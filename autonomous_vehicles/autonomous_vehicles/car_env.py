import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import time
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class GazeboLidarMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, H=256, W=256, time_step=0.1, max_steps=500):
        super().__init__()

        self.H, self.W = H, W
        self.time_step = float(time_step)
        self.max_steps = int(max_steps)
        self.offroad_flag = False # flaga od wykroczenia poza droge
        # ---------- Gym spaces ----------
        # przykładowo: obserwacja = (gray uint8) + (depth float32)
        self.observation_space = spaces.Dict({
            "lidar_img": spaces.Box(0, 255, shape=(H, W), dtype=np.uint8),
            "depth":     spaces.Box(0.0, 100.0, shape=(H, W), dtype=np.float32),
        })

        # akcja przykładowa (dopasuj do swojego robota)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---------- ROS2 init ----------
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node("gym_lidar_mask_env")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ---------- shared state ----------
        self._lock = threading.Lock()

        self.lidar_points = None  # (N,3) float32
        self.mask = None          # (H,W) uint8

        # liczniki świeżości (kluczowe dla RL)
        self._lidar_seq = 0
        self._mask_seq = 0

        # ---------- subscriptions ----------
        self.node.create_subscription(
            Float32MultiArray,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/tensor",
            self._lidar_callback,
            qos
        )

        self.node.create_subscription(
            Float32MultiArray,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/mask",
            self._mask_callback,
            qos
        )

        self.offroad_sub = self.node.create_subscription(
            Bool,
            "/off_road",
            self._offroad_cb,
            10
        )

        # krok epizodu
        self.step_count = 0

    # ---------------- callbacks ----------------
    def _lidar_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size % 3 != 0:
            self.node.get_logger().warn("LiDAR: liczba elementów nie dzieli się przez 3!")
            return

        points = data.reshape((-1, 3))
        with self._lock:
            self.lidar_points = points
            self._lidar_seq += 1

    def _mask_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)

        # Wariant A: maska przychodzi jako H*W (1D) -> reshape
        if data.size == self.H * self.W:
            mask = data.reshape((self.H, self.W))
        else:
            # jeżeli masz layout.dim w msg, możesz z niego odtworzyć shape
            # tutaj uproszczenie: fallback
            self.node.get_logger().warn(f"Mask: nieoczekiwany rozmiar {data.size}, oczekuję {self.H*self.W}")
            return

        # normalizacja do uint8 0..255
        mask = np.clip(mask, 0.0, 1.0) * 255.0
        mask = mask.astype(np.uint8)

        with self._lock:
            self.mask = mask
            self._mask_seq += 1

    # ---------------- lidar->image/depth ----------------
    def lidar_to_image_and_depth(self, lidar, mask,
                                horizontal_min=-0.6018333335,
                                horizontal_max=0.6018333335,
                                vertical_min=-0.338,
                                vertical_max=0.338):
        H, W = mask.shape[:2]

        valid = ~np.isnan(lidar).any(axis=1) & np.isfinite(lidar).all(axis=1)
        lidar = lidar[valid]
        if lidar.shape[0] == 0:
            return np.zeros((H, W), np.uint8), np.zeros((H, W), np.float32)

        x, y, z = lidar[:, 0], lidar[:, 1], lidar[:, 2]

        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x**2 + y**2))

        u = ((az - horizontal_min) / (horizontal_max - horizontal_min) * (W - 1)).astype(np.int32)
        v = ((el - vertical_min) / (vertical_max - vertical_min) * (H - 1)).astype(np.int32)

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        image_gray = np.zeros((H, W), dtype=np.uint8)
        depth_map = np.zeros((H, W), dtype=np.float32)

        # u Ciebie było flip + rot; zostawiam identyczną logikę
        mask_resized = np.flipud(np.fliplr(mask.copy()))
        depth = np.sqrt(x**2 + y**2 + z**2)

        image_gray[v, u] = mask_resized[v, u]
        depth_map[v, u] = depth

        return image_gray, depth_map

    # ---------------- obs helpers ----------------
    def _get_obs_blocking(self, timeout=1.0, require_new=True, last_lidar_seq=-1, last_mask_seq=-1):
        """
        Czeka aż:
        - będziemy mieli lidar + mask
        - oraz (jeśli require_new) ich seq będzie większy niż last_*
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.02)

            with self._lock:
                have = (self.lidar_points is not None) and (self.mask is not None)
                if not have:
                    continue

                if require_new:
                    if self._lidar_seq <= last_lidar_seq or self._mask_seq <= last_mask_seq:
                        continue

                lidar = self.lidar_points.copy()
                mask = self.mask.copy()
                lidar_seq = self._lidar_seq
                mask_seq = self._mask_seq

            lidar_img, depth = self.lidar_to_image_and_depth(lidar, mask)
            return {"lidar_img": lidar_img, "depth": depth}, lidar_seq, mask_seq

        # timeout -> zwróć “bezpieczne” zera
        obs = {
            "lidar_img": np.zeros((self.H, self.W), dtype=np.uint8),
            "depth": np.zeros((self.H, self.W), dtype=np.float32),
        }
        with self._lock:
            return obs, self._lidar_seq, self._mask_seq

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # ważne: jeśli chcesz wymusić świeże dane po "reset-sytuacji":
        with self._lock:
            # nie musisz zerować danych, ale warto wymusić require_new
            last_lidar_seq = self._lidar_seq
            last_mask_seq = self._mask_seq

        obs, _, _ = self._get_obs_blocking(timeout=2.0, require_new=True,
                                           last_lidar_seq=last_lidar_seq, last_mask_seq=last_mask_seq)
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # 1) Skala/clip akcji (dopasuj do siebie)
        v_norm = float(np.clip(action[0], 0.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))
        v = v_norm * self.max_lin
        w = w_norm * self.max_ang

        # 2) ZAPAMIĘTAJ seq PRZED akcją (żeby wymusić świeże dane po akcji)
        with self._lock:
            last_lidar_seq = self._lidar_seq
            last_mask_seq = self._mask_seq

        # 3) Wykonaj akcję w symulacji
        # Jeśli używasz pause/unpause:
        self._start_gz()                 # unpause (opcjonalnie)
        self._send_cmd(v, w)             # publish /cmd_vel

        # 4) Poczekaj dt (symulacja "idzie", callbacki zbierają dane)
        t0 = time.time()
        while time.time() - t0 < self.time_step:
            rclpy.spin_once(self.node, timeout_sec=0.02)

        self._stop_gz()                  # pause (opcjonalnie)

        # 5) Pobierz obserwację "po akcji" (musi być świeższa niż last_*)
        obs, _, _ = self._get_obs_blocking(
            timeout=1.0,
            require_new=True,
            last_lidar_seq=last_lidar_seq,
            last_mask_seq=last_mask_seq
        )

        # 6) Reward + warunki zakończenia (przykład)
        reward = self._compute_reward(obs)  # lub własna logika

        terminated = False
        truncated = False

        if self.collision_flag:
            terminated = True

        if self.destination_reached_flag:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        # jeśli terminated, zwykle truncated = False
        if terminated:
            truncated = False

        info = {}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
