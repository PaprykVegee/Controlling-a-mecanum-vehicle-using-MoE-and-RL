import gym
from gym import spaces

import numpy as np
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class GazeboLidarMaskEnv(gym.Env):
    """
    SB3 1.7.0 (Gym 0.21) compatible Env.

    Observation: np.uint8 array of shape (2, H, W)
      channel 0: lidar-projected mask image (0..255)
      channel 1: lidar depth (0..255), scaled from meters (0..100m)

    Action: np.float32 array shape (2,)
      [ v_norm in [0,1], w_norm in [-1,1] ]
    """
    metadata = {"render.modes": []}

    def __init__(self, H=256, W=256, time_step=0.1, max_steps=500):
        super().__init__()

        self.H, self.W = int(H), int(W)
        self.time_step = float(time_step)
        self.max_steps = int(max_steps)

        self.offroad_flag = False
        self.max_lin = 2.0   # [m/s]
        self.max_ang = 1.0   # [rad/s]

        # ---- Observation space: (C,H,W) = (2,H,W), uint8 for images ----
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(2, self.H, self.W),
            dtype=np.uint8
        )

        # ---- Action space ----
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---- ROS2 init ----
        if not rclpy.ok():
            rclpy.init(args=None)

        # unikalna nazwa node (żeby nie kolidować przy kilku uruchomieniach)
        self.node = Node(f"gym_lidar_mask_env_{int(time.time()*1000)}")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ---- shared state ----
        self._lock = threading.Lock()

        self.lidar_points = None  # (N,3) float32
        self.mask = None          # (H,W) uint8 already scaled 0..255

        self._lidar_seq = 0
        self._mask_seq = 0

        # ---- subscriptions ----
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

        self.node.create_subscription(
            Bool,
            "/off_road",
            self._offroad_cb,
            10
        )

        # publisher cmd_vel
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        self.step_count = 0

    # ---------------- control ----------------
    def _send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _offroad_cb(self, msg: Bool):
        self.offroad_flag = bool(msg.data)

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
        dims = [dim.size for dim in msg.layout.dim]
        data = np.array(msg.data, dtype=np.float32)

        try:
            mask_logits = data.reshape(dims)  # (H, W, C)
        except Exception as e:
            self.node.get_logger().warn(
                f"Mask: reshape fail dims={dims}, data.size={data.size}: {e}"
            )
            return

        if mask_logits.ndim != 3:
            self.node.get_logger().warn(f"Mask: oczekuję 3D (H,W,C), mam {mask_logits.shape}")
            return

        mask_class = np.argmax(mask_logits, axis=2).astype(np.uint8)  # (H, W)
        num_classes = int(mask_logits.shape[2])

        # skala do 0..255 (żeby było obrazowo)
        mask_scaled = (mask_class * (255 // max(1, num_classes - 1))).astype(np.uint8)

        with self._lock:
            self.mask = mask_scaled
            self._mask_seq += 1

    # ---------------- lidar -> (mask_img, depth_u8) ----------------
    def lidar_to_image_and_depth_u8(
        self,
        lidar,
        mask,
        horizontal_min=-0.6018333335,
        horizontal_max=0.6018333335,
        vertical_min=-0.338,
        vertical_max=0.338,
        depth_max_m=100.0
    ):
        H, W = mask.shape[:2]

        valid = ~np.isnan(lidar).any(axis=1) & np.isfinite(lidar).all(axis=1)
        lidar = lidar[valid]
        if lidar.shape[0] == 0:
            return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)

        x, y, z = lidar[:, 0], lidar[:, 1], lidar[:, 2]

        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x**2 + y**2))

        u = ((az - horizontal_min) / (horizontal_max - horizontal_min) * (W - 1)).astype(np.int32)
        v = ((el - vertical_min) / (vertical_max - vertical_min) * (H - 1)).astype(np.int32)

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        mask_img = np.zeros((H, W), dtype=np.uint8)
        depth_map = np.zeros((H, W), dtype=np.float32)

        # zachowanie jak w Twoim ControllerNode: flip mask
        mask_resized = np.flipud(np.fliplr(mask.copy()))

        depth = np.sqrt(x**2 + y**2 + z**2)
        depth = np.clip(depth, 0.0, float(depth_max_m))

        mask_img[v, u] = mask_resized[v, u]
        depth_map[v, u] = depth

        # obrót 180° jak w ControllerNode
        mask_img = np.rot90(mask_img, 2)
        depth_map = np.rot90(depth_map, 2)

        # depth -> uint8 0..255 (skala stała, nie per-frame)
        depth_u8 = (depth_map * (255.0 / float(depth_max_m))).astype(np.uint8)

        return mask_img, depth_u8

    # ---------------- obs helper ----------------
    def _get_obs_blocking(self, timeout=1.0, require_new=True, last_lidar_seq=-1, last_mask_seq=-1):
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

            mask_img, depth_u8 = self.lidar_to_image_and_depth_u8(lidar, mask)

            obs = np.stack([mask_img, depth_u8], axis=0).astype(np.uint8)  # (2,H,W)
            return obs, lidar_seq, mask_seq

        # fallback
        obs = np.zeros((2, self.H, self.W), dtype=np.uint8)
        with self._lock:
            return obs, self._lidar_seq, self._mask_seq

    # ---------------- reward ----------------
    def _compute_reward(self):
        # minimal: kara za offroad, inaczej 0
        return -1.0 if self.offroad_flag else 0.0

    # ---------------- Gym API (SB3 1.7 expects old gym API) ----------------
    def reset(self):
        self.step_count = 0
        self.offroad_flag = False

        self._send_cmd(0.0, 0.0)

        with self._lock:
            last_lidar_seq = self._lidar_seq
            last_mask_seq = self._mask_seq

        obs, _, _ = self._get_obs_blocking(
            timeout=2.0,
            require_new=True,
            last_lidar_seq=last_lidar_seq,
            last_mask_seq=last_mask_seq
        )
        return obs

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        v_norm = float(np.clip(action[0], 0.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))
        v = v_norm * self.max_lin
        w = w_norm * self.max_ang

        with self._lock:
            last_lidar_seq = self._lidar_seq
            last_mask_seq = self._mask_seq

        self._send_cmd(v, w)

        # odczekaj krok symulacji + spin
        t0 = time.time()
        while time.time() - t0 < self.time_step:
            rclpy.spin_once(self.node, timeout_sec=0.02)

        obs, _, _ = self._get_obs_blocking(
            timeout=1.0,
            require_new=True,
            last_lidar_seq=last_lidar_seq,
            last_mask_seq=last_mask_seq
        )

        reward = float(self._compute_reward())
        done = bool(self.offroad_flag or (self.step_count >= self.max_steps))

        info = {
            "offroad": bool(self.offroad_flag),
            "step": int(self.step_count),
            "v_cmd": float(v),
            "w_cmd": float(w),
        }

        return obs, reward, done, info

    def close(self):
        try:
            self._send_cmd(0.0, 0.0)
        except Exception:
            pass

        try:
            self.node.destroy_node()
        finally:
            # UWAGA: jeśli masz inne nody w tym samym procesie, to shutdown tu może przeszkadzać.
            # Jeśli env jest uruchamiany samodzielnie w procesie treningu — jest OK.
            if rclpy.ok():
                rclpy.shutdown()
