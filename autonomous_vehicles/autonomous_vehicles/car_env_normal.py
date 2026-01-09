import gym
from gym import spaces

import numpy as np
import threading
import time
import subprocess
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32


class GazeboLidarMaskEnvNormal(gym.Env):
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
        self._offroad_seq = 0
        self.offroad_hold_s = 0
        # ---- pose (z TFMessage) ----
        self.pose_xy = None          # (x,y)
        self._pose_seq = 0
        self.pose_frame_contains = "vehicle_blue"  # filtr po frame id


        #nagroda za jechanie 
        self.prev_pose_xy = None

        # strojenie reward
        # self.min_progress = 0.02   
        # self.idle_penalty_scale = 0.2 
        # self.progress_scale = 40.0      # ile nagrody za 1 metr (zacznij 3..10)
        # self.alive_reward = 0.1       # mały bonus za "życie"
        # self.turn_penalty_scale = 0.0 # kara za kręcenie (opcjonalnie)
        # self.offroad_penalty = 100.0    # kara za offroad (zostaje)
        # self.error_scale = 3 

        # ===== strojenie reward =====
        self.min_progress = 0.02

        self.progress_scale = 20.0        # MNIEJ niż 40 – nie dominuje
        self.alive_reward = 1

        self.idle_penalty_scale = 0.3

        self.turn_reward_scale = 0.8      # NAGRODA za skręt gdy jest błąd
        self.speed_turn_penalty_scale = 1.0  # kara: szybko + skręt

        self.error_scale = 1.5            # lżejsza kara globalna
        self.edge_penalty_scale = 15.0    # mocna kara blisko krawędzi

        self.offroad_penalty = 500.0

        self.error = 0 # pole przechowywujace ronce meidzy pose a gt





        # ---- offroad ----
        self.offroad_flag = False

        # ---- max commands ----
        self.max_lin = 2.0   # [m/s]
        self.max_ang = 1.0   # [rad/s]

        # ---- teleport start ----
        self.start_x = 25.0
        self.start_y = 2.0
        self.start_z = 0.325
        self.start_yaw = 0.0  # rad
        self.reset_grace_s = 0.5 
        # ---- Observation space ----
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

        self.node = Node(f"gym_lidar_mask_env_{int(time.time()*1000)}")
        # ---- random spawn points ----
        self.spawn_points = []
        self.spawn_file = "/home/developer/ros2_ws/src/xy8.txt"  # albo pełna ścieżka
        self.rng = np.random.default_rng()

        self._load_spawn_points(self.spawn_file)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # ===== GOAL / META =====
        self.goal_xy = None          # (x, y)
        self.goal_radius = 0.5       # [m]
        self.goal_reward = 100.0

        if self.goal_xy is None:
            gx, gy, _ = self.spawn_points[-1]
            self.goal_xy = (float(gx), float(gy))

        self._lock = threading.Lock()

        self.lidar_points = None
        self.mask = None
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

        # POSE (TFMessage) z bridge
        self.node.create_subscription(
            TFMessage,
            "/model/vehicle_blue/pose",
            self._pose_callback,
            qos
        )

        self.node.create_subscription(
            Float32,              
            '/groundtruth_error',  
            self._error_callback,
            10                  
        )

        # publisher cmd_vel
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        # ostatnie wysłane komendy (do reward)
        self.last_v_cmd = 0.0
        self.last_w_cmd = 0.0

        # skala nagrody za jazdę (możesz potem stroić)
        self.speed_reward_scale = 1.0


        self.step_count = 0

    def _sample_spawn_pose(self, k=20):
        n = len(self.spawn_points)
        idx = self.rng.integers(k, n - k)

        p = self.spawn_points[idx]
        x, y = float(p[0]), float(p[1])

        p_prev = self.spawn_points[idx - k]
        p_next = self.spawn_points[idx + k]

        x_prev, y_prev = float(p_prev[0]), float(p_prev[1])
        x_next, y_next = float(p_next[0]), float(p_next[1])

        # losowo wybieramy kierunek
        if self.rng.random() < 0.5:
            dx = x_next - x_prev
            dy = y_next - y_prev
        else:
            dx = x_prev - x_next
            dy = y_prev - y_next

        yaw = math.atan2(dy, dx)

        return x, y, yaw

    def _check_goal_reached(self):
        if self.goal_xy is None:
            return False

        with self._lock:
            cur = self.pose_xy

        if cur is None:
            return False

        dx = cur[0] - self.goal_xy[0]
        dy = cur[1] - self.goal_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)

        return dist <= self.goal_radius


    def _load_spawn_points(self, path: str):
        pts = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    line = line.replace(",", " ")
                    parts = [p for p in line.split() if p]
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        yaw = float(parts[2]) if len(parts) >= 3 else None
                        pts.append((x, y, yaw))
        except Exception as e:
            self.node.get_logger().warn(f"Nie mogę wczytać spawnów z {path}: {e}")

        self.spawn_points = pts
        self.node.get_logger().info(f"Wczytano {len(self.spawn_points)} punktów startowych z {path}")




    # ---------- utils ----------
    def _yaw_to_quat(self, yaw: float):
        half = 0.5 * float(yaw)
        return (math.cos(half), 0.0, 0.0, math.sin(half))  # (w,x,y,z)

    def _teleport_to_start(self):
        w, x, y, z = self._yaw_to_quat(self.start_yaw)

        req = (
            f'name: "vehicle_blue" '
            f'position {{x: {self.start_x} y: {self.start_y} z: {self.start_z}}} '
            f'orientation {{w: {w} x: {x} y: {y} z: {z}}}'
        )

        cmd = [
            "gz", "service",
            "-s", "/world/mecanum_drive/set_pose/blocking",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", req
        ]

        self._send_cmd(0.0, 0.0)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f"Teleport failed: {e.stderr or e.stdout}")
            return False

        # stop + chwilka na stabilizację i nowe wiadomości
        self._send_cmd(0.0, 0.0)
        t0 = time.time()
        while time.time() - t0 < 0.3:
            rclpy.spin_once(self.node, timeout_sec=0.05)

        return True

    def _wait_for_pose_update(self, timeout=1.0, last_pose_seq=-1):
        with self._lock:
            self.prev_pose_xy = self.pose_xy
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            with self._lock:
                if self._pose_seq > last_pose_seq and self.pose_xy is not None:
                    return True
        return False

    def _wait_offroad_fresh(self, timeout=1.0):
        """
        Po teleporcie często masz jeszcze starą flagę/offroad przez ułamek sekundy.
        Tu tylko spinujemy, żeby OffroadChecker zdążył policzyć i opublikować nowy stan.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    # ---------- control ----------
    def _send_cmd(self, v, w):
        self.last_v_cmd = float(v)
        self.last_w_cmd = float(w)

        msg = Twist()
        msg.linear.x = self.last_v_cmd
        msg.angular.z = self.last_w_cmd
        self.cmd_pub.publish(msg)



    def _offroad_cb(self, msg: Bool):
            off = bool(msg.data)
            self.offroad_flag = off
            self._offroad_seq += 1

            now = time.time()
            if off:
                if self._offroad_true_since is None:
                    self._offroad_true_since = now
            else:
                self._offroad_true_since = None

    # ---------- callbacks ----------
    def _pose_callback(self, msg: TFMessage):
        # TFMessage może mieć kilka transformacji -> bierzemy tę od auta
        best = None
        for tr in msg.transforms:
            child = tr.child_frame_id or ""
            parent = tr.header.frame_id or ""
            if (self.pose_frame_contains in child) or (self.pose_frame_contains in parent):
                best = tr
                break
        if best is None:
            # fallback: pierwsza transformacja
            if len(msg.transforms) == 0:
                return
            best = msg.transforms[0]

        x = float(best.transform.translation.x)
        y = float(best.transform.translation.y)
        with self._lock:
            self.pose_xy = (x, y)
            self._pose_seq += 1

    def _error_callback(self, msg: Float32):
        self.error = msg.data

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

        mask_class = np.argmax(mask_logits, axis=2).astype(np.uint8)
        num_classes = int(mask_logits.shape[2])
        mask_scaled = (mask_class * (255 // max(1, num_classes - 1))).astype(np.uint8)

        with self._lock:
            self.mask = mask_scaled
            self._mask_seq += 1

    # ---------- lidar -> images ----------
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

        mask_resized = np.flipud(np.fliplr(mask.copy()))

        depth = np.sqrt(x**2 + y**2 + z**2)
        depth = np.clip(depth, 0.0, float(depth_max_m))

        mask_img[v, u] = mask_resized[v, u]
        depth_map[v, u] = depth

        mask_img = np.rot90(mask_img, 2)
        depth_map = np.rot90(depth_map, 2)

        depth_u8 = (depth_map * (255.0 / float(depth_max_m))).astype(np.uint8)
        return mask_img, depth_u8

    # ---------- obs helper ----------
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
            obs = np.stack([mask_img, depth_u8], axis=0).astype(np.uint8)
            return obs, lidar_seq, mask_seq

        obs = np.zeros((2, self.H, self.W), dtype=np.uint8)
        with self._lock:
            return obs, self._lidar_seq, self._mask_seq

    # ---------- reward ----------
    def _compute_reward(self):
        # ===== TWARDY KONIEC EPIZODU =====
        if self.offroad_flag:
            return -100.0

        # ===== ODCZYT STANU =====
        with self._lock:
            cur = self.pose_xy
            prev = self.prev_pose_xy
            last_w = float(getattr(self, 'last_w_cmd', 0.0))
            dist_from_center = float(getattr(self, 'dist_from_center', 0.0))  # [m]

        # ===== POSTĘP =====
        progress = 0.0
        if cur is not None and prev is not None:
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]
            progress = math.sqrt(dx * dx + dy * dy)

        with self._lock:
            self.prev_pose_xy = cur

        # ===== ERROR (0..1) =====
        LANE_HALF_WIDTH = 2.0  # 4 m / 2
        error = abs(dist_from_center) / LANE_HALF_WIDTH
        error = max(0.0, min(error, 1.0))

        # ===== REWARD SKŁADOWE =====

        # --- progres (zdegradowany przy dużym błędzie) ---
        progress_reward = 25.0 * min(progress, 0.3)
        progress_reward *= math.exp(-2.0 * error)

        # --- nagroda za skręt TYLKO gdy jest błąd ---
        turn_reward = 0.0
        if error > 0.05:
            turn_reward = 0.8 * abs(last_w)

        # --- kara: skręt + prędkość (żeby nie prostował zakrętów) ---
        speed_turn_penalty = 3.0 * abs(last_w) * progress

        # --- kara za stanie ---
        idle_penalty = 0.0
        if progress < 0.02:
            idle_penalty = 0.3 * (0.02 - progress)

        # --- kara blisko krawędzi ---
        edge_penalty = 0.0
        if error > 0.7:
            edge_penalty = 15.0 * (error - 0.7)

        # --- łagodna kara za błąd ---
        error_penalty = 1.5 * error

        # --- mały bonus za życie ---
        alive_reward = 0.05

        # ===== SUMA =====
        reward = (
            alive_reward
            + progress_reward
            + turn_reward
            - speed_turn_penalty
            - idle_penalty
            - edge_penalty
            - error_penalty
        )

        if self._check_goal_reached():
            reward += self.goal_reward

        return float(reward)





    # ---------- Gym API ----------
    def reset(self):
        self.step_count = 0
        self.offroad_flag = False
        self._offroad_true_since = None
        self._reset_time = time.time()

        with self._lock:
            last_pose_seq = self._pose_seq
        last_offroad_seq = self._offroad_seq

        # if len(self.spawn_points) > 0:
        #     x, y, yaw = self.spawn_points[self.rng.integers(0, len(self.spawn_points))]
        #     self.start_x = float(x)
        #     self.start_y = float(y)
        #     if yaw is not None:
        #         self.start_yaw = float(yaw)

        if len(self.spawn_points) >= 5:
            self.start_x, self.start_y, self.start_yaw = self._sample_spawn_pose(k=10)



        self._teleport_to_start()

        # poczekaj na nową pose
        self._wait_for_pose_update(timeout=1.0, last_pose_seq=last_pose_seq)

        # poczekaj aż przyjdzie przynajmniej 1 nowy /off_road po resecie
        t0 = time.time()
        while time.time() - t0 < 1.0 and self._offroad_seq <= last_offroad_seq:
            rclpy.spin_once(self.node, timeout_sec=0.05)

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
            pose_xy = self.pose_xy

        self._send_cmd(v, w)

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

        now = time.time()
        in_grace = (now - self._reset_time) < self.reset_grace_s

        offroad_confirmed = False
        if not in_grace and self.offroad_flag and self._offroad_true_since is not None:
            if (now - self._offroad_true_since) >= self.offroad_hold_s:
                offroad_confirmed = True

        #done = bool(offroad_confirmed or (self.step_count >= self.max_steps))

        goal_reached = self._check_goal_reached()

        done = bool(
            goal_reached
            or offroad_confirmed
            or (self.step_count >= self.max_steps)
        )


        info = {
            "offroad": bool(self.offroad_flag),
            "step": int(self.step_count),
            "v_cmd": float(v),
            "w_cmd": float(w),
            "pose_xy": pose_xy,
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
            if rclpy.ok():
                rclpy.shutdown()