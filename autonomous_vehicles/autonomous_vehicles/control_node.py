import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

import numpy as np
import cv2
import math
import time

from scipy.interpolate import splrep, splev

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from autonomous_vehicles.view_manipulation import *
from autonomous_vehicles.Segmetation import *


# --- Klasa Regulatora PID ---
class regulatorPID():
    def __init__(self, P, I, D, dt, outlim):
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.outlim = outlim
        self.x_prev = 0
        self.x_sum = 0

    def output(self, x):
        P = self.P * x
        if np.abs(x) < self.outlim:
            self.x_sum += self.I * x * self.dt
        D = self.D * (x - self.x_prev) / self.dt
        out = P + self.x_sum + D
        self.x_prev = x

        if out > self.outlim:
            out = self.outlim
        elif out < -self.outlim:
            out = -self.outlim
        return out


def get_yellow_centroids(frame, visu=True):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    height, width = frame.shape[:2]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    centroids_list = []

    for i in range(1, num_labels):
        cx, cy = centroids[i]
        if 0 <= cx < width and 0 <= cy < height:
            centroids_list.append((int(cx), int(cy)))
            if visu:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), 2)

    return centroids_list, frame, mask


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        # --- sterowanie
        self.x = 0.0
        self.vx = 0.0
        self.PID = regulatorPID(0.05, 0.0, 0.02, 0.01, 1.0)

        self.send_msg_timer = 0.05
        self.main_timer = 0.05

        # --- kamera
        self.left_camera_img = None
        self.bridge = CvBridge()

        # --- lidar
        self.last_scan = None
        self.last_scan_time = 0.0
        self._lidar_seen = False

        self.lidar_img_size = 600
        self.lidar_scale = 40.0      # px / m
        self.lidar_max_draw = 12.0   # m

        # --- pub/sub
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self.get_left_img,
            10
        )

        # Dopasuj QoS do publishera z ros_gz_bridge (u Ciebie RELIABLE)
        lidar_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan",
            self.lidar_callback,
            lidar_qos
        )

        # --- timery (WAŻNE: osobne referencje!)
        self.speed_timer = self.create_timer(self.send_msg_timer, self.set_speed)
        self.control_timer = self.create_timer(self.main_timer, self.control_loop)

        # osobny timer do odświeżania lidara (niezależnie od segmentacji)
        self.lidar_viz_timer = self.create_timer(0.05, self.show_lidar_cloud)

        # --- model segmentacji
        self.model = EvalModel(r"/home/developer/ros2_ws/src/model_trening/best-unet-epoch=05-val_dice=0.9838.ckpt")

        self.get_logger().info("ControllerNode started (camera + lidar).")

    # ---------- callbacks ----------
    def get_left_img(self, msg: Image):
        try:
            self.left_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"[camera] CvBridge conversion error: {e}")
            self.left_camera_img = None

    def lidar_callback(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = time.time()

        if not self._lidar_seen:
            self._lidar_seen = True
            self.get_logger().info(f"LiDAR OK. frame_id={msg.header.frame_id}, ranges={len(msg.ranges)}")

    # ---------- viz ----------
    def show_lidar_cloud(self):
        if self.last_scan is None:
            return

        msg = self.last_scan
        N = self.lidar_img_size
        img = np.zeros((N, N, 3), dtype=np.uint8)
        cx, cy = N // 2, N // 2

        cv2.line(img, (0, cy), (N, cy), (60, 60, 60), 1)
        cv2.line(img, (cx, 0), (cx, N), (60, 60, 60), 1)

        angle = msg.angle_min
        rmax = min(msg.range_max, self.lidar_max_draw)

        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= rmax:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                px = int(cx + x * self.lidar_scale)
                py = int(cy - y * self.lidar_scale)
                if 0 <= px < N and 0 <= py < N:
                    img[py, px] = (0, 255, 0)
            angle += msg.angle_increment

        cv2.putText(img, f"LiDAR frame: {msg.header.frame_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # min range (bonus)
        finite_ranges = [r for r in msg.ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
        if finite_ranges:
            cv2.putText(img, f"min: {min(finite_ranges):.2f} m", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("LiDAR (2D point cloud)", img)
        cv2.waitKey(1)  # <- ważne dla OpenCV GUI

    # ---------- control ----------
    def set_speed(self):
        msg_out = Twist()
        out = self.PID.output(self.x)

        msg_out.linear.x = float(self.vx)
        msg_out.linear.y = 0.0
        msg_out.angular.z = float(out)

        self.cmd_vel_publisher.publish(msg_out)

    def control_loop(self):
        if self.left_camera_img is None:
            return

        # --- Predykcja maski segmentacyjnej ---
        mask_bool = self.model.predict(self.left_camera_img)
        h, w = self.left_camera_img.shape[:2]
        mask_bool = mask_bool.astype(np.uint8)
        mask_bool = cv2.resize(mask_bool, (w, h), interpolation=cv2.INTER_NEAREST)

        road_mask = cv2.dilate((mask_bool[:, :, 3] * 255).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=3)
        side_walk_mask = (mask_bool[:, :, 4] * 255).astype(np.uint8)

        # --- Bird Eye View ---
        frame_bev = perspectiveWarp(self.left_camera_img)
        side_walk_mask = perspectiveWarp(side_walk_mask)
        road_mask = perspectiveWarp(road_mask)

        centroids_list, frame_vis, line_mask = get_yellow_centroids(frame_bev)

        # --- punkty prawego pasa ---
        ys, xs = np.where(road_mask > 0)
        right_pts = []

        for y in range(0, road_mask.shape[0], 5):
            row_indices = np.where(ys == y)[0]
            if len(row_indices) == 0:
                continue
            row_xs = xs[row_indices]

            line_xs = np.where(line_mask[y] > 0)[0]
            if len(line_xs) == 0:
                continue
            line_x = int(np.mean(line_xs))

            right_xs = row_xs[row_xs > line_x]
            if len(right_xs) > 0:
                right_center_x = int(np.mean(right_xs))
                right_pts.append((y, right_center_x))

        center_x = frame_bev.shape[1] / 2
        y_samples = [frame_bev.shape[0] // 4, frame_bev.shape[0] // 3, frame_bev.shape[0] // 2]

        mask_spline_vis = np.zeros_like(road_mask)

        if len(right_pts) >= 5:
            y_pts, x_pts = zip(*right_pts)
            tck = splrep(y_pts, x_pts, s=20)
            self.last_tck = tck

            y_spline_full = np.arange(0, frame_bev.shape[0])
            x_spline_full = splev(y_spline_full, tck)
            for y_val, x_val in zip(y_spline_full, x_spline_full):
                yi, xi = int(y_val), int(x_val)
                if 0 <= yi < mask_spline_vis.shape[0] and 0 <= xi < mask_spline_vis.shape[1]:
                    mask_spline_vis[yi, xi] = 255

        elif hasattr(self, "last_tck"):
            tck = self.last_tck
        else:
            self.x = 0.0
            tck = None

        if tck is not None:
            errors = []
            for y in y_samples:
                x_sample = float(splev(y, tck))
                errors.append(center_x - x_sample)
            self.x = 5 * np.mean(errors) / center_x
            x_goal = float(splev(y_samples[0], tck))
            y_goal = y_samples[0]
        else:
            self.x = 0.0
            x_goal = center_x
            y_goal = frame_bev.shape[0] // 2

        # --- prędkość
        v_max = 5.0
        v_min = 0.5
        k = 0.7
        v = v_max * (1 - k * abs(self.x))
        self.vx = float(np.clip(v, v_min, v_max))

        # --- wizualizacje
        frame_vis_color = frame_vis.copy()
        frame_vis_color[mask_spline_vis > 0] = (0, 0, 255)
        cv2.line(frame_vis_color, (int(center_x), 0), (int(center_x), frame_vis.shape[0]), (255, 0, 0), 2)
        cv2.circle(frame_vis_color, (int(x_goal), y_goal), 6, (0, 255, 0), -1)

        cv2.putText(frame_vis_color, f"PID Error: {self.x:.3f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_vis_color, f"Speed: {self.vx:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        mask = np.argmax(mask_bool, axis=2).astype(np.uint8)
        color_mask = cv2.applyColorMap(mask * (255 // mask_bool.shape[2]), cv2.COLORMAP_JET)

        cv2.imshow("Segmentation Mask", color_mask)
        cv2.imshow("Bird eye", frame_vis_color)
        cv2.imshow("Right trajectory spline", mask_spline_vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ControllerNode()

    # MultiThreadedExecutor pomaga, gdy segmentacja blokuje callbacki/timery
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
