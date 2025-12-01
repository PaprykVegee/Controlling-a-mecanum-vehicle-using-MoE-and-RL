#to create bridge
    # ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

    # ros2 run ros_gz_bridge parameter_bridge \
    #   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
    #   /world/mecanum_drive/model/vehicle_blue/link/left_camera_link/sensor/left_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image \
    #   /world/mecanum_drive/model/vehicle_blue/link/right_camera_link/sensor/right_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip3 install segmentation_models_pytorch


# /world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, NavSatFix
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit

import subprocess
import shlex
from autonomous_vehicles.view_manipulation import *
from autonomous_vehicles.Segmetation import *
from rclpy.qos import qos_profile_sensor_data
import math
import time
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
        if out > self.outlim: out = self.outlim
        elif out < -self.outlim: out = -self.outlim
        
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
    
def find_edges(frame, visu=True):
    th_low = 100
    th_high = 200
    edges = cv2.Canny(frame, None, th_low, th_high)
    return edges

    

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        
        self.x = 0 
        self.vx = 0.0 
        
        self.PID = regulatorPID(0.05, 0, 0.02, 0.01, 1.0) 
        
        self.left_camera_img = None
        self.right_camera_img = None
        self.left_camera_status = False
        self.right_camera_status = False

        self.send_msg_timer = 0.05 
        self.main_timer = 0.05 
        #lidar
        self.last_scan = None
        self.last_scan_time = 0.0

        # podgląd lidara (metry -> piksele)
        self.lidar_img_size = 600
        self.lidar_scale = 40.0   # 40 px = 1 m  (dostosuj)
        self.lidar_max_draw = 12.0

        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self.get_left_img,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan",
            self.lidar_callback,
            qos_profile_sensor_data

        )
        self.create_subscription(
        NavSatFix,
        "/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/navsat",
        self.gps_cb,
        10
        )



        self.get_logger().info("Camera viewer and controller started.")

        self.timer = self.create_timer(self.send_msg_timer, self.set_speed)
        self.timer = self.create_timer(self.main_timer, self.control_loop)

        self.model = EvalModel(r"/home/developer/ros2_ws/src/model_trening/best-unet-epoch=05-val_dice=0.9838.ckpt")
        # ---
        self.stitcher = cv2.Stitcher_create()
    def gps_cb(self, msg: NavSatFix):
        self.get_logger().info(f"lat={msg.latitude:.7f}, lon={msg.longitude:.7f}, alt={msg.altitude:.2f}")
        
    def lidar_callback(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = time.time()

        # min tylko z sensownych wartości (finite + w zakresie)
        finite_ranges = [r for r in msg.ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
        if finite_ranges:
            min_dist = min(finite_ranges)
            # nie loguj co klatkę (bo zasypiesz terminal); np. co ~0.5 s:
            if int(self.get_clock().now().nanoseconds * 1e-9 * 2) % 1 == 0:
                self.get_logger().info(f"[LiDAR] Min range: {min_dist:.2f} m")

    def show_lidar_cloud(self):
        if self.last_scan is None:
            return

        msg = self.last_scan
        N = self.lidar_img_size
        img = np.zeros((N, N, 3), dtype=np.uint8)

        cx, cy = N // 2, N // 2

        # osie pomocnicze
        cv2.line(img, (0, cy), (N, cy), (60, 60, 60), 1)
        cv2.line(img, (cx, 0), (cx, N), (60, 60, 60), 1)

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= min(msg.range_max, self.lidar_max_draw):
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                px = int(cx + x * self.lidar_scale)
                py = int(cy - y * self.lidar_scale)  # minus, bo obraz ma y w dół

                if 0 <= px < N and 0 <= py < N:
                    img[py, px] = (0, 255, 0)
            angle += msg.angle_increment

        cv2.putText(img, f"LiDAR frame: {msg.header.frame_id}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("LiDAR Point Cloud (2D)", img)



    def set_speed(self): 
        msg_out = Twist()
        out = self.PID.output(self.x) 
        print(f'Output PID [deg]: {out:.4f}')

        msg_out.linear.x = self.vx 
        msg_out.linear.y = 0.0 
        msg_out.angular.z = float(out)
        
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Msg sent: v_x={msg_out.linear.x:.2f}, a_z={msg_out.angular.z:.4f}")

     

    def get_left_img(self, msg):
        try:
            self.left_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.left_camera_status = True
        except Exception as e:
            self.get_logger().error(f"[left camera] CvBridge conversion error: {e}")
            self.left_camera_status = False
            return
        
    def control_loop(self):
        if self.left_camera_img is None:
            return  

        # --- Predykcja maski segmentacyjnej ---
        mask_bool = self.model.predict(self.left_camera_img)
        h, w = self.left_camera_img.shape[:2]
        mask_bool = mask_bool.astype(np.uint8)
        mask_bool = cv2.resize(mask_bool, (w, h), interpolation=cv2.INTER_NEAREST)

        road_mask = cv2.dilate((mask_bool[:, :, 3] * 255).astype(np.uint8), np.ones((7,7), np.uint8), iterations=3)
        side_walk_mask = (mask_bool[:, :, 4] * 255).astype(np.uint8)

        # --- Bird Eye View ---
        # frame_bev = adaptivePerspectiveWarp(self.left_camera_img, road_mask)
        # side_walk_mask = adaptivePerspectiveWarp(side_walk_mask, road_mask)
        # road_mask = adaptivePerspectiveWarp(road_mask, road_mask)

        frame_bev = perspectiveWarp(self.left_camera_img)
        side_walk_mask = perspectiveWarp(side_walk_mask)
        road_mask = perspectiveWarp(road_mask)

        centroids_list, frame_vis, line_mask = get_yellow_centroids(frame_bev)

        # --- Wyciągamy punkty prawego pasa ---
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

        # --- Stałe punkty dla PID i wizualizacji ---
        center_x = frame_bev.shape[1] / 2
        y_samples = [frame_bev.shape[0] // 4, frame_bev.shape[0] // 3, frame_bev.shape[0] // 2]  # punkty predykcyjne

        mask_spline_vis = np.zeros_like(road_mask)

        if len(right_pts) >= 5:
            # Tworzymy gładki spline prawego pasa
            y_pts, x_pts = zip(*right_pts)
            tck = splrep(y_pts, x_pts, s=20)
            self.last_tck = tck  # zapisujemy spline do pamięci

            # Rysujemy trajektorię na mask_spline_vis
            y_spline_full = np.arange(0, frame_bev.shape[0])
            x_spline_full = splev(y_spline_full, tck)
            for y_val, x_val in zip(y_spline_full, x_spline_full):
                y_int, x_int = int(y_val), int(x_val)
                if 0 <= y_int < mask_spline_vis.shape[0] and 0 <= x_int < mask_spline_vis.shape[1]:
                    mask_spline_vis[y_int, x_int] = 255

        elif hasattr(self, "last_tck"):
            # Brak widocznych punktów, używamy ostatniego spline
            tck = self.last_tck
            y_spline_full = np.arange(0, frame_bev.shape[0])
            x_spline_full = splev(y_spline_full, tck)
            for y_val, x_val in zip(y_spline_full, x_spline_full):
                y_int, x_int = int(y_val), int(x_val)
                if 0 <= y_int < mask_spline_vis.shape[0] and 0 <= x_int < mask_spline_vis.shape[1]:
                    mask_spline_vis[y_int, x_int] = 255
        else:
            # Za mało danych, brak spline
            self.x = 0.0
            x_goal = center_x

        # --- Obliczenie błędu PID jako średnia kilku punktów w przód ---
        if len(right_pts) >= 5 or hasattr(self, "last_tck"):
            errors = []
            for y in y_samples:
                x_sample = float(splev(y, tck))
                errors.append(center_x - x_sample)
            self.x = 5 * np.mean(errors) / center_x  # wzmocnienie predykcyjne

            # Punkt docelowy do wizualizacji
            x_goal = float(splev(y_samples[0], tck))
            y_goal = y_samples[0]
        else:
            self.x = 0.0
            x_goal = center_x
            y_goal = frame_bev.shape[0] // 2

        # --- Prędkość i PID ---
        v_max = 5.0
        v_min = 0.5
        k = 0.7
        v = v_max * (1 - k * abs(self.x))
        self.vx = np.clip(v, v_min, v_max)

        out = self.PID.output(self.x)
        msg_out = Twist()
        msg_out.linear.x = self.vx
        msg_out.linear.y = 0.0
        msg_out.angular.z = float(out)
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Control: v_x={self.vx:.2f}, a_z={out:.4f}")

        # --- Wizualizacja trajektorii i punktu docelowego ---
        frame_vis_color = frame_vis.copy()
        frame_vis_color[mask_spline_vis > 0] = (0,0,255)  # czerwona trajektoria
        cv2.line(frame_vis_color, (int(center_x), 0), (int(center_x), frame_vis.shape[0]), (255,0,0), 2)
        cv2.circle(frame_vis_color, (int(x_goal), y_goal), 6, (0,255,0), -1)

        cv2.putText(frame_vis_color, f"PID Error: {self.x:.3f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame_vis_color, f"Speed: {self.vx:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        mask = np.argmax(mask_bool, axis=2).astype(np.uint8)

        # Mnożenie maski, żeby różne klasy dawały różne kolory
        color_mask = cv2.applyColorMap(mask * (255 // mask_bool.shape[2]), cv2.COLORMAP_JET)

        cv2.imshow("Segmentation Mask", color_mask)
        self.show_lidar_cloud()

        cv2.imshow("Bird eye", frame_vis_color)
        cv2.imshow("Right trajectory spline", mask_spline_vis)
        cv2.waitKey(1)

    # def control_loop(self):
    #     if self.left_camera_img is None:
    #         return  
        

    #     mask_bool = self.model.predict(self.left_camera_img)  # [2: 'Road Line', 3: 'Road', 4: 'Sidewalk', 'Car', 6: 'Vegetation', 7: 'Wall']
    #     h, w = self.left_camera_img.shape[:2]
    #     mask_bool = mask_bool.astype(np.uint8)
    #     mask_bool = cv2.resize(mask_bool, (w, h), interpolation=cv2.INTER_NEAREST)

    #     # mask_idx = np.argmax(mask_bool, axis=2).astype(np.uint8) 
    #     # mask_color = cv2.applyColorMap(mask_idx * (255 // mask_bool.shape[2]), cv2.COLORMAP_JET)

    #     road_mask = cv2.dilate((mask_bool[:, :, 3] * 255).astype(np.uint8), np.ones(7), iterations=3)
    #     # line_mask = cv2.morphologyEx((mask_bool[:, :, 2] * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones(7))
    #     #line_mask = cv2.dilate((mask_bool[:, :, 2] * 255).astype(np.uint8), np.ones(7)) # need to performe. Don't work becouse is bad mask for train data.
    #     side_walk_mask = (mask_bool[:, :, 4] * 255).astype(np.uint8)
        

    #     frame_bev = adaptivePerspectiveWarp(self.left_camera_img, road_mask)
    #     side_walk_mask = adaptivePerspectiveWarp(side_walk_mask, road_mask)
    #     road_mask = adaptivePerspectiveWarp(road_mask, road_mask)

    #     centroids_list, frame_vis, line_mask = get_yellow_centroids(frame_bev)


    #     mask3 = np.zeros_like(road_mask)


    #     mask3_spline = np.zeros_like(road_mask)
    #     prev_left = None
    #     prev_right = None

    #     # 1. Wyznacz krawędzie lewego i prawego pasa
    #     # Szukamy współrzędnych punktów niezerowych w road_mask
    #     ys, xs = np.where(road_mask > 0)

    #     for y in range(0, road_mask.shape[0], 5):
    #         row_indices = np.where(ys == y)[0]
    #         if len(row_indices) == 0:
    #             continue

    #         row_xs = xs[row_indices]

    #         line_xs = np.where(line_mask[y] > 0)[0]
    #         if len(line_xs) == 0:
    #             continue
    #         line_x = int(np.mean(line_xs))

    #         left_xs = row_xs[row_xs < line_x]
    #         if len(left_xs) > 0:
    #             left_center_x = int(np.mean(left_xs))
    #             if prev_left is not None:
    #                 cv2.line(mask3_spline, prev_left, (left_center_x, y), 255, 1)
    #             prev_left = (left_center_x, y)

    #         right_xs = row_xs[row_xs > line_x]
    #         if len(right_xs) > 0:
    #             right_center_x = int(np.mean(right_xs))
    #             if prev_right is not None:
    #                 cv2.line(mask3_spline, prev_right, (right_center_x, y), 255, 1)
    #             prev_right = (right_center_x, y)

    #     # Opcjonalnie pogrubienie linii
    #     mask3_spline = cv2.dilate(mask3_spline, np.ones((3,3), np.uint8), iterations=1)

    #     cv2.imshow("left right spline", mask3_spline)

    #     height, width = frame_bev.shape[:2]

    #     if len(centroids_list) > 0:
    #         avg_x = np.mean([c[0] for c in centroids_list])
    #         center_x = width / 2
    #         self.x = 5 * (center_x - avg_x) / center_x
            
    #         v_max = 5.0
    #         v_min = 0.5
    #         k = 0.7
    #         v = v_max * (1 - k * abs(self.x))
    #         self.vx = np.clip(v, v_min, v_max)

    #         cv2.line(frame_vis, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 2)
    #         cv2.line(frame_vis, (int(avg_x), 0), (int(avg_x), height), (0, 0, 255), 2)
    #         cv2.putText(frame_vis, f"Error: {self.x:.3f}", (30, 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    #         cv2.putText(frame_vis, f"Speed: {self.vx:.2f}", (30, 60),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #     else:
    #         self.x = 0.0
    #         self.vx = 0.0

    #     out = self.PID.output(self.x)
    #     msg_out = Twist()
    #     msg_out.linear.x = self.vx
    #     msg_out.linear.y = 0.0
    #     msg_out.angular.z = float(out)
    #     self.cmd_vel_publisher.publish(msg_out)
    #     self.get_logger().info(f"Control: v_x={self.vx:.2f}, a_z={out:.4f}")

    #     cv2.imshow("Mask road", find_edges(road_mask))
    #     cv2.imshow("Line road", line_mask)
    #     cv2.imshow("Side Walk road", side_walk_mask)
    #     # cv2.imshow("test", self.frame)
    #     cv2.imshow("Bird eye", frame_vis)
    #     cv2.waitKey(1)




def main():
    # processes = []
    # commands = ["ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
    #                 "ros2 run ros_gz_bridge parameter_bridge \
    #   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
    #   /world/mecanum_drive/model/vehicle_blue/link/left_camera_link/sensor/left_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image \
    #   /world/mecanum_drive/model/vehicle_blue/link/right_camera_link/sensor/right_camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image"
    # ]
    
    # for cmd in commands:
    #     p = subprocess.Popen(shlex.split(cmd)) 
    #     processes.append(p)
        
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# if __name__ == '__main__':
#     main()