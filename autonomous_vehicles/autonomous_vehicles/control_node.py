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
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped

import subprocess
import shlex
from autonomous_vehicles.view_manipulation import *
from autonomous_vehicles.Segmetation import *
from rclpy.qos import qos_profile_sensor_data
import math
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

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

        
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(
            PointCloud2,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/points",
            self.lidar_callback,
            qos
        )

        
        self.create_subscription(
            Float32MultiArray,
            '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/mask',
            self.mask_callback,
            10
        )

        self.create_subscription(
            PointStamped,
			"/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/position_xyz",
            self.gps_callback,
            10
        )


        self.get_logger().info("Camera viewer and controller started.")

        self.timer = self.create_timer(self.main_timer, self.control_loop)

        self.model = EvalModel(r"/home/developer/ros2_ws/src/UNET_trening/best-unet-epoch=05-val_dice=0.9838.ckpt")
        # ---
        self.stitcher = cv2.Stitcher_create()

    def mask_callback(self, msg: Float32MultiArray):
        dims = [dim.size for dim in msg.layout.dim]
        mask = np.array(msg.data, dtype=np.float32).reshape(dims)

        mask_class = np.argmax(mask, axis=2).astype(np.uint8) 

        color_mask = cv2.applyColorMap(mask_class * 32, cv2.COLORMAP_JET)

        cv2.imshow("Segmentation Mask", color_mask)
        cv2.waitKey(1)


    def lidar_callback(self, msg: PointCloud2):
        self.get_logger().info(f"Otrzymano wiadomość LiDAR 3D z {msg.width} punktami")
        data = np.array(msg.data, dtype=np.float32)
        
        if data.size % 3 != 0:
            self.get_logger().warn("Liczba elementów nie dzieli się przez 3!")
            return
        
        points = data.reshape((-1, 3))

        self.get_logger().info(f"Odebrano {points.shape[0]} punktów LiDAR")

        N = 600
        img = np.zeros((N, N, 3), dtype=np.uint8)
        cx, cy = N//2, N//2
        scale = 40.0 

        for x, y, z in points:
            px = int(cx + x*scale)
            py = int(cy - y*scale) 
            if 0 <= px < N and 0 <= py < N:
                img[py, px] = (0, 255, 0)

        cv2.imshow("LiDAR 3D Projection XY", img)
        cv2.waitKey(1)
  

    def gps_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        self.get_logger().info(f"XYZ = ({x:.3f}, {y:.3f}, {z:.3f}), frame={msg.header.frame_id}")

        
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


        
    def control_loop(self):
        pass



def main():
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