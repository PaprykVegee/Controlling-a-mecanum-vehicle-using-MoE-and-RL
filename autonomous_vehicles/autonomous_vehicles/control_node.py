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

from pynput import keyboard

from scipy.spatial.transform import Rotation as R
import re
    
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

        self.send_msg_timer = 0.05 
        self.main_timer = 0.05 

        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()

        self.mask = None #pole klasy do trzymania maski obrazu
        self.lidar_points = None #pole kalsy do trzymnaia pkt 3d lidaru

        
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(
            Float32MultiArray,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/tensor",
            self.lidar_callback,
            qos
        )

        
        self.create_subscription(
            Float32MultiArray,
            '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/mask',
            self.mask_callback,
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            "/cmd_vel", 
            10
        )

        # publisher przetworzonych danych dla wrapera
        # self.color_lidar_publisher = self.create_publisher(

        # )

        self.manual_control()


        self.get_logger().info("Camera viewer and controller started.")

        self.timer = self.create_timer(self.main_timer, self.control_loop)

        self.stitcher = cv2.Stitcher_create()

    def mask_callback(self, msg: Float32MultiArray):
        dims = [dim.size for dim in msg.layout.dim]
        mask = np.array(msg.data, dtype=np.float32).reshape(dims)
        
        mask_class = np.argmax(mask, axis=2).astype(np.uint8)  
        
        num_classes = mask.shape[2]  
        mask_scaled = (mask_class * (255 // max(1, num_classes - 1))).astype(np.uint8)
        
        self.get_logger().info(f"Odebrano maskę o rozmiarze {mask.shape}")
        
        self.mask = mask_scaled ##wyslanie
        
        cv2.imshow("Segmentation Mask (Gray)", mask_scaled)
        cv2.waitKey(1)



    def lidar_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)

        if data.size % 3 != 0:
            self.get_logger().warn("Liczba elementów nie dzieli się przez 3!")
            return

        points = data.reshape((-1, 3))

        self.lidar_points = points # wyslanie 

        self.get_logger().info(f"Odebrano {points.shape} punktów LiDAR")


    def gps_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        self.get_logger().info(f"XYZ = ({x:.3f}, {y:.3f}, {z:.3f}), frame={msg.header.frame_id}")

    def lidar_to_image_and_depth(self, lidar, mask,
                                horizontal_min=-0.6018333335,
                                horizontal_max=0.6018333335,
                                vertical_min=-0.338,
                                vertical_max=0.338):
        H, W = mask.shape[:2]  

        valid = ~np.isnan(lidar).any(axis=1) & np.isfinite(lidar).all(axis=1)
        lidar = lidar[valid]

        x, y, z = lidar[:, 0], lidar[:, 1], lidar[:, 2]

        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x**2 + y**2))

        u = ((az - horizontal_min) / (horizontal_max - horizontal_min) * (W - 1)).astype(int)
        v = ((el - vertical_min) / (vertical_max - vertical_min) * (H - 1)).astype(int)

        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        image_gray = np.zeros((H, W), dtype=np.uint8)
        depth_map = np.zeros((H, W), dtype=np.float32)

        mask_resized = np.flipud(np.fliplr(mask.copy()))

        depth = np.sqrt(x**2 + y**2 + z**2)

        image_gray[v, u] = mask_resized[v, u]
        depth_map[v, u] = depth

        return np.rot90(image_gray, 2), np.rot90(depth_map, 2)

        
    def control_loop(self):
        self.get_logger().info("Test control_loop")

        if self.lidar_points is None or self.mask is None:
            self.get_logger().warn("Brak danych lidar lub maski")
            return

        if self.mask.size == 0:
            self.get_logger().warn("Maska jest pusta, nie można przypisać kolorów")
            return
        img, img_deph = self.lidar_to_image_and_depth(lidar=self.lidar_points, mask=self.mask)

        depth_norm = cv2.normalize(img_deph, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        depth_3ch = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

        img_combined = cv2.hconcat([img_3ch, depth_3ch])

        cv2.imshow("test", img_combined)
        cv2.waitKey(1)


    def manual_control(self):

        self.twist = Twist()
        self.manual_enabled = True

        def on_press(key):
            try:
                k = key.char.lower()
            except:
                k = str(key)

            if k == 'w':
                self.twist.linear.x = 2.0
            elif k == 's':
                self.twist.linear.x = -2.0
            elif k == 'a':
                self.twist.angular.z = 0.6
            elif k == 'd':
                self.twist.angular.z = -0.6
            elif k == ' ':
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
            elif k == 'q':
                self.manual_enabled = False
                return False 

            self.cmd_vel_publisher.publish(self.twist)

        def on_release(key):
            if hasattr(key, 'char'):
                k = key.char.lower()

                if k in ['w', 's']:
                    self.twist.linear.x = 0.0

                if k in ['a', 'd']:
                    self.twist.angular.z = 0.0

                self.cmd_vel_publisher.publish(self.twist)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()



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