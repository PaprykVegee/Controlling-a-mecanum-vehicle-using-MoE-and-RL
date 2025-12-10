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

        self.create_subscription(
            PointStamped,
			"/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/position_xyz",
            self.gps_callback,
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            "/cmd_vel", 
            10
        )

        self.manual_control()


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

        self.get_logger().info(f"Odebrano {mask.shape} maski")

        self.mask = color_mask

        cv2.imshow("Segmentation Mask", color_mask)
        cv2.waitKey(1)


    def lidar_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)

        if data.size % 3 != 0:
            self.get_logger().warn("Liczba elementów nie dzieli się przez 3!")
            return

        points = data.reshape((-1, 3))

        self.lidar_points = points

        self.get_logger().info(f"Odebrano {points.shape} punktów LiDAR")

  

    def gps_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        self.get_logger().info(f"XYZ = ({x:.3f}, {y:.3f}, {z:.3f}), frame={msg.header.frame_id}")


    def lidar_to_colored_points(self, lidar, image, 
                                horizontal_samples=400, vertical_samples=64,
                                dx=0.0, dy=0.0, dz=0.0,
                                roll=0.0, pitch=0.0, yaw=0.0):
        rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

        lidar_cam = (rot @ lidar.T).T + np.array([dx, dy, dz])
        x, y, z = lidar_cam[:,0], lidar_cam[:,1], lidar_cam[:,2]

        mask_resized = cv2.resize(image, (horizontal_samples, vertical_samples), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.flipud(mask_resized)
        mask_resized = np.fliplr(mask_resized)

        colors_arr = mask_resized.reshape(-1, 3) / 255.0
        colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r,g,b in colors_arr]

        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x = x[valid_idx]
        y = y[valid_idx]
        z = z[valid_idx]
        colors = [c for i, c in enumerate(colors) if valid_idx[i]]
        
        return x, y, z, colors


    def lidar_to_img_top(self, x, y, colors, img_size=(800, 800), marker_size=1):
        H, W = img_size
        img_array = np.zeros((H, W, 3), dtype=np.uint8)

        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * (W-1)
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) * (H-1)

        y_norm = H - 1 - y_norm
        
        for xi, yi, color in zip(x_norm, y_norm, colors):
            r, g, b = [int(c) for c in color[4:-1].split(',')]
            xi, yi = int(xi), int(yi)
            x_min = max(xi - marker_size//2, 0)
            x_max = min(xi + marker_size//2 + 1, W)
            y_min = max(yi - marker_size//2, 0)
            y_max = min(yi + marker_size//2 + 1, H)
            img_array[y_min:y_max, x_min:x_max] = [r, g, b]
        img_array = np.rot90(img_array)
        
        cv2.imshow("Test", img_array)
        
    def control_loop(self):
        self.get_logger().info(f"test")

        # img_colored = self.lidar_to_colored_image()
        # if img_colored is not None:
        #     cv2.imshow("LiDAR Colored by Mask", img_colored)
        #     cv2.waitKey(1)

        if self.lidar_points is not None and self.mask is not None:
            if self.mask.size == 0:
                self.get_logger().warn("Maska jest pusta, nie można przypisać kolorów")
            else:
                x, y, z, colors_arr = self.lidar_to_colored_points(self.lidar_points, self.mask)
                self.lidar_to_img_top(x, y, colors_arr)




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
                return False  # stop listener

            self.cmd_vel_publisher.publish(self.twist)

        def on_release(key):
            if hasattr(key, 'char'):
                k = key.char.lower()

                # reset ruchu do przodu/tyłu
                if k in ['w', 's']:
                    self.twist.linear.x = 0.0

                # reset skręcania
                if k in ['a', 'd']:
                    self.twist.angular.z = 0.0

                self.cmd_vel_publisher.publish(self.twist)


        # uruchom listener w osobnym wątku
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