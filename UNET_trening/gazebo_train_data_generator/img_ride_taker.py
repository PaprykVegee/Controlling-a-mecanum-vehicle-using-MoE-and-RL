
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import pandas as pd
import time

from help_fun import *

df = pd.read_csv(r"/home/developer/ros2_ws/src/model_trening/carla_dataset/classes_rgb_values.csv")

df = df[['semantic_class', 'rgb_values']]

df['rgb_values'] = df['rgb_values'].apply(ast.literal_eval)
rgb_dict = dict(zip(df['semantic_class'], df['rgb_values']))

rgb_dict

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

    return centroids_list, frame
    
def find_edges(frame, visu=True):
    th_low = 100
    th_high = 200
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, None, th_low, th_high)
    return edges

    

class ControllerNode(Node):

    def __init__(self, rgb_dict):
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


        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/left_camera_link/sensor/left_camera_sensor/image",
            self.get_left_img,
            10
        )
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/right_camera_link/sensor/right_camera_sensor/image",
            self.get_right_img,
            10
        )
        target_colors = ((31, 42, 55), #Road
									(175, 104, 0), #Road Line
									(41, 65, 77), #Wall
                                    (77, 111, 86), # Vegetation
									(161, 161, 156)) #Sidewalk


        self.get_logger().info("Camera viewer and controller started.")

        self.timer = self.create_timer(self.send_msg_timer, self.set_speed)
        self.timer = self.create_timer(self.main_timer, self.control_loop)

        self.MaskRosGen = ROSMaskGenerator(15, target_colors, rgb_dict)
        # ---
        self.stitcher = cv2.Stitcher_create()

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
    
    def get_right_img(self, msg):
        try:
            self.right_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.right_camera_status = True
        except Exception as e:
            self.get_logger().error(f"[right camera] CvBridge conversion error: {e}")
            self.right_camera_status = False
            return
    

    def control_loop(self):
        if self.left_camera_img is None:
            return  

        frame_bev = perspectiveWarp(self.left_camera_img.copy())
        centroids_list, frame_vis = get_yellow_centroids(frame_bev)
        height, width = frame_bev.shape[:2]

        if len(centroids_list) > 0:
            avg_x = np.mean([c[0] for c in centroids_list])
            center_x = width / 2
            self.x = 5 * (center_x - avg_x) / center_x
            
            v_max = 3.0
            v_min = 0.5
            k = 0.7
            v = v_max * (1 - k * abs(self.x))
            self.vx = np.clip(v, v_min, v_max)

            cv2.line(frame_vis, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 2)
            cv2.line(frame_vis, (int(avg_x), 0), (int(avg_x), height), (0, 0, 255), 2)
            cv2.putText(frame_vis, f"Error: {self.x:.3f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_vis, f"Speed: {self.vx:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            self.x = 0.0
            self.vx = 0.0

        mask = self.MaskRosGen.process(self.left_camera_img)

        timestamp = int(time.time() * 1000) 
        filename = f"{timestamp}"

        # cv2.imwrite(f"/home/developer/ros2_ws/src/project1/gazebo_train_data_generator/imgs/{filename}.jpg", 
        #             self.left_camera_img)

        # np.save(f"/home/developer/ros2_ws/src/project1/gazebo_train_data_generator/masks/{filename}", mask)



        out = self.PID.output(self.x)
        msg_out = Twist()
        msg_out.linear.x = self.vx
        msg_out.linear.y = 0.0
        msg_out.angular.z = float(out)
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Control: v_x={self.vx:.2f}, a_z={out:.4f}")

        cv2.imshow("mask", mask)

        cv2.imshow("Bird eye", frame_vis)
        cv2.waitKey(1)



def main():
    rclpy.init()
    node = ControllerNode(rgb_dict)
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()