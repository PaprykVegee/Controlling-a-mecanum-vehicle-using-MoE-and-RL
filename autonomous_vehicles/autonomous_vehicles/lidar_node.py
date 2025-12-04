from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import rclpy
from rclpy.node import Node
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import cv2


def lidar_to_image(points, scale=50, img_size=500):
    """
    Konwertuje chmurę punktów 3D na obraz 2D do wyświetlenia w OpenCV.
    points: numpy array [N,3]
    scale: ile pikseli na 1 metr
    img_size: rozmiar obrazu w pikselach (kwadrat)
    """
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    x = points[:, 0]
    y = points[:, 1]

    px = (x * scale + img_size // 2).astype(np.int32)
    py = (y * scale + img_size // 2).astype(np.int32)

    mask = (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    px = px[mask]
    py = py[mask]

    img[py, px] = 255  # biały punkt

    img = cv2.flip(img, 0)
    return img

class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")

        self.subscription = self.create_subscription(
            PointCloud2,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/scan/points",
            self.lidar_callback,
            10
        )

        self.lidar_pub = self.create_publisher(
            Float32MultiArray,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/tensor",
            10
        )

    def lidar_callback(self, msg: PointCloud2):

        raw_points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))

        if not raw_points:
            return

        points = np.array([[p[0], p[1], p[2]] for p in raw_points], dtype=np.float32)

        #self.get_logger().info(f"Otrzymano LiDAR 3D: {points.shape[0]} punktów")

        arr = Float32MultiArray()
        arr.data = points.flatten().tolist()

        img = lidar_to_image(points)
        cv2.imshow("LiDAR top-down", img)
        cv2.waitKey(1)
        self.lidar_pub.publish(arr)



def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
