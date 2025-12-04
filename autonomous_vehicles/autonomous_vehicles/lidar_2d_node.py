import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class Lidar2DNode(Node):
    def __init__(self):
        super().__init__("lidar_2d_node")

        # Subskrypcja LiDAR 2D
        self.create_subscription(
            LaserScan,
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_2d_link/sensor/lidar_2d/points",
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg: LaserScan):
        # Zamiana zakresów na ndarray
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Liczba wykrytych punktów (nie-NaN)
        valid_points = np.sum(np.isfinite(ranges))
        self.get_logger().info(f"LiDAR 2D: {valid_points} punktów")

        # Dodatkowo można znaleźć minimalny dystans
        min_distance = np.nanmin(ranges)
        self.get_logger().info(f"Minimalny dystans: {min_distance:.2f} m")


def main(args=None):
    rclpy.init(args=args)
    node = Lidar2DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
