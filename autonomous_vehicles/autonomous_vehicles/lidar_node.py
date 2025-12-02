from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import rclpy
from rclpy.node import Node
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class LidarNode(Node):
	def __init__(self):
		super().__init__("lidar_node")

		# Subskrypcja chmury 3D LiDAR
		self.create_subscription(
			PointCloud2,
			"/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/points",
			self.lidar_callback,
			10
		)

		# Publikacja x,y,z jako spłaszczony Float32MultiArray
		self.lidar_pub = self.create_publisher(
			Float32MultiArray,
			"/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/tensor",
			10
		)

	def lidar_callback(self, msg: PointCloud2):
		# konwersja PointCloud2 -> ndarray [N,3]
		points = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)), dtype=np.float32)
		if points.size == 0:
			return

		arr_msg = Float32MultiArray()
		arr_msg.data = points.flatten().tolist()
		self.lidar_pub.publish(arr_msg)

		self.get_logger().info(f"[LiDAR3D] Publikowano {points.shape[0]} punktów")


def main(args=None):
	rclpy.init(args=args)
	node = LidarNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()
