from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
import numpy as np
import math


class GpsNode(Node):
    def __init__(self):
        super().__init__("gps_node")

        self.create_subscription(
            NavSatFix,
            "/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/navsat",
            self.gps_callback,
            10
        )

        self.gps_pub = self.create_publisher(
            PointStamped,                        
			"/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/position_xyz",
            10)

        self.origin_lat = 52.0
        self.origin_lon = 19.0
        self.origin_alt = 100.0

        self.lat0 = math.radians(self.origin_lat)
        self.lon0 = math.radians(self.origin_lon)


	# to ma transformowac dane gps na cordy x y z
    def gps_callback(self, msg: NavSatFix):
        lat = math.radians(msg.latitude)
        lon = math.radians(msg.longitude)
        alt = msg.altitude

        a = 6378137.0
        e2 = 6.69437999014e-3

        N = a / math.sqrt(1 - e2 * (math.sin(lat) ** 2))

        X = (N + alt) * math.cos(lat) * math.cos(lon)
        Y = (N + alt) * math.cos(lat) * math.sin(lon)
        Z = (N * (1 - e2) + alt) * math.sin(lat)

        N0 = a / math.sqrt(1 - e2 * (math.sin(self.lat0) ** 2))
        X0 = (N0 + self.origin_alt) * math.cos(self.lat0) * math.cos(self.lon0)
        Y0 = (N0 + self.origin_alt) * math.cos(self.lat0) * math.sin(self.lon0)
        Z0 = (N0 * (1 - e2) + self.origin_alt) * math.sin(self.lat0)

        dX = X - X0
        dY = Y - Y0
        dZ = Z - Z0

        t = self.lat0
        p = self.lon0

        e = -math.sin(p) * dX + math.cos(p) * dY
        n = -math.sin(t) * math.cos(p) * dX - math.sin(t) * math.sin(p) * dY + math.cos(t) * dZ
        u =  math.cos(t) * math.cos(p) * dX + math.cos(t) * math.sin(p) * dY + math.sin(t) * dZ

        out = PointStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"  

        out.point.x = float(e)
        out.point.y = float(n)
        out.point.z = float(u)

        self.gps_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GpsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()