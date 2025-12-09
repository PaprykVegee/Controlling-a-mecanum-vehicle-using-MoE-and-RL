import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


def dist_point_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def dist_to_polyline(px, py, pts):
    best = float("inf")
    for (x, y) in pts:
        d = math.hypot(px - x, py - y)
        if d < best:
            best = d
    return best


class OffroadChecker(Node):
    def __init__(self):
        super().__init__("offroad_checker")

        # Parametry
        self.declare_parameter("centerline_file", "")
        self.declare_parameter("road_half_width", 1.0)   # [m] połowa szerokości jezdni
        self.declare_parameter("margin", 0.1)            # [m] margines bezpieczeństwa
        self.declare_parameter("stop_on_offroad", True)
        self.declare_parameter("odom_topic", "/model/vehicle_blue/odometry")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        path = self.get_parameter("centerline_file").value
        if not path:
            raise RuntimeError("Brak parametru centerline_file (ścieżka do pliku x y).")

        self.centerline = self._load_xy(path)
        if len(self.centerline) < 2:
            raise RuntimeError("Plik centerline ma za mało punktów (min 2).")

        self.half_width = float(self.get_parameter("road_half_width").value)
        self.margin = float(self.get_parameter("margin").value)
        self.stop_on_offroad = bool(self.get_parameter("stop_on_offroad").value)

        self.offroad_pub = self.create_publisher(Bool, "/off_road", 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)

        odom_topic = self.get_parameter("odom_topic").value
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)

        self._last_state = None
        self.get_logger().info(
            f"Loaded centerline: {len(self.centerline)} pts, half_width={self.half_width}, margin={self.margin}"
        )

    def _load_xy(self, path: str):
        # Obsługuje spacje / taby / przecinki
        pts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                x, y = float(parts[0]), float(parts[1])
                pts.append((x, y))
        return pts

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        d = dist_to_polyline(x, y, self.centerline)
        limit = max(0.0, self.half_width - self.margin)
        off = d > limit

        # Publikuj tylko gdy zmienił się stan (mniej spamu w logach)
        if self._last_state is None or off != self._last_state:
            self._last_state = off
            self.offroad_pub.publish(Bool(data=off))
            if off:
                self.get_logger().warn(f"OFF ROAD: d={d:.3f} > {limit:.3f}  (x={x:.2f}, y={y:.2f})")
                if self.stop_on_offroad:
                    self.cmd_pub.publish(Twist())  # zero twist
            else:
                self.get_logger().info(f"ON ROAD: d={d:.3f} <= {limit:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = OffroadChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == "__main__":
    main()
