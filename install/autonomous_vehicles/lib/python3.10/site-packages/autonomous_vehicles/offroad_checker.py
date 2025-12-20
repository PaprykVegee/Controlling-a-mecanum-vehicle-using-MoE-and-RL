import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
import cv2
from std_msgs.msg import Float32

# Parametry obrazu
img_size = 600
margin_px = 50  # UWAGA: to są piksele do rysowania (żeby nie mieszać z margin w [m])

def draw_gt(centerline, vehicle_pos, half_width_m, margin_m, offroad):
    vx, vy = vehicle_pos

    d = dist_to_polyline(vx, vy, centerline)

    xs = [p[0] for p in centerline] + [vx]
    ys = [p[1] for p in centerline] + [vy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale_x = (img_size - 2 * margin_px) / (max_x - min_x + 1e-6)
    scale_y = (img_size - 2 * margin_px) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # Rysuj ścieżkę
    for i in range(len(centerline) - 1):
        pt1 = world_to_image(*centerline[i], center_x, center_y, scale)
        pt2 = world_to_image(*centerline[i + 1], center_x, center_y, scale)
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # Granice drogi
    offset_m = half_width_m + margin_m
    for i in range(len(centerline) - 1):
        x1, y1 = centerline[i]
        x2, y2 = centerline[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx, ny = -dy / length * offset_m, dx / length * offset_m

        left1  = world_to_image(x1 + nx, y1 + ny, center_x, center_y, scale)
        left2  = world_to_image(x2 + nx, y2 + ny, center_x, center_y, scale)
        right1 = world_to_image(x1 - nx, y1 - ny, center_x, center_y, scale)
        right2 = world_to_image(x2 - nx, y2 - ny, center_x, center_y, scale)

        cv2.line(img, left1, left2, (200, 200, 255), 1)
        cv2.line(img, right1, right2, (200, 200, 255), 1)

    color = (0, 0, 255) if offroad else (0, 200, 0)
    v_pt = world_to_image(vx, vy, center_x, center_y, scale)
    cv2.circle(img, v_pt, 5, color, -1)

    cv2.putText(img, f"Error: {d:.2f} m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

    cv2.imshow("Ground Truth", img)
    cv2.waitKey(1)


def world_to_image(x, y, center_x, center_y, scale):
    ix = int(img_size / 2 + (x - center_x) * scale)
    iy = int(img_size / 2 - (y - center_y) * scale)
    return ix, iy


def dist_to_polyline(px, py, pts):
    # (Twoja wersja: min dystans do punktów; działa jako szybki check)
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
        self.declare_parameter("road_half_width", 1.0)   # [m]
        self.declare_parameter("margin", 0.1)            # [m]
        self.declare_parameter("stop_on_offroad", True)
        self.declare_parameter("pose_topic", "/model/vehicle_blue/pose")
        self.declare_parameter("frame_contains", "vehicle_blue")  # dopasuj po echo
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        path = self.get_parameter("centerline_file").value
        if not path:
            raise RuntimeError("Brak parametru centerline_file (ścieżka do pliku x y).")

        self.centerline = self._load_xy(path)
        if len(self.centerline) < 2:
            raise RuntimeError("Plik centerline ma za mało punktów (min 2).")

        self.half_width = float(self.get_parameter("road_half_width").value)
        self.margin_m = float(self.get_parameter("margin").value)
        self.stop_on_offroad = bool(self.get_parameter("stop_on_offroad").value)

        self.pose_topic = self.get_parameter("pose_topic").value
        self.frame_contains = self.get_parameter("frame_contains").value

        self.offroad_pub = self.create_publisher(Bool, "/off_road", 10)
        self.error_pub = self.create_publisher(Float32, "/groundtruth_error", 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)

        # SUB: TFMessage zamiast Odometry
        self.create_subscription(TFMessage, self.pose_topic, self.pose_cb, 10)

        self._last_state = None
        self.get_logger().info(
            f"Loaded centerline: {len(self.centerline)} pts, half_width={self.half_width}, margin={self.margin_m}\n"
            f"Listening pose TF topic: {self.pose_topic}, frame_contains='{self.frame_contains}'"
        )

    def _load_xy(self, path: str):
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

    def pose_cb(self, msg: TFMessage):
        # TFMessage.transforms: lista TransformStamped
        chosen = None
        for tr in msg.transforms:
            # Najczęściej auto siedzi w child_frame_id
            cid = tr.child_frame_id or ""
            fid = tr.header.frame_id or ""
            if (self.frame_contains in cid) or (self.frame_contains in fid):
                chosen = tr
                break

        if chosen is None:
            # jak nie znalazł — możesz na start odkomentować log co przychodzi:
            # self.get_logger().warn("Nie znaleziono transformu dla auta w TFMessage")
            return

        x = chosen.transform.translation.x
        y = chosen.transform.translation.y

        d = dist_to_polyline(x, y, self.centerline)
        limit = max(0.0, self.half_width - self.margin_m)
        off = d > limit

        draw_gt(self.centerline, (x, y), self.half_width, self.margin_m, off)

        self.offroad_pub.publish(Bool(data=off))  # ZAWSZE

        self.error_pub.publish(Float32(data=d))

        if self._last_state is None or off != self._last_state:
            self._last_state = off
            if off:
                self.get_logger().warn(f"OFF ROAD: d={d:.3f} > {limit:.3f}  (x={x:.2f}, y={y:.2f})")
                if self.stop_on_offroad:
                    self.cmd_pub.publish(Twist())
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
