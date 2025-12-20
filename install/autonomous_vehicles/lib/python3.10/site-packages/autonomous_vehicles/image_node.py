import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import numpy as np
from autonomous_vehicles.Segmetation import *

class ImageNode(Node):
    def __init__(self):
        super().__init__("image_node")

        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image",
            self.get_img,
            10
        )

        self.mask_pub = self.create_publisher(
            Float32MultiArray,
            "/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/mask",
            10
        )
        
        self.model = EvalModel(
            r"/home/developer/ros2_ws/src/UNET_trening/best-unet-epoch=05-val_dice=0.9838.ckpt"
        )

    def get_img(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        mask = self.model.predict(img)    
        mask = mask.astype(np.float32)

        msg_out = Float32MultiArray()
        dims = []
        for i, size in enumerate(mask.shape):
            dims.append(MultiArrayDimension(label=f"dim{i}", size=size, stride=1))

        msg_out.layout.dim = dims

        msg_out.data = mask.flatten().tolist()

        self.mask_pub.publish(msg_out)
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

