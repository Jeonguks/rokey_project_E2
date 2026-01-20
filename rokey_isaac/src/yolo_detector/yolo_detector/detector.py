import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from ultralytics import YOLO


class YoloDetectorPub(Node):
    def __init__(self):
        super().__init__('yolo_detector_pub')

        # Params
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detections_topic', '/detections')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('device', 'cpu')   # '0' or 'cpu'
        self.declare_parameter('frame_id', 'camera')

        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/yolo/debug_image')

        self.image_topic = self.get_parameter('image_topic').value
        self.det_topic = self.get_parameter('detections_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf = float(self.get_parameter('conf').value)
        self.device = self.get_parameter('device').value
        self.frame_id = self.get_parameter('frame_id').value

        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)
        self.debug_topic = self.get_parameter('debug_image_topic').value

        # YOLO + CV bridge
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # Publishers/Subscribers
        self.pub = self.create_publisher(Detection2DArray, self.det_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)
        self.sub = self.create_subscription(Image, self.image_topic, self.cb, 10)

        self.get_logger().info(f"Sub: {self.image_topic}  Pub: {self.det_topic}")
        self.get_logger().info(f"Model: {self.model_path}, conf={self.conf}, device={self.device}")
        self.get_logger().info(f"Debug image pub: {self.debug_topic} (enabled={self.publish_debug})")

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        det_array = Detection2DArray()
        det_array.header = Header()
        det_array.header.stamp = msg.header.stamp
        det_array.header.frame_id = self.frame_id

        r0 = results[0]
        boxes = r0.boxes

        if boxes is not None:
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item())
                cls_id = int(b.cls[0].item())

                x1, y1, x2, y2 = xyxy
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                det = Detection2D()
                det.header = det_array.header

                bbox = BoundingBox2D()
                # center field compatibility
                try:
                    bbox.center.x = float(cx)
                    bbox.center.y = float(cy)
                except AttributeError:
                    bbox.center.position.x = float(cx)
                    bbox.center.position.y = float(cy)

                bbox.size_x = float(w)
                bbox.size_y = float(h)
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                # class_id is string in vision_msgs
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                det_array.detections.append(det)

        self.pub.publish(det_array)

        # Debug image publish
        if self.publish_debug:
            annotated = r0.plot()  # numpy(BGR)
            debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_msg.header.stamp = msg.header.stamp
            debug_msg.header.frame_id = self.frame_id
            self.debug_pub.publish(debug_msg)


def main():
    rclpy.init()
    node = YoloDetectorPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
