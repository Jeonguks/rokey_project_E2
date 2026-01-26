#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point, TwistStamped

import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class YoloIsaacSimNode(Node):
    def __init__(self):
        super().__init__('yolo_isaacsim_node')

        # 1) 모델 로드
        self.get_logger().info("⏳ YOLO 모델 로딩 중...")
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        WEIGHTS = os.path.join(BASE_DIR, "best.pt")
        self.model = YOLO(WEIGHTS)
        self.bridge = CvBridge()

        # ==========================================
        # Isaac Sim 설정 (EDIT HERE)
        # ==========================================
        self.RGB_TOPIC = "/camera/rgb"
        self.DEPTH_TOPIC = "/camera/depth"

        # Isaac Sim 카메라 Horizontal FOV (deg) - 카메라 속성에서 확인 권장
        self.H_FOV_DEG = 60.0

        # 타깃 클래스 (result.names와 정확히 일치해야 함)
        self.TARGET_CLASS = "Coke"

        # depth 유효 범위
        self.MIN_DEPTH = 0.10
        self.MAX_DEPTH = 5.00

        # ee_link 기준으로 변환할 때 카메라 오프셋(단순 Z 오프셋)
        self.CAMERA_OFFSET_Z = 0.08
        # ==========================================

        # 2) Publisher
        self.target_pub = self.create_publisher(Point, '/goal_point', 10)
        self.corr_pub = self.create_publisher(TwistStamped, '/vision_correction', 10)

        # 3) Subscriber
        # 외부에서 타깃 클래스 지정하는 토픽
        self.STOCK_TARGET_TOPIC = "/stock/target_class"
        self.create_subscription(String, self.STOCK_TARGET_TOPIC, self.on_target_class, 10)
        self.get_logger().info(f"   Sub: {self.STOCK_TARGET_TOPIC} (String) -> updates TARGET_CLASS")


        sub_rgb = message_filters.Subscriber(self, Image, self.RGB_TOPIC)
        sub_depth = message_filters.Subscriber(self, Image, self.DEPTH_TOPIC)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth],
            queue_size=10,
            slop=0.2
        )
        self.ts.registerCallback(self.callback)

        self.get_logger().info("🚀 Isaac Sim YOLO 3D 노드 시작")
        self.get_logger().info(f"   Sub: {self.RGB_TOPIC}, {self.DEPTH_TOPIC}")
        self.get_logger().info(f"   Target class: {self.TARGET_CLASS}")
        self.get_logger().info("   Pub: /vision_correction (TwistStamped), /goal_point (Point)")


    def on_target_class(self, msg: String):
        new_target = (msg.data or "").strip()
        if not new_target:
            self.get_logger().warn("[target_class] empty string received; ignoring")
            return

        if new_target != self.TARGET_CLASS:
            self.TARGET_CLASS = new_target
            self.get_logger().info(f"[target_class] TARGET_CLASS updated -> '{self.TARGET_CLASS}'")


    def callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            height, width, _ = cv_image.shape
            cx = width / 2.0
            cy = height / 2.0

            # pinhole: fx from HFOV (근사)
            fx = width / (2.0 * np.tan(np.deg2rad(self.H_FOV_DEG) / 2.0))
            fy = fx

            # YOLO 추론
            results = self.model.predict(cv_image, imgsz=1024, conf=0.3, verbose=False)

            # ---------------------------------------------------------
            # (1) 모든 bbox를 빨간색으로 시각화 + all_boxes 수집
            # ---------------------------------------------------------
            all_boxes = []  # (x1,y1,x2,y2,u,v,label,conf)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0

                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    u = (x1 + x2) / 2.0
                    v = (y1 + y2) / 2.0

                    all_boxes.append((x1, y1, x2, y2, u, v, label, conf))

                    # 빨간 bbox (모든 객체)
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        cv_image, f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                    )

            # ---------------------------------------------------------
            # (2) 타깃 클래스 + 중앙 기준으로 best bbox 선택
            #     (depth 유효한 bbox만 후보로 유지하는게 안정적)
            # ---------------------------------------------------------
            best = None
            best_dist = float("inf")

            h, w = depth_image.shape
            for (x1, y1, x2, y2, u, v, label, conf) in all_boxes:
                if label != self.TARGET_CLASS:
                    continue

                uu = int(np.clip(u, 0, w - 1))
                vv = int(np.clip(v, 0, h - 1))

                z_depth = float(depth_image[vv, uu])
                if not np.isfinite(z_depth):
                    continue
                if not (self.MIN_DEPTH < z_depth < self.MAX_DEPTH):
                    continue

                dist = (u - cx) ** 2 + (v - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (x1, y1, x2, y2, u, v, z_depth, conf)

            # 타깃 클래스가 없거나 depth 유효 후보가 없으면 빨간 bbox만 보여주고 종료
            if best is None:
                cv2.circle(cv_image, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                cv2.imshow("IsaacSim YOLO View", cv_image)
                cv2.waitKey(1)
                return

            x1, y1, x2, y2, u, v, z_depth, conf = best
            u = float(u)
            v = float(v)

            # ---------------------------------------------------------
            # (3) 선택된 타깃 bbox를 노란색으로 강조
            # ---------------------------------------------------------
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            cv2.putText(
                cv_image, f"TARGET:{self.TARGET_CLASS} {conf:.2f}",
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            cv2.circle(cv_image, (int(u), int(v)), 5, (0, 0, 255), -1)   # 타깃 중심
            cv2.circle(cv_image, (int(cx), int(cy)), 5, (255, 0, 0), -1) # 화면 중심

            # ---------------------------------------------------------
            # (4) 3D 위치 추정 (camera frame)
            #     X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy, Z = depth
            # ---------------------------------------------------------
            cam_x = (u - cx) * z_depth / fx
            cam_y = (v - cy) * z_depth / fy
            cam_z = z_depth

            # ee_link 기준 단순 오프셋 (필요시 축/부호 조정)
            real_x = cam_x
            real_y = cam_y
            real_z = cam_z + self.CAMERA_OFFSET_Z

            # ---------------------------------------------------------
            # (5) /goal_point 발행 (3D 근사)
            # ---------------------------------------------------------
            goal = Point()
            goal.x = float(real_x)
            goal.y = float(real_y)
            goal.z = float(real_z)
            self.target_pub.publish(goal)

            # ---------------------------------------------------------
            # (6) /vision_correction 발행
            #     - 미세조정용 dx,dy를 "미터"로 출력 (depth 기반)
            #     - dyaw는 추정 로직 없으므로 0
            # ---------------------------------------------------------
            # 주의: dx,dy는 "카메라 프레임에서의 오차"입니다.
            # Isaac 쪽에서 world XY로 바로 쓰려면 frame 변환이 필요합니다.
            dx = float(cam_x)
            dy = float(cam_y)
            dyaw = 0.0

            corr = TwistStamped()
            corr.header.stamp = self.get_clock().now().to_msg()
            corr.header.frame_id = "gripper_camera"  # 의미용
            corr.twist.linear.x = dx
            corr.twist.linear.y = dy
            corr.twist.linear.z = 0.0
            corr.twist.angular.x = 0.0
            corr.twist.angular.y = 0.0
            corr.twist.angular.z = float(dyaw)
            self.corr_pub.publish(corr)

            # 화면 표시 텍스트
            info = f"Z={z_depth:.2f}m cam_x={cam_x:.3f} cam_y={cam_y:.3f}"
            cv2.putText(
                cv_image, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            self.get_logger().info(
                f"[TARGET:{self.TARGET_CLASS}] Z={z_depth:.3f} "
                f"cam=(x={cam_x:.3f}, y={cam_y:.3f}) ee~=(x={real_x:.3f}, y={real_y:.3f}, z={real_z:.3f}) "
                f"dist={best_dist:.1f}"
            )

            cv2.imshow("IsaacSim YOLO View", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")


def main():
    rclpy.init()
    node = YoloIsaacSimNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
