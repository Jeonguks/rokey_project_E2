#!/usr/bin/env python3
import os
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class StockTargetPublisher(Node):
    """
    YAML 재고 파일을 주기적으로 읽고,
    임계값(threshold)보다 낮은 품목 중 '가장 부족한 품목'을 target_class로 1Hz publish.
    """

    def __init__(self):
        super().__init__("stock_target_publisher")

        # -------------------------
        # 설정
        # -------------------------
        # 1) 실행 디렉토리 기준 파일 
        self.yaml_path = os.path.abspath("stock.yaml")

        # 홈 디렉토리 기준 파일을 쓰고 싶으면 아래로 교체:
        # self.yaml_path = os.path.expanduser("~/stock.yaml")

        self.topic = "/stock/target_class"
        self.publish_hz = 1.0  # 1Hz
        self.publish_none = True  # 부족 품목 없을 때 "NONE"도 계속 보낼지 여부
        # -------------------------

        self.pub = self.create_publisher(String, self.topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_hz, self.tick)

        self.get_logger().info(
            f"Stock target publisher started.\n"
            f"  YAML:  {self.yaml_path}\n"
            f"  Topic: {self.topic}\n"
            f"  Rate:  {self.publish_hz} Hz"
        )

    def load_yaml(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML not found: {self.yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        items = data.get("items", {}) or {}
        thresholds = data.get("thresholds", {}) or {}

        # 타입 정리
        items = {str(k): int(v) for k, v in items.items()}
        thresholds = {str(k): int(v) for k, v in thresholds.items()}

        return items, thresholds

    def choose_target(self, items, thresholds):
        """
          - effective_threshold(item) = thresholds[item] if exists else thresholds.default
          - deficit = th - qty
          - deficit > 0 인 후보 중 deficit이 가장 큰 품목 선택
        """
        default_th = thresholds.get("default", 0)

        candidates = []
        for name, qty in items.items():
            th = thresholds.get(name, default_th)
            deficit = th - qty
            if deficit > 0:
                candidates.append((deficit, name, qty, th))

        if not candidates:
            return None, None

        # deficit 큰 순, 동률이면 이름 오름차순(원하면 우선순위 규칙으로 변경 가능)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        deficit, name, qty, th = candidates[0]
        return name, (deficit, qty, th)

    def tick(self):
        try:
            items, thresholds = self.load_yaml()
            target, info = self.choose_target(items, thresholds)

            if target is None:
                if self.publish_none:
                    self.pub.publish(String(data="NONE"))
                self.get_logger().info("No low-stock items. Published target_class=NONE")
                return

            deficit, qty, th = info

            # 1초마다 항상 publish
            self.pub.publish(String(data=target))
            self.get_logger().info(
                f"Published target_class={target} (deficit={deficit}, qty={qty}, th={th})"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to publish target: {e}")


def main():
    rclpy.init()
    node = StockTargetPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
