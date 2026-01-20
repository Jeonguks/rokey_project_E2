#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__("cmdvel_pub")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz
        self.v = 0.4
        self.w = 0.0

        self.get_logger().info("Publishing /cmd_vel at 10 Hz")

    def tick(self):
        msg = Twist()
        msg.linear.x = float(self.v)
        msg.angular.z = float(self.w)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
