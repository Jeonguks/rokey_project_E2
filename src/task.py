#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
    'finger_joint'
]


 #rad

OPEN_GRIPPER_ANGLE = 0.0
CLOSE_GRIPPER_ANGLE = 0.425


SCENARIO = [
    ("init", [0.00000, -1.57080, 0.00000, -1.57080, 1.57080, 1.57080, OPEN_GRIPPER_ANGLE], 15.0),
    ("pre_grasp (open)", [-0.18151, -1.88495, -0.65798, -0.58818, 1.72264, 3.14159, OPEN_GRIPPER_ANGLE], 5.0),
    ("grasp", [-0.18151, -1.88496, -0.65799, -0.58818, 1.72264, 3.14159, CLOSE_GRIPPER_ANGLE], 5.0),
    ("lift after grasp", [-0.18151, -1.69500, -0.65799, -0.58818, 1.72264, 3.14159, CLOSE_GRIPPER_ANGLE ], 5.0),
    ("rotate base to place", [3.25000, -1.69500, -0.65799, -0.58818, 1.72264, 3.14159, CLOSE_GRIPPER_ANGLE], 8.0),
    ("approach place", [3.25000, -1.80000, -0.60000, -0.80000, 1.72264, 3.14159, CLOSE_GRIPPER_ANGLE], 8.0),
    ("lower to place", [3.25000, -2.08000, -0.60000, -0.76000, 1.72300, 3.14159, CLOSE_GRIPPER_ANGLE], 8.0),
    ("release", [3.25000, -2.08000, -0.60000, -0.76000, 1.72300, 3.14159, OPEN_GRIPPER_ANGLE], 5.0),
    ("after release", [3.25000, -2.08000, -0.60000, -0.76000, 1.72300, 3.14159, OPEN_GRIPPER_ANGLE], 8.0),
    ("init", [0.00000, -1.57080, 0.00000, -1.57080, 1.57080, 1.57080, OPEN_GRIPPER_ANGLE], 5.0)
]

class JointScenarioRunner(Node):
    def __init__(self):
        super().__init__('joint_scenario_runner')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)

    def send_joint_target(self, positions):
        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = positions
        self.pub.publish(msg)

    def run_scenario(self):
        self.get_logger().info("=== Starting joint scenario ===")

        for step_name, target, timeout in SCENARIO:
            self.get_logger().info(f"[STEP] {step_name} | timeout={timeout}s | target={target}")

            start_t = time.time()
            while time.time() - start_t < timeout:
                self.send_joint_target(target)
                time.sleep(0.1)  # 10Hz로 계속 퍼블리시

            self.get_logger().info(f"[DONE] {step_name}")

        self.get_logger().info("=== Scenario complete ===")


def main():
    rclpy.init()
    node = JointScenarioRunner()

    # 약간 대기 후 실행 (Isaac Sim 연결 안정화용)
    time.sleep(2.0)
    node.run_scenario()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
