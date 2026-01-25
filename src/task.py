#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped 
from std_msgs.msg import String



JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
    'finger_joint'
]


# rad
OPEN_GRIPPER_ANGLE  = 0.00000
CLOSE_GRIPPER_ANGLE = 0.43633

INIT_POSITION = [0.00000, -1.57080, 0.00000, -1.57080, 1.57080, 3.14159, OPEN_GRIPPER_ANGLE]

SCENARIO = [
    ("init", INIT_POSITION, 5.0),

    ("approach", [0.00000, -1.39626, -0.69813, -1.39626, 1.67551, 3.14159, OPEN_GRIPPER_ANGLE], 5.0),

    ("pre_grasp_1", [0.00000, -1.16064, -1.21999, -0.75398, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),
    ("pre_grasp_2", [0.00000, -1.27758, -1.21999, -0.75398, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),

    ("grasp", [0.00000, -1.27758, -1.21999, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 5.0),

    # 네가 준 "후퇴"랑 "후퇴2"를 그대로 단계로 분리
    ("retreat_1", [0.00000, -1.27758, -1.21999, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 5.0),
    ("retreat_2", [0.00000, -0.92852, -1.21999, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 5.0),

    ("rotate", [-2.90597, -0.92852, -1.21999, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),

    ("place_approach_1", [-2.90597, -0.75398, -1.71566, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),
    ("place_approach_2", [-2.90597, -1.33518, -1.71566, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),
    ("place_approach_3", [-2.90597, -1.33518, -1.71566, -0.52185, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),

    ("place_align_0", [-2.90597, -1.50971, -1.71566, -0.28798, 1.63188, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),
    ("place_align",   [-2.90597, -1.50971, -1.71566, -0.28798, 1.68948, 3.24631, CLOSE_GRIPPER_ANGLE], 8.0),

    ("release_open", [-2.90597, -1.50971, -1.71566, -0.28798, 1.68948, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),

    ("retreat_back_1", [-2.90597, -1.50971, -1.71566, -0.28798, 1.63188, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),
    ("retreat_back_2", [-2.90597, -1.33518, -1.71566, -0.52185, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),
    ("retreat_back_3", [-2.90597, -1.33518, -1.71566, -0.75398, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),

    ("init", INIT_POSITION, 5.0),
]

class JointScenarioRunner(Node):
    def __init__(self):
        super().__init__('joint_scenario_runner')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)

        # 보정자세 수신
        self._corr_msg = None
        self._corr_time = None
        self.create_subscription(TwistStamped, '/vision_correction', self._on_corr, 10)


        # 타겟 수신
        self._target_msg = None
        self._target_time = None
        self.create_subscription(String, "/stock/target_class", self._on_target, 10)


        self.PRE_GRASP_TASK_TIMEOUT = 15.0 # pre_grasp 태스크 타임아웃(초)


    # 타겟 수신 콜백
    def _on_target(self, msg: String):
        self._target_msg = msg.data
        self._target_time = time.time()
        
    # 보정자세 수신 콜백
    def _on_corr(self, msg: TwistStamped):
        self._corr_msg = msg
        self._corr_time = time.time()


    def send_joint_target(self, positions):
        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = positions
        self.pub.publish(msg)


    def apply_corr_to_joint_target(self, base_target, corr: TwistStamped):
        """
        corr.twist.linear.x/y (m)를 관절각(rad)로 '근사 변환'하여 base_target을 수정.
        - 여기서는 가장 안전한 방식으로 base yaw만 보정(간이)
        """
        new_target = list(base_target)

        dx = corr.twist.linear.x
        dy = corr.twist.linear.y

        # 근사 게인: dy(좌우 오차)가 +면 yaw를 -로 돌려서 중앙으로 맞춘다고 가정
        # 값은 반드시 튜닝 필요 (예: 0.5 ~ 2.0 rad/m 사이에서 시작)
        K_YAW = 1.0   # rad per meter (튜닝)
        yaw_delta = -K_YAW * dy

        # 너무 큰 보정 방지(클램프)
        MAX_YAW_STEP = 0.15  # rad (약 8.6도)
        if yaw_delta > MAX_YAW_STEP:
            yaw_delta = MAX_YAW_STEP
        if yaw_delta < -MAX_YAW_STEP:
            yaw_delta = -MAX_YAW_STEP

        # shoulder_pan_joint가 index 0이라고 가정 (JOINT_NAMES 기준)
        new_target[0] = float(new_target[0] + yaw_delta)
        return new_target



    ## task 실패시 초기자세 이동 
    def task_goto_init(self):
        self.get_logger().warn("[RECOVERY] Go back to init")
        self.send_joint_target(INIT_POSITION)

    ## pre grasp task
    def task_on_pre_grasp(self, base_target, timeout_sec: float):
        """
        성공 시: (True, corrected_target)
        실패 시: (False, None)
        """
        self.get_logger().info(f"[TASK] pre_grasp wait correction (timeout={timeout_sec:.1f}s)")
        self._corr_msg = None
        self._corr_time = None

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._corr_msg is not None:
                corr = self._corr_msg

                corrected = self.apply_corr_to_joint_target(base_target, corr)

                self.get_logger().info(
                    f"[TASK] apply correction -> base_yaw {base_target[0]:.3f} -> {corrected[0]:.3f}"
                )
                return True, corrected

        self.get_logger().warn("[TASK] pre_grasp task TIMEOUT")
        return False, None


    def run_scenario(self):
        self.get_logger().info("=== Starting joint scenario ===")

        for step_name, target, timeout in SCENARIO:
            self.get_logger().info(f"[STEP] {step_name} | timeout={timeout}s | target={target}")

            if step_name == "pre_grasp_2":
                ok, corrected_target = self.task_on_pre_grasp(target, self.PRE_GRASP_TASK_TIMEOUT)
                if not ok:
                    self.task_goto_init()
                    self.get_logger().warn("[ABORT] Scenario aborted due to pre_grasp task timeout.")
                    return
                target = corrected_target

                
            start_t = time.time()

            while time.time() - start_t < timeout:
                self.send_joint_target(target)
                time.sleep(0.1)

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
