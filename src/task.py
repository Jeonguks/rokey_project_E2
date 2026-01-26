#!/usr/bin/env python3
import time
import os
import subprocess
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "finger_joint",
]

# ====================================
ROS_SETUP = "/opt/ros/humble/setup.bash"
WS_SETUP  = os.path.expanduser("~/ros2_ws/install/setup.bash")  # 없으면 "" 로 두기
YOLO_SCRIPT = "/home/rokey/Documents/project/rokey_project_E2/src/object_detector/yolo_isaacsim_node.py"
# ===============================================================


# rad
OPEN_GRIPPER_ANGLE = 0.00000
CLOSE_GRIPPER_ANGLE = 0.43633

INIT_POSITION = [0.00000, -1.57080, 0.00000, -1.57080, 1.57080, 3.14159, OPEN_GRIPPER_ANGLE]

SCENARIO = [
    ("init", INIT_POSITION, 5.0),

    ("approach", [0.00000, -1.39626, -0.69813, -1.39626, 1.67551, 3.14159, OPEN_GRIPPER_ANGLE], 5.0),

    ("pre_grasp_1", [0.00000, -1.16064, -1.21999, -0.75398, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),
    ("pre_grasp_2", [0.00000, -1.27758, -1.21999, -0.75398, 1.34041, 3.24631, OPEN_GRIPPER_ANGLE], 5.0),

    ("grasp", [0.00000, -1.27758, -1.21999, -0.75398, 1.34041, 3.24631, CLOSE_GRIPPER_ANGLE], 5.0),

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
        super().__init__("joint_scenario_runner")
        self.pub = self.create_publisher(JointState, "/joint_command", 10)

        self._corr_msg = None
        self._corr_time = None
        self.create_subscription(TwistStamped, "/vision_correction", self._on_corr, 10)

        self._target_msg = None
        self._target_time = None
        self.create_subscription(String, "/stock/target_class", self._on_target, 10)

        self.PRE_GRASP_TASK_TIMEOUT = 15.0

        self._yolo_proc = None
        self._yolo_log_thread = None
        self._yolo_stop_log = threading.Event()

    def _on_target(self, msg: String):
        self._target_msg = msg.data
        self._target_time = time.time()

    def _on_corr(self, msg: TwistStamped):
        self._corr_msg = msg
        self._corr_time = time.time()

    def send_joint_target(self, positions):
        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = positions
        self.pub.publish(msg)

    # -------- YOLO 로그를 실시간으로 찍는 쓰레드 --------
    def _yolo_log_worker(self):
        try:
            while (not self._yolo_stop_log.is_set()) and self._yolo_proc and self._yolo_proc.stdout:
                line = self._yolo_proc.stdout.readline()
                if not line:
                    break
                self.get_logger().info(f"[YOLO] {line.rstrip()}")
        except Exception as e:
            self.get_logger().error(f"[YOLO] log thread error: {e}")

    def start_yolo_node(self):
        if self._yolo_proc is not None and self._yolo_proc.poll() is None:
            self.get_logger().info("[YOLO] already running")
            return

        if not os.path.isfile(YOLO_SCRIPT):
            self.get_logger().error(f"[YOLO] script not found: {YOLO_SCRIPT}")
            return
        if not os.path.isfile(ROS_SETUP):
            self.get_logger().error(f"[YOLO] ROS setup not found: {ROS_SETUP}")
            return

        ws_source = ""
        if WS_SETUP:
            if os.path.isfile(WS_SETUP):
                ws_source = f"source '{WS_SETUP}' && "
            else:
                self.get_logger().warn(f"[YOLO] WS setup not found (skip): {WS_SETUP}")

        # ros환경설정
        cmd = (
            "bash -lc "
            f"\"source '{ROS_SETUP}' && "
            f"{ws_source}"
            f"python3 '{YOLO_SCRIPT}'\""
        )

        self.get_logger().info(f"[YOLO] start cmd: {cmd}")

        self._yolo_stop_log.clear()
        self._yolo_proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 로그 쓰레드 시작
        self._yolo_log_thread = threading.Thread(target=self._yolo_log_worker, daemon=True)
        self._yolo_log_thread.start()

    def stop_yolo_node(self):
        self._yolo_stop_log.set()
        if self._yolo_proc is None:
            return
        if self._yolo_proc.poll() is None:
            self.get_logger().info("[YOLO] stopping...")
            self._yolo_proc.terminate()
            try:
                self._yolo_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.get_logger().warn("[YOLO] kill")
                self._yolo_proc.kill()

        rc = self._yolo_proc.poll()
        self.get_logger().warn(f"[YOLO] stopped (returncode={rc})")
        self._yolo_proc = None

    # --------------------------------------------------

    def apply_corr_to_joint_target(self, base_target, corr: TwistStamped):
        new_target = list(base_target)
        dy = corr.twist.linear.y

        K_YAW = 1.0
        yaw_delta = -K_YAW * dy

        MAX_YAW_STEP = 0.15
        yaw_delta = max(-MAX_YAW_STEP, min(MAX_YAW_STEP, yaw_delta))

        new_target[0] = float(new_target[0] + yaw_delta)
        return new_target

    def task_goto_init(self):
        self.get_logger().warn("[RECOVERY] Go back to init")
        self.send_joint_target(INIT_POSITION)

    def task_on_pre_grasp(self, base_target, timeout_sec: float):
        self.get_logger().info(f"[TASK] pre_grasp wait correction (timeout={timeout_sec:.1f}s)")
        self._corr_msg = None
        self._corr_time = None

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

            # ✅ YOLO가 죽었으면 "왜 죽었는지" returncode 찍고 바로 실패 처리
            if self._yolo_proc is not None and self._yolo_proc.poll() is not None:
                rc = self._yolo_proc.poll()
                self.get_logger().error(f"[YOLO] process exited unexpectedly (returncode={rc})")
                return False, None

            if self._corr_msg is not None:
                corrected = self.apply_corr_to_joint_target(base_target, self._corr_msg)
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
                self.start_yolo_node()

                ok, corrected_target = self.task_on_pre_grasp(target, self.PRE_GRASP_TASK_TIMEOUT)
                if not ok:
                    self.stop_yolo_node()
                    self.task_goto_init()
                    self.get_logger().warn("[ABORT] Scenario aborted due to pre_grasp failure.")
                    return
                target = corrected_target

            start_t = time.time()
            while time.time() - start_t < timeout:
                self.send_joint_target(target)
                time.sleep(0.1)

            self.get_logger().info(f"[DONE] {step_name}")

        self.stop_yolo_node()
        self.get_logger().info("=== Scenario complete ===")


def main():
    rclpy.init()
    node = JointScenarioRunner()

    time.sleep(2.0)
    node.run_scenario()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
