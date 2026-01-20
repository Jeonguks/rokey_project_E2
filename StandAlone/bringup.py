#!/usr/bin/env python3
import threading
import time

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # GUI 켜기

import numpy as np

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid

# (Isaac Sim 버전에 따라 모듈 위치가 조금 다를 수 있어 try 처리)
try:
    from isaacsim.core.utils.physics import (
        set_rigid_body_linear_velocity,
        set_rigid_body_angular_velocity,
    )
except Exception:
    # 구버전/환경 차이 대비: physics util이 없으면 아래에서 안내하고 종료
    set_rigid_body_linear_velocity = None
    set_rigid_body_angular_velocity = None


# -----------------------------
# ROS2 (/cmd_vel) subscriber
# -----------------------------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__("isaacsim_cmdvel_subscriber")
        self._lock = threading.Lock()
        self._latest = (0.0, 0.0)  # (linear_x [m/s], angular_z [rad/s])
        self.create_subscription(Twist, "/cmd_vel", self._cb, 10)

    def _cb(self, msg: Twist):
        with self._lock:
            self._latest = (float(msg.linear.x), float(msg.angular.z))

    def get_latest(self):
        with self._lock:
            return self._latest


def spin_ros(node: Node):
    rclpy.spin(node)


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) World 생성
    world = World(stage_units_in_meters=1.0)

    # 2) USD 로드 (Reference)
    # canifixit.usd의 "파일 경로"를 절대경로로 넣는 것을 권장
    usd_path = "/home/rokey/Documents/project/canifixit.usd"

    cart_root_prim_path = "/World/nove_iwill_broken_you/fancy_cart"
    add_reference_to_stage(usd_path, cart_root_prim_path)

    # 3) 카트에서 실제로 velocity를 적용할 rigid body prim 경로 지정
    #    USD 구조에 따라 달라지므로, 보통 base 링크/바디 prim을 지정해야 합니다.
    #    예: "/World/Cart/base_link" 또는 "/World/Cart/chassis"
    cart_rigid_body_path = (
        "/World/nove_iwill_broken_you/fancy_cart/fancy_cart/chassis_link/base_link"
    )

    # 4) 로드 확인
    world.reset()

    if not is_prim_path_valid(cart_rigid_body_path):
        print(f"[ERROR] Rigid body prim path not found: {cart_rigid_body_path}")
        print(
            "        USD 내부에서 base rigid body prim 경로를 확인해서 cart_rigid_body_path를 수정하세요."
        )
        simulation_app.close()
        return

    if set_rigid_body_linear_velocity is None:
        print("[ERROR] physics velocity utils not available in this environment.")
        print("        isaacsim.core.utils.physics 모듈 제공 여부/버전을 확인하세요.")
        simulation_app.close()
        return

    cart_prim = get_prim_at_path(cart_rigid_body_path)

    # 5) ROS2 초기화 + /cmd_vel subscriber
    rclpy.init(args=None)
    cmd_node = CmdVelSubscriber()
    ros_thread = threading.Thread(target=spin_ros, args=(cmd_node,), daemon=True)
    ros_thread.start()

    # 6) 시뮬레이션 루프에서 /cmd_vel을 읽어 rigid body에 적용
    #    선속도는 카트의 local x축 기준 전진으로 가정(월드 좌표로 단순 적용)
    #    필요 시 yaw 방향에 따라 회전행렬로 변환해야 함.
    try:
        while simulation_app.is_running():
            world.step(render=True)

            v, w = cmd_node.get_latest()  # v: m/s, w: rad/s

            # 여기서는 "월드 좌표" 기준으로 X 전진 + Z 회전으로 단순 적용
            # 카트가 회전된 상태에서 local 기준으로 움직이려면 방향 변환이 필요합니다.
            linear_vel_world = np.array([v, 0.0, 0.0], dtype=np.float32)
            angular_vel_world = np.array([0.0, 0.0, w], dtype=np.float32)

            set_rigid_body_linear_velocity(cart_prim, linear_vel_world)
            set_rigid_body_angular_velocity(cart_prim, angular_vel_world)

    except KeyboardInterrupt:
        pass
    finally:
        # ROS2 종료
        cmd_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
