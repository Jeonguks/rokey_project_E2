import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.core.objects import DynamicCuboid


class MoveFrankaStandalone:
    def __init__(self):
        # ===== 사용자 환경에 맞게 수정 =====
        self.MAP_USD_PATH = "/home/rokey/Documents/project/test_world_origin.usd"
        self.MAP_PRIM_PATH = "/World/Map"
        self.FRANKA_PRIM_PATH = "/World/Fancy_Franka"
        self.FRANKA_POSITION = np.array([0.0, 1.0, 0.0])
        # ==================================

        # 큐브(집을 물체)
        self.CUBE_PRIM_PATH = "/World/DynamicCube"
        self.CUBE_SIZE = 0.05
        self.CUBE_START_POS = np.array([0.55, 0.55, 0.75])  # 공중에 두면 떨어지면서 잘 보임

        # Place 위치(놓을 위치)
        self.PLACE_POS = np.array([0.20, 0.20, 0.75])

        # (선택) 엔드이펙터 방향 - Franka 기본 grasp에 맞춰 필요시 조정
        self.EE_QUAT = euler_angles_to_quat([0.0, np.pi, 0.0])

        self._world = None
        self._franka = None
        self._cube = None
        self._pickplace = None

        self._done_printed = False

    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)

        if not os.path.isfile(self.MAP_USD_PATH):
            raise FileNotFoundError(self.MAP_USD_PATH)

        add_reference_to_stage(self.MAP_USD_PATH, self.MAP_PRIM_PATH)

        self._franka = self._world.scene.add(
            Franka(
                prim_path=self.FRANKA_PRIM_PATH,
                name="fancy_franka",
                position=self.FRANKA_POSITION,
            )
        )

        # Pick 대상: DynamicCuboid
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.CUBE_PRIM_PATH,
                name="dynamic_cube",
                position=self.CUBE_START_POS,
                size=self.CUBE_SIZE,
            )
        )

    def setup_post_load(self):
        self._world.reset()

        # PickPlaceController 생성
        self._pickplace = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        # 시작 시 그리퍼는 OPEN
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

        # 시뮬 콜백
        self._world.add_physics_callback("sim_step", self.physics_step)
        self._world.play()

    def _is_finished(self):
        """
        Isaac Sim 버전에 따라 완료 여부 API가 다를 수 있어서
        가능한 케이스를 넓게 커버.
        """
        # 1) 흔한 패턴: controller.is_done()
        if hasattr(self._pickplace, "is_done") and callable(self._pickplace.is_done):
            try:
                return bool(self._pickplace.is_done())
            except TypeError:
                pass

        # 2) 흔한 패턴: controller.is_done (property)
        if hasattr(self._pickplace, "is_done") and not callable(self._pickplace.is_done):
            return bool(self._pickplace.is_done)

        # 3) 예제들에서: controller.get_current_event() == "done" 류
        if hasattr(self._pickplace, "get_current_event") and callable(self._pickplace.get_current_event):
            ev = self._pickplace.get_current_event()
            if isinstance(ev, str) and ev.lower() in ["done", "finished", "complete", "completed"]:
                return True

        return False

    def physics_step(self, step_size):
        # 현재 큐브 pose를 pick 포지션으로 사용 (큐브가 떨어질 수도 있으니 매 step 갱신)
        pick_pos, _ = self._cube.get_world_pose()

        # PickPlaceController가 요구하는 입력은 버전별로 이름이 조금 다릅니다.
        # 아래는 가장 흔한 시그니처를 우선 시도하고, 실패하면 대체 시그니처를 시도합니다.
        current_joints = self._franka.get_joint_positions()

        actions = None
        try:
            # 흔한 형태
            actions = self._pickplace.forward(
                picking_position=np.array(pick_pos),
                placing_position=self.PLACE_POS,
                current_joint_positions=current_joints,
            )
        except TypeError:
            # 다른 예제들에서 쓰는 형태 (키 이름이 다른 경우)
            actions = self._pickplace.forward(
                pick_position=np.array(pick_pos),
                place_position=self.PLACE_POS,
                current_joint_positions=current_joints,
            )

        if actions is not None:
            self._franka.apply_action(actions)

        # 완료 출력(1회)
        if (not self._done_printed) and self._is_finished():
            self._done_printed = True
            print("[DONE] Pick & Place finished")


def main():
    app = MoveFrankaStandalone()
    app.setup_scene()
    app.setup_post_load()

    while simulation_app.is_running():
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
