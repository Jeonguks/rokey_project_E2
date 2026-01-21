import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka


class MoveJointInterpolatedFrankaStandalone:
    def __init__(self) -> None:
        self._interp_speed = 0.07
        self._threshold = 0.5

        self._target_joint_positions_1 = None
        self._target_joint_positions_2 = None
        self.task_phase = 1

        self._world = None
        self._franka = None

        # ====== 여기만 당신 환경에 맞게 수정 ======
        self.MAP_USD_PATH = "/home/rokey/Documents/project/test_world_origin.usd"  # 기존 맵 USD 경로
        self.MAP_PRIM_PATH = "/World/Map"                # 맵이 참조될 prim 경로
        self.FRANKA_PRIM_PATH = "/World/Fancy_Franka"    # 로봇 prim 경로 (맵과 겹치지 않게)
        # 맵 위에 로봇을 올릴 위치(미터 단위). 맵 원점/스케일에 맞춰 조정
        self.FRANKA_POSITION = np.array([0.0, 1.0, 0.0])
        # ======================================

    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)

        # 1) 기존 맵 USD를 Stage에 참조로 붙임
        if not os.path.exists(self.MAP_USD_PATH):
            raise FileNotFoundError(f"MAP_USD_PATH not found: {self.MAP_USD_PATH}")

        add_reference_to_stage(self.MAP_USD_PATH, self.MAP_PRIM_PATH)

        # 2) Franka 추가 (맵과는 별도의 prim path)
        self._franka = self._world.scene.add(
            Franka(
                prim_path=self.FRANKA_PRIM_PATH,
                name="fancy_franka",
                position=self.FRANKA_POSITION,
            )
        )

    def setup_targets(self):
        arm_target_deg_1 = np.array([0.0, -20.0, 0.0, -120.0, 0.0, 100.0, 45.0])
        gripper_target_pos_1 = np.array([0.04, 0.04])
        self._target_joint_positions_1 = np.concatenate([np.deg2rad(arm_target_deg_1), gripper_target_pos_1])

        arm_target_deg_2 = np.array([0.0, 20.0, 0.0, 20.0, 0.0, 0.0, 45.0])
        gripper_target_pos_2 = np.array([0.00, 0.00])
        self._target_joint_positions_2 = np.concatenate([np.deg2rad(arm_target_deg_2), gripper_target_pos_2])

    def move_joint_interpolated(self, target_joint_positions: np.ndarray) -> bool:
        current_joint_positions = self._franka.get_joint_positions()

        if current_joint_positions.shape[0] != target_joint_positions.shape[0]:
            raise RuntimeError(
                f"DoF mismatch: current={current_joint_positions.shape[0]}, "
                f"target={target_joint_positions.shape[0]}. "
                f"Franka가 arm만 로드됐는지(7) / arm+gripper(9)인지 확인 필요."
            )

        error = target_joint_positions - current_joint_positions
        error_norm = np.linalg.norm(error)

        if error_norm > self._threshold:
            next_joint_positions = current_joint_positions + (error * self._interp_speed)
            action = ArticulationAction(joint_positions=next_joint_positions)
            is_reached = False
        else:
            action = ArticulationAction(joint_positions=target_joint_positions)
            is_reached = True

        self._franka.apply_action(action)
        return is_reached

    def physics_step(self, step_size: float):
        if self.task_phase == 1:
            if self.move_joint_interpolated(self._target_joint_positions_1):
                print("Task Phase 1 (open) 완료.")
                self.task_phase = 2
        elif self.task_phase == 2:
            if self.move_joint_interpolated(self._target_joint_positions_2):
                print("Task Phase 2 (close) 완료.")
                self.task_phase = 3

    def run(self):
        self.setup_scene()
        self.setup_targets()

        self._world.reset()

        for _ in range(5):
            self._world.step(render=True)

        # 그리퍼 초기 open
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

        self._world.play()

        while simulation_app.is_running():
            try:
                step_size = self._world.get_physics_dt()
            except Exception:
                step_size = 1.0 / 60.0

            self.physics_step(step_size)
            self._world.step(render=True)

        simulation_app.close()


if __name__ == "__main__":
    MoveJointInterpolatedFrankaStandalone().run()
