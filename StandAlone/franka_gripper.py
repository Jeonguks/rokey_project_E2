import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import Franka


# ------------------------------------------------------------
# RMPFlow Controller
# ------------------------------------------------------------
class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt):
        cfg = mg.interface_config_loader.load_supported_motion_policy_config(
            "Franka", "RMPflow"
        )
        rmp = mg.lula.motion_policies.RmpFlow(**cfg)
        policy = mg.ArticulationMotionPolicy(robot_articulation, rmp, physics_dt)
        super().__init__(name=name, articulation_motion_policy=policy)

        pos, quat = robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(pos, quat)

    def reset(self):
        super().reset()
        pos, quat = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(pos, quat)


# ------------------------------------------------------------
# Main Application
# ------------------------------------------------------------
class MoveFrankaStandalone:
    def __init__(self):
        # ===== 사용자 환경에 맞게 수정 =====
        self.MAP_USD_PATH = "/home/rokey/Documents/project/test_world_origin.usd"
        self.MAP_PRIM_PATH = "/World/Map"
        self.FRANKA_PRIM_PATH = "/World/Fancy_Franka"
        self.FRANKA_POSITION = np.array([0.0, 1.0, 0.0])
        # ==================================

        # 시나리오 목표 좌표
        self.GOAL_1 = np.array([0.5, 0.5, 0.5])
        self.GOAL_2 = np.array([0.0, -0.3, 0.1])

        self._world = None
        self._franka = None
        self.controller = None

        # phase
        # 0: GOAL_1 이동
        # 1: gripper close
        # 2: GOAL_2 이동
        # 3: gripper open
        # 4: done
        self.phase = 0
        self.wait_steps = 0

        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])

    # --------------------------------------------------------
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

    # --------------------------------------------------------
    def setup_post_load(self):
        self._world.reset()

        self.controller = RMPFlowController(
            "rmp_controller",
            self._franka,
            self._world.get_physics_dt(),
        )

        # 시작 시 그리퍼 OPEN
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions
        )

        self._world.add_physics_callback("sim_step", self.physics_step)
        self._world.play()

    # --------------------------------------------------------
    def move_point(self, pos):
        action = self.controller.forward(pos, self.ee_quat)
        self._franka.apply_action(action)

        cur = self._franka.get_joint_positions()
        tgt = action.joint_positions

        return np.all(np.abs(cur[:7] - tgt[:7]) < 0.01)

    # --------------------------------------------------------
    def physics_step(self, step_size):
        if self.phase == 0:
            # 1. GOAL 1 이동
            if self.move_point(self.GOAL_1):
                self.controller.reset()
                self.phase = 1
                self.wait_steps = 0

        elif self.phase == 1:
            # 2. Gripper CLOSE
            if self.wait_steps == 0:
                self._franka.gripper.close()
            self.wait_steps += 1
            if self.wait_steps > 30:  # 약 0.5초
                self.phase = 2
                self.wait_steps = 0

        elif self.phase == 2:
            # 3. GOAL 2 이동
            if self.move_point(self.GOAL_2):
                self.controller.reset()
                self.phase = 3
                self.wait_steps = 0

        elif self.phase == 3:
            # 4. Gripper OPEN
            if self.wait_steps == 0:
                self._franka.gripper.open()
            self.wait_steps += 1
            if self.wait_steps > 30:
                self.phase = 4
                print("[DONE] Scenario finished")

        elif self.phase == 4:
            # 종료 상태
            pass


# ------------------------------------------------------------
def main():
    app = MoveFrankaStandalone()
    app.setup_scene()
    app.setup_post_load()

    while simulation_app.is_running():
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
