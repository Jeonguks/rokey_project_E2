import numpy as np

from isaacsim import SimulationApp

# GUI 켜기/끄기
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import Franka


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: SingleArticulation, physics_dt: float = 1.0 / 60.0) -> None:
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        self._default_position, self._default_orientation = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


class MoveFrankaStandalone:
    def __init__(self):
        self._world = None
        self._franka = None
        self.cspace_controller = None

        self._goal_points = [
            np.array([0.5, 0.5, 0.5]),
            np.array([0.0, -0.3, 0.1]),
        ]
        self.task_phase = 0
        self._goal_reached = False

    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        self._franka = self._world.scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="fancy_franka",
            )
        )

    def setup_post_load(self):
        # World reset 후 컨트롤러/콜백 세팅하는 게 안정적입니다.
        self._world.reset()

        self.cspace_controller = RMPFlowController(
            name="pick_place_controller_cspace_controller",
            robot_articulation=self._franka,
        )

        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

        # 물리 스텝마다 콜백
        self._world.add_physics_callback("sim_step", self.physics_step)

        self._world.play()

    def move_point(self, goal_position: np.ndarray, end_effector_orientation: np.ndarray = np.array([0, np.pi, 0])) -> bool:
        end_effector_orientation_q = euler_angles_to_quat(end_effector_orientation)

        target_action = self.cspace_controller.forward(
            target_end_effector_position=goal_position,
            target_end_effector_orientation=end_effector_orientation_q,
        )

        # Franka.apply_action 은 ArticulationAction을 받는 형태가 일반적입니다.
        self._franka.apply_action(target_action)

        current_joint_positions = self._franka.get_joint_positions()
        is_reached = np.all(np.abs(current_joint_positions[:7] - target_action.joint_positions) < 0.001)
        return bool(is_reached)

    def physics_step(self, step_size: float):
        if self.task_phase == 0:
            current_goal = self._goal_points[self.task_phase]
            self._goal_reached = self.move_point(current_goal)
            if self._goal_reached:
                self.cspace_controller.reset()
                self.task_phase = 1

        elif self.task_phase == 1:
            current_goal = self._goal_points[self.task_phase]
            self._goal_reached = self.move_point(current_goal, end_effector_orientation=np.array([np.pi / 2, np.pi / 2, 0]))
            if self._goal_reached:
                self.cspace_controller.reset()
                self.task_phase = 2


def main():
    app = MoveFrankaStandalone()
    app.setup_scene()
    app.setup_post_load()

    # Standalone의 “메인 루프”
    while simulation_app.is_running():
        # render=True면 화면 갱신, headless면 render=False 권장
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
