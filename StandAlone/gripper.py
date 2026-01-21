import os
import numpy as np

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.examples.interactive.base_sample import BaseSample


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: SingleArticulation, physics_dt: float = 1.0 / 60.0):
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        self._default_position, self._default_orientation = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(self._default_position, self._default_orientation)

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(self._default_position, self._default_orientation)


class RidgebackFrankaPick(BaseSample):
    def __init__(self):
        super().__init__()
        self.robot_prim_path = "/World/ridgeback_franka"
        self.task_phase = 0

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # USD 경로: 환경변수 ISAAC_USD 있으면 그걸 사용, 없으면 fallback
        USD_PATH = os.environ.get("ISAAC_USD", "/home/rokey/Documents/project/canifixit.usd")

        # Stage에 로봇 USD 붙이기
        add_reference_to_stage(USD_PATH, self.robot_prim_path)

        # Articulation으로 등록 (핸들링 가능하게)
        world.scene.add(SingleArticulation(prim_path=self.robot_prim_path, name="ridgeback_franka"))

        # 테스트용 큐브
        world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.6, 0.0, 0.0]),
                scale=np.array([0.05, 0.05, 0.05]),
            )
        )

    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("ridgeback_franka")
        self._cube = self._world.scene.get_object("fancy_cube")

        # 컨트롤러
        self.cspace_controller = RMPFlowController(
            name="rmp_cspace_controller",
            robot_articulation=self._robot,
            physics_dt=self._world.get_physics_dt(),
        )

        # 목표 EE 자세(예시)
        self.ee_quat = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))

        # 그리퍼 초기 open (joint 기반)
        self.set_gripper(close=False)

        # 스텝 콜백
        self._world.add_physics_callback("sim_step", self.physics_step)

        self.task_phase = 1
        self._goal_reached = False

        await self._world.play_async()

    def set_gripper(self, close: bool):
        """
        ridgeback_franka가 Franka wrapper가 아니라면 gripper helper가 없을 수 있어
        finger joint를 직접 제어하는 방식이 가장 안전합니다.
        """
        dof_names = self._robot.dof_names

        # 필요하면 여기 joint 이름을 stage 기준으로 수정
        j1 = dof_names.index("panda_finger_joint1")
        j2 = dof_names.index("panda_finger_joint2")

        q = self._robot.get_joint_positions()
        if close:
            q[j1], q[j2] = 0.0, 0.0
        else:
            q[j1], q[j2] = 0.04, 0.04

        self._robot.set_joint_positions(q)

    def move_point(self, goal_position: np.ndarray) -> bool:
        # 모바일 베이스가 움직인다면 base pose를 매 스텝 갱신하는 게 안전합니다.
        base_pos, base_quat = self._robot.get_world_pose()
        self.cspace_controller._motion_policy.set_robot_base_pose(base_pos, base_quat)

        action = self.cspace_controller.forward(
            target_end_effector_position=goal_position,
            target_end_effector_orientation=self.ee_quat,
        )

        # SingleArticulation은 ArticulationAction 적용을 지원합니다.
        self._robot.apply_action(action)

        # 수렴 판단(너무 타이트하면 영원히 false 될 수 있으니 여유 권장)
        cur = self._robot.get_joint_positions()
        tgt = action.joint_positions

        # tgt가 전체 dof 길이일 수도, arm dof만일 수도 있어 방어적으로 비교
        n = min(len(cur), len(tgt))
        return np.all(np.abs(cur[:n] - tgt[:n]) < 0.01)

    def physics_step(self, step_size):
        if self.task_phase == 1:
            cube_pos, _ = self._cube.get_world_pose()
            cube_pos[2] = 0.10  # 물체 위로 접근 높이(필요 조정)
            if self.move_point(cube_pos):
                self.cspace_controller.reset()
                self.task_phase = 2

        elif self.task_phase == 2:
            self.set_gripper(close=True)
            self.task_phase = 3

        elif self.task_phase == 3:
            cube_pos, _ = self._cube.get_world_pose()
            cube_pos[2] = 0.50  # 들어올리기 높이
            if self.move_point(cube_pos):
                self.cspace_controller.reset()
                self.task_phase = 4

        elif self.task_phase == 4:
            self.set_gripper(close=False)
            self.task_phase = 5

        return
