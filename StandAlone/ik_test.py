import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka import Franka
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

import isaacsim.robot_motion.motion_generation as mg


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


# ----------------------------
# Gripper helper (joint direct)
# ----------------------------
def set_gripper(robot: SingleArticulation, close: bool, finger_joints=("panda_finger_joint1", "panda_finger_joint2")):
    dof_names = robot.dof_names
    j1 = dof_names.index(finger_joints[0])
    j2 = dof_names.index(finger_joints[1])

    q = robot.get_joint_positions()
    if close:
        q[j1], q[j2] = 0.0, 0.0
    else:
        q[j1], q[j2] = 0.04, 0.04

    robot.set_joint_positions(q)

# ----------------------------
# Simple phase machine
# ----------------------------
class PickPlaceDemo:
    def __init__(self, world: World, robot: SingleArticulation, cube: DynamicCuboid):
        self.world = world
        self.robot = robot
        self.cube = cube

        self.controller = RMPFlowController("rmp_controller", robot, world.get_physics_dt())
        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])

        self.phase = 1
        self.phase_t0 = time.time()

        # 조인트 이름이 다르면 여기만 수정
        self.FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")

        set_gripper(self.robot, close=False, finger_joints=self.FINGER_JOINTS)

    def _apply_rmp_to(self, pos: np.ndarray):
        self.controller.update_base_pose()
        action = self.controller.forward(pos, self.ee_quat)
        self.robot.apply_action(action)

        cur = self.robot.get_joint_positions()
        tgt = action.joint_positions
        n = min(len(cur), len(tgt))
        # 수렴 기준은 너무 빡빡하면 영원히 false 될 수 있어 0.01 권장
        return np.all(np.abs(cur[:n] - tgt[:n]) < 0.01)

    def step(self):
        # phase timeout 방지 (디버깅용)
        if time.time() - self.phase_t0 > 8.0 and self.phase in (1, 3):
            print(f"[WARN] Phase {self.phase} taking too long. Check EE frame / RMP config / collisions.")
            self.phase_t0 = time.time()

        if self.phase == 1:
            # approach above cube
            p, _ = self.cube.get_world_pose()
            p = np.array(p, dtype=np.float64)
            p[2] = 0.10
            if self._apply_rmp_to(p):
                self.controller.reset()
                self.phase = 2
                self.phase_t0 = time.time()

        elif self.phase == 2:
            # close gripper
            set_gripper(self.robot, close=True, finger_joints=self.FINGER_JOINTS)
            self.phase = 3
            self.phase_t0 = time.time()

        elif self.phase == 3:
            # lift
            p, _ = self.cube.get_world_pose()
            p = np.array(p, dtype=np.float64)
            p[2] = 0.50
            if self._apply_rmp_to(p):
                self.controller.reset()
                self.phase = 4
                self.phase_t0 = time.time()

        elif self.phase == 4:
            # open gripper
            set_gripper(self.robot, close=False, finger_joints=self.FINGER_JOINTS)
            self.phase = 5
            print("[DONE] Pick/Place sequence finished.")

        # phase 5: idle


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # USD 경로: ISAAC_USD가 있으면 그걸 사용
    USD_PATH = os.environ.get("ISAAC_USD", "/home/rokey/Documents/project/canifixit.usd")
    ROBOT_PRIM_PATH = "/World/ridgeback_franka"

    add_reference_to_stage(USD_PATH, ROBOT_PRIM_PATH)

    robot = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="ridgeback_franka")
    world.scene.add(robot)

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0.6, 0.0, 0.0]),
            scale=np.array([0.05, 0.05, 0.05]),
        )
    )

    world.reset()

    # 디버깅용: DOF 이름 확인 (그리퍼/팔 조인트가 다른 경우 여기서 확인)
    print("Robot DOFs:")
    for i, n in enumerate(robot.dof_names):
        print(f"  [{i}] {n}")

    demo = PickPlaceDemo(world, robot, cube)

    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            demo.step()

    simulation_app.close()


if __name__ == "__main__":
    main()