import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.core.api.objects import DynamicCuboid

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics


# ------------------------------------------------------------
# RMPFlow Controller
# ------------------------------------------------------------
class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt):
        cfg = mg.interface_config_loader.load_supported_motion_policy_config(
            "UR10e", "RMPflow"
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
class MoveArmStandalone:
    def __init__(self):
        # ===== 사용자 환경에 맞게 수정 =====
        self.MAP_USD_PATH = "/home/rokey/Desktop/edit_map.usd"
        self.MAP_PRIM_PATH = "/World/Map"
        
        self.ARM_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/asset/project_arm.usd"
        self.ARM_PRIM_PATH = "/World/Arm"
        self.ARM_POSITION = np.array([0.0, 1.0, 0.9])






        # ==================================

        # (Pick&Place) 물체/목표
        self.CUBE_PRIM_PATH = "/World/pick_cube"
        self.CUBE_SIZE = 0.04
        self.CUBE_START_POS = np.array([0.50, 0.50, 0.40])   # 큐브 시작 위치

        self.PLACE_POS = np.array([0.00, -0.30, 0.40])       # 놓을 위치(바닥과 충돌 안 나게 z는 적당히)
        self.APPROACH_Z = 0.12   # approach offset (접근 높이)
        self.LIFT_Z = 0.18       # lift offset (들어올림 높이)



        # 고정 조인트 prim 경로
        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.FIXED_JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/pick_fixed_joint"

        self._world = None
        self._arm = None
        self._cube = None
        self.controller = None

        # phase
        # 0: approach above cube
        # 1: descend to cube
        # 2: close gripper
        # 3: attach (fixed joint)
        # 4: lift
        # 5: move above place
        # 6: descend to place
        # 7: open gripper
        # 8: detach (remove fixed joint)
        # 9: done
        self.phase = 0
        self.wait_steps = 0

        # EE 목표 자세(필요시 조정)
        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])

        self.EE_LINK_PATH = "/World/Arm/ee_link"


    # ---------------------------
    # USD Utils
    # ---------------------------
    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _get_world_transform(self, prim_path: str) -> Gf.Matrix4d:
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        xform_cache = UsdGeom.XformCache()
        return xform_cache.GetLocalToWorldTransform(prim)

    def _mat4_to_pos_quat(self, m: Gf.Matrix4d):
        # position
        t = m.ExtractTranslation()
        # rotation
        r = m.ExtractRotation()  # Gf.Rotation
        qd = r.GetQuat()         # Gf.Quatd
        # FixedJoint는 보통 float(quatf/vec3f)도 허용되나, 확실히 변환
        pos = Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))
        quat = Gf.Quatf(float(qd.GetReal()), Gf.Vec3f(float(qd.GetImaginary()[0]),
                                                     float(qd.GetImaginary()[1]),
                                                     float(qd.GetImaginary()[2])))
        return pos, quat


    # ---------------------------
    # Attach / Detach (Fixed Joint)
    # ---------------------------
    def attach_fixed_joint(self, cube_path: str, ee_link_path: str):
        """
        Attach (고정 조인트 생성):
        - FixedJoint joint frame을 '현재 큐브 월드 포즈'로 두고,
          cube의 local frame은 identity,
          ee_link의 local frame은 inv(ee_world) * cube_world 로 설정.
        """
        stage = self._get_stage()

        # Constraints root 보장
        if not stage.GetPrimAtPath(self.CONSTRAINTS_ROOT).IsValid():
            UsdGeom.Xform.Define(stage, self.CONSTRAINTS_ROOT)

        # 기존 joint 있으면 제거(중복 실행 안전)
        old = stage.GetPrimAtPath(self.FIXED_JOINT_PATH)
        if old and old.IsValid():
            stage.RemovePrim(self.FIXED_JOINT_PATH)

        # FixedJoint 생성
        joint = UsdPhysics.FixedJoint.Define(stage, self.FIXED_JOINT_PATH)

        # Body relationships 설정
        joint.CreateBody0Rel().SetTargets([cube_path])       # body0: cube
        joint.CreateBody1Rel().SetTargets([ee_link_path])    # body1: ee link

        # 월드 변환
        T_cube = self._get_world_transform(cube_path)
        T_ee = self._get_world_transform(ee_link_path)
        # joint frame = cube frame
        T_joint = T_cube

        # local0 = inv(T_cube) * T_joint = Identity
        # local1 = inv(T_ee) * T_joint
        T_local1 = T_ee.GetInverse() * T_joint

        # local pos/rot 설정
        # body0 (cube) 기준: identity
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        # body1 (ee) 기준: T_local1에서 pos/quat 추출
        pos1, quat1 = self._mat4_to_pos_quat(T_local1)
        joint.CreateLocalPos1Attr().Set(pos1)
        joint.CreateLocalRot1Attr().Set(quat1)

        print(f"[ATTACH] FixedJoint created: {self.FIXED_JOINT_PATH}")
        return True

    def detach_fixed_joint(self):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self.FIXED_JOINT_PATH)
        if prim and prim.IsValid():
            stage.RemovePrim(self.FIXED_JOINT_PATH)
            print(f"[DETACH] FixedJoint removed: {self.FIXED_JOINT_PATH}")

    # ---------------------------
    # Motion helper
    # ---------------------------
    def move_ee_to(self, pos: np.ndarray, thresh=0.01):
        action = self.controller.forward(
            target_end_effector_position=pos,
            target_end_effector_orientation=self.ee_quat
        )
        self._arm.apply_action(action)

        cur = self._arm.get_joint_positions()
        tgt = action.joint_positions

        return np.all(np.abs(cur[:6] - tgt[:6]) < thresh)
    # --------------------------------------------------------
    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)

        if not os.path.isfile(self.MAP_USD_PATH):
            raise FileNotFoundError(self.MAP_USD_PATH)

        add_reference_to_stage(self.MAP_USD_PATH, self.MAP_PRIM_PATH)

        if not os.path.isfile(self.ARM_USD_PATH):
            raise FileNotFoundError(self.ARM_USD_PATH)
        
        add_reference_to_stage(usd_path=self.ARM_USD_PATH, prim_path=self.ARM_PRIM_PATH)

        self._arm = self._world.scene.add(
            SingleArticulation(
                prim_path=self.ARM_PRIM_PATH,
                name="arm",
            )
        )

        self._arm.set_world_pose(position=self.ARM_POSITION)


        # Pick 대상 큐브 추가
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.CUBE_PRIM_PATH,
                name="pick_cube",
                position=self.CUBE_START_POS,
                scale=np.array([self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE]),
            )
        )

    # --------------------------------------------------------
    def setup_post_load(self):
        self._world.reset()
        dof_names = self._arm.dof_names

        # EE 링크 찾기
        self.ee_link_path = self.EE_LINK_PATH
        print(f"[INFO] EE link path = {self.ee_link_path}")

        self.controller = RMPFlowController(
            "rmp_controller",
            self._arm,
            self._world.get_physics_dt(),
        )

        # 초기자세 설정
        q_home_deg_by_name = {
            "shoulder_pan_joint":   0.0,
            "shoulder_lift_joint": -90.0,
            "elbow_joint":          90.0,
            "wrist_1_joint":       -90.0,
            "wrist_2_joint":       -90.0,
            "wrist_3_joint":        0.0,
        }
        q_current = self._arm.get_joint_positions()
        q_home_rad = np.array(q_current, dtype=np.float32)  # 기본: 현재값 유지

        # deg2rad
        for i, name in enumerate(dof_names):
            if name in q_home_deg_by_name:
                q_home_rad[i] = np.deg2rad(q_home_deg_by_name[name])

        self._arm.set_joint_positions(q_home_rad)
        self._arm.set_joint_velocities(np.zeros_like(q_home_rad))

        for _ in range(5):
            self._world.step(render=True)

        # 컨트롤러 생성/리셋
        self.controller = RMPFlowController(
            "rmp_controller",
            self._arm,
            self._world.get_physics_dt(),
        )
        self.controller.reset()

        self._world.add_physics_callback("sim_step", self.physics_step)
        self._world.play()

    # --------------------------------------------------------
    def physics_step(self, step_size):
        # 목표 포즈들 계산
        cube_pos = self.CUBE_START_POS.copy()
        pre_grasp = cube_pos.copy()
        pre_grasp[2] += self.APPROACH_Z

        grasp = cube_pos.copy()

        lift = cube_pos.copy()
        lift[2] += self.LIFT_Z

        place_pre = self.PLACE_POS.copy()
        place_pre[2] += self.APPROACH_Z

        place = self.PLACE_POS.copy()

        if self.phase == 0:
            # approach above cube
            if self.move_ee_to(pre_grasp):
                self.controller.reset()
                self.phase = 1

        elif self.phase == 1:
            # descend to cube
            if self.move_ee_to(grasp):
                self.controller.reset()
                self.phase = 2
                self.wait_steps = 0

        elif self.phase == 2:
            # close gripper
            if self.wait_steps == 0:
                pass
            self.wait_steps += 1
            if self.wait_steps > 25:
                self.phase = 3

        elif self.phase == 3:
            # attach (fixed joint)
            self.attach_fixed_joint(self.CUBE_PRIM_PATH, self.ee_link_path)
            self.phase = 4

        elif self.phase == 4:
            # lift
            if self.move_ee_to(lift):
                self.controller.reset()
                self.phase = 5

        elif self.phase == 5:
            # move above place
            if self.move_ee_to(place_pre):
                self.controller.reset()
                self.phase = 6

        elif self.phase == 6:
            # descend to place
            if self.move_ee_to(place):
                self.controller.reset()
                self.phase = 7
                self.wait_steps = 0

        elif self.phase == 7:
            # open gripper
            if self.wait_steps == 0:
                pass
            self.wait_steps += 1
            if self.wait_steps > 20:
                self.phase = 8

        elif self.phase == 8:
            # detach
            self.detach_fixed_joint()
            self.phase = 9
            print("[DONE] Pick & Place finished (FixedJoint attach/detach).")

        elif self.phase == 9:
            pass


# ------------------------------------------------------------
def main():
    app = MoveArmStandalone()
    app.setup_scene()
    app.setup_post_load()

    while simulation_app.is_running():
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
