import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics


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
        self.MAP_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/map/test_world_origin.usd"
        self.MAP_PRIM_PATH = "/World/Map"
        self.FRANKA_PRIM_PATH = "/World/Fancy_Franka"
        self.FRANKA_POSITION = np.array([0.0, 1.0, 0.0])
        # ==================================

        # (Pick&Place) 물체/목표
        self.CUBE_PRIM_PATH = "/World/pick_cube"
        self.CUBE_SIZE = 0.04
        self.CUBE_START_POS = np.array([0.50, 0.50, 0.40])   # 큐브 시작 위치

        self.PLACE_POS = np.array([0.00, -0.30, 0.40])       # 놓을 위치(바닥과 충돌 안 나게 z는 적당히)
        self.APPROACH_Z = 0.12   # approach offset (접근 높이)
        self.LIFT_Z = 0.18       # lift offset (들어올림 높이)

        # EE 링크 후보(환경마다 다릅니다). Stage에서 실제 링크 이름에 맞춰 수정하세요.
        # 예: panda_hand, hand, panda_link8 등
        self.EE_LINK_CANDIDATES = ["panda_hand", "hand", "panda_link8"] # panda_hand

        # 고정 조인트 prim 경로
        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.FIXED_JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/pick_fixed_joint"

        self._world = None
        self._franka = None
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

        # EE 링크 경로(런타임에 탐색해서 채움)
        self.ee_link_path = None

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
    # EE link 찾기
    # ---------------------------
    def find_ee_link_path(self):
        stage = self._get_stage()

        # Franka root 아래에서 후보 이름을 가진 prim을 찾아 첫 번째를 EE로 사용
        for cand in self.EE_LINK_CANDIDATES:
            # 가장 단순하게는 "프림 이름이 cand인 것을 전체 stage에서 찾기"를 합니다.
            # (여러 개 나오면 Franka 아래 첫 번째를 선택)
            for prim in stage.Traverse():
                if prim.GetName() == cand:
                    p = str(prim.GetPath())
                    if p.startswith(self.FRANKA_PRIM_PATH):
                        return p

        # 못 찾으면 fallback: Franka 아래 children 출력해서 사용자에게 힌트
        franka_prim = stage.GetPrimAtPath(self.FRANKA_PRIM_PATH)
        if franka_prim and franka_prim.IsValid():
            print("[EE LINK NOT FOUND] Franka children (top-level):")
            for c in franka_prim.GetChildren():
                print(" -", c.GetPath())

        raise RuntimeError(
            "EE link prim not found. Update EE_LINK_CANDIDATES to match your stage link names."
        )

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
        action = self.controller.forward(pos, self.ee_quat)
        self._franka.apply_action(action)

        cur = self._franka.get_joint_positions()
        tgt = action.joint_positions
        return np.all(np.abs(cur[:7] - tgt[:7]) < thresh)

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

        # EE 링크 찾기
        self.ee_link_path = self.find_ee_link_path()
        print(f"[INFO] EE link path = {self.ee_link_path}")

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
                self._franka.gripper.close()
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
                self._franka.gripper.open()
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
    app = MoveFrankaStandalone()
    app.setup_scene()
    app.setup_post_load()

    while simulation_app.is_running():
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
