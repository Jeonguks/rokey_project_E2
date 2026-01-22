import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import DynamicCuboid

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics


class JointPickPlaceStandalone:
    def __init__(self):
        # ===== 사용자 환경 =====
        self.MAP_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/map/test_world_origin.usd"
        self.MAP_PRIM_PATH = "/World/Map"

        self.ROBOT_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/asset/project_arm.usd"
        self.ROBOT_PRIM_PATH = "/World/Fancy_Franka"
        self.ROBOT_POSITION = np.array([0.0, 1.0, 0.0])

        # EE 링크(이미지에서 확인)
        self.EE_LINK_PATH = "/World/Fancy_Franka/ee_link"

        # 당신 로봇 조인트 이름(이미지에서 확인)
        self.JOINT_NAMES = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # ===== Pick&Place 대상 =====
        self.CUBE_PRIM_PATH = "/World/pick_cube"
        self.CUBE_SIZE = 0.04
        self.CUBE_START_POS = np.array([0.50, 0.50, 0.40])

        self.PLACE_POS = np.array([0.00, -0.30, 0.40])

        # FixedJoint
        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.FIXED_JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/pick_fixed_joint"

        # ===== 튜닝 파라미터 =====
        self.POS_THR = 0.03          # joint-space 도달 판정 허용오차(rad)
        self.HOLD_STEPS = 20         # 각 phase 전환 시 안정화 대기

        # ===== 관절 목표 (핵심: 여기 숫자만 튜닝하면 됨) =====
        # 주의: 단위 rad. UR-like 관절 범위 고려.
        # 아래는 "대충" 시작값. 당신 로봇/좌표계에 맞게 반드시 조정해야 함.
        self.Q_HOME     = np.array([ 0.0, -1.2,  1.6, -1.6, -1.6,  0.0], dtype=np.float32)
        self.Q_PREGRASP = np.array([ 0.4, -1.0,  1.4, -1.6, -1.6,  0.2], dtype=np.float32)
        self.Q_GRASP    = np.array([ 0.4, -0.8,  1.2, -1.6, -1.6,  0.2], dtype=np.float32)
        self.Q_LIFT     = np.array([ 0.4, -1.1,  1.5, -1.6, -1.6,  0.2], dtype=np.float32)
        self.Q_PREPLACE = np.array([-0.3, -1.0,  1.4, -1.6, -1.6, -0.3], dtype=np.float32)
        self.Q_PLACE    = np.array([-0.3, -0.8,  1.2, -1.6, -1.6, -0.3], dtype=np.float32)
        self.Q_RETREAT  = np.array([-0.3, -1.2,  1.6, -1.6, -1.6, -0.3], dtype=np.float32)

        # 시퀀스 정의
        self.SEQ = [
            ("home",     self.Q_HOME),
            ("pregrasp", self.Q_PREGRASP),
            ("grasp",    self.Q_GRASP),
            ("attach",   None),          # FixedJoint attach
            ("lift",     self.Q_LIFT),
            ("preplace", self.Q_PREPLACE),
            ("place",    self.Q_PLACE),
            ("detach",   None),          # FixedJoint detach
            ("retreat",  self.Q_RETREAT),
            ("done",     None),
        ]

        self._world = None
        self.robot = None
        self.cube = None

        self.phase = 0
        self.hold = 0

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
        t = m.ExtractTranslation()
        r = m.ExtractRotation()
        qd = r.GetQuat()
        pos = Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))
        quat = Gf.Quatf(float(qd.GetReal()), Gf.Vec3f(float(qd.GetImaginary()[0]),
                                                     float(qd.GetImaginary()[1]),
                                                     float(qd.GetImaginary()[2])))
        return pos, quat

    # ---------------------------
    # FixedJoint attach/detach
    # ---------------------------
    def attach_fixed_joint(self, cube_path: str, ee_link_path: str):
        stage = self._get_stage()

        if not stage.GetPrimAtPath(self.CONSTRAINTS_ROOT).IsValid():
            UsdGeom.Xform.Define(stage, self.CONSTRAINTS_ROOT)

        old = stage.GetPrimAtPath(self.FIXED_JOINT_PATH)
        if old and old.IsValid():
            stage.RemovePrim(self.FIXED_JOINT_PATH)

        joint = UsdPhysics.FixedJoint.Define(stage, self.FIXED_JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([cube_path])
        joint.CreateBody1Rel().SetTargets([ee_link_path])

        T_cube = self._get_world_transform(cube_path)
        T_ee = self._get_world_transform(ee_link_path)

        T_joint = T_cube
        T_local1 = T_ee.GetInverse() * T_joint

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        pos1, quat1 = self._mat4_to_pos_quat(T_local1)
        joint.CreateLocalPos1Attr().Set(pos1)
        joint.CreateLocalRot1Attr().Set(quat1)

        print(f"[ATTACH] {self.FIXED_JOINT_PATH}")

    def detach_fixed_joint(self):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self.FIXED_JOINT_PATH)
        if prim and prim.IsValid():
            stage.RemovePrim(self.FIXED_JOINT_PATH)
            print(f"[DETACH] {self.FIXED_JOINT_PATH}")

    # ---------------------------
    # Joint motion helpers
    # ---------------------------
    def _apply_joint_target(self, q_target: np.ndarray):
        # joint_names 명시해서 “이 6개만” 제어
        action = ArticulationAction(joint_positions=q_target, joint_names=self.JOINT_NAMES)
        self.robot.apply_action(action)

    def _reached(self, q_target: np.ndarray) -> bool:
        # 현재 전체 dof 중 우리가 지정한 조인트들만 비교해야 안전.
        # SingleArticulation.get_joint_positions(joint_names=...)가 되는 빌드도 있고,
        # 안 되는 빌드도 있어 보수적으로 dof_name 매핑을 직접 처리.
        cur_all = self.robot.get_joint_positions()
        dof_names = list(self.robot.dof_names)

        idx = []
        for jn in self.JOINT_NAMES:
            if jn not in dof_names:
                raise RuntimeError(f"Joint name not in dof_names: {jn}")
            idx.append(dof_names.index(jn))

        cur = cur_all[idx]
        return np.all(np.abs(cur - q_target) < self.POS_THR)

    # ---------------------------
    # Setup
    # ---------------------------
    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)

        if not os.path.isfile(self.MAP_USD_PATH):
            raise FileNotFoundError(self.MAP_USD_PATH)
        add_reference_to_stage(self.MAP_USD_PATH, self.MAP_PRIM_PATH)

        if not os.path.isfile(self.ROBOT_USD_PATH):
            raise FileNotFoundError(self.ROBOT_USD_PATH)
        add_reference_to_stage(self.ROBOT_USD_PATH, self.ROBOT_PRIM_PATH)

        self._world.reset()

        self.robot = self._world.scene.add(
            SingleArticulation(prim_path=self.ROBOT_PRIM_PATH, name="arm")
        )
        self.robot.set_world_pose(position=self.ROBOT_POSITION)

        print("[DOF NAMES]")
        for i, n in enumerate(self.robot.dof_names):
            print(i, n)

        self.cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.CUBE_PRIM_PATH,
                name="pick_cube",
                position=self.CUBE_START_POS,
                scale=np.array([self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE]),
            )
        )

        self._world.add_physics_callback("sim_step", self.physics_step)
        self._world.play()

    # ---------------------------
    # Main loop callback
    # ---------------------------
    def physics_step(self, dt):
        name, q = self.SEQ[self.phase]

        # phase 안정화 대기
        if self.hold > 0:
            self.hold -= 1
            return

        if name in ["home", "pregrasp", "grasp", "lift", "preplace", "place", "retreat"]:
            self._apply_joint_target(q)
            if self._reached(q):
                print(f"[PHASE] reached: {name}")
                self.phase += 1
                self.hold = self.HOLD_STEPS
            return

        if name == "attach":
            # 파지: FixedJoint
            self.attach_fixed_joint(self.CUBE_PRIM_PATH, self.EE_LINK_PATH)
            self.phase += 1
            self.hold = self.HOLD_STEPS
            return

        if name == "detach":
            self.detach_fixed_joint()
            self.phase += 1
            self.hold = self.HOLD_STEPS
            return

        if name == "done":
            # 완료 상태
            return


def main():
    app = JointPickPlaceStandalone()
    app.setup_scene()

    while simulation_app.is_running():
        app._world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
