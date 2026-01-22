import os
import numpy as np

# ---------------------------------------------------------
# 1) Isaac Sim App (가장 먼저)
# ---------------------------------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import DynamicCuboid, VisualSphere

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, Usd, Sdf


# ==============================================================================
# ✅ 너가 만질 건 여기 3개만
# ==============================================================================
MAP_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/map/project_map.usd"
ARM_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/asset/project_arm.usd"

ROBOT_BASE_POS = np.array([-7.55068, -4.08896, 0.0], dtype=np.float32)

CAN_POS   = np.array([-7.82042, -3.38997, 0.05], dtype=np.float32)
PLACE_POS = np.array([-7.82042, -3.70000, 0.05], dtype=np.float32)

# (1) 캔이랑 로봇이 "살짝 옆으로" 집으면 여기만 수정
#     예) 오른쪽으로 집음 -> Y를 +로 올렸다면 0.03 -> 0.00 또는 -0.02
GRASP_XY_OFFSET = np.array([0.00, 0.00, 0.00], dtype=np.float32)

# (2) 손목(wrist_3_link) 기준 "너무 깊게 내려가서 튕김" 방지용
#     깊게 내려감 -> 줄여 (0.16 -> 0.14 -> 0.12)
#     너무 위에서 붙음 -> 늘려 (0.16 -> 0.18 -> 0.20)
GRIPPER_FINGER_LEN = 0.16

# (3) 현재는 캔이 아니라 큐보이드(직육면체)임. 크기 여기서 바꿈
CAN_SCALE = np.array([0.06, 0.06, 0.12], dtype=np.float32)
# ==============================================================================


# -------------------------
# USD 타입 충돌 방지 헬퍼
# -------------------------
def f(x) -> float:
    return float(np.asarray(x, dtype=np.float64))

def v3d(arr) -> Gf.Vec3d:
    return Gf.Vec3d(f(arr[0]), f(arr[1]), f(arr[2]))

def v3f(arr) -> Gf.Vec3f:
    return Gf.Vec3f(f(arr[0]), f(arr[1]), f(arr[2]))

def quatf(real, imag_vec) -> Gf.Quatf:
    return Gf.Quatf(f(real), v3f(imag_vec))


class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt):
        cfg = mg.interface_config_loader.load_supported_motion_policy_config("UR10e", "RMPflow")
        rmp = mg.lula.motion_policies.RmpFlow(**cfg)
        policy = mg.ArticulationMotionPolicy(robot_articulation, rmp, physics_dt)
        super().__init__(name=name, articulation_motion_policy=policy)

        pos, quat = robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(pos, quat)

    def reset(self):
        super().reset()
        pos, quat = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(pos, quat)


class MoveArmFixed:
    def __init__(self):
        self._world = None
        self._arm = None
        self._can = None
        self.controller = None

        self.MAP_PRIM_PATH = "/World/Map"
        self.ARM_PRIM_PATH = "/World/Arm"
        self.CAN_PRIM_PATH = "/World/DynamicCan"

        self.EE_LINK_PATH = "/World/Arm/ee_link"
        self.ATTACH_LINK  = "/World/Arm/wrist_3_link"

        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/PickJoint"

        self.phase = 0
        self.wait_steps = 0
        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])

    # ---------------------------
    # Utils
    # ---------------------------
    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _step_n(self, n=5):
        # ⚠️ physics_step(콜백) 안에서 호출 금지
        for _ in range(n):
            self._world.step(render=True)

    def _prim_valid(self, prim_path: str) -> bool:
        prim = self._get_stage().GetPrimAtPath(prim_path)
        return bool(prim and prim.IsValid())

    def _assert_prim(self, prim_path: str):
        prim = self._get_stage().GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"[Prim Not Found] {prim_path}")
        return prim

    def _wait_for_prim(self, prim_path: str, steps: int = 180):
        for _ in range(steps):
            prim = self._get_stage().GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return prim
            self._world.step(render=True)
        raise RuntimeError(f"[Prim Timeout] {prim_path}")

    def _resolve_prim_by_name(self, prim_name: str):
        stage = self._get_stage()
        for prim in stage.Traverse():
            try:
                if prim and prim.IsValid() and prim.GetName() == prim_name:
                    return prim.GetPath().pathString
            except Exception:
                pass
        return None

    def _resolve_rigidbody_by_name(self, prim_name: str):
        stage = self._get_stage()
        fallback = None
        for prim in stage.Traverse():
            if not prim or not prim.IsValid():
                continue
            try:
                if prim.GetName() != prim_name:
                    continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return prim.GetPath().pathString
                for child in Usd.PrimRange(prim):
                    if child and child.IsValid() and child.HasAPI(UsdPhysics.RigidBodyAPI):
                        return child.GetPath().pathString
                fallback = prim.GetPath().pathString
            except Exception:
                pass
        return fallback

    def _ensure_link_paths(self):
        stage = self._get_stage()

        def has_rb(path: str) -> bool:
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI))

        if (not self._prim_valid(self.ATTACH_LINK)) or (not has_rb(self.ATTACH_LINK)):
            found = self._resolve_rigidbody_by_name("wrist_3_link") or self._resolve_prim_by_name("wrist_3_link")
            if found:
                print(f"🔧 ATTACH_LINK auto-fix: {self.ATTACH_LINK} -> {found}")
                self.ATTACH_LINK = found
            else:
                raise RuntimeError("🔥 wrist_3_link not found")

        if not self._prim_valid(self.EE_LINK_PATH):
            found = self._resolve_prim_by_name("ee_link")
            if found:
                print(f"🔧 EE_LINK auto-fix: {self.EE_LINK_PATH} -> {found}")
                self.EE_LINK_PATH = found

        self._assert_prim(self.ATTACH_LINK)

    def _get_world_pos(self, prim_path: str) -> np.ndarray:
        prim = self._assert_prim(prim_path)
        xf = UsdGeom.XformCache()
        t = xf.GetLocalToWorldTransform(prim).ExtractTranslation()
        return np.array([f(t[0]), f(t[1]), f(t[2])], dtype=np.float32)

    def _set_collision_enabled_recursive(self, root_path: str, enabled: bool):
        stage = self._get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            return

        for prim in Usd.PrimRange(root):
            if not prim or not prim.IsValid():
                continue

            try:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
            except Exception:
                pass

            try:
                attr = prim.GetAttribute("physics:collisionEnabled")
                if not attr or not attr.IsValid():
                    attr = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
                attr.Set(bool(enabled))
            except Exception:
                pass

    # ---------------------------
    # Setup
    # ---------------------------
    def setup_scene(self):
        self._world = World(stage_units_in_meters=1.0)

        if os.path.isfile(MAP_USD_PATH):
            add_reference_to_stage(MAP_USD_PATH, self.MAP_PRIM_PATH)
        else:
            print(f"⚠️ Map not found: {MAP_USD_PATH}")

        if os.path.isfile(ARM_USD_PATH):
            add_reference_to_stage(ARM_USD_PATH, self.ARM_PRIM_PATH)
        else:
            raise FileNotFoundError(f"Robot USD not found: {ARM_USD_PATH}")

        arm_prim = self._wait_for_prim(self.ARM_PRIM_PATH, steps=200)
        xform = UsdGeom.Xformable(arm_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(v3d(ROBOT_BASE_POS))

        self._step_n(10)

        # 물체(현재 큐보이드) 생성
        self._can = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.CAN_PRIM_PATH,
                name="dynamic_can",
                position=CAN_POS,
                scale=CAN_SCALE,
                color=np.array([1.0, 0.5, 0.0], dtype=np.float32),
                mass=0.1
            )
        )

        # 로봇 등록
        self._arm = self._world.scene.add(
            SingleArticulation(prim_path=self.ARM_PRIM_PATH, name="arm")
        )

        # ✅ 디버그: PLACE_POS 빨간 공
        self._world.scene.add(
            VisualSphere(
                prim_path="/World/DebugPlace",
                name="debug_place",
                position=PLACE_POS,
                scale=np.array([0.03, 0.03, 0.03], dtype=np.float32),
                color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        )

        # ✅ 디버그: 캔 윗면 파란 공
        can_top = np.array([CAN_POS[0], CAN_POS[1], CAN_POS[2] + CAN_SCALE[2] * 0.5], dtype=np.float32)
        self._world.scene.add(
            VisualSphere(
                prim_path="/World/DebugCanTop",
                name="debug_can_top",
                position=can_top,
                scale=np.array([0.02, 0.02, 0.02], dtype=np.float32),
                color=np.array([0.0, 0.4, 1.0], dtype=np.float32),
            )
        )

        print("🔴 빨간 공 = PLACE_POS (목표 위치)")
        print("🔵 파란 공 = 캔 윗면(참고)")
        print(f"🧭 GRASP_XY_OFFSET = {GRASP_XY_OFFSET}")

    def setup_post_load(self):
        self._world.reset()
        self._ensure_link_paths()

        self.controller = RMPFlowController("rmp_controller", self._arm, self._world.get_physics_dt())
        self.controller.reset()

        self._world.play()

        # 워밍업 후 동기화
        self._step_n(25)
        self.controller.reset()

        self._world.add_physics_callback("sim_step", self.physics_step)

    # ---------------------------
    # Motion
    # ---------------------------
    def move_ee_to(self, target_pos, thresh=0.03):
        action = self.controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=self.ee_quat
        )
        self._arm.apply_action(action)

        curr_pos = self._get_world_pos(self.ATTACH_LINK)
        return f(np.linalg.norm(curr_pos - target_pos)) < f(thresh)

    # ---------------------------
    # FixedJoint
    # ---------------------------
    def attach_fixed_joint(self):
        stage = self._get_stage()

        old = stage.GetPrimAtPath(self.JOINT_PATH)
        if old and old.IsValid():
            stage.RemovePrim(self.JOINT_PATH)

        # 튐 방지: 속도 0
        self._can.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self._can.set_angular_velocity(np.zeros(3, dtype=np.float32))

        root = stage.GetPrimAtPath(self.CONSTRAINTS_ROOT)
        if not root or not root.IsValid():
            UsdGeom.Xform.Define(stage, self.CONSTRAINTS_ROOT)

        joint = UsdPhysics.FixedJoint.Define(stage, self.JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([self.ATTACH_LINK])
        joint.CreateBody1Rel().SetTargets([self.CAN_PRIM_PATH])

        can_prim = self._assert_prim(self.CAN_PRIM_PATH)
        wrist_prim = self._assert_prim(self.ATTACH_LINK)
        xf = UsdGeom.XformCache()

        T_rel = xf.GetLocalToWorldTransform(wrist_prim).GetInverse() * xf.GetLocalToWorldTransform(can_prim)
        pos = T_rel.ExtractTranslation()
        rot = T_rel.ExtractRotation().GetQuat()
        im = rot.GetImaginary()

        joint.CreateLocalPos0Attr().Set(v3f([pos[0], pos[1], pos[2]]))
        joint.CreateLocalRot0Attr().Set(quatf(rot.GetReal(), [im[0], im[1], im[2]]))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

        # 붙어있는 동안 충돌 끄기
        self._set_collision_enabled_recursive(self.CAN_PRIM_PATH, False)
        print("[ACTION] Attached")

    def detach_fixed_joint(self):
        self._set_collision_enabled_recursive(self.CAN_PRIM_PATH, True)

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self.JOINT_PATH)
        if prim and prim.IsValid():
            stage.RemovePrim(self.JOINT_PATH)

        print("[ACTION] Detached")

    # ---------------------------
    # Callback Loop
    # ---------------------------
    def physics_step(self, step_size):
        # ⚠️ 절대 self._world.step() 호출 금지

        can_pos, _ = self._can.get_world_pose()

        # 캔 윗면 기준으로 "손목 목표 z" 자동 계산
        can_top_z = f(can_pos[2]) + f(CAN_SCALE[2]) * 0.5

        grasp_wrist_z    = can_top_z + f(GRIPPER_FINGER_LEN) + 0.02
        approach_wrist_z = grasp_wrist_z + 0.15
        lift_wrist_z     = grasp_wrist_z + 0.20

        pre_grasp = np.array([f(can_pos[0]), f(can_pos[1]), approach_wrist_z], dtype=np.float32) + GRASP_XY_OFFSET
        grasp     = np.array([f(can_pos[0]), f(can_pos[1]), grasp_wrist_z], dtype=np.float32)     + GRASP_XY_OFFSET
        lift      = np.array([f(can_pos[0]), f(can_pos[1]), lift_wrist_z], dtype=np.float32)      + GRASP_XY_OFFSET

        place_pre = np.array([f(PLACE_POS[0]), f(PLACE_POS[1]), approach_wrist_z], dtype=np.float32)
        place     = np.array([f(PLACE_POS[0]), f(PLACE_POS[1]), grasp_wrist_z + 0.02], dtype=np.float32)

        if self.phase == 0:
            if self.move_ee_to(pre_grasp):
                self.phase = 1
                self.wait_steps = 0
                print("Phase 0 -> 1: Approach")

        elif self.phase == 1:
            if self.move_ee_to(grasp, thresh=0.025):
                self.phase = 2
                self.wait_steps = 0
                print("Phase 1 -> 2: Grasp Pose")

        elif self.phase == 2:
            self.wait_steps += 1
            if self.wait_steps > 40:
                self.attach_fixed_joint()
                self.phase = 3
                self.wait_steps = 0
                print("Phase 2 -> 3: Attached")

        elif self.phase == 3:
            self.wait_steps += 1
            if self.wait_steps > 20:
                if self.move_ee_to(lift, thresh=0.04):
                    self.phase = 4
                    print("Phase 3 -> 4: Lift")

        elif self.phase == 4:
            if self.move_ee_to(place_pre, thresh=0.05):
                self.phase = 5
                print("Phase 4 -> 5: Move to Place")

        elif self.phase == 5:
            if self.move_ee_to(place, thresh=0.03):
                self.phase = 6
                self.wait_steps = 0
                print("Phase 5 -> 6: Place Pose")

        elif self.phase == 6:
            self.wait_steps += 1
            if self.wait_steps > 40:
                self.detach_fixed_joint()
                self.phase = 7
                self.wait_steps = 0
                print("Phase 6 -> 7: Detached")

        elif self.phase == 7:
            self.wait_steps += 1
            if self.wait_steps > 20:
                if self.move_ee_to(place_pre, thresh=0.06):
                    print("[DONE] Complete!")
                    self.phase = 8


def main():
    app = MoveArmFixed()
    try:
        app.setup_scene()
        app.setup_post_load()

        while simulation_app.is_running():
            app._world.step(render=True)

    except Exception as e:
        import traceback
        print(f"\n🔥 ERROR: {e}")
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
