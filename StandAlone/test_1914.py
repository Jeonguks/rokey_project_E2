import os
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd, Sdf
import omni.kit.viewport.utility as vp_utils


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
        # ===== Paths =====
        self.MAP_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/map/project_map.usd"
        self.MAP_PRIM_PATH = "/World/Map"

        self.ARM_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/asset/project_arm.usd"
        self.ARM_PRIM_PATH = "/World/Arm"
        # ================

        self.ARM_POSITION = np.array([-7.55068, -4.08896, 0.0], dtype=np.float32)

        # Pick/Place target
        self.PICK_TARGET_PRIM_PATH = "/World/Map/Fanta_Can"
        self.PLACE_POS = np.array([-7.82042, -3.38997, 0.2], dtype=np.float32)

        self.APPROACH_Z = 0.12
        self.LIFT_Z = 0.18

        # (optional) fixed joint
        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.FIXED_JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/pick_fixed_joint"

        # EE
        self.EE_LINK_PATH = "/World/Arm/ee_link"
        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])

        # World/robot/controller
        self._world = None
        self._arm = None
        self.controller = None

        # phase
        self.phase = 0
        self.wait_steps = 0

        # ----------------------------
        # Fake grasp options
        # ----------------------------
        self.fake_grasp = True              # True: pretend grasp (follow EE)
        self.make_dynamic_on_detach = True  # True: detach 후 중력으로 떨어지게(ScaleOrientation 경고는 뜰 수 있음)

        self._fake_attached = False
        self._fake_rel_T = None
        self._fake_obj_path = None          # 실제로 움직일 prim path (자동 탐색 결과)
        self._fake_collision_prev = None

    # ---------------------------
    # USD basic utils
    # ---------------------------

    def set_view_to_can(self, can_path: str, distance: float = 2.0, height: float = 1.0):
        """
        캔이 잘 보이도록 뷰포트 카메라를 캔 주변으로 이동.
        - distance: 캔으로부터 카메라 거리
        - height:   캔 기준 카메라 높이 오프셋
        """
        try:
            can_pos = self._get_prim_world_pos(can_path)

            # 카메라 위치: 캔 뒤쪽(-Y)에서 바라보게 (원하면 방향 바꾸면 됨)
            eye = Gf.Vec3d(float(can_pos[0]), float(can_pos[1] - distance), float(can_pos[2] + height))
            target = Gf.Vec3d(float(can_pos[0]), float(can_pos[1]), float(can_pos[2] + 0.05))
            up = Gf.Vec3d(0.0, 0.0, 1.0)

            viewport = vp_utils.get_active_viewport_window()
            if viewport is None:
                print("[WARN] No active viewport window.")
                return

            vp_api = viewport.viewport_api
            vp_api.set_camera_position(eye, False)      # False: 즉시 이동
            vp_api.set_camera_target(target, False)
            vp_api.set_camera_up_vector(up, False)

            print("[OK] View moved to can.")
        except Exception as e:
            print(f"[WARN] set_view_to_can failed: {e}")

    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _get_world_transform(self, prim_path: str) -> Gf.Matrix4d:
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")
        xform_cache = UsdGeom.XformCache()
        return xform_cache.GetLocalToWorldTransform(prim)

    def _get_prim_world_pos(self, prim_path: str) -> np.ndarray:
        T = self._get_world_transform(prim_path)
        t = T.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

    def _mat4_to_pos_quat(self, m: Gf.Matrix4d):
        t = m.ExtractTranslation()
        r = m.ExtractRotation()
        qd = r.GetQuat()
        pos = Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))
        quat = Gf.Quatf(
            float(qd.GetReal()),
            Gf.Vec3f(float(qd.GetImaginary()[0]), float(qd.GetImaginary()[1]), float(qd.GetImaginary()[2])),
        )
        return pos, quat

    def _assert_prim_exists(self, prim_path: str):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found in stage: {prim_path}")

    # ---------------------------
    # Prim resolve helpers
    # ---------------------------
    def wait_for_prim(self, prim_path: str, max_frames: int = 300, render: bool = True) -> bool:
        stage = self._get_stage()
        for _ in range(max_frames):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return True
            self._world.step(render=render)
        return False

    def resolve_prim_by_name(self, prim_name: str):
        stage = self._get_stage()
        for prim in stage.Traverse():
            try:
                if prim.GetName() == prim_name:
                    return prim.GetPath().pathString
            except Exception:
                pass
        return None

    # ---------------------------
    # 핵심: "실제로 움직일 prim" 자동 탐색
    # ---------------------------
    def find_movable_xform_target(self, root_path: str) -> str:
        """
        /World/Map/Fanta_Can 같은 prim이 'reference 루트'라서 xform authoring이 안 먹히는 경우가 흔함.
        그래서 root 아래에서 '움직여도 화면에 반영되는' Xform을 찾아서 그 prim을 이동 대상으로 사용한다.

        우선순위:
        1) root 자신이 UsdGeom.Xform 이면 root
        2) root 아래에서 처음 만나는 UsdGeom.Xform
        3) root 아래에서 Mesh를 만나면 "Mesh의 부모 prim"을 후보로
        4) 그래도 없으면 root
        """
        stage = self._get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Prim not found: {root_path}")

        # (1) root 자체가 Xform이면 best
        if root.IsA(UsdGeom.Xform):
            return root_path

        # (2)(3) traverse
        for prim in Usd.PrimRange(root):
            # skip root itself
            if prim.GetPath().pathString == root_path:
                continue

            if prim.IsA(UsdGeom.Xform):
                return prim.GetPath().pathString

            if prim.IsA(UsdGeom.Mesh):
                parent = prim.GetParent()
                if parent and parent.IsValid():
                    # parent가 Xformable이면 대개 여기 움직이는 게 화면에 반영됨
                    return parent.GetPath().pathString

        return root_path

    # ---------------------------
    # Collision approximation helper (fixed joint/real physics용)
    # ---------------------------
    def ensure_rigid_body_recursive(self, root_path: str, approximation: str = "convexHull"):
        stage = self._get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Prim not found: {root_path}")

        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root)
        if not root.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(root)

        targets = [p for p in Usd.PrimRange(root)]
        for prim in targets:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            try:
                physx_col = (
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
                    else PhysxSchema.PhysxCollisionAPI(prim)
                )
                if hasattr(physx_col, "GetCollisionApproximationAttr"):
                    attr = physx_col.GetCollisionApproximationAttr()
                    if attr:
                        attr.Set(approximation)
                        continue

                a = prim.GetAttribute("physxCollision:collisionApproximation")
                if a and a.IsValid():
                    a.Set(approximation)
                else:
                    prim.CreateAttribute(
                        "physxCollision:collisionApproximation", Sdf.ValueTypeNames.Token
                    ).Set(approximation)
            except Exception:
                pass

        print(f"[OK] ensure_rigid_body_recursive: {root_path} (approx={approximation}, prims={len(targets)})")

    # ---------------------------
    # Fake grasp
    # ---------------------------
    def attach_fake(self, obj_root_path: str, ee_link_path: str, disable_collision: bool = True):
        """
        obj_root_path 아래에서 실제로 움직일 prim을 자동으로 찾고(self._fake_obj_path),
        그 prim을 EE가 매 step 따라가도록 만든다.
        """
        # 1) "진짜로 움직일" 대상 prim 결정
        movable_path = self.find_movable_xform_target(obj_root_path)
        self._fake_obj_path = movable_path
        print("[DBG] movable target =", self._fake_obj_path)

        # 2) 상대변환 저장
        T_obj = self._get_world_transform(self._fake_obj_path)
        T_ee = self._get_world_transform(ee_link_path)
        self._fake_rel_T = T_ee.GetInverse() * T_obj
        self._fake_attached = True

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self._fake_obj_path)

        # 3) kinematic 켜기 (물리와 싸우지 않게)
        #    (RigidBodyAPI가 없어도 CreateKinematicEnabledAttr는 만들어질 수 있는데,
        #     안전하게 Apply 후 사용)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateKinematicEnabledAttr(True)

        # 4) collision off (연출용)
        if disable_collision:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            col = UsdPhysics.CollisionAPI(prim)
            prev = None
            if col.GetCollisionEnabledAttr():
                try:
                    prev = bool(col.GetCollisionEnabledAttr().Get())
                except Exception:
                    prev = None
            self._fake_collision_prev = prev

            if not col.GetCollisionEnabledAttr():
                col.CreateCollisionEnabledAttr(True)
            col.GetCollisionEnabledAttr().Set(False)

        print("[ATTACH-FAKE] object will follow EE (kinematic, collision off)")

    def detach_fake(self, restore_collision: bool = True, make_dynamic: bool = True):
        """
        가짜 그립 해제.
        - make_dynamic=True면 kinematic off로 풀어서 중력 받게 함.
        """
        if not self._fake_obj_path:
            # attach 전에 detach 호출된 케이스 방지
            self._fake_attached = False
            self._fake_rel_T = None
            print("[DETACH-FAKE] no target (skip)")
            return

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self._fake_obj_path)

        # collision 복구
        if restore_collision and prim and prim.IsValid():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            col = UsdPhysics.CollisionAPI(prim)
            if not col.GetCollisionEnabledAttr():
                col.CreateCollisionEnabledAttr(True)

            if self._fake_collision_prev is None:
                col.GetCollisionEnabledAttr().Set(True)
            else:
                col.GetCollisionEnabledAttr().Set(bool(self._fake_collision_prev))

        # kinematic 해제 (dynamic)
        if make_dynamic and prim and prim.IsValid():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateKinematicEnabledAttr(False)

        self._fake_attached = False
        self._fake_rel_T = None
        self._fake_collision_prev = None
        print("[DETACH-FAKE] object released (dynamic on)" if make_dynamic else "[DETACH-FAKE] object released")

        # 다음 attach를 위해 초기화
        self._fake_obj_path = None

    def _update_fake_attached_pose(self, ee_link_path: str):
        if not self._fake_attached or self._fake_rel_T is None or not self._fake_obj_path:
            return

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self._fake_obj_path)
        if not prim or not prim.IsValid():
            return

        T_ee = self._get_world_transform(ee_link_path)
        T_obj_target = T_ee * self._fake_rel_T

        xform = UsdGeom.Xformable(prim)
        t = T_obj_target.ExtractTranslation()
        r = T_obj_target.ExtractRotation().GetQuat()  # Quatd

        ops = xform.GetOrderedXformOps()
        op_t = None
        op_r = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op_t = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op_r = op

        if op_t is None:
            op_t = xform.AddTranslateOp()
        if op_r is None:
            op_r = xform.AddOrientOp()

        op_t.Set(Gf.Vec3d(t[0], t[1], t[2]))
        op_r.Set(Gf.Quatd(r.GetReal(), r.GetImaginary()))

    # ---------------------------
    # FixedJoint (옵션)
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

        print(f"[ATTACH] FixedJoint created: {self.FIXED_JOINT_PATH}")

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
            target_end_effector_orientation=self.ee_quat,
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
        add_reference_to_stage(self.ARM_USD_PATH, self.ARM_PRIM_PATH)

        # settle
        for _ in range(5):
            self._world.step(render=True)

        self._arm = self._world.scene.add(
            SingleArticulation(prim_path=self.ARM_PRIM_PATH, name="arm")
        )

    # --------------------------------------------------------
    def setup_post_load(self):
        self._world.reset()

        # dof names
        dof_names = self._arm.dof_names

        # home joints
        q_home_deg_by_name = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -90.0,
            "elbow_joint": 90.0,
            "wrist_1_joint": -90.0,
            "wrist_2_joint": -90.0,
            "wrist_3_joint": 0.0,
        }

        q_current = self._arm.get_joint_positions()
        q_home_rad = np.array(q_current, dtype=np.float32)
        for i, name in enumerate(dof_names):
            if name in q_home_deg_by_name:
                q_home_rad[i] = np.deg2rad(q_home_deg_by_name[name])

        self._arm.set_joint_positions(q_home_rad)
        self._arm.set_joint_velocities(np.zeros_like(q_home_rad))

        # base pose
        self._arm.set_world_pose(position=self.ARM_POSITION)
        self._arm.set_joint_velocities(np.zeros_like(self._arm.get_joint_positions()))

        for _ in range(5):
            self._world.step(render=True)

        # EE link path
        self.ee_link_path = self.EE_LINK_PATH
        print(f"[INFO] EE link path = {self.ee_link_path}")

        # controller
        self.controller = RMPFlowController(
            "rmp_controller", self._arm, self._world.get_physics_dt()
        )
        self.controller.reset()

        # pick prim resolve
        if not self.wait_for_prim(self.PICK_TARGET_PRIM_PATH, max_frames=240, render=True):
            name_guess = self.PICK_TARGET_PRIM_PATH.split("/")[-1]
            found = self.resolve_prim_by_name(name_guess)
            if found is None:
                raise RuntimeError(
                    f"Prim not found: {self.PICK_TARGET_PRIM_PATH} (also not found by name='{name_guess}')"
                )
            print(f"[WARN] Resolved by name: {self.PICK_TARGET_PRIM_PATH} -> {found}")
            self.PICK_TARGET_PRIM_PATH = found

        self._assert_prim_exists(self.PICK_TARGET_PRIM_PATH)

        # 뷰를 캔 근처로 이동
        self.set_view_to_can(self.PICK_TARGET_PRIM_PATH, distance=2.5, height=1.2)

        # fake_grasp면 초기부터 rigid body 강제 적용은 안 함 (ScaleOrientation 경고 원인 줄이기)
        if not self.fake_grasp:
            self.ensure_rigid_body_recursive(self.PICK_TARGET_PRIM_PATH, approximation="convexHull")

        self._world.play()
        self._world.add_physics_callback("sim_step", self.physics_step)

    # --------------------------------------------------------
    def physics_step(self, step_size):
        # fake attach 상태면 먼저 EE 따라가도록 업데이트
        self._update_fake_attached_pose(self.ee_link_path)

        # targets
        cube_pos = self._get_prim_world_pos(self.PICK_TARGET_PRIM_PATH)

        pre_grasp = cube_pos.copy()
        pre_grasp[2] += self.APPROACH_Z

        grasp = cube_pos.copy()

        lift = cube_pos.copy()
        lift[2] += self.LIFT_Z

        place_pre = self.PLACE_POS.copy()
        place_pre[2] += self.APPROACH_Z

        place = self.PLACE_POS.copy()

        # debug once
        if self.phase == 0 and self.wait_steps == 0:
            ee_T = self._get_world_transform(self.ee_link_path)
            ee_t = ee_T.ExtractTranslation()
            ee_pos = np.array([float(ee_t[0]), float(ee_t[1]), float(ee_t[2])], dtype=np.float32)

            print("[DBG] can_pos(world) =", cube_pos)
            print("[DBG] ee_pos(world)  =", ee_pos)
            print("[DBG] pre_grasp      =", pre_grasp)
            self.wait_steps = 1

        # phase machine
        if self.phase == 0:
            if self.move_ee_to(pre_grasp):
                self.controller.reset()
                self.phase = 1

        elif self.phase == 1:
            if self.move_ee_to(grasp):
                self.controller.reset()
                self.phase = 2
                self.wait_steps = 0

        elif self.phase == 2:
            self.wait_steps += 1
            if self.wait_steps > 25:
                self.phase = 3

        elif self.phase == 3:
            if self.fake_grasp:
                self.attach_fake(self.PICK_TARGET_PRIM_PATH, self.ee_link_path, disable_collision=True)
            else:
                # real physics mode (optional)
                self.ensure_rigid_body_recursive(self.PICK_TARGET_PRIM_PATH, approximation="convexHull")
                self.attach_fixed_joint(self.PICK_TARGET_PRIM_PATH, self.ee_link_path)
            self.phase = 4

        elif self.phase == 4:
            if self.move_ee_to(lift):
                self.controller.reset()
                self.phase = 5

        elif self.phase == 5:
            if self.move_ee_to(place_pre):
                self.controller.reset()
                self.phase = 6

        elif self.phase == 6:
            if self.move_ee_to(place):
                self.controller.reset()
                self.phase = 7
                self.wait_steps = 0

        elif self.phase == 7:
            self.wait_steps += 1
            if self.wait_steps > 20:
                self.phase = 8

        elif self.phase == 8:
            if self.fake_grasp:
                self.detach_fake(
                    restore_collision=True,
                    make_dynamic=self.make_dynamic_on_detach
                )
            else:
                self.detach_fixed_joint()
            self.phase = 9
            print("[DONE] Pick & Place finished.")

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
