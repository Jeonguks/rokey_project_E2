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
        # ===== Configuation Path =====
        self.MAP_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/map/project_map.usd"
        self.MAP_PRIM_PATH = "/World/Map"

        self.ARM_USD_PATH = "/home/rokey/Documents/project/rokey_project_E2/asset/project_arm.usd"
        self.ARM_PRIM_PATH = "/World/Arm"
        # ==================================

        # Initialize Arm position
        self.ARM_POSITION = np.array([-7.55068, -4.08896, 0.0], dtype=np.float32)

        # (Pick&Place) target prim / place position
        self.PICK_TARGET_PRIM_PATH = "/World/Map/Fanta_Can"
        self.PLACE_POS = np.array([-7.82042, -3.38997, 0.2], dtype=np.float32)

        self.APPROACH_Z = 0.12
        self.LIFT_Z = 0.18

        # FixedJoint prim paths (optional fallback)
        self.CONSTRAINTS_ROOT = "/World/Constraints"
        self.FIXED_JOINT_PATH = f"{self.CONSTRAINTS_ROOT}/pick_fixed_joint"

        self._world = None
        self._arm = None
        self.controller = None

        # phase machine
        self.phase = 0
        self.wait_steps = 0

        # EE target orientation
        self.ee_quat = euler_angles_to_quat([0.0, np.pi, 0.0])
        self.EE_LINK_PATH = "/World/Arm/ee_link"

        # fake grasp mode
        self.fake_grasp = True
        self._fake_attached = False
        self._fake_rel_T = None  # object transform in EE local at attach time
        self._fake_collision_prev = None  # remember collision state if needed

        self.OBJ_DRIVER_PATH = "/World/Map/Fanta_Can_driver"


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
    # Rigid body / collision approximation helpers
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
    # Fake grasp (recommended)
    # ---------------------------
    def attach_fake(self, obj_path: str, ee_link_path: str, disable_collision: bool = True):
        T_obj = self._get_world_transform(obj_path)
        T_ee = self._get_world_transform(ee_link_path)

        self._fake_rel_T = T_ee.GetInverse() * T_obj
        self._fake_attached = True

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(obj_path)

        # Make kinematic to avoid physics fighting
        rb = UsdPhysics.RigidBodyAPI(prim)
        if rb:
            rb.CreateKinematicEnabledAttr(True)

        # Disable collision while attached (recommended for "pretend grasp")
        if disable_collision:
            col = UsdPhysics.CollisionAPI(prim)
            if col:
                # remember previous if available
                prev = None
                if col.GetCollisionEnabledAttr():
                    prev = bool(col.GetCollisionEnabledAttr().Get())
                self._fake_collision_prev = prev

                if not col.GetCollisionEnabledAttr():
                    col.CreateCollisionEnabledAttr(True)
                col.GetCollisionEnabledAttr().Set(False)

        print("[ATTACH-FAKE] object will follow EE (kinematic, collision off)")

    def detach_fake(self, obj_path: str, restore_collision: bool = True, make_dynamic: bool = True):
        self._fake_attached = False
        self._fake_rel_T = None

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(obj_path)

        if make_dynamic:
            # detach 시점에만 rigid body 적용 (필요하면)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

        rb = UsdPhysics.RigidBodyAPI(prim)
        if rb:
            rb.CreateKinematicEnabledAttr(False)

        if restore_collision:
            col = UsdPhysics.CollisionAPI(prim)
            if col:
                if not col.GetCollisionEnabledAttr():
                    col.CreateCollisionEnabledAttr(True)
                col.GetCollisionEnabledAttr().Set(True)

        print("[DETACH-FAKE] object released (dynamic on)")

    def _update_fake_attached_pose(self, obj_path: str, ee_link_path: str):
        if not self._fake_attached or self._fake_rel_T is None:
            return

        stage = self._get_stage()
        prim = stage.GetPrimAtPath(obj_path)
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
    # Fixed joint (optional fallback)
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

        # Let stage load settle
        for _ in range(5):
            self._world.step(render=True)

        self._arm = self._world.scene.add(
            SingleArticulation(prim_path=self.ARM_PRIM_PATH, name="arm")
        )

    # --------------------------------------------------------
    def setup_post_load(self):
        self._world.reset()

        # 0) dof names
        dof_names = self._arm.dof_names

        # 1) set "home" joints before play
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

        # 2) set base pose
        self._arm.set_world_pose(position=self.ARM_POSITION)
        self._arm.set_joint_velocities(np.zeros_like(self._arm.get_joint_positions()))

        for _ in range(5):
            self._world.step(render=True)

        # 3) EE link path
        self.ee_link_path = self.EE_LINK_PATH
        print(f"[INFO] EE link path = {self.ee_link_path}")

        # 4) controller
        self.controller = RMPFlowController(
            "rmp_controller", self._arm, self._world.get_physics_dt()
        )
        self.controller.reset()

        # 5) resolve pick target prim
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

        # Ensure dynamic collision approximation etc.
        # 수정: fake_grasp일 땐 물리 바디를 만들지 않음 (경고 원천 제거)
        if not self.fake_grasp:
            self.ensure_rigid_body_recursive(self.PICK_TARGET_PRIM_PATH, approximation="convexHull")
            
        self._world.play()
        self._world.add_physics_callback("sim_step", self.physics_step)

    # --------------------------------------------------------
    def physics_step(self, step_size):
        # Keep fake attached object following EE BEFORE doing phase logic
        self._update_fake_attached_pose(self.PICK_TARGET_PRIM_PATH, self.ee_link_path)

        cube_pos = self._get_prim_world_pos(self.PICK_TARGET_PRIM_PATH)

        pre_grasp = cube_pos.copy()
        pre_grasp[2] += self.APPROACH_Z

        grasp = cube_pos.copy()

        lift = cube_pos.copy()
        lift[2] += self.LIFT_Z

        place_pre = self.PLACE_POS.copy()
        place_pre[2] += self.APPROACH_Z

        place = self.PLACE_POS.copy()

        if self.phase == 0 and self.wait_steps == 0:
            ee_T = self._get_world_transform(self.ee_link_path)
            ee_t = ee_T.ExtractTranslation()
            ee_pos = np.array([float(ee_t[0]), float(ee_t[1]), float(ee_t[2])], dtype=np.float32)
            print("[DBG] can_pos(world) =", cube_pos)
            print("[DBG] ee_pos(world)  =", ee_pos)
            print("[DBG] pre_grasp      =", pre_grasp)
            self.wait_steps = 1

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
                self.detach_fake(self.PICK_TARGET_PRIM_PATH, restore_collision=True)
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
