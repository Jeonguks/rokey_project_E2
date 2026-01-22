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
        self.ARM_POSITION = np.array([-7.55068, -4.08896, 0.0])

        # (Pick&Place) 물체/목표
        # self.CUBE_PRIM_PATH = "/World/pick_cube"
        # self.CUBE_SIZE = 0.04
        # self.CUBE_START_POS = np.array([0.50, 0.50, 0.40])   # 큐브 시작 위치

        self.PICK_TARGET_PRIM_PATH = "/World/Map/Fanta_Can"   # 맵에 이미 있는 프림 경로
        self.PLACE_POS = np.array([-7.82042, -3.38997, 0.2], dtype=np.float32) # 놔둘 위치
#-7.82042, -3.38997, 0.92712]
#-7.40
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

        for _ in range(5):
            self._world.step(render=True)


        self._arm = self._world.scene.add(
            SingleArticulation(
                prim_path=self.ARM_PRIM_PATH,
                name="arm",
            )
        )

        # # Pick 대상 큐브 추가
        # self._cube = self._world.scene.add(
        #     DynamicCuboid(
        #         prim_path=self.CUBE_PRIM_PATH,
        #         name="pick_cube",
        #         position=self.CUBE_START_POS,
        #         scale=np.array([self.CUBE_SIZE, self.CUBE_SIZE, self.CUBE_SIZE]),
        #     )
        # )

    # --------------------------------------------------------
    def setup_post_load(self):
        self._world.reset()

        # 0) dof names 확보
        dof_names = self._arm.dof_names

        # 1) 먼저 "홈 자세"를 세팅 (물리 play 전)
        q_home_deg_by_name = {
            "shoulder_pan_joint":   0.0,
            "shoulder_lift_joint": -90.0,
            "elbow_joint":          90.0,
            "wrist_1_joint":       -90.0,
            "wrist_2_joint":       -90.0,
            "wrist_3_joint":        0.0,
        }

        q_current = self._arm.get_joint_positions()
        q_home_rad = np.array(q_current, dtype=np.float32)
        for i, name in enumerate(dof_names):
            if name in q_home_deg_by_name:
                q_home_rad[i] = np.deg2rad(q_home_deg_by_name[name])

        # ★ 먼저 조인트 포즈부터
        self._arm.set_joint_positions(q_home_rad)
        self._arm.set_joint_velocities(np.zeros_like(q_home_rad))

        # 2) 그 다음 베이스 위치를 확정
        self._arm.set_world_pose(position=self.ARM_POSITION)
        self._arm.set_joint_velocities(np.zeros_like(self._arm.get_joint_positions()))

        # 3) 2~5프레임 반영 (끼임/충돌 전에 안정화)
        for _ in range(5):
            self._world.step(render=True)

        # 4) EE 링크 경로
        self.ee_link_path = self.EE_LINK_PATH
        print(f"[INFO] EE link path = {self.ee_link_path}")

        # 5) 컨트롤러 생성/리셋은 제일 마지막에
        self.controller = RMPFlowController(
            "rmp_controller",
            self._arm,
            self._world.get_physics_dt(),
        )
        self.controller.reset()

        # 6) pick prim 로드/경로 확정 (play 전)
        # 6-1) 먼저 지정 경로로 기다려본다
        if not self.wait_for_prim(self.PICK_TARGET_PRIM_PATH, max_frames=240, render=True):
            # 6-2) 여전히 없으면 "이름"으로 Stage 전체에서 찾아서 경로 보정
            name_guess = self.PICK_TARGET_PRIM_PATH.split("/")[-1]  # "Fanta_Can"
            found = self.resolve_prim_by_name(name_guess)
            if found is None:
                # 여기서 죽으면: 맵 USD에 진짜로 없거나, 더 늦게 로딩되거나, 이름이 다름
                raise RuntimeError(
                    f"Prim not found: {self.PICK_TARGET_PRIM_PATH} (also not found by name='{name_guess}'). "
                    f"Check USD contents / prim name / load timing."
                )
            print(f"[WARN] PICK_TARGET_PRIM_PATH not found. Resolved by name: {self.PICK_TARGET_PRIM_PATH} -> {found}")
            self.PICK_TARGET_PRIM_PATH = found

        # 최종 검증 + 물리 적용
        self._assert_prim_exists(self.PICK_TARGET_PRIM_PATH)
        self.ensure_rigid_body_recursive(self.PICK_TARGET_PRIM_PATH, approximation="convexHull")

        self._world.play()
        self._world.add_physics_callback("sim_step", self.physics_step)
    #-------------------------------------------------------------

    def wait_for_prim(self, prim_path: str, max_frames: int = 300, render: bool = True) -> bool:
        """
        prim이 stage에 나타날 때까지 world.step()을 돌리며 대기
        """
        stage = self._get_stage()
        for _ in range(max_frames):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return True
            # 로딩/초기화가 진행되도록 step
            self._world.step(render=render)
        return False

    def resolve_prim_by_name(self, prim_name: str) -> str | None:
        """
        Stage 전체를 훑어서 이름이 prim_name인 prim의 path를 찾아 반환
        (예: "Fanta_Can" 이름 프림을 찾아 실제 경로를 알아냄)
        """
        stage = self._get_stage()
        for prim in stage.Traverse():
            try:
                if prim.GetName() == prim_name:
                    return prim.GetPath().pathString
            except Exception:
                pass
        return None



    #################3
    def _assert_prim_exists(self, prim_path: str):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found in stage: {prim_path}")

    def _get_prim_world_pos(self, prim_path: str) -> np.ndarray:
        T = self._get_world_transform(prim_path)
        t = T.ExtractTranslation()
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
    

    # 집을 물체의 물리법칙 적용 유무 확인

    def ensure_rigid_body(self, prim_path: str, approximation: str = "convexHull"):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim not found: {prim_path}")

        # 1) RigidBody / Collision 적용
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        # 2) PhysX collision approximation 설정 (버전 호환)
        # - 어떤 버전은 CreateCollisionApproximationAttr가 없고 GetCollisionApproximationAttr만 있음
        # - 어떤 버전은 Attr 이름/네임스페이스가 다를 수 있음
        try:
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim) if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI) \
                        else PhysxSchema.PhysxCollisionAPI(prim)

            # (A) 가장 흔한 패턴: GetCollisionApproximationAttr().Set(...)
            if hasattr(physx_col, "GetCollisionApproximationAttr"):
                attr = physx_col.GetCollisionApproximationAttr()
                if attr:
                    attr.Set(approximation)
                    print(f"[OK] ensure_rigid_body: {prim_path} (approx={approximation})")
                    return

            # (B) 속성명을 직접 찾아서 세팅 (fallback)
            # 속성 후보들 (버전에 따라 다를 수 있음)
            candidate_attr_names = [
                "physxCollision:collisionApproximation",
                "physxCollision:collision_approximation",
                "physxCollision:approximation",
                "collisionApproximation",
            ]
            for name in candidate_attr_names:
                a = prim.GetAttribute(name)
                if a and a.IsValid():
                    a.Set(approximation)
                    print(f"[OK] ensure_rigid_body: {prim_path} (approx={approximation}, attr={name})")
                    return

            # (C) 아예 없으면 create 해서 세팅
            # USD에서 string token 계열로 들어가는 경우가 많아 Sdf.ValueTypeNames.Token을 사용
            from pxr import Sdf
            a = prim.CreateAttribute("physxCollision:collisionApproximation", Sdf.ValueTypeNames.Token)
            a.Set(approximation)
            print(f"[OK] ensure_rigid_body: {prim_path} (approx={approximation}, created attr)")

        except Exception as e:
            # 여기서 죽지 않게 하고, PhysX가 자동 fallback(convexHull) 하게 둠
            print(f"[WARN] ensure_rigid_body: could not set collision approximation for {prim_path}. "
                  f"Will rely on PhysX fallback. Error: {e}")
    def ensure_rigid_body_recursive(self, root_path: str, approximation: str = "convexHull"):
        """
        root_path 아래에서 PhysX가 실제로 collision을 읽는 Mesh/Gprim까지 포함하여
        RigidBody/Collision 및 collisionApproximation을 강제한다.
        """
        stage = self._get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Prim not found: {root_path}")

        # (1) root에는 RigidBody를 적용 (움직일 바디는 root로 잡는 편이 안정적)
        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root)
        if not root.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(root)

        # (2) root와 자식들의 collision approximation을 전부 강제
        #     - PhysX는 보통 Mesh/Gprim prim에서 collision을 파싱함
        targets = []
        for prim in Usd.PrimRange(root):
            # Xform이든 Mesh든 일단 collisionApproximation은 걸어두는 게 안전
            targets.append(prim)

        for prim in targets:
            # Collision API는 Mesh/Gprim에 붙이는 게 일반적이지만,
            # 어떤 USD는 collision이 상위에 붙어있을 수도 있어 광범위 적용
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

            # Physx collision approximation (호환형)
            try:
                physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim) if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI) \
                            else PhysxSchema.PhysxCollisionAPI(prim)

                if hasattr(physx_col, "GetCollisionApproximationAttr"):
                    attr = physx_col.GetCollisionApproximationAttr()
                    if attr:
                        attr.Set(approximation)
                        continue

                # attribute 직접 set / create
                a = prim.GetAttribute("physxCollision:collisionApproximation")
                if a and a.IsValid():
                    a.Set(approximation)
                else:
                    prim.CreateAttribute("physxCollision:collisionApproximation", Sdf.ValueTypeNames.Token).Set(approximation)

            except Exception:
                # 여기서 죽지 않게
                pass

        print(f"[OK] ensure_rigid_body_recursive: {root_path} (approx={approximation}, prims={len(targets)})")


    # --------------------------------------------------------
    def physics_step(self, step_size):
        # 0) 목표 포즈들 계산을 맨 먼저 (cube_pos 선계산)
        cube_pos = self._get_prim_world_pos(self.PICK_TARGET_PRIM_PATH)

        pre_grasp = cube_pos.copy()
        pre_grasp[2] += self.APPROACH_Z

        grasp = cube_pos.copy()

        lift = cube_pos.copy()
        lift[2] += self.LIFT_Z

        place_pre = self.PLACE_POS.copy()
        place_pre[2] += self.APPROACH_Z

        place = self.PLACE_POS.copy()

        # 1) 디버그는 cube_pos/pre_grasp 만든 뒤에
        if self.phase == 0 and self.wait_steps == 0:
            ee_T = self._get_world_transform(self.EE_LINK_PATH)
            ee_t = ee_T.ExtractTranslation()
            ee_pos = np.array([float(ee_t[0]), float(ee_t[1]), float(ee_t[2])], dtype=np.float32)

            print("[DBG] can_pos(world) =", cube_pos)
            print("[DBG] ee_pos(world)  =", ee_pos)
            print("[DBG] pre_grasp      =", pre_grasp)
            self.wait_steps = 1  # 한 번만 출력

        # 2) 기존 phase 로직
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
