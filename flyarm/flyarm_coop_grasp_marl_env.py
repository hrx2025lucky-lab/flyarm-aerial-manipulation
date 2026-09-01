# ================================================================
# 双机协同抓取 + 搬运 — 分布式 MARL 环境 (DirectMARLEnv, CTDE/MAPPO)
#
# 两台空中机械臂各抓共享长杆的一端, 协同把杆抬离台座并搬到空中目标。
# 与集中式版本任务/物理/参数完全相同, 区别只在"多机组织方式":
#   集中式: 单一网络看 75 维全局观测、出 18 维联合动作, 两机奖励相加
#           → 信用连坐(一机的好动作会被另一机的失败拖累)。
#   本版(MAPPO/CTDE): 每机一个独立 agent —— 各自的策略网络看 47 维 egocentric
#           观测、各出 9 维动作、各拿各的奖励(信用干净); 训练时另有一个中心化
#           critic 看 75 维全局 state。执行时每机只用自己的观测 + 队友广播
#           (CTDE: 集中训练、分散执行 → 真机去中心化部署友好)。
#
# 机间通信: 每机观测含 9 维通信块(队友广播相对位置/相对速度/是否夹住/是否
#   正在合爪/距其目标多远), 全部转到自己机体系(egocentric)。配合 yaw180 镜像,
#   两机看到的世界完全对称。真机上 = 50Hz 广播 9 个浮点。
#
# 参数单一来源: 直接 import 集中式版本的 cfg 实例(_P)。LADRC 增益、重心前馈、
#   抓取几何、全部奖励权重都从 _P 读 → 两版永远同参数。
#
# 奖励分账(MAPPO 的核心收益):
#   每机 18 项个人栈(dist/orientation/reaching/align/grasp_close/contact/...) → 归各自;
#   共享项(grasp_lift/lift_goal/lifting/bar_goal/bar_level/hold/internal/...) → 两机同发。
#   协作项共享是合作任务的标准做法; 个人技能项分账可消除信用串扰。
#
# 运行前置:
#   1) gymnasium 必须为 0.29.1 (DirectMARLEnv 在 1.0 上 state() 有问题):
#      isaaclab.bat -p -m pip install "gymnasium==0.29.1"
#   2) skrl 已随 isaaclab 全量安装。
#   训练: scripts/reinforcement_learning/skrl/train.py --task Isaac-Flyarm-CoopGrasp-MARL-v0
#         --algorithm MAPPO --headless --num_envs 1024
# ================================================================

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_apply_inverse
from isaaclab.sensors import ContactSensor

from .flyarm_coop_grasp_env import FlyarmCoopGraspEnvCfg
from .cascade_ladrc import CascadeLADRC

# 参数单一来源: 集中式版本的全部已调参数
_P = FlyarmCoopGraspEnvCfg()


# ──────────────────────────────────────────────────────────────
# 配置: DirectMARLEnvCfg 骨架 + 资产/参数全部引用 _P
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmCoopGraspMarlEnvCfg(DirectMARLEnvCfg):

    # === MARL 接口 ===
    possible_agents = ["drone_0", "drone_1"]
    action_spaces = {"drone_0": 9, "drone_1": 9}          # 每机: 加速度3+偏航率1+臂4+夹爪1
    # 每机 47 = 自己 32(本体29+杆轴3) + 杆状态转机体系 6 + 通信块 9(队友广播)。
    # 全部 egocentric, 配合 yaw180 镜像 → 两机视角对称。
    observation_spaces = {"drone_0": 47, "drone_1": 47}
    state_space = 75                                      # 中心化 critic 用(仅训练时存在): 32×2 + 世界系共享11

    # === 骨架(从 _P 搬, configclass 实例化时深拷贝) ===
    episode_length_s = 18.0
    decimation = 2
    sim = _P.sim
    scene = _P.scene
    terrain = _P.terrain
    debug_vis = False

    # === 资产(复用集中式 cfg 对象: prim 路径/碰撞/触觉全相同) ===
    robot_0 = _P.robot_0
    robot_1 = _P.robot_1
    bar = _P.bar
    pedestal_0 = _P.pedestal_0
    pedestal_1 = _P.pedestal_1
    contact_0_left = _P.contact_0_left
    contact_0_right = _P.contact_0_right
    contact_1_left = _P.contact_1_left
    contact_1_right = _P.contact_1_right


# ──────────────────────────────────────────────────────────────
# 环境主类(逻辑同集中式; 接口改成按 agent 的 dict)
# ──────────────────────────────────────────────────────────────
class FlyarmCoopGraspMarlEnv(DirectMARLEnv):
    """双机协同抓取 MARL 版。

    任务/物理/奖励逻辑与集中式版本一致, 接口为每机一个 agent。obs/action/
    reward/done 都按 agent 名("drone_0"/"drone_1")的字典进出。
    """

    cfg: FlyarmCoopGraspMarlEnvCfg

    def __init__(self, cfg: FlyarmCoopGraspMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device

        # ── 动作/飞控(内部仍用 (N,18) 拼接张量跑与集中式相同的逻辑) ──
        self._actions = torch.zeros(N, 18, device=dev)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(N, 2, 1, 3, device=dev)
        self._moment = torch.zeros(N, 2, 1, 3, device=dev)

        # ── 每机/任务状态 ──
        self._arm_dof_targets = torch.zeros(N, 2, 4, device=dev)
        self._desired_pos_w = torch.zeros(N, 2, 3, device=dev)
        self._prev_ee_dist = torch.zeros(N, 2, device=dev)
        self._bar_initial_pos = torch.zeros(N, 3, device=dev)
        self._max_bar_height = torch.zeros(N, device=dev)
        self._carry = torch.zeros(N, dtype=torch.bool, device=dev)
        # carry 后飞行目标 z 的 ease-out 爬升进度(0→1): 给去中心化两机一个平滑设定值,
        # 避免目标瞬跳 +carry_height 把杆甩飞。
        self._carry_ramp = torch.zeros(N, device=dev)
        self._in_goal_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._success = torch.zeros(N, dtype=torch.bool, device=dev)
        self._global_step = 0   # 全局训练步计数(每 env-step +1), 驱动 hold 课程的收紧进度; 不随 episode 重置
        self._contact_latch = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self._max_finger_force = torch.zeros(N, 2, device=dev)
        self._max_weak_finger = torch.zeros(N, 2, device=dev)
        self._lat_bias_sum = torch.zeros(N, 2, device=dev)   # 侧向偏置探针(标定夹持中心 vs 杆轴)
        self._lat_cnt = torch.zeros(N, 2, device=dev)
        # 接触去抖(距上次 both_contact 的步数; 999=本局没夹到过) + carry/all_contact 诊断计数。
        # 去抖原因: 多指多机时 PhysX 接触力逐帧闪烁, 二值门控会按帧掐断抬升奖励。
        self._since_both = torch.full((N, 2), 999, dtype=torch.long, device=dev)
        self._carry_cnt = torch.zeros(N, device=dev)
        self._allc_cnt = torch.zeros(N, device=dev)
        self._died_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self._steps_since_reset = torch.zeros(N, dtype=torch.long, device=dev)
        self._grasp_phase = torch.zeros(N, dtype=torch.bool, device=dev)

        # ── obs 缓存(_get_states 复用 _get_observations 的计算) ──
        self._own_obs = [torch.zeros(N, 32, device=dev), torch.zeros(N, 32, device=dev)]
        self._shared_obs = torch.zeros(N, 11, device=dev)

        self._robots = [self._robot_0, self._robot_1]
        self._contacts_l = [self._contact_0_left, self._contact_1_left]
        self._contacts_r = [self._contact_0_right, self._contact_1_right]
        self._end_sign = (1.0, -1.0) if _P.swap_ends else (-1.0, 1.0)   # 换端诊断开关(参数源 _P)

        # ── 串级 LADRC(num_agents=2, 参数全部来自 _P) ──
        self.ladrc = CascadeLADRC(
            N, num_agents=2, device=dev,
            in_wc_rp=_P.ladrc_in_wc_rp, in_wo_rp=_P.ladrc_in_wo_rp, in_b0_rp=_P.ladrc_in_b0_rp,
            in_wc_yaw=_P.ladrc_in_wc_yaw, in_wo_yaw=_P.ladrc_in_wo_yaw, in_b0_yaw=_P.ladrc_in_b0_yaw,
            out_wc=_P.ladrc_out_wc, out_wo=_P.ladrc_out_wo, out_b0=_P.ladrc_out_b0,
            z2_clamp_in_rp=0.3 * _P.ladrc_in_b0_rp,
            z2_clamp_in_yaw=0.2 * _P.ladrc_in_b0_yaw,
            z2_clamp_out=1.0,
            max_angle=_P.ladrc_max_angle, max_rate=_P.ladrc_max_rate,
        )

        # ── 索引 ──
        self._body_ids, self._rotor_ids, self._arm_ids, self._grip_ids = [], [], [], []
        self._ee_ids, self._lf_ids, self._rf_ids = [], [], []
        for rb in self._robots:
            self._body_ids.append(rb.find_bodies("base_link")[0])
            self._rotor_ids.append(rb.find_joints(".*_rotor_joint")[0])
            self._arm_ids.append(rb.find_joints(
                ["base_rotate_joint", "axis_B_joint", "axis_C_joint", "axis_D_joint"], preserve_order=True)[0])
            self._ee_ids.append(rb.find_bodies("gripper_base")[0])
            self._grip_ids.append(rb.find_joints(
                ["gripper_right_joint", "gripper_left_joint"], preserve_order=True)[0])
            self._lf_ids.append(rb.find_bodies("gripper_left")[0])
            self._rf_ids.append(rb.find_bodies("gripper_right")[0])

        # ── 臂限位(底座旋转收紧到 center ± clamp) ──
        if hasattr(self._robot_0.data, "soft_joint_pos_limits"):
            soft = self._robot_0.data.soft_joint_pos_limits[0]
        else:
            soft = self._robot_0.data.joint_pos_limits[0]
        aids0 = self._arm_ids[0]
        self._arm_lower = soft[aids0, 0].clone()
        self._arm_upper = soft[aids0, 1].clone()
        self._arm_lower[0] = torch.clamp(self._arm_lower[0], min=_P.base_rotate_center - _P.base_rotate_clamp)
        self._arm_upper[0] = torch.clamp(self._arm_upper[0], max=_P.base_rotate_center + _P.base_rotate_clamp)

        # ── 常量张量 ──
        self._gripper_open_t = torch.tensor(_P.gripper_open, device=dev)
        self._gripper_close_t = torch.tensor(_P.gripper_close, device=dev)
        self._arm_stow_t = torch.tensor(_P.arm_stow, device=dev)
        self._arm_straight_target_t = torch.tensor(_P.arm_straight_target, device=dev)
        self._arm_straight_pose_t = torch.tensor(
            [_P.base_rotate_center, _P.arm_straight_target[0], _P.arm_straight_target[1], _P.arm_stow[3]],
            device=dev,
        )
        self._grasp_offset_t = torch.tensor(_P.grasp_offset, device=dev)

        # ── 夹爪 effort 上限(温和合拢力, 防穿模卡死) ──
        try:
            eff = torch.full((N, 2), float(_P.gripper_effort_limit_2b), device=dev)
            for i, rb in enumerate(self._robots):
                rb.write_joint_effort_limit_to_sim(eff, joint_ids=self._grip_ids[i])
            print(f"[marl] 夹爪 effort={_P.gripper_effort_limit_2b}")
        except Exception as e:
            print(f"[marl] 夹爪 effort 写入跳过: {e}")

        # ── 质量/重力 + 重心前馈连杆质量 ──
        self._robot_mass = self._robot_0.root_physx_view.get_masses()[0].sum()
        self._body_masses = self._robot_0.root_physx_view.get_masses()[0].to(dev)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=dev).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ── 杆超重质量课程: 训练早期把杆设超重, 单台抬不动也拖不斜, 逼出"两机同时夹住"
        #    才有进展的同时性约束。──
        if _P.use_bar_heavy_curriculum:
            try:
                m = self._bar.root_physx_view.get_masses().clone()   # (num_envs,1) 通常在 CPU
                m[:, 0] = _P.bar_heavy_mass
                self._bar.root_physx_view.set_masses(m, torch.arange(m.shape[0]))
                print(f"[marl] 杆超重质量课程: bar {_P.bar_mass}→{_P.bar_heavy_mass}kg")
            except Exception as e:
                print(f"[marl] 杆超重设置跳过(API 不匹配): {e}")

        # ── 奖励记录(键名与集中式一致, 便于曲线对比) ──
        self._per_keys = [
            "dist_to_goal", "orientation", "reaching", "finger", "gripper_vertical", "arm_straight",
            "align", "grasp_close", "contact_grasp", "own_end_lift", "touch", "engagement", "progress",
            "lin_vel", "ang_vel", "action_rate", "joint_vel", "crash",
        ]
        shared_keys = ["grasp_lift", "lift_goal", "lifting", "bar_goal", "bar_level", "bar_level_pre", "hold_bonus", "internal", "bar_still",
                       "coop_ready", "coop_close", "coop_grasp", "coop_descend"]   # 协作阶梯 + 弱链下探
        sum_keys = [k + "_0" for k in self._per_keys] + [k + "_1" for k in self._per_keys] + shared_keys
        self._episode_sums = {k: torch.zeros(N, dtype=torch.float, device=dev) for k in sum_keys}

        print(f"[marl] MAPPO 版: 每机 obs{self.cfg.observation_spaces['drone_0']}(含9维通信块)/act9, "
              f"state{self.cfg.state_space}, cog_ff={_P.cog_ff_gain if _P.use_cog_feedforward else 0}")
        if hasattr(self, "set_debug_vis"):
            self.set_debug_vis(self.cfg.debug_vis)

    # ==============================================================
    # 场景
    # ==============================================================
    def _setup_scene(self):
        self._robot_0 = Articulation(self.cfg.robot_0)
        self._robot_1 = Articulation(self.cfg.robot_1)
        self.scene.articulations["robot_0"] = self._robot_0
        self.scene.articulations["robot_1"] = self._robot_1

        self._bar = RigidObject(self.cfg.bar)
        self.scene.rigid_objects["bar"] = self._bar
        self._pedestal_0 = RigidObject(self.cfg.pedestal_0)
        self._pedestal_1 = RigidObject(self.cfg.pedestal_1)
        self.scene.rigid_objects["pedestal_0"] = self._pedestal_0
        self.scene.rigid_objects["pedestal_1"] = self._pedestal_1

        self._contact_0_left = ContactSensor(self.cfg.contact_0_left)
        self._contact_0_right = ContactSensor(self.cfg.contact_0_right)
        self._contact_1_left = ContactSensor(self.cfg.contact_1_left)
        self._contact_1_right = ContactSensor(self.cfg.contact_1_right)
        self.scene.sensors["contact_0_left"] = self._contact_0_left
        self.scene.sensors["contact_0_right"] = self._contact_0_right
        self.scene.sensors["contact_1_left"] = self._contact_1_left
        self.scene.sensors["contact_1_right"] = self._contact_1_right

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # ── 焊接课程阶段(weld_phase<2): 在 base_link↔Bar 两端建焊接关节, 杆焊死在两机下方,
        #   先学"稳定悬停 + 协同搬运焊死的杆", 再退火到真抓取。出生时机体在杆端上方
        #   attach_drop 处 → 显式 local pose 零应力。excludeFromArticulation=True 保持杆为独立
        #   刚体(否则杆被吸进 articulation → reset 写不进位姿 + 约束爆炸)。
        if _P.weld_phase < 2:
            import omni.usd
            from pxr import Gf, PhysxSchema, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            for i, sx in enumerate((-1.0, 1.0)):
                jpath = f"/World/envs/env_0/BarWeld_{i}"
                if _P.attach_mode == "ball":
                    joint = UsdPhysics.SphericalJoint.Define(stage, jpath)
                else:
                    joint = UsdPhysics.FixedJoint.Define(stage, jpath)
                joint.CreateBody0Rel().SetTargets([f"/World/envs/env_0/Robot_{i}/base_link"])
                joint.CreateBody1Rel().SetTargets(["/World/envs/env_0/Bar"])
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -float(_P.attach_drop)))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(sx * float(_P.bar_grasp_x), 0.0, 0.0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                joint.CreateExcludeFromArticulationAttr().Set(True)
                pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
                pj.CreateJointFrictionAttr().Set(float(_P.attach_joint_friction))
            print(f"[marl] 焊接关节已建(weld_phase={_P.weld_phase}, mode={_P.attach_mode}, drop={_P.attach_drop})")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── 工具方法 ──
    def _finger_mag(self, sensor: ContactSensor) -> torch.Tensor:
        fmat = sensor.data.force_matrix_w
        if fmat is None:
            fmat = sensor.data.net_forces_w
        return torch.linalg.norm(fmat.reshape(self.num_envs, -1, 3), dim=-1).sum(dim=-1)

    def _both_contact(self, i: int) -> torch.Tensor:
        # 非对称判定: 主指 > 阈值 且 弱指 > 弱阈值。杆被外部约束时两指力可不等,
        # 抬离后自动均衡, 故弱指阈值放低。
        fl = self._finger_mag(self._contacts_l[i])
        fr = self._finger_mag(self._contacts_r[i])
        return (torch.maximum(fl, fr) > _P.contact_force_threshold) \
            & (torch.minimum(fl, fr) > _P.contact_weak_threshold)

    def _held(self, i: int) -> torch.Tensor:
        """去抖版"夹着": 最近 contact_hold_steps 步内出现过 both_contact。

        治瞬时接触力逐帧闪烁导致抬升奖励被按帧掐断。焊接阶段杆无碰撞 → 接触=0 →
        _held=False → 抬升/carry 不触发, 飞行目标钉死 hover(只学协同飞行)。
        weld_phase>=2(真抓取)走真实接触。
        """
        return self._since_both[:, i] <= _P.contact_hold_steps

    def _grasp_center(self, i: int) -> torch.Tensor:
        rb = self._robots[i]
        wp = rb.data.body_pos_w[:, self._ee_ids[i][0], :]
        wq = rb.data.body_quat_w[:, self._ee_ids[i][0], :]
        return wp + quat_apply(wq, self._grasp_offset_t.unsqueeze(0).expand(self.num_envs, 3))

    def _bar_end_w(self, i: int) -> torch.Tensor:
        e = torch.zeros(self.num_envs, 3, device=self.device)
        e[:, 0] = self._end_sign[i] * _P.bar_grasp_x
        return self._bar.data.root_pos_w + quat_apply(self._bar.data.root_quat_w, e)

    def _grasp_target_w(self, i: int) -> torch.Tensor:
        t = self._bar_end_w(i)
        t[:, 2] = t[:, 2] + _P.grasp_z_offset
        return t

    def _bar_axis_w(self) -> torch.Tensor:
        ex = torch.zeros(self.num_envs, 3, device=self.device); ex[:, 0] = 1.0
        return quat_apply(self._bar.data.root_quat_w, ex)

    def _at_end_strict(self, i: int) -> torch.Tensor:
        """机 i 指尖是否真在自己那端的抓取位(水平 < at_end_horiz 且 垂直 < at_end_vert)。

        垂直约束防把杆上方的空气当作"已到位"刷取闭合奖励。
        """
        d = self._grasp_center(i) - self._grasp_target_w(i)
        return (torch.linalg.norm(d[:, :2], dim=1) < _P.at_end_horiz) & (d[:, 2].abs() < _P.at_end_vert)

    # ==============================================================
    # 动作(dict→拼接) + LADRC + 重心前馈 + 臂课程 + 搬运态
    # ==============================================================
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        self._prev_actions = self._actions.clone()
        a_cat = torch.cat([actions["drone_0"], actions["drone_1"]], dim=-1)
        self._actions = a_cat.clone().clamp(-1.0, 1.0)
        self._steps_since_reset += 1
        N = self.num_envs
        a0, a1 = self._actions[:, 0:9], self._actions[:, 9:18]

        quat = torch.stack([rb.data.root_quat_w for rb in self._robots], dim=1)
        omega = torch.stack([rb.data.root_ang_vel_b for rb in self._robots], dim=1)
        grav_b = torch.stack([rb.data.projected_gravity_b for rb in self._robots], dim=1)
        g = self._gravity_magnitude
        a_des = torch.stack([a0[:, 0:3], a1[:, 0:3]], dim=1) * _P.accel_scale
        yaw_rate = torch.stack([a0[:, 3], a1[:, 3]], dim=1) * _P.yaw_rate_scale

        # RL 出期望加速度/偏航率 → 串级 LADRC(外环角度 + 内环角速率)算推力与力矩。
        self._thrust[:, :, 0, 2] = self.ladrc.accel_to_thrust(
            a_des, quat, self._robot_mass, g, _P.thrust_to_weight * self._robot_weight)
        roll_des, pitch_des = self.ladrc.accel_to_attitude_setpoint(a_des, quat, g, _P.ladrc_max_angle)
        pitch_meas = grav_b[:, :, 0:1]
        roll_meas = -grav_b[:, :, 1:2]
        h = self.cfg.sim.dt * self.cfg.decimation
        p_sp, q_sp = self.ladrc.outer_step(roll_des, pitch_des, roll_meas, pitch_meas, h)
        moment = self.ladrc.inner_step(p_sp, q_sp, yaw_rate.unsqueeze(-1), omega, h)
        moment = moment * _P.ladrc_moment_sign

        # 重心前馈: 挂臂后合重心水平偏移产生翻转力矩, 在 LADRC 力矩上提前减掉。
        if _P.use_cog_feedforward and _P.cog_ff_gain > 0.0:
            com_off = []
            for k, rb in enumerate(self._robots):
                body_pos_w = rb.data.body_pos_w
                com_w = (body_pos_w * self._body_masses.view(1, -1, 1)).sum(dim=1) / self._body_masses.sum()
                com_off.append(quat_apply_inverse(rb.data.root_quat_w, com_w - rb.data.root_pos_w))
            com_off_b = torch.stack(com_off, dim=1)
            ff = self.ladrc.cog_feedforward(com_off_b, self._thrust[:, :, 0, 2:3], _P.cog_ff_gain)
            moment = moment + ff

        # 持杆负载前馈(门控在去抖版"夹着"上): 半杆重挂夹爪产生翻转力矩, 前馈抵消。
        # 仅 weld_phase>=2(真抓取)启用: 焊接阶段杆重由焊接关节传给机体, 策略自学补偿,
        # 此处再加前馈会偏离纯飞行控制模型。
        if _P.use_payload_feedforward and _P.payload_mass_per_drone > 0.0 and _P.weld_phase >= 2:
            hold = torch.stack([self._held(0).float(), self._held(1).float()], dim=1).unsqueeze(-1)  # (N,2,1)
            r_grip_b = []
            for k, rb in enumerate(self._robots):
                gc_w = self._grasp_center(k)
                com_w = (rb.data.body_pos_w * self._body_masses.view(1, -1, 1)).sum(dim=1) / self._body_masses.sum()
                r_grip_b.append(quat_apply_inverse(rb.data.root_quat_w, gc_w - com_w))
            r_grip_b = torch.stack(r_grip_b, dim=1)  # (N,2,3)
            moment = moment + self.ladrc.payload_torque_feedforward(
                _P.payload_mass_per_drone, r_grip_b, grav_b, g, hold, _P.payload_ff_gain)
            self._thrust[:, :, 0, 2] += hold[:, :, 0] * _P.payload_mass_per_drone * g   # 推力补半杆重(开环)

        self._moment[:, :, 0, :] = moment
        # 让内环 ESO 看到含前馈的实际力矩, 避免重复补偿。
        self.ladrc.commit_inner_applied(moment)

        dt = self.cfg.sim.dt * self.cfg.decimation
        arm_delta = torch.stack([a0[:, 4:8], a1[:, 4:8]], dim=1) * _P.arm_speed_scale * dt
        self._arm_dof_targets = torch.clamp(self._arm_dof_targets + arm_delta, self._arm_lower, self._arm_upper)

        # 三阶段臂课程: 先蜷缩稳悬停(settle) → 强制插值到直臂(straighten) → 自由抓取。
        steps = self._steps_since_reset
        settle, straighten = _P.settle_steps, _P.straighten_steps
        stow = self._arm_stow_t.view(1, 1, 4).expand(N, 2, 4)
        straight_pose = self._arm_straight_pose_t.view(1, 1, 4).expand(N, 2, 4)
        settling = (steps < settle).view(N, 1, 1)
        self._arm_dof_targets = torch.where(settling, stow, self._arm_dof_targets)
        if straighten > 0:
            in_str = ((steps >= settle) & (steps < settle + straighten)).view(N, 1, 1)
            frac = ((steps - settle).clamp(min=0).float() / float(straighten)).clamp(0.0, 1.0).view(N, 1, 1)
            ramp = (1.0 - frac) * stow + frac * straight_pose
            self._arm_dof_targets = torch.where(in_str, ramp, self._arm_dof_targets)
        self._grasp_phase = steps >= (settle + straighten)

        # 搬运触发 _carry: 两机都夹住(去抖版) 且 杆升过触发高度, 带滞回(进 carry_trigger /
        # 退 carry_release)防在阈值上抖动 → 目标点高频跳变把两机当鞭子甩。
        bar_z = self._bar.data.root_pos_w[:, 2]
        all_held = self._held(0) & self._held(1)
        rise = bar_z - self._bar_initial_pos[:, 2]
        rise_gate = torch.where(self._carry, rise > _P.carry_release, rise > _P.carry_trigger)
        self._carry = all_held & rise_gate
        # carry 后飞行目标 z 的 ease-out 进度: carry 持续 → 逐步逼近 1; carry 断开 → 清零。
        if _P.carry_ramp_steps > 0:
            inc = 1.0 / float(_P.carry_ramp_steps)
            self._carry_ramp = torch.where(
                self._carry, (self._carry_ramp + inc).clamp(max=1.0), torch.zeros_like(self._carry_ramp))
        # carry 时锁臂到直臂 pose 保证搬运全程竖直。
        carry_arm = self._carry.view(N, 1, 1).expand(N, 2, 4)
        self._arm_dof_targets = torch.where(carry_arm, straight_pose, self._arm_dof_targets)

        # ── 焊接阶段(weld_phase<2): 锁臂到直臂 pose, 忽略策略臂动作 + 三阶段课程 + carry。
        #   FIXED 焊接把两机+杆锁成一个刚体, 若臂仍乱动会通过杆互相较劲 → 甩飞/翻。锁臂后
        #   每机 9 维只有前 4 维[加速度3+偏航率1]生效 = 等价纯飞行学习。
        #   用 [:] 原地写(不重绑/不 expand): expand 会与常量张量共享存储, reset 的原地索引写
        #   会反向篡改常量。
        if _P.weld_phase < 2:
            self._arm_dof_targets[:] = self._arm_straight_pose_t.view(1, 1, 4)

    def _apply_action(self):
        for i, rb in enumerate(self._robots):
            rb.set_external_force_and_torque(self._thrust[:, i], self._moment[:, i], body_ids=self._body_ids[i])
            rb.set_joint_position_target(self._arm_dof_targets[:, i], joint_ids=self._arm_ids[i])
            # 焊接阶段夹爪张开, 不夹焊死的杆: 焊接关节已握住杆, 再锁闭会形成焊接+夹爪挤压
            # 双约束 → 巨大内力 → 抬升时约束爆炸。抬升奖励用 _held(去抖接触)不依赖真接触,
            # 张爪不影响学习。weld_phase>=2 才听策略合爪命令。
            if _P.weld_phase < 2:
                gtarget = self._gripper_open_t.unsqueeze(0).expand(self.num_envs, 2)
            else:
                gcmd = self._actions[:, 8] if i == 0 else self._actions[:, 17]
                sel_close = (gcmd >= 0.0).float().unsqueeze(-1)
                sel_close = torch.maximum(sel_close, self._carry.float().unsqueeze(-1))   # 搬运态强制闭爪
                gtarget = sel_close * self._gripper_close_t + (1.0 - sel_close) * self._gripper_open_t
            rb.set_joint_position_target(gtarget, joint_ids=self._grip_ids[i])
            # 旋翼转速仅视觉, 跟随推力大小。
            tn = (self._thrust[:, i, 0, 2] / (_P.thrust_to_weight * self._robot_weight)).clamp(0.0, 1.0)
            spd = 50.0
            rs = torch.zeros(self.num_envs, 4, device=self.device)
            rs[:, 0] = spd * (0.5 + tn); rs[:, 1] = -spd * (0.5 + tn)
            rs[:, 2] = -spd * (0.5 + tn); rs[:, 3] = spd * (0.5 + tn)
            rb.set_joint_velocity_target(rs, joint_ids=self._rotor_ids[i])

    # ==============================================================
    # 观测: 每机 47(自己 32 + 通信块 15); state75 给中心化 critic
    # ==============================================================
    def _own_block(self, i: int) -> torch.Tensor:
        rb = self._robots[i]
        drone_pos_w = rb.data.root_pos_w
        drone_quat_w = rb.data.root_quat_w
        end_w = self._bar_end_w(i)

        des = end_w.clone()
        # z 钉在初始杆高+hover(非活动端实时 z): 拆掉"拉高自己那端→目标跟着升→越拉越高"的
        # 跷跷板正反馈。xy 跟活动端; 升空牵引交给 carry 后的 goal_end。
        des[:, 2] = self._bar_initial_pos[:, 2] + _P.hover_offset
        goal_end = self._bar_initial_pos.clone()
        goal_end[:, 0] = goal_end[:, 0] + self._end_sign[i] * _P.bar_grasp_x
        goal_end[:, 2] = goal_end[:, 2] + _P.carry_height + _P.hover_offset
        des = torch.where(self._carry.unsqueeze(-1), goal_end, des)
        # carry 后 z 目标从 hover 平滑(ease-out)升到 hover+carry_height, 避免瞬间 +carry_height
        # 跳变使两机猛窜甩飞杆。非 carry 时 _carry_ramp=0 → z=hover。
        if _P.carry_ramp_steps > 0:
            eased = self._carry_ramp * (2.0 - self._carry_ramp)   # ease-out, 0→1
            des[:, 2] = self._bar_initial_pos[:, 2] + _P.hover_offset + _P.carry_height * eased
        self._desired_pos_w[:, i] = des

        des_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, des)
        end_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, end_w)
        arm_pos = rb.data.joint_pos[:, self._arm_ids[i]]
        arm_n = 2.0 * (arm_pos - self._arm_lower) / (self._arm_upper - self._arm_lower + 1e-6) - 1.0
        arm_vel = rb.data.joint_vel[:, self._arm_ids[i]] * 0.1
        grasp_center_w = self._grasp_center(i)
        to_t = self._grasp_target_w(i) - grasp_center_w
        to_t_d = torch.linalg.norm(to_t, dim=1, keepdim=True)
        to_t_dir_w = to_t / (to_t_d + 1e-6)
        ee_w = rb.data.body_pos_w[:, self._ee_ids[i][0], :]
        ee_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, ee_w)
        dir_end_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, ee_w + to_t_dir_w)
        to_t_dir_b = dir_end_b - ee_b
        # 杆轴朝中心定向(乘 -end_sign): 杆轴在 world+x, 经各机机体系后 drone_0 得 body+x、
        # drone_1(yaw180) 得 body-x → 共享策略对两机输入相反。乘 -end_sign 让两机都得 body+x,
        # 实现真 egocentric 对称(reward 用 |cos| 与符号无关, 不受影响)。
        bar_axis_ego = -self._end_sign[i] * self._bar_axis_w()
        axis_end_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, drone_pos_w + bar_axis_ego)

        base = torch.cat([
            rb.data.root_lin_vel_b, rb.data.root_ang_vel_b, rb.data.projected_gravity_b,
            des_b, arm_n, arm_vel, end_b, to_t_dir_b, to_t_d, axis_end_b,
        ], dim=-1)
        base = torch.clamp(base, -5.0, 5.0)
        fl = torch.tanh(0.1 * self._finger_mag(self._contacts_l[i]))
        fr = torch.tanh(0.1 * self._finger_mag(self._contacts_r[i]))
        return torch.cat([base, torch.stack([fl, fr], dim=-1)], dim=-1)   # 32

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # 世界系共享块(只给中心化 critic 的 state 用, 缓存; 执行时不需要)
        ez = torch.zeros(self.num_envs, 3, device=self.device); ez[:, 2] = 1.0
        bar_zaxis_w = quat_apply(self._bar.data.root_quat_w, ez)
        rel_pos = self._robots[1].data.root_pos_w - self._robots[0].data.root_pos_w
        contacts = torch.stack([self._both_contact(0).float(), self._both_contact(1).float()], dim=-1)
        self._shared_obs = torch.clamp(torch.cat([
            self._bar.data.root_lin_vel_w * 0.5, bar_zaxis_w, rel_pos, contacts,
        ], dim=-1), -5.0, 5.0)
        self._own_obs[0] = self._own_block(0)
        self._own_obs[1] = self._own_block(1)

        # 分布式观测: 自己 32 + 杆状态转机体系 6 + 通信块(队友广播)9 = 47。
        # 全部 egocentric → 两机视角对称(配合 yaw180 镜像); 真机上通信块=每周期广播 9 个浮点。
        obs = {}
        for i, agent in enumerate(("drone_0", "drone_1")):
            j = 1 - i
            rb_i, rb_j = self._robots[i], self._robots[j]
            pos_i, quat_i = rb_i.data.root_pos_w, rb_i.data.root_quat_w
            # 杆状态(自己传感器可测) → 自己机体系
            bar_v_b = quat_apply_inverse(quat_i, self._bar.data.root_lin_vel_w) * 0.5
            bar_z_b = quat_apply_inverse(quat_i, bar_zaxis_w)
            # 通信块: 队友相对位置/相对速度(转机体系) + 是否夹住 + 是否正在合爪 + 距其目标多远
            mate_pos_b, _ = subtract_frame_transforms(pos_i, quat_i, rb_j.data.root_pos_w)
            mate_v_b = quat_apply_inverse(quat_i, rb_j.data.root_lin_vel_w - rb_i.data.root_lin_vel_w) * 0.5
            mate_contact = self._both_contact(j).float().unsqueeze(-1)
            mate_closing = (self._actions[:, 8 if j == 0 else 17] >= 0.0).float().unsqueeze(-1)
            mate_ee = torch.linalg.norm(self._grasp_center(j) - self._grasp_target_w(j), dim=1, keepdim=True)
            block = torch.cat([bar_v_b, bar_z_b, mate_pos_b, mate_v_b, mate_contact, mate_closing, mate_ee], dim=-1)  # 15
            obs[agent] = torch.cat([self._own_obs[i], torch.clamp(block, -5.0, 5.0)], dim=-1)   # 32+15=47
        return obs

    def _get_states(self) -> torch.Tensor:
        # 中心化 critic 的全局 state(= 集中式 obs 布局): 两机 own 块 + 共享块
        return torch.cat([self._own_obs[0], self._own_obs[1], self._shared_obs], dim=-1)   # 75

    # ==============================================================
    # 奖励: 个人栈归各自(MAPPO 分账) + 共享项两机同发
    # ==============================================================
    def _machine_reward(self, i: int):
        cfg = _P
        dt = self.step_dt
        rb = self._robots[i]
        drone_pos_w = rb.data.root_pos_w
        drone_h = drone_pos_w[:, 2]
        gclosed = ((self._actions[:, 8] if i == 0 else self._actions[:, 17]) >= 0.0).float()
        a_slice = slice(0, 9) if i == 0 else slice(9, 18)
        aids = self._arm_ids[i]

        lf = rb.data.body_pos_w[:, self._lf_ids[i][0], :]
        rf = rb.data.body_pos_w[:, self._rf_ids[i][0], :]
        finger_mid = (lf + rf) / 2.0
        wrist_pos = rb.data.body_pos_w[:, self._ee_ids[i][0], :]
        grasp_center = self._grasp_center(i)

        dist_to_goal = torch.linalg.norm(self._desired_pos_w[:, i] - drone_pos_w, dim=1)
        dist_reward = (1.0 / (1.0 + dist_to_goal ** 2)) ** 2
        ori_err = torch.sum(torch.square(rb.data.projected_gravity_b[:, :2]), dim=1)
        ee_d = torch.linalg.norm(grasp_center - self._grasp_target_w(i), dim=1)
        reaching = 1.0 - torch.tanh(ee_d / 0.2)
        is_far = (ee_d > 0.08).float(); is_near = (ee_d < 0.05).float()
        finger = -is_far * gclosed + is_near * gclosed   # 远处闭爪扣分、近处闭爪给分
        v = finger_mid - wrist_pos
        gvert = torch.clamp(-v[:, 2] / (torch.linalg.norm(v, dim=1) + 1e-6), min=0.0)
        arm_bc = rb.data.joint_pos[:, [aids[1], aids[2]]]
        arm_straight = 1.0 - torch.tanh(torch.linalg.norm(arm_bc - self._arm_straight_target_t, dim=1) / 1.0)
        # 姿态门控: 抓取奖励乘上"臂伸直 × 夹爪竖直", 强制先摆好姿态再解锁抓取分。
        straight_gate = ((arm_straight - 0.4) / 0.3).clamp(0.0, 1.0)
        vertical_gate = ((gvert - 0.5) / 0.3).clamp(0.0, 1.0)
        grasp_gate = straight_gate * vertical_gate
        phase3 = self._grasp_phase.float()

        # 对齐门控: 夹爪开合轴必须 ⊥ 杆轴(横跨截面)才解锁抓取分。
        finger_axis = rf - lf
        finger_axis = finger_axis / (torch.linalg.norm(finger_axis, dim=1, keepdim=True) + 1e-6)
        align = 1.0 - torch.abs(torch.sum(finger_axis * self._bar_axis_w(), dim=1))
        align_r = align * cfg.align_reward_scale * dt * phase3
        align_gate = ((align - 0.5) / 0.3).clamp(0.0, 1.0)

        fl_f = self._finger_mag(self._contacts_l[i])
        fr_f = self._finger_mag(self._contacts_r[i])
        self._max_finger_force[:, i] = torch.maximum(self._max_finger_force[:, i], fl_f + fr_f)
        self._max_weak_finger[:, i] = torch.maximum(self._max_weak_finger[:, i], torch.minimum(fl_f, fr_f))
        # 非对称双指接触判定, 与 _both_contact 一致
        both_contact = ((torch.maximum(fl_f, fr_f) > cfg.contact_force_threshold)
                        & (torch.minimum(fl_f, fr_f) > cfg.contact_weak_threshold)).float()
        contact_r = both_contact * cfg.contact_grasp_reward_scale * dt * grasp_gate * align_gate
        end_rise = (self._bar_end_w(i)[:, 2] - self._bar_initial_pos[:, 2]).clamp(min=0.0, max=cfg.own_end_lift_cap)
        # own_end_lift 门控在【两机都夹住】上: 只 _held(i) 时, 先抓那台独自抬自己那端会把杆
        # 翘斜, 害队友夹不住。改成两机都 held 才奖励抬 → 先抓那台 grab 后托平等队友, 双双夹住
        # 才一起抬(抓取本身仍由 contact_grasp/touch/grasp_close 这些 per-drone 项激励)。
        both_held = self._held(i).float() * self._held(1 - i).float()
        own_lift_r = end_rise * both_held * cfg.own_end_lift_reward_scale * dt
        progress = (self._prev_ee_dist[:, i] - ee_d).clamp(min=0.0, max=0.02)
        progress_r = progress * cfg.progress_reward_scale * phase3 * straight_gate
        self._prev_ee_dist[:, i] = ee_d.detach()
        engagement_r = (ee_d < 0.03).float() * cfg.engagement_reward_scale * dt * phase3 * straight_gate
        at_bar = self._at_end_strict(i).float()   # 严格到位判定, 杀"悬空合爪刷分"
        grasp_close_r = at_bar * gclosed * cfg.grasp_close_reward_scale * dt * grasp_gate * align_gate
        # touch 面包屑: 任一指碰到杆就给(对正 + 伸直门控), 补"空气→双指接触"之间的台阶
        any_touch = (torch.maximum(fl_f, fr_f) > cfg.contact_force_threshold).float()
        touch_r = any_touch * cfg.touch_reward_scale * dt * align_gate * straight_gate
        # 侧向偏置探针: 接触时刻夹持中心相对杆轴线的带符号横向(水平 ⊥ 杆轴)偏移。
        axis_w = self._bar_axis_w()
        lat_dir = torch.stack([-axis_w[:, 1], axis_w[:, 0], torch.zeros_like(axis_w[:, 0])], dim=1)
        lat_dir = lat_dir / (torch.linalg.norm(lat_dir, dim=1, keepdim=True) + 1e-6)
        lat_off = torch.sum((grasp_center - self._bar_end_w(i)) * lat_dir, dim=1)
        self._lat_bias_sum[:, i] += lat_off * any_touch
        self._lat_cnt[:, i] += any_touch

        lin_v = torch.sum(torch.square(rb.data.root_lin_vel_b), dim=1).clamp(max=100.0)
        ang_v = torch.sum(torch.square(rb.data.root_ang_vel_b), dim=1).clamp(max=100.0)
        action_rate = torch.sum(torch.square(self._actions[:, a_slice] - self._prev_actions[:, a_slice]), dim=1)
        jv = torch.sum(torch.square(rb.data.joint_vel[:, aids]), dim=1).clamp(max=100.0)
        crash_int = torch.clamp(torch.clamp(0.20 - drone_h, min=0.0) / 0.10
                                + torch.clamp(drone_h - 4.0, min=0.0) / 0.5, max=1.0)

        comp = {
            "dist_to_goal": dist_reward * cfg.dist_reward_scale * dt,
            "orientation": ori_err * cfg.orientation_penalty_scale * dt,
            "reaching": reaching * cfg.reaching_reward_scale * dt * phase3,
            "finger": finger * cfg.finger_reward_scale * dt * phase3,
            "gripper_vertical": gvert * cfg.gripper_vertical_reward_scale * dt,
            "arm_straight": arm_straight * cfg.arm_straight_reward_scale * dt,
            "align": align_r,
            "grasp_close": grasp_close_r,
            "contact_grasp": contact_r,
            "own_end_lift": own_lift_r,
            "touch": touch_r,
            "engagement": engagement_r,
            "progress": progress_r,
            "lin_vel": lin_v * cfg.lin_vel_penalty_scale * dt,
            "ang_vel": ang_v * cfg.ang_vel_penalty_scale * dt,
            "action_rate": action_rate * cfg.action_rate_penalty_scale * dt,
            "joint_vel": jv * cfg.joint_vel_penalty_scale * dt,
            "crash": crash_int * cfg.crash_penalty * dt,
        }
        total = torch.sum(torch.stack(list(comp.values())), dim=0)
        return total, comp

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = _P
        dt = self.step_dt
        self._global_step += 1   # 全局步进度(驱动 hold 课程收紧)
        # 去抖计数更新(每步一次, 必须在 _machine_reward / 共享块之前): 碰到→清零, 没碰→+1
        for i in range(2):
            raw_i = self._both_contact(i)
            self._since_both[:, i] = torch.where(
                raw_i, torch.zeros_like(self._since_both[:, i]), self._since_both[:, i] + 1)
        t0, c0 = self._machine_reward(0)
        t1, c1 = self._machine_reward(1)
        for k, v in c0.items():
            self._episode_sums[k + "_0"] += v
        for k, v in c1.items():
            self._episode_sums[k + "_1"] += v

        bar_pos = self._bar.data.root_pos_w
        bar_z = bar_pos[:, 2]
        # clamp(max=2.0) 防发散 env 的杆瞬间弹到极高拉爆均值; 任务真实最高 ≪ 2.0。仅影响日志。
        self._max_bar_height = torch.maximum(self._max_bar_height, bar_z.clamp(max=2.0))
        # 门控统一用去抖版(瞬时版只作诊断计数), 治接触力闪烁把抬升链按帧掐断。
        all_raw = (self._since_both[:, 0] == 0) & (self._since_both[:, 1] == 0)
        all_contact = (self._held(0) & self._held(1)).float()
        self._contact_latch[:, 0] |= (self._since_both[:, 0] == 0)
        self._contact_latch[:, 1] |= (self._since_both[:, 1] == 0)
        self._allc_cnt += all_raw.float()
        self._carry_cnt += self._carry.float()

        bar_rise = (bar_z - self._bar_initial_pos[:, 2]).clamp(min=0.0, max=cfg.lift_height)
        grasp_lift_r = bar_rise * all_contact * cfg.grasp_lift_reward_scale * dt
        lift_tz = self._bar_initial_pos[:, 2] + cfg.carry_height
        hgap = (lift_tz - bar_z).clamp(min=0.0)
        lift_goal_r = all_contact * (1.0 - torch.tanh(hgap / cfg.lift_goal_std)) * cfg.lift_goal_reward_scale * dt
        lifted = ((bar_z > self._bar_initial_pos[:, 2] + cfg.lift_height) * all_contact)
        lifting_r = lifted * cfg.lifting_reward_scale * dt

        # lift_gate: 杆离台进度(0=躺着 → 1=抬过 lift_height)。乘到下面的搬运近目标奖励
        # (bar_goal / hold_bonus)上, 使策略不会因夹住一个静止物体而拿到搬运奖励。bar_rise
        # 已 clamp[0, lift_height] → /lift_height ∈ [0,1]。flag 关时退化为标量 1.0。
        lift_gate = (bar_rise / cfg.lift_height) if cfg.lift_gate_transport_rewards else torch.ones_like(bar_rise)

        bar_goal = self._bar_initial_pos.clone()
        bar_goal[:, 2] = bar_goal[:, 2] + cfg.carry_height
        bar_goal_dist = torch.linalg.norm(bar_goal - bar_pos, dim=1)
        bar_goal_r = torch.exp(-cfg.bar_goal_alpha * bar_goal_dist) * cfg.bar_goal_reward_scale * dt * all_contact * lift_gate
        bar_tilt = torch.sum(torch.square(self._bar.data.root_quat_w[:, 1:3]), dim=1)
        bar_level_r = torch.exp(-bar_tilt / 0.05) * cfg.bar_level_reward_scale * dt * all_contact
        # 抓取阶段(grasp_phase 且未 all_contact)就奖励杆水平, 打击"先抓那台翘杆害队友"。
        pre_grasp_gate = self._grasp_phase.float() * (1.0 - all_contact)
        bar_level_pre_r = torch.exp(-bar_tilt / 0.05) * cfg.bar_level_pre_grasp_scale * dt * pre_grasp_gate
        # 严格成功判定(success 指标用): 杆到目标 < success_dist 且 水平 < success_tilt 且 两机同时夹住。
        in_goal_strict = (bar_goal_dist < cfg.success_dist) & (bar_tilt < cfg.success_tilt) & (all_contact > 0.5)
        # hold_bonus 奖励用课程放宽判定: 阈值从宽线性收紧到严格, 早期易拿分 → 学会"抬到目标附近保持"。
        if cfg.use_hold_curriculum:
            f = min(1.0, self._global_step / max(1.0, float(cfg.hold_curric_steps)))   # 0→1 over hold_curric_steps
            sdist = cfg.hold_loose_dist + f * (cfg.success_dist - cfg.hold_loose_dist)
            stilt = cfg.hold_loose_tilt + f * (cfg.success_tilt - cfg.hold_loose_tilt)
            in_goal_rew = (bar_goal_dist < sdist) & (bar_tilt < stilt) & (all_contact > 0.5)
        else:
            in_goal_rew = in_goal_strict
        hold_r = in_goal_rew.float() * cfg.hold_bonus_scale * dt * lift_gate   # 杆离台才发, 不为静止持物付费
        self._in_goal_steps = (self._in_goal_steps + 1) * in_goal_strict.long()   # 严格: success 指标
        self._success = self._success | (self._in_goal_steps >= cfg.success_hold_steps)

        # internal: 沿杆轴的两机相对速度 → 罚互相拉扯(内力代理), 防两机撕扯。
        v_rel = self._robots[0].data.root_lin_vel_w - self._robots[1].data.root_lin_vel_w
        v_k = torch.abs(torch.sum(v_rel * self._bar_axis_w(), dim=1)).clamp(max=10.0)
        internal_r = v_k * cfg.internal_penalty_scale * dt * all_contact

        # bar_still: 都没夹齐前别拖杆(防移动靶), all_contact 后不罚。
        bar_speed = torch.linalg.norm(self._bar.data.root_lin_vel_w, dim=1) \
            + 0.5 * torch.linalg.norm(self._bar.data.root_ang_vel_w, dim=1)
        bar_still_r = bar_speed.clamp(max=5.0) * cfg.bar_still_penalty_scale * dt * (1.0 - all_contact)

        # ── 协作三级阶梯(都到位 → 一起合爪 → 都夹稳; 两机同发, 专攻"同时性"这一稀有联合事件) ──
        both_ready = (self._at_end_strict(0) & self._at_end_strict(1)).float()
        both_closing = ((self._actions[:, 8] >= 0.0) & (self._actions[:, 17] >= 0.0)).float()
        coop_ready_r = both_ready * cfg.coop_ready_reward_scale * dt
        coop_close_r = both_ready * both_closing * cfg.coop_close_reward_scale * dt
        coop_grasp_r = all_contact * cfg.coop_grasp_reward_scale * dt
        # 弱链下探: min(两机 reaching) → 奖励"把落后那台也拽到杆端"。
        reach0 = 1.0 - torch.tanh(torch.linalg.norm(self._grasp_center(0) - self._grasp_target_w(0), dim=1) / 0.2)
        reach1 = 1.0 - torch.tanh(torch.linalg.norm(self._grasp_center(1) - self._grasp_target_w(1), dim=1) / 0.2)
        coop_descend_r = torch.minimum(reach0, reach1) * self._grasp_phase.float() * cfg.coop_descend_scale * dt

        shared = {
            "grasp_lift": grasp_lift_r, "lift_goal": lift_goal_r, "lifting": lifting_r,
            "bar_goal": bar_goal_r, "bar_level": bar_level_r, "bar_level_pre": bar_level_pre_r, "hold_bonus": hold_r,
            "internal": internal_r, "bar_still": bar_still_r,
            "coop_ready": coop_ready_r, "coop_close": coop_close_r, "coop_grasp": coop_grasp_r,
            "coop_descend": coop_descend_r,
        }
        for k, v in shared.items():
            self._episode_sums[k] += v
        shared_total = torch.sum(torch.stack(list(shared.values())), dim=0)
        # MAPPO 分账: 个人栈归各自 + 协作项两机同发
        return {"drone_0": t0 + shared_total, "drone_1": t1 + shared_total}

    # ==============================================================
    # 终止(两 agent 同生共死: 任一机坠 / 杆掉台 → 整局结束)
    # ==============================================================
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, rb in enumerate(self._robots):
            z = rb.data.root_pos_w[:, 2]
            died = died | (z < 0.12) | (z > 5.0)
            died = died | (torch.linalg.norm(rb.data.root_lin_vel_w, dim=1) > 30.0)
            died = died | (torch.linalg.norm(self._desired_pos_w[:, i] - rb.data.root_pos_w, dim=1) > 5.0)
        # 焊接阶段杆被 weld 拽住掉不下去, "杆掉台"只在真抓取(weld_phase>=2)时检查。
        if _P.weld_phase >= 2:
            died = died | (self._bar.data.root_pos_w[:, 2] < (_P.pedestal_height - 0.10))
        # 杆线速度 > 15m/s = 约束爆炸前兆(正常抬升 ~1-3m/s) → 尽早 reset 发散 env。
        died = died | (torch.linalg.norm(self._bar.data.root_lin_vel_w, dim=1) > 15.0)
        self._died_buf = died
        terminated = {agent: died for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    # ==============================================================
    # 重置(含抓 +x 端的机偏航 180° 镜像)
    # ==============================================================
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot_0._ALL_INDICES

        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        # Metrics 保持单元素张量, 不要 .item(): skrl trainer 只把 torch.Tensor 型 info 写进
        # TensorBoard, float 会被静默丢弃。
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self._died_buf[env_ids]).float()
        extras["Metrics/max_bar_height"] = self._max_bar_height[env_ids].mean()
        extras["Metrics/bar_height"] = self._bar.data.root_pos_w[env_ids, 2].mean()
        extras["Metrics/success_rate"] = self._success[env_ids].float().mean()
        extras["Metrics/contact_rate_0"] = self._contact_latch[env_ids, 0].float().mean()
        extras["Metrics/contact_rate_1"] = self._contact_latch[env_ids, 1].float().mean()
        extras["Metrics/max_finger_force_0"] = self._max_finger_force[env_ids, 0].mean()
        extras["Metrics/max_finger_force_1"] = self._max_finger_force[env_ids, 1].mean()
        extras["Metrics/max_weak_finger_0"] = self._max_weak_finger[env_ids, 0].mean()
        extras["Metrics/max_weak_finger_1"] = self._max_weak_finger[env_ids, 1].mean()
        for i in range(2):
            gc = self._grasp_center(i)[env_ids]
            tgt = self._grasp_target_w(i)[env_ids]
            extras[f"Metrics/final_ee_to_end_{i}"] = torch.linalg.norm(gc - tgt, dim=1).mean()
            # 侧向偏置读数(米, 带符号): 稳定非零 = grasp_offset.y 系统性偏差。
            extras[f"Metrics/grip_lat_bias_{i}"] = self._lat_bias_sum[env_ids, i].sum() \
                / (self._lat_cnt[env_ids, i].sum() + 1e-6)
        # carry / 瞬时 all_contact 本局占比: all_contact_frac 高而 carry_frac 低 = 高度门没过;
        # 双低 = 四指接触没凑齐过; carry_frac 起来 = 搬运态稳定点亮。
        steps_f = self._steps_since_reset[env_ids].clamp(min=1).float()
        extras["Metrics/carry_frac"] = (self._carry_cnt[env_ids] / steps_f).mean()
        extras["Metrics/all_contact_frac"] = (self._allc_cnt[env_ids] / steps_f).mean()
        self.extras["log"].update(extras)

        self._robot_0.reset(env_ids)
        self._robot_1.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._carry[env_ids] = False
        self._carry_ramp[env_ids] = 0.0
        self._success[env_ids] = False
        self._in_goal_steps[env_ids] = 0
        self._contact_latch[env_ids] = False
        self._max_finger_force[env_ids] = 0.0
        self._max_weak_finger[env_ids] = 0.0
        self._lat_bias_sum[env_ids] = 0.0
        self._lat_cnt[env_ids] = 0.0
        self._since_both[env_ids] = 999
        self._carry_cnt[env_ids] = 0.0
        self._allc_cnt[env_ids] = 0.0
        self._prev_ee_dist[env_ids] = 0.0
        self.ladrc.reset(env_ids)

        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]

        # 焊接阶段把杆 spawn 在台座上方的空中(清空气): 杆同时被 weld 拽住 + 台座顶住会形成
        # 冗余约束打架, 故焊接阶段杆抬到台顶上方 → 纯空中焊死飞行。真抓取(weld_phase>=2)杆回坐台上。
        bar_base_z = 0.55 if _P.weld_phase < 2 else 0.22

        bar_state = self._bar.data.default_root_state[env_ids].clone()
        bar_state[:, 0] = origins[:, 0] + torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        bar_state[:, 1] = origins[:, 1] + torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        bar_state[:, 2] = origins[:, 2] + bar_base_z
        bar_state[:, 3] = 1.0; bar_state[:, 4:7] = 0.0; bar_state[:, 7:] = 0.0
        self._bar.write_root_pose_to_sim(bar_state[:, :7], env_ids)
        self._bar.write_root_velocity_to_sim(bar_state[:, 7:], env_ids)
        self._bar_initial_pos[env_ids, 0] = origins[:, 0]
        self._bar_initial_pos[env_ids, 1] = origins[:, 1]
        self._bar_initial_pos[env_ids, 2] = bar_state[:, 2]
        self._max_bar_height[env_ids] = bar_state[:, 2]

        for i, ped in enumerate([self._pedestal_0, self._pedestal_1]):
            ps = ped.data.default_root_state[env_ids].clone()
            ps[:, 0] = origins[:, 0] + self._end_sign[i] * _P.bar_grasp_x
            ps[:, 1] = origins[:, 1]
            ps[:, 2] = origins[:, 2] + _P.pedestal_height / 2.0
            ps[:, 3] = 1.0; ps[:, 4:7] = 0.0; ps[:, 7:] = 0.0
            ped.write_root_pose_to_sim(ps[:, :7], env_ids)
            ped.write_root_velocity_to_sim(ps[:, 7:], env_ids)

        for i, rb in enumerate(self._robots):
            rs = rb.data.default_root_state[env_ids].clone()
            rs[:, 0] = origins[:, 0] + self._end_sign[i] * _P.bar_grasp_x
            rs[:, 1] = origins[:, 1]
            rs[:, 2] = origins[:, 2] + bar_base_z + _P.hover_offset \
                + torch.zeros(n, device=self.device).uniform_(-0.03, 0.03)
            # 抓 +x 端的机偏航 180°(镜像对称化, 使两机 egocentric 任务全同)。
            if self._end_sign[i] > 0:
                rs[:, 3] = 0.0; rs[:, 4:6] = 0.0; rs[:, 6] = 1.0
            else:
                rs[:, 3] = 1.0; rs[:, 4:7] = 0.0
            rs[:, 7:] = 0.0
            rb.write_root_pose_to_sim(rs[:, :7], env_ids)
            rb.write_root_velocity_to_sim(rs[:, 7:], env_ids)

            jp = rb.data.default_joint_pos[env_ids].clone()
            jv = rb.data.default_joint_vel[env_ids].clone()
            jp[:] = 0.0; jv[:] = 0.0
            # 焊接阶段起步姿态: 臂=直臂 pose、夹爪 state=张开。state 张开是为避免与焊死的杆穿插
            # 导致 depenetration 弹飞(闭合命令在 _apply_action 里下发, PD 轻夹)。真抓取从蜷缩 stow
            # +扰动起步、走三阶段课程。
            if _P.weld_phase < 2:
                arm_init = self._arm_straight_pose_t.unsqueeze(0).repeat(n, 1)
                grip_init = self._gripper_open_t
            else:
                arm_init = self._arm_stow_t.unsqueeze(0).repeat(n, 1) \
                    + torch.zeros(n, 4, device=self.device).uniform_(-0.1, 0.1)
                arm_init = torch.clamp(arm_init, self._arm_lower, self._arm_upper)
                grip_init = self._gripper_close_t
            for k, jid in enumerate(self._arm_ids[i]):
                jp[:, jid] = arm_init[:, k]
            for k, jid in enumerate(self._grip_ids[i]):
                jp[:, jid] = grip_init[k]
            rb.write_joint_state_to_sim(jp, jv, None, env_ids)
            self._arm_dof_targets[env_ids, i] = arm_init

            self._desired_pos_w[env_ids, i, 0] = origins[:, 0] + self._end_sign[i] * _P.bar_grasp_x
            self._desired_pos_w[env_ids, i, 1] = origins[:, 1]
            self._desired_pos_w[env_ids, i, 2] = origins[:, 2] + bar_base_z + _P.hover_offset

        # ── near-grasp 反向课程: assist 比例的 env 从"两机已夹住被托平的杆"起步 ──
        #   绕过从零乱飞: 直接从成功态学"夹住 + 一起抬" → all_contact_frac 开局就 >0 → 大协作
        #   奖励立刻触发。跑通后按成功率退火 grasp_assist_level → 0(全自己 reach + 抓)。
        if _P.use_grasp_assist and _P.grasp_assist_level > 0.0:
            assist = torch.rand(n, device=self.device) < _P.grasp_assist_level
            if bool(assist.any()):
                ai = assist.nonzero(as_tuple=False).squeeze(-1)   # 局部索引 0..n-1
                aenv = env_ids[ai]                                 # 全局 env 索引
                for i, rb in enumerate(self._robots):
                    jp = rb.data.joint_pos[aenv].clone()
                    jv = torch.zeros_like(rb.data.joint_vel[aenv])
                    for k, jid in enumerate(self._arm_ids[i]):
                        jp[:, jid] = self._arm_straight_pose_t[k]   # 臂=直臂 pose
                    for k, jid in enumerate(self._grip_ids[i]):
                        jp[:, jid] = self._gripper_close_t[k]       # 夹爪=闭合(夹住杆)
                    rb.write_joint_state_to_sim(jp, jv, None, aenv)
                    self._arm_dof_targets[aenv, i] = self._arm_straight_pose_t
                    # 机体降到 grasp 高度(直臂时夹爪正好落杆端)
                    pose = rb.data.default_root_state[aenv].clone()
                    pose[:, 0] = origins[ai, 0] + self._end_sign[i] * _P.bar_grasp_x
                    pose[:, 1] = origins[ai, 1]
                    pose[:, 2] = origins[ai, 2] + _P.assist_spawn_z
                    if self._end_sign[i] > 0:
                        pose[:, 3] = 0.0; pose[:, 4:6] = 0.0; pose[:, 6] = 1.0
                    else:
                        pose[:, 3] = 1.0; pose[:, 4:7] = 0.0
                    pose[:, 7:] = 0.0
                    rb.write_root_pose_to_sim(pose[:, :7], aenv)
                    rb.write_root_velocity_to_sim(pose[:, 7:], aenv)
                    # 飞控目标也降到该高度, 否则飞控立刻把无人机拉回 hover 高空、脱离杆端
                    self._desired_pos_w[aenv, i, 2] = origins[ai, 2] + _P.assist_spawn_z
                # 跳过 settle/straighten 课程(否则前若干步把臂拉回 stow 抹掉起步态), 当帧进抓取/搬运态
                self._steps_since_reset[aenv] = _P.settle_steps + _P.straighten_steps
                self._grasp_phase[aenv] = True

    # ==============================================================
    # 可视化(搬运目标)
    # ==============================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vis"):
                mc = CUBOID_MARKER_CFG.copy()
                mc.markers["cuboid"].size = (0.08, 0.08, 0.08)
                mc.prim_path = "/Visuals/Command/carry_goal_marl"
                self.goal_vis = VisualizationMarkers(mc)
            self.goal_vis.set_visibility(True)
        else:
            if hasattr(self, "goal_vis"):
                self.goal_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        goal = self._bar_initial_pos.clone()
        goal[:, 2] = goal[:, 2] + _P.carry_height
        self.goal_vis.visualize(goal)
