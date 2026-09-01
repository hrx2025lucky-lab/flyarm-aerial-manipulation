# ================================================================
# 空中机械臂 (flyarm) —— 基础强化学习环境
#
# 一架四旋翼搭载 4 自由度机械臂和二指夹爪, 学习起飞、悬停在台子上方的物体上、
# 竖直伸出机械臂、抓取物体并抬升/搬运。基于 Isaac Lab 的 DirectRLEnv 实现。
#
#   - Action (9):  飞行指令 + 机械臂 (base_rotate/B/C/D) + 夹爪
#   - Observation (27): 机体系速度、重力投影、目标、机械臂角度/速度、物体与抓取几何
#   - Reward (11 项): 飞向目标、姿态、靠近、夹爪时序、竖直/伸直姿态、抬升、平滑、安全
#   - 飞行控制: 可选 串级 LADRC / 几何 setpoint / 直接推力-力矩 内环
#
# 注意: 修改 URDF 需要重新转换 USD (用 --collider-type convex_decomposition)。
# 仅改参数/奖励则不需要。
# ================================================================

# ────────────────────────────────────────────────────────────────
# RL 四要素及其在本文件中的位置:
#   - Observation (策略看到的 27 个数) → _get_observations()
#   - Action (策略输出的 9 个数)        → _pre_physics_step() / _apply_action()
#   - Reward (每步得分, 11 项)          → _get_rewards()
#   - 回合终止 / 重置                    → _get_dones() / _reset_idx()
#   每帧顺序: 解析 action → 应用 → 物理 → 观测 → 奖励 → 终止 → 重置
#   两大块: FlyarmEnvCfg (参数) + FlyarmEnv (逻辑)。
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_apply_inverse
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import CUBOID_MARKER_CFG


# ──────────────────────────────────────────────────────────────
# UI 窗口
# ──────────────────────────────────────────────────────────────
class FlyarmEnvWindow(BaseEnvWindow):
    def __init__(self, env: FlyarmEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# ──────────────────────────────────────────────────────────────
# 环境配置
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmEnvCfg(DirectRLEnvCfg):

    # === 基本参数 ===
    episode_length_s = 15.0          # 15 秒一个回合
    decimation = 2                   # 100Hz 物理 / 50Hz RL —— 飞行任务的标准设置
    action_space = 9                 # 推力 1 + 力矩 3 + 机械臂 4 (base_rotate/B/C/D) + 夹爪 1
    observation_space = 27
    state_space = 0
    debug_vis = True
    ui_window_class_type = FlyarmEnvWindow

    # === 物理仿真 ===
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,                  # 100Hz —— 旋翼动力学需要 >=50Hz, 100Hz 留有余量
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # === 地形 (平面) ===
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # === 场景 ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # === 机器人 ===
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # USD 路径可通过环境变量 FLYARM_USD 在别的机器上覆盖。
            usd_path=os.environ.get(
                "FLYARM_USD", "C:/Users/zhekzhong2-c/Desktop/flyarmurdf/urdf/flyarm/flyarm.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 关闭自碰撞: 蜷缩时机械臂各连杆贴得很近且用整 STL 碰撞;
                # 开启自碰撞会在折叠姿态下死锁/抖动。
                enabled_self_collisions=False,
            ),
        ),
        actuators={
            # 旋翼: 仅视觉旋转, 不产生升力 (升力以外力施加)。
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=[".*_rotor_joint"],
                stiffness=0.0,
                damping=0.001,
                velocity_limit=200.0,
            ),
            # 机械臂: PD 位置控制。
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "base_rotate_joint", "axis_B_joint", "axis_C_joint", "axis_D_joint",
                ],
                stiffness=40.0,
                damping=4.0,
                velocity_limit=5.0,
            ),
            # 夹爪 (revolute)。刚度单位 Nm/rad; effort_limit 压低以避免过力穿模:
            # 按指长计算, URDF effort=5Nm 会在 15g 物体上产生 ~55N。1Nm (~11N) 已足够
            # 提供夹持力。
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_right_joint", "gripper_left_joint"],
                stiffness=50.0,
                damping=2.0,
                effort_limit=1.0,
            ),
        },
    )

    # === 物体 ===
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            # 4cm 立方。宽度 > 夹爪闭合间距 (~3.5cm), 闭合时产生夹持力; 宽高比 1.0 抗倾倒。
            # 安装在宽台上让抓取区位于台面上方 (grasp_z_offset 瞄准上半部)。
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,   # 更快把穿透的物体推出 (抗穿模)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
            # contact_offset 5mm: 在指尖真正穿入之前就检测到接触。
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.8, 0.2), metallic=0.1
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=5.0,     # 高摩擦 —— 纯物理抓取, 不靠粘附辅助
                dynamic_friction=5.0,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.22),       # 台面 0.20 + 半高 0.02
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # === 台子: 单个宽台 (kinematic 静态刚体) ===
    #   解决一个几何死结: 脚架 (≈臂长) 在够地面物体时会触地, 所以把物体垫到台子上。
    #   高物体让抓取区保持在台面上方 (手指夹物体的上中部, 永远够不到台面),
    #   这样宽台面就不会把张开的手指撑住。
    pedestal_height = 0.20            # 支撑顶面高度; 用于掉落阈值 / 重置 / 摆放
    pedestal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pedestal",
        spawn=sim_utils.CuboidCfg(
            # 25x25 台面: 足够大, 让台肩边缘位于夹爪操作区外 ~12.5cm, 手指不会卡到台肩。
            size=(0.25, 0.25, 0.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   # 静态
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),   # 中心 = 高度/2
    )

    # === 飞行动力学参数 ===
    thrust_to_weight = 1.9           # 最大推力 = 1.9 倍总重 (含机械臂, 取自 PhysX 质量)
    moment_scale = 0.4               # 用于直接力矩控制路径 (见 use_setpoint_control)

    # ══════════════════════════════════════════════════
    # 几何 setpoint + 内环 (Deshmukh 式 INDI): RL 输出期望加速度 (3, 世界系) + 偏航率 (1);
    # 内环几何控制器算出 推力 + 力矩。消融实验表明, 输出加速度 + 内环跟踪比直接输出力矩
    # 更稳、在机械臂运动时抗扰更强。
    # 增益/符号需在 sim 里调。默认关闭; use_setpoint_control=False 退回已验证的直接力矩控制器。
    # ══════════════════════════════════════════════════
    # 默认关闭: attitude_p=8 时力矩 (~8Nm) 远超本平台的小转动惯量 (~0.4Nm),
    # 会瞬间炸转/发散。增益已在此调小。
    use_setpoint_control = False     # True 启用 INDI (增益仍需调)
    accel_scale = 3.0                # 期望加速度幅度
    yaw_rate_scale = 1.5
    attitude_p = 1.0                 # 从 8 调小 —— 原力矩远超平台承受
    attitude_d = 0.15                # 角速度阻尼 (与直接控制的 attitude_rate_damping ~0.05 对应)
    yaw_rate_p = 0.1

    # ══════════════════════════════════════════════════
    # 串级 LADRC 内环 (控制方案取自 LADRC_Attitude_Control / Init_control)。
    #   RL 输出 [期望加速度 (3, 世界系) + 偏航率 (1)] → 外环角度 LADRC (roll/pitch)
    #   → 角速度 setpoint → 内环角速度 LADRC (roll/pitch/yaw) → 力矩。
    #   b0 已按本平台重标 (直接力矩输出下 b0 = 1/转动惯量)。
    #   必须在 sim 里调 (先验悬停; 若机身转错方向就翻力矩符号)。sim-to-real 时
    #   内环必须与真机 PX4 内环一致 (同方程、增益、离散、频率) —— 那时把 sim 提到 ~1kHz。
    # ══════════════════════════════════════════════════
    use_ladrc_control = True         # 直接 LADRC 飞控 (抑制机械臂引起的扰动)。
    ladrc_in_wc_rp = 10.5; ladrc_in_wo_rp = 40.0; ladrc_in_b0_rp = 80.0   # 内环角速度 roll/pitch; b0=1/Jxx (~80)。太激进就调大 b0; 太迟钝就调小。
    ladrc_in_wc_yaw = 8.0; ladrc_in_wo_yaw = 35.0; ladrc_in_b0_yaw = 130.0 # 内环角速度 yaw; b0=1/Jzz (~130)
    ladrc_out_wc = 2.0; ladrc_out_wo = 8.1; ladrc_out_b0 = 1.0            # 外环角度 roll/pitch; b0=1 (角度环 θ̇=ω → 输入即角速度)
    ladrc_z2_clamp_rp = 0.3; ladrc_z2_clamp_yaw = 0.2                     # 扰动估计 (z2) 钳制 = anti-windup
    ladrc_max_angle = 0.61           # 外环角度 setpoint 限幅, 35° = 0.61rad
    ladrc_max_rate = 4.0             # 角速度 setpoint 限幅, rad/s

    # === 机械臂控制参数 ===
    arm_speed_scale = 1.0            # 机械臂动作放慢 → 少甩动、稳住伸展姿态 (仍能在 ~2s 内伸开)
    # base_rotate (整臂偏航) 收紧在朝前的 "正面" 朝向附近 (center ± clamp), 而非 0;
    # 旋转整条机械臂会把机身甩偏。
    base_rotate_center = 0.45        # 朝前的中心 = arm_stow 的 base_rotate
    base_rotate_clamp = 0.3          # 偏航范围 (center ± 0.3 ≈ ±17°); 物体近乎居中。抓偏置物体时放宽。

    # 夹爪 (revolute), 每指目标角 (右, 左)。镜像的手指网格 → 对称值:
    # open=(0.8, 0.8), close=(0.20, 0.20)。方向已在 viewer 验证。
    gripper_open = (0.8, 0.8)        # 对称张开
    gripper_close = (0.20, 0.20)     # 对称闭合; 两个一起加大可夹得更紧

    # 机械臂蜷缩 (起飞前) 姿态 [base_rotate, B, C, D] (rad), viewer 标定。
    arm_stow = (0.45, -0.41, 0.35, -0.10)

    # === 任务参数 ===
    hover_offset = 0.45              # 物体垫到 ~0.215; 无人机悬停在 0.215+0.45=0.665, 直臂 (~0.45) 够到物体, 脚架 (~0.45) 离地 ~0.215。
    grasp_threshold = 0.05           # 抓取中心 < 5cm = 到位
    lift_height = 0.05               # 抬升 5cm = 成功
    object_height = 0.22             # 物体中心 = 台面 0.20 + 0.02; >0.27 (=+lift_height) 算抬起
    grasp_center_z_offset = 0.02     # 历史遗留偏置 (已被 grasp_offset 取代), 为兼容保留
    # 抓取点偏置 (gripper_base/腕部坐标系, 米): 真实指尖会合点相对腕部的位置。
    # 若改用两指根中点, 会让策略把 gripper_base 怼向物体而指尖够不到。
    #   z = -0.10: 3D 打印实测夹持面距手指质心 4.5-6cm (URDF 中质心约在腕下 4.7cm) →
    #   真实夹持面距腕 ~9.5-11cm。旧值 -0.062 瞄到了手指质心, 真实指尖低 ~3.5cm 会怼台面。
    #   -0.10 瞄准真实夹持面。
    #   x = 0.0: 镜像手指对称 → 居中 (避免把物体撞倒)。
    #   y = -0.020: 横向估值。
    grasp_offset = (0.0, -0.020, -0.10)
    # 瞄准点上移到物体上半部: 物体中心 + grasp_z_offset。
    # 瞄上半部 (而非中心) 让指尖保持在高处、远离台面。用于
    # reaching/engagement/grasp_close/obs 距离; 抬升高度仍按物体真实位置判定。
    grasp_z_offset = 0.015
    # 三阶段机械臂课程 (先伸直再抓取):
    #   1. settle: 强制蜷缩姿态 → 无人机先稳定悬停。
    #   2. straighten: 强制机械臂从蜷缩插值到直臂姿态。
    #   3. free: 放开机械臂, 策略在臂已伸直的情况下自由抓取。
    settle_steps = 75                # 阶段 1: 蜷缩悬停, 1.5s
    straighten_steps = 100           # 阶段 2: 缓慢伸直 (2s) 让重心平移更柔和; 0 = 关闭

    # ══════════════════════════════════════════════════
    # 抓取之后: 带物体上升并锁定机械臂竖直。
    #   触发: 物体高出初始高度超过 carry_trigger = 真抓住 (台上物体不会自己升高)。
    #   搬运模式下: (1) 抬高飞行目标 z, 让 dist_to_goal 驱动无人机直上;
    #   (2) 锁定机械臂到直臂竖直姿态 (压过策略); (3) 强制闭爪防掉块。滑脱则退出搬运模式。
    # ══════════════════════════════════════════════════
    carry_after_grasp = True         # 启用搬运 + 竖直锁臂; False = 仅悬停抓取
    carry_trigger = 0.02             # 物体升高 >2cm = "抓住" (台上判定可靠)
    carry_height = 0.30              # 抓住后, 把飞行目标抬到悬停高度上方 0.30m (→0.97)

    # ══════════════════════════════════════════════════
    # 奖励权重
    # ══════════════════════════════════════════════════
    # 第 1 组: 飞向目标
    dist_reward_scale = 8.0          # 飞行稳定最高优先
    orientation_penalty_scale = -8.0 # 强姿态惩罚, 抗翻滚
    # 第 2 组: 末端靠近物体
    reaching_reward_scale = 4.0      # 把夹爪拉向物体 (用 1-tanh 核保证远处也有梯度)
    finger_reward_scale = 2.0        # 远处闭爪扣分, 近处闭爪加分
    # 竖直抓取: 指尖在腕部下方 → 夹爪朝下。
    gripper_vertical_reward_scale = 4.0  # 加强 —— 配合把抓取奖励门控在竖直度上
    # 直臂奖励: 把机械臂从蜷缩拉到竖直朝下目标 (B≈0, C≈-1.48)。
    # 直臂 → 重心居于机体正下方 → 稳定悬停, 夹爪对准物体。
    arm_straight_target = (0.0, -1.48)   # B=0 让第一节臂垂直 (重心偏移最小); C=-1.48。也是抓后锁臂姿态的 B。
    arm_straight_reward_scale = 5.0      # 加大以坚决伸直机械臂; 门控的抓取奖励让它保持伸直
    # 第 3 组: 抬升物体
    lifting_reward_scale = 15.0      # 抬升 >=5cm 的稀疏 all-or-nothing 奖励
    # 连续抬升塑形: 夹爪闭合且靠近物体时, 物体每升高一点给一点分。
    lift_shaping_reward_scale = 20.0
    # 第 4 组: 正则化
    action_rate_penalty_scale = -1e-4
    joint_vel_penalty_scale = -1e-4
    # 安全
    crash_penalty = -50.0


# ──────────────────────────────────────────────────────────────
# 环境主类
# ──────────────────────────────────────────────────────────────
class FlyarmEnv(DirectRLEnv):
    """
    flyarm 空中机械臂环境。

    执行循环:
      _pre_physics_step(actions) → 解析 9 维 action
      _apply_action()            → 把指令下发给 sim (decimation 次)
      [PhysX]
      _get_observations()        → 27 维状态
      _get_rewards()             → 奖励各项
      _get_dones()               → 坠毁 / 超时
      _reset_idx()               → 重置
    """

    cfg: FlyarmEnvCfg

    def __init__(self, cfg: FlyarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── action / 状态张量 ──
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # LADRC 状态: 5 个一阶 LADRC (0,1 外环 roll/pitch; 2,3,4 内环 roll/pitch/yaw), 各存 z1/z2/u_prev。
        self._ladrc_z1 = torch.zeros(self.num_envs, 5, device=self.device)
        self._ladrc_z2 = torch.zeros(self.num_envs, 5, device=self.device)
        self._ladrc_u = torch.zeros(self.num_envs, 5, device=self.device)

        # 机械臂关节目标 —— 增量控制需要持续跟踪。
        self._arm_dof_targets = torch.zeros(self.num_envs, 4, device=self.device)

        # ── 飞行目标 ──
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── 抓取状态 ──
        self._object_initial_height = torch.zeros(self.num_envs, device=self.device)
        # 本回合物体到过的最高高度 —— 用于诊断瞬时抬升 (例如短暂抓起又掉)。
        self._max_object_height = torch.zeros(self.num_envs, device=self.device)

        # 每回合步数计数器 (驱动 settle 课程); 在 _reset_idx 里重置。
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 阶段 3 标志 (机械臂已伸直, 可以开始抓取)。门控 reaching/engagement/progress/finger,
        # 使前两阶段只悬停+伸直, 不被 "够到物体" 的奖励往下拉。
        self._grasp_phase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 搬运模式标志: 物体高出初始高度 → 带物上升 + 锁臂 + 强制闭爪。
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 抓取偏置张量 (腕部坐标系), 用于计算指尖会合点。
        self._grasp_offset_t = torch.tensor(self.cfg.grasp_offset, device=self.device)

        # ── 奖励记账 ──
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "dist_to_goal", "orientation", "reaching", "finger",
                "gripper_vertical", "arm_straight", "lifting", "lift_shaping", "action_rate", "joint_vel", "crash",
            ]
        }

        # ── 关节 / body 索引 ──
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._rotor_joint_ids = self._robot.find_joints(".*_rotor_joint")[0]
        # preserve_order=True 很关键: find_joints 返回 articulation 顺序而非传入顺序;
        # 不锁定的话 arm_stow/clamp 和夹爪左右会错位。
        self._arm_joint_ids = self._robot.find_joints(
            ["base_rotate_joint", "axis_B_joint", "axis_C_joint", "axis_D_joint"],
            preserve_order=True,
        )[0]
        self._ee_body_id = self._robot.find_bodies("gripper_base")[0]   # 腕部 (末端连杆)
        # 夹爪手指: 顺序固定为 (右, 左), 与 gripper_open/close 元组对应。
        self._gripper_joint_ids = self._robot.find_joints(
            ["gripper_right_joint", "gripper_left_joint"], preserve_order=True
        )[0]
        self._left_finger_id = self._robot.find_bodies("gripper_left")[0]
        self._right_finger_id = self._robot.find_bodies("gripper_right")[0]

        # ── 机械臂限位: 读 URDF 软限位, 并额外收紧 base_rotate ──
        if hasattr(self._robot.data, "soft_joint_pos_limits"):
            soft_limits = self._robot.data.soft_joint_pos_limits[0]   # (num_joints, 2)
        else:
            soft_limits = self._robot.data.joint_pos_limits[0]       # 回退字段名
        self._arm_lower = soft_limits[self._arm_joint_ids, 0].clone()
        self._arm_upper = soft_limits[self._arm_joint_ids, 1].clone()
        # base_rotate 是机械臂关节 0: 锁在朝前中心附近 (旋转整臂会把机身甩偏)。
        br = self.cfg.base_rotate_clamp
        brc = self.cfg.base_rotate_center
        self._arm_lower[0] = torch.clamp(self._arm_lower[0], min=brc - br)
        self._arm_upper[0] = torch.clamp(self._arm_upper[0], max=brc + br)

        # ── 夹爪张开/闭合目标张量 (2,), 顺序 (右, 左) ──
        self._gripper_open_t = torch.tensor(self.cfg.gripper_open, device=self.device)
        self._gripper_close_t = torch.tensor(self.cfg.gripper_close, device=self.device)
        # 机械臂蜷缩初始姿态 (4,)
        self._arm_stow_t = torch.tensor(self.cfg.arm_stow, device=self.device)
        # 直臂目标 (B, C) —— arm_straight 奖励把机械臂拉到这个竖直朝下姿态。
        self._arm_straight_target_t = torch.tensor(self.cfg.arm_straight_target, device=self.device)
        # 完整直臂姿态 [base_rotate, B, C, D], 供课程阶段 2 强制伸直用。
        self._arm_straight_pose_t = torch.tensor(
            [self.cfg.base_rotate_center, self.cfg.arm_straight_target[0],
             self.cfg.arm_straight_target[1], self.cfg.arm_stow[3]], device=self.device
        )

        # ── 质量与重力 ── (含机械臂的总质量; 自动求和, 让 thrust_to_weight 自适应)
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ── 信息 ──
        print(f"[flyarm]Robot mass: {self._robot_mass:.3f} kg, weight: {self._robot_weight:.3f} N")
        print(f"[flyarm]thrust_to_weight={self.cfg.thrust_to_weight}, moment_scale={self.cfg.moment_scale}")
        print(f"[flyarm]Arm joints (ids): {self._arm_joint_ids}")
        print(f"[flyarm]Arm clamp lower={self._arm_lower.tolist()}")
        print(f"[flyarm]Arm clamp upper={self._arm_upper.tolist()}")
        print(f"[flyarm]Arm stow={self.cfg.arm_stow}")
        print(f"[flyarm]Gripper ids(右,左)={self._gripper_joint_ids}, open={self.cfg.gripper_open}, close={self.cfg.gripper_close}")
        print(f"[flyarm]Init height = object_h({self.cfg.object_height}) + hover({self.cfg.hover_offset}) = {self.cfg.object_height + self.cfg.hover_offset:.3f}m")

        self.set_debug_vis(self.cfg.debug_vis)

    # ==============================================================
    # 场景搭建
    # ==============================================================
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self._object

        # 台子 (kinematic 静态刚体): 垫高物体让脚架离地。
        self._pedestal = RigidObject(self.cfg.pedestal)
        self.scene.rigid_objects["pedestal"] = self._pedestal

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ==============================================================
    # 一阶 LADRC (单轴, 按列 idx 向量化)。
    #   二阶 LESO (z1 跟踪输出 y, z2 跟踪总扰动) + LSEF 扰动补偿, 带宽参数化。
    #   外环: u = 角速度 setpoint; 内环: u = 力矩 setpoint。
    # ==============================================================
    def _ladrc_axis(self, idx, r, y, wc, wo, b0, h, z2_clamp):
        z1 = self._ladrc_z1[:, idx]; z2 = self._ladrc_z2[:, idx]; u_prev = self._ladrc_u[:, idx]
        beta1 = 2.0 * wo; beta2 = wo * wo
        e = z1 - y                                    # LESO 使用上一步实际施加的 u_prev
        z1 = z1 + h * (z2 - beta1 * e + b0 * u_prev)
        z2 = z2 + h * (-beta2 * e)
        z2 = z2.clamp(-z2_clamp, z2_clamp)            # anti-windup: 钳制扰动估计
        u = (wc * (r - z1) - z2) / b0                 # 控制律 u = (u0 - z2)/b0, u0 = wc*(r - z1)
        self._ladrc_z1[:, idx] = z1; self._ladrc_z2[:, idx] = z2; self._ladrc_u[:, idx] = u
        return u

    # ==============================================================
    # 解析 action —— 9 个 [-1,1] 的值 → 物理指令
    # (推力、力矩、机械臂动作、夹爪开/合)
    # ==============================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._steps_since_reset += 1     # 每回合步数计数器 (settle 课程)

        # --- 飞行控制 ---
        if self.cfg.use_ladrc_control:
            # 串级 LADRC: RL [加速度 (世界系) + 偏航率] → 外环角度 LADRC → 内环角速度 LADRC → 力矩。
            h = self.cfg.sim.dt * self.cfg.decimation               # 步长 (50Hz; sim-to-real 时提到 1kHz)
            quat = self._robot.data.root_quat_w
            omega = self._robot.data.root_ang_vel_b                 # 实测机体角速度 p,q,r
            grav_b = self._robot.data.projected_gravity_b           # 实测倾斜: 水平时为 [0,0,-1]
            g = self._gravity_magnitude
            a_des = self._actions[:, 0:3] * self.cfg.accel_scale            # 期望加速度 (世界系)
            yaw_rate_des = self._actions[:, 3] * self.cfg.yaw_rate_scale    # 期望偏航率
            # 推力 (沿机体 z) = m·(a_des + g·ẑ)·b3, 限幅。
            t_des = a_des.clone(); t_des[:, 2] = t_des[:, 2] + g
            ez = torch.zeros_like(a_des); ez[:, 2] = 1.0
            b3 = quat_apply(quat, ez)
            self._thrust[:, 0, 2] = (self._robot_mass * torch.sum(t_des * b3, dim=1)).clamp(
                0.0, self.cfg.thrust_to_weight * self._robot_weight)
            # 期望/实测 roll,pitch (小角度); 期望来自水平加速度。
            pitch_meas = grav_b[:, 0]; roll_meas = -grav_b[:, 1]
            pitch_des = (a_des[:, 0] / g).clamp(-self.cfg.ladrc_max_angle, self.cfg.ladrc_max_angle)
            roll_des = (-a_des[:, 1] / g).clamp(-self.cfg.ladrc_max_angle, self.cfg.ladrc_max_angle)
            # 外环角度 LADRC → 角速度 setpoint (限幅)。
            p_sp = self._ladrc_axis(0, roll_des, roll_meas, self.cfg.ladrc_out_wc, self.cfg.ladrc_out_wo,
                                    self.cfg.ladrc_out_b0, h, self.cfg.ladrc_z2_clamp_rp).clamp(
                                        -self.cfg.ladrc_max_rate, self.cfg.ladrc_max_rate)
            q_sp = self._ladrc_axis(1, pitch_des, pitch_meas, self.cfg.ladrc_out_wc, self.cfg.ladrc_out_wo,
                                    self.cfg.ladrc_out_b0, h, self.cfg.ladrc_z2_clamp_rp).clamp(
                                        -self.cfg.ladrc_max_rate, self.cfg.ladrc_max_rate)
            # 内环角速度 LADRC → 力矩。若机身转错方向, 翻这三个符号。
            self._moment[:, 0, 0] = self._ladrc_axis(2, p_sp, omega[:, 0], self.cfg.ladrc_in_wc_rp,
                                    self.cfg.ladrc_in_wo_rp, self.cfg.ladrc_in_b0_rp, h, self.cfg.ladrc_z2_clamp_rp)
            self._moment[:, 0, 1] = self._ladrc_axis(3, q_sp, omega[:, 1], self.cfg.ladrc_in_wc_rp,
                                    self.cfg.ladrc_in_wo_rp, self.cfg.ladrc_in_b0_rp, h, self.cfg.ladrc_z2_clamp_rp)
            self._moment[:, 0, 2] = self._ladrc_axis(4, yaw_rate_des, omega[:, 2], self.cfg.ladrc_in_wc_yaw,
                                    self.cfg.ladrc_in_wo_yaw, self.cfg.ladrc_in_b0_yaw, h, self.cfg.ladrc_z2_clamp_yaw)
        elif self.cfg.use_setpoint_control:
            # 几何 setpoint: RL [加速度 (3, 世界系), 偏航率 (1)] → 内环算出推力+力矩。
            quat = self._robot.data.root_quat_w          # (N,4) wxyz
            omega = self._robot.data.root_ang_vel_b      # (N,3) 机体角速度
            a_des_w = self._actions[:, 0:3] * self.cfg.accel_scale          # 期望加速度 (世界系)
            yaw_rate_des = self._actions[:, 3] * self.cfg.yaw_rate_scale    # 期望偏航率
            # 期望比力 (世界系) = a_des + g·ẑ (抵消重力)。
            t_des_w = a_des_w.clone()
            t_des_w[:, 2] = t_des_w[:, 2] + self._gravity_magnitude
            b3_des_w = t_des_w / (torch.linalg.norm(t_des_w, dim=1, keepdim=True) + 1e-6)   # 期望机体 z 方向
            # 当前机体 z (世界系) = R·[0,0,1]
            ez = torch.zeros_like(a_des_w); ez[:, 2] = 1.0
            b3_w = quat_apply(quat, ez)
            # 推力 (沿机体 z) = m·(t_des·b3), 限幅到 [0, T2W·weight]。
            thrust_mag = self._robot_mass * torch.sum(t_des_w * b3_w, dim=1)
            self._thrust[:, 0, 2] = torch.clamp(thrust_mag, 0.0, self.cfg.thrust_to_weight * self._robot_weight)
            # 姿态: 把机体 z 拉向 b3_des; 旋转轴 = b3×b3_des (世界系) → 机体系。
            tilt_b = quat_apply_inverse(quat, torch.cross(b3_w, b3_des_w, dim=1))
            torque = self.cfg.attitude_p * tilt_b                       # roll/pitch 倾斜 P
            torque[:, 2] = self.cfg.yaw_rate_p * (yaw_rate_des - omega[:, 2])   # 偏航率跟踪
            torque = torque - self.cfg.attitude_d * omega              # 角速度阻尼 (INDI 的 D)
            self._moment[:, 0, :] = torque
        else:
            # 直接力矩控制 (已验证的回退): a0 → 推力, a1:4 → 力矩。
            self._thrust[:, 0, 2] = self._robot_weight * (
                1.0 + self._actions[:, 0] * (self.cfg.thrust_to_weight - 1.0)
            )
            self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]

        # --- 机械臂: 增量位置控制 ---
        dt = self.cfg.sim.dt * self.cfg.decimation   # 0.02s
        arm_delta = self._actions[:, 4:8] * self.cfg.arm_speed_scale * dt
        self._arm_dof_targets = self._arm_dof_targets + arm_delta
        self._arm_dof_targets = torch.clamp(
            self._arm_dof_targets, self._arm_lower, self._arm_upper
        )
        # 三阶段课程: (1) settle = 强制蜷缩, (2) straighten = 蜷缩→直臂插值,
        # (3) 自由抓取。
        steps = self._steps_since_reset
        settle = self.cfg.settle_steps
        straighten = self.cfg.straighten_steps
        stow_b = self._arm_stow_t.unsqueeze(0).expand(self.num_envs, 4)
        # 阶段 1: 蜷缩悬停
        settling = (steps < settle).unsqueeze(-1)   # (N,1)
        self._arm_dof_targets = torch.where(settling, stow_b, self._arm_dof_targets)
        # 阶段 2: frac 0→1 从蜷缩插值到直臂姿态
        if straighten > 0:
            in_str = ((steps >= settle) & (steps < settle + straighten)).unsqueeze(-1)   # (N,1)
            frac = ((steps - settle).clamp(min=0).float() / float(straighten)).clamp(0.0, 1.0).unsqueeze(-1)   # (N,1)
            ramp = (1.0 - frac) * stow_b + frac * self._arm_straight_pose_t.unsqueeze(0).expand(self.num_envs, 4)
            self._arm_dof_targets = torch.where(in_str, ramp, self._arm_dof_targets)
        # 阶段 3 标志: 过了 settle+straighten = 可以开始抓取 (门控 reaching 类奖励)。
        self._grasp_phase = steps >= (settle + straighten)

        # 搬运模式: 物体高出初始高度 + carry_trigger = 抓住。
        # 最高优先覆盖: 锁定机械臂到直臂竖直姿态, 让它在搬运中保持竖直平稳。
        if self.cfg.carry_after_grasp:
            object_z = self._object.data.root_pos_w[:, 2]
            self._grasped = object_z > (self._object_initial_height + self.cfg.carry_trigger)
            grasped_arm = self._grasped.unsqueeze(-1).expand(self.num_envs, 4)   # (N,4)
            self._arm_dof_targets = torch.where(
                grasped_arm,
                self._arm_straight_pose_t.unsqueeze(0).expand(self.num_envs, 4),
                self._arm_dof_targets,
            )

    # ==============================================================
    # 应用 action —— 把指令下发给物理引擎
    # ==============================================================
    def _apply_action(self):
        # --- 1. 飞行推力 + 力矩 → base_link ---
        self._robot.set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._body_id
        )

        # --- 2. 机械臂 → PD 位置控制 ---
        self._robot.set_joint_position_target(
            self._arm_dof_targets, joint_ids=self._arm_joint_ids
        )

        # --- 3. 夹爪: 二值控制 (action[8] >= 0 → 闭合, < 0 → 张开) ---
        #   每指目标元组 (右, 左); revolute 镜像方向已内置。
        gripper_cmd = self._actions[:, 8]
        sel_close = (gripper_cmd >= 0.0).float().unsqueeze(-1)   # (N,1)
        # 搬运模式强制闭合 (抓住后不再松开): max → grasped=1 时强制闭合。
        if self.cfg.carry_after_grasp:
            sel_close = torch.maximum(sel_close, self._grasped.float().unsqueeze(-1))
        gripper_targets = (
            sel_close * self._gripper_close_t + (1.0 - sel_close) * self._gripper_open_t
        )   # (N,2)
        self._robot.set_joint_position_target(
            gripper_targets, joint_ids=self._gripper_joint_ids
        )

        # --- 4. 旋翼视觉旋转 (仅装饰) ---
        thrust_norm = (self._actions[:, 0] + 1.0) / 2.0
        spd = 50.0
        rotor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        rotor_speeds[:, 0] = spd * (0.5 + thrust_norm)
        rotor_speeds[:, 1] = -spd * (0.5 + thrust_norm)
        rotor_speeds[:, 2] = -spd * (0.5 + thrust_norm)
        rotor_speeds[:, 3] = spd * (0.5 + thrust_norm)
        self._robot.set_joint_velocity_target(
            rotor_speeds, joint_ids=self._rotor_joint_ids
        )

    # ==============================================================
    # 抓取点 = 指尖会合点 (obs / reward / metrics 共用)。
    #   真实抓取位置是指尖会合处, 不是腕部 (gripper_base)。把固定的 grasp_offset
    #   从腕部坐标系变换到世界系, 让策略把指尖对准物体 (腕部停在其上方)。
    # ==============================================================
    def _compute_grasp_center(self) -> torch.Tensor:
        wrist_pos = self._robot.data.body_pos_w[:, self._ee_body_id[0], :]
        wrist_quat = self._robot.data.body_quat_w[:, self._ee_body_id[0], :]
        return wrist_pos + quat_apply(wrist_quat, self._grasp_offset_t.unsqueeze(0).expand(self.num_envs, 3))

    # ==============================================================
    # 抓取瞄准点 = 物体中心 + 上移偏置 (grasp_z_offset)。
    #   瞄物体上半部, 让指尖保持在高处、避开台面。
    #   仅用于 "瞄哪里" (reaching/engagement/grasp_close/obs); 抬升高度按物体真实位置判定。
    # ==============================================================
    def _grasp_target_w(self) -> torch.Tensor:
        t = self._object.data.root_pos_w.clone()
        t[:, 2] = t[:, 2] + self.cfg.grasp_z_offset
        return t

    # ==============================================================
    # 观测 —— 27 个数
    # ==============================================================
    def _get_observations(self) -> dict:
        """
        27 维观测:
          [0:3]   机体线速度
          [3:6]   机体角速度
          [6:9]   重力投影 (机体系) —— 倾斜
          [9:12]  飞行目标 (机体系)
          [12:16] 机械臂关节角 (归一化 [-1,1])
          [16:20] 机械臂关节速度 (*0.1)
          [20:23] 物体位置 (机体系)
          [23:26] 抓取中心 → 物体方向 (机体系)
          [26]    抓取 → 物体距离
        """
        object_pos_w = self._object.data.root_pos_w
        drone_pos_w = self._robot.data.root_pos_w

        # 飞行目标 = 物体正上方。
        self._desired_pos_w[:, 0] = object_pos_w[:, 0]
        self._desired_pos_w[:, 1] = object_pos_w[:, 1]
        self._desired_pos_w[:, 2] = object_pos_w[:, 2] + self.cfg.hover_offset
        # 抓住后: 抬高目标 z, 让 dist_to_goal 驱动无人机直上;
        # xy 仍跟踪物体 (直上不横跑)。滑脱则重置回来。
        if self.cfg.carry_after_grasp:
            carry_z = self._object_initial_height + self.cfg.hover_offset + self.cfg.carry_height
            self._desired_pos_w[:, 2] = torch.where(self._grasped, carry_z, self._desired_pos_w[:, 2])

        desired_pos_b, _ = subtract_frame_transforms(
            drone_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        object_pos_b, _ = subtract_frame_transforms(
            drone_pos_w, self._robot.data.root_quat_w, object_pos_w
        )

        # 机械臂关节角归一化 → [-1,1]。
        arm_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        arm_pos_normalized = 2.0 * (arm_pos - self._arm_lower) / (
            self._arm_upper - self._arm_lower + 1e-6
        ) - 1.0

        # 抓取点 = 指尖会合点 (腕部位姿 + grasp_offset)。
        grasp_center_w = self._compute_grasp_center()

        # 瞄物体上半部 (中心 + grasp_z_offset), 而非中心。
        to_obj_vec = self._grasp_target_w() - grasp_center_w
        to_obj_dist = torch.linalg.norm(to_obj_vec, dim=1, keepdim=True)
        to_obj_dir_w = to_obj_vec / (to_obj_dist + 1e-6)

        # 方向转到机体系。
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id[0], :]
        ee_pos_b, _ = subtract_frame_transforms(
            drone_pos_w, self._robot.data.root_quat_w, ee_pos_w
        )
        dir_endpoint_b, _ = subtract_frame_transforms(
            drone_pos_w, self._robot.data.root_quat_w, ee_pos_w + to_obj_dir_w
        )
        to_obj_dir_b = dir_endpoint_b - ee_pos_b

        # 关节速度 (缩放 0.1)。
        arm_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        arm_vel_scaled = arm_vel * 0.1

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,        # (3)
            self._robot.data.root_ang_vel_b,        # (3)
            self._robot.data.projected_gravity_b,   # (3)
            desired_pos_b,                          # (3)
            arm_pos_normalized,                     # (4)
            arm_vel_scaled,                         # (4)
            object_pos_b,                           # (3)
            to_obj_dir_b,                           # (3)
            to_obj_dist,                            # (1)
        ], dim=-1)   # 27

        return {"policy": torch.clamp(obs, -5.0, 5.0)}   # 限幅防止数值爆炸

    # ==============================================================
    # 奖励
    # ==============================================================
    def _get_rewards(self) -> torch.Tensor:
        drone_pos_w = self._robot.data.root_pos_w
        drone_height = drone_pos_w[:, 2]
        object_pos_w = self._object.data.root_pos_w
        object_height = object_pos_w[:, 2]
        # 记录本回合物体到过的最高高度 (在 _reset_idx 里记录, 用于诊断)。
        self._max_object_height = torch.maximum(self._max_object_height, object_height)

        # 抓取点 = 指尖会合点; finger_mid/wrist 保留给 gripper_vertical 用。
        left_f = self._robot.data.body_pos_w[:, self._left_finger_id[0], :]
        right_f = self._robot.data.body_pos_w[:, self._right_finger_id[0], :]
        finger_mid = (left_f + right_f) / 2.0
        grasp_center = self._compute_grasp_center()
        wrist_pos = self._robot.data.body_pos_w[:, self._ee_body_id[0], :]

        # ── 1. 飞向目标 (1/(1+d²) 平方) ──
        dist_to_goal = torch.linalg.norm(self._desired_pos_w - drone_pos_w, dim=1)
        dist_reward = 1.0 / (1.0 + dist_to_goal ** 2)
        dist_reward = dist_reward * dist_reward

        # ── 2. 姿态 (水平时 projected_gravity_b 的 xy = 0) ──
        orientation_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # ── 3. 末端靠近物体 ──
        # 瞄物体上半部, 让 reaching/grasp 判定避开台面。
        ee_to_obj_dist = torch.linalg.norm(grasp_center - self._grasp_target_w(), dim=1)
        # 1-tanh(d/0.2) 远处有梯度 (exp(-d²/0.01) 核在 0.5m 处没有梯度),
        # 能从远处把夹爪拉向物体。
        reaching_reward = 1.0 - torch.tanh(ee_to_obj_dist / 0.2)

        # ── 4. 夹爪时序 (远处闭爪扣分, 近处闭爪加分) ──
        gripper_closed = (self._actions[:, 8] >= 0.0).float()
        is_far = (ee_to_obj_dist > 0.08).float()
        is_near = (ee_to_obj_dist < 0.05).float()
        finger_reward = -is_far * gripper_closed + is_near * gripper_closed

        # ── 5. 竖直抓取: 指尖在腕部下方 ──
        #   v = 两指中点 - 腕部; 朝下 → -v_z/|v| = 1。纯几何。
        v = finger_mid - wrist_pos
        v_norm = torch.linalg.norm(v, dim=1) + 1e-6
        gripper_vertical = torch.clamp(-v[:, 2] / v_norm, min=0.0)   # [0,1], 1 = 完全朝下
        # 暴露竖直度 [0,1] 供 vision_env 把抓取奖励门控在它上面。
        self._gripper_vertical_val = gripper_vertical

        # ── 直臂: B/C 朝向竖直朝下目标 → 重心居中、稳定悬停 ──
        #   宽 1-tanh 核: 拉直但不锁死, 留出对准物体的余地。
        arm_bc = self._robot.data.joint_pos[:, [self._arm_joint_ids[1], self._arm_joint_ids[2]]]
        # 用 1-tanh (而非 exp) 让从蜷缩 (dist~1.9) 到直臂 (0) 全程都有梯度。
        arm_straight_dist = torch.linalg.norm(arm_bc - self._arm_straight_target_t, dim=1)
        arm_straight = 1.0 - torch.tanh(arm_straight_dist / 1.0)    # [0,1], 1 = 完全伸直
        # 暴露伸直度 [0,1] 供 vision_env 把抓取奖励门控在它上面
        # (强制 "伸直 → 下降 → 抓取" 的顺序)。
        self._arm_straight_val = arm_straight

        # ── 6. 物体抬起 (高度 > 阈值 AND 抓取中心 < 5cm) ──
        ee_near_object = (ee_to_obj_dist < self.cfg.grasp_threshold).float()
        lifted = ((object_height > self._object_initial_height + self.cfg.lift_height) * ee_near_object)

        # ── 6b. 连续抬升塑形 ──
        #   只有稀疏 lifting 会让策略卡在 "够到但不抓"; 当 (夹爪闭合且靠近物体) 时,
        #   物体每升高一点给一点分。gripper_closed 门控防止靠顶/推获得高度。
        object_lift = torch.clamp(
            object_height - self._object_initial_height, min=0.0, max=self.cfg.lift_height
        )
        lift_shaping = object_lift * ee_near_object * gripper_closed

        # ── 7. 动作平滑 ──
        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        # ── 8. 关节速度平滑 ──
        arm_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        joint_vel = torch.sum(torch.square(arm_vel), dim=1).clamp(max=100.0)   # 限幅防爆炸

        # ── 9. 安全 (着陆不罚; 只罚真坠毁 / 飞太高) ──
        low_danger = torch.clamp(0.20 - drone_height, min=0.0) / 0.10
        high_danger = torch.clamp(drone_height - 4.0, min=0.0) / 0.5
        crash_intensity = torch.clamp(low_danger + high_danger, max=1.0)

        # 阶段 3 门控: "够到物体" 类奖励只在机械臂伸直后才发。
        #   阶段 1/2 机械臂被锁住, 此时 reaching 会把整架无人机往下拉。
        #   门控让它先悬停+伸直, 阶段 3 再去够。
        phase3 = self._grasp_phase.float()

        # ── 汇总 ──
        rewards = {
            "dist_to_goal": dist_reward * self.cfg.dist_reward_scale * self.step_dt,
            "orientation": orientation_error * self.cfg.orientation_penalty_scale * self.step_dt,
            "reaching": reaching_reward * self.cfg.reaching_reward_scale * self.step_dt * phase3,   # 门控
            "finger": finger_reward * self.cfg.finger_reward_scale * self.step_dt * phase3,         # 门控
            "gripper_vertical": gripper_vertical * self.cfg.gripper_vertical_reward_scale * self.step_dt,
            "arm_straight": arm_straight * self.cfg.arm_straight_reward_scale * self.step_dt,
            "lifting": lifted * self.cfg.lifting_reward_scale * self.step_dt,
            "lift_shaping": lift_shaping * self.cfg.lift_shaping_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_penalty_scale * self.step_dt,
            "joint_vel": joint_vel * self.cfg.joint_vel_penalty_scale * self.step_dt,
            "crash": crash_intensity * self.cfg.crash_penalty * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # ==============================================================
    # 终止
    # ==============================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        drone_died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.12,
            self._robot.data.root_pos_w[:, 2] > 5.0,
        )
        # 发散保护: 离太远或速度爆炸 → 在 v² 惩罚冲到 inf/NaN 之前终止
        # (避免 value_loss=inf / NaN-std 的训练崩溃)。
        lin_speed = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        too_far = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) > 5.0
        diverged = torch.logical_or(lin_speed > 30.0, too_far)
        drone_died = torch.logical_or(drone_died, diverged)
        # 物体被撞下台子 (低于台面) → 回合失败。
        # 阈值留出余地, 让夹爪可以挤压/微动物体而不结束回合 (结束会惩罚探索)。
        object_fell = self._object.data.root_pos_w[:, 2] < (self.cfg.pedestal_height - 0.10)
        died = torch.logical_or(drone_died, object_fell)
        return died, time_out

    # ==============================================================
    # 重置
    # ==============================================================
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ── 日志 ──
        final_dist = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        left_f = self._robot.data.body_pos_w[env_ids, self._left_finger_id[0], :]
        right_f = self._robot.data.body_pos_w[env_ids, self._right_finger_id[0], :]
        gc = self._compute_grasp_center()[env_ids]   # 指尖会合点
        obj_pos = self._object.data.root_pos_w[env_ids]
        final_ee = torch.linalg.norm(gc - obj_pos, dim=1).mean()
        # finger_gap: revolute 指根关节间距 ~恒定 (~3.7cm), 为兼容保留; 判抓取用接触/max_object_height。
        finger_gap = torch.linalg.norm(left_f - right_f, dim=1).mean().item()
        obj_height = obj_pos[:, 2].mean().item()
        max_obj_h = self._max_object_height[env_ids].mean().item()

        extras = dict()
        for key in self._episode_sums.keys():
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_dist.item()
        extras["Metrics/final_ee_to_object"] = final_ee.item()
        extras["Metrics/max_object_height"] = max_obj_h   # 本回合到过的最高高度
        extras["Metrics/finger_gap"] = finger_gap
        extras["Metrics/object_height"] = obj_height
        self.extras["log"].update(extras)

        # ── 重置骨架 ──
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0     # 重新进入 settle 阶段
        self._grasped[env_ids] = False           # 退出搬运模式
        # 清空 LADRC 状态 (z1/z2/u), 避免扰动估计跨回合带入。
        self._ladrc_z1[env_ids] = 0.0
        self._ladrc_z2[env_ids] = 0.0
        self._ladrc_u[env_ids] = 0.0

        # ── 物体: 固定位置 + 小随机化 ──
        n = len(env_ids)
        obj_state = self._object.data.default_root_state[env_ids].clone()
        # ±0.003: 让物体在支撑上近乎居中; 以后为泛化可放宽。
        obj_state[:, 0] += torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        obj_state[:, 1] += torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        obj_state[:, :3] += self._terrain.env_origins[env_ids]
        obj_state[:, 2] = self._terrain.env_origins[env_ids, 2] + self.cfg.object_height
        obj_state[:, 7:] = 0.0
        self._object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)
        self._object_initial_height[env_ids] = obj_state[:, 2]
        self._max_object_height[env_ids] = obj_state[:, 2]   # 最高高度从台面高度起算

        # 台子: 放到各 env 原点 (kinematic, 物体坐在其顶面上)。
        ped_state = self._pedestal.data.default_root_state[env_ids].clone()
        ped_state[:, 0] = self._terrain.env_origins[env_ids, 0]
        ped_state[:, 1] = self._terrain.env_origins[env_ids, 1]
        ped_state[:, 2] = self._terrain.env_origins[env_ids, 2] + self.cfg.pedestal_height / 2.0
        ped_state[:, 7:] = 0.0
        self._pedestal.write_root_pose_to_sim(ped_state[:, :7], env_ids)
        self._pedestal.write_root_velocity_to_sim(ped_state[:, 7:], env_ids)

        # ── 飞行目标 ──
        self._desired_pos_w[env_ids, 0] = obj_state[:, 0]
        self._desired_pos_w[env_ids, 1] = obj_state[:, 1]
        self._desired_pos_w[env_ids, 2] = obj_state[:, 2] + self.cfg.hover_offset

        # ── 无人机: 出生在悬停高度, 速度为零 ──
        root_state = self._robot.data.default_root_state[env_ids].clone()
        terrain_z = self._terrain.env_origins[env_ids, 2]
        init_z = terrain_z + self.cfg.object_height + self.cfg.hover_offset
        root_state[:, 0] = obj_state[:, 0]
        root_state[:, 1] = obj_state[:, 1]
        root_state[:, 2] = init_z + torch.zeros(n, device=self.device).uniform_(-0.03, 0.03)
        root_state[:, 3] = 1.0    # qw
        root_state[:, 4:7] = 0.0  # qx,qy,qz
        root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # ── 关节: 机械臂 = 蜷缩 + 扰动, 夹爪 = 闭合 ──
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:] = 0.0
        joint_vel[:] = 0.0
        # 机械臂 → 蜷缩 + 扰动 (有助泛化)。
        arm_init = self._arm_stow_t.unsqueeze(0).repeat(n, 1)
        arm_init = arm_init + torch.zeros(n, 4, device=self.device).uniform_(-0.1, 0.1)
        arm_init = torch.clamp(arm_init, self._arm_lower, self._arm_upper)
        for k, jid in enumerate(self._arm_joint_ids):
            joint_pos[:, jid] = arm_init[:, k]
        # 夹爪 → 闭合 (起飞前蜷缩)。
        for k, jid in enumerate(self._gripper_joint_ids):
            joint_pos[:, jid] = self._gripper_close_t[k]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 机械臂目标 = 初始机械臂角度 (避免第一步跳变)。
        self._arm_dof_targets[env_ids] = arm_init

    # ==============================================================
    # 可视化
    # ==============================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
