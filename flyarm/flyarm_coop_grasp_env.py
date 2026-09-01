# ================================================================
# 双机协同抓取 + 搬运（集中式 DirectRLEnv）
#
# 两台空中机械臂各自飞到一根共享刚性杆的一端，做真实的摩擦抓取，
# 一旦两端都被夹住、且杆已离开台座抬起，就切换到协同搬运阶段，
# 把杆搬到一个空中目标并保持水平。
#
# 本环境完全自包含：自行搭建场景，只导入单机的 cfg 积木和串级
# LADRC 飞控用于复用。
#
# 复用的、已验证的组件：
#   · 飞控：cascade LADRC（num_agents=2），内环 ESO 带宽放缓
#     （wo=15/13）以匹配 50 Hz 更新，z2 钳制与 b0 联动，并带
#     力矩符号安全兜底。
#   · 重心前馈：臂姿态 / 负载会偏移合重心；预测的翻转力矩在 LADRC
#     力矩之上提前补偿，随后 commit，使 ESO 观测到的是实际施加的
#     力矩（避免重复补偿）。
#   · 来自单机任务的抓取栈：指尖 grasp_offset 瞄准、grasp_z_offset
#     （瞄上半部）、三阶段臂课程（settle → straighten → free）、
#     grasp_gate（臂伸直 × 夹爪竖直）门控、grasp_close 桥梁奖励、
#     触觉接触、门控在臂伸直上的 engagement/progress，以及镜像 revolute
#     夹爪。
#   · 搬运奖励（bar_goal / bar_level / hold_bonus / internal-force
#     代理）和搬运态（锁臂伸直 + 强制闭爪）。
#
# 本阶段特有考点：夹爪闭合轴必须垂直于杆轴。方块各向同性，杆不是 ——
# 两指必须横跨 4 cm 截面，而不是沿杆轴落下。一个对齐奖励
# (1 - |finger_axis · bar_axis|) 加上每机在机体系下的杆轴观测，让策略
# 学会偏航对正。
#
# 回合流程：
#   1. 杆（1.2 m / 150 g / 4×4 cm）横搭在两个台座上；每台无人机在自己
#      那端上方悬停出场。
#   2. 每机：settle（稳定悬停）→ 强制 straighten → 自由抓取（瞄自己那端
#      的上半部）。
#   3. 两机都夹住（双指接触）且杆已抬起 → 进入搬运态：锁臂、强制闭爪、
#      抬高飞行目标。
#   4. 杆在目标容差内且保持水平 1 s = 成功；滑脱则自动退出搬运态重抓。
# ================================================================

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_apply_inverse
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .flyarm_env import FlyarmEnvCfg
from .flyarm_vision_env import FlyarmVisionEnvCfg
from .cascade_ladrc import CascadeLADRC


# ──────────────────────────────────────────────────────────────
# 复用单机的 robot / pedestal cfg 积木
# ──────────────────────────────────────────────────────────────
_SINGLE = FlyarmEnvCfg()


def _cg_robot(prim_path: str, init_pos: tuple) -> ArticulationCfg:
    r = _SINGLE.robot.replace(prim_path=prim_path)
    # 出场模板必须摆在工作位姿：没有 init_state 时两个机器人模板会和
    #   杆/台座/地面一起叠在原点，物理预热步会因接触缓冲溢出而爆炸。
    r.init_state = ArticulationCfg.InitialStateCfg(pos=init_pos)
    r.spawn.activate_contact_sensors = True   # 触觉传感器需要
    return r


def _cg_pedestal(prim_path: str, init_pos: tuple) -> RigidObjectCfg:
    p = _SINGLE.pedestal.replace(prim_path=prim_path)
    p.init_state = RigidObjectCfg.InitialStateCfg(pos=init_pos)   # 把模板摆开，同样的原因
    return p


def _cg_contact(prim_path: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=prim_path,
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/Bar"],   # 只统计对杆的接触力
    )


class FlyarmCoopGraspEnvWindow(BaseEnvWindow):
    def __init__(self, env: "FlyarmCoopGraspEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# ──────────────────────────────────────────────────────────────
# 配置：继承 vision cfg（复用全部单机抓取 / LADRC / 臂 / 门控参数），
# 追加杆、两个台座、4 个接触传感器，以及搬运目标。
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmCoopGraspEnvCfg(FlyarmVisionEnvCfg):

    # === 基础（覆盖单机） ===
    episode_length_s = 18.0          # 抓取（课程 + 接近 + 下探）+ 搬运
    action_space = 18                # 2×9（每机：accel 3 + yaw-rate 1 + 臂 4 + 夹爪 1）
    observation_space = 75           # 2×32（单机 29 + 机体系下杆轴 3）+ 共享 11
    state_space = 0
    ui_window_class_type = FlyarmCoopGraspEnvWindow

    # === 场景 ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # === LADRC：50 Hz 下内环 ESO 带宽必须放缓 ===
    ladrc_in_wo_rp = 15.0
    ladrc_in_wo_yaw = 13.0
    ladrc_moment_sign = 1.0

    # === 重心前馈 ===
    use_cog_feedforward = True
    cog_ff_gain = 0.6                # 增益取大，因为臂完全伸直时重心偏移已大到
                                     #   0.3 顶不住静态翻转力矩；若 orientation 变差则翻符号
    # === 负载前馈：cog_ff 只覆盖机体+臂自重，不含被夹住的杆载 ===
    use_payload_feedforward = True
    payload_ff_gain = 1.0            # 力矩前馈增益；设 -1 翻符号，0 关闭
    payload_mass_per_drone = 0.075   # ≈ bar_mass/2，每机夹住杆后多承担的额外负载（kg）

    # === 杆（1.2 m / 150 g / 4×4 cm 截面 = 单机方块宽度，比 3.5 cm 闭合
    #     指间距宽，故能挤出夹紧力） ===
    bar_length = 1.2
    bar_thickness = 0.04
    bar_mass = 0.15
    bar_grasp_x = 0.5                # 抓取点距杆中心的距离（每端内缩 0.1）
    # 可选重杆课程：早期把杆做得很重以逼同时性——台座托着它，单台无人机
    #   既抬不起也扳不倒，从而消掉"先抓那台把杆扳歪、害了队友"的死锁。
    #   接触率离开零后把质量退火降回。
    use_bar_heavy_curriculum = False
    bar_heavy_mass = 10.0            # kg
    # 近抓取反向课程：从"两机已夹住台座托着的水平杆"起步，让协作奖励立刻
    #   触发，策略不必先发现完整接近过程就能学"一起夹住 + 抬"。按成功率把
    #   grasp_assist_level 退火到 0。
    use_grasp_assist = False
    grasp_assist_level = 1.0          # 退火 1 → 0（use_grasp_assist=False 时无效）
    # === 焊接-解锁课程脚手架（weld → break → free） ===
    #   阶段 0：杆焊在两机夹爪下方 → 无人机飞不走/扳不倒 → 先学"飞 + 协同把
    #            焊死的杆搬到目标"。阶段 1：弱断裂力焊接以逼真实闭合接触。
    #            阶段 2：无焊接 = 真实抓取（当前任务）。
    weld_phase = 0                    # 0 = 焊死（学飞）/ 1 = 弱焊 / 2 = 无焊（真抓取）
    attach_drop = 0.45               # 焊接点在机体下方深度（= hover_offset：机体在杆端上方）
    attach_mode = "fixed"            # "fixed" 刚性 / "ball" 球关节（需 jointFriction）
    attach_joint_friction = 1.0      # ball 模式的转动阻力（≈ 夹爪夹持力矩）；fixed 模式无效
    weld_break_force = -1.0          # 阶段 1 焊接断裂力（<0 = 不可断）
    assist_spawn_z = 0.67             # 机体起始高度 = origins.z + 该值（0.22 + hover_offset 0.45；
                                      #   臂伸直时夹爪正好够到杆端）
    bar: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,   # 穿透时快速弹出
                angular_damping=0.2,
                linear_damping=0.05,
                # 杆从下方被台座托着、比单块物块重约 10 倍，所以手指往下压时求解器
                #   没有余量、夹得过紧；多迭代几次让接触（和焊接）约束收敛、减少爆炸。
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            # contact_offset 0.01 更早生成接触点，减小可见穿透深度
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2), metallic=0.1),
            # 低摩擦让杆恢复"自定心"：自由方块在近指把它推向远指时会自定心。高 μ 下
            #   杆的横向阻力超过指力，微小偏移就变死锁；μ=1 保留约 6 倍夹持余量，同时
            #   让近指把杆推到位。
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.22)),   # 台座顶面 0.20 + 半厚 0.02
    )

    # === 两个台座（复用单机 25×25×0.20 宽台，杆每端下方各一个） ===
    pedestal_0: RigidObjectCfg = _cg_pedestal("/World/envs/env_.*/Pedestal_0", (-0.5, 0.0, 0.10))
    pedestal_1: RigidObjectCfg = _cg_pedestal("/World/envs/env_.*/Pedestal_1", (0.5, 0.0, 0.10))

    # === 两台机器人（模板摆在各自悬停位姿以匹配 reset，避免出场接触爆炸） ===
    robot_0: ArticulationCfg = _cg_robot("/World/envs/env_.*/Robot_0", (-0.5, 0.0, 0.67))
    robot_1: ArticulationCfg = _cg_robot("/World/envs/env_.*/Robot_1", (0.5, 0.0, 0.67))

    # === 4 个接触传感器（每机左/右指，全部 filter 到杆上） ===
    contact_0_left: ContactSensorCfg = _cg_contact("/World/envs/env_.*/Robot_0/gripper_left")
    contact_0_right: ContactSensorCfg = _cg_contact("/World/envs/env_.*/Robot_0/gripper_right")
    contact_1_left: ContactSensorCfg = _cg_contact("/World/envs/env_.*/Robot_1/gripper_left")
    contact_1_right: ContactSensorCfg = _cg_contact("/World/envs/env_.*/Robot_1/gripper_right")

    # === 夹爪闭合轴 ⊥ 杆轴（方块各向同性，杆不是） ===
    #   两指（3.7 cm 间距）必须横跨杆的 4 cm 截面，而不是沿杆轴排布。
    align_reward_scale = 3.0         # 1 - |finger_axis · bar_axis|
    # 真实夹持奖励（持续双指接触）压过"在该点闭合"：本阶段 contact_grasp 必须
    #   承担单机里 grasp_lift 在接触时承担的角色。
    contact_grasp_reward_scale = 8.0
    # 非对称双指接触判定：主指 > 0.2 N 且 弱指 > contact_weak_threshold。
    #   杆被外部约束时两指力可以合理地不等，要求两指都 > 0.2 N 则永不触发。
    #   接受"一指夹牢 + 另一指真碰到"以让抬升链点火；杆抬离后两指力自动均衡。
    contact_weak_threshold = 0.02
    # 指尖横向偏置：镜像两指对称于 pivot 中线（gripper_base 系下 x ≈ -0.0128），
    #   而非对称于 x = 0。真实指尖会合点在那里，所以 x 偏置据此设定（单机从未
    #   暴露这点，因为自由方块会自定心）。
    grasp_offset = (-0.0128, -0.020, -0.10)

    # === 搬运目标：两端都夹住且杆离开台座 → 把杆抬到初始位置 + carry_height，
    #     保持水平 ===
    carry_trigger = 0.01             # 杆抬起 1 cm 后切换到搬运目标（缩小死区）
    # carry_height：温柔抬 12 cm，让搬运目标的跳变不会把焊死/耦合的两机猛拽进
    #   约束爆炸；"一起抬 12 cm"这个里程碑仍演示了协同搬运。
    carry_height = 0.12
    # 接触去抖 + 搬运滞回：PhysX 接触力逐帧闪烁，"本帧两指都过阈"的门控会切断
    #   抬升奖励；改用"最近 contact_hold_steps 内出现过 both_contact"算作 held。
    #   搬运在 carry_trigger 进入、在 carry_release 退出（滞回），使阈值化的
    #   搬运标志不抖、不绕杆轴乱拽。
    contact_hold_steps = 5
    carry_release = 0.002
    # carry_ramp：搬运飞行目标 z 从 hover 到 hover+carry_height 的 ease-out
    #   斜坡（步数）。0 = 瞬跳（默认）。对于去中心化的 agent，瞬跳 +carry_height
    #   会让两机猛窜、把杆甩飞；斜坡让它们平滑跟踪逐渐上升的目标，而最终高度
    #   （搬运目标）不变。默认 0 → 集中式版本不受影响；MARL 子类在 __init__ 里
    #   设正值。
    carry_ramp_steps = 0
    success_dist = 0.10
    success_tilt = 0.05
    success_hold_steps = 50          # 在目标处保持 1 s = 成功
    # hold 课程（默认关，仅 MARL 子类启用）：hold_bonus 的容差在 hold_curric_steps
    #   内从宽松收紧到严格的成功阈值，所以早期容易拿（学"抬到目标并保持"），
    #   摆脱"夹住但不抬"的局部最优。上报的 success 指标始终用严格阈值，因此
    #   保持诚实。
    use_hold_curriculum = False
    hold_curric_steps = 120000        # 容差收紧到严格所跨的 env-steps
    hold_loose_dist = 0.20            # hold_bonus 起始的宽松距离容差（严格 = success_dist 0.10）
    hold_loose_tilt = 0.15            # hold_bonus 起始的宽松倾角容差（严格 = success_tilt 0.05）
    # lift_gate（默认关，仅 MARL 子类启用）：近目标的搬运奖励 bar_goal / hold_bonus
    #   只门控在 all_contact（两机都夹）上，不管杆是否真的抬起。静止的杆离目标
    #   只有 carry_height（0.12），所以这些项在杆躺台座上就发放——策略学会
    #   "夹住、别动"。乘上 lift_gate（= bar_rise / lift_height，离台进度 0→1）
    #   使它们在杆静止时为零、只在抬起后才发放。这只整形奖励——严格成功判定不动，
    #   因此不会虚抬指标——而抓取锚点（coop_grasp / contact）不被门控。
    lift_gate_transport_rewards = False

    # === 协同搬运奖励 + 内力代理（Kim V_K） ===
    bar_goal_reward_scale = 8.0
    bar_goal_alpha = 1.5
    bar_level_reward_scale = 4.0
    # 在抓取阶段（all_contact 之前）就奖励保持杆水平，打击先抓那台把杆扳歪
    #   而害了队友的行为。
    bar_level_pre_grasp_scale = 1.5
    hold_bonus_scale = 15.0
    internal_penalty_scale = -0.5
    # "两端夹住前别拖杆"：在尚未 all_contact 时惩罚杆的运动，让先抓那台稳住等待
    #   ——否则队友的抓取目标（实时杆端）就成了移动靶。all_contact 后无惩罚
    #   （协同搬运必须移动杆）。
    bar_still_penalty_scale = -0.5
    # 夹爪力：约 10 N 的指尖力会把薄指尖怼进杆里、与凸分解块互锁（托住 1.5 N
    #   的分担只需每边约 0.15 N）；下调上限让指尖不至扎穿，而 μ=1 摩擦仍留约
    #   8 倍夹持余量。
    gripper_effort_limit_2b = 0.3
    # 三级协同奖励阶梯（把单机 grasp_close 桥梁思路提到联合事件层）：
    #   coop_ready（两机指尖都到各自那端）→ coop_close（都到位且都在合爪）→
    #   coop_grasp（all_contact）。三项都在两机间共享（标准 MAPPO 做法）。
    coop_ready_reward_scale = 2.0
    coop_close_reward_scale = 3.0
    coop_grasp_reward_scale = 6.0
    coop_descend_scale = 0.0          # 保留、未用（=0）
    # 严格"到端"判定（避免"悬在杆上方"刷分）：3.5 cm 的球会把杆上方的空气也算进去；
    #   要求水平 < 2 cm 且 垂直 < 1.5 cm，使奖励只在真实抓取位姿处发放。
    at_end_horiz = 0.02
    at_end_vert = 0.015
    # touch 面包屑（1.5/s）：任一指碰到杆（× align × straighten 门控）补上
    #   "空中合爪"到"双指夹持"之间缺失的梯度台阶。
    touch_reward_scale = 1.5
    # 换端诊断开关：True = 无人机 0 抓 +x 端 / 无人机 1 抓 -x 端，出场、台座和镜像
    #   偏航跟随换端（用于区分"实例侧"还是"端侧"的失败）。
    swap_ends = False
    # 每机"把自己那端抬一点"台阶：单机在接触时学会了"抬"，但这里的抬升奖励门控在
    #   all_contact（一个稀有的联合事件）上，所以每机学会夹住却不知为何而夹。这
    #   恢复了一个小的每机梯度：自己双指夹住 且 自己那端离开台座（封顶 3 cm）→ 奖励。
    #   3 cm 封顶意味着杆只倾约 1.4°（扳不翻）；每机学会"夹 → 提"，两端都升起，
    #   all_contact 自然达成。
    own_end_lift_cap = 0.03
    own_end_lift_reward_scale = 30.0   # 与单机 grasp_lift = 30 对齐

    # === 机械死锁修复（一根强指把杆钉死，另一指永远够不到） ===
    #   一根强指把杆横向钉死时，第二指碰不到，_held 永不触发，all_contact 永远
    #   达不成。下面三个旋钮默认为旧行为，只被平行夹爪 cfg 启用：
    #     1. weak_grip 连续奖励 min(两指力) → 把二值的双指接触悬崖变成斜坡
    #        （第二指碰上就逐渐加分）
    #     2. touch_require_both → touch 要求两指（min > 阈值），去掉那个强化
    #        自稳的"单指钉杆"局部最优的补贴
    #     3.（在平行夹爪 cfg 里）降低夹爪力，让强指不把杆钉死、在合爪前能自定心
    weak_grip_reward_scale = 0.0       # 默认关（旧任务 / MARL 不变）；平行夹爪设约 3.0
    weak_grip_force_ref = 1.0          # N，弱指力的归一化参考
    touch_require_both = False         # 默认 = 旧的单指 touch；True = 要求两指

    # 速度阻尼 / 正则化为每机一份（继承自单机任务）。

    def __post_init__(self):
        # 跳过父类 __post_init__（避免触碰闲置的单机 robot 字段 / 相机 obs 逻辑）；
        #   obs/action 由本类定义。
        # PhysX GPU narrowphase patch 缓冲翻倍（两台凸分解无人机 + 杆 + 两个台座）。
        self.sim.physx.gpu_max_rigid_patch_count = 327680
        # 阶段 0/1（焊死）关掉杆的碰撞：焊接关节持住杆，但夹爪/脚架正好落在杆位置
        #   → 杆-夹爪接触会是过约束、破坏飞行稳定甚至引发约束爆炸。阶段 0/1 的抬升
        #   用 held_for_lift（非真实接触），所以不需要碰撞；阶段 2（真抓取）需要杆
        #   碰撞、重新开启。（filter 到无碰撞杆上的接触传感器只读 0，无报错。）
        self.bar.spawn.collision_props.collision_enabled = (self.weld_phase >= 2)


# ──────────────────────────────────────────────────────────────
# 主环境
# ──────────────────────────────────────────────────────────────
class FlyarmCoopGraspEnv(DirectRLEnv):
    """
    杆躺在两个台座上 → 每台无人机飞到一端做真实摩擦抓取 →
    两端都夹住 → 协同搬运到空中目标。

    每步循环（每机一份单机逻辑 + 一层共享协作层）：
      _pre_physics_step → 18 维 action → cascade_ladrc (A=2) + 重心前馈 → 推力/力矩；
                          臂三阶段课程；搬运态锁臂
      _apply_action     → 驱动两机（臂 PD / 夹爪二值 / 旋翼）
      _get_observations → 75 维
      _get_rewards      → 每机抓取栈（瞄自己那端）+ align + 共享（抬升/搬运/内力）
      _get_dones        → 任一机坠毁/发散，或杆掉落，或超时
      _reset_idx        → 杆放回台座，无人机到各自那端上方，臂蜷缩
    """

    cfg: FlyarmCoopGraspEnvCfg

    def __init__(self, cfg: FlyarmCoopGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device

        # ── Action / 飞控 ──
        self._actions = torch.zeros(N, self.cfg.action_space, device=dev)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(N, 2, 1, 3, device=dev)
        self._moment = torch.zeros(N, 2, 1, 3, device=dev)

        # ── 每机状态 ──
        self._arm_dof_targets = torch.zeros(N, 2, 4, device=dev)
        self._desired_pos_w = torch.zeros(N, 2, 3, device=dev)
        self._prev_ee_dist = torch.zeros(N, 2, device=dev)

        # ── 任务状态 ──
        self._bar_initial_pos = torch.zeros(N, 3, device=dev)    # 杆初始世界位置；搬运目标 = 它 + carry_height
        self._max_bar_height = torch.zeros(N, device=dev)
        self._carry = torch.zeros(N, dtype=torch.bool, device=dev)        # 搬运态（两端都夹住 + 已抬起）
        self._in_goal_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self._success = torch.zeros(N, dtype=torch.bool, device=dev)      # 搬到目标 + 保持 1 s
        self._contact_latch = torch.zeros(N, 2, dtype=torch.bool, device=dev)   # 每机，本回合是否曾夹住过（诊断）
        # 传感器诊断：本回合每机两指力之和的最大值（用于核实接触链路）。
        self._max_finger_force = torch.zeros(N, 2, device=dev)
        # 弱指诊断：本回合 min(left, right) 的最大值。恒 0 = 有一指从不碰杆
        #   （横向偏置 / 单指按压）；> 阈值 = 两指真夹住。
        self._max_weak_finger = torch.zeros(N, 2, device=dev)
        # 横向偏置探针：接触期间累加抓取中心相对杆轴的带符号横向偏移 →
        #   grip_lat_bias_i 读出系统性偏移（米）以调 grasp_offset.y。
        self._lat_bias_sum = torch.zeros(N, 2, device=dev)
        self._lat_cnt = torch.zeros(N, 2, device=dev)
        # 接触去抖：距上次 both_contact 的步数（999 = 本回合尚未夹住）；每步在
        #   _get_rewards 顶部更新。
        self._since_both = torch.full((N, 2), 999, dtype=torch.long, device=dev)
        # 诊断：本回合搬运态 / 瞬时 all_contact 的步数 → carry_frac /
        #   all_contact_frac（区分"接触从未对齐"和"对齐了但搬运没 latch"）。
        self._carry_cnt = torch.zeros(N, device=dev)
        self._allc_cnt = torch.zeros(N, device=dev)

        # ── 共享回合时钟（三阶段课程） ──
        self._steps_since_reset = torch.zeros(N, dtype=torch.long, device=dev)
        self._grasp_phase = torch.zeros(N, dtype=torch.bool, device=dev)

        self._robots = [self._robot_0, self._robot_1]
        self._contacts_l = [self._contact_0_left, self._contact_1_left]
        self._contacts_r = [self._contact_0_right, self._contact_1_right]
        self._end_sign = (1.0, -1.0) if cfg.swap_ends else (-1.0, 1.0)   # 每机抓哪一端（可换端做诊断）

        # ── Cascade LADRC（A=2，wo15/13，z2 与 b0 联动） ──
        self.ladrc = CascadeLADRC(
            N, num_agents=2, device=dev,
            in_wc_rp=cfg.ladrc_in_wc_rp, in_wo_rp=cfg.ladrc_in_wo_rp, in_b0_rp=cfg.ladrc_in_b0_rp,
            in_wc_yaw=cfg.ladrc_in_wc_yaw, in_wo_yaw=cfg.ladrc_in_wo_yaw, in_b0_yaw=cfg.ladrc_in_b0_yaw,
            out_wc=cfg.ladrc_out_wc, out_wo=cfg.ladrc_out_wo, out_b0=cfg.ladrc_out_b0,
            z2_clamp_in_rp=0.3 * cfg.ladrc_in_b0_rp,
            z2_clamp_in_yaw=0.2 * cfg.ladrc_in_b0_yaw,
            z2_clamp_out=1.0,
            max_angle=cfg.ladrc_max_angle, max_rate=cfg.ladrc_max_rate,
        )

        # ── body/joint 索引（两机共用一个 USD） ──
        self._body_ids, self._rotor_ids, self._arm_ids = [], [], []
        self._ee_ids, self._grip_ids, self._lf_ids, self._rf_ids = [], [], [], []
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

        # ── 臂限位（读 URDF + 把 base_rotate 收紧到 center ± clamp） ──
        if hasattr(self._robot_0.data, "soft_joint_pos_limits"):
            soft = self._robot_0.data.soft_joint_pos_limits[0]
        else:
            soft = self._robot_0.data.joint_pos_limits[0]
        aids0 = self._arm_ids[0]
        self._arm_lower = soft[aids0, 0].clone()
        self._arm_upper = soft[aids0, 1].clone()
        brc, br = cfg.base_rotate_center, cfg.base_rotate_clamp
        self._arm_lower[0] = torch.clamp(self._arm_lower[0], min=brc - br)
        self._arm_upper[0] = torch.clamp(self._arm_upper[0], max=brc + br)

        # ── 常量张量 ──
        self._gripper_open_t = torch.tensor(cfg.gripper_open, device=dev)
        self._gripper_close_t = torch.tensor(cfg.gripper_close, device=dev)
        self._arm_stow_t = torch.tensor(cfg.arm_stow, device=dev)
        self._arm_straight_target_t = torch.tensor(cfg.arm_straight_target, device=dev)
        self._arm_straight_pose_t = torch.tensor(
            [cfg.base_rotate_center, cfg.arm_straight_target[0], cfg.arm_straight_target[1], cfg.arm_stow[3]],
            device=dev,
        )
        self._grasp_offset_t = torch.tensor(cfg.grasp_offset, device=dev)

        # ── 下调夹爪力上限，避免钉死 / 过度穿透杆（经 sim API 写入） ──
        try:
            eff = torch.full((N, 2), float(self.cfg.gripper_effort_limit_2b), device=dev)
            for i, rb in enumerate(self._robots):
                rb.write_joint_effort_limit_to_sim(eff, joint_ids=self._grip_ids[i])
            print(f"[coop-grasp] gripper effort limit set to {self.cfg.gripper_effort_limit_2b}")
        except Exception as e:
            print(f"[coop-grasp] gripper effort write skipped (API mismatch): {e}\n  -> fallback: set effort in URDF and re-convert USD.")

        # ── 质量 / 重力 + 各 link 质量（用于重心前馈，两机共用 URDF，取无人机 0） ──
        self._robot_mass = self._robot_0.root_physx_view.get_masses()[0].sum()
        self._body_masses = self._robot_0.root_physx_view.get_masses()[0].to(dev)   # (nb,)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=dev).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ── 奖励日志：每机一套（_0 / _1）+ 共享 ──
        self._per_keys = [
            "dist_to_goal", "orientation", "reaching", "finger", "gripper_vertical", "arm_straight",
            "align", "grasp_close", "contact_grasp", "own_end_lift", "touch", "weak_grip", "engagement", "progress",
            "lin_vel", "ang_vel", "action_rate", "joint_vel", "crash",
        ]
        shared_keys = ["grasp_lift", "lift_goal", "lifting", "bar_goal", "bar_level", "hold_bonus", "internal", "bar_still",
                       "coop_ready", "coop_close", "coop_grasp", "coop_descend"]
        sum_keys = [k + "_0" for k in self._per_keys] + [k + "_1" for k in self._per_keys] + shared_keys
        self._episode_sums = {k: torch.zeros(N, dtype=torch.float, device=dev) for k in sum_keys}

        print(f"[coop-grasp] real grasp + transport: bar {cfg.bar_length}m @ table 0.20, grasp point ±{cfg.bar_grasp_x}, "
              f"obs={cfg.observation_space}, act={cfg.action_space}, cog_ff={cfg.cog_ff_gain if cfg.use_cog_feedforward else 0}")
        self.set_debug_vis(self.cfg.debug_vis)

    # ==============================================================
    # 场景：两台机器人 + 杆 + 两个台座 + 4 个接触传感器（无关节——真实物理抓取）
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

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── 辅助函数 ──
    def _finger_mag(self, sensor: ContactSensor) -> torch.Tensor:
        fmat = sensor.data.force_matrix_w
        if fmat is None:
            fmat = sensor.data.net_forces_w
        return torch.linalg.norm(fmat.reshape(self.num_envs, -1, 3), dim=-1).sum(dim=-1)

    def _both_contact(self, i: int) -> torch.Tensor:
        """(N,) 无人机 i 有效地夹住杆：主指 > 0.2 N 且 弱指 > 0.02 N（非对称）。"""
        fl = self._finger_mag(self._contacts_l[i])
        fr = self._finger_mag(self._contacts_r[i])
        strong = torch.maximum(fl, fr)
        weak = torch.minimum(fl, fr)
        return (strong > self.cfg.contact_force_threshold) & (weak > self.cfg.contact_weak_threshold)

    def _held(self, i: int) -> torch.Tensor:
        """去抖后的"夹持中"：最近 contact_hold_steps 内出现过 both_contact（平滑接触力闪烁）。"""
        return self._since_both[:, i] <= self.cfg.contact_hold_steps

    def _grasp_center(self, i: int) -> torch.Tensor:
        """无人机 i 的指尖会合点（腕位姿 + grasp_offset，单机标定的几何）。"""
        rb = self._robots[i]
        wp = rb.data.body_pos_w[:, self._ee_ids[i][0], :]
        wq = rb.data.body_quat_w[:, self._ee_ids[i][0], :]
        return wp + quat_apply(wq, self._grasp_offset_t.unsqueeze(0).expand(self.num_envs, 3))

    def _bar_end_w(self, i: int) -> torch.Tensor:
        """无人机 i 那端的抓取点（杆系 ±bar_grasp_x → 世界系）。"""
        e = torch.zeros(self.num_envs, 3, device=self.device)
        e[:, 0] = self._end_sign[i] * self.cfg.bar_grasp_x
        return self._bar.data.root_pos_w + quat_apply(self._bar.data.root_quat_w, e)

    def _grasp_target_w(self, i: int) -> torch.Tensor:
        """无人机 i 的抓取瞄准点 = 它那端杆端 + grasp_z_offset（瞄上半部）。"""
        t = self._bar_end_w(i)
        t[:, 2] = t[:, 2] + self.cfg.grasp_z_offset
        return t

    def _bar_axis_w(self) -> torch.Tensor:
        ex = torch.zeros(self.num_envs, 3, device=self.device); ex[:, 0] = 1.0
        return quat_apply(self._bar.data.root_quat_w, ex)

    def _at_end_strict(self, i: int) -> torch.Tensor:
        """无人机 i 指尖真的在它的抓取位姿（水平 < 2 cm 且 垂直 < 1.5 cm；不是 3.5 cm 的球）。"""
        d = self._grasp_center(i) - self._grasp_target_w(i)
        return (torch.linalg.norm(d[:, :2], dim=1) < self.cfg.at_end_horiz) & (d[:, 2].abs() < self.cfg.at_end_vert)

    # ==============================================================
    # Action：18 维 → cascade_ladrc（两机）+ 重心前馈；臂三阶段课程；搬运态锁臂
    # ==============================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._steps_since_reset += 1
        N = self.num_envs
        a0, a1 = self._actions[:, 0:9], self._actions[:, 9:18]

        # ── 飞控：cascade LADRC（两机 stack 在一起、一次算完） ──
        quat = torch.stack([rb.data.root_quat_w for rb in self._robots], dim=1)
        omega = torch.stack([rb.data.root_ang_vel_b for rb in self._robots], dim=1)
        grav_b = torch.stack([rb.data.projected_gravity_b for rb in self._robots], dim=1)
        g = self._gravity_magnitude
        a_des = torch.stack([a0[:, 0:3], a1[:, 0:3]], dim=1) * self.cfg.accel_scale
        yaw_rate = torch.stack([a0[:, 3], a1[:, 3]], dim=1) * self.cfg.yaw_rate_scale

        self._thrust[:, :, 0, 2] = self.ladrc.accel_to_thrust(
            a_des, quat, self._robot_mass, g, self.cfg.thrust_to_weight * self._robot_weight)
        roll_des, pitch_des = self.ladrc.accel_to_attitude_setpoint(a_des, quat, g, self.cfg.ladrc_max_angle)
        pitch_meas = grav_b[:, :, 0:1]
        roll_meas = -grav_b[:, :, 1:2]
        h = self.cfg.sim.dt * self.cfg.decimation
        p_sp, q_sp = self.ladrc.outer_step(roll_des, pitch_des, roll_meas, pitch_meas, h)
        moment = self.ladrc.inner_step(p_sp, q_sp, yaw_rate.unsqueeze(-1), omega, h)
        moment = moment * self.cfg.ladrc_moment_sign

        # ── 重心前馈：合重心水平偏移 → 预测翻转力矩 → 提前补偿；
        #    commit 在相加之后做，使 ESO 看到实际施加的力矩 ──
        if self.cfg.use_cog_feedforward and self.cfg.cog_ff_gain > 0.0:
            com_off = []
            for k, rb in enumerate(self._robots):
                body_pos_w = rb.data.body_pos_w                                            # (N,nb,3)
                com_w = (body_pos_w * self._body_masses.view(1, -1, 1)).sum(dim=1) / self._body_masses.sum()
                com_off.append(quat_apply_inverse(rb.data.root_quat_w, com_w - rb.data.root_pos_w))
            com_off_b = torch.stack(com_off, dim=1)                                        # (N,2,3)
            ff = self.ladrc.cog_feedforward(com_off_b, self._thrust[:, :, 0, 2:3], self.cfg.cog_ff_gain)
            moment = moment + ff   # 若 orientation 变差则翻符号（cfg cog_ff_gain）

        self._moment[:, :, 0, :] = moment
        self.ladrc.commit_inner_applied(moment)   # ESO 观测到含前馈的力矩

        # ── 臂：增量控制 + 三阶段课程 ──
        dt = self.cfg.sim.dt * self.cfg.decimation
        arm_delta = torch.stack([a0[:, 4:8], a1[:, 4:8]], dim=1) * self.cfg.arm_speed_scale * dt
        self._arm_dof_targets = torch.clamp(self._arm_dof_targets + arm_delta, self._arm_lower, self._arm_upper)

        steps = self._steps_since_reset
        settle, straighten = self.cfg.settle_steps, self.cfg.straighten_steps
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

        # ── 搬运态：两机都夹住 且 杆已抬起 → 锁臂伸直 ──
        #   高度-与-接触双条件；真正滑脱（>0.1 s 无接触，或杆回到台面）会自动退出
        #   搬运态。去抖接触 + 抬升滞回（进入 1 cm / 退出 0.2 cm）防止搬运标志
        #   抖动、绕杆乱拽。
        bar_z = self._bar.data.root_pos_w[:, 2]
        all_held = self._held(0) & self._held(1)
        rise = bar_z - self._bar_initial_pos[:, 2]
        rise_gate = torch.where(self._carry, rise > self.cfg.carry_release, rise > self.cfg.carry_trigger)
        self._carry = all_held & rise_gate
        carry_arm = self._carry.view(N, 1, 1).expand(N, 2, 4)
        self._arm_dof_targets = torch.where(carry_arm, straight_pose, self._arm_dof_targets)

    # ==============================================================
    # 执行：用 推力/力矩、臂、夹爪、旋翼 驱动每台无人机
    # ==============================================================
    def _apply_action(self):
        for i, rb in enumerate(self._robots):
            rb.set_external_force_and_torque(self._thrust[:, i], self._moment[:, i], body_ids=self._body_ids[i])
            rb.set_joint_position_target(self._arm_dof_targets[:, i], joint_ids=self._arm_ids[i])

            gcmd = self._actions[:, 8] if i == 0 else self._actions[:, 17]
            sel_close = (gcmd >= 0.0).float().unsqueeze(-1)
            # 搬运态强制闭爪（搬运中别松手）；滑脱时自动把控制权交回策略。
            sel_close = torch.maximum(sel_close, self._carry.float().unsqueeze(-1))
            gtarget = sel_close * self._gripper_close_t + (1.0 - sel_close) * self._gripper_open_t
            rb.set_joint_position_target(gtarget, joint_ids=self._grip_ids[i])

            tn = (self._thrust[:, i, 0, 2] / (self.cfg.thrust_to_weight * self._robot_weight)).clamp(0.0, 1.0)
            spd = 50.0
            rs = torch.zeros(self.num_envs, 4, device=self.device)
            rs[:, 0] = spd * (0.5 + tn); rs[:, 1] = -spd * (0.5 + tn)
            rs[:, 2] = -spd * (0.5 + tn); rs[:, 3] = spd * (0.5 + tn)
            rb.set_joint_velocity_target(rs, joint_ids=self._rotor_ids[i])

    # ==============================================================
    # 观测：每机 32（单机 29 + 机体系杆轴 3），共享 11 → 75
    # ==============================================================
    def _drone_obs(self, i: int) -> torch.Tensor:
        rb = self._robots[i]
        drone_pos_w = rb.data.root_pos_w
        drone_quat_w = rb.data.root_quat_w
        end_w = self._bar_end_w(i)

        # 飞行目标：自己那端上方；搬运态 → 目标端上方（杆初始位置 + carry_height）
        des = end_w.clone()
        # z 钉在杆初始高度 + hover（不是实时端 z）：xy 钉在实时端保持水平跟踪，
        #   但 z 钉在实时端会让"抬我那端 → 我的目标升高"成为自激跷跷板。抬升的
        #   牵引由 carry 后的 goal_end 统一处理。
        des[:, 2] = self._bar_initial_pos[:, 2] + self.cfg.hover_offset
        goal_end = self._bar_initial_pos.clone()
        goal_end[:, 0] = goal_end[:, 0] + self._end_sign[i] * self.cfg.bar_grasp_x
        goal_end[:, 2] = goal_end[:, 2] + self.cfg.carry_height + self.cfg.hover_offset
        des = torch.where(self._carry.unsqueeze(-1), goal_end, des)
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

        # 机体系下的杆轴（用于对齐）。朝杆中心定向（× -end_sign），使两机在各自机体系
        #   里都看到 +x = egocentric 对称（world +x 轴对 yaw-180 那台会读成机体系 -x，
        #   破坏共享策略的对齐学习）。
        bar_axis_ego = -self._end_sign[i] * self._bar_axis_w()
        axis_end_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, drone_pos_w + bar_axis_ego)
        bar_axis_b = axis_end_b

        base = torch.cat([
            rb.data.root_lin_vel_b, rb.data.root_ang_vel_b, rb.data.projected_gravity_b,
            des_b, arm_n, arm_vel, end_b, to_t_dir_b, to_t_d, bar_axis_b,
        ], dim=-1)   # 3+3+3+3+4+4+3+3+1+3 = 30
        base = torch.clamp(base, -5.0, 5.0)

        fl = torch.tanh(0.1 * self._finger_mag(self._contacts_l[i]))
        fr = torch.tanh(0.1 * self._finger_mag(self._contacts_r[i]))
        tactile = torch.stack([fl, fr], dim=-1)
        return torch.cat([base, tactile], dim=-1)   # 32

    def _get_observations(self) -> dict:
        per = [self._drone_obs(0), self._drone_obs(1)]
        ez = torch.zeros(self.num_envs, 3, device=self.device); ez[:, 2] = 1.0
        bar_zaxis_w = quat_apply(self._bar.data.root_quat_w, ez)
        rel_pos = self._robots[1].data.root_pos_w - self._robots[0].data.root_pos_w
        contacts = torch.stack([self._both_contact(0).float(), self._both_contact(1).float()], dim=-1)
        shared = torch.cat([
            self._bar.data.root_lin_vel_w * 0.5,    # 3
            bar_zaxis_w,                            # 3
            rel_pos,                                # 3
            contacts,                               # 2 （各机是否在夹——协同等待信号）
        ], dim=-1)
        shared = torch.clamp(shared, -5.0, 5.0)
        obs = torch.cat(per + [shared], dim=-1)     # 32×2 + 11 = 75
        return {"policy": obs}

    # ==============================================================
    # 奖励：每机抓取栈（瞄自己那端）+ align + 共享（抬升/搬运/内力）
    # ==============================================================
    def _machine_reward(self, i: int):
        cfg = self.cfg
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

        # 1 飞到目标（自己那端上方 / 搬运目标）
        dist_to_goal = torch.linalg.norm(self._desired_pos_w[:, i] - drone_pos_w, dim=1)
        dist_reward = (1.0 / (1.0 + dist_to_goal ** 2)) ** 2
        # 2 姿态
        ori_err = torch.sum(torch.square(rb.data.projected_gravity_b[:, :2]), dim=1)
        # 3 reaching（瞄自己那端的上半部）
        ee_d = torch.linalg.norm(grasp_center - self._grasp_target_w(i), dim=1)
        reaching = 1.0 - torch.tanh(ee_d / 0.2)
        # 4 finger 时序（远处闭爪 = 扣分，近处闭爪 = 加分）
        is_far = (ee_d > 0.08).float(); is_near = (ee_d < 0.05).float()
        finger = -is_far * gclosed + is_near * gclosed
        # 5 夹爪竖直 + 臂伸直（门控来源）
        v = finger_mid - wrist_pos
        gvert = torch.clamp(-v[:, 2] / (torch.linalg.norm(v, dim=1) + 1e-6), min=0.0)
        arm_bc = rb.data.joint_pos[:, [aids[1], aids[2]]]
        arm_straight = 1.0 - torch.tanh(torch.linalg.norm(arm_bc - self._arm_straight_target_t, dim=1) / 1.0)
        straight_gate = ((arm_straight - 0.4) / 0.3).clamp(0.0, 1.0)
        vertical_gate = ((gvert - 0.5) / 0.3).clamp(0.0, 1.0)
        grasp_gate = straight_gate * vertical_gate
        phase3 = self._grasp_phase.float()

        # 6 对齐：指轴（过两指的直线）⊥ 杆轴 → 两指能横跨截面。1 - |cos| ∈ [0, 1]。
        finger_axis = rf - lf
        finger_axis = finger_axis / (torch.linalg.norm(finger_axis, dim=1, keepdim=True) + 1e-6)
        align = 1.0 - torch.abs(torch.sum(finger_axis * self._bar_axis_w(), dim=1))
        align_r = align * cfg.align_reward_scale * dt * phase3
        # align_gate：未对正则不给合爪/按压奖励（偏 > 约 60° 关闭）→ 防止
        #   "不对正就单指按压"的局部最优。真实横跨时 align ≥ 0.85。
        align_gate = ((align - 0.5) / 0.3).clamp(0.0, 1.0)

        # 7 抓取链（单机奖励 + 臂伸直门控）
        # 传感器诊断：最大力之和 + 最大弱指（min）；both_contact 在此计算以省一次传感器读取。
        fl = self._finger_mag(self._contacts_l[i])
        fr = self._finger_mag(self._contacts_r[i])
        self._max_finger_force[:, i] = torch.maximum(self._max_finger_force[:, i], fl + fr)
        self._max_weak_finger[:, i] = torch.maximum(self._max_weak_finger[:, i], torch.minimum(fl, fr))
        # 非对称双指接触判定（主 > 0.2，弱 > 0.02），与 _both_contact 一致
        both_contact = ((torch.maximum(fl, fr) > cfg.contact_force_threshold)
                        & (torch.minimum(fl, fr) > cfg.contact_weak_threshold)).float()
        contact_r = both_contact * cfg.contact_grasp_reward_scale * dt * grasp_gate * align_gate
        # 每机自己那端抬升台阶：门控在夹持上，封顶 3 cm → 恢复单机"接触时抬"的
        #   课程。用去抖后的 hold，使这课不被接触力闪烁切断。
        end_rise = (self._bar_end_w(i)[:, 2] - self._bar_initial_pos[:, 2]).clamp(min=0.0, max=cfg.own_end_lift_cap)
        # 门控在"两机都夹住"上：单台独自抬会把杆扳歪、队友夹不住，all_contact 保持
        #   0，否则 own_end_lift 反而强化扳歪。两机都夹住 → 先抓那台稳住等待 →
        #   两机都夹 → 一起抬。（抓取仍按每机激励。）
        both_held = self._held(i).float() * self._held(1 - i).float()
        own_lift_r = end_rise * both_held * cfg.own_end_lift_reward_scale * dt
        progress = (self._prev_ee_dist[:, i] - ee_d).clamp(min=0.0, max=0.02)
        progress_r = progress * cfg.progress_reward_scale * phase3 * straight_gate
        self._prev_ee_dist[:, i] = ee_d.detach()
        engagement_r = (ee_d < 0.03).float() * cfg.engagement_reward_scale * dt * phase3 * straight_gate
        at_bar = self._at_end_strict(i).float()   # 严格到端判定（杀"悬停-合爪"刷分）
        grasp_close_r = at_bar * gclosed * cfg.grasp_close_reward_scale * dt * grasp_gate * align_gate
        # touch 面包屑：任一指碰到杆（align + straighten 门控），补上"空中合爪"到
        #   "双指夹持"之间的台阶。
        # touch_require_both：默认 False = 单指（max > thr）；True（平行夹爪）= 两指（min > thr）
        #   → 去掉"单指按压"局部最优的补贴。
        touch_finger = torch.minimum(fl, fr) if cfg.touch_require_both else torch.maximum(fl, fr)
        any_touch = (touch_finger > cfg.contact_force_threshold).float()
        touch_r = any_touch * cfg.touch_reward_scale * dt * align_gate * straight_gate
        # 分级弱指奖励：连续的 min(指力)/ref → 把二值双指接触悬崖变成斜坡（"啮合第二指"
        #   的梯度）。门控 align × straighten 使它不靠空中捏合拿分。默认 scale = 0 →
        #   旧任务不受影响。
        weak_grip_r = (torch.minimum(fl, fr) / cfg.weak_grip_force_ref).clamp(max=1.0) \
            * cfg.weak_grip_reward_scale * dt * align_gate * straight_gate
        # 横向偏置探针：接触期间，抓取中心相对杆轴的带符号横向偏移。
        axis_w = self._bar_axis_w()
        lat_dir = torch.stack([-axis_w[:, 1], axis_w[:, 0], torch.zeros_like(axis_w[:, 0])], dim=1)
        lat_dir = lat_dir / (torch.linalg.norm(lat_dir, dim=1, keepdim=True) + 1e-6)
        lat_off = torch.sum((grasp_center - self._bar_end_w(i)) * lat_dir, dim=1)
        self._lat_bias_sum[:, i] += lat_off * any_touch
        self._lat_cnt[:, i] += any_touch

        # 8 正则化 / 安全
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
            "weak_grip": weak_grip_r,   # 连续弱指奖励（默认 scale=0 → 旧任务关；平行夹爪启用）
            "engagement": engagement_r,
            "progress": progress_r,
            "lin_vel": lin_v * cfg.lin_vel_penalty_scale * dt,
            "ang_vel": ang_v * cfg.ang_vel_penalty_scale * dt,
            "action_rate": action_rate * cfg.action_rate_penalty_scale * dt,
            "joint_vel": jv * cfg.joint_vel_penalty_scale * dt,
            "crash": crash_int * cfg.crash_penalty * dt,
        }
        total = torch.sum(torch.stack(list(comp.values())), dim=0)
        return total, comp, grasp_gate

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        dt = self.step_dt
        # 去抖计数器更新（每步一次，在 _machine_reward / 共享块之前）：碰到 → 清零，没碰 → +1
        for i in range(2):
            raw_i = self._both_contact(i)
            self._since_both[:, i] = torch.where(
                raw_i, torch.zeros_like(self._since_both[:, i]), self._since_both[:, i] + 1)
        t0, c0, gate0 = self._machine_reward(0)
        t1, c1, gate1 = self._machine_reward(1)
        for k, v in c0.items():
            self._episode_sums[k + "_0"] += v
        for k, v in c1.items():
            self._episode_sums[k + "_1"] += v

        # ── 共享：抬升链（单机 grasp_lift/lift_goal/lifting），门控在两机都夹住上 ──
        bar_pos = self._bar.data.root_pos_w
        bar_z = bar_pos[:, 2]
        self._max_bar_height = torch.maximum(self._max_bar_height, bar_z)
        # 门控用去抖后的"held"（瞬时版仅作诊断）：闪烁的接触力会逐帧切断抬升链，
        #   使大奖励实际消失。计数器刚在上面更新过 → _since_both == 0 表示本步瞬时 both_contact。
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

        # ── 共享：搬运奖励，门控在 all_contact 上（不夹住不发放） ──
        bar_goal = self._bar_initial_pos.clone()
        bar_goal[:, 2] = bar_goal[:, 2] + cfg.carry_height
        bar_goal_dist = torch.linalg.norm(bar_goal - bar_pos, dim=1)
        bar_goal_r = torch.exp(-cfg.bar_goal_alpha * bar_goal_dist) * cfg.bar_goal_reward_scale * dt * all_contact
        bar_tilt = torch.sum(torch.square(self._bar.data.root_quat_w[:, 1:3]), dim=1)
        bar_level_r = torch.exp(-bar_tilt / 0.05) * cfg.bar_level_reward_scale * dt * all_contact
        in_goal = (bar_goal_dist < cfg.success_dist) & (bar_tilt < cfg.success_tilt) & (all_contact > 0.5)
        hold_r = in_goal.float() * cfg.hold_bonus_scale * dt
        self._in_goal_steps = (self._in_goal_steps + 1) * in_goal.long()
        self._success = self._success | (self._in_goal_steps >= cfg.success_hold_steps)

        # ── 共享：内力代理（Kim V_K）：夹住后惩罚两机沿杆轴对拉 ──
        v_rel = self._robots[0].data.root_lin_vel_w - self._robots[1].data.root_lin_vel_w
        v_k = torch.abs(torch.sum(v_rel * self._bar_axis_w(), dim=1)).clamp(max=10.0)
        internal_r = v_k * cfg.internal_penalty_scale * dt * all_contact

        # ── "两端夹住前别拖杆"：未 all_contact 时惩罚杆运动，让先抓那台稳住等待；
        #    队友的抓取目标（杆端）不再是移动靶。all_contact 后无惩罚（协同搬运
        #    必须移动杆）。 ──
        bar_speed = torch.linalg.norm(self._bar.data.root_lin_vel_w, dim=1) \
            + 0.5 * torch.linalg.norm(self._bar.data.root_ang_vel_w, dim=1)
        bar_still_r = bar_speed.clamp(max=5.0) * cfg.bar_still_penalty_scale * dt * (1.0 - all_contact)

        # ── 协同三级阶梯：都到位 → 都合爪 → 都夹住（瞄准同时性） ──
        # "到位"用严格到端判定（水平 2 cm + 垂直 1.5 cm），不算杆上方的空气。
        both_ready = (self._at_end_strict(0) & self._at_end_strict(1)).float()
        both_closing = ((self._actions[:, 8] >= 0.0) & (self._actions[:, 17] >= 0.0)).float()
        coop_ready_r = both_ready * cfg.coop_ready_reward_scale * dt
        coop_close_r = both_ready * both_closing * cfg.coop_close_reward_scale * dt
        coop_grasp_r = all_contact * cfg.coop_grasp_reward_scale * dt
        # 弱环下探：min(两机 reaching) → 奖励把落后那台也拉到杆端。phase3 门控。
        reach0 = 1.0 - torch.tanh(torch.linalg.norm(self._grasp_center(0) - self._grasp_target_w(0), dim=1) / 0.2)
        reach1 = 1.0 - torch.tanh(torch.linalg.norm(self._grasp_center(1) - self._grasp_target_w(1), dim=1) / 0.2)
        coop_descend_r = torch.minimum(reach0, reach1) * self._grasp_phase.float() * cfg.coop_descend_scale * dt

        shared = {
            "grasp_lift": grasp_lift_r, "lift_goal": lift_goal_r, "lifting": lifting_r,
            "bar_goal": bar_goal_r, "bar_level": bar_level_r, "hold_bonus": hold_r, "internal": internal_r,
            "bar_still": bar_still_r,
            "coop_ready": coop_ready_r, "coop_close": coop_close_r, "coop_grasp": coop_grasp_r,
            "coop_descend": coop_descend_r,
        }
        for k, v in shared.items():
            self._episode_sums[k] += v
        return t0 + t1 + torch.sum(torch.stack(list(shared.values())), dim=0)

    # ==============================================================
    # 终止：任一机坠毁/过高/发散，或杆掉落，或超时
    # ==============================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, rb in enumerate(self._robots):
            z = rb.data.root_pos_w[:, 2]
            died = died | (z < 0.12) | (z > 5.0)
            died = died | (torch.linalg.norm(rb.data.root_lin_vel_w, dim=1) > 30.0)
            died = died | (torch.linalg.norm(self._desired_pos_w[:, i] - rb.data.root_pos_w, dim=1) > 5.0)
        # 杆被碰下台（与单机任务相同的掉落阈值语义）
        died = died | (self._bar.data.root_pos_w[:, 2] < (self.cfg.pedestal_height - 0.10))
        return died, time_out

    # ==============================================================
    # Reset：杆放回两个台座，无人机到各自那端上方，臂蜷缩
    # ==============================================================
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot_0._ALL_INDICES

        # ── 日志 ──
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/max_bar_height"] = self._max_bar_height[env_ids].mean().item()
        extras["Metrics/bar_height"] = self._bar.data.root_pos_w[env_ids, 2].mean().item()
        extras["Metrics/success_rate"] = self._success[env_ids].float().mean().item()          # 搬到目标 + 保持 1 s
        extras["Metrics/contact_rate_0"] = self._contact_latch[env_ids, 0].float().mean().item()   # 本回合是否曾夹住
        extras["Metrics/contact_rate_1"] = self._contact_latch[env_ids, 1].float().mean().item()
        extras["Metrics/max_finger_force_0"] = self._max_finger_force[env_ids, 0].mean().item()
        extras["Metrics/max_finger_force_1"] = self._max_finger_force[env_ids, 1].mean().item()
        extras["Metrics/max_weak_finger_0"] = self._max_weak_finger[env_ids, 0].mean().item()      # 0 = 有一指从不碰杆
        extras["Metrics/max_weak_finger_1"] = self._max_weak_finger[env_ids, 1].mean().item()
        for i in range(2):
            gc = self._grasp_center(i)[env_ids]
            tgt = self._grasp_target_w(i)[env_ids]
            extras[f"Metrics/final_ee_to_end_{i}"] = torch.linalg.norm(gc - tgt, dim=1).mean().item()
            # 横向偏置读数（m，带符号）：稳定的非零值表示存在 grasp_offset.y 偏置。
            extras[f"Metrics/grip_lat_bias_{i}"] = (self._lat_bias_sum[env_ids, i].sum()
                                                    / (self._lat_cnt[env_ids, i].sum() + 1e-6)).item()
        # 搬运 / 瞬时 all_contact 的回合占比：all_contact_frac 高但 carry_frac 低 =
        #   抬升 gate 未越过；两者都低 = 四指接触从未对齐；carry_frac 上升 = 搬运已 latch。
        steps_f = self._steps_since_reset[env_ids].clamp(min=1).float()
        extras["Metrics/carry_frac"] = (self._carry_cnt[env_ids] / steps_f).mean().item()
        extras["Metrics/all_contact_frac"] = (self._allc_cnt[env_ids] / steps_f).mean().item()
        self.extras["log"].update(extras)

        # ── 骨架 reset ──
        self._robot_0.reset(env_ids)
        self._robot_1.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._carry[env_ids] = False
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

        # ── 杆：横搭在两个台座上（居中、沿 x、微小随机化） ──
        bar_state = self._bar.data.default_root_state[env_ids].clone()
        bar_state[:, 0] = origins[:, 0] + torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        bar_state[:, 1] = origins[:, 1] + torch.zeros(n, device=self.device).uniform_(-0.003, 0.003)
        bar_state[:, 2] = origins[:, 2] + 0.22          # 台面 0.20 + 半厚 0.02
        bar_state[:, 3] = 1.0; bar_state[:, 4:7] = 0.0; bar_state[:, 7:] = 0.0
        self._bar.write_root_pose_to_sim(bar_state[:, :7], env_ids)
        self._bar.write_root_velocity_to_sim(bar_state[:, 7:], env_ids)
        self._bar_initial_pos[env_ids, 0] = origins[:, 0]
        self._bar_initial_pos[env_ids, 1] = origins[:, 1]
        self._bar_initial_pos[env_ids, 2] = bar_state[:, 2]
        self._max_bar_height[env_ids] = bar_state[:, 2]

        # ── 两个台座：各在一端的抓取点下方 ──
        for i, ped in enumerate([self._pedestal_0, self._pedestal_1]):
            ps = ped.data.default_root_state[env_ids].clone()
            ps[:, 0] = origins[:, 0] + self._end_sign[i] * self.cfg.bar_grasp_x
            ps[:, 1] = origins[:, 1]
            ps[:, 2] = origins[:, 2] + self.cfg.pedestal_height / 2.0
            ps[:, 3] = 1.0; ps[:, 4:7] = 0.0; ps[:, 7:] = 0.0
            ped.write_root_pose_to_sim(ps[:, :7], env_ids)
            ped.write_root_velocity_to_sim(ps[:, 7:], env_ids)

        # ── 两台无人机：在各端上方悬停，臂蜷缩 + 扰动，夹爪闭合（杆上方 0.45、不重叠） ──
        for i, rb in enumerate(self._robots):
            rs = rb.data.default_root_state[env_ids].clone()
            rs[:, 0] = origins[:, 0] + self._end_sign[i] * self.cfg.bar_grasp_x
            rs[:, 1] = origins[:, 1]
            rs[:, 2] = origins[:, 2] + 0.22 + self.cfg.hover_offset \
                + torch.zeros(n, device=self.device).uniform_(-0.03, 0.03)
            # 抓 +x 端的无人机出场偏航 180°（镜像对称）：若两机都 yaw-0，一台看到"杆朝
            #   队友伸来"、另一台看到"从后方伸来"，逼出两套镜像映射；yaw-180 使两个
            #   egocentric 任务一致。条件按端判断（便于换端）。
            if self._end_sign[i] > 0:
                rs[:, 3] = 0.0; rs[:, 4:6] = 0.0; rs[:, 6] = 1.0   # quat(w=0,z=1) = 绕 z 转 180°
            else:
                rs[:, 3] = 1.0; rs[:, 4:7] = 0.0
            rs[:, 7:] = 0.0
            rb.write_root_pose_to_sim(rs[:, :7], env_ids)
            rb.write_root_velocity_to_sim(rs[:, 7:], env_ids)

            jp = rb.data.default_joint_pos[env_ids].clone()
            jv = rb.data.default_joint_vel[env_ids].clone()
            jp[:] = 0.0; jv[:] = 0.0
            arm_init = self._arm_stow_t.unsqueeze(0).repeat(n, 1) \
                + torch.zeros(n, 4, device=self.device).uniform_(-0.1, 0.1)
            arm_init = torch.clamp(arm_init, self._arm_lower, self._arm_upper)
            for k, jid in enumerate(self._arm_ids[i]):
                jp[:, jid] = arm_init[:, k]
            for k, jid in enumerate(self._grip_ids[i]):
                jp[:, jid] = self._gripper_close_t[k]
            rb.write_joint_state_to_sim(jp, jv, None, env_ids)
            self._arm_dof_targets[env_ids, i] = arm_init

            # 飞行目标（在那端上方）
            self._desired_pos_w[env_ids, i, 0] = origins[:, 0] + self._end_sign[i] * self.cfg.bar_grasp_x
            self._desired_pos_w[env_ids, i, 1] = origins[:, 1]
            self._desired_pos_w[env_ids, i, 2] = origins[:, 2] + 0.22 + self.cfg.hover_offset

    # ==============================================================
    # 可视化：搬运目标（杆初始位置 + carry_height）
    # ==============================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vis"):
                mc = CUBOID_MARKER_CFG.copy()
                mc.markers["cuboid"].size = (0.08, 0.08, 0.08)
                mc.prim_path = "/Visuals/Command/carry_goal"
                self.goal_vis = VisualizationMarkers(mc)
            self.goal_vis.set_visibility(True)
        else:
            if hasattr(self, "goal_vis"):
                self.goal_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        goal = self._bar_initial_pos.clone()
        goal[:, 2] = goal[:, 2] + self.cfg.carry_height
        self.goal_vis.visualize(goal)
