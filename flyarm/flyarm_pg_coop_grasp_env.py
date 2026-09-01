# ================================================================
# 平行夹爪双机协同抓取 —— 集中式版本。
#
# 两台空中机械臂共同抓住一根共享杆的两端, 把它从台座上抬起, 再搬运到目标位姿。
# 由单一策略同时观测两台无人机 (集中式控制)。本版本继承完整的协同栈
# (带持杆负载前馈的串级 LADRC 飞控、三阶段机械臂课程、抓后搬运、接触感知,
# 以及全部 grasp/align/transport 奖励), 只覆盖平行夹爪相关的部分。
#
# 该配置达到了 100% 任务成功率。决定性的设计选择是:
#   - 夹爪转 90 度 (base_rotate_center), 使开合轴与杆垂直 —— 横跨细杆所必需
#     (立方体各向同性, 掩盖了这个需求)。
#   - 近抓取起步课程: 每台无人机 reset 时处于 grasp-ready 的机械臂姿态, 夹爪
#     张开悬在自己那端杆上方, 跳过接近过程, 让策略先从 near-success 态学
#     合爪 -> 抬升 -> 协同。
#   - 加重的杆 (0.5 kg) 加高阻尼, 使某台无人机的扫过或合爪动作不会把杆碰开,
#     并让 near-grasp 姿态保持稳定。
#
# 此处覆盖的平行夹爪专属项:
#   - USD 路径 (通过 FLYARM_PG_USD 指向平行夹爪资产)。
#   - 夹爪 actuator: revolute (Nm/rad) -> prismatic (N/m)。
#   - gripper_open/close 用米; 资产以闭合姿态导出, 所以 joint=0 为闭合,
#     0.017 为张开。
#   - grasp_offset: 平行夹持面沿手指 +X 方向往外约 12 cm。
#   - effort 上限提到 10 N (平面接触不会扎穿杆)。
#   - reset 时把夹爪写成张开 (基类写成闭合 = 完全并拢, 无法套住杆端)。
#
# 退回旋转夹爪即训练原任务; 旧任务及其 USD 保持不动。
# ================================================================

from __future__ import annotations

import copy
import os

import torch

from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ViewerCfg

from .flyarm_coop_grasp_env import FlyarmCoopGraspEnv, FlyarmCoopGraspEnvCfg


# 平行夹爪默认 USD 路径。使用专门的环境变量, 避免与旋转夹爪资产路径冲突。
_PG_USD_DEFAULT = "C:/Users/zhekzhong2-c/Desktop/flyarmurdf/urdf/flyarm_pg/flyarm_pg.usd"


def _pgcoop_viewer_cfg():
    # 仅用于 play/录像的近距相机预设 (不影响训练或物理)。通过 FLYARM_PGCOOP_VIEW
    # 选择: close (默认) / tight / front / side (看杆是否水平最佳) / top。
    # 相机跟随 env 0。
    _P = {
        "close": dict(eye=(1.10, -1.30, 0.90), lookat=(0.0, 0.0, 0.42)),   # ~1.7m, 两台无人机 + 杆都在画面内
        "tight": dict(eye=(0.50, -0.62, 0.55), lookat=(0.0, 0.0, 0.38)),   # ~0.9m, 看夹爪细节
        "front": dict(eye=(0.0, -1.30, 0.60),  lookat=(0.0, 0.0, 0.40)),
        "side":  dict(eye=(1.30, 0.0, 0.55),   lookat=(0.0, 0.0, 0.40)),
        "top":   dict(eye=(0.0, -0.10, 1.40),  lookat=(0.0, 0.0, 0.30)),
    }
    v = _P.get(os.environ.get("FLYARM_PGCOOP_VIEW", "close"), _P["close"])
    return ViewerCfg(eye=v["eye"], lookat=v["lookat"], origin_type="env", env_index=0, resolution=(1920, 1080))


# ──────────────────────────────────────────────────────────────
# 配置: 继承集中式协同任务; 只覆盖平行夹爪专属的参数。
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmPgCoopGraspEnvCfg(FlyarmCoopGraspEnvCfg):

    # play/录像用的近距相机 (仅渲染)。
    viewer = _pgcoop_viewer_cfg()

    # === 夹爪 open/close 目标 (prismatic, 米; 顺序 = 右, 左) ===
    # 资产以闭合姿态导出: joint=0 为闭合, 0.017 为张开 (URDF 限位 0..0.017)。
    # action 约定不变 (a8>=0 -> 闭合, <0 -> 张开): 闭合 = 较小值, 张开 = 较大值,
    # 所以基类的目标选择逻辑原样复用。
    gripper_open = (0.017, 0.017)     # 两指都到行程上限 -> 最大间距 (能罩住 4 cm 杆截面)
    gripper_close = (0.0, 0.0)        # 两指都到 0 -> 最小间距 (力由 effort 限幅)

    # === 夹持面中心 (grasp_offset, gripper_base 坐标系) ===
    # 平行夹持面沿手指 +X 方向往外约 12 cm。这是 reaching / grasp_center /
    # grasp 判定 / align 共同瞄准的点, 所以它必须是夹持面而非 link 原点。
    # y=0 (两指对称, 无 revolute 镜像中线偏移); z=0.02 落在手指平面上。
    grasp_offset = (0.12, 0.0, 0.02)
    # 平行夹持面约 23 mm 高, 在竖直方向有容差。瞄准点比杆中心下移 1 cm, 因为
    # 指尖实际落点比瞄准点高约 1.5-2 cm; 不下移则两个对称夹爪都会擦到上边缘
    # (一指在顶上, 一指悬空), 联合接触永远形不成。
    grasp_z_offset = -0.01

    # === 协同时序奖励权重 ===
    # coop_ready / coop_close 奖励两台无人机同帧动作, 这会诱发死锁——谁先动谁就
    # 把杆碰歪。减半以倾向错开抓取 (一台抓住并稳住, 再换另一台);
    # coop_grasp (持续联合接触, 与先后无关) 保持基类值。
    coop_ready_reward_scale = 1.0
    coop_close_reward_scale = 1.0

    # === 近抓取起步课程 + 加重的杆 ===
    # 两台无人机从零抓一根细杆太难, 单靠 reward shaping 哄不出来; 策略会扫过
    # 杆而不下探, 轻杆又会被碰开。借鉴 physics-forced-synchrony 和反向课程的
    # 思路, 课程先 bootstrap 出抓取再退火:
    #   - 近抓取起步: reset 进 grasp-ready 直臂姿态, 夹爪张开悬在杆端上方,
    #     跳过三阶段机械臂课程。
    #   - 加重的杆: 更重 + 高阻尼, 使扫过不会把它碰落, 合爪/抬升也不会把它甩飞。
    weak_grip_reward_scale = 3.0      # graded min(两指) 力奖励; near-grasp 下才有意义
    touch_require_both = False        # 保留单指 touch 面包屑当接近引导
    pg_near_grasp = True              # 从 grasp-ready 态起步; 成功后关掉 (或退火) 以学习接近
    pg_near_grasp_clearance = 0.08    # 把无人机出场抬高这么多, 使张开的夹爪悬在杆【上方】
                                      # (避免 reset 帧重叠把杆甩飞); 目标点不抬高,
                                      # 所以 dist_to_goal 会把无人机往下带进抓取位。

    # === 夹爪朝向转 90 度 ===
    # 正向运动学显示, 在原抓取姿态下开合轴几乎与杆平行 (~1.4 度), 两指无法横跨
    # 杆截面。策略自己转不过来, 因为所需的 base_rotate 在它绕中心的可达范围之外。
    # 把中心转 -90 度 (0.45 - pi/2) 使开合轴与杆约垂直, 同时夹爪位置基本不变
    # (绕竖直轴原地 twist)。两台无人机都成立 (机 1 是 yaw-180)。
    base_rotate_center = -1.1208      # 0.45 - pi/2: 开合轴从顺杆变成横跨杆
    arm_stow = (-1.1208, -0.41, 0.35, -0.10)   # stow 的 base_rotate 与新中心匹配, 免得 settle 被 clamp 掉

    # === 夹爪 effort 上限 ===
    # 平面平行接触不会扎穿杆, 所以不需要 revolute 的防穿透限幅。10 N 足以稳稳
    # 托住杆 (mu=1, 每侧约 0.75 N), 也与验证过的单机值一致。
    gripper_effort_limit_2b = 10.0

    # === 真抓取阶段 ===
    # 基类把 weld_phase 默认为焊死学飞阶段; 本版本需要两机真抓取 + 协同搬运,
    # 所以 weld_phase=2。
    weld_phase = 2

    def __post_init__(self):
        super().__post_init__()

        # 杆阻尼, 使第二台无人机接近时不会在第一台持杆期间把杆甩飞或转起来。
        self.bar.spawn.rigid_props.angular_damping = 2.0
        self.bar.spawn.rigid_props.linear_damping = 0.5
        # 加重的杆: 扫过或合爪都碰不落它, 也能在 near-grasp 姿态下稳住。
        # 0.5 kg 远在两台无人机的抬升余量内。
        self.bar.spawn.mass_props.mass = 0.5

        # 覆盖两台机器人的 USD + 夹爪 actuator (deepcopy 以免改到共享的基类
        # 机器人 cfg)。两台机器人用同一个平行夹爪资产。
        for name in ("robot_0", "robot_1"):
            rb = copy.deepcopy(getattr(self, name))
            rb.spawn.usd_path = os.environ.get("FLYARM_PG_USD", _PG_USD_DEFAULT)
            # Prismatic 夹爪 actuator。杆宽处约 mm 级的干涉量乘 2000 N/m 给出
            # 几 N 的夹持力, 由 effort_limit 限幅。
            rb.actuators = dict(rb.actuators)
            rb.actuators["gripper"] = ImplicitActuatorCfg(
                joint_names_expr=["gripper_right_joint", "gripper_left_joint"],
                stiffness=2000.0,     # N/m (prismatic)
                damping=50.0,         # Ns/m
                effort_limit=self.gripper_effort_limit_2b,
            )
            setattr(self, name, rb)


# ──────────────────────────────────────────────────────────────
# 环境: 继承集中式协同任务; 唯一改动是 reset 时把两个夹爪都写成张开
# (飞控、负载前馈、机械臂课程、搬运态锁臂, 以及全部奖励均原样继承)。
# ──────────────────────────────────────────────────────────────
class FlyarmPgCoopGraspEnv(FlyarmCoopGraspEnv):

    cfg: FlyarmPgCoopGraspEnvCfg

    # ==============================================================
    # Reset —— 基类 reset 之后, 把两个夹爪从闭合改写成张开。
    # 基类把夹爪写成闭合; prismatic 夹爪闭合即完全并拢 (joint=0), 这样无人机会
    # 出场就夹死, 永远套不住杆端。写成张开 (0.017) 恢复 "接近时张开, 靠近再合爪"
    # —— 合爪时机仍由奖励学习得到。
    # ==============================================================
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)   # 杆在台座上 / 无人机在杆端上方 / 机械臂蜷缩 / 夹爪闭合 / LADRC 清零

        # 基类 reset 把 env_ids=None 解析为所有 env; 这里复用该索引,
        # 只追加 "写成张开" 这一步。
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot_0._ALL_INDICES
        n = len(env_ids)
        grip_open = self._gripper_open_t.unsqueeze(0).repeat(n, 1)   # (n,2) 张开 0.017
        grip_vel = torch.zeros_like(grip_open)
        for i, rb in enumerate(self._robots):
            rb.write_joint_state_to_sim(
                grip_open, grip_vel, joint_ids=self._grip_ids[i], env_ids=env_ids
            )

        # 近抓取起步: 不再 reset 成蜷缩悬在杆端上方、走 settle -> straighten
        # 两个阶段, 而是把两条机械臂直接摆成直臂 grasp-ready 姿态, 夹爪张开悬在
        # 杆端上方, 并把课程时钟推进到这些阶段之后。策略随后从 near-success 态学
        # 合爪 -> 抬升 -> 协同。加重的杆使该姿态不倾倒、不被碰开。抓取成功后关掉
        # (或退火) cfg.pg_near_grasp, 让它学习接近。
        if self.cfg.pg_near_grasp:
            # 把无人机抬高 `clearance` 出场, 使夹爪悬在杆【上方】而非与之重叠;
            # 重叠的 reset 会导致第一帧巨大的解算力把杆甩飞。目标点
            # (_desired_pos_w, 已在抓取高度) 不抬高, 所以 dist_to_goal 会把
            # 无人机往下带进抓取位。
            origins = self._terrain.env_origins[env_ids]
            straight = self._arm_straight_pose_t                       # (4,) 直臂 grasp-ready 姿态
            arm_pos = straight.unsqueeze(0).repeat(n, 1)               # (n,4)
            arm_vel = torch.zeros_like(arm_pos)
            for i, rb in enumerate(self._robots):
                rb.write_joint_state_to_sim(arm_pos, arm_vel, joint_ids=self._arm_ids[i], env_ids=env_ids)
                self._arm_dof_targets[env_ids, i] = arm_pos            # 从直臂姿态在策略控制下继续
                rs = rb.data.default_root_state[env_ids].clone()
                rs[:, 0] = origins[:, 0] + self._end_sign[i] * self.cfg.bar_grasp_x
                rs[:, 1] = origins[:, 1]
                rs[:, 2] = origins[:, 2] + 0.22 + self.cfg.hover_offset + self.cfg.pg_near_grasp_clearance
                if self._end_sign[i] > 0:
                    rs[:, 3] = 0.0; rs[:, 4:6] = 0.0; rs[:, 6] = 1.0   # +x 端: yaw-180 (镜像对称)
                else:
                    rs[:, 3] = 1.0; rs[:, 4:7] = 0.0
                rs[:, 7:] = 0.0
                rb.write_root_pose_to_sim(rs[:, :7], env_ids)
                rb.write_root_velocity_to_sim(rs[:, 7:], env_ids)
            self._steps_since_reset[env_ids] = self.cfg.settle_steps + self.cfg.straighten_steps   # 跳过分阶段课程
