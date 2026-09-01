"""双机空中机械臂协同抓取环境 —— 平行夹爪的 MARL（MAPPO/CTDE）版本。

两台四旋翼机械臂学习共同抓住一根共享杆并把它抬到目标位置，按多智能体问题训练
（每个 agent 一份策略，配中心化 critic）。本文件继承旋转夹爪的 MARL 环境，只覆盖
平行夹爪特有的部分，让已验证的基类环境保持原样。

这里覆盖的内容：
- USD 资产：平行夹爪模型（经 FLYARM_PG_USD 环境变量，与旋转夹爪资产相互独立）。
- 夹爪 actuator：revolute（Nm/rad）-> prismatic（N/m），两台机器人都改。
- 夹爪 open/close 目标：弧度 -> 米（0 = 闭合，0.017 = 张开；CAD 模型按闭合姿态导出）。
- grasp_offset：平行夹爪的夹持面在腕系沿手指 +X 方向约 12 cm 处。
- Reset：夹爪写成张开，让两指在合拢前能横跨杆。
- 夹爪旋转 90 度，让开合轴垂直于杆。
- 近抓取课程：让机器人从 grasp-ready 姿态起步，绕开同时性死锁。

飞控（cascade LADRC）、分阶段臂课程、carry/交接逻辑、协作 reward 项以及 MAPPO 接口
全部原样继承。训练未改动的旋转夹爪任务，可作为受控对比基线。
"""

from __future__ import annotations

import copy
import os

import torch

from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ViewerCfg

from .flyarm_coop_grasp_marl_env import FlyarmCoopGraspMarlEnv, FlyarmCoopGraspMarlEnvCfg, _P


# 平行夹爪 USD 默认路径；用自己专属的环境变量，绝不与旋转夹爪资产冲突。
_PG_USD_DEFAULT = "C:/Users/zhekzhong2-c/Desktop/flyarmurdf/urdf/flyarm_pg/flyarm_pg.usd"

# prismatic 夹爪的 open/close 目标，单位米（顺序：右、左）。
# CAD 模型按闭合姿态导出，所以 joint=0 是闭合、0.017 是张开（URDF 限位 0..0.017）。
# action 约定不变（a8 >= 0 闭合）：闭合 = 较小值，张开 = 较大值，
# 因此父类的 gtarget = sel_close*close + (1-sel_close)*open 逻辑无需改动即可用。
_PG_GRIPPER_OPEN = (0.017, 0.017)
_PG_GRIPPER_CLOSE = (0.0, 0.0)
# 腕（gripper_base）系下的夹持中心：平行夹持面在沿手指 +X 约 12 cm 处。
# reaching / grasp_center / 抓取判定都瞄这个点，即真实夹持面而非 link 原点。
_PG_GRASP_OFFSET = (0.12, 0.0, 0.02)


def _pgcoop_marl_viewer_cfg():
    # 仅用于 play/录像的近距相机预设（不影响训练或物理）。
    # 经 FLYARM_PGCOOP_VIEW 环境变量选择；相机跟随 env 0。
    _V = {
        "close": dict(eye=(1.10, -1.30, 0.90), lookat=(0.0, 0.0, 0.42)),
        "tight": dict(eye=(0.50, -0.62, 0.55), lookat=(0.0, 0.0, 0.38)),
        "front": dict(eye=(0.0, -1.30, 0.60),  lookat=(0.0, 0.0, 0.40)),
        "side":  dict(eye=(1.30, 0.0, 0.55),   lookat=(0.0, 0.0, 0.40)),
        "top":   dict(eye=(0.0, -0.10, 1.40),  lookat=(0.0, 0.0, 0.30)),
    }
    v = _V.get(os.environ.get("FLYARM_PGCOOP_VIEW", "close"), _V["close"])
    return ViewerCfg(eye=v["eye"], lookat=v["lookat"], origin_type="env", env_index=0, resolution=(1920, 1080))


# ──────────────────────────────────────────────────────────────
# Config：继承双机 MARL 配置，只覆盖平行夹爪的资产/actuator
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmPgCoopGraspMarlEnvCfg(FlyarmCoopGraspMarlEnvCfg):

    # play/录像用的近距相机（仅渲染）。
    viewer = _pgcoop_marl_viewer_cfg()

    # 近抓取课程：出场预留间隙，让夹爪悬在杆上方不重叠，
    # 避免 reset 时的瞬时碰撞冲击。
    pg_near_grasp_clearance = 0.08

    def __post_init__(self):
        # 若父类定义了 __post_init__ 就调用（MARL 基类 cfg 可能没有）。
        sup = getattr(super(), "__post_init__", None)
        if sup is not None:
            sup()

        # 覆盖两台机器人的 USD 资产和夹爪 actuator。
        # robot_0/robot_1 由父类 _setup_scene 读取；逐个深拷贝后修改。
        pg_usd = os.environ.get("FLYARM_PG_USD", _PG_USD_DEFAULT)
        for name in ("robot_0", "robot_1"):
            rb = copy.deepcopy(getattr(self, name))
            rb.spawn.usd_path = pg_usd
            # 夹爪 actuator：revolute（Nm/rad）-> prismatic（N/m）。
            # 2000 N/m 刚度在与杆毫米级干涉量上给出几 N 夹紧力，
            # 由 effort_limit 封顶。10 N effort 避免平面接触穿透。
            rb.actuators = dict(rb.actuators)
            rb.actuators["gripper"] = ImplicitActuatorCfg(
                joint_names_expr=["gripper_right_joint", "gripper_left_joint"],
                stiffness=2000.0,     # N/m（prismatic）
                damping=50.0,         # Ns/m
                effort_limit=10.0,    # N
            )
            setattr(self, name, rb)


# ──────────────────────────────────────────────────────────────
# Environment：继承双机 MARL 环境；只覆盖夹爪常量张量
#（这些值从 _P 读取，cfg 够不到）以及把夹爪写成张开的 reset。
# 飞控 / 臂课程 / 抓取 / 协作 reward / MAPPO 接口都原样继承。
# ──────────────────────────────────────────────────────────────
class FlyarmPgCoopGraspMarlEnv(FlyarmCoopGraspMarlEnv):

    cfg: FlyarmPgCoopGraspMarlEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # 父类 MARL 环境在 _setup_scene/__init__ 里从模块级 _P 读取 weld_phase 和
        # gripper_effort_limit_2b，所以必须在 super().__init__() 之前设好（cfg 够不到它们）。
        # _P 是共享的模块级实例，但训练每个进程只跑一个任务，因此这里只影响本进程；
        # 旋转夹爪 MARL 任务跑在自己的进程里，_P 不受影响。
        #   weld_phase 0 -> 2：开启真抓取（父类默认把杆焊死做纯飞行训练）。
        #   gripper_effort_limit_2b -> 10：平面平行夹爪接触用 10 N（与 actuator 限值一致）。
        _P.weld_phase = 2
        _P.gripper_effort_limit_2b = 10.0
        # 把集中式 PgCoopGrasp 的修复 port 进 MARL 版（任务几何相同）。
        # grasp_z_offset / coop_ready / coop_close 在 reward 里从 _P 读；bar = _P.bar。
        # 在 super().__init__() 之前设：_setup_scene 读 _P.bar/coop，grasp target 读 _P.grasp_z_offset。
        _P.grasp_z_offset = -0.01            # 瞄夹持面；平行夹爪指尖在瞄准点之上，瞄中心会让一根指骑到顶面
        _P.coop_ready_reward_scale = 1.0     # 削弱"逼两机同帧到位"（死锁诱因）
        _P.coop_close_reward_scale = 1.0     # 削弱"逼两机同帧合爪"
        cfg.bar.spawn.rigid_props.angular_damping = 2.0   # 杆阻尼，避免第二台机器人的接触把杆甩飞
        cfg.bar.spawn.rigid_props.linear_damping = 0.5
        # port 已验证的集中式配方，让 MARL 学习的是同一个已可解的任务，
        # 近抓取课程绕开同时性死锁。这些值从 _P 读取，所以在 super 之前设。
        _P.base_rotate_center = -1.1208               # 把夹爪转 90 度，让开合轴垂直于杆（否则会想沿杆方向合爪，进不去）
        _P.arm_stow = (-1.1208, -0.41, 0.35, -0.10)   # stow 的 base_rotate 与新中心匹配
        _P.use_grasp_assist = False                   # 关掉旧版 assist（闭合夹爪、无间隙）；本环境用 _reset_idx 里的"张开夹爪 + 间隙"拐杖
        cfg.bar.spawn.mass_props.mass = 0.5           # 加重杆，使单点接触不会把它撞开
        # 重新开启杆碰撞：_P（= FlyarmCoopGraspEnvCfg()）在模块加载时按 weld_phase=0 实例化，
        # 其 __post_init__ 关闭了杆碰撞（焊死阶段不需要）。真抓取（weld_phase=2）需要碰撞；
        # __post_init__ 不会再跑，所以这里强制打开。cfg.bar 和 _P.bar 是同一个对象。
        cfg.bar.spawn.collision_props.collision_enabled = True
        print("[pg-marl] bar collision enabled")
        # hold 课程：把 hold_bonus 的距离/倾角阈值放宽（loose -> strict），引导策略
        # 抬升到目标并保持。success 指标始终用严格阈值，所以上报的 success_rate
        # 与集中式基线保持可比。
        _P.use_hold_curriculum = True
        print(f"[pg-marl] hold_bonus thresholds {_P.hold_loose_dist}/{_P.hold_loose_tilt} -> "
              f"{_P.success_dist}/{_P.success_tilt} over {_P.hold_curric_steps} steps; success metric stays strict")
        # lift_gate：用 0..1 的"杆离台高度"因子门控近目标搬运 reward（bar_goal/hold_bonus），
        # 让杆只有真正抬起后才拿到这些项。这消除了"夹住躺着不抬"的局部最优。
        # 抓取锚点（coop_grasp/contact）不动，保住已学会的抓取；success 判定（in_goal_strict）
        # 仍严格，集中式基线不受影响（那里该 flag 默认关）。
        _P.lift_gate_transport_rewards = True
        print("[pg-marl] transport rewards gated by lift progress; grasp anchor unchanged")
        # carry_ramp：carry 触发后，飞行目标 z 用 ease-out 斜坡在约 1.2s（60 步）内抬升，
        # 而非瞬间 +carry_height 跳变。去中心化的 agent 跟不上目标突跳（会把杆甩飞），
        # 斜坡让目标平滑爬升、两机稳定跟随。最终目标高度和 success 判定不变。
        _P.carry_ramp_steps = 60
        print(f"[pg-marl] carry flight-target z uses an ease-out ramp ({_P.carry_ramp_steps} steps ~1.2s)")
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device

        # 重设夹爪常量张量：父类 __init__ 从 _P 读的是旋转夹爪（弧度）值。
        # _P 与旋转夹爪任务共享，所以不去改它，而是只覆盖本实例的张量为
        # 平行夹爪的米值。父类的 _apply_action、_grasp_center 和抓取判定都用这些张量，
        # 因此整条链都切换到平行夹爪。
        self._gripper_open_t = torch.tensor(_PG_GRIPPER_OPEN, device=dev)
        self._gripper_close_t = torch.tensor(_PG_GRIPPER_CLOSE, device=dev)
        self._grasp_offset_t = torch.tensor(_PG_GRASP_OFFSET, device=dev)
        print(f"[pg-marl] parallel gripper override: open={_PG_GRIPPER_OPEN}(m)/close={_PG_GRIPPER_CLOSE}(m), "
              f"grasp_offset={_PG_GRASP_OFFSET}(wrist frame, +X grasp face)")

    # ==============================================================
    # Reset —— 在父类逻辑之后，把夹爪从闭合改写为张开。
    # 父类 _reset_idx 在真抓取阶段把夹爪状态设为闭合；对 prismatic 夹爪而言
    # close=0 会让两指完全合拢，机器人出场就夹死，approach 时无法横跨杆。
    # 把两台机器人的手指写成张开（0.017）符合"approach 张开、靠近时合爪"行为；
    # 合爪时机仍由抓取 reward 来学。
    #（焊死阶段父类已经写成张开，这里重写无害。）
    # ==============================================================
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)   # 父类：臂课程起步姿态、夹爪闭合（真抓取阶段）、LADRC/状态 reset

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot_0._ALL_INDICES
        n = len(env_ids)
        grip_open = self._gripper_open_t.unsqueeze(0).repeat(n, 1)   # (n,2) 张开 0.017
        grip_vel = torch.zeros_like(grip_open)
        for i, rb in enumerate(self._robots):                        # 两台机器人都张开
            rb.write_joint_state_to_sim(
                grip_open, grip_vel, joint_ids=self._grip_ids[i], env_ids=env_ids
            )

        # 近抓取课程（集中式成功配方应用到所有 env）：
        # grasp-ready 直臂（夹爪已转 90 度）、出场抬高一个 clearance 让夹爪悬在杆上方不重叠，
        # 并跳过从零开始的分阶段课程，让策略直接学"合爪 -> 协同抬升 -> 协同搬运"，绕开同时性死锁。
        # 夹爪刚被写成张开（靠近才合爪）；_desired_pos_w（杆端悬停高度）保留，
        # 让 dist_to_goal 把它缓缓降到杆端。
        origins = self._terrain.env_origins[env_ids]
        arm_pos = self._arm_straight_pose_t.unsqueeze(0).repeat(n, 1)   # (n,4) 直臂（base_rotate=-1.1208，90 度）
        arm_vel = torch.zeros_like(arm_pos)
        clr = self.cfg.pg_near_grasp_clearance
        for i, rb in enumerate(self._robots):
            rb.write_joint_state_to_sim(arm_pos, arm_vel, joint_ids=self._arm_ids[i], env_ids=env_ids)
            self._arm_dof_targets[env_ids, i] = arm_pos
            rs = rb.data.default_root_state[env_ids].clone()
            rs[:, 0] = origins[:, 0] + self._end_sign[i] * _P.bar_grasp_x
            rs[:, 1] = origins[:, 1]
            rs[:, 2] = origins[:, 2] + 0.22 + _P.hover_offset + clr   # 杆台座顶 0.22 + hover + clearance（夹爪在杆上方，不重叠）
            if self._end_sign[i] > 0:
                rs[:, 3] = 0.0; rs[:, 4:6] = 0.0; rs[:, 6] = 1.0       # +x 端 yaw 180（镜像对称，同父类）
            else:
                rs[:, 3] = 1.0; rs[:, 4:7] = 0.0
            rs[:, 7:] = 0.0
            rb.write_root_pose_to_sim(rs[:, :7], env_ids)
            rb.write_root_velocity_to_sim(rs[:, 7:], env_ids)
        self._steps_since_reset[env_ids] = _P.settle_steps + _P.straighten_steps   # 跳过从零的分阶段课程（直接进抓取阶段）
        self._grasp_phase[env_ids] = True
