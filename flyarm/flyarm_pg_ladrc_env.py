"""单机空中机械臂环境：平行夹爪 + 串级 LADRC 飞控。

本环境在已验证的单机 LADRC 控制栈之上，验证新的平行夹爪（prismatic 移动副手指）。
它是 ``FlyarmLadrcEnv`` 的轻量子类，原样继承飞控、分阶段臂课程、抓后搬运逻辑、
触觉感知以及全部 reward 项，只覆盖与夹爪相关的部分。

覆盖内容：
- USD 资产路径（平行夹爪模型），由 ``FLYARM_PG_USD`` 环境变量选择。
- 夹爪 actuator：revolute（Nm/rad）-> prismatic（N/m）。
- ``gripper_open`` / ``gripper_close`` 目标改用米为单位（CAD 在闭合姿态导出，
  所以 joint = 0 表示闭合，0.017 表示张开）。
- ``grasp_offset``：夹持平面位于腕部 +X 方向（手指伸出轴）而非腕部下方；
  这是所有 reward 项瞄准的点。
- ``hover_offset``：因平行夹爪更长而略微抬高。
- reset 时把夹爪张开（父类 reset 成闭合），让手指在 approach 阶段能横跨物体。
"""

from __future__ import annotations

import copy
import os

import torch

from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from .flyarm_ladrc_env import FlyarmLadrcEnv, FlyarmLadrcEnvCfg


# 平行夹爪模型的默认 USD 路径。使用专用环境变量，避免与旧夹爪的 FLYARM_USD 冲突。
_PG_USD_DEFAULT = "C:/Users/zhekzhong2-c/Desktop/flyarmurdf/urdf/flyarm_pg/flyarm_pg.usd"


# ──────────────────────────────────────────────────────────────
# Config：继承单机 LADRC env，只覆盖平行夹爪相关参数
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmPgLadrcEnvCfg(FlyarmLadrcEnvCfg):

    # prismatic 夹爪的张开/闭合目标，单位米（顺序：右、左）。
    # CAD 在闭合姿态导出，所以 joint = 0 表示闭合，0.017（行程上限）表示张开。
    # 动作约定（a8 >= 0 -> 闭合）不变，因此父类的目标选择逻辑直接复用、无需修改。
    gripper_open = (0.017, 0.017)     # 两指都到行程上限 -> 最大开口
    gripper_close = (0.0, 0.0)        # 两指都到 0 -> 最小开口（夹紧力由 effort 封顶）

    # 腕部（gripper_base）系下的夹持中心。平行夹爪的接触平面在手指伸出轴（+X）上
    # 约 12 cm 处。所有 reward 项（reaching、grasp_center、抓取成功判定）瞄准该点，
    # 而非 link 原点。
    grasp_offset = (0.12, 0.0, 0.02)
    # 瞄准物体中心：23 mm 宽的接触平面在竖直方向有较大宽容度，其下沿仍高于台座顶面。
    grasp_z_offset = 0.0

    # 悬停高度：平行夹爪长约 2 cm，抬高 hover_offset 以保持相同的脚架离地间隙。
    hover_offset = 0.47

    def __post_init__(self):
        super().__post_init__()

        # 覆盖机器人 USD + 夹爪 actuator（deepcopy 避免污染共享的父类 cfg）。
        self.robot = copy.deepcopy(self.robot)
        self.robot.spawn.usd_path = os.environ.get("FLYARM_PG_USD", _PG_USD_DEFAULT)
        # 夹爪 actuator：revolute -> prismatic。刚度 2000 N/m 在手指闭合到物体上、
        # 产生毫米级干涉量时给出几 N 的夹紧力；effort_limit 给它封顶。平面接触面
        # 避免了旧的薄指尖的扎穿问题。
        self.robot.actuators = dict(self.robot.actuators)
        self.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper_right_joint", "gripper_left_joint"],
            stiffness=2000.0,     # N/m (prismatic)
            damping=50.0,         # Ns/m
            effort_limit=10.0,    # N
        )


# ──────────────────────────────────────────────────────────────
# Env：继承单机 LADRC env；唯一覆盖 = reset 时张开夹爪
# ──────────────────────────────────────────────────────────────
class FlyarmPgLadrcEnv(FlyarmLadrcEnv):

    cfg: FlyarmPgLadrcEnvCfg

    def _reset_idx(self, env_ids):
        # 父类把夹爪 reset 到闭合目标。对 prismatic 夹爪而言那会把手指完全夹拢
        # （joint = 0），导致 approach 阶段手指无法横跨物体。这里在父类 reset 之后
        # 重写为张开；策略仍通过 finger reward 学习何时闭合。
        super()._reset_idx(env_ids)

        n = len(env_ids)
        grip_open = self._gripper_open_t.unsqueeze(0).repeat(n, 1)   # (n, 2) 张开在 0.017
        grip_vel = torch.zeros_like(grip_open)
        self._robot.write_joint_state_to_sim(
            grip_open, grip_vel, joint_ids=self._gripper_joint_ids, env_ids=env_ids
        )
