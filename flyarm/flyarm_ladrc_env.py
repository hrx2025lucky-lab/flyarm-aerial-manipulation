"""采用串级 LADRC 飞控的单机空中机械臂环境。

本模块是单机视觉环境的轻量子类, 只把内环飞控换成独立的 `CascadeLADRC` 模块,
reward 塑形、抓取逻辑、三阶段机械臂课程和触觉感知都保持不变。
它作为 LADRC 模块在双机任务复用之前的验证台。

要点:
- 继承 FlyarmVisionEnv; 只覆盖飞控(关掉父类内联的控制器, 再用 LADRC 模块重算 thrust/moment)。
- action/obs 空间不变(obs 29, action 9: 加速度 3 + yaw-rate 1 + 臂 4 + 夹爪 1),
  所以策略既能从单机视觉 checkpoint warm-start, 也能从零训练。
- LADRC 增益: 内环 b0 = 1/J (roll/pitch ~80, yaw ~130); ESO 状态钳制随 b0 缩放,
  在 50 Hz 下既保证 observer 有效又不放大测量噪声。
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from .flyarm_vision_env import FlyarmVisionEnv, FlyarmVisionEnvCfg
from .cascade_ladrc import CascadeLADRC


def _ladrc_viewer_cfg() -> ViewerCfg:
    """选一个近景 play 录像相机, 让回放时抓取动作始终在画面内。"""

    view = os.environ.get("FLYARM_LADRC_VIEW", "close").lower()
    presets = {
        # 中近景斜视: 平台、夹爪、机械臂和机身都看得到。
        "close": ((1.35, -1.35, 0.95), (0.50, 0.00, 0.36)),
        "tight": ((0.95, -0.85, 0.62), (0.50, 0.00, 0.24)),
        "front": ((0.50, -1.45, 0.78), (0.50, 0.00, 0.34)),
        "side": ((1.45, 0.00, 0.78), (0.50, 0.00, 0.34)),
        "top": ((0.50, -0.05, 1.55), (0.50, 0.00, 0.25)),
        "diag": ((1.55, -1.50, 1.00), (0.50, 0.00, 0.36)),
    }
    eye, lookat = presets.get(view, presets["close"])
    return ViewerCfg(eye=eye, lookat=lookat, origin_type="env", env_index=0, resolution=(1920, 1080))


# ──────────────────────────────────────────────────────────────
# 配置: 继承视觉环境, 关掉父类内联的飞控, 提供串级 LADRC 增益
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmLadrcEnvCfg(FlyarmVisionEnvCfg):

    # 仅用于回放的相机; 影响 play/录像, 不影响训练或物理。
    # 设置 FLYARM_LADRC_VIEW=close/tight/front/side/top/diag。
    viewer = _ladrc_viewer_cfg()

    # 关掉父类两个内联飞控分支, 让它的 _pre_physics_step 走廉价的
    # 直接出力矩路径作占位; 本文件再通过串级 LADRC 模块覆盖 self._thrust/self._moment。
    # 机械臂课程、抓后搬运和夹爪逻辑仍交给父类。
    use_ladrc_control = False        # 关掉父类内联 LADRC
    use_setpoint_control = False     # 关掉父类几何内环

    # === cascade_ladrc 模块增益(b0 = 1/J; z2 钳制随 b0 缩放) ===
    ladrc_module_in_wc_rp = 10.5
    # 内环 observer 带宽按 50 Hz 更新率调: h·wo ≈ 0.3 让 ESO 平滑(特征值 ~0.7)。
    # wo 更高接近 deadbeat 会放大测量噪声; 真正的高带宽 ESO 需要物理频率的内环。
    ladrc_module_in_wo_rp = 15.0
    ladrc_module_in_b0_rp = 80.0     # ≈ 1/Jxx (裸机近似)
    ladrc_module_in_wc_yaw = 8.0
    ladrc_module_in_wo_yaw = 13.0
    ladrc_module_in_b0_yaw = 130.0   # ≈ 1/Jzz
    ladrc_module_out_wc = 2.0
    ladrc_module_out_wo = 8.1
    ladrc_module_out_b0 = 1.0
    # 力矩符号兜底: 上电后若机体往反方向翻, 设为 -1.0。
    ladrc_moment_sign = 1.0


# ──────────────────────────────────────────────────────────────
# 环境: 继承视觉环境, 只覆盖 __init__/_pre_physics_step/_reset_idx
# 来接入模块。每个覆盖都先调 super(), 再覆盖飞控,
# 从而保留 reward/抓取/课程 逻辑。
# ──────────────────────────────────────────────────────────────
class FlyarmLadrcEnv(FlyarmVisionEnv):

    cfg: FlyarmLadrcEnvCfg

    def __init__(self, cfg: FlyarmLadrcEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 串级 LADRC 模块(单 agent, A=1)。ESO 状态钳制与 b0 联动
        # (roll/pitch 0.3×b0, yaw 0.2×b0), 改增益时保持一致。
        self.ladrc = CascadeLADRC(
            self.num_envs, num_agents=1, device=self.device,
            in_wc_rp=cfg.ladrc_module_in_wc_rp, in_wo_rp=cfg.ladrc_module_in_wo_rp, in_b0_rp=cfg.ladrc_module_in_b0_rp,
            in_wc_yaw=cfg.ladrc_module_in_wc_yaw, in_wo_yaw=cfg.ladrc_module_in_wo_yaw, in_b0_yaw=cfg.ladrc_module_in_b0_yaw,
            out_wc=cfg.ladrc_module_out_wc, out_wo=cfg.ladrc_module_out_wo, out_b0=cfg.ladrc_module_out_b0,
            z2_clamp_in_rp=0.3 * cfg.ladrc_module_in_b0_rp,
            z2_clamp_in_yaw=0.2 * cfg.ladrc_module_in_b0_yaw,
            z2_clamp_out=1.0,
            max_angle=cfg.ladrc_max_angle, max_rate=cfg.ladrc_max_rate,
        )
        print(f"[flyarm-ladrc] cascade_ladrc 模块已接入(单机A=1). b0_rp={cfg.ladrc_module_in_b0_rp}, "
              f"b0_yaw={cfg.ladrc_module_in_b0_yaw}, z2_rp={0.3*cfg.ladrc_module_in_b0_rp:.1f}, "
              f"z2_yaw={0.2*cfg.ladrc_module_in_b0_yaw:.1f}, moment_sign={cfg.ladrc_moment_sign}")

    # ==============================================================
    # 解析动作: 先跑父类(机械臂课程 / 搬运 / 夹爪 + 占位飞控),
    # 再用模块覆盖飞控。
    # ==============================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)   # 父类算臂目标/搬运 + 占位 thrust/moment

        # 用串级 LADRC 模块重算飞控。
        quat = self._robot.data.root_quat_w                        # (N,4) wxyz
        omega = self._robot.data.root_ang_vel_b                    # (N,3) 实测机体角速度
        grav_b = self._robot.data.projected_gravity_b             # (N,3) 实测倾斜(水平时为 [0,0,-1])
        g = self._gravity_magnitude
        a_des_w = self._actions[:, 0:3] * self.cfg.accel_scale     # (N,3) 世界系期望加速度
        yaw_rate_des = self._actions[:, 3] * self.cfg.yaw_rate_scale   # (N,)

        # 沿机体 z 轴的推力。
        self._thrust[:, 0, 2] = self.ladrc.accel_to_thrust(
            a_des_w[:, None], quat[:, None], self._robot_mass, g,
            self.cfg.thrust_to_weight * self._robot_weight,
        )[:, 0]

        # 外环姿态 LADRC(RL 频率) → 角速率 setpoint(考虑 yaw 的映射)。
        roll_des, pitch_des = self.ladrc.accel_to_attitude_setpoint(
            a_des_w[:, None], quat[:, None], g, self.cfg.ladrc_max_angle,
        )   # 各 (N,1,1)
        pitch_meas = grav_b[:, 0:1].unsqueeze(1)        # (N,1,1)
        roll_meas = (-grav_b[:, 1:2]).unsqueeze(1)      # (N,1,1)
        h = self.cfg.sim.dt * self.cfg.decimation       # 0.02s (50Hz); sim-to-real 时提到 1kHz
        p_sp, q_sp = self.ladrc.outer_step(roll_des, pitch_des, roll_meas, pitch_meas, h)   # (N,1,1)
        r_sp = yaw_rate_des.view(self.num_envs, 1, 1)   # (N,1,1) yaw 直接跟踪角速率(无外环)

        # 内环角速率 LADRC → 力矩(此处 RL 频率; sim-to-real 时内环 1kHz)。
        moment = self.ladrc.inner_step(p_sp, q_sp, r_sp, omega[:, None], h)   # (N,1,3)
        moment = moment * self.cfg.ladrc_moment_sign     # 符号兜底(翻了就把 cfg 设 -1.0)
        self._moment[:, 0, :] = moment[:, 0, :]
        # 把实际施加的力矩回灌进 ESO 的 u_prev(此处无前馈/扰动)。
        self.ladrc.commit_inner_applied(moment)

    # ==============================================================
    # 重置: 父类逻辑 + 清空 LADRC 模块状态(z1/z2/u_prev)。
    # ==============================================================
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.ladrc.reset(env_ids)
