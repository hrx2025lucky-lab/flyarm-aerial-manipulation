# ================================================================
# cascade_ladrc.py — 串级 LADRC 姿态控制器（纯 torch 实现）
#
# 面向空中机械臂的向量化、多智能体串级 LADRC（线性自抗扰控制器）。
# 它充当底层内环飞控：RL 策略输出高层 setpoint（期望加速度 + 偏航率），
# 本模块把它们转换成稳定平台所需的推力与机体力矩。
#
#   - 串级结构：外环角度环（rad）-> 内环角速率环（rad/s），内环直接输出
#     机体力矩（N·m）。
#   - 每个单轴 LADRC 由一个二阶 LESO（扩张状态观测器）和一个 LSEF 控制律
#     配对：LESO 估计总扰动，LSEF 将其抵消：u = (wc*(ref - z1) - z2) / b0，
#     其中 b0 ≈ 1 / 该轴转动惯量。
#   - 自抗扰让姿态环在机械臂摆动时仍保持水平（机械臂的反作用力矩被观测
#     并补偿）。
#   - 为 sim-to-real 一致性设计：仿真器与最终的 PX4 固件运行的是同一套逻辑。
#     仅依赖 torch（不依赖 Isaac Lab），因此该循环可干净地移植到嵌入式 C++。
#
# 频率拆分（sim-to-real 正确做法）：
#   - outer_step() 在 RL/控制频率运行（例如 50 Hz）。
#   - inner_step() 在物理频率运行（例如 100–1000 Hz）。
#   各自使用独立的积分步长 h。
#
# 多智能体：状态缓存形状为 (num_envs, num_agents, 5)。单机用 num_agents=1；
# 双机协作任务用 num_agents=2，无需改动本文件。
#
# 调参说明 —— b0 必须按平台转动惯量重新标定：
#   内环输出物理力矩（N·m），所以 b0 ≈ 1 / J_axis（单位力矩产生的角加速度）。
#   裸机架 Jxx≈0.013 对应 b0≈80；挂上机械臂会增大 J，故 b0 按
#   b0_new ≈ b0_ref · (J_ref / J_new) 缩小。外环角度环用 b0=1
#   （θ̇≈ω，输入本身就是角速率）。只要量级正确，LESO 能吸收 b0 的偏差。
# ================================================================

from __future__ import annotations

import torch


# 状态缓存中 5 个单轴 LADRC 环的索引。
L_OUT_ROLL, L_OUT_PITCH = 0, 1     # 外环角度环（输出 = 角速率 setpoint）
L_IN_ROLL, L_IN_PITCH, L_IN_YAW = 2, 3, 4   # 内环角速率环（输出 = 力矩 N·m）


# ──────────────────────────────────────────────────────────────
# 四元数辅助函数（wxyz），纯 torch 实现以保证 PX4 可移植性
# ──────────────────────────────────────────────────────────────
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """用四元数 q(...,4 wxyz) 旋转向量 v(...,3)：v_world = R(q) · v。"""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """从四元数 (...,4 wxyz) 提取偏航角 (...)。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ──────────────────────────────────────────────────────────────
# 串级 LADRC
# ──────────────────────────────────────────────────────────────
class CascadeLADRC:
    """
    串级一阶 LADRC（外环角度环 + 内环角速率环）。
    向量化，且支持多智能体。

    每个轴均为带宽参数化的一阶 LADRC（Han/Gao 形式）：
        被控对象近似       ẏ = b0·u + f          (f = 总扰动，先估计再抵消)
        LESO（二阶）       e  = z1 - y
                          ż1 = z2 - β1·e + b0·u_prev
                          ż2 = -β2·e             (β1 = 2·wo, β2 = wo²)
        控制律            u0 = wc·(r - z1)
                          u  = (u0 - z2) / b0
        在 _leso_step 中用前向欧拉法（步长 h）离散化。

    z2 在扰动量纲下被钳制：
        - 内环：y = 角速率（rad/s）⇒ f 量纲为 rad/s²；z2_clamp 量纲为
          rad/s²（rp=24, yaw=26）。它限定了能补偿的最大机械臂诱导
          角加速度扰动：τ_dist/J ≈ 0.34/0.013 ≈ 26 rad/s²，故 24 覆盖绝大部分。
        - 外环：y = 角度（rad），b0=1 ⇒ f 量纲为 rad/s（残余角速率
          扰动）；z2_clamp 量纲为 rad/s（1.0）。
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int = 1,
        device: str | torch.device = "cuda",
        # —— 内环角速率 roll/pitch ——
        in_wc_rp: float = 10.5, in_wo_rp: float = 40.0, in_b0_rp: float = 80.0,
        # —— 内环角速率 yaw ——
        in_wc_yaw: float = 8.0, in_wo_yaw: float = 35.0, in_b0_yaw: float = 130.0,
        # —— 外环角度 roll/pitch ——
        out_wc: float = 2.0, out_wo: float = 8.1, out_b0: float = 1.0,
        # —— z2 钳制（扰动量纲；见类 docstring）——
        z2_clamp_in_rp: float = 24.0,    # rad/s²
        z2_clamp_in_yaw: float = 26.0,   # rad/s²
        z2_clamp_out: float = 1.0,       # rad/s
        # —— setpoint 限幅 ——
        max_angle: float = 0.61,         # 外环角度 setpoint 上限（≈35°）
        max_rate: float = 4.0,           # 角速率 setpoint 上限（rad/s）
    ):
        self.N = num_envs
        self.A = num_agents
        self.device = device

        self.in_wc_rp, self.in_wo_rp, self.in_b0_rp = in_wc_rp, in_wo_rp, in_b0_rp
        self.in_wc_yaw, self.in_wo_yaw, self.in_b0_yaw = in_wc_yaw, in_wo_yaw, in_b0_yaw
        self.out_wc, self.out_wo, self.out_b0 = out_wc, out_wo, out_b0
        self.z2_clamp_in_rp = z2_clamp_in_rp
        self.z2_clamp_in_yaw = z2_clamp_in_yaw
        self.z2_clamp_out = z2_clamp_out
        self.max_angle = max_angle
        self.max_rate = max_rate

        # 每个环的状态 (N, A, 5)：每个一阶 LADRC 保存 z1/z2/u_prev。
        self._z1 = torch.zeros(self.N, self.A, 5, device=device)
        self._z2 = torch.zeros(self.N, self.A, 5, device=device)
        self._u_prev = torch.zeros(self.N, self.A, 5, device=device)

    # ──────────────────────────────────────────────
    # 每回合 reset：清空 z1/z2/u_prev，防止上一回合的陈旧扰动估计
    # 泄漏到下一回合。
    # ──────────────────────────────────────────────
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._z1.zero_(); self._z2.zero_(); self._u_prev.zero_()
        else:
            self._z1[env_ids] = 0.0
            self._z2[env_ids] = 0.0
            self._u_prev[env_ids] = 0.0

    # ──────────────────────────────────────────────
    # 一阶 LADRC 离散更新，对给定的一组环索引向量化执行
    # （这些环共用同一套标量增益）。
    #   r, y: (N, A, len(loops))；返回 u: (N, A, len(loops))
    # ──────────────────────────────────────────────
    def _leso_step(self, loops, r, y, wc, wo, b0, h, z2_clamp):
        z1 = self._z1[..., loops]
        z2 = self._z2[..., loops]
        up = self._u_prev[..., loops]
        beta1 = 2.0 * wo
        beta2 = wo * wo
        e = z1 - y                                   # LESO 使用上一次实际施加的输入 u_prev
        z1 = z1 + h * (z2 - beta1 * e + b0 * up)
        z2 = (z2 + h * (-beta2 * e)).clamp(-z2_clamp, z2_clamp)   # 在扰动量纲下钳制
        u = (wc * (r - z1) - z2) / b0                # LSEF 控制律
        self._z1[..., loops] = z1
        self._z2[..., loops] = z2
        self._u_prev[..., loops] = u                 # 暂存值；内环会被 commit_inner_applied 覆盖
        return u

    # ──────────────────────────────────────────────
    # 外环角度环 —— 在 RL/控制频率调用（h = sim.dt × decimation）。
    #   输入：期望/实测 roll、pitch（各 (N,A,1)）。
    #   输出：角速率 setpoint p_sp、q_sp（各 (N,A,1)，已限幅）。
    # ──────────────────────────────────────────────
    def outer_step(self, roll_des, pitch_des, roll_meas, pitch_meas, h):
        r = torch.cat([roll_des, pitch_des], dim=-1)      # (N,A,2)
        y = torch.cat([roll_meas, pitch_meas], dim=-1)
        u = self._leso_step([L_OUT_ROLL, L_OUT_PITCH], r, y,
                            self.out_wc, self.out_wo, self.out_b0, h, self.z2_clamp_out)
        u = u.clamp(-self.max_rate, self.max_rate)
        p_sp = u[..., 0:1]
        q_sp = u[..., 1:2]
        return p_sp, q_sp

    # ──────────────────────────────────────────────
    # 内环角速率环 —— 在物理频率调用（h = sim.dt），传入最新实测角速率。
    #   p_sp/q_sp/r_sp：角速率 setpoint (N,A,1)；omega：实测机体角速率
    #   (N,A,3)。返回力矩 (N,A,3) [N·m]。
    #   若力矩方向相反，对返回的 moment 取负（符号须与仿真器核对）。
    # ──────────────────────────────────────────────
    def inner_step(self, p_sp, q_sp, r_sp, omega, h):
        # roll/pitch 共用增益
        r_rp = torch.cat([p_sp, q_sp], dim=-1)            # (N,A,2)
        y_rp = omega[..., 0:2]
        u_rp = self._leso_step([L_IN_ROLL, L_IN_PITCH], r_rp, y_rp,
                            self.in_wc_rp, self.in_wo_rp, self.in_b0_rp, h, self.z2_clamp_in_rp)
        # yaw
        y_yaw = omega[..., 2:3]
        u_yaw = self._leso_step([L_IN_YAW], r_sp, y_yaw,
                            self.in_wc_yaw, self.in_wo_yaw, self.in_b0_yaw, h, self.z2_clamp_in_yaw)
        moment = torch.cat([u_rp[..., 0:1], u_rp[..., 1:2], u_yaw], dim=-1)   # (N,A,3)
        return moment

    # ──────────────────────────────────────────────
    # 把实际施加的总力矩写回内环 u_prev。
    # 在前馈/扰动项修改过 moment 且 wrench 已施加之后调用。这样 LESO 把前馈
    # 当作已知输入，只估计真正的残余扰动，前馈与观测器便不会重复补偿。
    #   applied_moment: (N,A,3)
    # ──────────────────────────────────────────────
    def commit_inner_applied(self, applied_moment):
        self._u_prev[..., [L_IN_ROLL, L_IN_PITCH, L_IN_YAW]] = applied_moment

    # ──────────────────────────────────────────────
    # 加速度 -> 姿态 setpoint（考虑偏航）。
    # 先把世界系的期望水平加速度旋转到偏航系，再映射成 roll/pitch_des，
    # 这样平台偏航时控制方向仍然正确。
    #   a_des_w: (N,A,3) 世界系加速度；quat: (N,A,4) wxyz；g 为标量。
    #   返回 roll_des、pitch_des，各 (N,A,1)。
    # ──────────────────────────────────────────────
    def accel_to_attitude_setpoint(self, a_des_w, quat, g, max_angle=None):
        if max_angle is None:
            max_angle = self.max_angle
        psi = yaw_from_quat(quat)                          # (N,A)
        ax, ay = a_des_w[..., 0], a_des_w[..., 1]          # (N,A)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        ax_y = cpsi * ax + spsi * ay                       # 旋转到偏航系
        ay_y = -spsi * ax + cpsi * ay
        pitch_des = (ax_y / g).clamp(-max_angle, max_angle)
        roll_des = (-ay_y / g).clamp(-max_angle, max_angle)
        return roll_des.unsqueeze(-1), pitch_des.unsqueeze(-1)

    # ──────────────────────────────────────────────
    # 加速度 -> 沿机体 z 轴的总推力（N）。仿真中直接施加；
    # 在真机上则映射为归一化油门。
    #   a_des_w: (N,A,3)；quat: (N,A,4)；mass、g、max_thrust 为标量。
    #   返回 (N,A)。
    # ──────────────────────────────────────────────
    def accel_to_thrust(self, a_des_w, quat, mass, g, max_thrust):
        t_des = a_des_w.clone()
        t_des[..., 2] = t_des[..., 2] + g                  # 抵消重力
        ez = torch.zeros_like(a_des_w); ez[..., 2] = 1.0
        b3 = quat_rotate(quat, ez)                         # 世界系下的机体 z 轴
        thrust = mass * torch.sum(t_des * b3, dim=-1)      # (N,A)
        return thrust.clamp(0.0, max_thrust)

    # ──────────────────────────────────────────────
    # 重心前馈力矩（可选，在 inner_step 之后叠加）。
    # 抵消推力作用在偏移重心上产生的翻转力矩。
    #   com_off_b: (N,A,3) 机体系下重心相对 base 的偏移；thrust_z:
    #   (N,A,1) 施加的推力；gain 为标量。返回 (N,A,3)。
    #   符号须在仿真中核对：开启前馈后若 orientation/存活率改善则符号正确，
    #   否则翻转 gain 的符号。
    # ──────────────────────────────────────────────
    def cog_feedforward(self, com_off_b, thrust_z, gain):
        cx = com_off_b[..., 0:1]
        cy = com_off_b[..., 1:2]
        T = thrust_z
        ff_roll = gain * T * cy        # 抵消 -T·cy
        ff_pitch = -gain * T * cx      # 抵消 +T·cx
        ff_yaw = torch.zeros_like(ff_roll)
        return torch.cat([ff_roll, ff_pitch, ff_yaw], dim=-1)

    # ──────────────────────────────────────────────
    # 负载（所搬运杆）的载荷前馈力矩。
    # 抓住后，半根杆的质量（m_load）悬挂在夹爪位置 r_grip_b（相对质心）处，
    # 产生扰动力矩 τ = r × F；本前馈将其抵消。门控在"该机正在抓取"上。
    # 在 inner_step 之后、commit_inner_applied 之前施加，使 LESO 把它当作
    # 已知输入。cog_feedforward 只覆盖机体 + 机械臂自重，不含负载，故这是
    # 额外的一项。
    #   m_load: 标量 kg；r_grip_b: (N,A,3) 夹爪相对机体质心的位置（机体系）；
    #   grav_b_unit: (N,A,3) projected_gravity_b；g: 标量重力；
    #   gate: (N,A,1) 0/1 抓取标志；gain: 符号/强度调节。
    #   返回 (N,A,3)。
    # ──────────────────────────────────────────────
    def payload_torque_feedforward(self, m_load, r_grip_b, grav_b_unit, g, gate, gain=1.0):
        F_b = (m_load * g) * grav_b_unit                       # 负载重力（机体系，指向下方）
        tau_dist = torch.linalg.cross(r_grip_b, F_b, dim=-1)   # 扰动力矩 = r × F
        return (-gain) * gate * tau_dist                       # 前馈将其抵消


# ──────────────────────────────────────────────────────────────
# 离线自检（纯 torch，不依赖 Isaac Lab）：用内环跟踪一个角速率阶跃，
# 确认 LESO 能抑制恒定扰动。
#   python cascade_ladrc.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cpu"
    N, A = 4, 2                                   # A=2 用于检验双机的张量形状
    J = 0.013                                     # 假定的单轴转动惯量
    ladrc = CascadeLADRC(N, num_agents=A, device=dev, in_b0_rp=1.0 / J)

    # 简化被控对象：单轴角速率 ω，ω̇ = (1/J)·τ + 扰动。
    # 测试内环 roll 跟踪 1.0 rad/s 的阶跃。
    omega = torch.zeros(N, A, 3, device=dev)
    p_sp = torch.ones(N, A, 1, device=dev) * 1.0  # 目标角速率 1 rad/s
    q_sp = torch.zeros(N, A, 1, device=dev)
    r_sp = torch.zeros(N, A, 1, device=dev)
    disturb = 5.0                                 # 以角加速度 rad/s² 表示的恒定扰动

    h = 1.0 / 1000.0                              # 内环 1 kHz
    for k in range(2000):
        moment = ladrc.inner_step(p_sp, q_sp, r_sp, omega, h)   # (N,A,3) 力矩
        # 简化刚体：ω̇ = moment/J + 扰动（仅 roll 轴）
        ang_acc = moment / J
        ang_acc[..., 0] += disturb
        omega = omega + h * ang_acc
        ladrc.commit_inner_applied(moment)
        if k % 400 == 0:
            print(f"step {k:4d}  roll_rate={omega[0,0,0].item():+.4f} (target 1.0)  "
                  f"moment_roll={moment[0,0,0].item():+.5f}  z2_roll={ladrc._z2[0,0,L_IN_ROLL].item():+.3f}")

    err = (omega[..., 0] - 1.0).abs().max().item()
    print(f"\nFinal roll-rate tracking error (with constant disturbance): {err:.4f} rad/s")
    print("Small error (<~0.05) means the LESO rejected the disturbance: z2 clamp units correct.")
    print("Shapes stay (N,A,·) throughout: the two-drone case (A=2) reuses this module directly.")
