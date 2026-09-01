# flyarm 运动学、动力学、参数辨识与控制分析

## 1. 建模定位

本文的 MATLAB/Simulink 模型用于验证 flyarm 空中机械臂的动力学与底层控制合理性。它和 Isaac Lab 的分工是：

| 平台 | 负责内容 |
|---|---|
| MATLAB/Simulink | 运动学、动力学、控制器、电机响应、参数辨识、轨迹跟踪 |
| Isaac Lab | 高维接触抓取、强化学习训练、触觉/视觉、双机协作 |
| MuJoCo | CPU 物理交叉验证 |

第一版采用 `flyarm_parallel_gripper.urdf`，因为它对应最新论文/展示路线中的平行夹爪机构。

## 2. 坐标系与结构参数

flyarm 采用四旋翼常用机体系：

```text
x_b: 机头前方
y_b: 机体左方
z_b: 机体上方
```

平行夹爪 URDF 解析结果：

```text
总质量 m = 2.7320 kg
旋翼数量 = 4
夹爪关节 = prismatic, prismatic
控制用惯量 J_control = diag(1/80, 1/80, 1/130) kg*m^2
```

这里要区分两个惯量：

- `J_base_urdf`：URDF 中 `base_link` 自身惯量，只描述机体本体。
- `J_control`：论文控制模型采用的等效控制惯量，来自 Isaac LADRC 设计中的 `b0 ~= 1/J` 标定，包含挂臂后控制量级。

## 3. 旋翼推力与力矩

每个旋翼建模为：

```text
f_i = k_f * omega_i^2
tau_i = k_m * omega_i^2
```

第一版没有电机台架数据，因此按最大推重比估计：

```text
T_max = 1.9 * m * g
k_f = (T_max / 4) / omega_max^2
omega_hover = sqrt((m*g/4) / k_f)
```

当前验证输出：

```text
omega_hover = 652.93 rad/s
```

电机动态不是瞬时响应，而是一阶惯性：

```text
tau_m * omega_dot = omega_cmd - omega
```

这能反映电机/桨响应滞后，避免论文模型退化成“直接把总推力打到机体上”。

## 4. 混控矩阵

旋翼位置为 `r_i = [x_i, y_i, z_i]^T`，升力沿 `+z_b`。单个旋翼对机体系力矩的贡献：

```text
tau_force_i = r_i x [0, 0, f_i]^T = [y_i*f_i, -x_i*f_i, 0]^T
tau_yaw_i = s_i * k_m * omega_i^2
```

因此：

```text
[T; Mx; My; Mz] = B * [omega_1^2; omega_2^2; omega_3^2; omega_4^2]
```

其中第 `i` 列为：

```text
[k_f; y_i*k_f; -x_i*k_f; s_i*k_m]
```

`s_i` 是旋翼反扭矩方向，第一版采用 `[+1, -1, +1, -1]` 的交替方向。

## 5. 刚体动力学与耦合动力学

状态定义：

```text
x = [p_w; v_w; phi theta psi; omega_b]
```

平移动力学：

```text
m * v_dot = R(phi,theta,psi) * [0;0;T] - m*g*[0;0;1] + F_dist
```

转动动力学：

```text
J * omega_dot = M + tau_arm + tau_payload - omega x (J*omega)
```

其中：

- `tau_arm` 表示机械臂运动引起的等效扰动力矩；
- `tau_payload` 表示抓取负载后，由夹爪偏心和负载重力产生的扰动力矩；
- 第一版 demo 中 `payload_step` 用小力矩阶跃模拟抓取负载扰动。

### 5.1 第1层：机械臂本体动力学

机械臂关节层现在显式写成：

```text
M_a(q) * qddot + C_a(q,qdot) * qdot + G_a(q) = tau_a
```

其中：

- `q`：6 个主动关节 `[base_rotate, axis_B, axis_C, axis_D, gripper_right, gripper_left]`；
- `M_a(q)`：机械臂质量矩阵，来自各连杆质量、惯量和 COM 雅可比；
- `C_a(q,qdot)qdot`：离心/科氏项，用 `M_a(q)` 对 `q` 的有限差分近似 Christoffel 项；
- `G_a(q)`：重力广义力；
- `tau_a`：关节电机/舵机输出力矩或力。

核心计算关系是：

```text
T = 1/2 * sum_i( m_i * v_i^T v_i + omega_i^T I_i omega_i )
M_a = sum_i( m_i * Jv_i^T Jv_i + Jw_i^T I_i Jw_i )
G_a = -sum_i( Jv_i^T * m_i * g_b )
```

对应 MATLAB 文件：

```text
matlab/flyarm_link_kinematic_terms.m
matlab/flyarm_manipulator_dynamics.m
matlab/flyarm_arm_computed_torque_control.m
```

这层的意义是：机械臂不再只是“给机体一个扰动”，而是先有自己的关节动力学。后面做 computed torque、阻抗控制或关节轨迹跟踪，都可以基于这层模型。

### 5.2 第2层：飞行器-机械臂耦合动力学

系统层使用机体系广义速度：

```text
chi = [p_w; roll pitch yaw; q]
nu  = [v_b; omega_b; qdot]
```

耦合动力学写成：

```text
M(chi) * nudot + C(chi,nu) * nu + G(chi)
    = tau_rotor + tau_joint + J_c^T * F_c + d
```

其中：

- `tau_rotor = [F_b; M_b; 0_q]`：旋翼总推力和机体系力矩；
- `tau_joint = [0_6; tau_a]`：机械臂关节广义力；
- `J_c^T F_c`：夹爪接触力映射成系统广义力；
- `d`：风扰、参数误差、未建模摩擦、接触误差等总扰动。

这里 `C(chi,nu)nu` 由两部分组成：机体系浮动基座的 Newton-Euler 输运项，以及 `M(chi)` 对机械臂关节坐标的有限差分 Christoffel 项。这样能覆盖机械臂运动引起的主要速度耦合，而不需要手写一大套符号展开式。

质量矩阵按机体和机械臂分块：

```text
M = [M_bb  M_ba
     M_ab  M_aa]
```

这里最关键的是 `M_ba/M_ab`。它表示机械臂关节运动会通过惯量耦合影响机体平动和转动，也就是论文里常说的“机械臂运动对飞行平台产生动态扰动”。

对应 MATLAB 文件：

```text
matlab/flyarm_aerial_manipulator_dynamics.m
```

这个函数会输出：

```text
dyn.M
dyn.Cnu
dyn.G
dyn.blocks.M_bb, M_ba, M_ab, M_aa
dyn.inputMap.rotorWrench
dyn.inputMap.jointTorque
dyn.inputMap.Jc_grip_b
```

### 5.3 原 24 状态闭环模型的作用

当前版本仍保留和 Isaac 逻辑一致的 24 状态耦合仿真：

```text
x_c = [p_w; v_w; phi theta psi; omega_b; q; qdot]
q   = [base_rotate; axis_B; axis_C; axis_D; gripper_right; gripper_left]
```

机械臂和夹爪不再只是一个外部力矩阶跃，而是按 Isaac 的关节位置 PD drive 抽象：

```text
q_cmd -> 关节 PD drive -> 关节力/力矩 -> 机体耦合扰动
tau_q = Kp*(q_cmd-q) + Kd*(qdot_cmd-qdot)
```

耦合扰动由三部分组成：

```text
1. 关节驱动反作用:
   revolute:  tau_base += -tau_i * axis_i
   prismatic: F_base += -F_i * axis_i, tau_base += r_i x F_base

2. 机械臂质心运动:
   a_arm_com ~= J_com(q) * qddot
   F_arm = -m_arm * a_arm_com
   tau_arm = r_arm_com x F_arm

3. 质心/负载偏心:
   tau_com = (r_com(q)-r_com(q_trim)) x R^T*[0;0;-m*g]
   tau_payload = r_grip x R^T*[0;0;-m_payload*g]
```

对应 MATLAB 文件：

```text
matlab/flyarm_articulation_config.m
matlab/flyarm_articulated_kinematics.m
matlab/flyarm_articulated_pd_control.m
matlab/flyarm_coupled_dynamics.m
matlab/flyarm_coupled_tracking.m
```

这部分主要用于跑闭环曲线和 Simulink 对比。它不是要取代上面的 `M/C/G` 耦合动力学，而是把 Isaac 中最重要的控制链路抽象成可运行仿真：

```text
URDF 关节/质量 -> 关节 PD -> 机械臂质心变化 -> 机体扰动 -> 飞控抗扰
```

论文中可以这样理解：第1层和第2层负责“建模公式站得住”，24 状态闭环模型负责“控制仿真能跑通、能画图、能比较 PD 和 LADRC”。

## 6. 控制结构

第一版控制器采用经典分层结构：

```text
轨迹/位置参考
  -> 位置/高度 PD 外环
  -> 期望加速度
  -> 期望 roll/pitch/yaw
  -> 姿态 PD 内环
  -> 总推力 T 和力矩 M
  -> 混控到四个电机
  -> 电机一阶响应
  -> 四旋翼动力学
```

这和 Isaac 项目中的分层思想一致：

```text
PPO/MAPPO 负责高层决策
LADRC/PD 负责底层稳定执行
```

后续可以把姿态 PD 内环替换为 LADRC：

```text
LESO:
e = z1 - y
z1_dot = z2 - beta1*e + b0*u_prev
z2_dot = -beta2*e

LSEF:
u = (wc*(r - z1) - z2) / b0
```

这正好对应展示仓库中的 `flyarm/cascade_ladrc.py`。

## 7. 参数辨识路线

第一版参数来源分三类：

| 参数 | 当前来源 | 后续辨识方法 |
|---|---|---|
| 质量、关节、旋翼位置 | URDF | CAD/称重复核 |
| 控制等效惯量 | LADRC 的 `b0 ~= 1/J` | 姿态阶跃响应拟合 |
| `k_f`, `k_m` | 推重比估计 | 电机台架或悬停日志最小二乘 |
| `tau_m` | 经验值 0.05 s | 电机转速阶跃响应 |
| 负载扰动力矩 | 等效阶跃 | 抓取后姿态/角速度日志反推 |

最小二乘形式可以写为：

```text
T_i = k_f * omega_i^2
tau_i = k_m * omega_i^2
```

收集多个转速点后：

```text
k_f = argmin sum_j (T_j - k_f*omega_j^2)^2
k_m = argmin sum_j (tau_j - k_m*omega_j^2)^2
```

## 8. 轨迹跟踪验证

当前实现四个场景：

1. `hover_step`：高度阶跃，验证悬停和高度控制。
2. `payload_step`：2 秒后加入负载扰动力矩，验证抗扰能力。
3. `circle`：水平圆轨迹，验证轨迹跟踪接口。
4. `arm_reach_payload`：机械臂从蜷缩到伸直、夹爪闭合并附着负载，验证机械臂/夹爪/四旋翼耦合扰动。

生成命令：

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab'); run_flyarm_matlab_demo"
```

输出图：

```text
output/flyarm_hover_step.png
output/flyarm_payload_step.png
output/flyarm_circle.png
output/flyarm_coupled_arm_payload.png
```

## 9. PID 与 LADRC 对比验证

对比目的很简单：flyarm 的机械臂伸展、夹爪闭合和负载附着都会给机体带来额外扰动力矩。PID 飞控在 PD 基础上加入有界积分项，可补偿慢变化偏差和稳态误差；LADRC 把机械臂扰动、负载扰动和模型误差统一看成总扰动，通过 ESO 在线估计并补偿。

当前对比使用同一个 MATLAB 动力学模型、同一个电机一阶响应、同一个负载扰动场景：

```text
payload_step 场景：
2 s 时加入等效负载扰动力矩，模拟抓取物体后夹爪偏心载荷对机体的影响。
```

PID 姿态控制可概括为：

```text
M = J * (Kp * attitude_error + Ki * integral(attitude_error) + Kd * rate_error)
```

串级 LADRC 结构为：

```text
位置/高度外环
  -> 期望 roll/pitch/yaw
  -> 外环 LADRC：姿态角 -> 期望角速度
  -> 内环 LADRC：角速度 -> 机体力矩
```

单轴 LADRC 写法：

```text
y_dot = b0 * u + f

e = z1 - y
z1_dot = z2 - beta1 * e + b0 * u_prev
z2_dot = -beta2 * e

u = (wc * (r - z1) - z2) / b0
```

当前结果：

```text
payload_step:
PID 最大姿态误差范数    由脚本自动生成
LADRC 最大姿态误差范数  由脚本自动生成
```

新增机械臂/夹爪耦合场景：

```text
arm_reach_payload:
控制器 LADRC
机械臂动作：蜷缩 -> 张爪 -> 伸直 -> 合爪
负载：2.2 s 后附着在夹爪中心
最大姿态误差范数 0.0377 rad
最大机械臂耦合力矩 0.1970 Nm
最大负载偏心力矩 0.0550 Nm
最终高度 0.9338 m
```

这说明：在简化动力学、等效机械臂耦合和负载扰动下，LADRC 比 PD 更能抑制持续扰动力矩。但它还不能直接等价于真机一定优于 PD，真机仍需要电机参数、传感器噪声、控制频率和真实负载扰动辨识。

## 10. Simulink 闭环验证

当前主 Simulink 模型是：

```text
simulink/flyarm_closed_loop_compare.slx
```

生成脚本：

```text
simulink/build_flyarm_closed_loop_model.m
```

运行脚本：

```text
matlab/run_flyarm_simulink_closed_loop.m
```

当前模型采用第一版最稳妥的连接方式：

```text
Clock
  ├─> PID closed loop interpreted MATLAB function
  └─> LADRC closed loop interpreted MATLAB function
        ↓
      To Workspace
```

每个 interpreted MATLAB function block 内部执行一整步闭环：

```text
参考高度/轨迹
  -> PID 或 LADRC 控制器
  -> 旋翼混控
  -> 电机一阶响应
  -> 四旋翼 6DOF 动力学
  -> 下一步状态
```

为什么第一版没有把每个公式都拆成 Simulink 小模块：第一版目标是可运行、可验证、和 MATLAB 结果一致、能输出论文图。把 6DOF 动力学、旋翼混控、电机响应、LADRC ESO 全拆成几十个方块会显著增加接口错误风险。后续如果需要答辩展示用框图，可以再拆成：

```text
Reference
Position Controller
Attitude Controller
Motor Mixer
Motor Dynamics
6DOF Plant
Scope / To Workspace
```

运行命令：

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab'); run_flyarm_simulink_closed_loop"
```

输出：

```text
output/flyarm_simulink_closed_loop_compare.png
output/flyarm_simulink_closed_loop_log.mat
```

论文可用说法：

```text
在 Simulink 中建立了 PID 与串级 LADRC 的闭环对比模型。模型包含参考输入、控制器、旋翼混控、电机一阶响应和四旋翼刚体动力学。仿真结果表明，在等效负载扰动力矩作用下，串级 LADRC 相比传统 PID 能显著降低机体姿态偏差，验证了其用于空中机械臂底层抗扰控制的合理性。
```

## 11. 论文写法建议

可以这样表述：

```text
为避免强化学习策略完全依赖黑箱仿真，本文基于 flyarm 的 URDF/CAD 参数建立了分层动力学模型。机械臂层采用 `M_a(q)qddot + C_a(q,qdot)qdot + G_a(q) = tau_a` 描述关节动力学；系统层采用 `M(chi)nudot + C(chi,nu)nu + G(chi) = tau_rotor + tau_joint + J_c^T F_c + d` 描述飞行器与机械臂的惯量耦合、接触广义力和总扰动。在 MATLAB/Simulink 中进一步加入电机一阶响应、旋翼混控、PID/LADRC 飞控和 computed torque/关节 PD 控制，用于验证底层控制系统稳定性和抗扰能力。复杂接触抓取、摩擦滑移和双机协作策略仍在 Isaac Lab 中完成训练与验证。
```

## 12. 后续工作

- 精读扫描 PDF 中第 3/5/6/7/10 章，补充书中符号推导和传递函数。
- 用真实电机/桨数据替换估计的 `k_f/k_m/tau_m`。
- 如果答辩需要更直观框图，再把当前 interpreted MATLAB function 版本拆成教材式 Simulink 子模块。
- 将 `M_a/C_a/G_a` 和 `M_ba/M_ab/J_c^T F_c` 整理成论文动力学建模小节。
- 用 Simscape Multibody 导入 URDF，验证机械臂关节轴、质心、惯量和臂动扰动。
- 如果面向真机，继续对齐 PX4 的 body-rate + thrust 接口。
