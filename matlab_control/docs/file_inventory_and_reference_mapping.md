# flyarm MATLAB/Simulink 文件总索引与论文映射

## 1. 结论

合并后只保留三份说明文档：

```text
docs/file_inventory_and_reference_mapping.md   唯一总索引
docs/flyarm_kinematics_dynamics_control.md    主论文技术说明
docs/isaac_coupled_dynamics_mapping.md        Isaac-MATLAB 对应关系
```

被合并的旧文档内容已经进入上述文件：

| 原文件 | 合并去向 |
|---|---|
| `pdf_reference_index.md` | 本文件第 3 节“参考资料映射” |
| `pd_vs_ladrc_comparison.md` | `flyarm_kinematics_dynamics_control.md` 的控制验证部分 |
| `simulink_closed_loop_notes.md` | `flyarm_kinematics_dynamics_control.md` 的 Simulink 闭环部分 |

正式阅读时不要再从文件夹里乱点，按下面顺序走。

## 2. 推荐打开顺序与验证内容

### 2.1 第一遍：理解整体结构

| 顺序 | 文件 | 验证/理解什么 |
|---:|---|---|
| 1 | `README.md` | 这个包的定位：MATLAB/Simulink 验证动力学和底层控制，不替代 Isaac 抓取仿真 |
| 2 | `docs/file_inventory_and_reference_mapping.md` | 每个文件是什么、参考什么资料、论文放在哪里 |
| 3 | `docs/flyarm_kinematics_dynamics_control.md` | 论文主线：参数、运动学、动力学、控制器、Simulink、结果 |
| 4 | `docs/isaac_coupled_dynamics_mapping.md` | Isaac 的关节 PD、夹爪、负载、接触如何映射到 MATLAB 公式 |

### 2.2 第二遍：按代码数据流验证

| 顺序 | 文件 | 要验证的内容 |
|---:|---|---|
| 1 | `matlab/flyarm_params_from_urdf.m` | URDF 是否读取到 flyarm 平行夹爪的真实质量、惯量、旋翼位置、关节类型 |
| 2 | `matlab/flyarm_articulation_config.m` | 机械臂 4 关节 + 2 个 prismatic 夹爪的 PD、限位、姿态目标是否和 Isaac 项目一致 |
| 3 | `matlab/flyarm_motor_mixer.m` | 4 个旋翼转速平方是否能合成 `[T, Mx, My, Mz]` |
| 4 | `matlab/flyarm_motor_model.m` | 电机是否按一阶惯性响应，而不是瞬时达到命令 |
| 5 | `matlab/flyarm_quad_dynamics.m` | 四旋翼 12 状态刚体动力学是否能悬停平衡 |
| 6 | `matlab/flyarm_articulated_kinematics.m` | 机械臂质心、夹爪中心、关节轴是否由 URDF 链式运动学得到 |
| 7 | `matlab/flyarm_link_kinematic_terms.m` | 每个连杆 COM、惯量、雅可比是否为动力学准备好 |
| 8 | `matlab/flyarm_manipulator_dynamics.m` | 第 1 层 `M_a/C_a/G_a` 是否对称、正定、有限 |
| 9 | `matlab/flyarm_aerial_manipulator_dynamics.m` | 第 2 层 `M/C/G`、`M_ba/M_ab`、`J_c^T F_c` 接口是否存在 |
| 10 | `matlab/flyarm_articulated_pd_control.m` | Isaac-like 关节位置 PD 是否生成有限且限幅的关节力/力矩 |
| 11 | `matlab/flyarm_arm_computed_torque_control.m` | computed torque 是否能调用第 1 层动力学生成关节控制输入 |
| 12 | `matlab/flyarm_pid_init.m`、`matlab/flyarm_attitude_altitude_pid_control.m` | PID 飞控是否能输出推力、姿态力矩和有界积分项 |
| 13 | `matlab/flyarm_ladrc_init.m`、`matlab/flyarm_cascade_ladrc_control.m` | LADRC 的 `b0/wc/wo/ESO/z2` 是否初始化并输出有限控制力矩 |
| 14 | `matlab/flyarm_trajectory_tracking.m` | `hover_step/payload_step/circle` 是否能跑出轨迹日志 |
| 15 | `matlab/flyarm_coupled_dynamics.m`、`matlab/flyarm_coupled_tracking.m` | 机械臂伸展、夹爪闭合、负载附着是否能形成机体扰动 |
| 16 | `matlab/flyarm_compare_pid_ladrc.m` | PID 与 LADRC 对比指标是否能复现 |
| 17 | `matlab/run_flyarm_matlab_demo.m` | MATLAB 结果图和 `.mat` 日志是否能生成 |
| 18 | `simulink/build_flyarm_closed_loop_model.m`、`matlab/run_flyarm_simulink_closed_loop.m` | Simulink 闭环模型是否能生成、运行、导出图 |
| 19 | `simulink/build_flyarm_control_block_diagram.m` | 论文/答辩展示用控制框图是否能生成 |
| 20 | `tests/run_first_version_checks.m` | 一键回归：所有关键接口和结果是否仍可运行 |

### 2.3 第三遍：按论文写作打开

| 论文小节 | 应该打开的文件 | 论文放什么 |
|---|---|---|
| 参数来源 | `flyarm_params_from_urdf.m`、`flyarm_articulation_config.m` | 参数表：质量、惯量、旋翼位置、关节、夹爪、推重比、电机估计 |
| 四旋翼动力学 | `flyarm_quad_dynamics.m`、`flyarm_motor_mixer.m`、`flyarm_motor_model.m` | 平动/转动方程、混控矩阵、电机一阶响应 |
| 机械臂本体动力学 | `flyarm_link_kinematic_terms.m`、`flyarm_manipulator_dynamics.m` | `M_a(q)qddot+C_a(q,qdot)qdot+G_a(q)=tau_a` 和各项含义 |
| 飞行器-机械臂耦合动力学 | `flyarm_aerial_manipulator_dynamics.m`、`isaac_coupled_dynamics_mapping.md` | `M(chi)nudot+C(chi,nu)nu+G(chi)=tau_rotor+tau_joint+J_c^T F_c+d` |
| 控制器 | `flyarm_attitude_altitude_pid_control.m`、`flyarm_cascade_ladrc_control.m`、`flyarm_arm_computed_torque_control.m` | PID 飞控、LADRC 飞控、关节 PD 和 computed torque 的结构与作用边界 |
| MATLAB 实验 | `run_flyarm_matlab_demo.m`、`output/*.png` | hover、payload、circle、arm_reach_payload 结果图 |
| Simulink 实验 | `run_flyarm_simulink_closed_loop.m`、`flyarm_closed_loop_compare.slx`、`flyarm_control_block_diagram.slx` | 可运行闭环模型、论文展示控制框图和 PID/LADRC 对比图 |
| Isaac 分工 | `isaac_coupled_dynamics_mapping.md` | MATLAB 负责名义动力学和控制验证，Isaac 负责接触抓取/RL/视觉触觉/双机 |

## 3. 参考资料映射

### 3.1 参数来源

当前 MATLAB/Simulink 包默认使用：

```text
D:/20489/Desktop/flyarm_aerial_manipulation/urdf/flyarm_parallel_gripper.urdf
```

这份 URDF 对应论文展示路线中的 flyarm 平行夹爪模型。`matlab/flyarm_params_from_urdf.m` 从它读取：

- 总质量和各连杆质量；
- base_link、机械臂连杆、夹爪连杆惯量；
- 旋翼安装位置；
- 机械臂和夹爪关节类型、轴向、限位、effort、velocity；
- 平行夹爪 prismatic 关节结构。

非 URDF 直接给出的参数单独标明来源：

| 参数 | 当前来源 | 文件 |
|---|---|---|
| `J_control = diag(1/80,1/80,1/130)` | Isaac/LADRC 控制量级标定，作为底层飞控等效惯量 | `flyarm_params_from_urdf.m` |
| `thrustToWeight = 1.9` | Isaac 项目推重比配置 | `flyarm_params_from_urdf.m` |
| `maxOmega = 900 rad/s`、`tau_m = 0.05 s` | 第一版电机估计值，后续需要台架或日志辨识 | `flyarm_params_from_urdf.m`、`flyarm_motor_model.m` |
| 机械臂关节 PD | Isaac 关节 drive 抽象 | `flyarm_articulation_config.m` |
| 平行夹爪 PD `kp=2000 N/m, kd=100 Ns/m` | 参考官方 Franka 类平行夹爪量级 | `flyarm_articulation_config.m` |
| `qStow/qStraight` | flyarm 折叠/伸展验证姿态 | `flyarm_articulation_config.m` |
| `payloadMassDefault = 0.15 kg` | 抓取负载扰动验证估计值 | `flyarm_articulation_config.m` |

结论：公式结构参考论文/教材的建模形式，数值参数使用我们自己的 flyarm 平行夹爪 URDF 和项目控制配置。

### 3.2 四旋翼扫描资料

扫描 PDF 第一版只作为建模路线参考，不直接套用里面的样机数值。

| 资料 | 用途 |
|---|---|
| 四旋翼第二章 飞行器原理 | 旋翼升力、反扭矩、电机/桨模型 |
| 四旋翼第三章 刚体运动方程 | 6DOF 平动/转动方程、欧拉角/角速度关系 |
| 四旋翼第五章 小扰动线性化 | 悬停附近线性化和状态方程写法 |
| 四旋翼第六章 姿态角控制 | 姿态角/角速度控制结构 |
| 四旋翼第七章 垂直速度和高度控制 | 高度外环和推力控制 |
| 四旋翼第十章 悬停控制 | 悬停控制、扰动响应、仿真实例 |

论文里可以写“参考四旋翼教材中的建模与控制框架，并将参数替换为 flyarm URDF/CAD 参数”，不要写成“完全复现教材全部公式”。

### 3.3 空中机械臂论文映射

| 资料 | 对当前 MATLAB 包的作用 |
|---|---|
| Yang/Lee quadrotor-manipulator dynamics | 支撑飞行器-机械臂耦合动力学、机体/机械臂分块质量矩阵、分层控制思想 |
| Jimenez-Cano aerial robot with multi-link arm | 支撑飞行器挂载多连杆机械臂的建模写法 |
| Cooperative aerial manipulation papers | 支撑后续双机负载、接触 wrench、协同搬运框架 |
| LADRC/ADRC 资料 | 支撑 ESO、`wc/wo/b0`、串级姿态/角速度控制 |
| Isaac Lab flyarm 项目代码 | 支撑关节 PD、动作语义、夹爪闭合、抓取/搬运逻辑 |
| Osprey/Deshmukh/Xie/Zhou 等论文 | 主要用于 Isaac/RL/视觉伺服和后续扩展，不是当前 MATLAB 第一版的全部公式来源 |

## 4. 文件清单

### 4.1 顶层

| 文件 | 内容 |
|---|---|
| `README.md` | 总入口、目录结构、打开顺序、运行命令、论文放置建议 |
| `docs/file_inventory_and_reference_mapping.md` | 本文件，唯一总索引 |

### 4.2 文档

| 文件 | 内容 |
|---|---|
| `docs/flyarm_kinematics_dynamics_control.md` | 主论文技术说明：运动学、动力学、PID/LADRC、关节 PD、Simulink、结果 |
| `docs/isaac_coupled_dynamics_mapping.md` | Isaac 与 MATLAB/Simulink 的接口映射和分工 |

### 4.3 MATLAB 函数

| 文件 | 内容 |
|---|---|
| `flyarm_params_from_urdf.m` | 从平行夹爪 URDF 读取参数，并补充第一版控制参数 |
| `flyarm_motor_mixer.m` | 旋翼混控矩阵 |
| `flyarm_motor_model.m` | 电机一阶响应 |
| `flyarm_quad_dynamics.m` | 四旋翼 12 状态刚体动力学 |
| `flyarm_articulation_config.m` | 机械臂/夹爪 PD、限位、目标姿态 |
| `flyarm_articulated_kinematics.m` | URDF 链式运动学、机械臂 COM 和夹爪中心 |
| `flyarm_link_kinematic_terms.m` | 连杆 COM、惯量、雅可比 |
| `flyarm_manipulator_dynamics.m` | 第 1 层机械臂本体 `M_a/C_a/G_a` |
| `flyarm_aerial_manipulator_dynamics.m` | 第 2 层飞行器-机械臂 `M/C/G`、`M_ba/M_ab`、`J_c` |
| `flyarm_articulated_pd_control.m` | Isaac-like 关节 PD |
| `flyarm_arm_computed_torque_control.m` | 机械臂 computed torque 控制接口 |
| `flyarm_attitude_altitude_control.m` | 旧版 PD 飞控基线，保留用于兼容 |
| `flyarm_pid_init.m` | PID 飞控积分状态和限幅初始化 |
| `flyarm_attitude_altitude_pid_control.m` | PID 飞控基线 |
| `flyarm_ladrc_init.m` | LADRC 参数和 ESO 状态初始化 |
| `flyarm_cascade_ladrc_control.m` | 串级 LADRC 姿态/角速度控制 |
| `flyarm_trajectory_tracking.m` | hover、payload、circle 轨迹仿真 |
| `flyarm_coupled_dynamics.m` | 24 状态机体-机械臂-负载耦合 ODE |
| `flyarm_coupled_tracking.m` | arm_reach_payload 耦合闭环仿真 |
| `flyarm_compare_pid_ladrc.m` | PID/LADRC 对比指标 |
| `flyarm_compare_pd_ladrc.m` | 旧名称兼容入口，当前内部生成 PID/LADRC 对比 |
| `run_flyarm_matlab_demo.m` | 一键生成 MATLAB 图和日志 |
| `run_flyarm_simulink_closed_loop.m` | 一键生成并运行 Simulink 闭环 |
| `flyarm_simulink_step_pid.m` | Simulink PID step wrapper |
| `flyarm_simulink_step_pd.m` | 旧名称兼容 wrapper，当前路由到 PID |
| `flyarm_simulink_step_ladrc.m` | Simulink LADRC step wrapper |
| `flyarm_simulink_step_impl.m` | Simulink 共用闭环 step 实现 |

### 4.4 Simulink

| 文件 | 内容 |
|---|---|
| `simulink/build_flyarm_closed_loop_model.m` | 生成可运行 PID/LADRC 闭环模型 |
| `simulink/build_flyarm_control_block_diagram.m` | 生成论文/答辩展示用控制框图 |
| `simulink/flyarm_closed_loop_compare.slx` | 当前主 Simulink 闭环模型 |
| `simulink/flyarm_control_block_diagram.slx` | 论文/答辩展示用 Simulink 控制框图 |

### 4.5 输出

| 文件 | 内容 |
|---|---|
| `output/flyarm_hover_step.png/.mat` | 高度阶跃总览图和日志 |
| `output/flyarm_payload_step.png/.mat` | 负载扰动总览图和日志 |
| `output/flyarm_circle.png/.mat` | 圆轨迹总览图和日志 |
| `output/flyarm_coupled_arm_payload.png/.mat` | 机械臂伸展、夹爪闭合、负载附着总览图和日志 |
| `output/flyarm_pid_vs_ladrc_payload_step.png/.mat` | MATLAB PID/LADRC 对比总览图和日志 |
| `output/flyarm_simulink_closed_loop_compare.png/.mat` | Simulink PID/LADRC 对比总览图和日志 |
| `output/*_altitude_tracking.png` | 单图版高度跟踪，横轴为 `time (s)`，纵轴为高度 `z (m)` |
| `output/*_attitude_response.png`、`output/*_attitude_norm.png` | 单图版姿态角或姿态误差范数 |
| `output/*_thrust_command.png` | 单图版总推力命令 |
| `output/*_arm_joint_motion.png` | 单图版机械臂 base/B/C/D 关节运动 |
| `output/*_arm_coupling_torque.png` | 单图版机械臂运动引起的机体耦合力矩 |
| `output/*_payload_coupling_torque.png` | 单图版负载偏心引起的机体耦合力矩 |
| `output/*_minor_coupling_torques.png` | `arm x` 主耦合力矩单独绘制后，小量级耦合力矩的分开展示图 |
| `output/*_pitch_response.png`、`output/*_ladrc_z2_pitch.png` | 单图版 pitch 响应和 LADRC 扰动估计状态 |

### 4.6 测试

| 文件 | 内容 |
|---|---|
| `tests/run_first_version_checks.m` | 总回归检查，覆盖 URDF、混控、电机、四旋翼、机械臂 M/C/G、耦合 M/C/G、computed torque、PID/LADRC、关节 PD、Simulink |

## 5. 一键验证命令

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/tests'); run_first_version_checks"
```

通过后说明：

- URDF 参数读取正常；
- 旋翼混控矩阵非奇异；
- 电机响应向命令收敛；
- 悬停加速度近似为零；
- 机械臂 `M_a` 和系统 `M` 对称、正定；
- `M_ba/M_ab` 耦合块存在且非零；
- computed torque 和关节 PD 有限且满足 effort 限幅；
- PID/LADRC 对比能生成；
- Simulink 闭环模型能保存并运行。

## 6. 论文应该放什么

建议放在论文的“机器人建模与控制验证”章节中，结构如下：

1. **参数来源**：说明参数来自 flyarm 平行夹爪 URDF，而不是教材样机参数。
2. **四旋翼动力学**：写平动方程、转动方程、旋翼混控和电机一阶响应。
3. **机械臂本体动力学**：写 `M_a(q)qddot+C_a(q,qdot)qdot+G_a(q)=tau_a`。
4. **飞行器-机械臂耦合动力学**：写 `M(chi)nudot+C(chi,nu)nu+G(chi)=tau_rotor+tau_joint+J_c^T F_c+d`。
5. **控制器设计**：PID 飞控基线、串级 LADRC、ESO 扰动估计、关节 PD 和 computed torque 关节控制接口。
6. **MATLAB 验证实验**：hover、payload、circle、arm_reach_payload。
7. **Simulink 闭环验证**：展示 `flyarm_closed_loop_compare.slx` 的框图和 PID/LADRC 对比曲线。
8. **与 Isaac Lab 的分工**：MATLAB 验证名义动力学和控制，Isaac 验证接触抓取、摩擦、掉落、视觉/触觉和双机协作。

当前可以报告的核心数值：

```text
payload_step:
PID 最大姿态误差范数    由脚本自动生成
LADRC 最大姿态误差范数  由脚本自动生成

arm_reach_payload:
控制器 LADRC
最大姿态误差范数 0.0377 rad
最大机械臂耦合力矩 0.1970 Nm
最大负载偏心力矩 0.0550 Nm
最终高度 0.9338 m
```

论文中要写清边界：

- 这是名义动力学和控制验证，不是完整 PhysX 接触求解器；
- 两指接触、摩擦滑移、掉落和抓取成功率仍由 Isaac Lab 验证；
- 电机参数 `kf/km/tau_m` 当前是第一版估计，后续需要电机台架或飞行日志辨识。
