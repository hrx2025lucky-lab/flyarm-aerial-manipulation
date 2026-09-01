# flyarm MATLAB/Simulink 控制验证包

本目录是 flyarm 空中机械臂论文中的动力学建模与底层控制验证包。它不替代 Isaac Lab 的接触抓取和强化学习训练，而是用于回答三个问题：

1. flyarm 的质量、惯量、旋翼位置、机械臂关节和夹爪结构是否来自我们自己的平行夹爪 URDF。
2. 机械臂本体动力学、飞行器-机械臂耦合动力学、旋翼混控、电机响应和底层控制是否能独立跑通。
3. 在等效负载/机械臂扰动下，PID 与 LADRC 的抗扰效果是否有可复现实验曲线。

## 1. 合并后的目录结构

```text
matlab_control/
├── README.md
├── docs/
│   ├── file_inventory_and_reference_mapping.md
│   ├── flyarm_kinematics_dynamics_control.md
│   └── isaac_coupled_dynamics_mapping.md
├── matlab/
│   └── 保留当前函数文件
├── simulink/
│   ├── build_flyarm_closed_loop_model.m
│   ├── build_flyarm_control_block_diagram.m
│   ├── flyarm_closed_loop_compare.slx
│   └── flyarm_control_block_diagram.slx
├── output/
│   └── 可再生成的结果图和 .mat 日志
└── tests/
    └── run_first_version_checks.m
```

合并原则：

- `docs/file_inventory_and_reference_mapping.md` 是唯一总索引，包含文件清单、资料映射、阅读顺序、验证内容和论文放置建议。
- `docs/flyarm_kinematics_dynamics_control.md` 是主论文技术说明，讲运动学、动力学、PID/LADRC、关节 PD、Simulink 闭环和结果解释。
- `docs/isaac_coupled_dynamics_mapping.md` 只讲 Isaac Lab 与 MATLAB/Simulink 的对应关系。
- `matlab/` 中 `.m` 文件暂不合并。MATLAB 推荐一个主函数一个文件，强行合并会让调用关系和 Simulink wrapper 更难维护。
- `simulink/` 保留两类模型：`flyarm_closed_loop_compare.slx` 用于可运行闭环验证，`flyarm_control_block_diagram.slx` 用于论文/答辩展示控制框图。

## 2. 推荐打开顺序

第一遍先理解整体：

1. `README.md`：看这个包解决什么问题、怎么运行。
2. `docs/file_inventory_and_reference_mapping.md`：看每个文件是什么、参考了什么资料、论文放哪里。
3. `docs/flyarm_kinematics_dynamics_control.md`：看论文主线技术说明。
4. `docs/isaac_coupled_dynamics_mapping.md`：看 MATLAB 与 Isaac 的分工。

第二遍按验证链路打开：

1. `matlab/flyarm_params_from_urdf.m`：验证参数来自 flyarm 平行夹爪 URDF。
2. `matlab/flyarm_motor_mixer.m`、`matlab/flyarm_motor_model.m`：验证旋翼混控和电机一阶响应。
3. `matlab/flyarm_quad_dynamics.m`：验证四旋翼 6DOF 刚体动力学。
4. `matlab/flyarm_manipulator_dynamics.m`：验证第 1 层机械臂本体动力学。
5. `matlab/flyarm_aerial_manipulator_dynamics.m`：验证第 2 层飞行器-机械臂耦合动力学。
6. `matlab/flyarm_attitude_altitude_pid_control.m`、`matlab/flyarm_cascade_ladrc_control.m`：验证 PID 与 LADRC 飞控；机械臂/夹爪仍由关节 PD 验证。
7. `matlab/run_flyarm_matlab_demo.m`：生成 MATLAB 结果图。
8. `matlab/run_flyarm_simulink_closed_loop.m`：生成/运行 Simulink 闭环模型。
9. `simulink/build_flyarm_control_block_diagram.m`：生成论文展示用控制框图。
10. `tests/run_first_version_checks.m`：最终一键回归检查。

## 3. 运行命令

一键检查：

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/tests'); run_first_version_checks"
```

生成 MATLAB 演示图：

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab'); run_flyarm_matlab_demo"
```

该命令会同时生成两类图片：

- `flyarm_*.png`：多子图总览图，适合快速检查一次实验。
- `flyarm_*_altitude_tracking.png`、`*_attitude_response.png`、`*_arm_coupling_torque.png` 等：单图版，适合论文插图和逐项讲解。

在 MATLAB 界面中实时弹出图窗，便于调参：

```matlab
cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab')
run_flyarm_matlab_demo(show=true)
```

生成并运行 Simulink 闭环：

```powershell
& 'D:\matlab2026\bin\matlab.exe' -batch "cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab'); run_flyarm_simulink_closed_loop"
```

在 MATLAB/Simulink 界面中打开模型并弹出结果图：

```matlab
cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/matlab')
run_flyarm_simulink_closed_loop(show=true, openModel=true)
```

生成论文/答辩展示用 Simulink 控制框图：

```matlab
cd('D:/20489/Desktop/flyarm_aerial_manipulation/matlab_control/simulink')
build_flyarm_control_block_diagram
open_system('flyarm_control_block_diagram')
```

## 4. 论文中应该放什么

论文建议放在“机器人建模与控制验证”章节，不要把它写成替代 Isaac Lab 的完整抓取仿真。

可以放：

- flyarm 平行夹爪 URDF 参数来源表：质量、惯量、旋翼位置、机械臂关节、夹爪类型。
- 四旋翼刚体动力学：平动方程、转动方程、旋翼混控、电机一阶响应。
- 第 1 层机械臂本体动力学：
  `M_a(q) qddot + C_a(q,qdot) qdot + G_a(q) = tau_a`
- 第 2 层飞行器-机械臂耦合动力学：
  `M(chi) nudot + C(chi,nu)nu + G(chi) = tau_rotor + tau_joint + J_c^T F_c + d`
- 控制结构图：轨迹/参考 -> PID 或 LADRC -> 混控 -> 电机 -> 四旋翼/机械臂耦合模型。机械臂/夹爪关节仍采用 PD。展示图对应 `simulink/flyarm_control_block_diagram.slx`。
- PID vs LADRC 对比结果：`payload_step` 中 PID 最大姿态误差与 LADRC 最大姿态误差由脚本自动生成。
- `arm_reach_payload` 结果：机械臂伸展、夹爪闭合、负载附着后，LADRC 仍能保持较小姿态误差。
- 单图版结果：高度跟踪、姿态响应、推力命令、机械臂关节运动、机械臂耦合力矩、负载耦合力矩和姿态范数分别单独展示。
- Simulink 闭环结果图，作为 MATLAB 脚本结果的模型级验证。

不建议第一版论文夸大：

- 不要写“MATLAB 已完整复现 Isaac/PhysX 接触摩擦”。
- 不要写“已经完成真机控制验证”。
- 不要把扫描教材里的样机参数当成 flyarm 参数。
- 不要把 MATLAB 的等效负载扰动说成真实夹爪接触动力学。

## 5. 当前边界

MATLAB/Simulink 已覆盖名义动力学和底层控制验证；Isaac Lab 仍负责高保真接触、摩擦、抓取成功率、视觉/触觉和双机协作策略验证。这个分工在论文里最稳。
