# Isaac 逻辑到 MATLAB 耦合动力学的对应关系

## 1. 结论

这次新增的模型不是把 Isaac/PhysX 重写一遍，而是在 MATLAB 中补上论文需要的两层名义动力学，同时保留 Isaac 中最关键的控制和接触逻辑：

```text
四旋翼机体: 外力/外力矩作用在 base_link
机械臂: M_a(q), C_a(q,qdot), G_a(q) + 关节位置目标 + PD drive
夹爪: 开合目标 + PD drive
系统耦合: M(chi), C(chi,nu), G(chi) + J_c^T F_c + d
负载: 抓住后在夹爪位置产生重力和偏心力矩
耦合: 机械臂运动、质心偏移、夹爪负载共同形成机体扰动
```

因此它适合用于论文中的动力学建模和控制验证；接触细节、穿模、摩擦、两指是否真正夹住物体，仍以 Isaac Lab 为主。

## 2. Isaac 中实际怎么控制

Isaac 任务里的动作语义是：

```text
action = [flight, arm, gripper]
```

当前单机任务中，飞行部分有两类路线：

```text
旧路线: action -> thrust + body moment
LADRC/设定值路线: action -> desired acceleration + yaw rate -> thrust + body moment
```

机械臂和夹爪不是直接施加“理想位移”，而是：

```text
q_cmd -> joint PD drive -> joint torque/force -> PhysX articulated body
```

对应到公式就是：

```text
tau_q = Kp * (q_cmd - q) + Kd * (qdot_cmd - qdot)
tau_q = saturate(tau_q, effort_limit)
```

本文 MATLAB 模型中的对应文件：

```text
matlab/flyarm_articulation_config.m
matlab/flyarm_articulated_pd_control.m
```

## 3. 机械臂和夹爪状态

新增耦合模型使用 24 维状态：

```text
x_c = [p_w; v_w; roll pitch yaw; omega_b; q; qdot]
q   = [base_rotate; axis_B; axis_C; axis_D; gripper_right; gripper_left]
```

其中前 12 维仍是四旋翼刚体状态，后 12 维是机械臂和夹爪的关节位置/速度。

## 4. 耦合动力学怎么建

四旋翼主体仍采用：

```text
m * v_dot = R * ([0;0;T] + F_couple) - m*g*e3
J * omega_dot = M + tau_couple - omega x (J*omega)
```

在论文公式层，完整系统写成：

```text
chi = [p_w; roll pitch yaw; q]
nu  = [v_b; omega_b; qdot]

M(chi) * nudot + C(chi,nu) * nu + G(chi)
    = tau_rotor + tau_joint + J_c^T * F_c + d
```

其中机械臂本体先满足：

```text
M_a(q) * qddot + C_a(q,qdot) * qdot + G_a(q) = tau_a
```

对应新增文件：

```text
matlab/flyarm_link_kinematic_terms.m
matlab/flyarm_manipulator_dynamics.m
matlab/flyarm_aerial_manipulator_dynamics.m
matlab/flyarm_arm_computed_torque_control.m
```

在闭环仿真层，耦合项继续拆成三部分，方便和 Isaac 的控制逻辑对齐。

第一，关节 PD drive 的反作用：

```text
revolute joint:  tau_base += -tau_i * axis_i
prismatic joint: F_base   += -F_i * axis_i
                 tau_base += r_i x F_base
```

第二，机械臂质心移动带来的等效惯性反力：

```text
a_arm_com ~= J_com(q) * qddot
F_arm = -m_arm * a_arm_com
tau_arm = r_arm_com x F_arm
```

第三，机械臂展开或抓取后，系统质心相对初始姿态偏移，重力会产生偏心力矩：

```text
tau_com = (r_com(q) - r_com(q_trim)) x R^T * [0;0;-m*g]
```

如果夹爪抓住负载，则负载重力在夹爪中心产生：

```text
F_payload = R^T * [0;0;-m_payload*g]
tau_payload = r_grip x F_payload
```

对应文件：

```text
matlab/flyarm_articulated_kinematics.m
matlab/flyarm_coupled_dynamics.m
matlab/flyarm_coupled_tracking.m
```

## 5. 为什么仍不把 Isaac/PhysX 全部重写

当前 MATLAB 已经有论文需要的名义多体动力学：

```text
M_a(q) * qddot + C_a(q,qdot) * qdot + G_a(q) = tau_a
M(chi) * nudot + C(chi,nu) * nu + G(chi) = tau_rotor + tau_joint + J_c^T F_c + d
```

但仍不建议在 MATLAB 里重写完整 PhysX 接触求解器，原因是：

- Isaac/PhysX 已经负责高保真多刚体、接触和摩擦；
- 论文控制验证更关心“机械臂扰动出现后，飞控能否压住姿态”；
- LADRC 的方法论本来就是把未精确建模的接触、风扰和参数误差看成总扰动 `d`，由 ESO 估计补偿；
- 两指接触、摩擦锥、滑移和掉落需要接触参数辨识，否则手写模型容易形式复杂但不可验证。

因此当前版本采用“MATLAB 名义动力学 + MATLAB 控制验证 + Isaac 高保真接触/策略验证”的分工。

## 6. 和论文资料章节的关系

这部分可以对应到你给的资料：

| 资料章节 | 用在本文哪里 |
|---|---|
| 第二章 飞行器原理 | 旋翼推力、反扭矩、混控 |
| 第三章 刚体运动方程 | 四旋翼 6DOF 平动/转动方程 |
| 第五章 小扰动线性化 | 悬停附近控制器设计和稳定性说明 |
| 第六章 姿态角控制 | PID/LADRC 姿态环结构 |
| 第七章 垂直速度和高度控制 | 高度外环与推力控制 |
| 第十章 悬停控制 | 悬停、抓取后负载扰动、抗扰验证 |
| LADRC 资料 | 串级 ESO、带宽、b0 和 z2 限幅设计 |
| flyarm URDF | 质量、连杆、关节、夹爪类型、旋翼位置 |
| Isaac 任务代码 | 关节 PD drive、动作语义、接触抓取、抓后搬运逻辑 |

## 7. 当前边界

当前 MATLAB 耦合模型已经包含：

- 四旋翼电机一阶响应；
- 旋翼混控；
- 机体 6DOF 动力学；
- 机械臂本体 `M_a(q), C_a(q,qdot), G_a(q)`；
- 飞行器-机械臂耦合 `M(chi), C(chi,nu), G(chi)`；
- 机体-机械臂交叉惯量块 `M_ba/M_ab`；
- 接触广义力接口 `J_c^T F_c`；
- 机械臂 4 关节 PD；
- 夹爪 2 关节 PD；
- 机械臂质心移动；
- 关节反作用力矩；
- 抓取负载偏心扰动；
- PID/LADRC 飞控闭环。

当前还没有包含：

- 两指接触摩擦的真实约束求解；
- 物体在夹爪中滑动/掉落的接触动力学；
- 符号级闭式推导，当前实现是 URDF 数值动力学；
- Simscape Multibody 导入 URDF 后的可视化验证。

这些内容后续可以作为论文增强项，而不是第一版控制验证的阻塞项。
