# ================================================================
# Flyarm 视觉 + 触觉环境
#
# 在基础空中机械臂任务之上扩展 FlyarmEnv, 增加感知与抓取塑形. 共三处新增,
# 功能上均可选:
#   - 触觉: 每个夹爪手指上挂 ContactSensor, 把 finger-object 接触力
#     送入 obs 和抓取 reward. 纯物理量(无渲染开销), MLP 策略可直接用;
#     默认开启.
#   - 抓取塑形 reward: 纯几何项(progress / engagement / contact /
#     grasp_close / grasp_lift / lift_goal), 把策略从"够到但不抓"
#     引导到"合爪夹住并抬起".
#   - Camera/IBVS 脚手架: 可选的腕部 + 机身 TiledCamera, 配绿色物体的
#     像素误差信号做 image-based visual servoing.
#     默认关闭(enable_cameras=False), 因为原始像素不能直接喂给 MLP ——
#     开启时应改用 residual visual servoing 或 CNN 特征提取器.
#
# 另外在 PPO 力矩上叠加一项 inner-loop attitude-rate damping
# (仅在旧的 direct-torque 控制下生效; setpoint/LADRC 路径自带阻尼).
# ================================================================

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from .flyarm_env import FlyarmEnv, FlyarmEnvCfg


# ──────────────────────────────────────────────────────────────
# 配置: 继承基础 env, 只添加新增的部分
# ──────────────────────────────────────────────────────────────
@configclass
class FlyarmVisionEnvCfg(FlyarmEnvCfg):

    # 27 基础维 + 2 触觉(左/右指接触力) = 29
    observation_space = 29

    # === 触觉(contact sensor) ===
    # 接触阈值: 物块轻(15g)+温和夹紧时接触力可能 <0.5N. 用 0.2N 让"轻夹"也计为接触,
    # 从而解锁 grasp_lift / lift_goal 抬升梯度.
    contact_force_threshold = 0.2

    # === 相机开关(默认关; 见文件头注释) ===
    enable_cameras = False
    camera_width = 84
    camera_height = 84

    # === 抓取塑形 reward 权重 ===
    contact_grasp_reward_scale = 3.0   # 两指都接触物体(真咬住)
    progress_reward_scale = 8.0        # 末端→物体距离持续减小; 权重偏低避免"来回晃"刷分
    engagement_reward_scale = 2.0      # 末端贴到物体很近(<3cm)
    # grasp_lift: 物体被两指真夹住且离地越高给越多分, 连续(0~lift_height), 由 both_contact 门控
    # 防止顶/撞骗分. 权重高于接近类奖励总和, 让"夹住抬起"压过"贴着不抓"的局部最优.
    grasp_lift_reward_scale = 30.0
    # grasp_close: 不依赖接触的桥梁奖励 — 指尖会合点贴物块且夹爪命令闭合即给分,
    # 破"张开撑台不合爪"的局部最优, 引导先合爪→产生接触→解锁后续抬升奖励.
    grasp_close_reward_scale = 6.0
    # lift_goal: 对标官方 lift 的 object_goal_tracking — both_contact 后把物体往空中目标高度拉,
    # 提供"一路搬上去"的持续牵引(连续, 主导抬升).
    lift_goal_height = 0.30            # 对齐父类 carry_height: 抓住后带物块直上飞 0.30m
    lift_goal_std = 0.10              # tanh 软度尺度(m)
    lift_goal_reward_scale = 16.0     # 对标官方 lift object_goal_tracking 权重

    # === 内环稳定(PID inner loop) + 速度阻尼 ===
    attitude_rate_damping = 0.20   # 角速度阻尼(D项), 压住伸臂后的姿态振荡
    attitude_level_gain = 1.0      # 水平P项: 用重力投影把机身拉回水平; 若符号与机型相反则置0
    lin_vel_penalty_scale = -0.05  # 线速度阻尼(别乱飘)
    ang_vel_penalty_scale = -0.02  # 角速度阻尼(别乱转)

    # === Visual servoing(仅 enable_cameras=True 时): IBVS 居中 ===
    wrist_center_reward_scale = 1.0    # 物体在画面越居中给分越高
    # 用哪个相机做像素误差信号: "body" 机身相机俯视台子可靠看到物块(默认),
    # "wrist" 腕相机眼在手上易被夹爪手指遮挡.
    ibvs_camera = "body"

    # === 手指接触传感器 ===
    # prim_path 中的 body 名 gripper_left / gripper_right; 若 USD 层级更深需改路径或用通配.
    # filter 只统计与被抓物体 Object 的接触力.
    contact_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_left",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )
    contact_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_right",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
    )

    # === 腕部相机(挂在 gripper_base 上; 仅 enable_cameras=True 时创建) ===
    # rot 为相机相对父连杆朝向(四元数 wxyz). 相机在 gripper_base 侧边, 光轴从侧边斜对抓取点
    # (grasp_offset), 而非垂直俯视(会被夹爪手指挡). 四元数 (0,0.973,0,0.230) 使 +Z 光轴
    # 对准视线方向 (0.447,0,-0.894), 即朝内(夹爪方向)且朝下.
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_base/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.05, -0.02, 0.0), rot=(0.0, 0.973, 0.0, 0.230), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 2.0)
        ),
        width=84, height=84,
    )
    # === 机身相机(挂在 base_link 上, 俯视场景/物体) ===
    # rot (0,1,0,0) 朝下避免拍到自身旋翼; pos 下移到旋翼下方避免遮挡. 仅作脚手架, 不参与奖励/obs.
    body_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/body_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.10, 0.0, -0.12), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
        ),
        width=84, height=84,
    )

    def __post_init__(self):
        # 接触传感器要求对应刚体开启接触上报
        self.robot.spawn.activate_contact_sensors = True
        # 开相机时观测追加 2 维"物体在相机里的像素误差(cx,cy)" → 29+2=31
        if self.enable_cameras:
            self.observation_space = 31


# ──────────────────────────────────────────────────────────────
# Env: 继承基础环境, 为新增部分覆写 scene/obs/reward/reset
# ──────────────────────────────────────────────────────────────
class FlyarmVisionEnv(FlyarmEnv):

    cfg: FlyarmVisionEnvCfg

    def __init__(self, cfg: FlyarmVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)   # 触发 _setup_scene()

        # 为新增奖励项补建累计记录键(父类已建好 self._episode_sums)
        sum_keys = ["contact_grasp", "progress", "engagement", "lin_vel", "ang_vel", "grasp_lift", "lift_goal", "grasp_close"]
        if self.cfg.enable_cameras:
            sum_keys.append("wrist_center")
        for key in sum_keys:
            self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # progress 奖励需要记住上一帧的 末端→物体 距离
        self._prev_ee_obj_dist = torch.zeros(self.num_envs, device=self.device)

        print(f"[vision] tactile=ON, camera={'ON' if self.cfg.enable_cameras else 'OFF'}, obs=29")

    # ==============================================================
    # 内环稳定: 在 PPO 力矩上叠加 attitude-rate damping(+可选 leveling),
    # 防止平台翻转.
    # PPO 仍输出期望力矩; 我们加上 D(damping)项, 以及可选的 P(leveling)项 ——
    # 后者默认关闭, 因为其符号需按机型逐一验证.
    # ==============================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)          # 父类: 算好 self._thrust / self._moment / 臂目标
        # 仅在旧的直接出力矩控制下补阻尼; setpoint(INDI) 和 LADRC 自带阻尼(LESO).
        if not self.cfg.use_setpoint_control and not self.cfg.use_ladrc_control:
            omega = self._robot.data.root_ang_vel_b     # 机体角速度
            self._moment[:, 0, :] -= self.cfg.attitude_rate_damping * omega   # D项: 抗旋转
            kp = self.cfg.attitude_level_gain
            if kp > 0.0:                                  # P项: 用重力投影把机身拉回水平
                grav_b = self._robot.data.projected_gravity_b
                self._moment[:, 0, 0] -= kp * grav_b[:, 1]   # 抗 roll
                self._moment[:, 0, 1] += kp * grav_b[:, 0]   # 抗 pitch

    # ==============================================================
    # 场景搭建: 基础流程 + 在克隆前加入传感器
    # ==============================================================
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self._object
        # 宽台(kinematic): 把物体垫高让脚架离地
        self._pedestal = RigidObject(self.cfg.pedestal)
        self.scene.rigid_objects["pedestal"] = self._pedestal

        # 触觉: 两指接触传感器(物理量, 不渲染, 默认加)
        self._contact_left = ContactSensor(self.cfg.contact_left)
        self._contact_right = ContactSensor(self.cfg.contact_right)
        self.scene.sensors["contact_left"] = self._contact_left
        self.scene.sensors["contact_right"] = self._contact_right

        # 相机: 仅在开启时创建(渲染开销大)
        if self.cfg.enable_cameras:
            self._wrist_cam = TiledCamera(self.cfg.wrist_camera)
            self._body_cam = TiledCamera(self.cfg.body_camera)
            self.scene.sensors["wrist_cam"] = self._wrist_cam
            self.scene.sensors["body_cam"] = self._body_cam

        # 地形 + 克隆 + 灯光
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # 读某个指与物体的接触力大小 (N,)
    def _finger_contact_mag(self, sensor: ContactSensor) -> torch.Tensor:
        # force_matrix_w: (num_envs, num_bodies=1, num_filtered=1, 3) → 取范数 → (num_envs,)
        fmat = sensor.data.force_matrix_w
        if fmat is None:                                   # 兜底: 无过滤数据则用净接触力
            fmat = sensor.data.net_forces_w
        return torch.linalg.norm(fmat.reshape(self.num_envs, -1, 3), dim=-1).sum(dim=-1)

    # ==============================================================
    # 观测: 27 基础维 + 2 触觉 = 29 (相机开启时再 +2 像素误差)
    # ==============================================================
    def _get_observations(self) -> dict:
        obs = super()._get_observations()                 # {"policy": (N,27)}, 已 clamp 到[-5,5]
        fl = self._finger_contact_mag(self._contact_left)
        fr = self._finger_contact_mag(self._contact_right)
        # tanh 压到 [0,1) 区间(力越大越接近1), 数值友好
        tactile = torch.stack([torch.tanh(0.1 * fl), torch.tanh(0.1 * fr)], dim=-1)   # (N,2)
        obs["policy"] = torch.cat([obs["policy"], tactile], dim=-1)                    # (N,29)
        # 视觉伺服: 把"物体在相机里的像素误差(cx,cy)"喂进观测, 策略才能学会视觉对准.
        # 仅相机开启时执行(obs→31).
        if self.cfg.enable_cameras:
            pix_err = self._wrist_object_pixel_error()                                 # (N,2) ≈[-1,1]
            obs["policy"] = torch.cat([obs["policy"], pix_err], dim=-1)                # (N,31)
        return obs

    # ==============================================================
    # 奖励: 基础 10 项 + 抓取塑形(contact / progress / engagement /
    # grasp_lift / lift_goal / grasp_close) + 速度阻尼
    # ==============================================================
    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()                   # 原版 10 项已求和并记账

        # 夹持点 = 指尖会合点(父类 _compute_grasp_center)
        grasp_center = self._compute_grasp_center()
        obj_pos = self._object.data.root_pos_w
        # 瞄物块上半部(物块中心+grasp_z_offset): 不夹底/不卡台; 抬升判定仍用物块真实高度(见下).
        ee_d = torch.linalg.norm(grasp_center - self._grasp_target_w(), dim=1)   # 指尖→上半部瞄准点距离

        # 臂伸直门控: 闸门 臂够直≈1、卷曲≈0, 乘到所有抓/抬奖励上, 逼"先伸直再抓".
        # 软斜坡 0.4→0.7. (_arm_straight_val 由父类 _get_rewards 暂存)
        straight_gate = ((self._arm_straight_val - 0.4) / 0.3).clamp(0.0, 1.0)
        # 夹爪竖直门控: 软斜坡 0.5→0.8, 逼"先竖直再下抓". (_gripper_vertical_val 由父类暂存)
        vertical_gate = ((self._gripper_vertical_val - 0.5) / 0.3).clamp(0.0, 1.0)
        # 必须 臂伸直 AND 夹爪竖直 才解锁抓取回报
        grasp_gate = straight_gate * vertical_gate

        # ① contact: 两指都和物体有接触力(>阈值) → 真咬住
        fl = self._finger_contact_mag(self._contact_left)
        fr = self._finger_contact_mag(self._contact_right)
        both_contact = ((fl > self.cfg.contact_force_threshold) &
                        (fr > self.cfg.contact_force_threshold)).float()
        contact_r = both_contact * self.cfg.contact_grasp_reward_scale * self.step_dt * grasp_gate

        # ② progress: 距离持续减小才给分(每步幅度 clamp 防尖峰); 首帧 prev=0 不产生假尖峰
        progress = (self._prev_ee_obj_dist - ee_d).clamp(min=0.0, max=0.02)
        progress_r = progress * self.cfg.progress_reward_scale * self._grasp_phase.float()   # 门控在第三阶段
        self._prev_ee_obj_dist = ee_d.detach()

        # ③ engagement: 末端贴到物体很近(<3cm) → 阶跃奖励, 逼它从悬停上方下探
        engagement = (ee_d < 0.03).float()
        engagement_r = engagement * self.cfg.engagement_reward_scale * self.step_dt * self._grasp_phase.float()

        # ④ grasp_lift: 物体被两指真夹住(both_contact) 且离地越高 → 连续给越多分, 治"抬不起"的主力梯度.
        object_rise = (self._object.data.root_pos_w[:, 2] - self._object_initial_height).clamp(
            min=0.0, max=self.cfg.lift_height
        )
        grasp_lift_r = object_rise * both_contact * self.cfg.grasp_lift_reward_scale * self.step_dt * grasp_gate

        # ⑤ lift_goal: 真夹住后把物体往空中目标高度拉(主导抬升); 越接近目标 1-tanh 越接近1.
        lift_target_z = self._object_initial_height + self.cfg.lift_goal_height
        height_gap = (lift_target_z - self._object.data.root_pos_w[:, 2]).clamp(min=0.0)
        lift_goal_r = both_contact * (1.0 - torch.tanh(height_gap / self.cfg.lift_goal_std)) \
            * self.cfg.lift_goal_reward_scale * self.step_dt * grasp_gate

        # ⑥ grasp_close: 桥梁奖励 — 指尖会合点贴物块(<3.5cm) 且夹爪命令闭合(a8≥0) 即给分(不依赖接触),
        # 破"张开撑台"局部最优, 引导合爪→产生接触→后续 contact/grasp_lift/lift_goal 接管真信号.
        gripper_closed = (self._actions[:, 8] >= 0.0).float()
        at_block = (ee_d < 0.035).float()
        grasp_close_r = at_block * gripper_closed * self.cfg.grasp_close_reward_scale * self.step_dt * grasp_gate

        # ⑦⑧ 速度阻尼: 别乱飘/别乱转, 与内环角速度阻尼互补.
        # clamp(max=100) 防速度发散→v²→奖励炸 inf→value_loss/std 崩溃(配合 _get_dones 的发散终止).
        lin_v = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1).clamp(max=100.0)
        ang_v = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1).clamp(max=100.0)
        lin_vel_r = lin_v * self.cfg.lin_vel_penalty_scale * self.step_dt
        ang_vel_r = ang_v * self.cfg.ang_vel_penalty_scale * self.step_dt

        reward = reward + contact_r + progress_r + engagement_r + lin_vel_r + ang_vel_r + grasp_lift_r + lift_goal_r + grasp_close_r
        self._episode_sums["contact_grasp"] += contact_r
        self._episode_sums["progress"] += progress_r
        self._episode_sums["engagement"] += engagement_r
        self._episode_sums["lin_vel"] += lin_vel_r
        self._episode_sums["ang_vel"] += ang_vel_r
        self._episode_sums["grasp_lift"] += grasp_lift_r
        self._episode_sums["lift_goal"] += lift_goal_r
        self._episode_sums["grasp_close"] += grasp_close_r

        # 视觉伺服(仅相机开启): 物体在腕相机里越居中越加分 (IBVS)
        if self.cfg.enable_cameras:
            pix_err = self._wrist_object_pixel_error()
            centering = torch.exp(-torch.sum(pix_err ** 2, dim=1) / 0.1)
            center_r = centering * self.cfg.wrist_center_reward_scale * self.step_dt
            reward = reward + center_r
            self._episode_sums["wrist_center"] += center_r
        return reward

    # ==============================================================
    # Reset: 基础逻辑 + 重置 progress 的上一帧距离
    # ==============================================================
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)                       # 顺带记账+清零新增奖励键
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._prev_ee_obj_dist[env_ids] = 0.0             # 新一局: 上一帧距离清0(配合 clamp(min=0) 不尖峰)

    # ==============================================================
    # IBVS 像素误差: 从相机 RGB 中分割出绿色物体, 返回其相对画面中心的归一化
    # 偏移 (N,2) ≈[-1,1]; ~0 表示物体已居中. 仅 enable_cameras=True 时有效.
    # 绿色阈值按物体颜色 (0,0.8,0.2) 估算. 这就是策略在下探前用来保持物体居中的
    # residual-visual-servoing 误差.
    # ==============================================================
    def _wrist_object_pixel_error(self) -> torch.Tensor:
        # 选相机: body(机身相机, 俯视台子清楚看到绿物块, 默认) 或 wrist(腕相机, 易被夹爪挡).
        cam = self._body_cam if getattr(self.cfg, "ibvs_camera", "body") == "body" else self._wrist_cam
        rgb = cam.data.output["rgb"][..., :3].float()   # (N,H,W,3) uint8→float
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = ((g > 80) & (g > 1.3 * r) & (g > 1.3 * b)).float()   # 绿色物体掩膜
        H, W = mask.shape[1], mask.shape[2]
        xs = torch.linspace(-1.0, 1.0, W, device=self.device).view(1, 1, W)
        ys = torch.linspace(-1.0, 1.0, H, device=self.device).view(1, H, 1)
        tot = mask.sum(dim=(1, 2)) + 1e-6
        cx = (mask * xs).sum(dim=(1, 2)) / tot
        cy = (mask * ys).sum(dim=(1, 2)) / tot
        return torch.stack([cx, cy], dim=-1)                        # (N,2)

    # ==============================================================
    # (可选) 抓取相机图像 — 用于核验相机视角或喂给未来的视觉编码器.
    # 仅 enable_cameras=True 时有效. 返回 (wrist, body) RGB, 形状 (N,H,W,3).
    # ==============================================================
    def get_camera_rgb(self):
        if not self.cfg.enable_cameras:
            raise RuntimeError("相机没开: 把 FlyarmVisionEnvCfg.enable_cameras 设为 True, 启动加 --enable_cameras")
        return self._wrist_cam.data.output["rgb"], self._body_cam.data.output["rgb"]
