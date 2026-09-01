"""flyarm 空中机械臂任务的 RSL-RL PPO 训练配置。

定义 on-policy PPO runner 的配置层级, 在单机抓取任务及其派生变体
(LADRC 飞控、平行夹爪硬件、以及双机协作任务) 之间共享。所有变体共用同一套
actor-critic 网络和 PPO 超参数; 子类主要覆盖实验日志目录, 以及在 action
空间变化时覆盖 entropy 系数。

- 基础单机任务 obs=27、action=9; 网络 [256, 256, 128]。
- 双机任务扩大 obs/action 并降低 entropy 系数。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FlyarmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """单机抓取任务的基础 PPO 配置 (obs=27、action=9)。"""

    seed = 42
    num_steps_per_env = 24                 # 每个 env 每次更新的 rollout 长度
    max_iterations = 10000

    # 观测归一化 (running mean/std), 加速收敛
    actor_obs_normalization = True
    critic_obs_normalization = True

    save_interval = 200
    experiment_name = "flyarm_grasp"
    run_name = ""
    logger = "tensorboard"
    resume = False

    # Actor-critic MLP: obs27 → 256 → 256 → 128 → action9
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,                # 较大初始噪声促进探索, 训练中自动衰减
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,                # 抑制动作 std 失控 (机械臂乱挥导致抓不稳)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",               # 按 KL 自适应学习率
        gamma=0.99,                        # 长视野, 适配多阶段任务
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ─── 视觉 + 触觉变体: 网络/超参数相同, 日志目录分开 ───
@configclass
class FlyarmVisionPPORunnerCfg(FlyarmPPORunnerCfg):
    experiment_name = "flyarm_vision_grasp"


# ─── 单机 LADRC 飞控验证 (obs29/action9) ───
@configclass
class FlyarmLadrcPPORunnerCfg(FlyarmVisionPPORunnerCfg):
    experiment_name = "flyarm_ladrc"
    # 关闭 obs 归一化, 让策略直接在 raw obs 上训练, 保证与下游复用时输入分布一致。
    # obs 各量级均为 O(1) (位置 ~0.2m / 速度 ~m/s / 单位向量 ±1), 不归一化仍可收敛。
    actor_obs_normalization = False
    critic_obs_normalization = False


# ─── 双机第一阶段 (obs58/action18, 集中式单策略) ───
@configclass
class FlyarmDualPPORunnerCfg(FlyarmVisionPPORunnerCfg):
    experiment_name = "flyarm_dual"

    # 双机 18 维动作下熵奖励随维度翻倍, 同样 coef 会把 std 推得更高 (探索甩动作→机身被掀翻)。
    # 减半压低 std; 其余参数与单机一致。
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ─── 双机第二阶段 2b: 协作抓取 + 搬运 (obs75/action18) ───
#   继承 Dual 以沿用 entropy=0.001 的双机减熵设置, 仅日志目录分开。
@configclass
class FlyarmCoopGraspPPORunnerCfg(FlyarmDualPPORunnerCfg):
    experiment_name = "flyarm_coop_grasp"


# ─── 单机 平行夹爪 + LADRC 验证 (obs29/action9) ───
#   继承 Vision (obs 归一化开启), 从零验证新夹爪, 不导入 teacher。
@configclass
class FlyarmPgLadrcPPORunnerCfg(FlyarmVisionPPORunnerCfg):
    experiment_name = "flyarm_pg_ladrc"


# ─── 双机 平行夹爪 2b 集中式 (rsl_rl) ───
#   继承 CoopGrasp 以沿用双机减熵设置, 仅日志目录分开。
@configclass
class FlyarmPgCoopGraspPPORunnerCfg(FlyarmCoopGraspPPORunnerCfg):
    experiment_name = "flyarm_pg_coop_grasp"
