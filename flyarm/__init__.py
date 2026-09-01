# ================================================================
# Flyarm 空中机械臂 —— Gymnasium 任务注册
#
# 注册用于 RL 训练的空中机械臂环境。
# 机器人是一架四旋翼, 搭载 4 自由度机械臂与夹爪; 飞行内环
# 为串级 LADRC 控制器, 由 PPO/MAPPO 学习高层的抓取与搬运行为。
#
# 任务:
#   - 单机 LADRC:
#       Isaac-Flyarm-Ladrc-Direct-v0    (旋转夹爪)
#       Isaac-Flyarm-PgLadrc-Direct-v0  (平行夹爪)
#   - 双机集中式 (单策略同时控制两机):
#       Isaac-Flyarm-CoopGrasp-Direct-v0    (旋转夹爪)
#       Isaac-Flyarm-PgCoopGrasp-Direct-v0  (平行夹爪)
#   - 双机分布式 MARL (CTDE / MAPPO):
#       Isaac-Flyarm-CoopGrasp-MARL-v0    (旋转夹爪)
#       Isaac-Flyarm-PgCoopGrasp-MARL-v0  (平行夹爪)
# ================================================================

import gymnasium as gym

from . import agents

# ── 单机 LADRC 验证 (旋转夹爪) ──
gym.register(
    id="Isaac-Flyarm-Ladrc-Direct-v0",
    entry_point=f"{__name__}.flyarm_ladrc_env:FlyarmLadrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_ladrc_env:FlyarmLadrcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlyarmLadrcPPORunnerCfg",    },
)

# ── 单机 LADRC 验证 (平行夹爪) ──
# 机械臂/飞行/reward 栈与旋转版完全相同; 仅夹爪子树
# (prismatic 平行指) 及其 USD 不同。
gym.register(
    id="Isaac-Flyarm-PgLadrc-Direct-v0",
    entry_point=f"{__name__}.flyarm_pg_ladrc_env:FlyarmPgLadrcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_pg_ladrc_env:FlyarmPgLadrcEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlyarmPgLadrcPPORunnerCfg",    },
)

# ── 双机协同抓取 + 搬运, 集中式 (旋转夹爪) ──
# 单策略同时观测两机并输出两套 action;
# 两机各抓共享杆的一端, 把它抬到空中目标。
# 飞行 = 串级 LADRC (num_agents=2) + 负载前馈。
gym.register(
    id="Isaac-Flyarm-CoopGrasp-Direct-v0",
    entry_point=f"{__name__}.flyarm_coop_grasp_env:FlyarmCoopGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_coop_grasp_env:FlyarmCoopGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlyarmCoopGraspPPORunnerCfg",    },
)

# ── 双机协同抓取 + 搬运, 集中式 (平行夹爪) ──
gym.register(
    id="Isaac-Flyarm-PgCoopGrasp-Direct-v0",
    entry_point=f"{__name__}.flyarm_pg_coop_grasp_env:FlyarmPgCoopGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_pg_coop_grasp_env:FlyarmPgCoopGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlyarmPgCoopGraspPPORunnerCfg",    },
)

# ── 双机协同抓取, 分布式 MARL / CTDE (旋转夹爪) ──
# 每架无人机是独立 agent (各自的 reward), 配中心化 critic;
# 用 skrl MAPPO 训练。任务与参数和集中式版本一致。
gym.register(
    id="Isaac-Flyarm-CoopGrasp-MARL-v0",
    entry_point=f"{__name__}.flyarm_coop_grasp_marl_env:FlyarmCoopGraspMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_coop_grasp_marl_env:FlyarmCoopGraspMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# ── 双机协同抓取, 分布式 MARL / CTDE (平行夹爪) ──
# 使用专门的 yaml, 使其日志与旋转夹爪 MARL 的训练分开。
gym.register(
    id="Isaac-Flyarm-PgCoopGrasp-MARL-v0",
    entry_point=f"{__name__}.flyarm_pg_coop_grasp_marl_env:FlyarmPgCoopGraspMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flyarm_pg_coop_grasp_marl_env:FlyarmPgCoopGraspMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_pg_cfg.yaml",
    },
)
