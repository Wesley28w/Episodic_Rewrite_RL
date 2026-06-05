"""
Stack environment
"""

import gymnasium as gym
from . import agents
from .stack_env import StackEnv, StackEnvCfg
from .stack_shake_env import StackShakeEnv, StackShakeEnvCfg
from .stack_camera_shake_env  import StackCameraShakeEnv, StackCameraShakeEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Stack-Direct-v0",
    entry_point=f"{__name__}.stack_env:StackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_env:StackEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StackPPORunnerCfg"
    },
)

# shares agent files with Isaac Stack Direct v0
gym.register(
    id="Isaac-Stack-Shake-Direct-v0",
    entry_point=f"{__name__}.stack_shake_env:StackShakeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_shake_env:StackShakeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StackPPORunnerCfg"
    },
)

gym.register(
    id="Isaac-Stack-Camera-Shake-Direct-v0",
    entry_point=f"{__name__}.stack_camera_shake_env:StackCameraShakeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_camera_shake_env:StackCameraShakeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_camera_shake_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StackCameraShakePPORunnerCfg"
    },
)