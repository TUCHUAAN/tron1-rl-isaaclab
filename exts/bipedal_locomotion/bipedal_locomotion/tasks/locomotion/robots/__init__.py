import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    PF_TRON1AFlatPPORunnerCfg,
    SF_TRON1AFlatPPORunnerCfg,
    WF_TRON1AFootAllTerrainPPORunnerCfg,
    WF_TRON1AFlatPPORunnerCfg,
    WF_TRON1AWheelModePPORunnerCfg,
)

from . import limx_pointfoot_env_cfg, limx_solefoot_env_cfg, limx_wheelfoot_env_cfg, limx_wheelfoot_mode_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_pf_blind_flat_runner_cfg = PF_TRON1AFlatPPORunnerCfg()

limx_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg()
limx_wf_wheel_mode_runner_cfg = WF_TRON1AWheelModePPORunnerCfg()
limx_wf_foot_all_terrain_runner_cfg = WF_TRON1AFootAllTerrainPPORunnerCfg()

limx_sf_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()



##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# WF Blind Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)


############################
# SF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

####################################
# WF dual-expert locomotion tasks
####################################

gym.register(
    id="Isaac-Limx-WF-Wheel-Mode-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_mode_env_cfg.WFWheelModeEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_wheel_mode_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Wheel-Mode-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_mode_env_cfg.WFWheelModeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_wheel_mode_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Foot-AllTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_mode_env_cfg.WFFootAllTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_foot_all_terrain_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Foot-AllTerrain-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_mode_env_cfg.WFFootAllTerrainEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_foot_all_terrain_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Dual-Mode-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_mode_env_cfg.WFDualModeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_wheel_mode_runner_cfg,
    },
)
