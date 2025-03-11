import gymnasium as gym
import fw_jsbgym


def make_env(cfg):
    """
        Make an env for TD-MPC2 experiments in the JSBSim simulator.
    """
    try:
        env = gym.make(f'{cfg.rl.task}-v0', cfg_env=cfg.env, telemetry_file='telemetry/telemetry.csv',
                   render_mode=cfg.env.jsbsim.render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
    except:
        raise ValueError(f'Unknown task: {cfg.rl.task}')
    # env = gym.make('SimpleAC_OMAC-v0', config_file = '../../../jsbsim_cfg.yaml', telemetry_file = 'telemetry.csv',
    #                render_mode = 'none')

    print("observation space: ", env.observation_space.shape)
    print("action space: ", env.action_space.shape)
    env.init()
    return env