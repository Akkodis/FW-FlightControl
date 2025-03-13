import os
import numpy as np
import torch
import gymnasium as gym
import fw_jsbgym
import pandas as pd
import wandb
import plotly.graph_objects as go
from math import pi
from fw_flightcontrol.agents import sac_norm, sac, ppo_norm, ppo
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from fw_jsbgym.utils import conversions
from omegaconf import DictConfig, OmegaConf

# Global variables
# Sequence of roll and pitch references for the the periodic evaluation
attitude_seq: np.ndarray = np.array([
                                        [	# roll, pitch
                                            [np.deg2rad(25), np.deg2rad(15)], # easy
                                            [np.deg2rad(-25), np.deg2rad(-15)],
                                            [np.deg2rad(25), np.deg2rad(-15)],
                                            [np.deg2rad(-25), np.deg2rad(15)]
                                        ],
                                        [
                                            [np.deg2rad(40), np.deg2rad(22)], # medium
                                            [np.deg2rad(-40), np.deg2rad(-22)],
                                            [np.deg2rad(40), np.deg2rad(-22)],
                                            [np.deg2rad(-40), np.deg2rad(22)]
                                        ],
                                        [
                                            [np.deg2rad(55), np.deg2rad(28)], # hard
                                            [np.deg2rad(-55), np.deg2rad(-28)],
                                            [np.deg2rad(55), np.deg2rad(-28)],
                                            [np.deg2rad(-55), np.deg2rad(28)]
                                        ]
                                    ])

# attitude_seq: np.ndarray = np.array([
# 										[	# roll			,pitch
# 											[np.deg2rad(25), np.deg2rad(15)], # easy
# 										],
# 										[
# 											[np.deg2rad(40), np.deg2rad(22)], # medium
# 										],
# 										[
# 											[np.deg2rad(55), np.deg2rad(28)], # hard
# 										]
# 									])

# Waypoint Tracking sequence for the periodic evaluation
waypoint_seq: np.ndarray = np.array([       # x, y, z
                                        [1.88, -49.1, 609.256],
                                        [-4.027, 49.837, 600.229],
                                        [29.338, -39.304, 590.279],
                                        [-37.94, 31.081, 609.722],
                                        [-29.219, 39.939, 592.849]
                                    ])

# waypoint_seq: np.ndarray = np.array([
#                                         [0, 50, 600],
#                                     ])

# Altitude Tracking sequence for the periodic evaluation
altitude_seq: np.ndarray = np.array([[550], [570], [590], [600], [620], [640], [650]])


# Run periodic attitude control evaluation during training
def periodic_eval_AC(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device):
    """Periodically evaluate a given agent."""
    ep_rewards = []
    dif_obs = []
    dif_fcs_fluct = [] # dicts storing all obs across all episodes and fluctuation of the flight controls for all episodes
    for dif_idx, ref_dif in enumerate(ref_seq): # iterate over the difficulty levels
        dif_obs.append([])
        dif_fcs_fluct.append([])
        for ref_idx, ref_ep in enumerate(ref_dif): # iterate over the ref for 1 episode
            obs, info = env.reset(options=cfg_sim.eval_sim_options)
            obs, info, done, ep_reward, t = torch.Tensor(obs).unsqueeze(0).to(device), info, False, 0, 0
            while not done:
                env.set_target_state(ref_ep)
                with torch.no_grad():
                    if isinstance(agent, sac.Actor_SAC) or isinstance(agent, sac_norm.Actor_SAC):
                        action = agent.get_action(obs)[2].squeeze_(0).detach().cpu().numpy()
                    elif isinstance(agent, ppo.Agent_PPO) or isinstance(agent, ppo_norm.Agent_PPO):
                        action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
                    elif isinstance(agent, TDMPC2):
                        action = agent.act(obs.squeeze(0), t0=t==0, eval_mode=True)
                obs, reward, term, trunc, info = env.step(action)
                obs = torch.Tensor(obs).unsqueeze(0).to(device)
                done = np.logical_or(term, trunc)
                dif_obs[dif_idx].append(info['non_norm_obs']) # append the non-normalized observation to the list
                ep_reward += info['non_norm_reward']
                t += 1

            ep_fcs_pos_hist = np.array(info['fcs_pos_hist'])
            dif_fcs_fluct[dif_idx].append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # compute the fcs fluctuation of the episode being reset and append to the list

            ep_rewards.append(ep_reward)
    env.reset(options=cfg_sim.train_sim_options) # reset the env with the training options for the following of the training

    # computing the mean fcs fluctuation across all episodes for each difficulty level
    dif_fcs_fluct = np.array(dif_fcs_fluct)
    easy_fcs_fluct = np.mean(np.array(dif_fcs_fluct[0]), axis=0)
    medium_fcs_fluct = np.mean(np.array(dif_fcs_fluct[1]), axis=0)
    hard_fcs_fluct = np.mean(np.array(dif_fcs_fluct[2]), axis=0)

    # computing the rmse of the roll and pitch angles across all episodes for each difficulty level
    obs_hist_size = cfg_mdp.obs_hist_size

    if isinstance(agent, sac.Actor_SAC) or isinstance(agent, sac_norm.Actor_SAC):
    # Check if dif_obs has an inhomogeneous shape and pad the dif_obs array with np.pi (if episode truncated fill the errors with np.pi)
    # only happens with SAC
    # (copilot generated snippet careful)
        if len(set(np.shape(obs) for obs in dif_obs)) > 1:
            max_shape = max(np.shape(obs) for obs in dif_obs)
            dif_obs = [np.pad(obs, [(0, max_shape[0]-np.shape(obs)[0]), (0, max_shape[1]-np.shape(obs)[1])], constant_values=np.pi) for obs in dif_obs]

    dif_obs = np.array(dif_obs)
    if obs_hist_size == 1 and not cfg_mdp.obs_is_matrix:
        easy_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 6])))
        easy_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 7])))
        medium_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 6])))
        medium_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 7])))
        hard_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 6])))
        hard_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 7])))
    elif obs_hist_size > 1 and cfg_mdp.obs_is_matrix:
        easy_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, :, obs_hist_size-1, 6])))
        easy_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, :, obs_hist_size-1, 7])))
        medium_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, :, obs_hist_size-1, 6])))
        medium_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, :, obs_hist_size-1, 7])))
        hard_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, :, obs_hist_size-1, 6])))
        hard_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, :, obs_hist_size-1, 7])))

    return dict(
        episode_reward=np.nanmean(ep_rewards),
        easy_roll_rmse=easy_roll_rmse,
        easy_pitch_rmse=easy_pitch_rmse,
        medium_roll_rmse=medium_roll_rmse,
        medium_pitch_rmse=medium_pitch_rmse,
        hard_roll_rmse=hard_roll_rmse,
        hard_pitch_rmse=hard_pitch_rmse,
        easy_ail_fluct=easy_fcs_fluct[0],
        easy_ele_fluct=easy_fcs_fluct[1],
        medium_ail_fluct=medium_fcs_fluct[0],
        medium_ele_fluct=medium_fcs_fluct[1],
        hard_ail_fluct=hard_fcs_fluct[0],
        hard_ele_fluct=hard_fcs_fluct[1],
    )


def periodic_eval_alt(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device):
    ep_rewards = []
    non_norm_obs = []
    for ref_ep in ref_seq: # iterate over the ref for 1 episode
        obs, info = env.reset(options=cfg_sim.eval_sim_options)
        obs, info, done, ep_reward, t = torch.Tensor(obs).unsqueeze(0).to(device), info, False, 0, 0
        while not done:
            env.set_target_state(np.array(ref_ep))
            with torch.no_grad():
                if isinstance(agent, sac.Actor_SAC) or isinstance(agent, sac_norm.Actor_SAC):
                    action = agent.get_action(obs)[2].squeeze_(0).detach().cpu().numpy()
                elif isinstance(agent, ppo.Agent_PPO) or isinstance(agent, ppo_norm.Agent_PPO):
                    action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
                elif isinstance(agent, TDMPC2):
                    action = agent.act(obs.squeeze(0), t0=t==0, eval_mode=True)
            obs, reward, term, trunc, info = env.step(action)
            obs = torch.Tensor(obs).unsqueeze(0).to(device)
            done = np.logical_or(term, trunc)
            non_norm_obs.append(info['non_norm_obs']) # append the non-normalized observation to the list
            ep_reward += info['non_norm_reward']
            t += 1

        ep_rewards.append(ep_reward)
    non_norm_obs = np.array(non_norm_obs)
    # compute RMSE of the altitude errors
    alt_rmse = np.sqrt(np.mean(np.square(non_norm_obs[:, 0])))

    env.reset(options=cfg_sim.train_sim_options) # reset the env with the training options for the following of the training
    return dict(
        episode_reward=np.nanmean(ep_rewards),  # mean of the episode rewards
        alt_rmse=alt_rmse,  # RMSE of the altitude errors
    )


def periodic_eval_waypoints(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device):
    non_norm_obs = np.full((ref_seq.shape[0], env.max_episode_steps) + env.observation_space.shape, np.nan)
    ep_rewards, fcs_fluct, targets_missed, targets_reached, successes = [[] for _ in range(5)]
    for ep_idx, ref_ep in enumerate(ref_seq):
        # non_norm_obs.append([])
        obs, info = env.reset(options=cfg_sim.eval_sim_options)
        obs, info, done, ep_reward, t = torch.Tensor(obs).unsqueeze(0).to(device), info, False, 0, 0
        ref_ep = conversions.enu2ecef(*ref_ep, env.unwrapped.sim['ic/lat-geod-deg'], env.unwrapped.sim['ic/long-gc-deg'], 0.0)
        if 'WaypointVa' in env_id:
            ref_ep = np.hstack((ref_ep, np.array([60.0])))
        while not done:
            env.set_target_state(ref_ep)
            with torch.no_grad():
                if isinstance(agent, sac.Actor_SAC) or isinstance(agent, sac_norm.Actor_SAC):
                    action = agent.get_action(obs)[2].squeeze_(0).detach().cpu().numpy()
                elif isinstance(agent, ppo.Agent_PPO) or isinstance(agent, ppo_norm.Agent_PPO):
                    action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
                elif isinstance(agent, TDMPC2):
                    action = agent.act(obs.squeeze(0), t0=t==0, eval_mode=True)
            obs, reward, term, trunc, info = env.step(action)
            obs = torch.Tensor(obs).unsqueeze(0).to(device)
            done = np.logical_or(term, trunc)
            non_norm_obs[ep_idx, t] = info['non_norm_obs'] # append the non-normalized observation to the list
            ep_reward += info['non_norm_reward']
            t += 1
        ep_rewards.append(ep_reward)

        targets_missed.append(int(info['target_missed']))
        targets_reached.append(int(info['target_reached']))
        ep_fcs_pos_hist = np.array(info['fcs_pos_hist'])
        fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # compute the fcs fluctuation of the episode being reset and append to the list

    non_norm_obs = np.array(non_norm_obs)
    # Compute the distances to the target for the episode
    dists_to_target = np.sqrt(np.square(non_norm_obs[:, :, 0]) + np.square(non_norm_obs[:, :, 1]) + np.square(non_norm_obs[:, :, 2]))
    dists_to_target_mean = np.nanmean(dists_to_target)
    x_rmse = np.sqrt(np.nanmean(np.square(non_norm_obs[:, :, 0])))
    y_rmse = np.sqrt(np.nanmean(np.square(non_norm_obs[:, :, 1])))
    z_rmse = np.sqrt(np.nanmean(np.square(non_norm_obs[:, :, 2])))

    # Convert fcs fluctuations to numpy array
    fcs_fluct = np.array(fcs_fluct).mean(axis=0)

    # Compute the RMSE of the airspeed errors
    va_rmse = np.nan
    if 'WaypointVa' in env_id:
        va_rmse = np.sqrt(np.mean(np.square(non_norm_obs[:, :, 4])))

    env.reset(options=cfg_sim.train_sim_options) # reset the env with the training options for the following of the training

    return dict(
        episode_reward=np.nanmean(ep_rewards),  # mean of the episode rewards
        dists_to_target_mean=dists_to_target_mean,  # mean of the distances to the target for 1 episode
        targets_reached=np.sum(targets_reached),  # if the target was reached
        targets_missed=np.sum(targets_missed),  # if the target was missed
        x_rmse=x_rmse,  # RMSE of the x errors
        y_rmse=y_rmse,  # RMSE of the y errors
        z_rmse=z_rmse,  # RMSE of the z errors
        va_rmse=va_rmse,  # RMSE of the airspeed errors
        ail_fluct=fcs_fluct[0],  # aileron fluctuation
        ele_fluct=fcs_fluct[1],  # elevator fluctuation
        thr_fluct=fcs_fluct[2],  # throttle fluctuation
    )


def periodic_eval(env_id, cfg_mdp, cfg_sim, env, agent, device):
    """Periodically evaluate a given agent."""
    print("*** Evaluating the agent ***")
    env.eval = True
    results: dict = {}
    if 'AC' in env_id:
        ref_seq = attitude_seq
        results = periodic_eval_AC(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device)
    elif 'Altitude' in env_id:
        ref_seq = altitude_seq
        results = periodic_eval_alt(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device)
    elif 'Waypoint' in env_id:
        ref_seq = waypoint_seq
        results = periodic_eval_waypoints(env_id, ref_seq, cfg_mdp, cfg_sim, env, agent, device)
    env.eval = False
    return results


def clip_obs(obs):
    low_bounds = np.array([300, -300, -136, -136, -pi, -2*pi, -pi, -pi, 0, -1, 0])
    high_bounds = np.array([900, 300, 136, 136, pi, 2*pi, pi, pi, 260, 1, 1])
    return np.clip(obs, low_bounds, high_bounds)


def make_env(env_id, cfg_env, render_mode, telemetry_file=None, eval=False, gamma=0.99, run_name='', idx=0):
    def thunk():
        env = gym.make(env_id, cfg_env=cfg_env, telemetry_file=telemetry_file,
                        render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.TransformObservation(env, clip_obs)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = MyNormalizeObservation(env, eval=eval)
        # env = NormalizeObservationEnvMinMax(env)
        if not eval:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk


def constrained_waypoint_sample(n_points, radius=50, z_center=600, min_z=-10, max_z=10, min_y=None):
    # Sample all z values at once
    z = np.random.uniform(min_z, max_z, size=n_points)

    # Compute horizontal radii r for all z values
    r = np.sqrt(radius**2 - z**2)

    if min_y is not None:
        # Check if min_y is geometrically possible
        if np.any(min_y > r):
            raise ValueError(f"min_y={min_y} is larger than some computed horizontal radii (r), "
                             f"which means no valid points can be sampled at those z levels.")

        # Compute theta_min for all radii
        theta_min = np.arcsin(min_y / r)
        # Sample theta within the allowed arc [theta_min, π - theta_min]
        theta = np.random.uniform(theta_min, np.pi - theta_min, size=n_points)
    else:
        # Sample theta uniformly from 0 to 2*pi when no min_y constraint is provided
        theta = np.random.uniform(0, 2*np.pi, size=n_points)

    # Convert to x, y, z using polar to Cartesian conversion
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    z += z_center

    # Stack into final array
    points = np.column_stack((x, y, z))

    return points


def sample_targets(single_target: bool, env_id: str, env, cfg_rl: DictConfig):
    targets = None
    if 'AC' in env_id:
        roll_high = np.full((cfg_rl.num_envs, 1), np.deg2rad(cfg_rl.roll_limit))
        pitch_high = np.full((cfg_rl.num_envs, 1), np.deg2rad(cfg_rl.pitch_limit))
        roll_targets = np.random.uniform(-roll_high, roll_high)
        pitch_targets = np.random.uniform(-pitch_high, pitch_high)
        targets = np.hstack((roll_targets, pitch_targets))
    elif 'Waypoint' in env_id:
        # randomly sampling a unit vector (direction) in the ENU frame
        # don't sample if y distance is too close from starting point
        # unit_vecs = np.random.uniform(low=[-1, 0.5, -0.2],
        #                               high=[1, 1, 0.2],
        #                               size=(cfg_rl.num_envs, 3))
        # unit_vecs /= np.linalg.norm(unit_vecs)

        # # targets are 100m away and z is around the starting altitude at 600m
        # targets_enu = unit_vecs * 100 + np.full((cfg_rl.num_envs, 3), [0, 0, 600])

        targets_enu = constrained_waypoint_sample(cfg_rl.num_envs, radius=50, z_center=600, 
                                                  min_z=-10, max_z=10, min_y=None)

        # Straight line (the waypoint is always at the same x, z and y=50)
        # x_targets = np.full((cfg_rl.num_envs, 1), 0)
        # y_targets = np.full((cfg_rl.num_envs, 1), 50.0)
        # z_targets = np.full((cfg_rl.num_envs, 1), 600.0)
        # targets_enu = np.hstack((x_targets, y_targets, z_targets))
        targets = np.zeros_like(targets_enu)
        for i in range(cfg_rl.num_envs):
            targets[i] = conversions.enu2ecef(*targets_enu[i],
                                            env.unwrapped.sim['ic/lat-geod-deg'],
                                            env.unwrapped.sim['ic/long-gc-deg'],
                                            0.0)
        if 'WaypointVa' in env_id:
            targets = np.hstack((targets, np.full((cfg_rl.num_envs, 1), 60.0))) # add the airspeed target (60 kph)
    elif 'Altitude' in env_id:
        z_targets = np.random.uniform(550, 650, (cfg_rl.num_envs, 1))
        targets = z_targets
    # take the first sampled target if we want a single target and not a batch of targets for n envs
    if single_target:
        targets = targets[0]
    assert targets is not None
    return targets


# Save the model PPO
def save_model_PPO(save_path, run_name, agent, env, seed):
    save_path: str = f"models/train/ppo/{seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = f"{save_path}{run_name}.pt"
    train_dict = {}
    # train_dict["obs_rms.mean"] = env.obs_rms.mean
    # train_dict["obs_rms.var"] = env.obs_rms.var
    # print("obs_rms.mean", env.obs_rms.mean)
    # print("obs_rms.var", env.obs_rms.var)
    train_dict["seed"] = seed
    train_dict["agent"] = agent.state_dict()
    torch.save(train_dict, f"{save_path}{run_name}.pt")
    print(f"agent saved to {model_path}")


# Save the model TD3/SAC
def save_model_SAC(run_name, actor, qf1, qf2, seed):
    save_path: str = f"models/train/sac/{seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = f"{save_path}{run_name}.pt"
    torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
    print(f"agent saved to {model_path}")


# Plot 
def final_traj_plot(e_env, env_id, cfg_sim, agent, device, run_name):
    print("\n******** Plotting... ***********")
    e_env.eval = True
    telemetry_file = f"telemetry/{run_name}.csv"
    cfg_sim.eval_sim_options.seed = 10 # set a specific seed for the test traj plot
    # reset the environment with the evaluation options + modifications for rendering and telemetry
    e_obs, info = e_env.reset(options=OmegaConf.to_container(cfg_sim.eval_sim_options, resolve=True) |
                              {"render_mode": "log", "telemetry_file": telemetry_file})
    e_obs, info, done, ep_reward, t = e_obs, info, False, 0, 0
    e_obs = torch.Tensor(e_obs).unsqueeze(0).to(device)
    if 'AC' in env_id:
        roll_ref = np.deg2rad(30)
        pitch_ref = np.deg2rad(15)
        target = np.array([roll_ref, pitch_ref])
    elif 'Altitude' in env_id:
        target = np.array([630])
    elif 'Waypoint' in env_id:
        # target_enu = np.array([0, 300.0, 600.0])
        target_enu = np.array([0, 50.0, 600.0])
        target = conversions.enu2ecef(*target_enu,
                                      e_env.unwrapped.sim['ic/lat-geod-deg'],
                                      e_env.unwrapped.sim['ic/long-gc-deg'],
                                      0.0)
        if 'WaypointVa' in env_id:
            target = np.hstack((target, np.array([60.0])))

    while not done:
        e_env.unwrapped.set_target_state(target)
        if isinstance(agent, sac.Actor_SAC) or isinstance(agent, sac_norm.Actor_SAC):
            action = agent.get_action(e_obs)[2].squeeze_().detach().cpu().numpy()
        elif isinstance(agent, ppo.Agent_PPO) or isinstance(agent, ppo_norm.Agent_PPO):
            action = agent.get_action_and_value(e_obs)[1][0].detach().cpu().numpy()
        elif isinstance(agent, TDMPC2):
            action = agent.act(e_obs.squeeze(0), t0=t==0, eval_mode=True)
        e_obs, reward, truncated, terminated, info = e_env.step(action)
        e_obs = torch.Tensor(e_obs).unsqueeze(0).to(device)
        done = np.logical_or(truncated, terminated)
        t += 1

        if done:
            print(f"Episode reward: {info['episode']['r']}")
            break
    telemetry_df = pd.read_csv(telemetry_file)
    traj_3d_points = telemetry_df[['position_enu_x_m', 'position_enu_y_m', 'position_enu_z_m']].to_numpy()
    traj_3d = go.Scatter3d(x=traj_3d_points[:, 0], y=traj_3d_points[:, 1], z=traj_3d_points[:, 2], mode='lines')
    start_point = go.Scatter3d(x=[traj_3d_points[0, 0]], y=[traj_3d_points[0, 1]], z=[traj_3d_points[0, 2]], mode='markers', marker=dict(size=5, color='red'))
    target_point = go.Scatter3d(x=[target_enu[0]], y=[target_enu[1]], z=[target_enu[2]], mode='markers', marker=dict(size=5, color='green'))
    fig = go.Figure(data=[traj_3d, start_point, target_point])

    # compute figure axis limits based on the trajectory points and target point
    x_min, x_max = min(traj_3d_points[:, 0].min(), target_enu[0]) - 10, max(traj_3d_points[:, 0].max(), target_enu[0]) + 10
    y_min, y_max = min(traj_3d_points[:, 1].min(), target_enu[1]) - 10, max(traj_3d_points[:, 1].max(), target_enu[1]) + 10
    z_min, z_max = min(traj_3d_points[:, 2].min(), target_enu[2]) - 10, max(traj_3d_points[:, 2].max(), target_enu[2]) + 10

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max]),
        )
    ) 
    wandb.log({"FinalTraj/telemetry": wandb.Table(dataframe=telemetry_df),
               "FinalTraj/3D_trajectory": wandb.Plotly(fig)})
