import numpy as np
import torch
from fw_jsbgym.utils import conversions
from fw_jsbgym.utils import jsbsim_properties as prp


def prepare_targets(env, targets_enu, cfg_rl, pid=False):
    """Prepare target coordinates for simulation"""
    targets = np.zeros_like(targets_enu)
    pid_targets = None

    # Convert ENU to ECEF coordinates if not in ENU mode
    if "ENU" not in cfg_rl.task:
        for i, target_enu in enumerate(targets_enu):
                targets[i] = conversions.enu2ecef(
                    *target_enu,
                    env.unwrapped.sim['ic/lat-geod-deg'],
                    env.unwrapped.sim['ic/long-gc-deg'],
                    0.0
                )
    else: # else use the directly provided ENU coordinates
        targets = targets_enu

    if cfg_rl.task == 'WaypointVaTracking':
        print(f"targets: {targets}")
        airspeed_targets = np.full((targets.shape[0], 1), 60.0)
        targets = np.hstack((targets, airspeed_targets))
    
    if pid:
        pid_targets = prepare_pid_targets(env, targets_enu)
        
    return targets, pid_targets


def prepare_pid_targets(env, targets_enu):
    """Prepare target coordinates for PID simulation"""
    pid_targets: list[dict[str, float]] = [{} for _ in range(len(targets_enu))]
    for i, target_enu in enumerate(targets_enu):
        print(f"Target ENU: {target_enu}")
        course_target = np.arctan2(target_enu[0], target_enu[1])
        altitude_target = target_enu[2]
        airspeed_target = 60.0
        pid_targets[i] = {
            'course_target': course_target,
            'altitude_target': altitude_target,
            'airspeed_target': airspeed_target
        }
    print(f"PID Targets: {np.array(pid_targets)}")
    return np.array(pid_targets)


def pid_action(agent, env, pid_targets, ep_cnt) -> torch.Tensor:
    # print(f"Target NED: x: {env.unwrapped.sim[prp.target_ned_x_m]}, "\
    #                 f"y: {env.unwrapped.sim[prp.target_ned_y_m]}, "\
    #                 f"z: {env.unwrapped.sim[prp.target_ned_z_m]}")

    # Longitudinal control
    # setting PID references
    agent["altitude_pid"].set_reference(pid_targets[ep_cnt]['altitude_target'])
    agent["airspeed_pid"].set_reference(pid_targets[ep_cnt]['airspeed_target'])
     # airspeed -> throttle
    throttle, _, _ = agent["airspeed_pid"].update(state=env.unwrapped.sim[prp.airspeed_kph], 
                                                    saturate=True)
    # get actions from PIDs
    # altitude -> pitch -> elevator
    pitch_ref, _, _ = agent["altitude_pid"].update(state=env.unwrapped.sim[prp.enu_z_m], 
                                                    saturate=True)
    agent["pitch_pid"].set_reference(pitch_ref)
    elevator_cmd, _, _ = agent["pitch_pid"].update(state=env.unwrapped.sim[prp.pitch_rad],
                                                    state_dot=env.unwrapped.sim[prp.q_radps],
                                                    saturate=True, normalize=True)
    # Lateral control
    # course -> roll -> aileron
    # cross track error
    xi_q = pid_targets[ep_cnt]['course_target'] # course angle of the target waypoint
    xi_uav = (np.arctan2(env.unwrapped.sim[prp.v_east_fps], env.unwrapped.sim[prp.v_north_fps]
        )
        uav_x_n = env.unwrapped.sim[prp.enu_n_m]
        uav_y_e = env.unwrapped.sim[prp.enu_e_m]
    wp_prev_x_n = 0.0
    wp_prev_y_e = 0.0

    if xi_q - xi_uav < -np.pi:
        xi_q = xi_q + 2*np.pi
    elif xi_q - xi_uav > np.pi:
        xi_q = xi_q - 2*np.pi
    e_c = -np.sin(xi_q) * (uav_x_n - wp_prev_x_n) + np.cos(xi_q) * (uav_y_e - wp_prev_y_e)
    xi_inf = np.pi/2
    k_path = 0.01
    course_desired = xi_q - xi_inf * 2 / np.pi * np.arctan(k_path * e_c)
    agent["course_pid"].set_reference(course_desired)

    roll_ref, error_course, _ = agent["course_pid"].update(state=xi_uav,
                                                saturate=True, is_course=False)

    agent["roll_pid"].set_reference(roll_ref)
    aileron_cmd, _, _ = agent["roll_pid"].update(state=env.unwrapped.sim[prp.roll_rad],
                                                    state_dot=env.unwrapped.sim[prp.p_radps],
                                                    saturate=True, normalize=True)

    action = torch.Tensor([aileron_cmd, elevator_cmd, throttle])
    return action


def run_simulations(env, agent, targets, severity_range, jsbsim_seeds, cfg_sim, pid_targets=None, trim=None):
    """Run simulations for all severity levels and episodes"""
    num_ep = targets.shape[0]
    total_num_ep = len(severity_range) * num_ep
    total_ep_cnt = 0
    
    # Initialize arrays to store results
    enu_positions = np.full((len(severity_range), num_ep, env.max_episode_steps, 3), np.nan)
    orientations = np.full((len(severity_range), num_ep, env.max_episode_steps, 3), np.nan)
    wind_vector = np.full((len(severity_range), num_ep, env.max_episode_steps, 3), np.nan)
    ep_fcs_fluct = np.full((len(severity_range), num_ep, 3), np.nan)
    target_success = np.zeros((len(severity_range), num_ep))
    
    # Run simulations for all severity levels
    for sev_cnt, severity in enumerate(severity_range):
        # Set atmosphere severity and wind settings
        cfg_sim.eval_sim_options.atmosphere.severity = severity
        cfg_sim.eval_sim_options.atmosphere.wind.enable = (severity != 'off')
        # if the substring "wind" is the current severity, set the wind severity
        # by getting the substring after the underscore (e.g. "wind_5kph" -> 5)
        if ("wind" in severity):
            cfg_sim.eval_sim_options.atmosphere.wind.wind_severity = float(severity.split('_')[1][:-3])


        print(f"********** TDMPC2 METRICS {severity} **********")
        
        for ep_cnt, target_ecef in enumerate(targets):
            # print episode number / total number of episodes
            print(f"-- Episode {total_ep_cnt + 1} / {total_num_ep} --")

            t = 0  # Reset timestep counter for each episode
            
            # Change seed for each episode
            cfg_sim.eval_sim_options.seed = float(jsbsim_seeds[ep_cnt])
            
            # Reset environment
            obs, _ = env.reset(cfg_sim.eval_sim_options)
            
            # Run episode
            while True:
                # Set target and get action
                env.set_target_state(target_ecef)
                if isinstance(agent, dict) and pid_targets is not None:
                    action = pid_action(agent, env, pid_targets, ep_cnt)
                else:
                    action = agent.act(obs, t0=t==0, eval_mode=True)

                # action = torch.Tensor([trim.aileron, trim.elevator, trim.throttle])

                # Record position and orientation
                enu_positions[sev_cnt, ep_cnt, t] = [
                    env.unwrapped.sim[prp.enu_e_m],
                    env.unwrapped.sim[prp.enu_n_m],
                    env.unwrapped.sim[prp.enu_u_m]
                ]
                
                orientations[sev_cnt, ep_cnt, t] = [
                    env.unwrapped.sim[prp.roll_rad],
                    env.unwrapped.sim[prp.pitch_rad],
                    env.unwrapped.sim[prp.heading_rad]
                ]
                
                wind_vector[sev_cnt, ep_cnt, t] = [
                    env.unwrapped.sim[prp.windspeed_north_mps],
                    env.unwrapped.sim[prp.windspeed_east_mps],
                    env.unwrapped.sim[prp.windspeed_down_mps]
                ]

                # Execute action
                obs, reward, term, trunc, info = env.step(action)
                t+=1
                
                done = np.logical_or(term, trunc)
                
                if done:
                    total_ep_cnt += 1
                    # Record success and FCS fluctuation
                    target_success[sev_cnt, ep_cnt] = info['success']
                    print(f"Episode reward: {info['episode']['r']}, finished at step {t}")
                    
                    # Get FCS fluctuation
                    ep_fcs_pos_hist = np.array(env.get_fcs_hist())
                    ep_fcs_fluct[sev_cnt, ep_cnt] = np.nanmean(
                        np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0
                    )
                    if isinstance(agent, dict):
                        for pid in agent.values():
                            pid.reset() # reset PID controllers (integral and prev_error)
                    break

    return enu_positions, orientations, wind_vector, ep_fcs_fluct, target_success