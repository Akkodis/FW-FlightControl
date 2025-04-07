import numpy as np
import torch
from fw_jsbgym.utils import conversions
from fw_jsbgym.utils import jsbsim_properties as prp
from fw_flightcontrol.agents.pid import PID


def prepare_targets(env, targets_enu, cfg_rl, pid=False):
    """Prepare target coordinates for simulation"""
    targets = None
    targets_ecef = np.zeros((targets_enu.shape[0], 3))

    if cfg_rl.task == "WaypointTrackingENU" or cfg_rl.task == "StraightPathTracking" \
        or cfg_rl.task == "CourseAltTracking":
        targets = targets_enu
    elif cfg_rl.task == "WaypointTracking":
        for i, target_enu in enumerate(targets_enu):
            # Convert ENU to ECEF coordinates
            targets_ecef[i] = conversions.enu2ecef(
                *target_enu,
                env.unwrapped.sim[prp.ic_lat_gd_deg],
                env.unwrapped.sim[prp.ic_long_gc_deg],
                0.0,
            )
        targets = targets_ecef

    if "Va" in cfg_rl.task:
        airspeed_targets = np.full((targets_enu.shape[0], 1), 60.0)
        targets = np.hstack((targets, airspeed_targets))

    assert targets is not None, "Targets is None, check the eval_sim.prepare_targets() function"
    print(targets)
    return targets


def pid_action(agent, env, path_target, wp_target) -> torch.Tensor:
    """
        Cascaded PID controller for straight line path tracking
    """
    # Longitudinal control
    # setting PID references
    agent["altitude_pid"].set_reference(path_target[1])
    agent["airspeed_pid"].set_reference(path_target[2])
     # airspeed -> throttle
    throttle, _, _ = agent["airspeed_pid"].update(state=env.unwrapped.sim[prp.airspeed_kph], 
                                                    saturate=True)
    # get actions from PIDs
    # altitude -> pitch -> elevator
    pitch_ref, _, _ = agent["altitude_pid"].update(state=env.unwrapped.sim[prp.enu_u_m], 
                                                    saturate=True)
    agent["pitch_pid"].set_reference(pitch_ref)
    elevator_cmd, _, _ = agent["pitch_pid"].update(state=env.unwrapped.sim[prp.pitch_rad],
                                                    state_dot=env.unwrapped.sim[prp.q_radps],
                                                    saturate=True, normalize=True)
    # Lateral control
    # course -> roll -> aileron
    # cross track error
    path_tracking = False
    xi_uav = np.arctan2(  # course angle of the UAV
        env.unwrapped.sim[prp.v_east_fps],
        env.unwrapped.sim[prp.v_north_fps]
    )
    if path_tracking:
        xi_q = path_target[0] # course angle of the target waypoint
        xi_uav = np.arctan2(  # course angle of the UAV
            env.unwrapped.sim[prp.v_east_fps],
            env.unwrapped.sim[prp.v_north_fps]
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
    else:
        uav_to_wp_n = wp_target[1] - env.unwrapped.sim[prp.enu_n_m]
        uav_to_wp_e = wp_target[0] - env.unwrapped.sim[prp.enu_e_m]
        course_desired = np.arctan2(uav_to_wp_e, uav_to_wp_n)
        if (course_desired - xi_uav) < -np.pi:
            course_desired = course_desired + 2*np.pi
        elif (course_desired - xi_uav) > np.pi:
            course_desired = course_desired - 2*np.pi
        agent["course_pid"].set_reference(course_desired)

    roll_ref, error_course, _ = agent["course_pid"].update(state=xi_uav,
                                                saturate=True, is_course=False)

    agent["roll_pid"].set_reference(roll_ref)
    aileron_cmd, _, _ = agent["roll_pid"].update(state=env.unwrapped.sim[prp.roll_rad],
                                                    state_dot=env.unwrapped.sim[prp.p_radps],
                                                    saturate=True, normalize=True)

    action = torch.Tensor([aileron_cmd, elevator_cmd, throttle])
    return action


def run_simulations(env, agent, agent_name, targets_wp, severity_range, jsbsim_seeds, cfg_sim, trim=None):
    """Run simulations for all severity levels and episodes"""
    num_ep = targets_wp.shape[0]
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

        print(f"********** {agent_name.upper()} METRICS {severity} **********")

        for ep_cnt, target_wp in enumerate(targets_wp):
            # print episode number / total number of episodes
            print(f"-- Episode {total_ep_cnt + 1} / {total_num_ep} --") 

            # Prepare path target for PID
            if agent_name.casefold() == "pid":
                path_target = conversions.wpENU_to_wpCourseAlt(target_wp)
                path_target = np.hstack((path_target, 60.0)) # add airspeed target

            t = 0  # Reset timestep counter for each episode
            
            # Change seed for each episode
            cfg_sim.eval_sim_options.seed = float(jsbsim_seeds[ep_cnt])
            
            # Reset environment
            obs, _ = env.reset(cfg_sim.eval_sim_options)

            # Run episode
            while True:
                # Set target and get action
                env.set_target_state(target_wp)
                if agent_name.casefold() == "pid":
                    action = pid_action(agent, env, path_target, target_wp)
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