import numpy as np
from fw_jsbgym.utils import conversions
from fw_jsbgym.utils import jsbsim_properties as prp


def prepare_targets(env, targets_enu, cfg_rl):
    """Prepare target coordinates for simulation"""
    targets_ecef = np.zeros_like(targets_enu)
    for i, target_enu in enumerate(targets_enu):
        targets_ecef[i] = conversions.enu2ecef(*target_enu,
                                              env.unwrapped.sim['ic/lat-geod-deg'],
                                              env.unwrapped.sim['ic/long-gc-deg'],
                                              0.0)

    if cfg_rl.task == 'WaypointVaTracking':
        targets_ecef = np.hstack((targets_ecef, np.array([60.0])))
        
    return targets_ecef

def run_simulations(env, agent, targets_ecef, severity_range, jsbsim_seeds, cfg_sim):
    """Run simulations for all severity levels and episodes"""
    num_ep = targets_ecef.shape[0]
    total_num_ep = len(severity_range) * num_ep
    total_ep_cnt = 0
    
    # Initialize arrays to store results
    enu_positions = np.full((len(severity_range), num_ep, 2000, 3), np.nan)
    orientations = np.full((len(severity_range), num_ep, 2000, 3), np.nan)
    wind_vector = np.full((len(severity_range), num_ep, 2000, 3), np.nan)
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
        else:
            cfg_sim.eval_sim_options.atmosphere.wind.wind_severity = severity


        print(f"********** TDMPC2 METRICS {severity} **********")
        
        for ep_cnt, target_ecef in enumerate(targets_ecef):
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
                action = agent.act(obs, t0=t==0, eval_mode=True)

                # Record position and orientation
                enu_positions[sev_cnt, ep_cnt, t] = [
                    env.unwrapped.sim[prp.enu_x_m],
                    env.unwrapped.sim[prp.enu_y_m],
                    env.unwrapped.sim[prp.enu_z_m]
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
                    break
    
    return enu_positions, orientations, wind_vector, ep_fcs_fluct, target_success