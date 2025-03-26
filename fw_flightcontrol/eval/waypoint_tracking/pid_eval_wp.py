import torch
import numpy as np
import os
import sys
import hydra
import random
import matplotlib.pyplot as plt
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../agents/tdmpc2/tdmpc2/')

from omegaconf import DictConfig
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from fw_flightcontrol.eval.waypoint_tracking.utils import eval_sim, metrics

from fw_flightcontrol.agents.pid import PID


@hydra.main(version_base=None, config_path="../../config", config_name="tdmpc2_default")
def eval(cfg: DictConfig):
    """Main evaluation function"""
    # Setup environment and device
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**** Using Device: {device} ****")

    # Parse config
    cfg.rl = parse_cfg(cfg.rl)
    os.chdir(hydra.utils.get_original_cwd())

    # Shorter cfg aliases
    cfg_rl = cfg.rl
    cfg_sim = cfg.env.jsbsim

    # Environment setup
    env = make_env(cfg)
    x8 = AeroModel()
    trim = TrimPoint('x8')

    # Define the PID loops
    # lateral dynamics
    # inner loop
    kp_roll: float = 0.5
    ki_roll: float = 0.0
    kd_roll: float = 0.3 # default 0.3
    roll_pid: PID = PID(kp=kp_roll, ki=ki_roll, kd=kd_roll,
                    dt=env.fdm_dt, limit=x8.aileron_limit)

    # outer loop
    kp_course: float = 0.6 # default 0.4
    ki_course: float = 0.0 # default 0.0
    course_pid: PID = PID(kp=kp_course, ki=ki_course,
                     dt=env.fdm_dt, limit=x8.roll_max)
    
    # longitudinal dynamics
    # inner loop
    kp_pitch: float = -19.0
    ki_pitch: float = -0.0
    kd_pitch: float = -2.0
    pitch_pid: PID = PID(kp=kp_pitch, ki=ki_pitch, kd=kd_pitch,
                        dt=env.fdm_dt, trim=trim, limit=x8.aileron_limit)

    kp_alt: float = 0.1
    ki_alt: float = 0.0001
    kd_alt: float = 0.0
    altitude_pid: PID = PID(kp=kp_alt, ki=ki_alt, kd=kd_alt,
                            dt=env.fdm_dt, trim=trim, limit=x8.pitch_max)

    kp_airspeed: float = 1.2
    ki_airspeed: float = 0.0
    kd_airspeed: float = 0.0
    airspeed_pid: PID = PID(kp=kp_airspeed, ki=ki_airspeed, kd=kd_airspeed,
                        dt=env.fdm_dt, trim=trim, limit=x8.throttle_limit, is_throttle=True)
    
    agent: dict[str, PID] = {
        'roll_pid': roll_pid,
        'course_pid': course_pid,
        'pitch_pid': pitch_pid,
        'altitude_pid': altitude_pid,
        'airspeed_pid': airspeed_pid
    }

    # Load seeds and determine severity levels
    jsbsim_seeds = np.load(f'eval/waypoint_tracking/targets/jsbsim_seeds.npy')
    
    if cfg_sim.eval_sim_options.atmosphere.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [cfg_sim.eval_sim_options.atmosphere.severity]

    atmo_type: str = 'noatmo'
    if cfg_sim.eval_sim_options.atmosphere.turb.enable:
        atmo_type = 'turb'
    if cfg_sim.eval_sim_options.atmosphere.gust.enable:
        atmo_type = 'gusts'
    if cfg_sim.eval_sim_options.atmosphere.wind.enable and not cfg_sim.eval_sim_options.atmosphere.turb.enable:
        atmo_type = 'wind'
        # severity_range = ["off", "wind_5kph", "wind_10kph", "wind_20kph", "wind_30kph"]
        severity_range = ["off"]
    print(f"**** Using Atmosphere Type: {atmo_type} ****")

    # save simulated episodes to a single numpy file
    npz_file = f'eval/waypoint_tracking/outputs/eval_trajs/{atmo_type}_pid.npz'

    # Load and prepare targets
    targets_np_file = 'eval/waypoint_tracking/targets/target_points360_200m.npy'
    targets_enu = np.load(targets_np_file)
    targets_enu = targets_enu[:10]
    # targets_enu = np.array([targets_enu[0]])
    # targets_enu = np.array([[47.473, 15.478, 602.59]])
    # targets_enu = np.array([[15.473, 43.478, 602.59]])
    targets_ecef, pid_targets = eval_sim.prepare_targets(env, targets_enu, cfg_rl, pid=True)

    if cfg_rl.eval.run_eval_sims:
        # Run all simulations
        enu_positions, orientations, wind_vector, ep_fcs_fluct, target_success = eval_sim.run_simulations(
            env, agent, targets_ecef, severity_range, jsbsim_seeds, cfg_sim, 
            trim=trim, pid_targets=pid_targets
        )

        np.savez(
            npz_file, 
            enu_positions=enu_positions, 
            orientations=orientations,
            wind_vector=wind_vector,
            ep_fcs_fluct=ep_fcs_fluct, 
            target_success=target_success
        )

        # Close environment
        env.close()

    # Read trajs from the saved npz file
    npz_data = np.load(npz_file)
    enu_positions = npz_data['enu_positions']
    orientations = npz_data['orientations']
    wind_vector = npz_data['wind_vector']
    ep_fcs_fluct = npz_data['ep_fcs_fluct']
    target_success = npz_data['target_success']

    # Compute metrics
    total_targets, success_percent, success_dict = metrics.compute_target_success(
        target_success, severity_range
    )
    
    avg_fcs_fluct = metrics.compute_fcs_fluctuation(
        ep_fcs_fluct, severity_range
    )
    
    _, avg_distance = metrics.compute_distance(
        enu_positions, severity_range, targets_enu.shape[0]
    )
    
    _, avg_time = metrics.compute_time(
        enu_positions, severity_range, targets_enu.shape[0], env.fdm_dt
    )
    
    # Save metrics to CSV
    csv_filename = f"eval/waypoint_tracking/outputs/metrics/{atmo_type}_pid.csv"
    csv_file = metrics.save_metrics_summary(
        csv_filename,
        severity_range, total_targets, success_dict, success_percent, 
        avg_fcs_fluct, avg_distance, avg_time
    )

    # Plot trajectories if requested
    if cfg_rl.eval.plot_trajs:
        fig = metrics.plot_trajectories(
            enu_positions, orientations, wind_vector, targets_enu, 
            target_success, severity_range, cfg_rl.eval.plot_frames
        )
        plt.show()

if __name__ == '__main__':
    eval()