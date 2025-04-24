import torch
import numpy as np
import os
import sys
import hydra
import matplotlib.pyplot as plt
from fw_jsbgym.trim.trim_point import TrimPoint

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../agents/tdmpc2/tdmpc2/')

from omegaconf import DictConfig
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from fw_flightcontrol.eval.waypoint_tracking.utils import eval_sim, metrics


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
    trim = TrimPoint('x8')

    # Load agent
    agent = TDMPC2(cfg.rl)
    assert os.path.exists(cfg.rl.checkpoint), f"Checkpoint {cfg.rl.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.rl.checkpoint)

    # get the checkpoint file name (without the parent directories, and the .pt at the end)
    checkpoint_name = '_'.join(cfg.rl.checkpoint.split('/')[-1].split('_'))[:-3]
    print(f"**** Using Checkpoint: {checkpoint_name} ****")

    # Load seeds and determine severity levels
    jsbsim_seeds = np.load(f'eval/waypoint_tracking/targets/jsbsim_100seeds.npy')
    
    if cfg_sim.eval_sim_options.atmosphere.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = cfg_sim.eval_sim_options.atmosphere.severity

    print(f"{severity_range=}")
    atmo_type: str = 'noatmo'
    if cfg_sim.eval_sim_options.atmosphere.turb.enable:
        atmo_type = 'turb'
    if cfg_sim.eval_sim_options.atmosphere.gust.enable:
        atmo_type = 'gusts'
    if cfg_sim.eval_sim_options.atmosphere.wind.enable and not cfg_sim.eval_sim_options.atmosphere.turb.enable:
        atmo_type = 'wind'
    print(f"**** Using Atmosphere Type: {atmo_type} ****")

    # save simulated episodes to a single numpy file
    npz_file = f'eval/waypoint_tracking/outputs/eval_trajs/{atmo_type}_{checkpoint_name}.npz'

    # Load and prepare targets
    targets_np_file = 'eval/waypoint_tracking/targets/target_25points360_50-200m.npy'
    targets_enu = np.load(targets_np_file)
    targets_wp: np.ndarray = eval_sim.prepare_targets(env, targets_enu, cfg_rl)

    if cfg_rl.eval.run_eval_sims:
        # Run all simulations
        enu_positions, orientations, wind_vector, ep_fcs_fluct, target_success = eval_sim.run_simulations(
            env, agent, "TDMPC2", targets_wp, severity_range, jsbsim_seeds, cfg_sim, trim=trim
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
    
    _, avg_distance, avg_distance_normalized = metrics.compute_distance(
        enu_positions, severity_range, targets_enu, targets_enu.shape[0]
    )
    
    _, avg_time = metrics.compute_time(
        enu_positions, severity_range, targets_enu.shape[0], env.fdm_dt
    )
    
    # Save metrics to CSV
    csv_filename = f"eval/waypoint_tracking/outputs/metrics/{atmo_type}_{checkpoint_name}.csv"
    csv_file = metrics.save_metrics_summary(
        csv_filename,
        severity_range, total_targets, success_dict, success_percent, 
        avg_fcs_fluct, avg_distance,avg_distance_normalized, avg_time
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