import torch
import numpy as np
import os
import sys
import csv
import hydra

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../../agents/tdmpc2/tdmpc2/')

from omegaconf import DictConfig
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2


@hydra.main(version_base=None, config_path="../../config", config_name="tdmpc2_default")
def eval(cfg: DictConfig):
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    print(f"**** Using Device: {device} ****")

    cfg.rl = parse_cfg(cfg.rl)
    os.chdir(hydra.utils.get_original_cwd())

    # shorter cfg aliases
    cfg_rl = cfg.rl
    cfg_sim = cfg.env.jsbsim

    # env setup
    env = make_env(cfg)
    state_names = list(prp.get_legal_name() for prp in env.state_prps)

    # Load agent
    agent = TDMPC2(cfg.rl)
    assert os.path.exists(cfg.rl.checkpoint), f"Checkpoint {cfg.rl.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.rl.checkpoint)

    # load the reference sequence and initialize the evaluation arrays
    simple_ref_data = np.load(f'eval/attitude_control/targets/{cfg_rl.ref_file}.npy')

    # load the jsbsim seeds to apply at each reset and set the first seed
    jsbsim_seeds = np.load(f'eval/attitude_control/targets/jsbsim_seeds.npy')
    cfg_sim.eval_sim_options.seed = float(jsbsim_seeds[0])

    # set default target values
    # roll_ref: float = np.deg2rad(58)
    # pitch_ref: float = np.deg2rad(28)

    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if cfg_sim.render_mode == "none":
        total_steps = 50_000
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 2000

    if cfg_sim.eval_sim_options.atmosphere.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [cfg_sim.eval_sim_options.atmosphere.severity]

    all_rmse = []
    all_fcs_fluct = []

    if not os.path.exists("eval/outputs"):
        os.makedirs("eval/outputs")

    eval_res_csv = f"eval/outputs/{cfg_rl.res_file}.csv"
    eval_fieldnames = ["severity", "roll_rmse", "pitch_rmse",
                        "roll_fcs_fluct", "pitch_fcs_fluct",
                        "avg_rmse", "avg_fcs_fluct"]

    with open(eval_res_csv, "w") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
        csv_writer.writeheader()

    for i, severity in enumerate(severity_range):
        cfg_sim.eval_sim_options.atmosphere.severity = severity
        e_obs = []
        eps_fcs_fluct = []
        print(f"********** TDMPC2 METRICS {severity} **********")
        obs, _ = env.reset(options=cfg_sim.eval_sim_options)
        z_obs = agent.model.encode(torch.Tensor(obs).to(device), None)
        ep_cnt = 0 # episode counter
        ep_step = 0 # step counter within an episode
        step, t = 0, 0
        targets = simple_ref_data[ep_cnt]
        # default target values
        # roll_ref = np.deg2rad(-10)
        # pitch_ref = np.deg2rad(15)
        # targets = np.array([roll_ref, pitch_ref])
        while step < total_steps:
            env.set_target_state(targets)
            # action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
            action = agent.act(obs, t0=t==0, eval_mode=True)
            obs, reward, term, trunc, info = env.step(action)
            done = np.logical_or(term, trunc)

            # imagined trajectory
            if cfg.rl.im_traj:
                z_obs = agent.model.next(z_obs, torch.Tensor(action).to(device), None)
                im_obs = agent.model.decode(z_obs, None)
                im_dec_state_names = list('im_dec_' + state_name for state_name in state_names)
                im_dec_obs_dict = dict(zip(im_dec_state_names, im_obs.cpu().detach().numpy()))
                env.telemetry_logging(im_dec_obs_dict)

            # decoded observation
            if cfg.rl.dec_obs:
                obs = torch.Tensor(obs).to(device)
                encoded_obs = agent.model.encode(obs, None)
                decoded_obs = agent.model.decode(encoded_obs, None)
                dec_state_names = list('dec_' + state_name for state_name in state_names)
                dec_obs_dict = dict(zip(dec_state_names, decoded_obs.cpu().detach().numpy()))
                env.telemetry_logging(dec_obs_dict)

            e_obs.append(info["non_norm_obs"])
            t += 1
            # if t == 30:
            #     break

            if done:
                t = 0
                if info['out_of_bounds']:
                    print("Out of bounds")
                    e_obs[len(e_obs)-ep_step:] = [] # delete the last observations if the ep is oob
                    step -= ep_step # set the step counter back to the last episode
                    ep_step = 0 # reset the episode step counter
                else:
                    ep_step = 0 # reset the episode step counter
                    ep_cnt += 1 # increment the episode counter
                print(f"Episode reward: {info['episode']['r']}")
                print(f"******* {step}/{total_steps} *******")
                # break
                obs, last_info = env.reset(options={"seed": float(jsbsim_seeds[ep_cnt])})
                ep_fcs_pos_hist = np.array(last_info["fcs_pos_hist"]) # get fcs pos history of the finished episode
                eps_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # get fcs fluctuation of the episode and append it to the list of all fcs fluctuations
                if ep_cnt < len(simple_ref_data):
                    targets = simple_ref_data[ep_cnt]
                roll_ref, pitch_ref = targets[0], targets[1]
            ep_step += 1
            step += 1

        all_fcs_fluct.append(np.mean(np.array(eps_fcs_fluct), axis=0))
        e_obs = np.array(e_obs)
        print(f"e_obs shape: {e_obs.shape}")
        print(f"eps_fcs_fluct shape: {np.array(eps_fcs_fluct).shape}")
        roll_rmse = np.sqrt(np.mean(np.square(e_obs[:, 6])))
        pitch_rmse = np.sqrt(np.mean(np.square(e_obs[:, 7])))
        all_rmse.append([roll_rmse, pitch_rmse])

    for rmse, fcs_fluct, severity in zip(all_rmse, all_fcs_fluct, severity_range):
        print("\nSeverity: ", severity)
        print(f"  Roll RMSE: {rmse[0]:.4f}\n  Pitch RMSE: {rmse[1]:.4f}")
        print(f"  Roll fluctuation: {fcs_fluct[0]:.4f}\n  Pitch fluctuation: {fcs_fluct[1]:.4f}")
        print(f" Average RMSE: {np.mean(rmse):.4f}\n Average fluctuation: {np.mean(fcs_fluct):.4f}")
        with open(eval_res_csv, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
            csv_writer.writerow({"severity": severity, "roll_rmse": rmse[0], "pitch_rmse": rmse[1], 
                                "roll_fcs_fluct": fcs_fluct[0], "pitch_fcs_fluct": fcs_fluct[1],
                                "avg_rmse": np.mean(rmse), "avg_fcs_fluct": np.mean(fcs_fluct)})

    env.close()


if __name__ == '__main__':
    eval()