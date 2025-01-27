import random
import torch
import numpy as np
import os
import csv
import hydra
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from fw_flightcontrol.agents.sac_norm import Actor_SAC
from fw_flightcontrol.utils.train_utils import make_env
from fw_jsbgym.trim.trim_point import TrimPoint


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def eval(cfg: DictConfig):
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**** Using Device: {device} ****")

    # seeding
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # shorter cfg aliases
    cfg_sac = cfg.rl.SAC
    cfg_sim = cfg.env.jsbsim
    cfg_task = cfg.env.task

    # env setup
    env = make_env('AltitudeTracking-v0', cfg.env, cfg_sim.render_mode, 
                   'telemetry/telemetry.csv', eval=True)()

    # loading the agent
    train_dict = torch.load(cfg.model_path, map_location=device)[0] # only load the actor's state dict
    sac_agent = Actor_SAC(env).to(device)
    sac_agent.load_state_dict(train_dict)
    sac_agent.eval()

    trim = TrimPoint('x8')
    trim_action = np.array([trim.aileron, trim.elevator, trim.throttle])

    obs, _ = env.reset(options=cfg_sim.eval_sim_options)
    ep_obss = [obs]
    obs = torch.Tensor(obs).unsqueeze_(0).to(device)

    ep_rewards = [0]
    rolls = [env.unwrapped.sim['attitude/roll-rad']]
    alts = [env.unwrapped.sim['position/enu-z-m']]
    enu_xs = [env.unwrapped.sim['position/enu-x-m']]
    enu_ys = [env.unwrapped.sim['position/enu-y-m']]
    step = 0
    target = np.array([560])
    total_steps = 2000

    while step < total_steps:
        env.set_target_state(target)
        action = sac_agent.get_action(torch.Tensor(obs).unsqueeze(0).to(device))[2].squeeze_().detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_obss.append(obs)
        ep_rewards.append(reward)
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        rolls.append(env.unwrapped.sim['attitude/roll-rad'])
        alts.append(env.unwrapped.sim['position/enu-z-m'])
        enu_xs.append(env.unwrapped.sim['position/enu-x-m'])
        enu_ys.append(env.unwrapped.sim['position/enu-y-m'])
        done = np.logical_or(terminated, truncated)

        if done:
            if info['out_of_bounds']:
                print("Out of bounds")
                break

            print(f"Episode reward: {info['episode']['r']}")
            print(f"******* {step}/{total_steps} *******")
            break
            obs, last_info = env.reset()
        step += 1
    env.close()

    ep_obss = np.array(ep_obss)
    ep_rewards = np.array(ep_rewards)
    enu_xs = np.array(enu_xs)
    enu_ys = np.array(enu_ys)
    alts = np.array(alts)
    errs_z = ep_obss[:, 0]
    dist_to_target = np.sqrt(errs_z**2)
    tsteps = np.linspace(0, ep_obss.shape[0], ep_obss.shape[0])
    fig, ax = plt.subplots(2, 3)

    ax[0,0].plot(tsteps, np.rad2deg(ep_obss[:, 1]))
    ax[0,0].set_title("Pitch")

    ax[0,1].plot(tsteps, alts)
    ax[0,1].set_title("Altitude")

    ax[0,2].plot(tsteps, ep_obss[:, 3])
    ax[0,2].set_title("Airspeed")
    
    # plot the positions in a 3D plot with the line being a gradient of colors between red and purple, red being the first timestep and purple the last
    ax[0,2].remove()
    ax[0,2] = fig.add_subplot(1,3,3, projection='3d')
    # ax[0,2].plot([target[0]], [target[1]], [target[2]], 'ro')
    ax[0,2].set_title("X")
    ax[0,2].set_ylabel("Y")
    ax[0,2].set_zlabel("Z")
    # ax[0,2].set_xlim(-400, 400)
    # ax[0,2].set_ylim(-100, 400)
    # ax[0,2].set_zlim(400, 800)

 
    for i in range(ep_obss.shape[0] - 1):
        ax[0,2].plot(enu_xs[i:i+2], enu_ys[i:i+2], alts[i:i+2], c=plt.cm.plasma(i/enu_xs.shape[0]))

    ax[1,0].plot(tsteps, dist_to_target)
    ax[1,0].set_title("Distance to target")

    ax[1,1].plot(tsteps, ep_rewards)
    ax[1,1].set_title("Rewards")

    # ax[1,1].plot(tsteps, np.rad2deg(rolls))
    # ax[1,1].set_title("Roll")

    ax[1,2].plot(tsteps, ep_obss[:, 9], label="elevator")
    ax[1,2].plot(tsteps, ep_obss[:, 10], label="throttle")
    ax[1,2].set_title("Commands")
    ax[1,2].legend()

    # print(f"Last position: {ep_obss[-1, :3]}")

    plt.show()



if __name__ == '__main__':
    eval()