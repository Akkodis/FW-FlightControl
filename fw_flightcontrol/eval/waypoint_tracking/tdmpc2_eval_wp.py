import torch
import numpy as np
import os
import sys
import hydra
import random
import matplotlib.pyplot as plt

sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../agents/tdmpc2/tdmpc2/')

from omegaconf import DictConfig
from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env
from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2


@hydra.main(version_base=None, config_path="../config", config_name="tdmpc2_default")
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

    cfg.rl = parse_cfg(cfg.rl)
    os.chdir(hydra.utils.get_original_cwd())

    # shorter cfg aliases
    cfg_rl = cfg.rl
    cfg_sim = cfg.env.jsbsim

    # Load agent
    agent = TDMPC2(cfg.rl)
    assert os.path.exists(cfg.rl.checkpoint), f"Checkpoint {cfg.rl.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.rl.checkpoint)

    # env setup
    env = make_env(cfg)
    state_names = list(prp.get_legal_name() for prp in env.state_prps)

    ep_rewards = [0]
    enu_xs = [env.unwrapped.sim['position/enu-x-m']]
    enu_ys = [env.unwrapped.sim['position/enu-y-m']]
    enu_zs = [env.unwrapped.sim['position/enu-z-m']]
    step = 0
    target = np.array([0, 300, 600])
    total_steps = 2000 

    while step < total_steps:
        env.set_target_state(target)
        # action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
        action = agent.act(obs, t0=t==0, eval_mode=True)
        obs, reward, term, trunc, info = env.step(action)
        ep_obss.append(obs)
        ep_rewards.append(reward)
        enu_xs.append(env.unwrapped.sim['position/enu-x-m'])
        enu_ys.append(env.unwrapped.sim['position/enu-y-m'])
        enu_zs.append(env.unwrapped.sim['position/enu-z-m'])
        done = np.logical_or(term, trunc)
        t += 1
        step += 1

        if done:
            t = 0
            print(f"Episode reward: {info['episode']['r']}")
            print(f"******* {step}/{total_steps} *******")

    env.close()

    ep_obss = np.array(ep_obss)
    ep_rewards = np.array(ep_rewards)
    enu_xs = np.array(enu_xs)
    enu_ys = np.array(enu_ys)
    enu_zs = np.array(enu_zs)
    errs_x, errs_y, errs_z = ep_obss[:, 0], ep_obss[:, 1], ep_obss[:, 2]
    dist_to_target = np.sqrt(errs_x**2 + errs_y**2 + errs_z**2)
    tsteps = np.linspace(0, ep_obss.shape[0], ep_obss.shape[0])
    fig, ax = plt.subplots(2, 3)

    ax[0, 0].plot(tsteps, np.rad2deg(ep_obss[:, 3]), label='roll')
    ax[0, 0].plot(tsteps, np.rad2deg(ep_obss[:, 4]), label='pitch')
    ax[0, 0].set_title('Roll and Pitch [deg]')
    ax[0, 0].legend()

    ax[0, 1].plot(tsteps, ep_obss[:, 5])
    ax[0, 1].set_title('Airspeed [kph]')

    ax[0, 2].remove()
    ax[0, 2] = fig.add_subplot(1, 3, 3, projection='3d')
    ax[0,2].set_xlabel("X")
    ax[0,2].set_ylabel("Y")
    ax[0,2].set_zlabel("Z")
    for i in range(ep_obss.shape[0] - 1):
        ax[0,2].plot(enu_xs[i:i+2], enu_ys[i:i+2], enu_zs[i:i+2], c=plt.cm.plasma(i/enu_xs.shape[0]))

    ax[1, 0].plot(tsteps, dist_to_target)
    ax[1, 0].set_title('Distance to Target [m]')

    ax[1, 1].plot(tsteps, ep_rewards)
    ax[1, 1].set_title('Rewards')

    ax[1, 2].plot(tsteps, ep_obss[:, 11], label='aileron')
    ax[1, 2].plot(tsteps, ep_obss[:, 12], label='elevator')
    ax[1, 2].plot(tsteps, ep_obss[:, 13], label='throttle')
    ax[1, 2].set_title('Control Inputs')
    ax[1, 2].legend()

    print(f"Last position: X:{enu_xs[-1]}, Y:{enu_ys[-1]}, Z:{enu_zs[-1]}")

    plt.show()

if __name__ == '__main__':
    eval()