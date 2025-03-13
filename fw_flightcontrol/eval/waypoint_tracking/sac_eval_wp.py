import random
import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from fw_flightcontrol.agents.sac_norm import Actor_SAC
from fw_flightcontrol.utils.train_utils import make_env
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.utils import conversions
from fw_jsbgym.utils import jsbsim_properties as prp


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
    env = make_env('WaypointVaTracking-v0', cfg.env, cfg_sim.render_mode, 
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
    enu_xs = [env.unwrapped.sim['position/enu-x-m']]
    enu_ys = [env.unwrapped.sim['position/enu-y-m']]
    enu_zs = [env.unwrapped.sim['position/enu-z-m']]
    step = 0
    target_enu = np.array([0, 300, 600])
    target = conversions.enu2ecef(*target_enu,
                                  env.unwrapped.sim['ic/lat-geod-deg'],
                                  env.unwrapped.sim['ic/long-gc-deg'],
                                  0.0)
    target = np.hstack((target, np.array([60.0])))
    total_steps = 2000

    while step < total_steps:
        env.set_target_state(target)
        # action = trim_action
        action = sac_agent.get_action(torch.Tensor(obs).unsqueeze(0).to(device))[2].squeeze_().detach().cpu().numpy()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_obss.append(obs)
        ep_rewards.append(reward)
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        enu_xs.append(env.unwrapped.sim['position/enu-x-m'])
        enu_ys.append(env.unwrapped.sim['position/enu-y-m'])
        enu_zs.append(env.unwrapped.sim['position/enu-z-m'])
        done = np.logical_or(terminated, truncated)

        if done:
            if info['out_of_bounds']:
                print("Out of bounds")
                break

            print(f"Episode reward: {info['episode']['r']}")
            print(f"******* {step}/{total_steps} *******")
            obs, last_info = env.reset()
        step += 1
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
    ax[0, 2] = fig.add_subplot(2, 3, 3, projection='3d')
    ax[0, 2].set_xlim(enu_xs.min()-10, enu_xs.max()+10)
    ax[0, 2].set_ylim(enu_ys.min()-10, enu_ys.max()+10)
    ax[0, 2].set_zlim(enu_zs.min()-10, enu_zs.max()+10)
    ax[0, 2].set_xlabel("X")
    ax[0, 2].set_ylabel("Y")
    ax[0, 2].set_zlabel("Z")
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


if __name__ == '__main__':
    eval()