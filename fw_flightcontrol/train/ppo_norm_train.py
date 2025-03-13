# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from fw_jsbgym.trim.trim_point import TrimPoint
from fw_jsbgym.models.aerodynamics import AeroModel
from fw_flightcontrol.agents.ppo_norm import Agent_PPO
from fw_flightcontrol.agents.pid import torchPID
from fw_flightcontrol.utils import train_utils
import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="../config", config_name="default")
def train(cfg: DictConfig):
    cfg.rl.PPO.batch_size = int(cfg.rl.PPO.num_envs * cfg.rl.PPO.num_steps)
    cfg.rl.PPO.minibatch_size = int(cfg.rl.PPO.batch_size // cfg.rl.PPO.num_minibatches)
    cfg.rl.PPO.total_timesteps = int(cfg.rl.PPO.total_timesteps)


    if OmegaConf.is_missing(cfg.rl.PPO, "seed"):
        cfg.rl.PPO.seed = random.randint(0, 9999)
        print(f"Seed not provided, using random seed: {cfg.rl.PPO.seed}")
    else:
        print(f"Seed provided, using seed from config: {cfg.rl.PPO.seed}")

    # shorter cfg aliases
    cfg_ppo = cfg.rl.PPO
    cfg_sim = cfg.env.jsbsim
    cfg_mdp = cfg.env.task.mdp

    np.set_printoptions(precision=3)

    run_name = f"ppo_{cfg_ppo.exp_name}_{cfg_ppo.seed}"

    save_path: str = "models/train/ppo"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if cfg_ppo.track:
        wandb.init(
            project=cfg_ppo.wandb_project_name,
            entity=cfg_ppo.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=["PPO"]
        )
        wandb.define_metric("global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg_ppo).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg_ppo.seed)
    np.random.seed(cfg_ppo.seed)
    torch.manual_seed(cfg_ppo.seed)
    torch.backends.cudnn.deterministic = cfg_ppo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg_ppo.cuda else "cpu")
    print(f"**** Using Device: {device} ****")

    # env setup
    print(f"ENV ID: {cfg_ppo.env_id}")
    envs = gym.vector.SyncVectorEnv(
        [train_utils.make_env(cfg_ppo.env_id, cfg.env, cfg_sim.render_mode, None, eval=False, gamma=cfg_ppo.gamma) for i in range(cfg_ppo.num_envs)]
    )
    unwr_envs = [envs.envs[i].unwrapped for i in range(cfg_ppo.num_envs)]
    print("Single Env Observation Space Shape = ", envs.single_observation_space.shape)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent_PPO(envs, cfg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg_ppo.learning_rate, eps=1e-5, weight_decay=cfg_ppo.weight_decay)
    trim_point: TrimPoint = TrimPoint(aircraft_id='x8')
    if "NoVa" in cfg_ppo.env_id or "Vanilla" in cfg_ppo.env_id:
        trim_acts = torch.tensor([trim_point.aileron, trim_point.elevator]).to(device)
    else:
        trim_acts = torch.tensor([trim_point.aileron, trim_point.elevator, trim_point.throttle]).to(device)


    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs) + envs.single_observation_space.shape).to(device)
    obs_t1 = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs)).to(device)
    rewards = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs)).to(device)
    terminateds = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs)).to(device)
    truncateds = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs)).to(device)
    values = torch.zeros((cfg_ppo.num_steps, cfg_ppo.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    prev_gl_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(options=cfg_sim.train_sim_options)
    next_obs = torch.Tensor(next_obs).to(device)
    next_terminated = torch.zeros(cfg_ppo.num_envs).to(device)
    num_updates = cfg_ppo.total_timesteps // cfg_ppo.batch_size

    targets = train_utils.sample_targets(False, cfg_ppo.env_id, cfg_ppo)

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if cfg_ppo.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg_ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # save checkpoints periodically
        if cfg_ppo.save_cp and update % 8 == 0:
            run_name = f"ppo_{cfg_ppo.exp_name}_cp{global_step}_{cfg_ppo.seed}"
            train_utils.save_model_PPO(save_path, run_name, agent, envs.envs[0], cfg_ppo.seed)

        # run periodic evaluation
        prev_div, _ = divmod(prev_gl_step, cfg_ppo.eval_freq)
        curr_div, _ = divmod(global_step, cfg_ppo.eval_freq)
        print(f"prev_gl_step = {prev_gl_step}, global_step = {global_step}, prev_div = {prev_div}, curr_div = {curr_div}")
        if cfg_ppo.periodic_eval and (prev_div != curr_div or global_step == 0):
            eval_dict = train_utils.periodic_eval(cfg_ppo.env_id, cfg_mdp, cfg_sim, envs.envs[0], agent, device)
            for k, v in eval_dict.items():
                writer.add_scalar("eval/" + k, v, global_step)

        prev_gl_step = global_step

        # at half the updates, change the beta params
        if cfg_ppo.ref_sampler == "beta" and global_step > 7.5e5:
            print("change beta params, make the targets harder")
            cfg_ppo.beta_params = [0.1, 0.1]

        for step in range(0, cfg_ppo.num_steps):
            if cfg_ppo.track:
                wandb.log({"global_step": global_step})
            global_step += 1 * cfg_ppo.num_envs
            obs[step] = next_obs
            terminateds[step] = next_terminated

            # send reference (target) points
            for i in range(cfg_ppo.num_envs):
                ith_env_step = unwr_envs[i].sim[unwr_envs[i].current_step]
                unwr_envs[i].set_target_state(targets[i])

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_mean, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            truncateds[step] = torch.Tensor(truncated).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_terminated = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)

            dones = np.logical_or(terminated, truncated)

            # At every env reset, sample and send a new reference
            for env_i, done in enumerate(dones):
                if done:
                    obs_t1[step][env_i] = obs[step][env_i]
                    targets[env_i] = train_utils.sample_targets(True, cfg_ppo.env_id, cfg_ppo)
                    # if cfg_ppo.ref_sampler == "beta":
                    #     if cfg_ppo.beta_params is not None:
                    #         roll_targets[env_i] = np.random.beta(cfg_ppo.beta_params[0], cfg_ppo.beta_params[1]) * roll_limit*2 - roll_limit
                    #         pitch_targets[env_i] = np.random.beta(cfg_ppo.beta_params[0], cfg_ppo.beta_params[1]) * pitch_limit*2 - pitch_limit
                    #         print(f"Sampled from beta with params {cfg_ppo.beta_params}")

                    # print(f"Env {env_i} done, new targets : "\
                    #       f"roll = {np.rad2deg(targets[env_i, 0]):.3f}, "\
                    #       f"pitch = {np.rad2deg(targets[env_i, 1]):.3f}")
                    if 'AC' in cfg_ppo.env_id:
                        print(f"Env {env_i} done, new targets : "\
                            f"roll = {np.rad2deg(targets[env_i, 0]):.3f}, "\
                            f"pitch = {np.rad2deg(targets[env_i, 1]):.3f}")
                    elif 'Waypoint' in cfg_ppo.env_id:
                        print(f"Env {env_i} done, new targets : "\
                              f"x = {targets[env_i, 0]:.3f}, "\
                              f"y = {targets[env_i, 1]:.3f}, "\
                              f"z = {targets[env_i, 2]:.3f}")

                else:
                    obs_t1[step][env_i] = next_obs[env_i]

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']} \n" + \
                      f"episode_end={info['episode_end']}, out_of_bounds={info['out_of_bounds']}\n********")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                r_per_step = info["episode"]["r"]/info["episode"]["l"]
                # writer.add_scalar("charts/reward_per_step", r_per_step, global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg_ppo.num_steps)):
                if t == cfg_ppo.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminateds[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg_ppo.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg_ppo.gamma * cfg_ppo.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.transpose(0,1).reshape((-1,) + envs.single_observation_space.shape)
        b_obs_t1 = obs_t1.transpose(0, 1).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.transpose(0, 1).reshape(-1)
        b_actions = actions.transpose(0, 1).reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.transpose(0, 1).reshape(-1)
        b_returns = returns.transpose(0, 1).reshape(-1)
        b_values = values.transpose(0, 1).reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg_ppo.batch_size)
        clipfracs = []
        for epoch in range(cfg_ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg_ppo.batch_size, cfg_ppo.minibatch_size):
                end = start + cfg_ppo.minibatch_size
                mb_inds = b_inds[start:end]

                _, __, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg_ppo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg_ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg_ppo.clip_coef, 1 + cfg_ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg_ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg_ppo.clip_coef,
                        cfg_ppo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # CAPS losses
                # Temporal Smoothing
                act_means = agent.get_action_and_value(b_obs[mb_inds])[1]
                next_act_means = agent.get_action_and_value(b_obs_t1[mb_inds])[1]
                ts_loss = torch.Tensor([0.0]).to(device)
                b_cmd = torch.zeros((cfg_ppo.minibatch_size, 2)).to(device)
                if cfg_ppo.env_id not in ["ACNoVaPIDRLAdd-v0", "ACNoVaPIDRL-v0"]:
                    ts_loss = F.mse_loss(act_means, next_act_means)
                else:
                    # get all the relevant variables for computing the PID output given the PIDRL action at time t
                    roll_gains = act_means[:, 0:3]
                    roll_err = b_obs[mb_inds][:, 0, 4, 6].reshape(-1, 1)
                    roll_int_err = b_obs[mb_inds][:, 0, 4, 8].reshape(-1, 1)
                    roll_p = b_obs[mb_inds][:, 0, 4, 3].reshape(-1, 1)
                    roll_errs = torch.cat((roll_err, roll_int_err, -roll_p), dim=1)
                    b_roll_cmd = torchPID(roll_gains, roll_errs, AeroModel().aileron_limit, saturate=True, normalize=True)
                    # roll_pid_terms = roll_gains * roll_errs
                    # b_roll_cmd = roll_pid_terms.sum(dim=1).reshape(-1, 1) # batch of aileron (roll) commands

                    pitch_gains = act_means[:, 3:6]
                    pitch_err = b_obs[mb_inds][:, 0, 4, 7].reshape(-1, 1)
                    pitch_int_err = b_obs[mb_inds][:, 0, 4, 9].reshape(-1, 1)
                    pitch_q = b_obs[mb_inds][:, 0, 4, 4].reshape(-1, 1)
                    pitch_errs = torch.cat((pitch_err, pitch_int_err, -pitch_q), dim=1)
                    b_pitch_cmd = torchPID(pitch_gains, pitch_errs, AeroModel().elevator_limit, saturate=True, normalize=True)
                    # pitch_pid_terms = pitch_gains * pitch_errs
                    # b_pitch_cmd = pitch_pid_terms.sum(dim=1).reshape(-1, 1) # batch of elevator (pitch) commands

                    roll_gains_t1 = next_act_means[:, 0:3]
                    roll_err_t1 = b_obs_t1[mb_inds][:, 0, 4, 6].reshape(-1, 1)
                    roll_int_err_t1 = b_obs_t1[mb_inds][:, 0, 4, 8].reshape(-1, 1)
                    roll_p_t1 = b_obs_t1[mb_inds][:, 0, 4, 3].reshape(-1, 1)
                    roll_errs_t1 = torch.cat((roll_err_t1, roll_int_err_t1, -roll_p_t1), dim=1)
                    b_roll_cmd_t1 = torchPID(roll_gains_t1, roll_errs_t1, AeroModel().aileron_limit, saturate=True, normalize=True)
                    # roll_pid_terms_t1 = roll_gains_t1 * roll_errs_t1
                    # b_roll_cmd_t1 = roll_pid_terms_t1.sum(dim=1).reshape(-1, 1)

                    pitch_gains_t1 = next_act_means[:, 3:6]
                    pitch_err_t1 = b_obs_t1[mb_inds][:, 0, 4, 7].reshape(-1, 1)
                    pitch_int_err_t1 = b_obs_t1[mb_inds][:, 0, 4, 9].reshape(-1, 1)
                    pitch_q_t1 = b_obs_t1[mb_inds][:, 0, 4, 4].reshape(-1, 1)
                    pitch_errs_t1 = torch.cat((pitch_err_t1, pitch_int_err_t1, -pitch_q_t1), dim=1)
                    b_pitch_cmd_t1 = torchPID(pitch_gains_t1, pitch_errs_t1, AeroModel().elevator_limit, saturate=True, normalize=True)
                    # pitch_pid_terms_t1 = pitch_gains_t1 * pitch_errs_t1
                    # b_pitch_cmd_t1 = pitch_pid_terms_t1.sum(dim=1).reshape(-1, 1)

                    b_cmd = torch.cat((b_roll_cmd, b_pitch_cmd), dim=1)
                    b_cmd_t1 = torch.cat((b_roll_cmd_t1, b_pitch_cmd_t1), dim=1)
                    ts_loss = torch.linalg.norm(b_cmd - b_cmd_t1, ord=2)


                # Spatial Smoothing
                state_problaw = Normal(b_obs[mb_inds], 0.01)
                state_sampled = state_problaw.sample()
                act_means_bar = agent.get_action_and_value(state_sampled)[1]
                ss_loss = torch.Tensor([0.0]).to(device)
                if cfg_ppo.env_id not in ["ACNoVaPIDRLAdd-v0", "ACNoVaPIDRL-v0"]:
                    ss_loss = torch.linalg.norm(act_means - act_means_bar, ord=2)
                else:
                    roll_gains_bar = act_means_bar[:, 0:3]
                    roll_err_bar = state_sampled[:, 0, 4, 6].reshape(-1, 1)
                    roll_int_err_bar = state_sampled[:, 0, 4, 8].reshape(-1, 1)
                    roll_p_bar = state_sampled[:, 0, 4, 3].reshape(-1, 1)
                    roll_errs_bar = torch.cat((roll_err_bar, roll_int_err_bar, -roll_p_bar), dim=1)
                    b_roll_cmd_bar = torchPID(roll_gains_bar, roll_errs_bar, AeroModel().aileron_limit, saturate=True, normalize=True)

                    pitch_gains_bar = act_means_bar[:, 3:6]
                    pitch_err_bar = state_sampled[:, 0, 4, 7].reshape(-1, 1)
                    pitch_int_err_bar = state_sampled[:, 0, 4, 9].reshape(-1, 1)
                    pitch_q_bar = state_sampled[:, 0, 4, 4].reshape(-1, 1)
                    pitch_errs_bar = torch.cat((pitch_err_bar, pitch_int_err_bar, -pitch_q_bar), dim=1)
                    b_pitch_cmd_bar = torchPID(pitch_gains_bar, pitch_errs_bar, AeroModel().elevator_limit, saturate=True, normalize=True)

                    b_cmd_bar = torch.cat((b_roll_cmd_bar, b_pitch_cmd_bar), dim=1)
                    ss_loss = torch.linalg.norm(b_cmd - b_cmd_bar, ord=2)

                # preactivation loss
                pa_loss = torch.Tensor([0.0]).to(device)
                if cfg_ppo.env_id not in ["ACNoVaPIDRLAdd-v0", "ACNoVaPIDRL-v0", "WaypointTracking-v0", "AltitudeTracking-v0"]:
                    pa_loss = torch.linalg.norm(act_means - trim_acts, ord=2)
                elif cfg_ppo.env_id == "AltitudeTracking-v0":
                    pa_loss = torch.linalg.norm(act_means - trim_acts[-2:], ord=2)

                loss = pg_loss - cfg_ppo.ent_coef * entropy_loss + v_loss * cfg_ppo.vf_coef \
                      + cfg_ppo.ts_coef * ts_loss + cfg_ppo.ss_coef * ss_loss + cfg_ppo.pa_coef * pa_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg_ppo.max_grad_norm)
                optimizer.step()

            if cfg_ppo.target_kl is not None:
                if approx_kl > cfg_ppo.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        action_std = agent.get_action_and_value(b_obs)[5]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/value_loss_term", cfg_ppo.vf_coef * v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/ts", ts_loss.item(), global_step)
        writer.add_scalar("losses/ts_term", cfg_ppo.ts_coef * ts_loss.item(), global_step)
        writer.add_scalar("losses/ss", ss_loss.item(), global_step)
        writer.add_scalar("losses/ss_term", cfg_ppo.ss_coef * ss_loss.item(), global_step)
        writer.add_scalar("losses/pa", ss_loss.item(), global_step)
        writer.add_scalar("losses/pa_term", cfg_ppo.pa_coef * pa_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/entropy_term", cfg_ppo.ent_coef * entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        if 'Altitude' not in cfg_ppo.env_id:
            writer.add_scalar("action_std/da", action_std[0].mean(), global_step)
            writer.add_scalar("action_std/de", action_std[1].mean(), global_step)
        else:
            writer.add_scalar("action_std/de", action_std[0].mean(), global_step)
            writer.add_scalar("action_std/dt", action_std[1].mean(), global_step)
        if cfg_ppo.env_id == "ACBohn-v0":
            writer.add_scalar("action_std/dt", action_std[2].mean(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    # Evaluate the agent once with a single traj to be plotted in wandb
    if cfg_ppo.final_traj_plot:
        train_utils.final_traj_plot(envs.envs[0], cfg_ppo.env_id, cfg_sim, agent, device, run_name)

    train_utils.save_model_PPO(save_path, run_name, agent, envs.envs[0], cfg_ppo.seed)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()