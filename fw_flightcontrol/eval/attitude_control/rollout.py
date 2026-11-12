"""Generic attitude-control rollouts for PID, PPO, SAC and TD-MPC2."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .metrics import compute_metrics
from .protocol import PACKAGE, resolve_path, target_name, validate_target_array


RENDER_MODES = {
    "none": "none",
    "matplotlib": "plot_anim",
    "flightgear": "fgear",
    "both": "fgear_plot",
}


def seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_path(entry):
    return resolve_path(entry.path)


def simulator_seed_values(config, needed):
    seeds = config.simulator_seeds
    if seeds.mode == "generated":
        base = seeds.base_seed
        if type(base) is not int or base < 1:
            raise ValueError("simulator_seeds.base_seed must be a positive integer")
        values = np.arange(base, base + needed, dtype=np.int64)
    elif seeds.mode == "values":
        values = np.asarray(list(seeds.seeds))
    else:
        if not seeds.file:
            raise ValueError("simulator_seeds.file is required when mode is file")
        seed_path = resolve_path(seeds.file)
        if not seed_path.is_file():
            raise FileNotFoundError(f"Simulator seed file does not exist: {seed_path}")
        values = np.load(seed_path, allow_pickle=False)
    if values.ndim != 1 or len(values) < needed:
        found = len(values) if values.ndim == 1 else values.shape
        raise ValueError(f"Need {needed} simulator seeds, found {found}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Simulator seeds must be numeric")
    if not np.isfinite(values).all() or np.any(values < 1):
        raise ValueError("Simulator seeds must be finite positive numbers")
    if np.any(values != np.floor(values)):
        raise ValueError("Simulator seeds must be integers")
    return values.astype(np.int64)


def selected_targets(cfg):
    selected = []
    for entry in cfg.evaluation.targets:
        path = target_path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"Target array does not exist: {path}")
        selected.append((target_name(entry), path, validate_target_array(path)))
    return selected


def selected_simulator_seeds(cfg, target_count):
    start = cfg.evaluation.episodes.start
    count = cfg.evaluation.episodes.count
    end = target_count if count is None else start + count
    if start >= target_count or end > target_count:
        raise ValueError(
            f"Episode selection [{start}, {end}) is outside target array of length {target_count}"
        )
    episodes = end - start
    needed = episodes + int(bool(cfg.evaluation.episodes.include_final_reset))
    return start, end, simulator_seed_values(cfg.evaluation, needed)


def reset_options(cfg, severity, seed):
    cfg.env.jsbsim.eval_sim_options.seed = float(seed)
    cfg.env.jsbsim.eval_sim_options.atmosphere.severity = severity
    return cfg.env.jsbsim.eval_sim_options


class Controller:
    def __init__(self, cfg, telemetry, policy_seed):
        import gymnasium as gym
        import fw_jsbgym  # noqa: F401 -- registers environment IDs

        self.cfg = cfg
        self.algorithm = cfg.algorithm.name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.deterministic = True
        print(f"**** Using Device: {self.device} ****", flush=True)
        sim = cfg.env.jsbsim
        sim.render_mode = RENDER_MODES[cfg.evaluation.visualization.mode]
        if cfg.evaluation.visualization.interactive:
            sim.plot_at_episode_end = False

        if self.algorithm == "tdmpc2":
            submodule = str(PACKAGE / "agents/tdmpc2/tdmpc2")
            if submodule not in sys.path:
                sys.path.insert(0, submodule)
            from fw_flightcontrol.agents.tdmpc2.tdmpc2.common.parser import parse_cfg
            from fw_flightcontrol.agents.tdmpc2.tdmpc2.envs import make_env

            self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            self.cfg.rl.seed = policy_seed
            self.cfg.rl = parse_cfg(self.cfg.rl)
            self.env = make_env(self.cfg)
            self.state_names = [prop.get_legal_name() for prop in self.env.state_prps]
            self.env.unwrapped.telemetry_file = str(telemetry)
        else:
            if self.algorithm == "pid":
                env_id = cfg.PID.env_id
            else:
                env_id = cfg.rl[self.algorithm.upper()].env_id
            self.env = gym.make(
                env_id, cfg_env=cfg.env, render_mode=sim.render_mode,
                telemetry_file=str(telemetry),
            )
            self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
            if self.algorithm != "pid":
                self.env = gym.wrappers.ClipAction(self.env)

    def load_agent(self, checkpoint):
        if self.algorithm == "tdmpc2":
            from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2

            self.agent = TDMPC2(self.cfg.rl)
            self.agent.load(str(checkpoint))
            return

        self.env.init()
        if self.algorithm == "pid":
            from fw_flightcontrol.agents.pid import PID
            from fw_jsbgym.models.aerodynamics import AeroModel
            from fw_jsbgym.trim.trim_point import TrimPoint

            if self.cfg.env.task.mdp.obs_is_matrix:
                raise ValueError("PID requires vector observations")
            trim = TrimPoint("x8")
            aircraft = AeroModel()
            gains = self.cfg.evaluation.pid
            timestep = gains.timestep
            if timestep is None:
                timestep = 1.0 / self.cfg.env.jsbsim.agent_freq
            self.roll_pid = PID(
                kp=gains.kp[0], ki=gains.ki[0], kd=gains.kd[0],
                dt=timestep, trim=trim, limit=aircraft.aileron_limit,
            )
            self.pitch_pid = PID(
                kp=gains.kp[1], ki=gains.ki[1], kd=gains.kd[1],
                dt=timestep, trim=trim, limit=aircraft.elevator_limit,
            )
            return

        if self.algorithm == "ppo":
            from fw_flightcontrol.agents.ppo import Agent_PPO

            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.agent = Agent_PPO(self.env, self.cfg).to(self.device)
            self.agent.load_state_dict(state["agent"])
        elif self.algorithm == "sac":
            from fw_flightcontrol.agents.sac import Actor_SAC

            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.agent = Actor_SAC(self.env).to(self.device)
            self.agent.load_state_dict(state[0])
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        self.agent.eval()

    def start_condition(self, observation):
        if self.algorithm == "pid":
            self.pid_state = observation
        elif self.algorithm == "tdmpc2":
            self.z_obs = self.agent.model.encode(torch.as_tensor(observation).to(self.device), None)

    def reset_policy(self, observation):
        if self.algorithm == "pid":
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.pid_state = observation

    def action(self, observation, target, timestep):
        if self.algorithm == "pid":
            self.roll_pid.set_reference(float(target[0]))
            self.pitch_pid.set_reference(float(target[1]))
            roll, pitch, _, p, q = self.pid_state[:5]
            aileron = self.roll_pid.update(roll, p, saturate=True, normalize=True)[0]
            elevator = self.pitch_pid.update(pitch, q, saturate=True, normalize=True)[0]
            return np.array([aileron, elevator])
        if self.algorithm == "tdmpc2":
            return self.agent.act(observation, t0=timestep == 0, eval_mode=True)
        tensor = torch.as_tensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.algorithm == "ppo":
                return self.agent.get_action_and_value(tensor)[1].squeeze(0).cpu().numpy()
            return self.agent.get_action(tensor)[2].squeeze(0).cpu().numpy()

    def after_step(self, observation, action):
        if self.algorithm == "pid":
            self.pid_state = observation
            return
        if self.algorithm != "tdmpc2":
            return
        if self.cfg.rl.im_traj:
            self.z_obs = self.agent.model.next(
                self.z_obs, torch.as_tensor(action).to(self.device), None,
            )
            decoded = self.agent.model.decode(self.z_obs, None).detach().cpu().numpy()
            self.env.telemetry_logging(dict(zip(
                ("im_dec_" + name for name in self.state_names), decoded,
            )))
        if self.cfg.rl.dec_obs:
            encoded = self.agent.model.encode(torch.as_tensor(observation).to(self.device), None)
            decoded = self.agent.model.decode(encoded, None).detach().cpu().numpy()
            self.env.telemetry_logging(dict(zip(
                ("dec_" + name for name in self.state_names), decoded,
            )))


def _observation_for_metrics(algorithm, cfg, observation, info):
    raw = observation if algorithm == "pid" else info["non_norm_obs"]
    if algorithm in {"ppo", "sac"} and cfg.env.task.mdp.obs_is_matrix:
        raw = raw[0, -1]
    return np.asarray(raw)


def _interactive_continue(message):
    print(message, flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer not in {"q", "quit"}


def run_condition(
    cfg, controller, target_name, targets, severity, policy_seed, checkpoint, telemetry,
):
    start, end, seeds = selected_simulator_seeds(cfg, len(targets))
    sim = cfg.env.jsbsim
    sim.eval_sim_options.atmosphere.severity = severity
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    controller.env.unwrapped.telemetry_file = str(telemetry)
    observation, _ = controller.env.reset(
        options=reset_options(cfg, severity, seeds[0]),
    )
    controller.start_condition(observation)
    episode_observations = []
    episode_fluctuations = []
    episode_rows = []
    max_steps = int(sim.episode_length_s * sim.agent_freq)
    completed = 0
    stopped = False

    for target_index in range(start, end):
        if target_index > start:
            controller.reset_policy(observation)
        target = targets[target_index]
        observations = []
        terminated = truncated = False
        step_info = {}
        for timestep in range(max_steps):
            controller.env.set_target_state(target)
            action = controller.action(observation, target, timestep)
            observation, _, terminated, truncated, step_info = controller.env.step(action)
            controller.after_step(observation, action)
            observations.append(_observation_for_metrics(
                controller.algorithm, cfg, observation, step_info,
            ))
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise RuntimeError(
                f"{target_name} episode {target_index} exceeded {max_steps} steps without ending"
            )

        out_of_bounds = bool(step_info["out_of_bounds"])
        if out_of_bounds:
            print(
                f"ERROR: Out of bounds: target={target_name}, severity={severity}, "
                f"episode={target_index}, policy_seed={policy_seed}, "
                f"simulator_seed={int(seeds[target_index - start])}; continuing evaluation.",
                file=sys.stderr, flush=True,
            )
        episode_observations.extend(observations)

        should_reset = (
            target_index + 1 < end
            or bool(cfg.evaluation.episodes.include_final_reset)
        )
        if should_reset:
            next_seed_index = target_index - start + 1
            observation, reset_info = controller.env.reset(options=reset_options(
                cfg, severity, seeds[next_seed_index],
            ))
            positions = reset_info.get("fcs_pos_hist")
        else:
            positions = step_info.get("fcs_pos_hist")
        if positions is None:
            raise RuntimeError(
                f"Missing control-surface history for {target_name} episode {target_index}"
            )
        fluctuation_unavailable_reason = None
        if len(positions) < 2:
            fluctuation_unavailable_reason = "fewer than two control-surface samples"
        else:
            episode_fluctuations.append(
                np.mean(np.abs(np.diff(np.asarray(positions), axis=0)), axis=0)
            )
        reward = float(np.asarray(step_info["episode"]["r"]).item())
        print(f"Episode reward: {reward}", flush=True)
        episode_rows.append({
            "algorithm": controller.algorithm,
            "checkpoint": checkpoint.name if checkpoint is not None else "pid",
            "training_seed": checkpoint.get("training_seed") if checkpoint is not None else None,
            "target": target_name,
            "severity": severity,
            "policy_seed": policy_seed,
            "episode": target_index,
            "simulator_seed": int(seeds[target_index - start]),
            "reward": reward,
            "steps": len(observations),
            "duration_s": len(observations) / float(sim.agent_freq),
            "termination_reason": (
                "out_of_bounds" if out_of_bounds
                else "terminated" if terminated
                else "truncated"
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "early_termination": len(observations) < max_steps,
            "out_of_bounds": out_of_bounds,
            "error": "OUT_OF_BOUNDS" if out_of_bounds else None,
            "fluctuation_unavailable_reason": fluctuation_unavailable_reason,
        })
        completed += 1
        if cfg.evaluation.visualization.interactive:
            if not _interactive_continue(
                f"Completed {target_name} episode {target_index}; "
                "press Enter to continue or q to stop this evaluation."
            ):
                stopped = True
                break

    if not episode_observations:
        raise RuntimeError(f"No episodes completed for {target_name} at severity {severity}")
    print(f"e_obs shape: {np.asarray(episode_observations).shape}", flush=True)
    print(f"eps_fcs_fluct shape: {np.asarray(episode_fluctuations).shape}", flush=True)
    metrics = compute_metrics(
        np.asarray(episode_observations),
        np.asarray(episode_fluctuations),
    )
    result = {
        "algorithm": controller.algorithm,
        "checkpoint": checkpoint.name if checkpoint is not None else "pid",
        "training_seed": checkpoint.get("training_seed") if checkpoint is not None else None,
        "target": target_name,
        "severity": severity,
        "policy_seed": policy_seed,
        "episodes": completed,
        "episodes_attempted": len(episode_rows),
        "episodes_remaining": (end - start) - completed,
        "episodes_out_of_bounds": sum(row["out_of_bounds"] for row in episode_rows),
        "out_of_bounds_error": any(row["out_of_bounds"] for row in episode_rows),
        "episodes_early_termination": sum(row["early_termination"] for row in episode_rows),
        "episodes_fluctuation_included": len(episode_fluctuations),
        "episodes_fluctuation_unavailable": completed - len(episode_fluctuations),
        "steps_completed": sum(row["steps"] for row in episode_rows),
        **metrics,
    }
    return result, episode_rows, stopped


def run_evaluation(cfg, output):
    algorithm = cfg.algorithm.name
    targets = selected_targets(cfg)
    severities = list(cfg.evaluation.atmo_severities)
    rows = []
    episode_rows = []
    stopped = False
    output = Path(output)
    for checkpoint_entry in cfg.evaluation.checkpoints:
        print(f"===== Evaluating checkpoint: {checkpoint_entry.name!r} =====", flush=True)
        checkpoint = None
        if algorithm != "pid":
            checkpoint = resolve_path(checkpoint_entry.path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        for policy_seed in cfg.evaluation.policy_seeds:
            if algorithm != "tdmpc2":
                seed_policy(policy_seed)
            telemetry = output / "telemetry" / str(checkpoint_entry.name)
            telemetry.mkdir(parents=True, exist_ok=True)
            controller = Controller(cfg, telemetry / "telemetry.csv", policy_seed)
            try:
                controller.load_agent(checkpoint)
                for target_name, _, target_values in targets:
                    start, end, _ = selected_simulator_seeds(cfg, len(target_values))
                    for severity in severities:
                        telemetry_file = (
                            telemetry / target_name / f"policy_{policy_seed}_{severity}.csv"
                        )
                        result, episodes, user_stopped = run_condition(
                            cfg, controller, target_name, target_values, severity,
                            policy_seed, checkpoint_entry if checkpoint else None,
                            telemetry_file,
                        )
                        rows.append(result)
                        episode_rows.extend(episodes)
                        if user_stopped or len(episodes) < end - start:
                            stopped = True
                            break
                    if stopped:
                        break
            finally:
                controller.env.close()
            if stopped:
                break
        if stopped:
            break
    planned_jobs = (
        len(cfg.evaluation.checkpoints)
        * len(cfg.evaluation.policy_seeds)
        * len(targets)
        * len(severities)
    )
    remaining_jobs = planned_jobs - len(rows)
    return rows, episode_rows, stopped, remaining_jobs
