"""Check local-source PPO checkpoint compatibility without writing experiment results."""

from pathlib import Path
import random

import gymnasium as gym
from hydra import compose, initialize_config_dir
import numpy as np
import torch

import fw_flightcontrol
from fw_flightcontrol.agents.ppo import Agent_PPO
import fw_jsbgym


def main() -> None:
    package = Path(__file__).resolve().parents[1] / "fw_flightcontrol"
    expected_simulator = package.parents[1] / "FW-JSBGym" / "fw_jsbgym"
    if Path(fw_flightcontrol.__file__).resolve().parent != package:
        raise RuntimeError("fw_flightcontrol resolves to a different checkout")
    if Path(fw_jsbgym.__file__).resolve().parent != expected_simulator:
        raise RuntimeError("fw_jsbgym resolves to a different checkout")

    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    with initialize_config_dir(version_base=None, config_dir=str(package / "config")):
        cfg = compose(config_name="default", overrides=["env.jsbsim.debug=false"])

    checkpoint_path = (
        package / "models/icinco/ppo/noactreg"
        / "ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    env = gym.make(
        cfg.rl.PPO.env_id, cfg_env=cfg.env, telemetry_file=None, render_mode="none"
    )
    try:
        env.unwrapped.init()
        model = Agent_PPO(env, cfg)
        model.load_state_dict(checkpoint["agent"], strict=True)
        model.eval()
        options = cfg.env.jsbsim.eval_sim_options
        options.seed = 10
        options.atmosphere.severity = "off"
        observation, _ = env.reset(options=options)
        print(f"Observation shape: {observation.shape}; actions: {env.action_space.shape}")
        for step in range(10):
            env.unwrapped.set_target_state(np.deg2rad(np.array([25.0, 15.0])))
            with torch.no_grad():
                action = model.get_action_and_value(
                    torch.as_tensor(observation).unsqueeze(0)
                )[1].squeeze(0).numpy()
            observation, reward, terminated, truncated, _ = env.step(
                np.clip(action, env.action_space.low, env.action_space.high)
            )
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise RuntimeError(f"Non-finite observation or reward at step {step}")
            if terminated or truncated:
                raise RuntimeError(f"Unexpected episode end at step {step}")
        print("PASS: strict seed-1 PPO checkpoint load and ten finite simulator steps")
        print("This is not a paper-score reproduction.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
