"""Train seed-1 ICINCO PPO (no action regularization), then evaluate easy/no-wind.

Outputs stay in a new directory. No archived weights/configs are overwritten.
The historical 750,000-step request yields 61 complete rollouts: 749,568 steps.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_ppo_nominal import (
    git_output, paper_surface_fluctuation, read_metrics, sha256, simulator_source_files,
)


def training_overrides(device: str, smoke: bool = False) -> list[str]:
    overrides = [
        "rl.PPO.seed=1", "rl.PPO.exp_name=icinco2024_noactreg",
        "rl.PPO.track=false", f"rl.PPO.cuda={str(device == 'cuda').lower()}",
        "rl.PPO.total_timesteps=750000", "rl.PPO.num_envs=6", "rl.PPO.num_steps=2048",
        "rl.PPO.input_arch=mlp", "rl.PPO.env_id=ACBohnNoVaIErr-v0",
        "rl.PPO.ts_coef=0", "rl.PPO.ss_coef=0", "rl.PPO.pa_coef=0",
        "rl.PPO.save_cp=false", "rl.PPO.final_traj_plot=false", "rl.PPO.final_eval=false",
        "rl.PPO.periodic_eval=true", "rl.PPO.ref_sampler=uniform",
        "roll_limit=60", "pitch_limit=30",
        "env.jsbsim.debug=false", "+env.jsbsim.sim_profile=icinco2024",
        "env.jsbsim.eval_sim_options.atmosphere.severity=off",
        "hydra.run.dir=hydra", "hydra.job.chdir=false",
    ]
    if smoke:
        overrides += [
            "rl.PPO.total_timesteps=4096", "rl.PPO.num_envs=2", "rl.PPO.num_steps=1024",
            "rl.PPO.periodic_eval=false",
        ]
    return overrides


def archive_ranges(repo: Path) -> dict[str, dict[str, float]]:
    directory = repo / "fw_flightcontrol/eval/attitude_control/outputs/icinco/ppo/noactreg_noatmo"
    rows = [
        read_metrics(directory / f"ppo_bohnmlp_noactreg_nohist_noatmo_easyrefs_{seed}.csv")
        for seed in range(1, 6)
    ]
    values = {
        "avg_rmse": [row["avg_rmse"] for row in rows],
        "paper_surface_fluct": [paper_surface_fluctuation(row) for row in rows],
    }
    return {name: {"min": min(numbers), "max": max(numbers)} for name, numbers in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory; must not exist")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="4096 training steps, not a paper gate")
    args = parser.parse_args()
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu explicitly")
    repo = Path(__file__).resolve().parents[1]
    simulator = repo.parent / "FW-JSBGym"
    output = args.output_dir.resolve()
    ranges = archive_ranges(repo)
    sources = simulator_source_files(simulator)
    names = git_output(
        repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", "fw_flightcontrol", "reproducibility",
    ).split("\0")
    sources += [
        repo / name for name in sorted(set(names))
        if name.endswith((".py", ".yaml", ".sha256"))
    ]
    archives = repo / "fw_flightcontrol/eval/attitude_control/outputs/icinco/ppo/noactreg_noatmo"
    sources += sorted(archives.glob("*easyrefs_[1-5].csv"))
    hashes = {str(path): sha256(path) for path in sources}
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(simulator), str(repo))),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "WANDB_MODE": "disabled", "MPLBACKEND": "Agg",
    }
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    command = [
        sys.executable, str(repo / "fw_flightcontrol/train/ppo_train.py"),
        *training_overrides(args.device, args.smoke),
    ]
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "packages": {name: version(name) for name in (
            "torch", "numpy", "gymnasium", "jsbsim", "hydra-core", "omegaconf",
        )},
        "sources": {root.name: {
            "commit": git_output(root, "rev-parse", "HEAD"),
            "status": git_output(root, "status", "--short"),
            "diff": git_output(root, "diff", "HEAD"),
        } for root in (repo, simulator)},
        "input_sha256": hashes, "command": command, "process_environment": environment,
        "device": args.device, "smoke": args.smoke,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "requested_steps": 4096 if args.smoke else 750000,
        "rollout_batch_size": 2048 if args.smoke else 12288,
        "minibatch_size": 64 if args.smoke else 384,
        "comparison_rule": "Each mean metric inside the archived five-seed min/max; not a confidence interval",
        "archive_ranges": ranges,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Training seed 1; log: {output / 'training.log'}", flush=True)
    with (output / "training.log").open("x") as log:
        subprocess.run(
            command, cwd=output, env={**os.environ, **environment},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10800,
        )
    log_text = (output / "training.log").read_text()
    num_envs = 2 if args.smoke else 6
    if log_text.count("Simulator profile: icinco2024") != num_envs:
        raise RuntimeError("Not every training environment confirmed the historical profile")
    if f"Using Device: {args.device}" not in log_text:
        raise RuntimeError(f"Trainer did not use requested device {args.device}")
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("Inputs changed during training; the run is not a stable baseline")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    checkpoint = output / "models/train/ppo/1/ppo_icinco2024_noactreg_1.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["seed"] != 1 or not all(torch.isfinite(value).all() for value in payload["agent"].values()):
        raise RuntimeError("Invalid training checkpoint")
    events = EventAccumulator(str(output / "runs/ppo_icinco2024_noactreg_1")).Reload()
    completed_steps = events.Scalars("charts/SPS")[-1].step
    expected_steps = 4096 if args.smoke else 749568
    if completed_steps != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} optimizer steps' worth of samples, got {completed_steps}")
    result = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": completed_steps, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "smoke": args.smoke,
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.smoke:
        print(f"Training smoke passed: {completed_steps} transitions and a finite checkpoint.")
        return
    subprocess.run([
        sys.executable, str(repo / "reproducibility/evaluate_ppo_nominal.py"),
        "--checkpoint", str(checkpoint), "--sim-profile", "icinco2024",
        "--device", args.device, str(output / "evaluation"),
    ], check=True, env={**os.environ, **environment}, timeout=660)
    metrics = read_metrics(output / "evaluation/eval/outputs/ppo_seed1_easy.csv")
    metrics["paper_surface_fluct"] = paper_surface_fluctuation(metrics)
    comparison = {
        name: {**bounds, "fresh": metrics[name], "inside_archived_range":
               bounds["min"] <= metrics[name] <= bounds["max"]}
        for name, bounds in ranges.items()
    }
    (output / "fresh_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    if not all(row["inside_archived_range"] for row in comparison.values()):
        raise RuntimeError(f"Fresh training is outside the pre-recorded archive range: {comparison}")
    print("Fresh seed-1 metrics are inside the archived five-seed range (easy/no-wind only).")


if __name__ == "__main__":
    main()
