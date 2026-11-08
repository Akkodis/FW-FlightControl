"""Fresh current TD-MPC2 seed 1 on ICINCO, followed by nominal easy evaluation.

The legacy trainer's inclusive step loop collects 750,001 transitions for a
750,000-step request. Metric comparisons are descriptive, not acceptance gates.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluate_ppo_nominal import (
    RUNTIME_PACKAGES, evaluation_log_problems, git_output, paper_surface_fluctuation,
    read_metrics, runtime_package_info, sha256, simulator_source_files,
)


def training_overrides(smoke: bool = False) -> list[str]:
    overrides = [
        "rl=attitude_control/ac_tdmpc2", "rl.seed=1", "rl.model_size=5",
        "rl.steps=750000", "rl.exp_name=icinco2024_noactreg",
        "rl.use_caps=false", "rl.ts_coef=0", "rl.compile=false",
        "rl.save_agent=true", "rl.disable_wandb=true", "rl.final_traj_plot=false",
        "rl.periodic_eval=true", "rl.eval_freq=37500",
        "env/task/reward=ac_my_bohn", "env.jsbsim.debug=false",
        "env.jsbsim.sim_profile=icinco2024",
        "env.jsbsim.eval_sim_options.atmosphere.severity=off",
        "hydra.run.dir=hydra", "hydra.job.chdir=false",
    ]
    if smoke:
        overrides += ["rl.steps=2010", "rl.seed_steps=2000", "rl.periodic_eval=false"]
    return overrides


def archive_metrics(repo: Path) -> list[dict[str, float]]:
    directory = repo / "fw_flightcontrol/eval/attitude_control/outputs/icinco/tdmpc2/noactreg_noatmo"
    return [
        read_metrics(directory / f"bohn_noactreg_noatmo_easyrefs_{seed}.csv")
        for seed in range(1, 6)
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_checkpoint(path: Path) -> int:
    import torch

    weights = torch.load(path, map_location="cpu", weights_only=True)["model"]
    tensor_count = 0
    for name, value in weights.items():
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite checkpoint tensor: {name}")
            tensor_count += 1
        elif name.endswith(".__batch_size") and isinstance(value, torch.Size):
            if value != torch.Size([5]):
                raise RuntimeError(f"Unexpected critic ensemble size: {name}={value}")
        elif name.endswith(".__device") and (value is None or isinstance(value, torch.device)):
            continue
        else:
            raise RuntimeError(f"Unexpected checkpoint metadata: {name}")
    if not tensor_count:
        raise RuntimeError("Checkpoint contains no model tensors")
    return tensor_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory; must not exist")
    parser.add_argument("--smoke", action="store_true", help="2011 transitions; not a paper-score check")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    repo = Path(__file__).resolve().parents[1]
    simulator = repo.parent / "FW-JSBGym"
    package = repo / "fw_flightcontrol"
    submodule = package / "agents/tdmpc2"
    output = args.output_dir.resolve()
    archive = archive_metrics(repo)
    sources = simulator_source_files(simulator)
    for root in (package, repo / "reproducibility"):
        names = git_output(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
        sources.extend(root / name for name in sorted(set(names)) if name.endswith((".py", ".yaml", ".sha256")))
    sources.extend(submodule.rglob("*.py"))
    sources.extend((package / "eval/attitude_control/targets").glob("*.npy"))
    archive_dir = package / "eval/attitude_control/outputs/icinco/tdmpc2/noactreg_noatmo"
    sources.extend(archive_dir.glob("*easyrefs_[1-5].csv"))
    hashes = {str(path): sha256(path) for path in sorted(set(sources))}
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join((str(simulator), str(repo))),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "WANDB_MODE": "disabled", "MPLBACKEND": "Agg", "HYDRA_FULL_ERROR": "1",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    command = [
        sys.executable, str(package / "train/tdmpc2_train.py"),
        *training_overrides(args.smoke),
    ]
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        **runtime_package_info(
            environment, {**RUNTIME_PACKAGES, "torchrl": "torchrl", "tensordict": "tensordict"},
            entrypoint=Path(command[1]),
        ),
        "sources": {str(root): {
            "commit": git_output(root, "rev-parse", "HEAD"),
            "status": git_output(root, "status", "--short"),
            "diff": git_output(root, "diff", "HEAD"),
        } for root in (repo, simulator, submodule)},
        "input_sha256": hashes, "command": command, "process_environment": environment,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "smoke": args.smoke, "requested_steps": 2010 if args.smoke else 750000,
        "expected_transitions": 2011 if args.smoke else 750001,
        "comparison_rule": "Descriptive seed-1 and five-seed archive comparisons; no new tolerance or author acceptance",
        "archive_metrics": archive,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", provenance)
    print(f"Training seed 1; log: {output / 'training.log'}", flush=True)
    with (output / "training.log").open("x") as log:
        subprocess.run(
            command, cwd=output, env={**os.environ, **environment},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=86400,
        )
    log_text = (output / "training.log").read_text()
    if "Simulator profile: icinco2024" not in log_text or "Pretraining agent on seed data..." not in log_text:
        raise RuntimeError("Training did not confirm the profile and optimizer warmup")
    if "Training completed successfully" not in log_text or "Failed to save model" in log_text:
        raise RuntimeError("Training did not finish and save successfully")
    # The console logs completed episodes; the inclusive loop takes one more step.
    steps = re.findall(r"\bI:\s+([\d,]+)", log_text)
    expected_last_episode = 2000 if args.smoke else 750000
    if not steps or int(steps[-1].replace(",", "")) != expected_last_episode:
        raise RuntimeError(f"Unexpected final completed-episode step: {steps[-1:]}")
    checkpoint = output / "models/train/tdmpc2/1/tdmpc2_icinco2024_noactreg_1.pt"
    tensor_count = validate_checkpoint(checkpoint)
    changed = [path for path, digest in hashes.items() if sha256(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"Inputs changed during training: {changed}")
    write_json(output / "training_result.json", {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "last_logged_episode_step": expected_last_episode,
        "completed_transitions": provenance["expected_transitions"],
        "finite_model_tensors": tensor_count,
        "smoke": args.smoke,
    })
    if args.smoke:
        print("CUDA optimizer/checkpoint smoke passed; no paper-score claim.", flush=True)
        return

    evaluation = output / "evaluation"
    targets = evaluation / "eval/attitude_control/targets"
    targets.parent.mkdir(parents=True)
    targets.symlink_to(package / "eval/attitude_control/targets", target_is_directory=True)
    eval_command = [
        sys.executable, str(package / "eval/attitude_control/tdmpc2_eval_simple.py"),
        "rl=attitude_control/ac_tdmpc2", "rl.seed=10", "rl.model_size=5",
        f"rl.checkpoint={checkpoint}", "rl.ref_file=simple_easy", "rl.res_file=metrics",
        "rl.disable_wandb=true", "env/task/reward=ac_my_bohn",
        "env.jsbsim.debug=false", "env.jsbsim.sim_profile=icinco2024",
        "env.jsbsim.eval_sim_options.atmosphere.severity=off",
        "hydra.run.dir=hydra", "hydra.job.chdir=false",
    ]
    write_json(evaluation / "provenance.json", {
        "command": eval_command, "checkpoint_sha256": sha256(checkpoint),
        "training_provenance": str(output / "provenance.json"),
        "evaluation_seed": 10, "episodes": 25, "steps": 50000, "device": "cuda",
    })
    print(f"Evaluating fresh checkpoint; log: {evaluation / 'evaluation.log'}", flush=True)
    with (evaluation / "evaluation.log").open("x") as log:
        subprocess.run(
            eval_command, cwd=evaluation, env={**os.environ, **environment},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200,
        )
    eval_log = (evaluation / "evaluation.log").read_text()
    problems = evaluation_log_problems(eval_log, "icinco2024")
    if "Using Device: cuda" not in eval_log:
        problems.append("evaluation did not confirm CUDA")
    if problems:
        raise RuntimeError("; ".join(problems))
    changed = [path for path, digest in hashes.items() if sha256(Path(path)) != digest]
    if changed:
        raise RuntimeError(f"Inputs changed during evaluation: {changed}")
    fresh = read_metrics(evaluation / "eval/outputs/metrics.csv")
    comparison = {}
    for metric in ("avg_rmse", "avg_fcs_fluct"):
        values = [row[metric] for row in archive]
        comparison[metric] = {
            "fresh": fresh[metric], "archived_seed1": values[0],
            "change_percent": 100 * (fresh[metric] / values[0] - 1),
            "five_seed_min": min(values), "five_seed_max": max(values),
            "inside_archived_range": min(values) <= fresh[metric] <= max(values),
        }
    if abs(fresh["avg_fcs_fluct"] - paper_surface_fluctuation(fresh)) > 1e-12:
        raise RuntimeError("Fluctuation average does not match the two control surfaces")
    write_json(output / "fresh_comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
