"""Run the unchanged ICINCO PPO evaluator on seed 1 / easy references / no wind.

Outputs are isolated, CPU inference is selected, and archived results are only read.
A completed evaluation is not a successful paper-reproduction gate.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


METRICS = (
    "roll_rmse", "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct",
    "avg_rmse", "avg_fcs_fluct",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def read_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("severity") != "off":
        raise ValueError(f"Expected exactly one nominal-atmosphere row in {path}")
    if set(rows[0]) != {"severity", *METRICS}:
        raise ValueError(f"Unexpected metric schema in {path}")
    metrics = {key: float(rows[0][key]) for key in METRICS}
    if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
        raise ValueError(f"Non-finite or negative metrics in {path}")
    return metrics


def paper_surface_fluctuation(metrics: dict[str, float]) -> float:
    return (metrics["roll_fcs_fluct"] + metrics["pitch_fcs_fluct"]) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory; must not exist")
    parser.add_argument(
        "--simulator-root", type=Path,
        help="Separate FW-JSBGym Git worktree for controlled diagnostic comparisons",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    simulator = (args.simulator_root or repo.parent / "FW-JSBGym").resolve()
    if Path(git_output(simulator, "rev-parse", "--show-toplevel")).resolve() != simulator:
        raise ValueError(f"Simulator path must be a Git worktree root: {simulator}")
    if not (simulator / "fw_jsbgym/__init__.py").is_file():
        raise ValueError(f"Simulator package is missing from {simulator}")
    simulator_files = [
        simulator / name
        for name in git_output(simulator, "ls-files", "-z", "--", "fw_jsbgym").split("\0")
        if name
    ]
    simulator_hashes = {
        str(path.relative_to(simulator)): sha256(path) for path in simulator_files
    }
    package = repo / "fw_flightcontrol"
    evaluation = package / "eval/attitude_control"
    checkpoint = (
        package / "models/icinco/ppo/noactreg"
        / "ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt"
    )
    archive = (
        evaluation / "outputs/icinco/ppo/noactreg_noatmo"
        / "ppo_bohnmlp_noactreg_nohist_noatmo_easyrefs_1.csv"
    )
    historical = read_metrics(archive)
    inputs = [
        Path(__file__).resolve(),
        repo / "reproducibility/historical_inputs.sha256",
        checkpoint, archive, evaluation / "ppo_eval_simple.py",
        evaluation / "targets/simple_easy.npy",
        evaluation / "targets/jsbsim_seeds.npy",
        package / "config/default.yaml",
        package / "config/rl/ppo.yaml",
        package / "config/env/jsbsim/noatmo.yaml",
        package / "config/env/task/mdp/no_hist_all.yaml",
        package / "config/env/task/reward/ac_my_bohn.yaml",
    ]
    input_hashes = {str(path.relative_to(repo.parent)): sha256(path) for path in inputs}
    child_env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(simulator), str(repo))),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable, str(evaluation / "ppo_eval_simple.py"),
        f"model_path={checkpoint}", "ref_file=simple_easy", "res_file=ppo_seed1_easy",
        "env.jsbsim.debug=false",
        "env.jsbsim.eval_sim_options.atmosphere.severity=off",
        "hydra.run.dir=hydra", "hydra.job.chdir=false",
    ]
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("torch", "numpy", "gymnasium", "jsbsim", "hydra-core", "omegaconf")
        },
        "sources": {
            source.name: {
                "path": str(source),
                "commit": git_output(source, "rev-parse", "HEAD"),
                "status": git_output(source, "status", "--short"),
            }
            for source in (repo, simulator)
        },
        "input_sha256": input_hashes,
        "simulator_sha256": simulator_hashes,
        "simulator_worktree_diff": git_output(simulator, "diff", "HEAD", "--", "fw_jsbgym"),
        "command": command,
        "process_environment": child_env,
        "protocol": {
            "checkpoint_seed": 1, "runtime_seed": 10, "device": "cpu",
            "steps": 50000, "episodes": 25, "steps_per_episode": 2000,
            "agent_hz": 100, "fdm_hz": 100,
            "reference_subset": "simple_easy", "atmosphere": "off",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    targets = output / "eval/attitude_control/targets"
    targets.parent.mkdir(parents=True)
    targets.symlink_to(evaluation / "targets", target_is_directory=True)
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")

    log_path = output / "evaluation.log"
    print(f"Running legacy evaluator; log: {log_path}", flush=True)
    with log_path.open("x") as log:
        subprocess.run(
            command, cwd=output, env={**os.environ, **child_env},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
        )
    log_text = log_path.read_text()
    if (
        log_text.count("Episode reward:") != 25
        or "Out of bounds" in log_text
        or "e_obs shape: (50000, 14)" not in log_text
        or "eps_fcs_fluct shape: (25, 3)" not in log_text
    ):
        raise RuntimeError(f"Nominal episode/observation checks failed; inspect {log_path}")
    if any(sha256(path) != input_hashes[str(path.relative_to(repo.parent))] for path in inputs):
        raise RuntimeError("An input changed during evaluation; results are not a stable baseline")
    if any(sha256(path) != simulator_hashes[str(path.relative_to(simulator))] for path in simulator_files):
        raise RuntimeError("Simulator sources changed during evaluation; results are not stable")

    current = read_metrics(output / "eval/outputs/ppo_seed1_easy.csv")
    historical["paper_surface_fluct"] = paper_surface_fluctuation(historical)
    current["paper_surface_fluct"] = paper_surface_fluctuation(current)
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["metric", "archived", "current", "delta", "delta_percent"]
        )
        writer.writeheader()
        for metric in (*METRICS, "paper_surface_fluct"):
            previous, value = historical[metric], current[metric]
            writer.writerow({
                "metric": metric, "archived": previous, "current": value,
                "delta": value - previous,
                "delta_percent": 100 * (value / previous - 1) if previous else None,
            })
    print(f"Completed 25 episodes / 50,000 steps. Comparison: {output / 'comparison.csv'}")
    print(f"Average RMSE: {current['avg_rmse']:.8f}; archive: {historical['avg_rmse']:.8f}")
    print(
        f"Paper two-surface fluctuation: {current['paper_surface_fluct']:.8f}; "
        f"legacy aggregate (includes throttle): {current['avg_fcs_fluct']:.8f}"
    )
    print("No paper-reproduction tolerance has been established; inspect the differences.")


if __name__ == "__main__":
    main()
