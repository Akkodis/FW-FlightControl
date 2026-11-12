"""Hydra-configured attitude-control evaluation entry point."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from . import metrics, protocol, rollout, scheduler
from .worker import input_files, runtime_info, source_info, write_csv


def run(cfg):
    if cfg.evaluation.checkpoint_directory is not None:
        cfg = deepcopy(cfg)
    protocol.prepare_checkpoint_directory(cfg)
    protocol.validate_config(cfg)
    algorithm = cfg.algorithm.name
    runtime = runtime_info(algorithm)
    actual_jsbsim = runtime["packages"]["jsbsim"]["imported_version"]
    expected_jsbsim = cfg.evaluation.runtime.jsbsim_version
    if expected_jsbsim and actual_jsbsim != expected_jsbsim:
        raise RuntimeError(
            f"Configured JSBSim version {expected_jsbsim} is required; found {actual_jsbsim}"
        )

    import torch

    runtime["cuda"] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }
    output = protocol.resolve_path(cfg.evaluation.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sources, simulator = source_info()
    files = input_files(cfg, simulator)
    hashes = {str(path): protocol.sha256(path) for path in files}
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    provenance = {
        "schema": "fw-rl-attitude-evaluation-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "runtime": runtime,
        "sources": sources,
        "input_sha256": hashes,
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    if len(cfg.evaluation.checkpoints) > 1 and cfg.evaluation.visualization.mode == "none":
        rows, episodes, stopped, remaining_jobs = scheduler.run_parallel(cfg, output)
    else:
        rows, episodes, stopped, remaining_jobs = rollout.run_evaluation(cfg, output)
    changed = [path for path, digest in hashes.items()
               if protocol.sha256(path) != digest]
    if changed:
        raise RuntimeError(f"Evaluation inputs changed during execution: {changed}")
    write_csv(output / "metrics.csv", rows)
    aggregates = metrics.aggregate(rows, cfg.evaluation.aggregation.target_groups)
    write_csv(output / "aggregates.csv", aggregates)
    (output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
    status = "stopped" if stopped else "completed"
    out_of_bounds_episodes = sum(episode["out_of_bounds"] for episode in episodes)
    result = {
        "status": status,
        "out_of_bounds_error": out_of_bounds_episodes > 0,
        "episodes_out_of_bounds": out_of_bounds_episodes,
        "metrics": rows,
        "aggregates": aggregates,
        "episode_count": len(episodes),
        "remaining_jobs": remaining_jobs,
        "inputs_verified_after_run": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if out_of_bounds_episodes:
        print(
            f"ERROR: Evaluation {status} with {out_of_bounds_episodes} "
            f"out-of-bounds episode(s). See {output / 'result.json'} for details.",
            file=sys.stderr, flush=True,
        )
    return result


@hydra.main(version_base=None, config_path="../../config", config_name="evaluate")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
