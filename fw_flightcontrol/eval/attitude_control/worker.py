"""Runtime, provenance, and output helpers for attitude-control evaluation."""

from importlib.metadata import PackageNotFoundError, version
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import hydra

from . import protocol


def runtime_info(algorithm):
    modules = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("jsbsim", "jsbsim"),
        ("gymnasium", "gymnasium"),
        ("hydra-core", "hydra"),
        ("omegaconf", "omegaconf"),
    ]
    if algorithm == "tdmpc2":
        modules.extend((("torchrl", "torchrl"), ("tensordict", "tensordict")))
    packages = {}
    for distribution, module_name in modules:
        module = importlib.import_module(module_name)
        try:
            distribution_version = version(distribution)
        except PackageNotFoundError:
            distribution_version = None
        packages[distribution] = {
            "imported_version": getattr(module, "__version__", None),
            "distribution_version": distribution_version,
            "file": module.__file__,
        }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "environment": {
            name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "MPLBACKEND", "PYTHONPATH", "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def source_info():
    import fw_jsbgym

    simulator = Path(fw_jsbgym.__file__).resolve().parent
    result = {}
    for root in (protocol.PACKAGE, simulator):
        repository = root.parent
        if (repository / ".git").exists():
            result[str(repository)] = {
                "commit": subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
                ).strip(),
                "status": subprocess.check_output(
                    ["git", "-C", str(repository), "status", "--short"], text=True,
                ).strip(),
            }
    return result, simulator


def input_files(cfg, simulator):
    files = [
        *sorted((protocol.PACKAGE / "eval/attitude_control").glob("*.py")),
        *sorted((protocol.PACKAGE / "config").rglob("*.yaml")),
    ]
    files.extend(path for path in simulator.rglob("*")
                 if path.is_file() and path.suffix in {".py", ".xml", ".yaml"})
    for target in cfg.evaluation.targets:
        files.append(protocol.resolve_path(target.path))
    seed_config = cfg.evaluation.simulator_seeds
    if seed_config.mode == "file":
        files.append(protocol.resolve_path(seed_config.file))
    if cfg.algorithm.name != "pid":
        for checkpoint in cfg.evaluation.checkpoints:
            files.append(protocol.resolve_path(checkpoint.path))
    jsbsim_path = Path(importlib.import_module("jsbsim").__file__).resolve()
    files.append(jsbsim_path)
    files.extend(sorted(jsbsim_path.parent.glob("_jsbsim*.so")))
    unique = []
    seen = set()
    for path in files:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation input files do not exist: {missing}")
    return unique


def write_csv(path, rows):
    import csv

    if not rows:
        raise ValueError(f"Cannot write an empty result file: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg):
    from .rollout import run_evaluation

    protocol.validate_config(cfg)
    if len(cfg.evaluation.checkpoints) != 1:
        raise ValueError("A checkpoint worker requires exactly one checkpoint")
    rows, episodes, stopped, remaining_jobs = run_evaluation(
        cfg, protocol.resolve_path(cfg.evaluation.output),
    )
    result = {
        "rows": rows,
        "episodes": episodes,
        "stopped": stopped,
        "remaining_jobs": remaining_jobs,
    }
    result_path = Path(os.environ["FW_ATTITUDE_WORKER_RESULT"])
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
