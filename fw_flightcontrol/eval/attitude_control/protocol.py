"""Configuration and input helpers for attitude-control evaluations."""

import hashlib
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


PACKAGE = Path(__file__).resolve().parents[2]
SEVERITIES = ("off", "light", "moderate", "severe")
ALGORITHMS = ("pid", "ppo", "sac", "tdmpc2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path, *, follow_symlinks=True):
    candidate = Path(path).expanduser()

    def absolute(value):
        return value.resolve() if follow_symlinks else value.parent.resolve() / value.name

    if candidate.is_absolute():
        return absolute(candidate)
    from_cwd = Path.cwd() / candidate
    if from_cwd.exists():
        return absolute(from_cwd)
    return absolute(PACKAGE / candidate)


def target_name(entry):
    return Path(entry.path).stem


def prepare_checkpoint_directory(cfg):
    directory = cfg.evaluation.checkpoint_directory
    if directory is None:
        return
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("evaluation.checkpoint_directory must be a nonempty path or null")
    if cfg.algorithm.name == "pid":
        raise ValueError("PID evaluations do not accept a checkpoint directory")

    entries = list(cfg.evaluation.checkpoints)
    if entries and (
        len(entries) != 1
        or entries[0].name != "default"
        or entries[0].path is not None
        or entries[0].get("training_seed") is not None
    ):
        raise ValueError(
            "Choose either evaluation.checkpoint_directory or explicit checkpoint entries"
        )

    path = resolve_path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {path}")
    files = sorted(
        (file for file in path.iterdir()
         if file.is_file() and file.suffix.lower() in {".pt", ".pth"}),
        key=lambda file: file.name,
    )
    if not files:
        raise ValueError(f"No .pt or .pth checkpoints found in directory: {path}")
    cfg.evaluation.checkpoints = [
        {"name": file.name, "path": str(file.resolve()), "training_seed": None}
        for file in files
    ]


def validate_config(cfg):
    algorithm = cfg.algorithm.name
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    evaluation = cfg.evaluation
    if not evaluation.targets:
        raise ValueError("Configure at least one target array in evaluation.targets")
    if any(
        not isinstance(target, DictConfig)
        or set(target.keys()) != {"path"}
        or not isinstance(target.path, str)
        or not target.path.strip()
        for target in evaluation.targets
    ):
        raise ValueError("Every target entry must contain only a nonempty path")
    target_names = [target_name(target) for target in evaluation.targets]
    if len(target_names) != len(set(target_names)):
        raise ValueError("Target filenames without extensions must be unique")
    if any(not _safe_label(name) for name in target_names):
        raise ValueError("Target names must be nonempty path-safe labels")
    if not evaluation.atmo_severities:
        raise ValueError("Configure at least one atmosphere severity")
    if any(severity not in SEVERITIES for severity in evaluation.atmo_severities):
        raise ValueError(f"Atmosphere severities must be selected from {SEVERITIES}")
    if len(evaluation.atmo_severities) != len(set(evaluation.atmo_severities)):
        raise ValueError("Atmosphere severities must be unique")

    policy_seeds = list(evaluation.policy_seeds)
    if not policy_seeds or any(type(seed) is not int or seed < 0 for seed in policy_seeds):
        raise ValueError("policy_seeds must contain nonnegative integers")
    if len(policy_seeds) != len(set(policy_seeds)):
        raise ValueError("policy_seeds must be unique")

    checkpoints = list(evaluation.checkpoints)
    if not checkpoints:
        raise ValueError("Configure at least one checkpoint entry")
    if any(not entry.name for entry in checkpoints):
        raise ValueError("Every checkpoint entry requires a name")
    if algorithm != "pid":
        missing = [entry.name for entry in checkpoints if not entry.path]
        if missing:
            raise ValueError(
                f"{algorithm} requires a checkpoint path for every entry; missing: {missing}"
            )
        invalid_seeds = [
            entry.name for entry in checkpoints
            if entry.get("training_seed") is not None
            and (type(entry.training_seed) is not int or entry.training_seed < 0)
        ]
        if invalid_seeds:
            raise ValueError(f"Checkpoint training seeds must be nonnegative integers: {invalid_seeds}")
    elif len(checkpoints) != 1 or any(entry.path for entry in checkpoints):
        raise ValueError("PID evaluations do not accept checkpoint files")
    checkpoint_names = [entry.name for entry in checkpoints]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("Checkpoint names must be unique")
    if any(not _safe_label(name) for name in checkpoint_names):
        raise ValueError("Checkpoint names must be nonempty path-safe labels")

    start = evaluation.episodes.start
    count = evaluation.episodes.count
    if type(start) is not int or start < 0:
        raise ValueError("episodes.start must be a nonnegative integer")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("episodes.count must be a positive integer or null")
    if evaluation.simulator_seeds.mode not in {"generated", "values", "file"}:
        raise ValueError("simulator_seeds.mode must be generated, values, or file")
    if evaluation.visualization.mode not in {"none", "matplotlib", "flightgear", "both"}:
        raise ValueError("Unsupported visualization mode")
    if evaluation.visualization.interactive and evaluation.visualization.mode == "none":
        raise ValueError("Interactive visualization requires a non-none visualization mode")
    execution = evaluation.execution
    if type(execution.workers_per_gpu) is not int or not 1 <= execution.workers_per_gpu <= 4:
        raise ValueError("execution.workers_per_gpu must be an integer between 1 and 4")
    if type(execution.cpu_workers) is not int or execution.cpu_workers < 1:
        raise ValueError("execution.cpu_workers must be a positive integer")
    group_names = []
    for group in evaluation.aggregation.target_groups:
        if not group.name or not _safe_label(group.name):
            raise ValueError("Target aggregation groups require a path-safe name")
        names = list(group.targets)
        if not names or len(names) != len(set(names)) or any(
            name not in target_names for name in names
        ):
            raise ValueError(f"Target aggregation {group.name} references an unknown target")
        if group.method != "mean_of_means" or group.sem != "mean_of_sems":
            raise ValueError(f"Unsupported aggregation method in target group {group.name}")
        group_names.append(group.name)
    if len(group_names) != len(set(group_names)) or set(group_names) & set(target_names):
        raise ValueError("Target aggregation group names must be unique and distinct from targets")


def _safe_label(name):
    return (
        isinstance(name, str)
        and bool(name)
        and name not in {".", ".."}
        and "/" not in name
        and "\\" not in name
    )


def validate_target_array(path):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError(f"Attitude target array must have shape (episodes, 2): {path}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Attitude target array must be numeric: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Target array contains non-finite values: {path}")
    return values
