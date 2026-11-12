"""Run independent checkpoint evaluations in isolated worker processes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event, Lock

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import torch

from . import metrics, protocol, rollout


def worker_slots(cfg, count):
    if not torch.cuda.is_available():
        return [None] * min(count, cfg.evaluation.execution.cpu_workers)

    device_count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        devices = [str(index) for index in range(device_count)]
    else:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if len(devices) < device_count:
            raise RuntimeError("Cannot map visible CUDA devices to worker processes")
        devices = devices[:device_count]
    if not devices:
        raise RuntimeError("CUDA is available but no visible GPU was found")

    slots = []
    for _ in range(cfg.evaluation.execution.workers_per_gpu):
        for device in devices:
            slots.append(device)
            if len(slots) == count:
                return slots
    return slots


def validate_worker_result(cfg, checkpoint, result):
    if not isinstance(result, dict):
        raise ValueError(f"Invalid worker result for checkpoint {checkpoint.name!r}")
    rows = result.get("rows")
    episodes = result.get("episodes")
    if (
        not isinstance(rows, list)
        or not isinstance(episodes, list)
        or result.get("stopped") is not False
        or result.get("remaining_jobs") != 0
    ):
        raise ValueError(f"Incomplete worker result for checkpoint {checkpoint.name!r}")

    expected_conditions = set()
    expected_episodes = set()
    for target_name, _, targets in rollout.selected_targets(cfg):
        start, end, seeds = rollout.selected_simulator_seeds(cfg, len(targets))
        for policy_seed in cfg.evaluation.policy_seeds:
            for severity in cfg.evaluation.atmo_severities:
                condition = (target_name, severity, policy_seed)
                expected_conditions.add(condition)
                for episode in range(start, end):
                    expected_episodes.add((*condition, episode, int(seeds[episode - start])))
    try:
        actual_conditions = [
            (row["target"], row["severity"], row["policy_seed"]) for row in rows
        ]
        actual_episodes = [
            (row["target"], row["severity"], row["policy_seed"],
             row["episode"], row["simulator_seed"])
            for row in episodes
        ]
        wrong_checkpoint = any(
            row["checkpoint"] != checkpoint.name for row in (*rows, *episodes)
        )
        episode_fields = {
            "algorithm", "checkpoint", "training_seed", "target", "severity",
            "policy_seed", "episode", "simulator_seed", "reward", "steps",
            "duration_s", "termination_reason", "terminated", "truncated",
            "early_termination", "out_of_bounds", "error", "fluctuation_unavailable_reason",
        }
        row_fields = {
            "algorithm", "checkpoint", "training_seed", "target", "severity",
            "policy_seed", "episodes", "episodes_attempted", "episodes_remaining",
            "episodes_out_of_bounds", "out_of_bounds_error", "episodes_early_termination",
            "episodes_fluctuation_included", "episodes_fluctuation_unavailable",
            "steps_completed", *metrics.METRICS,
        }
        if any(not episode_fields <= row.keys() for row in episodes) or any(
            not row_fields <= row.keys() for row in rows
        ):
            raise ValueError("Missing result fields")
        if any(
            row["algorithm"] != cfg.algorithm.name
            or row["training_seed"] != checkpoint.get("training_seed")
            for row in (*rows, *episodes)
        ):
            raise ValueError("Incorrect algorithm or training seed")
        for row in rows:
            condition_episodes = [
                episode for episode in episodes
                if (episode["target"], episode["severity"], episode["policy_seed"])
                == (row["target"], row["severity"], row["policy_seed"])
            ]
            count = len(condition_episodes)
            if (
                any(type(episode["out_of_bounds"]) is not bool for episode in condition_episodes)
                or row["episodes"] != count
                or row["episodes_attempted"] != count
                or row["episodes_remaining"] != 0
                or row["episodes_out_of_bounds"] != sum(
                    episode["out_of_bounds"] for episode in condition_episodes
                )
                or row["out_of_bounds_error"] != (row["episodes_out_of_bounds"] > 0)
            ):
                raise ValueError("Inconsistent episode counts")
        metrics.aggregate(rows)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed worker result for checkpoint {checkpoint.name!r}") from error
    if (
        wrong_checkpoint
        or len(actual_conditions) != len(expected_conditions)
        or set(actual_conditions) != expected_conditions
        or len(actual_episodes) != len(expected_episodes)
        or set(actual_episodes) != expected_episodes
    ):
        raise ValueError(f"Unexpected worker results for checkpoint {checkpoint.name!r}")
    return rows, episodes


def run_worker(cfg, output, checkpoint, device, log_lock, original_cwd, active, active_lock, stop):
    job_dir = output / "jobs" / checkpoint.name
    job_dir.mkdir(parents=True, exist_ok=False)
    job_cfg = deepcopy(cfg)
    job_cfg.evaluation.checkpoint_directory = None
    job_checkpoint = deepcopy(checkpoint)
    job_checkpoint.path = str(protocol.resolve_path(checkpoint.path))
    job_cfg.evaluation.checkpoints = [job_checkpoint]
    job_cfg.evaluation.output = str(output)
    for target in job_cfg.evaluation.targets:
        target.path = str(protocol.resolve_path(target.path, follow_symlinks=False))
    if job_cfg.evaluation.simulator_seeds.mode == "file":
        seeds = job_cfg.evaluation.simulator_seeds
        seeds.file = str(protocol.resolve_path(seeds.file))
    config_path = job_dir / "config.yaml"
    result_path = job_dir / "result.json"
    log_path = job_dir / "worker.log"
    OmegaConf.save(job_cfg, config_path, resolve=True)

    environment = os.environ.copy()
    environment["FW_ATTITUDE_WORKER_RESULT"] = str(result_path)
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    (job_dir / "job.json").write_text(json.dumps({
        "checkpoint": checkpoint.name,
        "checkpoint_path": checkpoint.path,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    command = [
        sys.executable, "-u", "-m", "fw_flightcontrol.eval.attitude_control.worker",
        "--config-dir", str(job_dir), "--config-name", "config",
        f"hydra.run.dir={json.dumps(str(job_dir / 'hydra'))}",
        "hydra.output_subdir=null", "hydra.job.chdir=false",
    ]
    prefix = f"[checkpoint {checkpoint.name!r}] "
    with log_path.open("w", encoding="utf-8") as log:
        with log_lock:
            print(f"{prefix}Starting worker; log: {log_path}", flush=True)
        with active_lock:
            if stop.is_set():
                raise RuntimeError(f"Checkpoint {checkpoint.name!r} worker was not started")
            process = subprocess.Popen(
                command, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, cwd=original_cwd,
            )
            active.add(process)
        try:
            if process.stdout is None:
                raise RuntimeError("Worker output pipe was not created")
            with process.stdout as stdout:
                for line in stdout:
                    if not line.endswith("\n"):
                        line += "\n"
                    formatted = prefix + line
                    log.write(formatted)
                    log.flush()
                    with log_lock:
                        sys.stdout.write(formatted)
                        sys.stdout.flush()
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        finally:
            with active_lock:
                active.discard(process)
    if returncode:
        raise RuntimeError(
            f"Checkpoint {checkpoint.name!r} worker exited with code {returncode}; see {log_path}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"Checkpoint {checkpoint.name!r} worker did not write {result_path}")
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read worker result for checkpoint {checkpoint.name!r}") from error
    return validate_worker_result(cfg, checkpoint, result)


def run_parallel(cfg, output):
    checkpoints = list(cfg.evaluation.checkpoints)
    output = Path(output).resolve()
    original_cwd = get_original_cwd() if HydraConfig.initialized() else os.getcwd()
    slots = worker_slots(cfg, len(checkpoints))
    jobs = Queue()
    for index, checkpoint in enumerate(checkpoints):
        jobs.put((index, checkpoint))
    stop = Event()
    log_lock = Lock()
    active_lock = Lock()
    active = set()
    results = {}

    def run_slot(device):
        while not stop.is_set():
            try:
                index, checkpoint = jobs.get_nowait()
            except Empty:
                return
            try:
                results[index] = run_worker(
                    cfg, output, checkpoint, device, log_lock, original_cwd,
                    active, active_lock, stop,
                )
            except Exception:
                stop.set()
                raise

    failures = []
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run_slot, device) for device in slots]
        try:
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(error)
        except BaseException:
            stop.set()
            with active_lock:
                for process in active:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
            raise
    if failures:
        raise RuntimeError(
            "Checkpoint evaluation failed: " + "; ".join(str(error) for error in failures)
        ) from failures[0]
    if len(results) != len(checkpoints):
        raise RuntimeError("Checkpoint evaluation finished without all worker results")
    rows = []
    episodes = []
    for index in range(len(checkpoints)):
        job_rows, job_episodes = results[index]
        rows.extend(job_rows)
        episodes.extend(job_episodes)
    return rows, episodes, False, 0
