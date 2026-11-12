from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

from fw_flightcontrol.eval.attitude_control.metrics import (
    METRICS,
    aggregate,
    compute_metrics,
)
from fw_flightcontrol.eval.attitude_control import __main__ as entry
from fw_flightcontrol.eval.attitude_control import scheduler
from fw_flightcontrol.eval.attitude_control.protocol import (
    PACKAGE,
    prepare_checkpoint_directory,
    sha256,
    target_name,
    validate_config,
)
from fw_flightcontrol.eval.attitude_control.rollout import run_condition, run_evaluation
from fw_flightcontrol.eval.attitude_control.rollout import (
    Controller,
    selected_simulator_seeds,
    selected_targets,
    simulator_seed_values,
)


CONFIG_DIR = PACKAGE / "config"


def compose_config(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="evaluate", overrides=list(overrides))


def metric_row(checkpoint, training_seed, value):
    return {
        "algorithm": "ppo",
        "checkpoint": checkpoint,
        "training_seed": training_seed,
        "target": "custom",
        "severity": "light",
        "policy_seed": 10,
        "episodes": 1,
        "episodes_attempted": 1,
        "episodes_out_of_bounds": 0,
        "episodes_fluctuation_included": 1,
        "episodes_fluctuation_unavailable": 0,
        **{metric: value for metric in METRICS},
    }


class MetricsTests(unittest.TestCase):
    def test_computes_attitude_rmse_and_surface_fluctuations(self):
        observations = np.zeros((2, 14), dtype=np.float32)
        observations[:, 6] = [3, 4]
        observations[:, 7] = [0, 4]
        result = compute_metrics(observations, [[1, 3, 99], [3, 5, 99]])
        self.assertAlmostEqual(result["roll_rmse"], np.sqrt(12.5))
        self.assertAlmostEqual(result["pitch_rmse"], np.sqrt(8))
        self.assertAlmostEqual(
            result["avg_rmse"], np.mean([np.sqrt(12.5), np.sqrt(8)]), places=6,
        )
        self.assertEqual(result["avg_fcs_fluct"], 3)
        self.assertEqual(
            compute_metrics(observations, [[1, 3, np.nan], [3, 5, np.nan]])[
                "avg_fcs_fluct"
            ],
            3,
        )
        with self.assertRaises(ValueError):
            compute_metrics(observations, [[np.nan, 3, 0], [3, 5, 0]])

    def test_aggregates_checkpoint_seeds_without_collapsing_conditions(self):
        result = aggregate([
            metric_row("seed1", 1, 1),
            metric_row("seed2", 2, 3),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["n_checkpoints"], 2)
        self.assertEqual(result[0]["avg_rmse"], 2)
        self.assertEqual(result[0]["avg_rmse_sem"], 1)

    def test_aggregates_declared_target_groups(self):
        easy = {**metric_row("seed1", 1, 1), "target": "easy"}
        medium = {**metric_row("seed1", 1, 3), "target": "medium"}
        easy["episodes_out_of_bounds"] = 1
        easy["episodes_fluctuation_included"] = 0
        easy["episodes_fluctuation_unavailable"] = 1
        for metric in ("roll_fcs_fluct", "pitch_fcs_fluct", "avg_fcs_fluct"):
            easy[metric] = None
        group = SimpleNamespace(
            name="nominal_refs",
            targets=["easy", "medium"],
            method="mean_of_means",
            sem="mean_of_sems",
        )
        result = aggregate([easy, medium], [group])
        combined = next(row for row in result if row["target"] == "nominal_refs")
        self.assertEqual(combined["avg_rmse"], 2)
        self.assertIsNone(combined["avg_rmse_sem"])
        self.assertTrue(combined["out_of_bounds_error"])
        self.assertEqual(combined["episodes_out_of_bounds"], 1)
        self.assertEqual(combined["episodes_attempted"], 2)
        self.assertIsNone(combined["avg_fcs_fluct"])

    def test_rejects_duplicate_checkpoints_and_invalid_metrics(self):
        for rows in (
            [metric_row("seed1", 1, 1), metric_row("seed1", 1, 1)],
            [metric_row("seed1", 1, float("nan"))],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate(rows)

    def test_aggregation_marks_errors_and_unavailable_fluctuations(self):
        incomplete = metric_row("seed1", 1, 1)
        incomplete["episodes_out_of_bounds"] = 1
        incomplete["episodes_fluctuation_included"] = 0
        incomplete["episodes_fluctuation_unavailable"] = 1
        for metric in ("roll_fcs_fluct", "pitch_fcs_fluct", "avg_fcs_fluct"):
            incomplete[metric] = None
        result = aggregate([incomplete, metric_row("seed2", 2, 3)])[0]
        self.assertTrue(result["out_of_bounds_error"])
        self.assertEqual(result["episodes_out_of_bounds"], 1)
        self.assertEqual(result["episodes_attempted"], 2)
        self.assertEqual(result["episodes_fluctuation_included"], 1)
        self.assertEqual(result["episodes_fluctuation_unavailable"], 1)
        self.assertEqual(result["avg_rmse"], 2)
        self.assertIsNone(result["avg_fcs_fluct"])
        self.assertIsNone(result["avg_fcs_fluct_sem"])


class ConfigTests(unittest.TestCase):
    def test_discovers_only_direct_checkpoints_in_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.pth", "a.pt", "c.PT", "notes.txt"):
                (root / name).write_bytes(b"checkpoint")
            (root / "nested").mkdir()
            (root / "nested" / "ignored.pt").write_bytes(b"checkpoint")
            cfg = compose_config(
                "algorithm=ppo",
                f"evaluation.checkpoint_directory={root}",
                "evaluation.targets=[{path:/tmp/targets.npy}]",
            )
            prepare_checkpoint_directory(cfg)
            validate_config(cfg)

            entries = cfg.evaluation.checkpoints
            self.assertEqual([entry.name for entry in entries], ["a.pt", "b.pth", "c.PT"])
            self.assertEqual([entry.path for entry in entries], [
                str(root / "a.pt"), str(root / "b.pth"), str(root / "c.PT"),
            ])
            self.assertTrue(all(entry.training_seed is None for entry in entries))

    def test_checkpoint_directory_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "not-a-directory"
            file.write_bytes(b"checkpoint")
            target = "evaluation.targets=[{path:/tmp/targets.npy}]"
            cases = (
                (("algorithm=ppo", f"evaluation.checkpoint_directory={root / 'missing'}", target),
                 FileNotFoundError),
                (("algorithm=ppo", f"evaluation.checkpoint_directory={file}", target),
                 NotADirectoryError),
                (("algorithm=ppo", f"evaluation.checkpoint_directory={root}", target),
                 ValueError),
                ((f"evaluation.checkpoint_directory={root}", target), ValueError),
                (("algorithm=ppo", f"evaluation.checkpoint_directory={root}",
                  "checkpoints=single", "checkpoints.entries.0.path=/tmp/model.pt", target),
                 ValueError),
            )
            for overrides, error in cases:
                with self.subTest(overrides=overrides):
                    cfg = compose_config(*overrides)
                    with self.assertRaises(error):
                        prepare_checkpoint_directory(cfg)

    def test_generic_and_icinco_policy_seed_defaults_to_ten(self):
        for overrides in ((), ("experiment=icinco2024",)):
            with self.subTest(overrides=overrides):
                cfg = compose_config(*overrides)
                self.assertEqual(list(cfg.evaluation.policy_seeds), [10])

    def test_algorithm_groups_compose_with_their_own_rl_configs(self):
        expected_rl_group = {
            "pid": "PPO",
            "ppo": "PPO",
            "sac": "SAC",
            "tdmpc2": "task",
        }
        for algorithm, first_key in expected_rl_group.items():
            with self.subTest(algorithm=algorithm):
                cfg = compose_config(f"algorithm={algorithm}")
                self.assertEqual(cfg.algorithm.name, algorithm)
                self.assertIn(first_key, cfg.rl)

    def test_icinco_preset_is_a_configurable_historical_default(self):
        cfg = compose_config("experiment=icinco2024")
        validate_config(cfg)
        self.assertEqual(list(cfg.evaluation.atmo_severities), ["off"])
        self.assertEqual(cfg.evaluation.episodes.count, 25)
        self.assertTrue(cfg.evaluation.episodes.include_final_reset)
        self.assertEqual(len(cfg.evaluation.targets), 3)
        targets = selected_targets(cfg)
        self.assertEqual([name for name, _, _ in targets], ["easy", "medium", "hard"])
        self.assertEqual(list(cfg.evaluation.aggregation.target_groups[0].targets),
                         ["easy", "medium"])
        self.assertEqual([len(values) for _, _, values in targets], [25, 25, 25])
        start, end, seeds = selected_simulator_seeds(cfg, len(targets[0][2]))
        self.assertEqual((start, end), (0, 25))
        self.assertEqual(len(seeds), 26)
        self.assertTrue(np.all(seeds > 0))

    def test_custom_checkpoints_and_visualization_compose_independently(self):
        cfg = compose_config(
            "algorithm=ppo",
            "checkpoints=single",
            "checkpoints.entries.0.path=/tmp/custom-checkpoint.pt",
            "evaluation.targets=[{path:/tmp/custom-targets.npy}]",
            "evaluation.atmo_severities=[light,severe]",
            "evaluation.visualization.mode=matplotlib",
        )
        validate_config(cfg)
        self.assertEqual(list(cfg.evaluation.atmo_severities), ["light", "severe"])
        self.assertEqual(cfg.evaluation.visualization.mode, "matplotlib")
        self.assertEqual(target_name(cfg.evaluation.targets[0]), "custom-targets")

    def test_target_filenames_must_be_unique_and_names_are_not_configurable(self):
        for targets in (
            "[{path:/one/easy.npy},{path:/two/easy.npy}]",
            "[{path:/one/easy.npy},{path:/two/easy.npz}]",
            "[{name:alias,path:/one/easy.npy}]",
            "[{path:''}]",
        ):
            with self.subTest(targets=targets):
                cfg = compose_config(f"evaluation.targets={targets}")
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_simulator_seed_values_reject_zero_and_fractional_seeds(self):
        cfg = compose_config()
        cfg.evaluation.simulator_seeds.mode = "values"
        for values in ([0], [1.5]):
            with self.subTest(values=values):
                cfg.evaluation.simulator_seeds.seeds = values
                with self.assertRaises(ValueError):
                    simulator_seed_values(cfg.evaluation, 1)

    def test_execution_limits_are_validated(self):
        cfg = compose_config("evaluation.targets=[{path:/tmp/targets.npy}]")
        for field, value in (("workers_per_gpu", 0), ("workers_per_gpu", 5),
                             ("workers_per_gpu", True), ("cpu_workers", 0)):
            with self.subTest(field=field, value=value):
                original = cfg.evaluation.execution[field]
                cfg.evaluation.execution[field] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)
                cfg.evaluation.execution[field] = original


FAKE_WORKER = """
import os
import sys
import time
import types
import numpy as np

module = types.ModuleType("fw_flightcontrol.eval.attitude_control.rollout")
def run_evaluation(cfg, output):
    from hydra.utils import get_original_cwd
    if get_original_cwd() != os.environ["FAKE_ORIGINAL_CWD"]:
        raise RuntimeError("Incorrect Hydra original working directory")
    checkpoint = cfg.evaluation.checkpoints[0]
    name = checkpoint.name
    target_name = os.path.splitext(os.path.basename(cfg.evaluation.targets[0].path))[0]
    target_values = np.load(cfg.evaluation.targets[0].path, allow_pickle=False)[0].tolist()
    print("fake rollout for " + name, flush=True)
    if os.environ.get("FAKE_WORKER_MODE") == "failure" and name == "a.pt":
        raise RuntimeError("intentional fake worker failure")
    if os.environ.get("FAKE_WORKER_MODE") == "missing" and name == "a.pt":
        os._exit(0)
    started = time.monotonic()
    time.sleep(2.5 if name == "a.pt" else 0.1)
    finished = time.monotonic()
    out_of_bounds = name == "b.pt"
    row = {
        "algorithm": "ppo", "checkpoint": name, "training_seed": None,
        "target": target_name, "severity": "off", "policy_seed": 10,
        "episodes": 1, "episodes_attempted": 1, "episodes_remaining": 0,
        "episodes_out_of_bounds": int(out_of_bounds), "out_of_bounds_error": out_of_bounds,
        "episodes_early_termination": 0, "episodes_fluctuation_included": 1,
        "episodes_fluctuation_unavailable": 0, "steps_completed": 2,
        "roll_rmse": 1, "pitch_rmse": 1, "avg_rmse": 1,
        "roll_fcs_fluct": 1, "pitch_fcs_fluct": 1, "avg_fcs_fluct": 1,
    }
    episode = {
        "algorithm": "ppo", "checkpoint": name, "training_seed": None,
        "target": target_name, "severity": "off", "policy_seed": 10,
        "episode": 0, "simulator_seed": 10, "reward": 1, "steps": 2,
        "duration_s": 0.02, "termination_reason": "out_of_bounds" if out_of_bounds else "truncated",
        "terminated": out_of_bounds, "truncated": not out_of_bounds,
        "early_termination": False, "out_of_bounds": out_of_bounds,
        "error": "OUT_OF_BOUNDS" if out_of_bounds else None,
        "fluctuation_unavailable_reason": None, "started": started, "finished": finished,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "target_values": target_values,
    }
    if os.environ.get("FAKE_WORKER_MODE") == "malformed" and name == "a.pt":
        del episode["out_of_bounds"]
    if os.environ.get("FAKE_WORKER_MODE") == "incomplete" and name == "a.pt":
        return [row], [], False, 0
    return [row], [episode], False, 0
module.run_evaluation = run_evaluation
sys.modules[module.__name__] = module
"""


class SchedulerTests(unittest.TestCase):
    def test_slots_respect_visible_gpu_order_and_cpu_limits(self):
        cfg = compose_config()
        with patch.object(scheduler.torch.cuda, "is_available", return_value=True), \
                patch.object(scheduler.torch.cuda, "device_count", return_value=2), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5,GPU-abc"}):
            self.assertEqual(
                scheduler.worker_slots(cfg, 7),
                ["5", "GPU-abc", "5", "GPU-abc", "5", "GPU-abc", "5"],
            )
            cfg.evaluation.execution.workers_per_gpu = 1
            self.assertEqual(scheduler.worker_slots(cfg, 8), ["5", "GPU-abc"])
        with patch.object(scheduler.torch.cuda, "is_available", return_value=False):
            self.assertEqual(scheduler.worker_slots(cfg, 9), [None, None])
            cfg.evaluation.execution.cpu_workers = 3
            self.assertEqual(scheduler.worker_slots(cfg, 9), [None, None, None])

    def run_fake_workers(
        self, root, mode="normal", via_entry=False, gpu=False, original_cwd=None,
        relative_target=False, target_symlink=False,
    ):
        (root / "sitecustomize.py").write_text(FAKE_WORKER)
        models = root / "models"
        models.mkdir()
        for name in ("a.pt", "b.pt", "c model.pth"):
            (models / name).write_bytes(b"fake checkpoint")
        target = root / "targets.npy"
        target_data = root / "actual.npy" if target_symlink else target
        np.save(target_data, np.array([[0.25, -0.5]], dtype=np.float32))
        if target_symlink:
            target.symlink_to(target_data)
        target_override = os.path.relpath(target, Path.cwd()) if relative_target else target
        cfg = compose_config(
            "algorithm=ppo", f"evaluation.checkpoint_directory={models}",
            f"evaluation.targets=[{{path:{target_override}}}]",
            f"evaluation.output={root / 'output'}",
        )
        if not via_entry:
            prepare_checkpoint_directory(cfg)
            validate_config(cfg)
        env = {
            "PYTHONPATH": os.pathsep.join((str(root), str(PACKAGE.parent))),
            "FAKE_ORIGINAL_CWD": str(original_cwd) if original_cwd else os.getcwd(),
            "FAKE_WORKER_MODE": mode,
        }
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = "5,GPU-abc"
        terminal = io.StringIO()
        with patch.dict(os.environ, env), \
                patch.object(scheduler.torch.cuda, "is_available", return_value=gpu), \
                patch.object(scheduler.torch.cuda, "device_count", return_value=2 if gpu else 0), \
                patch.object(scheduler.HydraConfig, "initialized",
                             return_value=original_cwd is not None), \
                patch.object(scheduler, "get_original_cwd", return_value=str(original_cwd)), \
                redirect_stdout(terminal):
            if via_entry:
                runtime = {"packages": {"jsbsim": {"imported_version": "test"}}}
                with patch.object(entry, "runtime_info", return_value=runtime), \
                        patch.object(entry, "source_info", return_value=({}, root)), \
                        patch.object(entry, "input_files", return_value=[]), \
                        redirect_stderr(io.StringIO()):
                    result = entry.run(cfg)
            else:
                result = scheduler.run_parallel(cfg, root / "output")
        return result, terminal.getvalue()

    def test_real_workers_run_concurrently_and_preserve_order_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (rows, episodes, stopped, remaining), terminal = self.run_fake_workers(root)
            self.assertEqual([row["checkpoint"] for row in rows], ["a.pt", "b.pt", "c model.pth"])
            self.assertFalse(stopped)
            self.assertEqual(remaining, 0)
            self.assertTrue(episodes[1]["out_of_bounds"])
            self.assertLess(episodes[2]["started"], episodes[0]["finished"])
            events = sorted(
                [(episode["started"], 1) for episode in episodes]
                + [(episode["finished"], -1) for episode in episodes]
            )
            active = maximum = 0
            for _, delta in events:
                active += delta
                maximum = max(maximum, active)
            self.assertEqual(maximum, 2)
            for name in ("a.pt", "b.pt", "c model.pth"):
                prefix = f"[checkpoint {name!r}] "
                self.assertIn(prefix + "fake rollout for " + name, terminal)
                job = root / "output" / "jobs" / name
                self.assertIn(prefix + "fake rollout for " + name, (job / "worker.log").read_text())
                self.assertTrue((job / "config.yaml").is_file())
                self.assertTrue((job / "result.json").is_file())

    def test_entry_aggregates_real_worker_results_and_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _ = self.run_fake_workers(root, via_entry=True)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["out_of_bounds_error"])
            self.assertEqual(result["episodes_out_of_bounds"], 1)
            self.assertEqual(result["aggregates"][0]["n_checkpoints"], 3)
            self.assertEqual(json.loads((root / "output" / "result.json").read_text()), result)

    def test_worker_hydra_context_preserves_original_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_cwd = root / "original working directory"
            original_cwd.mkdir()
            (rows, _, _, _), _ = self.run_fake_workers(root, original_cwd=original_cwd)
            self.assertEqual(len(rows), 3)

    def test_workers_load_same_configured_relative_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_cwd = root / "different worker cwd"
            original_cwd.mkdir()
            result, _ = self.run_fake_workers(
                root, original_cwd=original_cwd, relative_target=True, target_symlink=True,
            )
            if not isinstance(result, tuple):
                self.fail("Expected parallel worker results")
            _, episodes, _, _ = result
            target = (root / "targets.npy").absolute()
            for episode in episodes:
                job_dir = root / "output" / "jobs" / episode["checkpoint"]
                cfg = OmegaConf.load(job_dir / "config.yaml")
                self.assertEqual(cfg.evaluation.targets[0].path, str(target))
                self.assertEqual(episode["target_values"], [0.25, -0.5])
                self.assertEqual(episode["target"], "targets")

    def test_worker_gpu_visibility_matches_assigned_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (_, episodes, _, _), _ = self.run_fake_workers(root, gpu=True)
            devices = []
            for episode in episodes:
                name = episode["checkpoint"]
                job = json.loads((root / "output" / "jobs" / name / "job.json").read_text())
                self.assertEqual(job["cuda_visible_devices"], episode["device"])
                devices.append(episode["device"])
            self.assertCountEqual(devices, ["5", "5", "GPU-abc"])

    def test_single_checkpoint_and_visualized_runs_use_serial_rollout(self):
        for mode, names in (("none", ("a.pt",)), ("matplotlib", ("a.pt", "b.pt"))):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                models = root / "models"
                models.mkdir()
                for name in names:
                    (models / name).write_bytes(b"checkpoint")
                cfg = compose_config(
                    "algorithm=ppo", f"evaluation.checkpoint_directory={models}",
                    "evaluation.targets=[{path:unused.npy}]",
                    f"evaluation.visualization.mode={mode}",
                    f"evaluation.output={root / 'output'}",
                )
                runtime = {"packages": {"jsbsim": {"imported_version": "test"}}}
                rows = [metric_row(name, None, 1) for name in names]
                with patch.object(entry, "runtime_info", return_value=runtime), \
                        patch.object(entry, "source_info", return_value=({}, root)), \
                        patch.object(entry, "input_files", return_value=[]), \
                        patch.object(entry.scheduler, "run_parallel") as parallel, \
                        patch.object(entry.rollout, "run_evaluation",
                                     return_value=(rows, [], False, 0)) as serial:
                    entry.run(cfg)
                serial.assert_called_once()
                parallel.assert_not_called()

    def test_worker_failure_and_malformed_results_fail_explicitly(self):
        cases = (
            ("failure", "exited with code"),
            ("missing", "did not write"),
            ("malformed", "Malformed worker"),
            ("incomplete", "Inconsistent episode counts"),
        )
        for mode, message in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = "Malformed worker" if mode == "incomplete" else message
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.run_fake_workers(root, mode, via_entry=True)
                self.assertFalse((root / "output" / "result.json").exists())
                self.assertFalse((root / "output" / "aggregates.csv").exists())


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.steps = 0
        self.positions = []
        self.previous_positions = []

    def reset(self, options):
        self.previous_positions = self.positions
        self.positions = []
        self.steps = 0
        return np.zeros(14, dtype=np.float32), {
            "fcs_pos_hist": self.previous_positions,
        }

    def set_target_state(self, target):
        self.target = target

    def step(self, action):
        self.steps += 1
        observation = np.zeros(14, dtype=np.float32)
        observation[6] = self.steps
        observation[7] = 2 * self.steps
        self.positions.append([self.steps, 2 * self.steps, 100 * self.steps])
        info = {
            "non_norm_obs": observation,
            "episode": {"r": np.float32(1)},
            "out_of_bounds": False,
            "fcs_pos_hist": self.positions.copy(),
        }
        done = self.steps == 2
        return observation, 1.0, False, done, info


class FakeController:
    algorithm = "pid"

    def __init__(self):
        self.env = FakeEnv()

    def start_condition(self, observation):
        pass

    def reset_policy(self, observation):
        pass

    def action(self, observation, target, timestep):
        return np.zeros(2)

    def after_step(self, observation, action):
        pass


class OutOfBoundsEnv(FakeEnv):
    def __init__(self, *, every_episode=False):
        super().__init__()
        self.episode = 0
        self.every_episode = every_episode

    def reset(self, options):
        self.episode += 1
        return super().reset(options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.steps == 1 and (self.every_episode or self.episode == 1):
            info["out_of_bounds"] = True
            return observation, reward, True, False, info
        return observation, reward, terminated, truncated, info


@unittest.skipUnless(
    os.environ.get("FW_RUN_REAL_GPU_TESTS") == "1",
    "Set FW_RUN_REAL_GPU_TESTS=1 to run the real GPU/JSBSim integration test",
)
class RealGpuIntegrationTests(unittest.TestCase):
    def test_ppo_checkpoints_run_with_the_configured_target(self):
        self.assertTrue(scheduler.torch.cuda.is_available(), "A CUDA GPU is required")
        checkpoints = PACKAGE / "models/icinco/ppo/actreg/caps"
        names = sorted(
            path.name for path in checkpoints.iterdir()
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
        )
        self.assertGreaterEqual(len(names), 2, "At least two PPO CAPS checkpoints are required")
        source_target = PACKAGE / "eval/attitude_control/targets/easy.npy"

        with tempfile.TemporaryDirectory(prefix="attitude-gpu-test-") as directory:
            root = Path(directory)
            target = root / "configured-target.npy"
            np.save(target, np.load(source_target, allow_pickle=False))
            output = root / "output"
            command = [
                sys.executable, "-u", "-m", "fw_flightcontrol.eval.attitude_control",
                "experiment=icinco2024", "algorithm=ppo",
                f"evaluation.checkpoint_directory={checkpoints}",
                f"evaluation.targets=[{{path:{json.dumps(str(target))}}}]",
                "evaluation.aggregation.target_groups=[]", "evaluation.episodes.count=1",
                f"evaluation.output={output}", f"hydra.run.dir={root / 'hydra'}",
                "hydra.output_subdir=null",
            ]
            environment = os.environ.copy()
            environment["HYDRA_FULL_ERROR"] = "1"
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment.setdefault("MKL_NUM_THREADS", "1")
            completed = subprocess.run(
                command, cwd=PACKAGE.parent, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            report = json.loads((output / "result.json").read_text())
            provenance = json.loads((output / "provenance.json").read_text())
            episodes = json.loads((output / "episodes.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["inputs_verified_after_run"])
            self.assertTrue(provenance["runtime"]["cuda"]["available"])
            self.assertEqual(provenance["input_sha256"][str(target)], sha256(target))
            self.assertEqual(report["episode_count"], len(names))
            self.assertEqual(report["aggregates"][0]["n_checkpoints"], len(names))
            self.assertEqual([episode["checkpoint"] for episode in episodes], names)
            self.assertTrue(all(episode["target"] == "configured-target" and episode["steps"] > 0
                                for episode in episodes))
            self.assertEqual(len({episode["simulator_seed"] for episode in episodes}), 1)
            first_reward = completed.stdout.index("Episode reward:")
            self.assertGreaterEqual(completed.stdout[:first_reward].count("Starting worker;"), 2)

            devices = set()
            for name in names:
                job_dir = output / "jobs" / name
                job_cfg = OmegaConf.load(job_dir / "config.yaml")
                job = json.loads((job_dir / "job.json").read_text())
                log = (job_dir / "worker.log").read_text()
                self.assertEqual(job_cfg.evaluation.targets[0].path, str(target))
                self.assertIsNotNone(job["cuda_visible_devices"])
                devices.add(job["cuda_visible_devices"])
                self.assertIn(f"[checkpoint {name!r}] **** Using Device: cuda ****", log)
            self.assertEqual(len(devices), min(len(names), scheduler.torch.cuda.device_count()))


class RolloutTests(unittest.TestCase):
    def test_runs_discovered_checkpoints_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            first = models / "a.pt"
            second = models / "b.pth"
            first.write_bytes(b"first model")
            second.write_bytes(b"second model")
            target = root / "targets.npy"
            np.save(target, np.zeros((1, 2), dtype=np.float32))
            cfg = compose_config(
                "algorithm=ppo",
                f"evaluation.checkpoint_directory={models}",
                f"evaluation.targets=[{{path:{target}}}]",
            )
            prepare_checkpoint_directory(cfg)
            validate_config(cfg)
            controllers = [FakeController(), FakeController()]
            for controller in controllers:
                controller.load_agent = Mock()
                controller.env.close = Mock()
            with patch(
                "fw_flightcontrol.eval.attitude_control.rollout.Controller",
                side_effect=controllers,
            ):
                rows, _, stopped, remaining = run_evaluation(cfg, root)
            self.assertEqual([row["checkpoint"] for row in rows], ["a.pt", "b.pth"])
            controllers[0].load_agent.assert_called_once_with(first)
            controllers[1].load_agent.assert_called_once_with(second)
            self.assertFalse(stopped)
            self.assertEqual(remaining, 0)

    def test_directory_checkpoints_are_recorded_and_hashed_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            first = models / "a.pt"
            second = models / "b.pth"
            first.write_bytes(b"first model")
            second.write_bytes(b"second model")
            target = root / "targets.npy"
            np.save(target, np.zeros((1, 2), dtype=np.float32))
            output = root / "output"
            cfg = compose_config(
                "algorithm=ppo",
                f"evaluation.checkpoint_directory={models}",
                f"evaluation.targets=[{{path:{target}}}]",
                f"evaluation.output={output}",
            )
            runtime = {"packages": {"jsbsim": {"imported_version": "test"}}}

            def fake_evaluation(run_cfg, run_output):
                self.assertEqual(run_output, output)
                self.assertEqual(
                    [item.path for item in run_cfg.evaluation.checkpoints],
                    [str(first), str(second)],
                )
                rows = [
                    metric_row(item.name, item.training_seed, 1)
                    for item in run_cfg.evaluation.checkpoints
                ]
                return rows, [], False, 0

            with patch.object(entry, "runtime_info", return_value=runtime), \
                    patch.object(entry, "source_info", return_value=({}, root)), \
                    patch.object(entry.scheduler, "run_parallel", side_effect=fake_evaluation):
                entry.run(cfg)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(
                [item["name"] for item in provenance["configuration"]["evaluation"]["checkpoints"]],
                ["a.pt", "b.pth"],
            )
            self.assertIn(str(first), provenance["input_sha256"])
            self.assertIn(str(second), provenance["input_sha256"])
            self.assertEqual(cfg.evaluation.checkpoints[0].name, "default")

    def test_detects_discovered_checkpoint_changes_during_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            checkpoint = models / "model.pt"
            checkpoint.write_bytes(b"original")
            target = root / "targets.npy"
            np.save(target, np.zeros((1, 2), dtype=np.float32))
            cfg = compose_config(
                "algorithm=ppo",
                f"evaluation.checkpoint_directory={models}",
                f"evaluation.targets=[{{path:{target}}}]",
                f"evaluation.output={root / 'output'}",
            )
            runtime = {"packages": {"jsbsim": {"imported_version": "test"}}}

            def change_checkpoint(run_cfg, run_output):
                checkpoint.write_bytes(b"changed")
                return [], [], False, 0

            with patch.object(entry, "runtime_info", return_value=runtime), \
                    patch.object(entry, "source_info", return_value=({}, root)), \
                    patch.object(entry.rollout, "run_evaluation", side_effect=change_checkpoint):
                with self.assertRaisesRegex(RuntimeError, "Evaluation inputs changed"):
                    entry.run(cfg)

    def test_out_of_bounds_is_reported_and_next_episode_runs(self):
        for overrides in ((), ("experiment=icinco2024",)):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                cfg = compose_config(
                    *overrides,
                    "evaluation.episodes.count=2",
                    "evaluation.episodes.include_final_reset=false",
                    "evaluation.simulator_seeds.mode=values",
                    "evaluation.simulator_seeds.seeds=[123,456]",
                )
                controller = FakeController()
                controller.env = OutOfBoundsEnv()
                errors = io.StringIO()
                with redirect_stderr(errors):
                    result, episodes, stopped = run_condition(
                        cfg, controller, "custom", np.zeros((2, 2), dtype=np.float32),
                        "off", 10, None, Path(directory) / "telemetry.csv",
                    )
                self.assertIn("ERROR: Out of bounds", errors.getvalue())
                self.assertEqual(result["episodes"], 2)
                self.assertTrue(result["out_of_bounds_error"])
                self.assertEqual(result["episodes_out_of_bounds"], 1)
                self.assertEqual(result["episodes_fluctuation_included"], 1)
                self.assertEqual(result["episodes_fluctuation_unavailable"], 1)
                self.assertEqual(episodes[0]["error"], "OUT_OF_BOUNDS")
                self.assertEqual(
                    episodes[0]["fluctuation_unavailable_reason"],
                    "fewer than two control-surface samples",
                )
                self.assertFalse(episodes[1]["out_of_bounds"])
                self.assertFalse(stopped)

    def test_all_short_out_of_bounds_episodes_leave_fluctuation_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = compose_config(
                "evaluation.episodes.count=1",
                "evaluation.simulator_seeds.mode=values",
                "evaluation.simulator_seeds.seeds=[123]",
            )
            controller = FakeController()
            controller.env = OutOfBoundsEnv(every_episode=True)
            with redirect_stderr(io.StringIO()):
                result, episodes, _ = run_condition(
                    cfg, controller, "custom", np.zeros((1, 2), dtype=np.float32),
                    "off", 10, None, Path(directory) / "telemetry.csv",
                )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(result["episodes_fluctuation_unavailable"], 1)
            self.assertIsNone(result["avg_fcs_fluct"])
            self.assertIsNotNone(result["avg_rmse"])
            self.assertIsNone(aggregate([result])[0]["avg_fcs_fluct"])

    def test_result_report_and_terminal_mark_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = compose_config(
                f"evaluation.output={directory}/output",
                "evaluation.targets=[{path:unused.npy}]",
            )
            row = metric_row("pid", None, 1)
            row["algorithm"] = "pid"
            row["episodes_out_of_bounds"] = 1
            episodes = [{"out_of_bounds": True, "error": "OUT_OF_BOUNDS"}]
            runtime = {"packages": {"jsbsim": {"imported_version": "test"}}}
            errors = io.StringIO()
            with patch.object(entry, "runtime_info", return_value=runtime), \
                    patch.object(entry, "source_info", return_value=({}, Path(directory))), \
                    patch.object(entry, "input_files", return_value=[]), \
                    patch.object(entry.rollout, "run_evaluation", return_value=([row], episodes, False, 0)), \
                    redirect_stderr(errors):
                result = entry.run(cfg)
            report = json.loads((Path(directory) / "output" / "result.json").read_text())
            self.assertTrue(result["out_of_bounds_error"])
            self.assertTrue(report["out_of_bounds_error"])
            self.assertEqual(report["episodes_out_of_bounds"], 1)
            self.assertEqual(report["status"], "completed")
            self.assertIn("ERROR: Evaluation completed with 1 out-of-bounds", errors.getvalue())

    def test_default_policy_seed_is_initialized_once_across_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            first_target = Path(directory) / "first.npy"
            second_target = Path(directory) / "second.npy"
            np.save(first_target, np.zeros((2, 2), dtype=np.float32))
            np.save(second_target, np.zeros((2, 2), dtype=np.float32))
            cfg = compose_config(
                f"evaluation.targets=[{{path:{first_target}}},{{path:{second_target}}}]",
                "evaluation.atmo_severities=[off,light]",
                "evaluation.policy_seeds=[10,17]",
                "evaluation.simulator_seeds.mode=values",
                "evaluation.simulator_seeds.seeds=[123,456]",
            )
            controller = FakeController()
            controller.load_agent = Mock()
            controller.env.close = Mock()
            with patch(
                "fw_flightcontrol.eval.attitude_control.rollout.Controller",
                return_value=controller,
            ), patch(
                "fw_flightcontrol.eval.attitude_control.rollout.seed_policy",
            ) as seed_policy:
                rows, episodes, stopped, remaining = run_evaluation(cfg, Path(directory))

        self.assertEqual(seed_policy.call_args_list, [call(10), call(17)])
        self.assertEqual(len(rows), 8)
        for policy_seed in (10, 17):
            simulator_seeds = [
                row["simulator_seed"] for row in episodes
                if row["policy_seed"] == policy_seed
            ]
            self.assertEqual(simulator_seeds, [123, 456] * 4)
        self.assertFalse(stopped)
        self.assertEqual(remaining, 0)

    def test_tdmpc2_parser_uses_the_selected_policy_seed(self):
        cfg = compose_config("algorithm=tdmpc2", "rl.seed=999")
        submodule = str(PACKAGE / "agents/tdmpc2/tdmpc2")
        fake_env = SimpleNamespace(state_prps=[], unwrapped=SimpleNamespace())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "path", [submodule, *sys.path]), \
                patch("hydra.utils.get_original_cwd", return_value=directory):
            from fw_flightcontrol.agents.tdmpc2.tdmpc2.common import parser
            from fw_flightcontrol.agents.tdmpc2.tdmpc2 import envs

            with patch.object(parser, "set_seed", wraps=parser.set_seed) as set_seed, \
                    patch.object(envs, "make_env", return_value=fake_env):
                controller = Controller(cfg, Path(directory) / "telemetry.csv", 17)
            set_seed.assert_called_once_with(17)
            self.assertEqual(controller.cfg.rl.seed, 17)
            self.assertEqual(cfg.rl.seed, 999)

    def test_tdmpc2_evaluation_passes_seeds_without_seeding_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            cfg = compose_config(
                "algorithm=tdmpc2", "checkpoints=single",
                f"checkpoints.entries.0.path={checkpoint}",
                "evaluation.policy_seeds=[10,17]",
                "evaluation.episodes.count=1",
                "evaluation.simulator_seeds.mode=values",
                "evaluation.simulator_seeds.seeds=[123]",
            )
            controller = FakeController()
            controller.load_agent = Mock()
            controller.env.close = Mock()
            with patch("fw_flightcontrol.eval.attitude_control.rollout.selected_targets",
                       return_value=[("easy", None, np.zeros((1, 2)))]), \
                    patch("fw_flightcontrol.eval.attitude_control.rollout.run_condition",
                          return_value=({}, [{}], False)), \
                    patch("fw_flightcontrol.eval.attitude_control.rollout.Controller",
                          return_value=controller) as factory, \
                    patch("fw_flightcontrol.eval.attitude_control.rollout.seed_policy") as seed:
                _, _, stopped, remaining = run_evaluation(cfg, Path(directory))
            self.assertEqual([args.args[2] for args in factory.call_args_list], [10, 17])
            seed.assert_not_called()
            self.assertFalse(stopped)
            self.assertEqual(remaining, 0)

    def test_pid_first_action_after_reset_uses_new_observation(self):
        for overrides in ((), ("experiment=icinco2024",)):
            with self.subTest(overrides=overrides):
                controller = Controller.__new__(Controller)
                controller.cfg = compose_config(*overrides)
                controller.algorithm = "pid"
                controller.roll_pid = Mock()
                controller.pitch_pid = Mock()
                controller.roll_pid.update.return_value = (0.0,)
                controller.pitch_pid.update.return_value = (0.0,)
                controller.pid_state = np.array([1., 2., 0., 3., 4.])
                initial = np.array([5., 6., 0., 7., 8.])

                controller.reset_policy(initial)
                controller.action(initial, np.array([0., 0.]), 0)

                controller.roll_pid.reset.assert_called_once_with()
                controller.pitch_pid.reset.assert_called_once_with()
                controller.roll_pid.update.assert_called_once_with(
                    5., 7., saturate=True, normalize=True,
                )
                controller.pitch_pid.update.assert_called_once_with(
                    6., 8., saturate=True, normalize=True,
                )

    def test_generic_rollout_records_each_episode_and_returns_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "targets.npy"
            np.save(target_path, np.zeros((2, 2), dtype=np.float32))
            cfg = compose_config(
                f"evaluation.targets=[{{path:{target_path}}}]",
                "evaluation.episodes.count=2",
                "evaluation.episodes.include_final_reset=true",
                "evaluation.simulator_seeds.mode=values",
                "evaluation.simulator_seeds.seeds=[1,2,3]",
            )
            telemetry = Path(directory) / "telemetry.csv"
            result, episodes, stopped = run_condition(
                cfg,
                FakeController(),
                "custom",
                np.load(target_path),
                "off",
                10,
                None,
                telemetry,
            )
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["episodes_early_termination"], 2)
        self.assertEqual(result["episodes_out_of_bounds"], 0)
        self.assertEqual(result["steps_completed"], 4)
        self.assertEqual(result["severity"], "off")
        self.assertEqual([row["episode"] for row in episodes], [0, 1])
        self.assertEqual([row["simulator_seed"] for row in episodes], [1, 2])
        self.assertEqual([row["duration_s"] for row in episodes], [0.02, 0.02])
        self.assertEqual([row["termination_reason"] for row in episodes], ["truncated", "truncated"])
        self.assertTrue(all(row["early_termination"] for row in episodes))
        self.assertFalse(stopped)
        self.assertAlmostEqual(result["roll_rmse"], np.sqrt(2.5))
        self.assertEqual(result["roll_fcs_fluct"], 1)
        self.assertEqual(result["avg_fcs_fluct"], 1.5)

    def test_quit_stops_after_episode_even_when_selection_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "targets.npy"
            np.save(target_path, np.zeros((2, 2), dtype=np.float32))
            cfg = compose_config(
                f"evaluation.targets=[{{path:{target_path}}}]",
                "evaluation.episodes.count=2",
                "evaluation.simulator_seeds.mode=values",
                "evaluation.simulator_seeds.seeds=[1,2]",
                "evaluation.visualization.interactive=true",
            )
            telemetry = Path(directory) / "telemetry.csv"
            with patch("builtins.input", return_value="q"):
                result, episodes, stopped = run_condition(
                    cfg,
                    FakeController(),
                    "custom",
                    np.load(target_path),
                    "off",
                    10,
                    None,
                    telemetry,
                )
        self.assertTrue(stopped)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["episodes_remaining"], 1)
        self.assertEqual(len(episodes), 1)


if __name__ == "__main__":
    unittest.main()
