import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from train_tdmpc2_icinco import archive_metrics, training_overrides, validate_checkpoint


SUBMODULE = Path(__file__).resolve().parents[1] / "fw_flightcontrol/agents/tdmpc2/tdmpc2"


class TDMPC2TrainingTests(unittest.TestCase):
    def test_selected_training_protocol(self):
        overrides = training_overrides()
        for value in (
            "rl.seed=1", "rl.model_size=5", "rl.steps=750000", "rl.compile=false",
            "rl.use_caps=false", "rl.save_agent=true", "rl.disable_wandb=true",
            "rl.periodic_eval=true", "rl.eval_freq=37500",
            "+env.jsbsim.sim_profile=icinco2024",
        ):
            self.assertIn(value, overrides)
        self.assertIn("rl.steps=2010", training_overrides(smoke=True))
        self.assertIn("rl.seed_steps=2000", training_overrides(smoke=True))

    def test_five_seed_archive_comparison(self):
        rows = archive_metrics(Path(__file__).resolve().parents[1])
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]["avg_rmse"], 0.0199749635)
        self.assertAlmostEqual(rows[0]["avg_fcs_fluct"], 0.03365601978)

    def test_checkpoint_tensor_and_tensordict_metadata_validation(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            weights = {
                "weights": torch.ones(3), "_Qs.params.__batch_size": torch.Size([5]),
                "_Qs.params.__device": None,
            }
            torch.save({"model": weights}, path)
            self.assertEqual(validate_checkpoint(path), 1)
            weights["weights"][0] = float("nan")
            torch.save({"model": weights}, path)
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                validate_checkpoint(path)
            torch.save({"model": {"unexpected": None}}, path)
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                validate_checkpoint(path)

    def test_disabled_tracking_still_saves_checkpoint(self):
        with patch.object(sys, "path", [str(SUBMODULE), *sys.path]):
            from common.logger import Logger

        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create({
                "work_dir": directory, "seed": 1, "save_csv": True, "save_agent": True,
                "task": "ACBohnNoVaIErr", "task_title": "Attitude", "exp_name": "test",
                "obs_shape": {"state": [14]}, "action_dim": 2, "steps": 10,
                "disable_wandb": True, "save_video": False,
            })
            previous = Path.cwd()
            try:
                os.chdir(directory)
                logger = Logger(cfg, cfg)
                agent = Mock()
                logger.save_agent(agent)
                agent.save.assert_called_once_with(Path("models/train/tdmpc2/1/tdmpc2_test_1.pt"))
                self.assertIsNone(logger._wandb)
            finally:
                os.chdir(previous)

    def test_training_does_not_hide_checkpoint_failure(self):
        with patch.object(sys, "path", [str(SUBMODULE), *sys.path]):
            from trainer.online_trainer import OnlineTrainer

        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer._step = 1
        trainer.cfg = OmegaConf.create({"steps": 0, "final_traj_plot": False})
        trainer.cfg_all = OmegaConf.create({"rl": {"task": "ACBohnNoVaIErr"}})
        trainer.agent = Mock()
        trainer.logger = Mock()
        trainer.logger.save_agent.side_effect = OSError("checkpoint write failed")
        with self.assertRaisesRegex(OSError, "checkpoint write failed"):
            trainer.train()

    def test_cuda_training_periodic_eval_and_checkpoint(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("TD-MPC2 requires CUDA")
        repo = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(repo), str(repo.parent / "FW-JSBGym"))),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "WANDB_MODE": "disabled", "MPLBACKEND": "Agg", "HYDRA_FULL_ERROR": "1",
        }
        for mode in ("current", "icinco2024"):
            with self.subTest(inference_mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [
                        sys.executable, str(repo / "fw_flightcontrol/train/tdmpc2_train.py"),
                        *training_overrides(), f"+rl.inference_mode={mode}",
                        "rl.steps=40", "rl.seed_steps=20", "rl.eval_freq=20",
                        "env.jsbsim.episode_length_s=0.1",
                    ],
                    cwd=directory, env=environment, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                self.assertEqual(result.stdout.count("*** Evaluating the agent ***"), 3)
                self.assertIn(f"Inference mode: {mode}", result.stdout)
                self.assertIn("Simulator profile: icinco2024", result.stdout)
                self.assertNotIn("Failed to save model", result.stdout)
                checkpoint = Path(directory) / "models/train/tdmpc2/1/tdmpc2_icinco2024_noactreg_1.pt"
                self.assertEqual(validate_checkpoint(checkpoint), 72)


if __name__ == "__main__":
    unittest.main()
