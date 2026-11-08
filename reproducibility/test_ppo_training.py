import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from train_ppo_icinco import archive_ranges, training_overrides


class PPOTrainingTests(unittest.TestCase):
    def test_paper_overrides(self):
        overrides = training_overrides("cuda")
        for value in (
            "rl.PPO.total_timesteps=750000", "rl.PPO.num_envs=6", "rl.PPO.num_steps=2048",
            "rl.PPO.ts_coef=0", "rl.PPO.ss_coef=0", "rl.PPO.pa_coef=0",
            "roll_limit=60", "pitch_limit=30", "rl.PPO.cuda=true",
            "env.jsbsim.sim_profile=icinco2024",
        ):
            self.assertIn(value, overrides)

    def test_archive_range(self):
        ranges = archive_ranges(Path(__file__).resolve().parents[1])
        self.assertEqual(ranges["avg_rmse"], {"min": 0.028291479, "max": 0.03489262})
        self.assertLess(ranges["paper_surface_fluct"]["min"], 0.001)
        self.assertGreater(ranges["paper_surface_fluct"]["max"], 0.011)

    def test_optimizer_and_autoreset_for_both_profiles(self):
        import torch

        repo = Path(__file__).resolve().parents[1]
        environment = {
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(repo.parent / "FW-JSBGym"), str(repo))),
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "WANDB_MODE": "disabled",
        }
        for profile in ("current", "icinco2024"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                overrides = [
                    value for value in training_overrides("cpu", smoke=True)
                    if not value.startswith("env.jsbsim.sim_profile=")
                ]
                overrides += [
                    f"env.jsbsim.sim_profile={profile}",
                    "rl.PPO.num_steps=16", "rl.PPO.total_timesteps=64",
                    "rl.PPO.num_minibatches=2", "rl.PPO.update_epochs=1",
                    "env.jsbsim.episode_length_s=0.04",
                    f"rl.PPO.periodic_eval={str(profile == 'icinco2024').lower()}",
                ]
                result = subprocess.run(
                    [sys.executable, str(repo / "fw_flightcontrol/train/ppo_train.py"), *overrides],
                    cwd=directory, env=environment, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                self.assertEqual(result.stdout.count(f"Simulator profile: {profile}"), 2)
                self.assertIn("new targets", result.stdout)
                self.assertEqual(result.stdout.count("SPS:"), 2)
                checkpoint = Path(directory) / "models/train/ppo/1/ppo_icinco2024_noactreg_1.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["seed"], 1)
                self.assertTrue(all(torch.isfinite(value).all() for value in payload["agent"].values()))
                self.assertTrue(torch.any(payload["agent"]["actor_logstd"] != 0))


if __name__ == "__main__":
    unittest.main()
