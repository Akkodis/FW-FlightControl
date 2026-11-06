from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch


REPO = Path(__file__).resolve().parents[1]

from fw_flightcontrol.agents.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from common import math as td_math


def agent_config(mode):
    config_dir = REPO / "fw_flightcontrol/config"
    cfg = OmegaConf.merge(
        OmegaConf.load(config_dir / "tdmpc2_default.yaml").rl,
        OmegaConf.load(config_dir / "rl/attitude_control/ac_tdmpc2.yaml"),
    )
    cfg.inference_mode = mode
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.obs_shape = {"state": [14]}
    cfg.action_dim = 2
    cfg.episode_length = 2000
    cfg.enc_dim = 32
    cfg.latent_dim = 32
    cfg.mlp_dim = 64
    cfg.iterations = 2
    cfg.num_samples = 32
    cfg.num_pi_trajs = 4
    cfg.num_elites = 8
    return cfg


def make_agents():
    torch.manual_seed(123)
    default_cfg = agent_config("current")
    del default_cfg.inference_mode
    current = TDMPC2(default_cfg)
    legacy = TDMPC2(agent_config("icinco2024"))
    with torch.no_grad():
        current.model._reward[-1].weight.normal_(std=0.01)
        current.model._Qs.params["2", "weight"].normal_(std=0.01)
    legacy.model.load_state_dict(current.model.state_dict())
    return current, legacy


def trace(agent, eval_mode):
    np.random.seed(10)
    torch.manual_seed(10)
    observations = [torch.linspace(-0.2, 0.2, 14) + i * 0.01 for i in range(6)]
    actions = [
        agent.act(obs, t0=i in (0, 3), eval_mode=eval_mode)
        for i, obs in enumerate(observations)
    ]
    return actions, np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()


class InferenceConfigurationTests(unittest.TestCase):
    def test_unknown_mode_is_rejected(self):
        cfg = agent_config("legacy_typo")
        with self.assertRaisesRegex(ValueError, "Unknown TD-MPC2 inference mode"):
            TDMPC2(cfg)

    def test_unsupported_legacy_configuration_is_rejected(self):
        cfg = agent_config("icinco2024")
        cfg.compile = True
        cfg.mppi_actfluct = True
        with self.assertRaisesRegex(ValueError, "compile.*mppi_actfluct"):
            TDMPC2(cfg)

    def test_legacy_dropout_state_is_restored_even_on_error(self):
        agent = TDMPC2.__new__(TDMPC2)
        torch.nn.Module.__init__(agent)
        dropout = torch.nn.Dropout(0.01)
        agent._legacy_dropouts = (dropout,)
        for training, inplace in ((False, False), (True, False), (False, True)):
            dropout.train(training)
            dropout.inplace = inplace
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with agent._legacy_critic_dropout():
                    self.assertTrue(dropout.training)
                    self.assertTrue(dropout.inplace)
                    raise RuntimeError("failure")
            self.assertEqual(dropout.training, training)
            self.assertEqual(dropout.inplace, inplace)


@unittest.skipUnless(torch.cuda.is_available(), "TD-MPC2 requires CUDA")
class LegacyInferenceTests(unittest.TestCase):
    def assert_traces_equal(self, first, second):
        first_actions, first_numpy, first_cpu, first_cuda = first
        second_actions, second_numpy, second_cpu, second_cuda = second
        for first_action, second_action in zip(first_actions, second_actions):
            self.assertTrue(torch.equal(first_action, second_action))
        self.assertEqual(first_numpy[0], second_numpy[0])
        np.testing.assert_array_equal(first_numpy[1], second_numpy[1])
        self.assertEqual(first_numpy[2:], second_numpy[2:])
        self.assertTrue(torch.equal(first_cpu, second_cpu))
        self.assertTrue(torch.equal(first_cuda, second_cuda))

    def test_legacy_matches_validated_diagnostic_and_rng(self):
        oracle, legacy = make_agents()
        original_pi = oracle.model.pi
        original_q = oracle.model.Q
        dropout = next(m for m in oracle.model._Qs.modules() if isinstance(m, torch.nn.Dropout))
        dropout.train(True)
        dropout.inplace = True

        def historical_pi(z, task):
            _, info = original_pi(z, task)
            return info["mean"], info

        def historical_q(z, action, task, return_type="min", target=False, detach=False):
            out = original_q(z, action, task, return_type="all", target=target, detach=detach)
            if return_type == "all":
                return out
            q1, q2 = out[np.random.choice(oracle.cfg.num_q, 2, replace=False)]
            q1 = td_math.two_hot_inv(q1, oracle.cfg)
            q2 = td_math.two_hot_inv(q2, oracle.cfg)
            return torch.minimum(q1, q2) if return_type == "min" else (q1 + q2) / 2

        def historical_elite(probabilities, temperature=1.0, dim=0):
            index = np.random.choice(np.arange(probabilities.shape[0]), p=probabilities.cpu().numpy())
            return torch.tensor(index, dtype=torch.long, device=probabilities.device)

        oracle.model.pi = historical_pi
        oracle.model.Q = historical_q
        with patch.object(td_math, "gumbel_softmax_sample", historical_elite):
            expected = trace(oracle, eval_mode=True)
        actual = trace(legacy, eval_mode=True)
        self.assert_traces_equal(expected, actual)
        self.assertTrue(torch.equal(oracle._prev_mean, legacy._prev_mean))
        self.assertFalse(legacy._legacy_dropouts[0].training)
        self.assertFalse(legacy._legacy_dropouts[0].inplace)

    def test_training_action_semantics_remain_current_after_legacy_evaluation(self):
        current, legacy = make_agents()
        legacy.act(torch.zeros(14), t0=True, eval_mode=True)
        self.assert_traces_equal(trace(current, eval_mode=False), trace(legacy, eval_mode=False))


if __name__ == "__main__":
    unittest.main()
