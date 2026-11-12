# PPO: inference device and fresh seed-1 training

[Run guide and status](../README.md)

- **Device:** CUDA shrinks the preserved seed-1 fluctuation gap from +3.73%
  (CPU) to -0.15%.
- **Fresh training:** inside archived five-seed ranges but +44% fluctuation vs
  archived seed 1; not a reconstruction. Medium/hard evaluation remains.

All runs: nominal atmosphere, no action regularization, `icinco2024`.

## Inference device (preserved seed 1)

| Runtime / device | Mean RMSE (rad) | Surface fluctuation (rad/sample) | Fluctuation vs archive |
|---|---:|---:|---:|
| Archived seed 1 | 0.029099766 | 0.006079950 | reference |
| `fw-310` or `fw-tdmpc2-39`, CPU | 0.029106578 | 0.006306472 | +3.726% |
| `fw-310` or `fw-tdmpc2-39`, CUDA | 0.029096860 | 0.006070544 | -0.155% |
| `fw-old`, CUDA | 0.029098084 | 0.006214391 | +2.211% |

- `fw-310`: Python 3.10.13, JSBSim 1.2.3, NumPy 2.2.6 imported (1.24.4 in
  metadata). `fw-tdmpc2-39`: Python 3.9.18, NumPy 1.24.4, JSBSim 1.2.3.dev1.
  Both: PyTorch 2.6.0.dev20241112, Gymnasium 0.29.1. `fw-old`: Python 3.10.0,
  NumPy 1.26.4, Gymnasium 0.28.1, JSBSim 1.1.13 (confounded).
- Archive runtime unknown. Later work used `fw-310` CUDA;
  [disturbance reproduction requires JSBSim 1.1.13](../README.md).

## Fresh seed-1 training

61 updates / 749,568 transitions, 17 minutes (RTX A6000). Final model, 25 easy
episodes:

| Metric | Archived seed 1 | Fresh seed 1 (CUDA) | Change |
|---|---:|---:|---:|
| Roll / pitch RMSE (rad) | 0.0346 / 0.0236 | 0.0325 / 0.0271 | -6.1% / +14.8% |
| Mean RMSE (rad) | 0.029099766 | 0.029783230 | +2.35% |
| Aileron / elevator fluctuation (rad/sample) | 0.00947 / 0.00269 | 0.01319 / 0.00432 | +39.4% / +60.3% |
| Mean surface fluctuation (rad/sample) | 0.006079950 | 0.008754006 | +43.98% |

- Inside predeclared archived five-seed easy min/max ranges (RMSE
  0.028291–0.034893, fluctuation 0.000377–0.011520), on CPU too.
- Not an RNG replay of the May 2024 trainer.
- Checkpoint `models/train/ppo/1/ppo_icinco2024_noactreg_1.pt` in local
  [`fresh-seed1`](../runs/training-checkpoint-20261103/fresh-seed1/), SHA-256
  `1a803ee742013974be42b67e3ae07d8e59995e625403561577f77645c22d258c`.

<details>
<summary>Training settings</summary>

| Setting | Value |
|---|---|
| Task | `ACBohnNoVaIErr-v0`, `icinco2024`, 14 observations, two actions; 100 Hz, 20 s episodes |
| Network | Separate actor/critic MLPs, two 64-unit Tanh layers |
| Targets | Uniform roll ±60°, pitch ±30° |
| Budget | 750,000 requested; 749,568 run in complete rollouts of 6 environments × 2,048 steps = 12,288 |
| Optimization | LR 3e-4 annealed, 10 epochs, 32 minibatches of 384, Adam eps 1e-5 |
| PPO | gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, value coefficient 0.5 |
| Regularization | None: CAPS coefficients zero, action-variation reward off |
| Periodic evaluation | Every 37,500 steps, including update zero |

From FW-JSBGym `b55d942` (May 22, 2024). Paper prints 12,228; code's 12,288
kept.

</details>

## Training repairs

| Failure | Repair |
|---|---|
| `2ca8d57` moved simulator creation to `init()`; trainers reset first | Initialize each environment before reset |
| `abc02fe` / `943cb7c` changed `sample_targets` and its limits | Pass environment/config to initial and autoreset calls; explicit `rl.PPO` limits win |

- No archived YAML, algorithm, reward or default behaviour changed.
- Normalized PPO, SAC and TD3 trainers are not repaired.
- The former `train_ppo_icinco.py` launcher (available in Git history)
  checked input hashes and failed if a metric left its predeclared range.

## Source pair and branches

| Repository | Validated training commit | Local `paper/icinco` candidate |
|---|---|---|
| FW-JSBGym | `052d8b434dac1a2ebee67c5b2216c341086a3a8d` | same |
| FW-FlightControl | `d8db31374e060ed8c76676679b7619d5894299cf` | `8366758504100ab3219593f62d0c699f42a7551c` (adds the report only) |

Ran from `e0d3474` plus recorded diffs. Candidates cover PPO seed 1, easy, no
wind only. Evidence:
[`training-checkpoint-20261103`](../runs/training-checkpoint-20261103/).
