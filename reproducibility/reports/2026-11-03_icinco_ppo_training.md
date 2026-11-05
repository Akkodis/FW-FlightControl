# PPO: inference device and fresh seed-1 training

[Run guide and status](../README.md)

- **Device matters:** for the preserved seed-1 policy on easy references, CUDA
  inference shrinks the archive fluctuation gap from +3.73% (CPU) to -0.15%.
  Python 3.9 and 3.10 environments give byte-identical results per device.
- **Fresh training works but differs:** a full-budget fresh seed-1 run lands
  inside the archived five-seed ranges, but its fluctuation is +44% vs archived
  seed 1, so it does not reconstruct that policy. Medium/hard evaluation remains.
- **Two training entry-point repairs** were needed, for standard PPO only.

All runs: nominal atmosphere, no action regularization, `icinco2024`.

## Inference device (preserved seed 1)

| Runtime / device | Mean RMSE (rad) | Surface fluctuation (rad/sample) | Fluctuation vs archive |
|---|---:|---:|---:|
| Archived seed 1 | 0.029099766 | 0.006079950 | reference |
| `fw-310` or `fw-tdmpc2-39`, CPU | 0.029106578 | 0.006306472 | +3.726% |
| `fw-310` or `fw-tdmpc2-39`, CUDA | 0.029096860 | 0.006070544 | -0.155% |
| `fw-old`, CUDA | 0.029098084 | 0.006214391 | +2.211% |

- `fw-310`: Python 3.10.13, JSBSim 1.2.3. `fw-tdmpc2-39`: Python 3.9.18,
  JSBSim 1.2.3.dev1. Both: PyTorch 2.6.0.dev20241112, NumPy 1.24.4,
  Gymnasium 0.29.1. `fw-old` (Python 3.10.0, NumPy 1.26.4, Gymnasium 0.28.1,
  JSBSim 1.1.13) changes several dependencies at once, so it isolates nothing.
- This shows device sensitivity; the archive's exact runtime remains unknown.
  All later work uses `fw-310` on RTX A6000 CUDA.

## Fresh seed-1 training

61 updates / 749,568 transitions in about 17 minutes on an RTX A6000, with
finite weights and no out-of-bounds episode. The final model (no best-model
selection) on 25 easy-reference episodes:

| Metric | Archived seed 1 | Fresh seed 1 (CUDA) | Change |
|---|---:|---:|---:|
| Roll / pitch RMSE (rad) | 0.0346 / 0.0236 | 0.0325 / 0.0271 | -6.1% / +14.8% |
| Mean RMSE (rad) | 0.029099766 | 0.029783230 | +2.35% |
| Aileron / elevator fluctuation (rad/sample) | 0.00947 / 0.00269 | 0.01319 / 0.00432 | +39.4% / +60.3% |
| Mean surface fluctuation (rad/sample) | 0.006079950 | 0.008754006 | +43.98% |

- Both means fall inside the archived five-seed easy ranges, fixed before
  training: mean RMSE 0.028291–0.034893, fluctuation 0.000377–0.011520. These
  are min/max ranges, not confidence intervals or tolerances.
- CUDA re-evaluation is byte-identical; a CPU re-evaluation also falls inside
  both ranges.
- This is not an RNG replay of the May 2024 trainer, which drew targets
  differently.
- Checkpoint: `models/train/ppo/1/ppo_icinco2024_noactreg_1.pt` in the local
  [`fresh-seed1`](../runs/training-checkpoint-20261103/fresh-seed1/) run,
  SHA-256 `1a803ee742013974be42b67e3ae07d8e59995e625403561577f77645c22d258c`.

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

Settings come from FW-JSBGym `b55d942` (May 22, 2024). The paper prints batch
size 12,228; the code's 12,288 is kept.

</details>

## Training repairs

Two pre-existing failures blocked PPO training on current code:

| Failure | Repair |
|---|---|
| `2ca8d57` moved simulator creation to `init()`, but trainers still reset first | Initialize each vector environment once before reset |
| `abc02fe` / `943cb7c` changed `sample_targets` and moved its limits | Pass the environment/config to the initial and autoreset calls; explicit `rl.PPO` limits take precedence |

No archived YAML, PPO algorithm, reward, architecture or default simulator
behaviour changed, and default-profile evaluation stayed byte-identical.
Normalized PPO, SAC and TD3 trainers are not repaired yet.
[`train_ppo_icinco.py`](../train_ppo_icinco.py) refuses existing output
directories, checks input hashes and fails if a metric leaves its predeclared
range.

## Source pair and branches

| Repository | Validated training commit | Local `paper/icinco` candidate |
|---|---|---|
| FW-JSBGym | `052d8b434dac1a2ebee67c5b2216c341086a3a8d` | same |
| FW-FlightControl | `d8db31374e060ed8c76676679b7619d5894299cf` | `8366758504100ab3219593f62d0c699f42a7551c` (adds the report only) |

Training ran from `e0d3474` plus recorded diffs; all 265 inputs match the
committed pair. The candidates cover PPO seed 1, easy references and no wind
only; they are not release tags. Local, Git-ignored evidence:
[`training-checkpoint-20261103`](../runs/training-checkpoint-20261103/)
(device runs, smoke test, fresh run and re-evaluations).
