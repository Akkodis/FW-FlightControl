# PPO training and inference-device experiments

[Investigation overview](./README.md) | [Run commands](../README.md#run-the-validated-slice)

**Conclusions:** CUDA substantially reduces the preserved-policy fluctuation
gap. Fresh seed-1 PPO training completes and falls inside predeclared archive
ranges, but does **not** reconstruct the archived seed-1 policy.
All experiments here are nominal atmosphere, without action regularization.

## Preserved checkpoint: inference device matters

Only runtime/device varied; policy, evaluator, profile, references, seeds and
metric definitions stayed fixed. Each easy-reference run completed
25 episodes / 50,000 steps without out-of-bounds retries.

| Runtime / device | Mean RMSE (rad) | Surface fluctuation (rad/sample) | Fluctuation vs archive |
|---|---:|---:|---:|
| Archived seed 1 | 0.029099766 | 0.006079950355 | reference |
| `fw-310`, CPU | 0.029106578 | 0.006306471580 | +3.726% |
| `fw-tdmpc2-39`, CPU | 0.029106578 | 0.006306471580 | +3.726% |
| `fw-310`, CUDA | 0.029096860 | 0.006070544465 | -0.155% |
| `fw-tdmpc2-39`, CUDA | 0.029096860 | 0.006070544465 | -0.155% |
| `fw-old`, CUDA | 0.029098084 | 0.006214390815 | +2.211% |

`fw-310` and `fw-tdmpc2-39` produce byte-identical CSVs **per device** despite
Python 3.10/3.9 and JSBSim 1.2.3/1.2.3.dev1 differences. They share PyTorch,
NumPy and Gymnasium versions; see the [runtime inventory](./2026-10-30_baseline_inventory.md#runtime-and-import-checks).
`fw-old` changes several dependencies at once, so it cannot isolate JSBSim.

This establishes device sensitivity, not bitwise portability or the archive's
exact runtime. The paper reports RTX A6000 training; these CUDA evaluations
also use RTX A6000. The archive's exact evaluation device remains unverified.
The author later [accepted the small CUDA residuals](./2026-11-03_icinco_submission_pair_evaluation.md#author-acceptance-2026-11-03);
no numerical correction was applied.

## Full-run result

Fresh PPO seed 1 completed **61 updates / 749,568 transitions**, 20 periodic
evaluation checkpoints, and about 17 minutes of training. Model parameters
and TensorBoard scalars were finite; no training episode went out of bounds.
The final model, not a selected best model, was evaluated on 25 easy targets:

| Metric | Archived seed 1 | Fresh seed 1, CUDA | Change vs seed 1 |
|---|---:|---:|---:|
| Roll RMSE (rad) | 0.034604420 | 0.032488886 | -6.11% |
| Pitch RMSE (rad) | 0.023595111 | 0.027077574 | +14.76% |
| Mean RMSE (rad) | 0.029099766 | 0.029783230 | +2.35% |
| Aileron fluctuation (rad/sample) | 0.009466399785 | 0.013191098174 | +39.35% |
| Elevator fluctuation (rad/sample) | 0.002693500925 | 0.004316913609 | +60.27% |
| Mean surface fluctuation (rad/sample) | 0.006079950355 | 0.008754005891 | +43.98% |

### Predeclared fresh-training comparison

The five archived easy/no-atmosphere seeds supplied these ranges **before**
training; neither was adjusted afterwards:

| Aggregate metric | Archive minimum | Archive maximum | Fresh result |
|---|---:|---:|---|
| Mean RMSE | 0.028291479 | 0.034892620 | Inside |
| Mean two-surface fluctuation | 0.000376523485 | 0.011519680748 | Inside |

These are descriptive min/max ranges, **not confidence intervals**. Passing
them does not erase the +43.98% seed-1 fluctuation difference or reproduce the
paper's easy-plus-medium aggregate. The acceptance of small preserved-policy
residuals does not extend to this fresh-training difference.

CUDA reevaluation was byte-identical, with no out-of-bounds retries. Historical
CPU reevaluation also passed: mean RMSE `0.029781040`, fluctuation `0.00884751`,
inside both ranges. Future runs use CUDA.

Final checkpoint: `models/train/ppo/1/ppo_icinco2024_noactreg_1.pt` inside
[`fresh-seed1`](../runs/training-checkpoint-20261103/fresh-seed1/).
SHA-256: `1a803ee742013974be42b67e3ae07d8e59995e625403561577f77645c22d258c`.
The archived policy was neither overwritten nor used to initialize training.

## Training protocol and provenance

| Setting | Value |
|---|---|
| Seed/task/profile | 1; `ACBohnNoVaIErr-v0`; `icinco2024`; 14 observations, two actions |
| Simulator | 100 Hz, 20 s episodes, no atmosphere/randomized FDM |
| Network | Separate actor/critic MLPs, two 64-unit Tanh layers |
| Targets | Uniform roll +/-60 degrees, pitch +/-30 degrees |
| Requested/actual budget | 750,000 / 749,568 transitions; legacy complete-rollout floor division |
| Rollout | Six environments x 2,048 steps = 12,288 |
| Optimization | LR 3e-4 annealed, 10 epochs, 32 minibatches of 384, Adam eps 1e-5 |
| PPO | gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, value coefficient 0.5 |
| Regularization | CAPS temporal/spatial/preactivation coefficients zero; action-variation reward off |
| Periodic evaluation | Existing 37,500-step cadence, including update zero |
| Device/output | RTX A6000 CUDA; final model, TensorBoard, Hydra config and provenance; tracking/final diagnostic plot disabled |

Source `FW-JSBGym b55d942` (May 22) supplies the training limits and PPO
hyperparameters. The paper prints batch size 12,228, but six times 2,048 is
12,288: the code's calculation is retained. The later 55/28-degree values in
`f71acbe` concern evaluation, not training.

**Not an exact historical RNG replay:** the current sampler draws independent
initial targets and a batch before selecting an autoreset target. May's trainer
shared the initial pair, drew scalar autoreset targets and consumed unused
reference-sequence draws. Matching seed/distribution does not imply matching
samples or tensor bytes.

[`train_ppo_icinco.py`](../train_ppo_icinco.py) supplies explicit overrides,
records/checks source and input hashes, refuses existing output directories,
evaluates the final checkpoint and fails if a metric falls outside its range.
Its `--smoke` run covers 4,096 transitions, not paper performance.

## PPO entry-point repairs

The profile smoke exposed two pre-existing failures, also present with
unmodified JSBGym `571a52f`:

| Failure | Standard PPO repair |
|---|---|
| `2ca8d57` moved simulator creation to `init()`; trainers still reset first | Initialize each unwrapped vector environment once before reset |
| `abc02fe` / `943cb7c` changed `sample_targets` and moved limits | Pass environment/config to initial and autoreset calls; merge root limits, with explicit `rl.PPO` limits taking precedence |

No archived YAML, PPO algorithm, reward, architecture, normalization wrappers
or default simulator behaviour changed. The integration test exercises two
optimizer updates and multiple autoresets under both profiles, finite weights,
changed policy standard deviations and periodic historical-profile evaluation.
Normalized PPO, SAC and TD3 remain outside this repair; TD3 also uses the stale
pre-Hydra environment API.

Validation at the training checkpoint: **17 FlightControl tests**, **14 profile
tests**, **204/204 protected inputs** and **265/265 training-input hashes**.
Default-profile evaluation stayed byte-identical, including explicit
`--checkpoint` selection.

## Preserved checkpoint: medium and hard references

The later `--reference-set easy|medium|hard` extension completed medium/hard
CUDA runs (25 episodes / 50,000 steps each, no retries). Its default easy/CPU
regression was byte-identical, and 18 focused FlightControl tests passed.
The authoritative numbers now live in the
[three-way results table](./2026-11-03_icinco_submission_pair_evaluation.md#comparison-with-archived-seed-1-results).
This extension evaluated the **archived** policy, not the fresh one.

## Evidence and branch boundary

| Local, Git-ignored evidence | Contents |
|---|---|
| [`training-checkpoint-20261103`](../runs/training-checkpoint-20261103/) | `runtime-39`, `runtime-310-cuda`, `runtime-39-cuda`, `runtime-old-cuda`; `training-smoke`, `fresh-seed1`, `fresh-repeat-cuda`, `fresh-cpu`, `default-regression` |
| [`sim-profile-20261103/icinco2024`](../runs/sim-profile-20261103/icinco2024/) | Original `fw-310` CPU comparison |
| [`followup-20261103`](../runs/followup-20261103/) | `easy-regression`, `medium-cuda`, `hard-cuda` |

Exact validated training source pair:

| Repository | Commit |
|---|---|
| FW-JSBGym | `052d8b434dac1a2ebee67c5b2216c341086a3a8d` |
| FW-FlightControl | `d8db31374e060ed8c76676679b7619d5894299cf` |

Training began before the FlightControl commit: provenance records `e0d3474`
plus diff/source hashes; all 265 inputs were checked against this committed pair.
Initial local `paper/icinco` candidates point to JSBGym `052d8b4` and
FlightControl `8366758504100ab3219593f62d0c699f42a7551c` (report-only successor
to the validated code). Both original `main` branches remain untouched.
These are limited PPO/easy/no-wind candidates, not tagged full-paper releases;
future refactors need preserved-policy and fresh-training checks.
