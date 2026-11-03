# ICINCO PPO training repair and CPU/CUDA discrepancy

This continues the [simulator-profile checkpoint](./2026-11-03_icinco2024_sim_profile.md).
Scope: PPO without action regularization, seed 1, easy references, no atmosphere.
It is not a reproduction of the whole paper.

## Preserved checkpoint: inference device matters

The evaluator, policy, references, seeds, profile and metric definitions were
held fixed. Only the installed runtime and inference device were varied:

| Runtime / device | Mean RMSE (rad) | Surface fluctuation (rad/sample) | Fluctuation vs archive |
|---|---:|---:|---:|
| Archived seed 1 | 0.029099766 | 0.006079950355 | reference |
| `fw-310`, CPU | 0.029106578 | 0.006306471580 | +3.726% |
| `fw-tdmpc2-39`, CPU | 0.029106578 | 0.006306471580 | +3.726% |
| `fw-310`, CUDA | 0.029096860 | 0.006070544465 | -0.155% |
| `fw-tdmpc2-39`, CUDA | 0.029096860 | 0.006070544465 | -0.155% |
| `fw-old`, CUDA | 0.029098084 | 0.006214390815 | +2.211% |

Both CPU cross-runtime CSVs are byte-identical; so are both CUDA CSVs for
`fw-310` and `fw-tdmpc2-39`. Each run completed 25 episodes / 50,000 steps
without out-of-bounds retries. Their Python versions are 3.10.13 and 3.9.18,
and their JSBSim versions are 1.2.3 and 1.2.3.dev1, respectively.
They share PyTorch 2.6.0.dev20241112, NumPy 1.24.4 and Gymnasium 0.29.1.
`fw-old` also changes NumPy, Gymnasium and JSBSim, so it is a multi-factor
comparison, not proof of a JSBSim-only effect.

**Conclusion:** CPU/CUDA inference explains the dominant +3.7% residual on
this closed-loop policy. Switching Python environments does not remove it.
CUDA tracking differs from the archive by only -0.010%. The final -0.155%
fluctuation residual is still not attributed to one exact historical binary
or source change; no numerical correction has been applied. This is measured
device sensitivity, not a claim of bitwise portability or proof of the archive's
exact runtime.

The ICINCO paper reports RTX A6000 training. CUDA evaluation here also uses an
RTX A6000. The archive's exact evaluation device is not established by that
statement alone.

Local run directories are under `../runs/training-checkpoint-20261103/`:
`runtime-39`, `runtime-310-cuda`, `runtime-39-cuda`, and `runtime-old-cuda`.
The original CPU result is in `../runs/sim-profile-20261103/icinco2024/`.
Every run retains its invocation, input hashes, package versions, log, Hydra
config, raw metric CSV and comparison CSV.

## Preserved checkpoint: medium and hard references

The isolated evaluator now accepts `--reference-set easy|medium|hard`; easy
remains the default. The fixed reference arrays and seed-1 archived CSVs were
selected by matching reference-set names, and each new run records the chosen
set and its input hashes.

The archived seed-1 checkpoint was evaluated on all 25 medium and 25 hard
references with `fw-310`, CUDA inference, nominal atmosphere, and the
`icinco2024` profile. The easy CUDA row is included from the prior controlled
runtime comparison:

| Reference set | Metric | Archived | Re-evaluated | Difference |
|---|---|---:|---:|---:|
| Easy | Mean RMSE (rad) | 0.029099766 | 0.029096860 | -0.010% |
| Easy | Two-surface fluctuation | 0.006079950 | 0.006070544 | -0.155% |
| Medium | Mean RMSE (rad) | 0.073654940 | 0.073655420 | +0.0007% |
| Medium | Two-surface fluctuation | 0.043597637 | 0.043554799 | -0.098% |
| Hard | Mean RMSE (rad) | 0.228911640 | 0.228910620 | -0.0004% |
| Hard | Two-surface fluctuation | 0.068063825 | 0.068104233 | +0.059% |

All runs completed 25 episodes / 50,000 steps without out-of-bounds retries.
As a backward-compatibility check, the new launcher's default easy/CPU result
was byte-identical to the earlier `icinco2024` result. These are seed-1,
nominal-atmosphere subset comparisons, not a reproduction of the paper's
multi-seed aggregates or a tolerance claim.

The raw provenance, logs, metrics and comparisons are in the local,
Git-ignored `../runs/followup-20261103/{easy-regression,medium-cuda,hard-cuda}/`
directories. Run medium or hard with:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py --sim-profile icinco2024 \
    --device cuda --reference-set medium \
    reproducibility/runs/preserved-medium-cuda
```

## PPO entry-point repairs

In `fw_flightcontrol/train/ppo_train.py`:

1. Initialize each unwrapped vector environment exactly once before reset.
2. Pass the environment and target-sampling config to both `sample_targets`
   calls: initial targets and autoreset targets.
3. Merge the still-present root `roll_limit`/`pitch_limit` into the PPO target
   config, allowing explicit `rl.PPO` limits to take precedence. No archived
   YAML is changed.

The PPO algorithm, reward, architecture, normalization wrappers and simulator
default are otherwise unchanged. The new integration test runs two optimizer
updates and multiple autoresets for both simulator profiles, checks finite
checkpoint parameters and changed policy standard deviations, and exercises
periodic evaluation with the historical profile.

This is deliberately a PPO slice. The normalized PPO, SAC and TD3 entry points
are not repaired or certified by it.

## Training protocol and provenance

[`train_ppo_icinco.py`](../train_ppo_icinco.py) supplies explicit overrides,
disables remote tracking, refuses existing output directories, records source
hashes before training, checks them afterwards, and evaluates the final saved
checkpoint without replacing any archive. A short `--smoke` mode checks
4,096 transitions but cannot satisfy a paper-score gate.

| Setting | Selected value / evidence |
|---|---|
| Seed | 1; Python, NumPy and PyTorch seeded by the existing trainer |
| Simulator | `icinco2024`; 100 Hz, 20 s episodes, no atmosphere/randomized FDM |
| Task | `ACBohnNoVaIErr-v0`, 14-vector observation, two actions |
| Network | Separate actor/critic MLPs, two 64-unit tanh layers; unchanged agent |
| Target distribution | Uniform, roll +/-60 degrees, pitch +/-30 degrees |
| Training request | 750,000 transitions, ICINCO Table 3 |
| Rollout | Six environments x 2,048 steps = 12,288 transitions |
| Actual completed budget | 61 full rollouts = 749,568 transitions (legacy floor division) |
| Optimization | LR 3e-4 annealed, 10 epochs, 32 minibatches of 384, Adam eps 1e-5 |
| PPO | gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, value coefficient 0.5 |
| Regularization | CAPS temporal/spatial/preactivation coefficients all zero; action-variation reward off |
| Periodic evaluation | Enabled at the existing 37,500-step cadence, including update zero |
| Device | RTX A6000 CUDA for the full run; CPU for the smoke |
| Outputs | Final model, TensorBoard curves, Hydra config, provenance and metric comparison |

The May 22 source `FW-JSBGym b55d942:jsbgym/ppo_train.py` explicitly samples
60/30-degree training limits; its `jsbgym/config/rl/ppo.yaml` supplies the PPO
hyperparameters. The 55/28-degree values previously noted in
`FW-FlightControl f71acbe` concern evaluation targets, not training limits.
The paper's printed batch size 12,228 does not equal six times 2,048; code
uses 12,288. We retain the code's calculation and do not silently change the
rollout or increase the requested budget to a partial extra rollout.

**Not an exact replay of May's RNG stream:** the current shared sampler draws
independent initial targets for each environment and draws a batch before
selecting a single autoreset target. May's trainer shared the initial pair,
drew scalar autoreset targets, and consumed RNG draws for unused reference
sequences. The distribution and limits match, but identical seed numbers do
not imply identical training samples or weights. This check is for comparable
fresh-training performance, not reconstruction of the archived tensor bytes.
Online tracking and the final diagnostic plot are disabled; the final checkpoint
is selected without tuning or selecting the best evaluation.

## Predeclared fresh-training comparison

Before launching the full run, the launcher records the min/max of the five
archived easy/no-atmosphere seed results:

| Metric | Minimum | Maximum |
|---|---:|---:|
| Mean roll/pitch RMSE | 0.028291479 | 0.034892620 |
| Mean two-surface fluctuation | 0.000376523485 | 0.011519680748 |

The fresh result is checked against both intervals, unchanged after training.
These are descriptive empirical ranges, **not confidence intervals**, and
the wide fluctuation spread must not be interpreted as precise seed-1
reproduction. A result outside either interval is recorded and makes the
launcher fail explicitly. Easy references alone do not reproduce the
easy-plus-medium nominal metric in ICINCO Tables 6/7.

## Full-run result

The full-budget run is in `../runs/training-checkpoint-20261103/fresh-seed1/`.
It completed all **61 optimizer updates / 749,568 training transitions** and
20 periodic evaluation checkpoints. All TensorBoard scalars and saved model
parameters were finite. No training episode went out of bounds. The measured
training duration was approximately 17 minutes.

The final checkpoint was evaluated on the fixed 25 easy episodes / 50,000
steps, without out-of-bounds retries:

| Metric | Archived seed 1 | Fresh seed 1, CUDA | Change vs seed 1 |
|---|---:|---:|---:|
| Roll RMSE | 0.034604420 | 0.032488886 | -6.11% |
| Pitch RMSE | 0.023595111 | 0.027077574 | +14.76% |
| Mean RMSE | 0.029099766 | 0.029783230 | +2.35% |
| Aileron fluctuation | 0.009466399785 | 0.013191098174 | +39.35% |
| Elevator fluctuation | 0.002693500925 | 0.004316913609 | +60.27% |
| Mean surface fluctuation | 0.006079950355 | 0.008754005891 | +43.98% |

Both aggregate metrics are **inside the predeclared five-seed archive ranges**.
The actuator value is nevertheless substantially different from archived
seed 1; this is comparable performance across the original seed spread, not
an exact seed-1 policy reproduction.

A separate CUDA evaluation was byte-identical to the first. The same fresh
checkpoint also evaluated successfully on CPU: mean RMSE `0.029781040`,
surface fluctuation `0.00884751`, also inside both archive ranges.

Checkpoint:
`models/train/ppo/1/ppo_icinco2024_noactreg_1.pt` inside the full-run directory.
SHA-256:
`1a803ee742013974be42b67e3ae07d8e59995e625403561577f77645c22d258c`.
The original archived checkpoint was never overwritten or used to initialize
the new policy.

Additional verification:

- 17 FlightControl tests, including the two-profile training integration test;
- 14 JSBGym profile tests;
- 204/204 historical input hashes unchanged;
- all 265 inputs recorded before full training still match after the run;
- default-profile evaluation remains byte-identical to the previous baseline,
  also when selecting the archived checkpoint through the new `--checkpoint` flag.

## Branch/release boundary

Exact validated source pair (later report-only commits do not change these sources):

- FW-JSBGym: `052d8b434dac1a2ebee67c5b2216c341086a3a8d`
- FW-FlightControl: `d8db31374e060ed8c76676679b7619d5894299cf`

The full run began before the FlightControl commit was made; its provenance
therefore records `e0d3474` plus the diff and source hashes. All 265 recorded
input hashes were checked against the committed sources above.

The paired `paper/icinco` branches are maintained **reproduction candidates**
for this validated PPO/easy/no-wind slice. They can receive future refactors
only after rerunning the preserved-checkpoint and fresh-training checks.
They do not certify all ICINCO tables, algorithm families, reference sets or
gust conditions; no `icinco-repro-*` release tag is justified yet.
The enclosing `revamp_plan.md` records the exact paired commit hashes.
Both original `main` branches remain untouched.

Next independent checkpoint: evaluate other archived seeds and nominal
easy-plus-medium aggregation for PPO, then cover the other algorithm families
and atmosphere cases. The normalized PPO/SAC/TD3 repairs, SNCS and dependency
modernization remain separate work rather than being silently included in this
slice.

## Reproduce locally

From the FlightControl root, with the paired JSBGym checkout as its sibling:

```bash
# Preserved seed-1 checkpoint, paper-style device.
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py --sim-profile icinco2024 --device cuda \
    reproducibility/runs/preserved-cuda-new

# Full training followed by evaluation of the new final checkpoint.
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_ppo_icinco.py --device cuda \
    reproducibility/runs/fresh-seed1-new
```

Use `--device cpu` explicitly on a CPU-only host and retain the device in all
comparisons. No shared Conda environment was modified by this work.
