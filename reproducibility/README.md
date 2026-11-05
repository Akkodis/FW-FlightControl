# Reproducibility guide

**Status (2026-11-05):**

- The nominal rows of ICINCO Tables 6–7 reproduce for PID, PPO and SAC.
- TD-MPC2 matches the nominal RMSE with lower fluctuation, kept by author
  decision.
- Action-regulation rows were evaluated for PPO, SAC and TD-MPC2 AVP over five
  seeds; residuals are accepted as measured, not as a general tolerance.
- Measured nominal residuals for all five seeds are
  [accepted by the author](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#author-decisions),
  not as a general tolerance.
- Not yet covered: turbulence and gusts, fresh TD-MPC2 training and SNCS. The
  full papers are **not** reproduced yet.

| Report | Contents |
|---|---|
| [ICINCO results, five seeds](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md) | Nominal and action-regulation tables vs archive/current, per-seed residuals, author decisions |
| [PPO seed-1 drift](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md) | Why current code drifted, the `icinco2024` profile, June 12, 2024 source check |
| [PPO training](./reports/2026-11-03_icinco_ppo_training.md) | CPU vs CUDA inference, fresh seed-1 training, trainer repairs |
| [TD-MPC2 inference ablation](./reports/2026-11-04_icinco_tdmpc2_inference_ablation.md) | Why current TD-MPC2 fluctuates less; decision to keep it |

## Run the validated slice

Use the existing **`fw-310`** environment and an **RTX GPU through CUDA**. Do
not upgrade the shared environment or fall back to CPU (JSBSim physics still
runs on CPU). Keep FW-JSBGym and FW-FlightControl as sibling checkouts and run
from the FlightControl root. The PPO evaluator defaults to `current` and CPU,
so **pass `--sim-profile icinco2024` and `--device cuda`**. Every output
directory must be new; archives are never overwritten.

### Preserved seed-1 PPO evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py \
    --sim-profile icinco2024 --device cuda --reference-set easy \
    reproducibility/runs/preserved-easy-cuda-new
```

- Use `--reference-set medium` or `hard`, each in its own output directory.
- Add `--checkpoint /path/to/fresh.pt` to compare another checkpoint with
  archived seed 1.
- The launcher checks device, profile, finite metrics, episode and observation
  counts, channels and input hashes.
- **Exit zero means the evaluation completed, not that paper scores matched.**

### Fresh seed-1 PPO training, then evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_ppo_icinco.py --device cuda \
    reproducibility/runs/fresh-seed1-new
```

- Selects `icinco2024`, disables remote tracking, trains 749,568 transitions
  (the 750,000 request in full rollouts), then evaluates the final checkpoint
  on easy references.
- The full run fails if either metric leaves the archived five-seed range;
  [results](./reports/2026-11-03_icinco_ppo_training.md).
- `--smoke` runs 4,096 transitions, which is not a paper-score check.
- Direct Hydra entry points take `+env.jsbsim.sim_profile=icinco2024`; the `+`
  leaves protected YAML unchanged. Not every legacy trainer works yet.

### Fresh seed-1 TD-MPC2 training, then evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_tdmpc2_icinco.py \
    reproducibility/runs/fresh-tdmpc2-seed1-new
```

- **What it runs:** current TD-MPC2, kept by
  [author decision](./reports/2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04)
  with no legacy inference mode. Model size 5, `icinco2024`, no action
  regularization, CUDA only, tracking off.
- **Training:** 750,001 transitions (the inclusive loop for 750,000) with
  periodic evaluation every 37,500 steps.
- **Final evaluation:** the final checkpoint on 25 easy-reference episodes,
  evaluation seed 10.
- **Comparisons:** the launcher reports the seed-1 difference and the position
  within the five-seed archive range descriptively. **A completed run is not an
  acceptance of its scores.**
- `--smoke`: 2,011 transitions, shorter warmup, no evaluation.
- **Status:** six regression tests and the CUDA smoke pass.
  - The full run was cancelled at the author's request on Nov 4 after about
    24,000 steps. No checkpoint was saved.
  - Restart only on request: from scratch, in a new directory, in byobu.
  - Local evidence: [smoke](./runs/2026-11-04_tdmpc2_training_smoke_verified/)
    and [cancelled run](./runs/2026-11-04_tdmpc2_seed1_full_cuda/).
- **Submodule fixes**, on branch `revamp/reproducibility-first` of
  Dobid/tdmpc2. Algorithm, planner, networks and protected YAML are unchanged.
  - `96fe24e`: saves checkpoints when W&B tracking is disabled.
  - `bce5911`: training fails instead of reporting success when a checkpoint
    save fails.

## Common preserved-checkpoint protocol

| Item | Value |
|---|---|
| Checkpoints | Preserved ICINCO checkpoints without action regularization, e.g. [PPO seed 1](../fw_flightcontrol/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt) |
| Evaluation RNG seed | 10; distinct from the training seed |
| Task | `ACBohnNoVaIErr-v0`, X8, 14 observations, two actions |
| PPO actor | Two 64-unit Tanh hidden layers; deterministic mean |
| Episodes | 25 targets × 2,000 steps; 100 Hz agent/FDM; 20 s each |
| References | Fixed `simple_easy.npy`, `simple_medium.npy`, `simple_hard.npy` in [targets](../fw_flightcontrol/eval/attitude_control/targets/) |
| Simulator seeds | `jsbsim_seeds.npy`; the 26th entry serves the evaluator's final reset |
| Atmosphere | No wind, turbulence, gusts or randomized FDM; severity `off` |
| PPO wrappers | Episode statistics and action clipping; no observation/reward normalization |
| Execution | `fw-310`, RTX A6000 CUDA (PID: CPU), one OpenMP/MKL thread |

Observations: roll, pitch, airspeed (kph), p/q/r, roll/pitch errors,
alpha/beta, aileron/elevator commands, roll/pitch integral errors. The
integrals accumulate only with `icinco2024`; the default `current` profile
resets them on every target-setter call.

- **Mean RMSE (rad):** each axis's RMSE over all 50,000 raw errors (columns
  6/7), then the mean of the two; not a pooled RMS.
- **Surface fluctuation, ActFluct (rad/sample):** mean absolute difference of
  successive actual aileron/elevator positions, 1,999 per episode, averaged
  over episodes and both surfaces. It is not command fluctuation. The launcher
  reports `paper_surface_fluct` separately from legacy exports that may
  include normalized throttle.
- **Seed aggregation:** mean and SEM over the five seeds per reference level;
  see [how the paper aggregates](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#how-the-paper-aggregates-seeds).
  A single seed is not a paper value.

## Evidence and checks

- **Launcher records:** invocation, source commits, dirty diffs, package
  versions, input and source hashes, profile, Hydra config, log, raw metrics
  and comparison. Training also keeps the final model and TensorBoard curves.
- **What a clone contains:** `runs/` is Git-ignored, so run paths in reports
  are local evidence. The reports, [`ppo_diagnosis.csv`](./ppo_diagnosis.csv)
  and [`tdmpc2_inference_ablation.csv`](./tdmpc2_inference_ablation.csv) are
  tracked. Legacy evaluators do not export full raw trajectories.
- **Protected inputs:** [`historical_inputs.sha256`](./historical_inputs.sha256)
  covers 204 inputs, with paths relative to `FW-RL`:
  - 118 paper checkpoints: 70 ICINCO and 48 SNCS.
  - Four attitude and five waypoint reference/seed arrays.
  - FlightControl and X8 trim YAML, and the simulator XML assets.
  - Both paper PDFs, package setup files and JSBGym environment manifests.

  The snapshot detects accidental changes; it does not prove numerical
  reproducibility.
- **Revamp starting commits,** both clean `main` (not paper-era):
  - FlightControl `bffe6c33e7268d39811a652a83218e20ae2e9384`
  - JSBGym `571a52fe02ec3f7fc78968a076fa68a536be1455`

```bash
(cd .. && sha256sum --check FW-FlightControl/reproducibility/historical_inputs.sha256)
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    -m unittest discover -s reproducibility -p 'test_*.py'
(cd ../FW-JSBGym && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../FW-FlightControl \
    /opt/anaconda3/envs/fw-310/bin/python -m unittest discover -s tests)
```

To run the unmodified June 12, 2024 source instead of today's profile, see the
[historical replay](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#evidence-and-replay).
