# Reproducibility guide

> [!WARNING]
> **Use JSBSim 1.1.13 for ICINCO disturbance results.** Under 1.2.3,
> `icinco2024` misses archived turbulence (PID severe/easy RMSE 0.051151 →
> 0.041369 rad); 1.1.13 matches all four PID levels. Do not switch versions in
> a shared environment. Nominal and action-regulation reruns under 1.1.13 are
> [reported here](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#jsbsim-1113-nominal-revalidation);
> measured residuals are accepted. A code tag does not pin the runtime.

**Status (2026-11-06):**

- Nominal (ICINCO Tables 6–7): PID, PPO and SAC reproduce; TD-MPC2 matches
  RMSE with lower fluctuation, kept by author decision.
- Five-seed nominal and action-regulation residuals under 1.2.3:
  [author-accepted](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#author-decisions)
  as measured, not as a tolerance; 1.1.13 nominal/action-regulation residuals
  for all agents are also accepted as measured.
- Turbulence/gusts: PID and five-seed PPO/SAC/TD-MPC2 done under 1.1.13;
  residuals accepted. The TD-MPC2 turbulence-off ranking reversal traces to
  inference changes in one run; production-mode sweeps match the earlier diagnostics.
- TD-MPC2 has an opt-in `icinco2024` inference mode (below); historical
  training is unvalidated.
- The 1.1.13 nominal revalidation is complete. In the locked runtime, PID,
  PPO, SAC and TD-MPC2 pilots match their `fw-310` controls exactly (PID/PPO/SAC
  ran before the TorchRL swap below; they do not import TorchRL).
- The lock uses stable `tensordict==0.6.1` + `torchrl==0.6.0` instead of the
  unavailable 2025.1.1 nightlies (TD-MPC2 pilots identical; TorchRL's unused
  C++ extension does not load). Three wheels are still built locally.
- Not covered: fresh TD-MPC2 training and SNCS. The papers are **not** fully
  reproduced.

| Report | Contents |
|---|---|
| [ICINCO results, five seeds](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md) | Nominal, action-regulation and disturbance results; author decisions |
| [PPO seed-1 drift](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md) | Why current code drifted, the `icinco2024` profile, June 12, 2024 source check |
| [PPO training](./reports/2026-11-03_icinco_ppo_training.md) | CPU vs CUDA inference, fresh seed-1 training, trainer repairs |
| [TD-MPC2 inference ablation](./reports/2026-11-04_icinco_tdmpc2_inference_ablation.md) | Inference differences, opt-in mode and training assessment |

## Run the validated slice

- Nominal only: **`fw-310`**, JSBSim 1.2.3, **RTX GPU via CUDA**. Do not
  upgrade it or fall back to CPU (JSBSim physics runs on CPU).
- Run from FlightControl, with FW-JSBGym as a sibling checkout.
- PPO evaluator defaults to `current` and CPU: **pass
  `--sim-profile icinco2024 --device cuda`**.
- Always use a new output directory; archives are never overwritten.

### Candidate 1.1.13 runtime (local only)

The Linux x86-64 [Conda](./locks/icinco2024-linux-64-conda.lock) and
[pip](./locks/icinco2024-linux-64-pip.lock) locks capture the tested candidate,
not the original paper environment or a portable release. They require the
exact, Git-ignored wheelhouse in
`runs/2026-11-07_icinco2024_stable_torchrl/runtime_candidate/wheels/`:
`antlr4-python3-runtime`, `ffmpeg` and `moviepy` were built locally from PyPI
source distributions, so their hashes are local. Preserve it.

From `FW-FlightControl`, choose a **new** prefix and the preserved wheelhouse:

```bash
PREFIX=/absolute/path/to/new-icinco2024-env
WHEELHOUSE=/absolute/path/to/preserved/wheels
conda create --yes --prefix "$PREFIX" --file reproducibility/locks/icinco2024-linux-64-conda.lock
PYTHONNOUSERSITE=1 "$PREFIX/bin/python" -m pip install --no-index --no-deps --require-hashes \
    --find-links "$WHEELHOUSE" -r reproducibility/locks/icinco2024-linux-64-pip.lock
PYTHONNOUSERSITE=1 "$PREFIX/bin/python" -m pip check
```

Keep `PYTHONNOUSERSITE=1` when running; the source checkouts are separate from
the lock. Only the ICINCO attitude-evaluation slice is being validated.

### Preserved seed-1 PPO evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py \
    --sim-profile icinco2024 --device cuda --reference-set easy \
    reproducibility/runs/preserved-easy-cuda-new
```

- `--reference-set medium` or `hard`: separate directories.
- `--checkpoint /path/to/fresh.pt`: compare another checkpoint with seed 1.
- Validates device, profile, metrics, counts and hashes. **Exit zero means
  completed, not matched.**

### Fresh seed-1 PPO training, then evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_ppo_icinco.py --device cuda \
    reproducibility/runs/fresh-seed1-new
```

- `icinco2024`, no tracking; 749,568 transitions (750,000 in full rollouts),
  then easy-reference evaluation. Fails if either metric leaves the archived
  five-seed range ([results](./reports/2026-11-03_icinco_ppo_training.md)).
- `--smoke`: 4,096 transitions, not a paper-score check.
- Hydra entry points take `+env.jsbsim.sim_profile=icinco2024` (`+` keeps
  protected YAML unchanged). Not every legacy trainer works yet.

### Fresh seed-1 TD-MPC2 training, then evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_tdmpc2_icinco.py \
    reproducibility/runs/fresh-tdmpc2-seed1-new
```

- Current TD-MPC2 ([author decision](./reports/2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04)),
  size 5, `icinco2024`, no action regularization, CUDA only, no tracking.
- 750,001 transitions (inclusive loop for 750,000), evaluation every 37,500
  steps; final checkpoint on 25 easy episodes, seed 10.
- Compares with seed 1 and the five-seed range descriptively; **completion does
  not accept scores.**
- `--smoke`: 2,011 transitions, shorter warmup, no evaluation.
- Six regression tests and CUDA smoke pass. Full run cancelled Nov 4 at
  ~24,000 steps, no checkpoint; restart only on request, fresh, in a new
  directory, in byobu. Evidence: [smoke](./runs/2026-11-04_tdmpc2_training_smoke_verified/),
  [cancelled run](./runs/2026-11-04_tdmpc2_seed1_full_cuda/).
- Submodule fixes on `revamp/reproducibility-first` of Dobid/tdmpc2:
  `96fe24e` saves checkpoints with W&B disabled; `bce5911` fails training when
  a checkpoint save fails.

### Paper-era TD-MPC2 inference

Pass `+rl.inference_mode=icinco2024` to the
[TD-MPC2 attitude evaluator](../fw_flightcontrol/eval/attitude_control/tdmpc2_eval_simple.py)
(`current` is the default; the `+` leaves protected YAML unchanged).

- Restores policy means for planner trajectories and terminal values, NumPy
  critic and elite sampling, and active critic dropout.
- Sets no profile or dependency: add `+env.jsbsim.sim_profile=icinco2024` and,
  for disturbances, JSBSim 1.1.13.
- Only single-task, state-based MPC with an encoder, five critics and dropout
  0.01; compilation, episodic/model-free modes, MPPI action-fluctuation
  planning, a decoder and Gaussian dynamics must be off.
- Affects only `act(..., eval_mode=True)`, including periodic evaluations,
  which share RNG streams with training, so later training can change.
  Training actions and updates stay current: **not historical training.**

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

- **Observations:** roll, pitch, airspeed (kph), p/q/r, roll/pitch errors,
  alpha/beta, aileron/elevator commands, roll/pitch integral errors. Integrals
  accumulate only under `icinco2024`; `current` resets them per target.
- **Mean RMSE (rad):** mean of per-axis RMSEs over 50,000 raw errors (columns
  6/7); not a pooled RMS.
- **ActFluct (rad/sample):** mean absolute change of actual aileron/elevator
  positions (1,999 per episode), averaged over episodes and surfaces; not
  commands. Reported as `paper_surface_fluct`; legacy exports may include
  throttle.
- **Seeds:** five-seed mean and SEM per level
  ([details](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#how-the-paper-aggregates-seeds));
  one seed is not a paper value.

## Evidence and checks

- **Launchers record** invocation, commits, dirty diffs, package versions,
  module paths, hashes, profile, Hydra config, log, metrics and comparison;
  training adds the final model and TensorBoard curves. Distribution metadata
  is separate: in `fw-310` NumPy imports as 2.2.6 but metadata says 1.24.4.
- **`runs/` is Git-ignored**; run paths in reports are local. Tracked:
  reports, [`ppo_diagnosis.csv`](./ppo_diagnosis.csv), [TD-MPC2 ablation](./tdmpc2_inference_ablation.csv),
  [disturbance aggregates](./icinco_disturbance_aggregates.csv) and
  [historical-inference per-seed results](./tdmpc2_disturbance_historical_inference.csv).
  Legacy evaluators do not export full raw trajectories.
- **Protected inputs:** [`historical_inputs.sha256`](./historical_inputs.sha256)
  covers 204 inputs (paths relative to `FW-RL`): 118 paper checkpoints (70
  ICINCO, 48 SNCS); four attitude and five waypoint reference/seed arrays;
  FlightControl and X8 trim YAML and simulator XML assets; both paper PDFs,
  package setup files and JSBGym environment manifests. Detects accidental
  changes only.
- **Revamp starting commits** (clean `main`, not paper-era): FlightControl
  `bffe6c33e7268d39811a652a83218e20ae2e9384`, JSBGym
  `571a52fe02ec3f7fc78968a076fa68a536be1455`.

```bash
(cd .. && sha256sum --check FW-FlightControl/reproducibility/historical_inputs.sha256)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../FW-JSBGym /opt/anaconda3/envs/fw-310/bin/python \
    -m unittest discover -s reproducibility -p 'test_*.py'
(cd ../FW-JSBGym && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../FW-FlightControl \
    /opt/anaconda3/envs/fw-310/bin/python -m unittest discover -s tests)
```

For the unmodified June 12, 2024 source, see the
[historical replay](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#evidence-and-replay).
