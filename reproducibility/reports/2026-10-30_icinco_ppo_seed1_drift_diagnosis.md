# ICINCO PPO seed 1: why current code no longer reproduces the archived scores

| | |
|---|---|
| Experiments run | 2026-10-30 |
| Journaled | 2026-11-03 |
| Branches | `revamp/reproducibility-first` in both repositories |
| Source commits | FW-FlightControl `81661f6`; FW-JSBGym `571a52f` (= original `main`) |
| Runtime | `fw-310`: Python 3.10.13, PyTorch 2.6.0.dev20241112, NumPy 1.24.4, Gymnasium 0.29.1, JSBSim 1.2.3 (build 1561), Hydra 1.3.2 |
| Scope | One preserved checkpoint, one reference subset, nominal atmosphere |
| Status | Large discrepancies explained; small fluctuation residual still open |

## Summary

Evaluating the preserved ICINCO PPO seed-1 checkpoint with the current code gave
a mean roll/pitch RMSE **87% higher** than the archived result and a surface
fluctuation that was **more than ten times lower**. Neither the weights nor the
Python version caused it. Controlled experiments identified three post-ICINCO
changes in FW-JSBGym:

1. **Integral-error observations were broken for this evaluator** (Dec 2024).
   They are now reset on *every* `set_target_state` call, which the evaluator
   makes at every step. The policy therefore no longer sees accumulated
   integral errors. This alone explains the tracking-error regression.
2. **The X8 drag model changed** (Jun 2025). Once the integral observations are
   restored, the earlier drag table is needed to recover the archived actuator
   fluctuation.
3. **The exported fluctuation metric changed meaning** (Feb 2025). Throttle was
   added to the surface history, so the legacy evaluator's `avg_fcs_fluct` became
   an average of aileron, elevator, *and throttle*. The paper metric averages
   only aileron and elevator.

With historical behaviour restored for these three factors, plus the historical
initial conditions and surface-sampling instant, the evaluation gives:

| Metric | Archived | Restored | Difference |
|---|---:|---:|---:|
| Mean roll/pitch RMSE | 0.029099766 | 0.029106578 | **+0.023%** |
| Paper surface fluctuation | 0.006079950 | 0.006306472 | **+3.726%** |

The restored run was repeated with byte-identical results. The remaining
fluctuation difference is **not explained yet** and has not been accepted as a
tolerance.

## Evaluated protocol

- Checkpoint
  `fw_flightcontrol/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt`
  (training seed 1; strict, weights-only load).
- Reference archive
  `fw_flightcontrol/eval/attitude_control/outputs/icinco/ppo/noactreg_noatmo/ppo_bohnmlp_noactreg_nohist_noatmo_easyrefs_1.csv`.
  A similarly named CSV outside `outputs/icinco/` has different numbers and is
  not the ICINCO result.
- Unmodified legacy evaluator `eval/attitude_control/ppo_eval_simple.py`:
  evaluator seed 10, deterministic actor mean, CPU, one OpenMP/MKL thread.
- Task `ACBohnNoVaIErr-v0`, X8, 100 Hz agent and FDM, 20 s episodes.
- All 25 targets of `targets/simple_easy.npy`; simulator seeds from
  `targets/jsbsim_seeds.npy` (26 values; the last one is used by the final reset).
- No wind, turbulence, gusts, or FDM randomization. The legacy evaluator needs
  `env.jsbsim.eval_sim_options.atmosphere.severity=off` as a scalar.
- 25 episodes, 2,000 steps each, 50,000 observations; no out-of-bounds retries
  occurred in any run.

### Metric definitions used

- **RMSE**: separate roll and pitch RMSE over all 50,000 raw error observations
  (observation columns 6 and 7), then the arithmetic mean of the two.
- **Paper surface fluctuation**: for each episode, the mean absolute difference
  between consecutive *actual* aileron and elevator positions (radians, 1,999
  differences per episode); averaged over episodes, then over the two surfaces.
  It is not command fluctuation or angular velocity.

The easy subset is only part of the paper's nominal reference set. Archived
easy and medium CSVs combined over the five seeds reproduce ICINCO Table 6
(PPO nominal `5.29 ± 0.12`, hard `15.78 ± 2.48`, ×1e-2) and Table 7's nominal
fluctuation mean `1.98` ×1e-2. This links the archived CSVs to the paper.

## Symptom

| Metric | Archived | Current code | Change |
|---|---:|---:|---:|
| Roll RMSE (rad) | 0.034604420 | 0.049398348 | +42.75% |
| Pitch RMSE (rad) | 0.023595111 | 0.059699472 | +153.02% |
| Mean RMSE (rad) | 0.029099766 | 0.054548910 | +87.45% |
| Aileron fluctuation (rad/step) | 0.009466400 | 0.000496550 | -94.75% |
| Elevator fluctuation (rad/step) | 0.002693501 | 0.000306005 | -88.64% |
| Paper surface fluctuation | 0.006079950 | 0.000401277 | -93.40% |
| Exported `avg_fcs_fluct` | 0.006079950 | 0.000334999 | not comparable |

Two independent runs of the current code produced byte-identical metric CSVs,
so the difference is deterministic and not evaluation noise.

## Method

Everything was held fixed except the simulator factor under test: the
interpreter and packages, device and threads, checkpoint, policy code,
evaluator, targets, and seeds. Each factor was restored in a separate detached
FW-JSBGym worktree under the Git-ignored `reproducibility/runs/` directory.
The working repositories, checkpoints, references, and configurations were not
modified; all 204 hashes in `historical_inputs.sha256` still match.

| Simulator variant | Mean RMSE | Paper surface fluctuation |
|---|---:|---:|
| Archived seed-1 result | 0.029099766 | 0.006079950 |
| Current code | 0.054548910 | 0.000401277 |
| Historical drag only | 0.053859490 | 0.001099867 |
| Historical integral resets only | 0.028797068 | 0.002585542 |
| Historical integral resets + drag | 0.029137757 | 0.006261801 |
| ... + historical initial conditions | 0.029106578 | 0.006299480 |
| ... + historical post-step surface sampling | 0.029106578 | 0.006306472 |

Full-precision values, including per-axis numbers and the legacy exported
average, are in [`../ppo_diagnosis.csv`](../ppo_diagnosis.csv).

## Findings

### 1. Integral-error observations: confirmed regression

- **Change:** FW-JSBGym `95d4b531` (2024-12-06),
  `ACBohnNoVaIErrTask.set_target_state` in
  `fw_jsbgym/envs/tasks/attitude_control/ac_bohn_nova.py`.
- **Before:** each integral error was reset only when its own target changed.
- **After:** both integral errors are reset on every call.
- **Why it matters:** the attitude evaluators call `set_target_state` every
  step, even for a constant target. Observations 13 and 14 then contain one
  step's error contribution instead of the accumulated history that the policy
  was trained with.
- **Evidence:** restoring only the conditional reset brings mean RMSE from
  0.05455 back to 0.02880. A probe in the real environment injected integral
  values `[1.25, -2.5]`:

  | Operation | Current | Restored |
  |---|---|---|
  | Same target again | `[0, 0]` | `[1.25, -2.5]` |
  | Roll target changes only | `[0, 0]` | `[0, -2.5]` |
  | Pitch target changes only | `[0, 0]` | `[1.25, 0]` |
  | Episode reset | `[0, 0]` (±1e-20) | `[0, 0]` (±1e-20) |
- **Paper alignment:** the restored behaviour (`icinco2024` profile) is the
  one the ICINCO paper describes: Eq. 14, `Ie = Ie + e·dt` with `dt = 0.01`,
  "reset at the beginning of each episode or target change". The current
  default deviates from the paper and from its own docstring ("If the target
  state changes, reset the integral errors"). Because every call resets the
  integral, it never builds up and only holds `e·dt` from the latest step.
  The paper does not say whether a change on one axis also resets the other
  axis's integral. That makes no difference here: ICINCO targets change only
  at episode boundaries, where `reset_target_state` zeroes both integrals.
  `current` is still the default, but this regression does not affect SNCS
  (checked 2026-11-03, see "SNCS exposure" below), so making the paper rule
  the default awaits only the author's decision.
- **SNCS exposure:** none for the published metrics.
  - The SNCS waypoint MDPs (`CourseAltTracking`, `CourseAltNoVaTracking`) have no
    integral-error observations and do not inherit from `ACBohnNoVaIErrTask`.
    All three saved SNCS variants (SS-Vanilla, SS-ActPos, PI-Throttle) take a
    19-dimensional input, which matches that integral-free state.
  - The airspeed PI controller and the waypoint PID baseline keep their own
    integrator state, independent of this code.
  - Every SNCS attitude-control table and figure is marked "reproduced from"
    ICINCO, so those results belong to the `icinco2024` semantics.

### 2. X8 drag model: confirmed dynamics change

- **Change:** FW-JSBGym `97e356e07` (2025-06-30) replaced the angle-of-attack
  drag table in `fw_jsbgym/fdm_descriptions/aircraft/x8/x8.xml`.
- **Evidence:** with integral semantics restored, the earlier table raises
  surface fluctuation from 0.002586 to 0.006262, close to the archived
  0.006080. On its own it barely affects RMSE.
- **Interpretation:** ICINCO used the earlier model. This does not show that the
  newer model is wrong; historical experiments should select the earlier model
  explicitly instead of reverting later flight-model work.

### 3. Exported fluctuation: confirmed metric-definition drift

- **Change:** FW-JSBGym `98580b4` (2025-02-13) appended throttle position to
  `fcs_pos_hist` in `fw_jsbgym/envs/jsbsim_env.py`.
- **Effect:** the attitude evaluators still compute `np.mean(fcs_fluct)` over all
  channels, so their `avg_fcs_fluct` silently became a three-channel average
  mixing radians and normalized throttle.
- **Evidence:** for the fully restored run, the legacy export reports
  0.004255815, while `(aileron + elevator) / 2` gives 0.006306472.
- **Consequence:** compare only the two-surface quantity with the paper. The
  launcher reports both values separately without rewriting legacy outputs.

### 4. Smaller protocol differences

- **Initial conditions:** the X8 initial-condition file at FW-JSBGym `1072782`
  (2024-10-04) uses 16.97 m/s and latitude/longitude 0; the current file uses
  16.67 m/s and a location in France. Restoring the file reduces the RMSE gap
  from +0.131% to +0.023%. The speed and location effects were not separated.
- **Surface-sampling instant:** at `1072782` the task recorded surface positions
  after advancing the FDM; current code records them before applying the
  action. Restoring this leaves RMSE unchanged and moves fluctuation from
  0.006299480 to 0.006306472, i.e. slightly away from the archive.

## Open questions

- **Remaining +3.7% surface fluctuation.** Candidates include other source
  differences since the paper-era commit, the JSBSim build (the archive may have
  been produced with a different JSBSim version), and numerical or device
  differences. None is established. A same-stack comparison with
  `fw-tdmpc2-39` (JSBSim 1.2.3.dev1) is a cheap next diagnostic.
- **Exact paper-era commits** are still unknown. Local Hydra snapshots support
  the selected configuration, but FW-FlightControl `f71acbe` hard-codes
  55°/28° targets, so a saved config alone does not prove the historical
  protocol.
- **Coverage:** only one seed, one algorithm, one reference subset, and the
  nominal atmosphere were evaluated. SAC, TD-MPC2, PID, medium/hard references,
  atmosphere conditions, other seeds, and fresh training remain unchecked.
- **SNCS:** the integral-reset regression does not affect SNCS (Finding 1).
  The drag table and surface-history changes still need checking against the
  SNCS waypoint checkpoints. Apply restorations only through explicit profiles,
  not globally.

## Related lead found while journaling (not yet evaluated)

The ICINCO paper (Section 6.2.2) states that gusts are triggered at timesteps
**500 and 1500**. The simulator at `1072782` did this, while current
`JSBSimEnv.step` triggers a single gust at step **250**. This does not affect the
nominal-atmosphere results above, but gust-condition reproductions must account
for it.

## Evidence and reproduction

Tracked:

- [`../ppo_diagnosis.csv`](../ppo_diagnosis.csv): one row per variant.
- [`../icinco_ppo_diagnostic.patch`](../icinco_ppo_diagnostic.patch): the four
  isolated restorations used for the final row. It is an experiment artifact,
  not a production fix, and applies to FW-JSBGym `571a52f`.
- [`../evaluate_ppo_nominal.py`](../evaluate_ppo_nominal.py): isolated evaluation
  launcher with provenance and comparison output.

Local, Git-ignored:

- `../runs/icinco-ppo-seed1-easy-fw310-20261030/`: current-code runs.
- `../runs/ppo-diagnosis-20261030/`: detached worktrees, per-variant results,
  repeats, and integral-probe JSON files.

To repeat the final diagnostic row from the FW-FlightControl root:

```bash
git -C ../FW-JSBGym worktree add --detach \
    "$PWD/reproducibility/runs/ppo-historical-sim" \
    571a52fe02ec3f7fc78968a076fa68a536be1455
git -C reproducibility/runs/ppo-historical-sim apply \
    "$PWD/reproducibility/icinco_ppo_diagnostic.patch"
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py \
    reproducibility/runs/ppo-historical-evaluation \
    --simulator-root reproducibility/runs/ppo-historical-sim
```

Use `git apply --include=<path>` to restore one factor at a time.

## Follow-up decisions

The first two decisions were implemented on 2026-11-03; see
[2026-11-03: ICINCO 2024 simulator profile](./2026-11-03_icinco2024_sim_profile.md).

- Represent the restored behaviour as an explicit, opt-in ICINCO compatibility
  profile; keep current behaviour as the default.
- Report the paper's two-surface fluctuation explicitly in reproduction tools.
- Investigate the remaining fluctuation gap before setting tolerances.
- Extend checks to other conditions, seeds, and algorithms, then fresh seed-1
  training.
