# ICINCO PPO seed-1 drift: diagnosis and fix

[Run guide and status](../README.md)

- **Cause:** three post-paper simulator changes made current code drift from the
  archive (preserved PPO seed 1, easy references: mean RMSE +87%). They are
  integral-error resets, the X8 drag table, and throttle in the fluctuation
  metric. Initial conditions and surface-sampling timing refine the match.
- **Fix:** the opt-in [`icinco2024` profile](#the-fix-the-icinco2024-simulator-profile)
  in FW-JSBGym restores that behaviour. The default `current` profile is unchanged.
- **Check:** the June 12, 2024 source and current code + `icinco2024` give
  byte-identical metric CSVs on easy, medium and hard references (CUDA). The
  remaining archive residual (at most 0.16%) was accepted on Nov 3.
- **SNCS:** waypoint tasks do not depend on the integral-reset change.

## Controlled restorations (Oct 30, CPU)

Preserved PPO seed 1, easy references, nominal atmosphere, 25 episodes /
50,000 steps, `fw-310` on CPU with one thread. Each row restores simulator
behaviour in a separate worktree; policy, references, seeds and evaluator stay
fixed.

| Simulator variant | Mean RMSE (rad) | Surface fluctuation (rad/sample) |
|---|---:|---:|
| Archived seed-1 result | 0.029099766 | 0.006079950 |
| Current code | 0.054548910 | 0.000401277 |
| Historical drag only | 0.053859490 | 0.001099867 |
| Historical integral resets only | 0.028797068 | 0.002585542 |
| Historical integral resets + drag | 0.029137757 | 0.006261801 |
| ... + historical initial conditions | 0.029106578 | 0.006299480 |
| ... + historical post-step surface sampling | 0.029106578 | 0.006306472 |

The current and fully restored rows each repeated exactly. Current code's low
fluctuation is not an improvement: tracking is worse and the metric changed
meaning. The remaining +3.7% fluctuation is a CPU effect; on CUDA it is -0.15%
([device comparison](./2026-11-03_icinco_ppo_training.md#inference-device-preserved-seed-1)).
Full-precision values: [`ppo_diagnosis.csv`](../ppo_diagnosis.csv).

## Causes

| FW-JSBGym change | Effect |
|---|---|
| `95d4b531` (2024-12-06): integral errors reset on every target-setter call | The evaluator sets the same target every step, so the integrals hold only one step's error. Restoring conditional resets recovers most tracking accuracy. |
| `97e356e07` (2025-06-30): new X8 angle-of-attack drag table | With integrals restored, the old table raises fluctuation from 0.0026 to 0.0063, near the archive. The newer model is not shown to be wrong. |
| `98580b4` (2025-02-13): throttle added to the surface history | `avg_fcs_fluct` averages three channels, mixing radians with normalized throttle: 0.0043 instead of the paper's two-surface 0.0063. |
| `ad64129`, `76dc60d`: new initial conditions | Restoring 16.97 m/s at lat/lon 0 (instead of 16.67 m/s over France) cuts the RMSE gap from +0.131% to +0.023%. Speed and location effects were not separated. |
| Surface sampling moved before the action | Restoring post-FDM-step sampling leaves RMSE unchanged and moves fluctuation slightly. |

These are compatibility requirements for reproducing ICINCO, not reasons to
revert the newer dynamics.

## The fix: the `icinco2024` simulator profile

Defined in FW-JSBGym
[`sim_profiles.py`](https://github.com/Akkodis/FW-JSBGym/blob/96e9cda0e0ca3d5eace0ca2ca9b4675e3d2b9332/fw_jsbgym/envs/sim_profiles.py).
Select it with `--sim-profile icinco2024` (reproduction launchers) or
`+env.jsbsim.sim_profile=icinco2024` (Hydra); protected YAML stays unedited.

| Behaviour | `current` (default) | `icinco2024` |
|---|---|---|
| X8 model | `x8.xml` | `x8_icinco2024.xml`, drag table before `97e356e` |
| Initial conditions | 16.67 m/s, France | 16.97 m/s, lat/lon 0 |
| Integral reset on a target-setter call | Both axes, every call | Only the axis whose target changed |
| Surface-history channels | Aileron, elevator, throttle | Aileron, elevator |
| Surface-history sampling | Before applying the action | After the FDM step |
| Gust start steps | 250 | 500 and 1500 (paper Section 6.2.2) |

- Both profiles reset the integrals at episode start. Only `icinco2024` matches
  ICINCO Eq. 14: `I_next = I + error * 0.01`, reset at episode start or target
  change.
- **Validated:** the default profile's metric CSV is byte-identical to the
  pre-profile code. `icinco2024` equals the fully restored diagnostic case and
  repeats exactly. 14 profile tests (selection, asset hashes, resets, history
  channels, sampling, gust timing) and 14 launcher tests pass. Implemented in
  JSBGym `052d8b4` and FlightControl `e0d3474`.
- **Not restored:** the base task's airspeed-PI reset on target change. ICINCO
  targets change only at episode boundaries, right after that reset, so it
  should have no effect there.
- **Not validated:** gust-condition scores. Gust timing is unit-tested only.

## Integral errors and SNCS

Integral state after injecting `[1.25, -2.5]` and then:

| Operation | `current` | `icinco2024` and June 12 code |
|---|---|---|
| Same target again | `[0, 0]` | `[1.25, -2.5]` |
| Change roll target only | `[0, 0]` | `[0, -2.5]` |
| Change pitch target only | `[0, 0]` | `[1.25, 0]` |
| Reset episode | `[0, 0]` | `[0, 0]` |

SNCS waypoint tasks (`CourseAltTracking`, `CourseAltNoVaTracking`) do not use
the attitude-integral class. For SS-Vanilla, SS-ActPos and PI-Throttle,
toggling only the reset flag leaves observations, rewards, flags, positions and
actuators exactly equal (seed 1, two ten-step episodes with a waypoint change).
This does not validate SNCS. SNCS Tables 9–10 reuse ICINCO attitude results,
which need `icinco2024` semantics when re-evaluated.

## June 12, 2024 source and author acceptance

The author recalls FW-JSBGym `733a2a8bffe9deb7e0f9ca6e202b7da132d1b168` and
FW-FlightControl `85767615f3206f92c0dd997b21071d5615da8d56` (both 2024-06-12)
as the ICINCO submission commits. They hold the same 70 ICINCO checkpoints and
4 reference/seed arrays as today, and all 203 archived result CSVs match after
CRLF/LF normalization. In that code, PPO and SAC live in `FW-JSBGym/jsbgym`
and TD-MPC2 in `FW-FlightControl/tdmpc2`.

The unmodified June 12 PPO evaluator and current code + `icinco2024` (both
`fw-310`, RTX A6000 CUDA) produce byte-identical metric CSVs, so one rerun
column covers both; an easy repeat was also identical:

| References | Archived RMSE | Rerun RMSE | Δ | Archived fluctuation | Rerun fluctuation | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 0.029099766 | 0.029096860 | -0.010% | 0.006079950 | 0.006070544 | -0.155% |
| Medium | 0.073654940 | 0.073655420 | +0.001% | 0.043597637 | 0.043554799 | -0.098% |
| Hard | 0.228911640 | 0.228910620 | -0.0004% | 0.068063825 | 0.068104233 | +0.059% |

**Author decision (2026-11-03):** these measured residuals are accepted for
preserved PPO seed 1, nominal atmosphere, CUDA. The cause is unknown; JSBSim or
PyTorch updates between the ICINCO and SNCS papers are a recalled hypothesis.
This is not a tolerance for other runs.

## Evidence and replay

- Tracked: [`ppo_diagnosis.csv`](../ppo_diagnosis.csv) and
  [`icinco_ppo_diagnostic.patch`](../icinco_ppo_diagnostic.patch), the
  restoration patch for JSBGym `571a52f` (an experiment artifact, not the
  production fix). To replay it, apply it to a detached `571a52f` worktree and
  pass that worktree to `evaluate_ppo_nominal.py --simulator-root`.
- Local, Git-ignored runs:
  [`icinco-ppo-seed1-easy-fw310-20261030`](../runs/icinco-ppo-seed1-easy-fw310-20261030/) and
  [`ppo-diagnosis-20261030`](../runs/ppo-diagnosis-20261030/) (diagnosis, FlightControl `81661f6`, JSBGym `571a52f`);
  [`sim-profile-20261103`](../runs/sim-profile-20261103/) (profile validation);
  [`submission-pair-20261103`](../runs/submission-pair-20261103/) (June 12
  worktrees, `run_evaluation.py` helper and runs).

<details>
<summary>Replay the unmodified June 12 PPO evaluation</summary>

With the retained local helper, from `FW-RL`, into a new directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    FW-FlightControl/reproducibility/runs/submission-pair-20261103/run_evaluation.py \
    easy easy-cuda-new
```

In a fresh workspace, from `FW-RL` (use `simple_medium` or `simple_hard` in
separate directories for the other reference sets):

```bash
BASE="$PWD/FW-FlightControl/reproducibility/runs/submission-pair-replay"
git -C FW-JSBGym worktree add --detach "$BASE/FW-JSBGym" \
    733a2a8bffe9deb7e0f9ca6e202b7da132d1b168
git -C FW-FlightControl worktree add --detach "$BASE/FW-FlightControl" \
    85767615f3206f92c0dd997b21071d5615da8d56
mkdir -p "$BASE/easy-cuda/eval"
ln -s "$BASE/FW-JSBGym/jsbgym/eval/refs" "$BASE/easy-cuda/eval/refs"
cd "$BASE/easy-cuda"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg
export PYTHONPATH="$BASE/FW-JSBGym:$BASE/FW-JSBGym/jsbgym:$BASE/FW-FlightControl"
/opt/anaconda3/envs/fw-310/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA required; do not fall back to CPU"' &&
/opt/anaconda3/envs/fw-310/bin/python "$BASE/FW-JSBGym/jsbgym/ppo_eval_simple.py" \
    "model_path=$BASE/FW-JSBGym/jsbgym/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt" \
    ref_file=simple_easy res_file=ppo_seed1_easy \
    env.jsbsim.debug=false env.jsbsim.eval_sim_options.atmosphere.severity=off \
    hydra.run.dir=hydra hydra.job.chdir=false
```

The historical evaluator selects CUDA itself when available; check that its
log says `Using Device: cuda`.

</details>
