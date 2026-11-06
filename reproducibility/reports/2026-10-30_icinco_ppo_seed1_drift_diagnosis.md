# ICINCO PPO seed-1 drift: diagnosis and fix

[Run guide and status](../README.md)

- **Cause:** post-paper simulator changes (integral resets, X8 drag table,
  throttle in the fluctuation metric) raised preserved PPO seed-1 easy RMSE
  by 87%.
- **Fix:** opt-in [`icinco2024` profile](#the-fix-the-icinco2024-simulator-profile);
  default `current` unchanged.
- **Check:** matches the June 12, 2024 source byte-for-byte (CUDA); archive
  residual (at most 0.16%) accepted Nov 3.

## Controlled restorations (Oct 30, CPU)

Easy references, nominal atmosphere, 25 episodes, `fw-310`, one CPU thread;
only simulator behaviour varies.

| Simulator variant | Mean RMSE (rad) | Surface fluctuation (rad/sample) |
|---|---:|---:|
| Archived seed-1 result | 0.029099766 | 0.006079950 |
| Current code | 0.054548910 | 0.000401277 |
| Historical drag only | 0.053859490 | 0.001099867 |
| Historical integral resets only | 0.028797068 | 0.002585542 |
| Historical integral resets + drag | 0.029137757 | 0.006261801 |
| ... + historical initial conditions | 0.029106578 | 0.006299480 |
| ... + historical post-step surface sampling | 0.029106578 | 0.006306472 |

- Current code's low fluctuation is not an improvement: the metric changed.
- Remaining +3.7% is a CPU effect; CUDA gives -0.15%
  ([device comparison](./2026-11-03_icinco_ppo_training.md#inference-device-preserved-seed-1)).
- Full precision: [`ppo_diagnosis.csv`](../ppo_diagnosis.csv).

## Causes

| FW-JSBGym change | Effect |
|---|---|
| `95d4b531` (2024-12-06): integrals reset on every target-setter call | Called every step, so integrals hold one step's error; main RMSE cause. |
| `97e356e07` (2025-06-30): new X8 drag table | Old table raises fluctuation 0.0026 to 0.0063; new model not shown wrong. |
| `98580b4` (2025-02-13): throttle in surface history | Mixed units: 0.0043 vs two-surface 0.0063. |
| `ad64129`, `76dc60d`: new initial conditions | 16.97 m/s at lat/lon 0 cuts RMSE gap +0.131% to +0.023%. |
| Surface sampling moved before the action | Minor fluctuation change. |

Compatibility requirements, not reasons to revert newer dynamics.

## The fix: the `icinco2024` simulator profile

[`sim_profiles.py`](https://github.com/Akkodis/FW-JSBGym/blob/96e9cda0e0ca3d5eace0ca2ca9b4675e3d2b9332/fw_jsbgym/envs/sim_profiles.py);
select with `--sim-profile icinco2024` or `+env.jsbsim.sim_profile=icinco2024`.

| Behaviour | `current` (default) | `icinco2024` |
|---|---|---|
| X8 model | `x8.xml` | `x8_icinco2024.xml`, drag table before `97e356e` |
| Initial conditions | 16.67 m/s, France | 16.97 m/s, lat/lon 0 |
| Integral reset on a target-setter call | Both axes, every call | Only the axis whose target changed |
| Surface-history channels | Aileron, elevator, throttle | Aileron, elevator |
| Surface-history sampling | Before applying the action | After the FDM step |
| Gust start steps | 250 | 500 and 1500 (paper Section 6.2.2) |

- Only `icinco2024` matches ICINCO Eq. 14 (`I_next = I + error * 0.01`).
- **Validated:** default unchanged byte-for-byte; `icinco2024` equals the full
  restoration; 28 tests pass. JSBGym `052d8b4`, FlightControl `e0d3474`.
- **Not restored:** airspeed-PI reset on target change (no ICINCO effect).
- **Not validated:** gust scores; gust timing unit-tested only.

## Integral errors and SNCS

Integrals after injecting `[1.25, -2.5]`:

| Operation | `current` | `icinco2024` and June 12 code |
|---|---|---|
| Same target again | `[0, 0]` | `[1.25, -2.5]` |
| Change roll target only | `[0, 0]` | `[0, -2.5]` |
| Change pitch target only | `[0, 0]` | `[1.25, 0]` |
| Reset episode | `[0, 0]` | `[0, 0]` |

SNCS waypoint tasks do not use these integrals; toggling the reset leaves
their outputs identical. SNCS Tables 9–10 reuse ICINCO attitude results, which
need `icinco2024`.

## June 12, 2024 source and author acceptance

Author-recalled submission commits: FW-JSBGym
`733a2a8bffe9deb7e0f9ca6e202b7da132d1b168`, FW-FlightControl
`85767615f3206f92c0dd997b21071d5615da8d56`. Checkpoints, references and all
203 archived CSVs match today's.

June 12 evaluator and current + `icinco2024` (CUDA) give identical CSVs:

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

- [`icinco_ppo_diagnostic.patch`](../icinco_ppo_diagnostic.patch): diagnostic
  patch for JSBGym `571a52f`; pass its worktree to `--simulator-root`.
- Local runs:
  [`icinco-ppo-seed1-easy-fw310-20261030`](../runs/icinco-ppo-seed1-easy-fw310-20261030/),
  [`ppo-diagnosis-20261030`](../runs/ppo-diagnosis-20261030/) (FlightControl `81661f6`),
  [`sim-profile-20261103`](../runs/sim-profile-20261103/),
  [`submission-pair-20261103`](../runs/submission-pair-20261103/).

<details>
<summary>Replay the unmodified June 12 PPO evaluation</summary>

Local helper, from `FW-RL`:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    FW-FlightControl/reproducibility/runs/submission-pair-20261103/run_evaluation.py \
    easy easy-cuda-new
```

Fresh workspace (`simple_medium`/`simple_hard` for other sets):

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

Check the log says `Using Device: cuda`.

</details>
