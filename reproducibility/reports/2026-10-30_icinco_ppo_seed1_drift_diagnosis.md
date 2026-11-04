# ICINCO PPO drift: causes and controlled experiments

[Investigation overview](./README.md) | [Common protocol](../README.md#common-preserved-checkpoint-protocol)

**Conclusion:** three post-paper simulator changes explain the large drift:
integral resets, X8 drag, and the fluctuation metric. Initial conditions and
surface-sampling timing refine the match.

This report records the **October 30 CPU diagnosis**, with November 3
clarifications. Its +3.726% fluctuation residual is not the final CUDA result;
the later [nominal seed-1 CUDA residuals were accepted](./2026-11-03_icinco_submission_pair_evaluation.md#author-acceptance-2026-11-03).
That does not accept the larger CPU difference.

## Controlled results

One preserved PPO seed-1 policy, easy references, nominal atmosphere:
25 episodes / 50,000 steps in `fw-310`, CPU, one OpenMP/MKL thread.
Evaluation sources: FlightControl `81661f6`, JSBGym `571a52f`.
Each restoration used a separate detached simulator worktree; policy,
references, RNG seeds, runtime and evaluator were held fixed. No archived
inputs changed (204/204 hashes matched).

| Simulator variant | Mean RMSE (rad) | Paper surface fluctuation (rad/sample) |
|---|---:|---:|
| Archived seed-1 result | 0.029099766 | 0.006079950 |
| Current code | 0.054548910 | 0.000401277 |
| Historical drag only | 0.053859490 | 0.001099867 |
| Historical integral resets only | 0.028797068 | 0.002585542 |
| Historical integral resets + drag | 0.029137757 | 0.006261801 |
| ... + historical initial conditions | 0.029106578 | 0.006299480 |
| ... + historical post-step surface sampling | 0.029106578 | 0.006306472 |

The unchanged baseline and fully restored case each repeated exactly.
Baseline mean RMSE is **+87.45%** versus archive. Its apparently low fluctuation
is not evidence of improvement: tracking worsened, and the legacy export also
changed meaning. Full-precision per-axis and legacy-average values remain in
[`ppo_diagnosis.csv`](../ppo_diagnosis.csv).

## Why it drifted

| Change in FW-JSBGym | Effect established by the experiments |
|---|---|
| `95d4b531` (2024-12-06): unconditional integral resets | Repeated same-target setter calls erase accumulation. Restoring conditional resets alone recovers most tracking accuracy. |
| `97e356e07` (2025-06-30): X8 angle-of-attack drag table | With integrals restored, historical drag raises fluctuation from 0.002586 to 0.006262, near archive. The newer model is not thereby proved wrong. |
| `98580b4` (2025-02-13): throttle added to surface history | `avg_fcs_fluct` becomes a three-channel mean mixing radians and normalized throttle. The fully restored diagnostic export is 0.004255815, but the paper's two-surface value is 0.006306472. |
| Initial conditions differ from `1072782` | Historical 16.97 m/s and lat/lon 0 replace 16.67 m/s and France; RMSE gap falls from +0.131% to +0.023%. Speed/location effects were not separated. |
| Surface sampling differs from `1072782` | Sampling after the FDM step rather than before the action leaves RMSE unchanged and moves fluctuation slightly away from archive. |

These are historical compatibility requirements, not reasons to globally
revert newer dynamics. The production solution is the explicit
[`icinco2024` profile](./2026-11-03_icinco2024_sim_profile.md).

## Integral errors and SNCS

**The restored behaviour, not the faulty default, matches ICINCO Eq. 14:**
`I_next = I + error * 0.01`, reset at episode start or target change.

The attitude evaluator/trainer calls `set_target_state` each control step,
usually with the **same target**. A setter call is not a target change.
In `current`, the final two observations (indices 12/13) therefore hold only
one step's contribution. In `icinco2024`, errors accumulate throughout the
episode until a real change/reset.

A real-environment probe injected integrals `[1.25, -2.5]`:

| Operation | `current` | Historical / `icinco2024` |
|---|---|---|
| Same target again | `[0, 0]` | `[1.25, -2.5]` |
| Change roll target only | `[0, 0]` | `[0, -2.5]` |
| Change pitch target only | `[0, 0]` | `[1.25, 0]` |
| Reset episode | `[0, 0]` within `1e-20` | `[0, 0]` within `1e-20` |

The paper does not specify whether a one-axis change resets both integrals.
Historical code resets only the changed axis; this distinction is immaterial
here because ICINCO targets change at episode boundaries, where both reset.
Simulator defaults were not changed globally.

| SNCS check | Finding |
|---|---|
| Waypoint MDPs | `CourseAltTracking` / `CourseAltNoVaTracking` do not inherit the affected attitude-integral class. SS-Vanilla, SS-ActPos and PI-Throttle each have 19 state values without roll/pitch integral errors. |
| Isolated runtime probe | Each variant: seed 1, fixed actions, no atmosphere, two ten-step episodes at 100 Hz, waypoint change halfway. Toggle only `reset_integral_errors_every_target_call`; observations (raw/normalized), rewards, flags, ENU and actuator positions remain exactly equal. |
| Separate integrators | Airspeed PI and waypoint PID controllers have their own state; PI-Throttle computed its own throttle during the probe. |
| Reused attitude results | SNCS Eq. 16 still has attitude integral errors. Tables 9/10, reproduced from ICINCO, require historical semantics when re-evaluated. |

This proves independence from **one reset flag**, not equivalence of the full
profiles or reproduction of SNCS policies/dynamics.

## Author-supplied submission candidates (2026-11-03)

The later [source audit and historical execution](./2026-11-03_icinco_submission_pair_evaluation.md#source-audit)
use JSBGym `733a2a8` and FlightControl `8576761` from June 12, 2024.
They confirm the restored semantics and reproduce the profile's CUDA metric
CSVs exactly. Full commit IDs, artifact counts and the acceptance decision live
in that report rather than being duplicated here.

## Evidence and reproduction

| Evidence | Location |
|---|---|
| All diagnostic values | [`ppo_diagnosis.csv`](../ppo_diagnosis.csv), tracked |
| Isolated restoration patch | [`icinco_ppo_diagnostic.patch`](../icinco_ppo_diagnostic.patch), tracked experiment artifact, not a production fix |
| Original/repeat evaluation | [`icinco-ppo-seed1-easy-fw310-20261030`](../runs/icinco-ppo-seed1-easy-fw310-20261030/), local/ignored |
| Per-variant runs and integral probe | [`ppo-diagnosis-20261030`](../runs/ppo-diagnosis-20261030/), local/ignored |

The patch applies to JSBGym `571a52f`. To rerun that restoration on the required
**CUDA** device, from FlightControl with new directory names:

```bash
git -C ../FW-JSBGym worktree add --detach \
    "$PWD/reproducibility/runs/diagnostic-source-new" \
    571a52fe02ec3f7fc78968a076fa68a536be1455
git -C reproducibility/runs/diagnostic-source-new apply \
    "$PWD/reproducibility/icinco_ppo_diagnostic.patch"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py --device cuda \
    --simulator-root reproducibility/runs/diagnostic-source-new \
    reproducibility/runs/diagnostic-cuda-new
```

Use `git apply --include=<path>` for an individual restoration. This command
replays the simulator patch on CUDA; it is **not** expected to reproduce the
CPU table bit-for-bit. Routine evaluations should instead use the profile and
[current run guide](../README.md).

Gusts at steps 500/1500 (paper Section 6.2.2 and `1072782`) versus the later
single gust at 250 were another source finding. They do not affect these
nominal runs; the profile restores the timing, but gust-score validation remains.
