# ICINCO conclusions checklist (Phase 1b acceptance gate)

[Run guide and status](./README.md)

The gate is: **every conclusion below still holds on the simplified stack; numbers may vary.**
Each table follows the paper and shows `paper → simplified harness`, five seeds
(1, 2164, 2989, 4508, 6930), mean ± SEM over seeds as the paper aggregates them
([details](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#how-the-paper-aggregates-seeds)).
PID is deterministic and has no SEM.

- Units: 1e-2. RMSE is roll/pitch RMSE (rad); ActFluct is roll/pitch fluctuation (rad/sample).
- "Nominal" is the mean of the easy and medium references; "Hard" is the hard reference.
- **Paper** values are the paper's published numbers, or, where the paper tables are not reproduced
  in the repo (action-regulation variants, per-level disturbances), the archived per-seed CSVs
  aggregated the same way (identical to the paper wherever both exist).
- **Simplified harness** values come from `python -m fw_flightcontrol.eval.attitude_control
  experiment=icinco2024 ...` with the `eval.yaml` config groups, evaluating the five seed checkpoints
  of each variant with `evaluation.checkpoint_directory=<variant dir>`, 25 episodes per target, JSBSim 1.1.13,
  TD-MPC2 with `rl.inference_mode=current`. Raw outputs and `run_five_seeds.sh`:
  `runs/2026-11-24_phase1b_five_seeds/` (Git-ignored, 25 runs).
- Disturbance runs use the disturbance-trained checkpoints (`models/icinco/<algo>/{turbonly,gustonly}`).
- The harness reruns at the end of Phase 1b reuse these tables: replace the right-hand value.

## Table 1 — Baselines, nominal and hard (paper Tables 6–7)

| Agent | RMSE nominal | ActFluct nominal | RMSE hard | ActFluct hard |
|---|---|---|---|---|
| PID | 4.20 → 4.17 | 0.07 → 0.06 | 20.10 → 20.03 | 0.17 → 0.16 |
| TD-MPC2 | 3.81 ± 0.10 → 3.81 ± 0.10 | 4.37 ± 0.48 → 3.99 ± 0.59 | 9.10 ± 0.69 → 9.43 ± 0.94 | 6.02 ± 0.59 → 5.70 ± 0.79 |
| PPO | 5.29 ± 0.12 → 5.29 ± 0.12 | 1.98 ± 0.58 → 1.98 ± 0.58 | 15.78 ± 2.48 → 15.78 ± 2.48 | 5.54 ± 0.93 → 5.53 ± 0.93 |
| SAC | 6.64 ± 0.83 → 6.64 ± 0.83 | 10.07 ± 1.12 → 10.08 ± 1.12 | 11.15 ± 0.65 → 11.15 ± 0.65 | 12.00 ± 0.71 → 12.00 ± 0.71 |

Checks (orderings, best to worst, in the simplified harness):

| Check | Paper | Simplified | Holds |
|---|---|---|---|
| Nominal RMSE | TD-MPC2 < PID < PPO < SAC | TD-MPC2 3.81 < PID 4.17 < PPO 5.29 < SAC 6.64 | ✅ |
| Hard RMSE | TD-MPC2 < SAC < PPO < PID | TD-MPC2 9.43 < SAC 11.15 < PPO 15.78 < PID 20.03 | ✅ |
| Nominal ActFluct | PID < PPO < TD-MPC2 < SAC | PID 0.06 < PPO 1.98 < TD-MPC2 3.99 < SAC 10.08 | ✅ |
| Hard ActFluct | PID < PPO < TD-MPC2 < SAC | PID 0.16 < PPO 5.53 < TD-MPC2 5.70 < SAC 12.00 | ✅ |

Accepted differences:

- TD-MPC2 ActFluct is lower than the paper (3.99 vs 4.37, 5.70 vs 6.02) because of the
  accepted *current* inference (`rl.inference_mode=current`, `icinco2024` is opt-in); orderings
  are unchanged.
- TD-MPC2 hard RMSE moved from 9.10 to 9.43 (within the SEM of both, 0.69 and 0.94).
- PID differs by 0.03 / 0.07 RMSE from the published values: the generic harness feeds the new
  reset observation to the first action (see the
  [harness guide](../fw_flightcontrol/eval/attitude_control/README.md)).

## Table 2 — Action-regulation variants (CAPS / AVP)

| Variant | RMSE nominal | ActFluct nominal | RMSE hard | ActFluct hard |
|---|---|---|---|---|
| PPO CAPS | 5.05 ± 0.15 → 5.05 ± 0.15 | 1.47 ± 0.27 → 1.47 ± 0.26 | 19.17 ± 4.68 → 19.17 ± 4.68 | 4.56 ± 0.60 → 4.55 ± 0.61 |
| PPO AVP | 5.78 ± 0.21 → 5.78 ± 0.21 | 13.78 ± 3.16 → 13.78 ± 3.16 | 13.80 ± 0.82 → 13.80 ± 0.82 | 13.48 ± 2.50 → 13.48 ± 2.50 |
| SAC CAPS | 6.79 ± 0.74 → 6.79 ± 0.74 | 0.53 ± 0.27 → 0.53 ± 0.27 | 12.28 ± 1.17 → 12.28 ± 1.17 | 1.44 ± 0.55 → 1.44 ± 0.55 |
| SAC AVP | 6.54 ± 0.75 → 6.54 ± 0.75 | 0.50 ± 0.18 → 0.50 ± 0.18 | 15.14 ± 3.39 → 15.14 ± 3.39 | 1.20 ± 0.32 → 1.20 ± 0.32 |
| TD-MPC2 AVP | 3.58 ± 0.01 → 3.60 ± 0.01 | 0.54 ± 0.02 → 0.68 ± 0.01 | 8.09 ± 0.03 → 8.13 ± 0.04 | 0.60 ± 0.02 → 0.69 ± 0.02 |

Checks (compare with the plain agent of Table 1):

| Check | Paper | Simplified | Holds |
|---|---|---|---|
| PPO CAPS lowers ActFluct and nominal RMSE, raises hard RMSE | 1.98 → 1.47, 5.29 → 5.05, 15.78 → 19.17 | same | ✅ |
| PPO AVP raises ActFluct and gives no RMSE improvement | ActFluct 1.98 → 13.78 | same | ✅ |
| SAC CAPS / AVP cut ActFluct by about 10× at a small RMSE cost | 10.07 → 0.53 / 0.50 | 10.08 → 0.53 / 0.50 | ✅ |
| TD-MPC2 AVP cuts ActFluct by about 6× without worse RMSE | 3.81 → 3.58, 4.37 → 0.54 | 3.81 → 3.60, 3.99 → 0.68 | ✅ |

TD-MPC2 AVP ActFluct is 0.14 / 0.09 above the paper (still far below plain TD-MPC2); accepted
current-inference difference.

## Table 3 — Turbulence (paper Figure 2, nominal references)

Light steady wind is present at every level including `off`.

**RMSE**

| Agent | off | light | moderate | severe |
|---|---|---|---|---|
| PID | 4.93 → 4.90 | 5.25 → 5.31 | 5.69 → 5.66 | 7.25 → 7.20 |
| PPO CAPS | 5.68 ± 0.21 → 5.68 ± 0.21 | 5.98 ± 0.24 → 6.05 ± 0.24 | 6.22 ± 0.24 → 6.22 ± 0.24 | 7.28 ± 0.27 → 7.27 ± 0.27 |
| SAC CAPS | 8.24 ± 1.07 → 8.24 ± 1.07 | 8.27 ± 0.87 → 8.40 ± 0.86 | 8.52 ± 0.80 → 8.52 ± 0.80 | 9.45 ± 0.68 → 9.40 ± 0.68 |
| TD-MPC2 AVP | 5.13 ± 0.52 → 5.47 ± 0.82 | 6.12 ± 0.69 → 6.63 ± 1.07 | 7.23 ± 0.91 → 6.86 ± 1.06 | 6.78 ± 0.36 → 6.94 ± 0.59 |

**ActFluct**

| Agent | off | light | moderate | severe |
|---|---|---|---|---|
| PID | 0.08 → 0.07 | 0.19 → 0.19 | 0.25 → 0.25 | 0.40 → 0.40 |
| PPO CAPS | 1.87 ± 0.10 → 1.87 ± 0.10 | 2.42 ± 0.11 → 2.44 ± 0.11 | 2.59 ± 0.12 → 2.58 ± 0.12 | 3.04 ± 0.17 → 3.04 ± 0.17 |
| SAC CAPS | 0.98 ± 0.30 → 0.98 ± 0.30 | 1.84 ± 0.23 → 1.84 ± 0.23 | 2.14 ± 0.20 → 2.14 ± 0.20 | 2.89 ± 0.18 → 2.88 ± 0.18 |
| TD-MPC2 AVP | 0.65 ± 0.03 → 0.72 ± 0.03 | 0.77 ± 0.04 → 0.82 ± 0.04 | 0.86 ± 0.06 → 0.89 ± 0.05 | 1.02 ± 0.02 → 1.03 ± 0.02 |

## Table 4 — Gusts (paper Figure 3, nominal references)

**RMSE**

| Agent | off | light | moderate | severe |
|---|---|---|---|---|
| PID | 4.20 → 4.17 | 4.85 → 4.82 | 7.78 → 7.76 | 19.11 → 19.06 |
| PPO CAPS | 5.63 ± 0.24 → 5.63 ± 0.24 | 6.04 ± 0.22 → 6.04 ± 0.22 | 8.38 ± 0.18 → 8.38 ± 0.18 | 19.35 ± 0.74 → 19.35 ± 0.74 |
| SAC CAPS | 22.05 ± 8.78 → 22.05 ± 8.78 | 22.29 ± 8.20 → 22.29 ± 8.20 | 25.12 ± 7.75 → 25.12 ± 7.75 | 34.90 ± 6.50 → 34.89 ± 6.51 |
| TD-MPC2 AVP | 5.16 ± 0.21 → 5.42 ± 0.21 | 7.55 ± 0.33 → 7.83 ± 0.43 | 11.65 ± 0.38 → 12.67 ± 0.34 | 23.86 ± 1.55 → 26.15 ± 1.59 |

**ActFluct**

| Agent | off | light | moderate | severe |
|---|---|---|---|---|
| PID | 0.07 → 0.06 | 0.12 → 0.12 | 0.20 → 0.19 | 0.28 → 0.27 |
| PPO CAPS | 1.47 ± 0.23 → 1.47 ± 0.23 | 1.68 ± 0.21 → 1.68 ± 0.21 | 1.74 ± 0.24 → 1.74 ± 0.24 | 1.89 ± 0.24 → 1.89 ± 0.24 |
| SAC CAPS | 2.19 ± 0.49 → 2.19 ± 0.49 | 2.56 ± 0.46 → 2.56 ± 0.46 | 2.80 ± 0.43 → 2.80 ± 0.43 | 2.97 ± 0.41 → 2.97 ± 0.41 |
| TD-MPC2 AVP | 2.26 ± 0.20 → 1.85 ± 0.22 | 2.75 ± 0.21 → 2.31 ± 0.21 | 2.87 ± 0.17 → 2.36 ± 0.15 | 3.07 ± 0.14 → 2.63 ± 0.11 |

## Table 5 — Robustness conclusions

| Check | Paper | Simplified | Holds |
|---|---|---|---|
| Turbulence RMSE, `off`: PID best, SAC worst | PID 4.93 < TD-MPC2 5.13 < PPO 5.68 < SAC 8.24 | PID 4.90 < TD-MPC2 5.47 < PPO 5.68 < SAC 8.24 | ✅ |
| Turbulence RMSE, `moderate` | PID < PPO < TD-MPC2 < SAC | PID 5.66 < PPO 6.22 < TD-MPC2 6.86 < SAC 8.52 | ✅ |
| Turbulence RMSE, `severe`: TD-MPC2 below PID and PPO | TD-MPC2 6.78 < PID 7.25 < PPO 7.28 < SAC 9.45 | TD-MPC2 6.94 < PID 7.20 < PPO 7.27 < SAC 9.40 | ✅ |
| Gust RMSE, every level: PID < PPO < TD-MPC2 < SAC (`off`: TD-MPC2 below PPO) | moderate 7.78 < 8.38 < 11.65 < 25.12 | moderate 7.76 < 8.38 < 12.67 < 25.12; severe 19.06 < 19.35 < 26.15 < 34.89 | ✅ |
| PID has the lowest ActFluct at every level, both conditions | yes | yes | ✅ |
| TD-MPC2 AVP has lower ActFluct than PPO CAPS and SAC CAPS under turbulence, at every level | 0.65–1.02 vs 1.87–3.04 and 0.98–2.89 | 0.72–1.03 vs 1.87–3.04 and 0.98–2.88 | ✅ |
| Gust RMSE rises monotonically with severity for every agent | yes | yes | ✅ |
| Turbulence ActFluct rises monotonically with severity for every agent | yes | yes | ✅ |

Accepted differences: TD-MPC2 gust RMSE is 0.3 to 2.3 above the paper and its ActFluct 0.4 to 0.5 below
it (current inference); TD-MPC2 turbulence `moderate` RMSE is 6.86 vs 7.23, with overlapping SEMs. In
every case the orderings above are unchanged. PPO and SAC match the paper to the second decimal
(except 0.01 to 0.13 at turbulence light, a JSBSim 1.1.13 residual).

## Safety and scope

| Check | Result |
|---|---|
| Out-of-bounds episodes | 1 of 25 runs: SAC CAPS gusts, severe, hard target (1 episode in 125, excluded from RMSE). The nominal references and all other runs have none. Not in the paper's nominal figures. |
| Disturbance checkpoints | `turbonly`/`gustonly` checkpoints used for all disturbance rows |
| JSBSim version | 1.1.13 for every run |
| TD-MPC2 inference | `rl.inference_mode=current` (default); `icinco2024` stays an opt-in setting |
