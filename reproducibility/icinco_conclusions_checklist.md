# ICINCO conclusions checklist (Phase 1b acceptance gate)

[Run guide and status](./README.md)

Fixed 2026-11-20, before the simplified stack was tested. The gate is **every item below holds
on the simplified stack; numbers may vary**, residuals are reported, and items are never relaxed
retroactively. Paper values are five-seed means (1e-2 units) from ICINCO Tables 6–7 as recorded in the
[five-seed report](./reports/2026-11-05_icinco_phase1_nominal_five_seeds.md#results-against-the-paper).
Nominal = mean of the easy and medium references; "Hard" = the hard reference.

Two tiers:

- **Paper tier (P):** conclusions the paper states, supported by five-seed evidence. They are
  checked on five seeds, not on seed 1 alone.
- **Reference tier (R):** the same comparisons on seed 1 only, checked by a rerun of the
  [Phase 1b reference set](#reference-set) on the simplified stack: signs and orderings must equal
  the frozen current-harness references below. Seed 1 can disagree with the five-seed paper
  ordering (marked *seed-sensitive*); R then guards against change, not against the paper.

## P1 — Method ranking, nominal and hard (Tables 6–7)

| Condition | Metric | Paper ordering (best → worst) |
|---|---|---|
| Nominal | RMSE | TD-MPC2 3.81 < PID 4.20 < PPO 5.29 < SAC 6.64 |
| Hard | RMSE | TD-MPC2 9.10 < SAC 11.15 < PPO 15.78 < PID 20.10 |
| Nominal | ActFluct | PID 0.07 < PPO 1.98 < TD-MPC2 4.37 < SAC 10.07 |
| Hard | ActFluct | PID 0.17 < PPO 5.54 < TD-MPC2 6.02 < SAC 12.00 |

Accepted: TD-MPC2 under current inference has lower ActFluct (3.93 / 5.63) than the paper; the
ordering above is unchanged (see P5).

## P2 — Effect of CAPS / AVP (action-regulation variants)

- PPO CAPS lowers nominal and hard ActFluct (1.98 → 1.47, 5.54 → 4.56) and nominal RMSE; hard RMSE rises.
- PPO AVP raises ActFluct (13.78 nominal) and is not an improvement.
- SAC CAPS and AVP cut ActFluct by an order of magnitude (10.07 → ≈0.5; 12.00 → ≈1.2–1.4) at a small RMSE cost.
- TD-MPC2 AVP cuts ActFluct from 4.37 to ≈0.54 without worsening RMSE.

## P3 — Robustness under disturbances (Figures 2–3; PPO CAPS, SAC CAPS, TD-MPC2 AVP, PID)

- RMSE and ActFluct do not decrease with severity for every controller (turbulence is a smooth
  rise; gusts rise sharply for SAC).
- PID has the lowest ActFluct at every level.
- Five-seed gust ordering of RMSE (easy/medium mean, moderate): PID 7.78 < PPO 8.38 < TD-MPC2 11.65 < SAC 25.12.
- Five-seed turbulence ordering at moderate: PID 5.69 < PPO 6.22 < TD-MPC2 7.23 < SAC 8.52.
- Known accepted exception: at turbulence `off`, current-inference TD-MPC2 mean RMSE is above PPO
  (SEMs overlap); all other rankings hold. Requires JSBSim 1.1.13.

## P4 — Scope

No controller leaves the safe bounds in any reference run (0 out-of-bounds episodes).

## P5 — TD-MPC2 inference decision

`rl.inference_mode=current` stays the default; `icinco2024` is opt-in. The simplified stack must
keep both modes selectable by one option and must not change the default behaviour.

## Reference set

Seed-1 references, 25 episodes per target, evaluation seed 10, JSBSim 1.1.13, `experiment=icinco2024`,
TD-MPC2 with `rl.inference_mode=current`. Nominal and hard use the `noactreg`/`actreg` (CAPS, AVP)
checkpoints under `fw_flightcontrol/models/icinco`; disturbance runs use the **disturbance-trained**
`turbonly`/`gustonly` checkpoints (CAPS for PPO/SAC, AVP for TD-MPC2), as in Figures 2–3 and the
earlier disturbance runs. Local outputs: `reproducibility/runs/2026-11-19_phase1b_references/`
(Git-ignored; `summary_all.csv`, `seed1_vs_archive.csv`). Values in 1e-2 units, RMSE / ActFluct.

| Controller | Nominal (easy+medium) | Hard |
|---|---:|---:|
| PID | 4.17 / 0.06 | 20.03 / 0.16 |
| PPO | 5.14 / 2.50 | 22.89 / 6.80 |
| SAC | 5.27 / 9.28 | 10.36 / 12.16 |
| TD-MPC2 | 3.59 / 2.81 | 8.33 / 3.99 |
| PPO CAPS | 4.69 / 1.38 | 36.69 / 3.86 |
| SAC CAPS | 6.87 / 0.50 | 12.31 / 0.69 |
| TD-MPC2 AVP | 3.63 / 0.72 | 8.23 / 0.78 |

Disturbances (easy/medium mean), `off` → `moderate`:

| Controller | Turbulence | Gust |
|---|---|---|
| PID | 4.89 / 0.08 → 5.66 / 0.24 | 4.17 / 0.07 → 7.76 / 0.20 |
| PPO CAPS | 5.27 / 2.08 → 5.75 / 2.88 | 5.57 / 2.01 → 8.32 / 2.23 |
| SAC CAPS | 5.75 / 1.21 → 6.46 / 2.19 | 13.49 / 1.17 → 20.90 / 2.10 |
| TD-MPC2 AVP | 5.21 / 0.73 → 5.77 / 0.97 | 5.58 / 1.70 → 12.81 / 2.50 |

## Reference comparison against the archived seed-1 files (current harness)

The archived per-seed CSVs (`fw_flightcontrol/eval/attitude_control/outputs/icinco/<algo>/<variant>/*_1.csv`,
seed 1 = file `_1`; PID has a single file) are compared run by run in `seed1_vs_archive.csv`:

- **PPO and SAC (nominal, CAPS, gust):** identical to the archive to the displayed precision (≤ 0.01 RMSE,
  ≤ 0.01 ActFluct). Turbulence moderate differs by ≤ 0.04 RMSE (JSBSim turbulence residual, accepted).
- **PID:** within 0.05 RMSE of the archive (first-action observation fix).
- **TD-MPC2 (current inference):** RMSE within 0.01–0.21 at nominal; ActFluct lower for the plain
  checkpoint (-0.5 to -1.0) and higher for AVP (+0.13 to +0.16), as in the accepted inference decision (P5).
  Disturbance runs differ more (gust moderate +0.6 to +1.3 RMSE; turbulence medium/moderate 15.92 archive → 8.65,
  an archive outlier already documented for this seed family); these are accepted current-inference residuals.

## Reference comparison against the checklist

| Item | Result |
|---|---|
| P1 nominal RMSE | Holds: TD-MPC2 < PID < PPO < SAC (3.59, 4.17, 5.14, 5.27). |
| P1 hard RMSE | **Seed-sensitive:** TD-MPC2 < SAC < PID < PPO; seed-1 PPO (22.89) is worse than PID, as in the archive. |
| P1 nominal ActFluct | Holds: PID < PPO < TD-MPC2 < SAC. |
| P1 hard ActFluct | **Seed-sensitive:** PID < TD-MPC2 < PPO < SAC; the paper's PPO/TD-MPC2 gap is within SEM. |
| P2 PPO CAPS | ActFluct and nominal RMSE fall; hard RMSE rises (22.89 → 36.69). Holds. (PPO AVP not in the reference set.) |
| P2 SAC CAPS | ActFluct 9.28 → 0.50; RMSE rises (5.27 → 6.87; hard 10.36 → 12.31). Holds. |
| P2 TD-MPC2 AVP | ActFluct 2.81 → 0.72; nominal RMSE flat (3.59 → 3.63) rather than lower (paper 3.81 → 3.58). Holds within noise; flag. |
| P3 severity | RMSE and ActFluct rise from `off` to `moderate` for every controller. PID has the lowest ActFluct. Holds. |
| P3 gust ordering (moderate) | PID 7.76 < PPO 8.32 < TD-MPC2 12.81 < SAC 20.90, same as the five-seed ordering. Holds. |
| P3 turbulence ordering (moderate) | PID 5.66 < PPO 5.75 < TD-MPC2 5.77 < SAC 6.46, same as the five-seed ordering. Holds. |
| P3 turbulence off | PID < TD-MPC2 5.21 < PPO 5.27 < SAC; five-seed current inference has PPO ahead of TD-MPC2 (accepted exception, SEMs overlap). |
| P4 bounds | 0 out-of-bounds episodes in all 15 runs. Holds. |
| P5 | TD-MPC2 references use `current`. Holds. |

PID differs slightly from the published figure (nominal 4.17 vs 4.20, hard 20.03 vs 20.10)
because the generic harness uses the new reset observation for the first action (documented in the
[harness guide](../fw_flightcontrol/eval/attitude_control/README.md)); the published PID agreement is within the accepted residuals.
