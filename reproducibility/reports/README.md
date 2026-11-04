# Reproduction investigation: start here

Updated through **2026-11-03**. This is the short record of what we tested,
what we learned, and what remains. It does **not** certify either complete paper.

## What to remember

- **Three later changes caused the large ICINCO drift:** integral-error resets,
  X8 drag, and inclusion of throttle in the fluctuation metric.
- **`icinco2024` restores the paper-era behaviour.** It is opt-in; the default
  `current` behaviour has not been changed globally.
- **Historical source and the restored profile agree exactly** on the exported
  seed-1 PPO CUDA metrics for nominal easy, medium and hard references.
  See the [archived / June 12 code / restored-profile table](./2026-11-03_icinco_submission_pair_evaluation.md#comparison-with-archived-seed-1-results).
- **The author accepts the small remaining CUDA residuals.** Their exact cause
  is unknown; recalled JSBSim/PyTorch updates are a hypothesis, not proof.
  This investigation is closed for that slice, not a general tolerance.
- **Fresh PPO training works**, but is not an exact seed-1 reconstruction:
  both easy-reference metrics fall inside the predeclared five-seed ranges,
  while fluctuation is still **+43.98%** versus archived seed 1.
- **SNCS waypoint tracking is unaffected by this particular integral-reset
  bug.** That narrow check does not validate its policies or flight dynamics.

## Experiment ledger

Dates below are investigation dates, not the dates of the historical code.

| Date | Experiment | Result and detailed record |
|---|---|---|
| Oct 30 | Freeze inputs and check local imports | 204 inputs protected; 118 checkpoints inventoried; strict PPO load and ten-step CPU smoke passed in two environments. [Baseline](./2026-10-30_baseline_inventory.md) |
| Oct 30 | Repeat unchanged seed-1 PPO evaluation | Both easy/CPU runs identical, but mean RMSE was +87.45% versus archive. [Diagnosis](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#controlled-results) |
| Oct 30 | Restore simulator factors separately | Integral resets explain tracking drift; drag and metric semantics explain most fluctuation drift. Initial conditions and sampling instant refine the match. [Ablations](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#controlled-results) |
| Nov 3 | Validate the opt-in profile and repeat | Default unchanged; historical profile matches the diagnostic metrics and exports the correct two-surface average. [Profile](./2026-11-03_icinco2024_sim_profile.md) |
| Nov 3 | Vary runtime and inference device | Python 3.9/3.10 environments match per device; CUDA reduces the easy fluctuation gap from +3.726% to -0.155%. [Device table](./2026-11-03_icinco_ppo_training.md#preserved-checkpoint-inference-device-matters) |
| Nov 3 | Repair PPO, smoke-test, then train seed 1 | 749,568 transitions; finite outputs; fresh easy/CUDA evaluation repeats exactly. [Training](./2026-11-03_icinco_ppo_training.md#full-run-result) |
| Nov 3 | Check reset semantics and SNCS exposure | Same-target calls preserve historical integrals. Three waypoint variants are unchanged by toggling only the reset flag. [Integral and SNCS checks](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#integral-errors-and-sncs) |
| Nov 3 | Extend preserved PPO to medium/hard | Both CUDA runs complete; the launcher's easy/CPU regression remains byte-identical. [Coverage extension](./2026-11-03_icinco_ppo_training.md#preserved-checkpoint-medium-and-hard-references) |
| Nov 3 | Audit the June 12, 2024 source pair | 70 checkpoints, four reference/seed arrays and 203 result CSVs match; CSV matching allows line-ending normalization only. [Source audit](./2026-11-03_icinco_submission_pair_evaluation.md#source-audit) |
| Nov 3 | Execute unmodified historical PPO | Easy, medium, hard and an easy repeat: 100 episodes / 200,000 steps. All metric CSVs match the restored profile. [Historical evaluation](./2026-11-03_icinco_submission_pair_evaluation.md) |
| Nov 3 | Review the residual with the author | Measured nominal seed-1 CUDA residuals accepted; no metric correction or blanket tolerance. [Decision](./2026-11-03_icinco_submission_pair_evaluation.md#author-acceptance-2026-11-03) |

## Coverage and remaining work

| Area | Current boundary |
|---|---|
| Preserved PPO seed 1 | Nominal easy/medium/hard checked on CUDA; small residuals accepted. |
| Fresh PPO seed 1 | Full-budget training and easy evaluation checked; medium/hard remain. |
| Full ICINCO tables | Other seeds, easy-plus-medium multi-seed aggregation, other algorithms and atmosphere conditions remain. Gust timing has unit tests, not archived-score validation. |
| SNCS | Only the isolated integral-reset independence check is complete; full reproduction remains. |
| Historical runtime | Exact 2024 binaries/settings are unknown; further reconstruction is not required for the accepted slice. |
| Release status | Local `paper/icinco` branches are limited-scope candidates, not full-paper releases. No release tags or pushes. |

Next: broaden preserved/fresh PPO coverage, then algorithm and atmosphere
coverage. Other trainer repairs, SNCS reproduction, structural refactors and
dependency modernization remain separate work.

## Finding the evidence

- [Run commands and common protocol](../README.md): use existing `fw-310`,
  RTX CUDA, explicit `icinco2024`, and new output directories.
- [Baseline inventory](./2026-10-30_baseline_inventory.md): checksum scope,
  installed environments, first smoke checks, and archived-table mapping.
- Each dated report owns its detailed tables and evidence paths. `runs/` is
  local and Git-ignored; committed reports/CSVs preserve the findings, not all
  raw logs. No new experiments were run for this documentation reorganization.
