# Reproduction investigation: start here

Updated through **2026-11-04**. This is the short record of what we tested,
what we learned, and what remains. It does **not** certify either complete paper.

## What to remember

- **Three later changes caused the large ICINCO PPO drift:** integral-error resets,
  X8 drag, and inclusion of throttle in the fluctuation metric.
- **`icinco2024` restores the paper-era simulator behaviour.** It is opt-in; the default
  `current` behaviour has not been changed globally.
- **Historical source and the restored profile agree exactly** on the exported
  seed-1 PPO CUDA metrics for nominal easy, medium and hard references.
  See the [archived / June 12 code / restored-profile table](./2026-11-03_icinco_submission_pair_evaluation.md#comparison-with-archived-seed-1-results).
- **The author accepts the documented PPO/SAC CUDA residuals.** Their exact cause
  is unknown; recalled JSBSim/PyTorch updates are a hypothesis, not proof.
  Acceptance covers the documented PPO slice and, following the Nov 4
  decision, the nominal easy SAC seed-1 slice. TD-MPC2 has a separate decision below.
- **Fresh PPO training works**, but is not an exact seed-1 reconstruction:
  both easy-reference metrics fall inside the predeclared five-seed ranges,
  while fluctuation is still **+43.98%** versus archived seed 1.
- **TD-MPC2's fluctuation gap is explained.** Historical critic dropout stayed
  active during evaluation. Restoring it with historical policy/sampling choices
  makes current code + `icinco2024` match the June 12 metric CSV exactly.
- **Keep current TD-MPC2 for training/evaluation.** The author accepts the lower
  ActFluct as supporting ICINCO's claims; no rollback or legacy inference mode.
  Historical restoration remains diagnostic evidence, not the selected setup.
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
| Nov 4 | Evaluate preserved SAC and TD-MPC2 | SAC repeats exactly; its archive residuals were subsequently accepted by the author. TD-MPC2's historical inference is closer to archive than the current planner, especially for fluctuation. [Comparison and SAC acceptance](./2026-11-04_icinco_sac_tdmpc2_evaluation.md) |
| Nov 4 | Isolate TD-MPC2 inference changes | Six full CUDA ablations identify active historical critic dropout as the dominant fluctuation factor. Complete legacy inference restores all six June 12 metrics byte-for-byte. [Ablations](./2026-11-04_icinco_tdmpc2_inference_ablation.md) |
| Nov 4 | Retain current TD-MPC2 | Author chooses current training/evaluation code and accepts the explained ActFluct improvement; supersedes the proposed legacy mode. [Decision and scope](./2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04) |

## Coverage and remaining work

| Area | Current boundary |
|---|---|
| Preserved PPO seed 1 | Nominal easy/medium/hard checked on CUDA; small residuals accepted. |
| Preserved SAC seed 1 | Nominal easy checked on CUDA, including a GPU repeat and June 12 source control; measured residuals accepted, cause unknown. |
| Preserved TD-MPC2 seed 1 | Nominal easy CUDA gap explained and accepted; current code + `icinco2024` is retained. Historical metrics remain diagnostic references, not rollback targets. Fresh training and wider coverage remain. |
| Fresh PPO seed 1 | Full-budget training and easy evaluation checked; medium/hard remain. |
| Full ICINCO tables | Other seeds, easy-plus-medium multi-seed aggregation, remaining algorithms/conditions and full atmosphere coverage remain. Gust timing has unit tests, not archived-score validation. |
| SNCS | Only the isolated integral-reset independence check is complete; full reproduction remains. |
| Historical runtime | Exact 2024 binaries/settings are unknown; further reconstruction is not required for the accepted slice. |
| Release status | `revamp/reproducibility-first` is pushed to Akkodis `origin` in both repositories. Local `paper/icinco` branches remain limited-scope candidates; no release tags. |

Next: continue wider seed, reference, atmosphere and fresh-training coverage
using current TD-MPC2 before structural refactors or dependency modernization.

## Finding the evidence

- [Run commands and common protocol](../README.md): use existing `fw-310`,
  RTX CUDA, explicit `icinco2024`, and new output directories.
- [Baseline inventory](./2026-10-30_baseline_inventory.md): checksum scope,
  installed environments, first smoke checks, and archived-table mapping.
- Each dated report owns its detailed tables and evidence paths. `runs/` is
  local and Git-ignored; committed reports/CSVs preserve the findings, not all
  raw logs.
