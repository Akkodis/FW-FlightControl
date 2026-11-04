# Baseline inventory and first checks

[Investigation overview](./README.md) | [Run guide](../README.md)

**Conclusion:** the archived inputs are protected and the local seed-1 PPO
checkpoint loads and runs. These checks alone do not establish paper-score or
fresh-training agreement.

## Frozen starting point

Both repositories were clean on local `main` before creating
`revamp/reproducibility-first`. Neither original `main` was moved.

| Repository | Starting commit |
|---|---|
| FW-FlightControl | `bffe6c33e7268d39811a652a83218e20ae2e9384` |
| FW-JSBGym | `571a52fe02ec3f7fc78968a076fa68a536be1455` |

These are revamp starting points, not paper-era commits. The later
[June 12 source audit](./2026-11-03_icinco_submission_pair_evaluation.md#source-audit)
identifies the author-supplied historical candidates.

[`historical_inputs.sha256`](../historical_inputs.sha256) protects 204 files,
with paths relative to `FW-RL`:

| Inputs | Scope |
|---|---|
| Paper checkpoints | 118: 70 ICINCO, 48 SNCS; about 2,332 MiB, hashed without deserialization |
| References and seeds | Four attitude arrays and five waypoint arrays |
| Experiment configuration | FlightControl YAML, X8 trim YAML |
| Simulator assets | Aircraft, engine, system and initial-condition XML |
| Publication/runtime records | Both paper PDFs, both package setup files, two JSBGym environment manifests |

The snapshot detects accidental input changes; it does not prove numerical
reproducibility. Starting commits identify the remaining tracked source.
All 204 hashes remained unchanged through the latest experiments.

## Runtime and import checks

Installed metadata was inspected without installing packages. Runtime values
must still be recorded per run: shared environment names are not immutable.

| Environment | Python | PyTorch | NumPy | Gymnasium | JSBSim |
|---|---|---|---|---|---|
| `fw-310` | 3.10.13 | 2.6.0.dev20241112 | 1.24.4 | 0.29.1 | 1.2.3 |
| `fw-old` | 3.10.0 | 2.6.0.dev20241112 | 1.26.4 | 0.28.1 | 1.1.13 |
| `fw-tdmpc2-39` | 3.9.18 | 2.6.0.dev20241112 | 1.24.4 | 0.29.1 | 1.2.3.dev1 |

All reported Hydra 1.3.2 and OmegaConf 2.3.0. Two RTX A6000 GPUs were visible,
with driver 560.35.05. The initial inventory reported NumPy 2.2.6 for `fw-310`;
the October 30 runtime recheck reported 1.24.4. This work changed no packages.

- `fw-old` initially imported another checkout; the other candidates did not
  resolve the local packages by default. Process-local `PYTHONPATH` fixed the
  source selection without installing anything.
- [`smoke_ppo.py`](../smoke_ppo.py) passed in both `fw-tdmpc2-39` and `fw-310`:
  strict `weights_only=True` checkpoint load, 14 observations, two actions,
  ten finite nominal steps, no unexpected episode end. These were CPU probes,
  not GPU-training or full-episode checks.
- The probes used JSBSim `1.2.3.dev1` / `JSBSim-ML v2.0` and `1.2.3`
  (build 1561, commit `570e8115a102df8f877b11e0e59b964ea483e3c0`), respectively.
  Run legacy probes in a scratch directory: telemetry directories are created
  even when logging is disabled.

The author selected **`fw-310` first**, recalling it for most thesis work and
possibly `fw-tdmpc2-39` for early ICINCO work. Future policy inference/training
uses RTX CUDA; CPU results are retained only as historical diagnostic evidence.
Cross-Python compatibility is a validation goal, not a general guarantee.

## Archived results really map to the paper

Combining archived easy and medium scores within each of the five PPO seeds
gives the nominal aggregate below. Hard uses the five archived hard scores.

| Quantity | Archived calculation | Paper value, units of `1e-2` |
|---|---:|---|
| Nominal mean RMSE / SEM | 0.0528708032 / 0.0011684226 | Table 6: `5.29 +/- 0.12` |
| Hard mean RMSE / SEM | 0.1578438998 / 0.0248007204 | Table 6: `15.78 +/- 2.48` |
| Nominal mean surface fluctuation | 0.0197599483 | Table 7 mean: `1.98` |

This identifies the archived CSV family; it is **not** a rerun of the full
paper. Use archives under `eval/attitude_control/outputs/icinco/`, not similarly
named files outside `icinco`.

The initial configuration audit found 3,374 local, ignored Hydra `config.yaml`
files, six naming the seed-1 checkpoint. The 2024-10-07 16:54:55 snapshot
supports the selected task, no history/atmosphere and easy references.
Configuration alone is not definitive provenance: later evaluator `f71acbe`
hard-codes 55/28-degree targets despite loading references; the June 12
candidate evaluator does not.

## What followed

Two unchanged easy/CPU evaluations completed 25 episodes / 50,000 steps each
with byte-identical CSVs, but disagreed strongly with the archive. Their
results and controlled restorations are in the
[diagnosis](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md).
Local outputs: [`icinco-ppo-seed1-easy-fw310-20261030`](../runs/icinco-ppo-seed1-easy-fw310-20261030/)
(`legacy` and `repeat`).

A definitive paper-run manifest, complete saved-config/checkpoint-schema
inventory, and broader SNCS mapping remain unfinished. The historical-source
audit strengthens ICINCO provenance but does not close those tasks.
