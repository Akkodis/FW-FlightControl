# ICINCO nominal results: all agents, five seeds

[Run guide and status](../README.md)

Nominal atmosphere, no action regulation, easy/medium/hard references. Preserved
PPO, SAC and TD-MPC2 (model size 5) checkpoints of training seeds 1, 2164, 2989,
4508 and 6930 (45 CUDA runs), plus the deterministic PID controller (CPU).
Current code + `icinco2024`, [common protocol](../README.md#common-preserved-checkpoint-protocol).

- **The nominal rows of ICINCO Tables 6–7 reproduce.** Current PID, PPO and SAC
  are within 0.01 (units of 1e-2) of every published mean and SEM.
- **TD-MPC2** matches the nominal RMSE. Hard RMSE is 2.0% lower, and ActFluct is
  10.1% (nominal) and 6.5% (hard) lower, as expected from the
  [inference ablation](./2026-11-04_icinco_tdmpc2_inference_ablation.md). The
  agent ranking is unchanged.
- **Paper method:** "Nominal Refs" averages the easy and medium means *and SEMs*.
  Table 7's PPO "5.29 ± 1.20" is a typo for ± 0.12.
- **Archive files** `_1` to `_5` hold seeds 1, 2164, 6930, 4508 and 2989.
- **Review:** the measured nominal residuals for all five seeds are
  [accepted by the author](#author-decisions).

## Results against the paper

Five-seed mean ± SEM in units of 1e-2: mean roll/pitch RMSE (rad) and
two-surface fluctuation, ActFluct (rad/sample). *Archive* applies the paper's
aggregation to the archived per-seed CSVs; *current* applies it to this
evaluation. The paper calls TD-MPC2 "TD-MPC".

| Agent | Metric | Paper nominal | Archive nominal | Current nominal | Paper hard | Archive hard | Current hard |
|---|---|---:|---:|---:|---:|---:|---:|
| PID | RMSE | 4.20 | 4.20 | 4.20 | 20.10 | 20.10 | 20.10 |
| PID | ActFluct | 0.07 | 0.07 | 0.07 | 0.17 | 0.17 | 0.17 |
| TD-MPC2 | RMSE | 3.81 ± 0.10 | 3.81 ± 0.10 | 3.80 ± 0.10 | 9.10 ± 0.69 | 9.10 ± 0.69 | 8.91 ± 0.54 |
| TD-MPC2 | ActFluct | 4.37 ± 0.48 | 4.37 ± 0.48 | 3.93 ± 0.58 | 6.02 ± 0.59 | 6.02 ± 0.59 | 5.63 ± 0.78 |
| PPO | RMSE | 5.29 ± 0.12 | 5.29 ± 0.12 | 5.29 ± 0.12 | 15.78 ± 2.48 | 15.78 ± 2.48 | 15.78 ± 2.48 |
| PPO | ActFluct | 1.98 ± 0.58 | 1.98 ± 0.58 | 1.97 ± 0.58 | 5.54 ± 0.93 | 5.54 ± 0.93 | 5.53 ± 0.93 |
| SAC | RMSE | 6.64 ± 0.83 | 6.64 ± 0.83 | 6.63 ± 0.84 | 11.15 ± 0.65 | 11.15 ± 0.65 | 11.15 ± 0.65 |
| SAC | ActFluct | 10.07 ± 1.12 | 10.07 ± 1.12 | 10.08 ± 1.12 | 12.00 ± 0.71 | 12.00 ± 0.71 | 12.00 ± 0.71 |

PID is one deterministic controller, so it has no SEM. The action-regulated
rows of Table 7 are not covered yet.

## How the paper aggregates seeds

- **Per reference level:** mean over the five seeds; SEM = sample standard
  deviation / √5. This reproduces the historical `compute_seed_statistics.py`
  output (`all_total.csv`).
- **Nominal Refs:** average of the easy and medium means, and of their SEMs.
  **Hard Refs:** the hard aggregate.
- The SEM of the five per-seed easy-medium averages does not reproduce the
  paper (for example, SAC RMSE 0.74 instead of 0.83).

## Per-seed residuals

Mean RMSE / mean ActFluct of each run vs its archive file, in percent.
Per-axis values are in the local `per_seed.csv`.

| Agent | Seed (file) | Easy | Medium | Hard |
|---|---|---:|---:|---:|
| PPO | 1 (#1) | -0.01 / -0.15 | 0.00 / -0.10 | 0.00 / +0.06 |
| PPO | 2164 (#2) | 0.00 / 0.00 | 0.00 / -1.64 | 0.00 / +0.28 |
| PPO | 2989 (#5) | 0.00 / -0.04 | 0.00 / -0.04 | 0.00 / -0.38 |
| PPO | 4508 (#4) | 0.00 / +0.76 | 0.00 / -0.43 | 0.00 / -0.15 |
| PPO | 6930 (#3) | 0.00 / -0.43 | 0.00 / +0.16 | 0.00 / -0.02 |
| SAC | 1 (#1) | -3.23 / +0.51 | 0.00 / 0.00 | 0.00 / 0.00 |
| SAC | 2164 (#2) | 0.00 / 0.00 | -0.03 / +0.01 | -0.06 / -0.04 |
| SAC | 2989 (#5) | 0.00 / 0.00 | +0.06 / +0.06 | -0.01 / +0.03 |
| SAC | 4508 (#4) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| SAC | 6930 (#3) | -0.04 / +0.53 | 0.00 / -0.11 | -0.01 / +0.06 |
| TD-MPC2 | 1 (#1) | +0.83 / -21.72 | -0.66 / -15.37 | -0.45 / -21.76 |
| TD-MPC2 | 2164 (#2) | -0.19 / -3.75 | +1.26 / +2.69 | -6.57 / +0.20 |
| TD-MPC2 | 2989 (#5) | +0.60 / -7.05 | -0.53 / -10.82 | -0.39 / +2.20 |
| TD-MPC2 | 4508 (#4) | -2.83 / -15.25 | -0.16 / -14.79 | -0.41 / -8.14 |
| TD-MPC2 | 6930 (#3) | +0.64 / -13.61 | +0.44 / -14.29 | -0.52 / -11.44 |

- **PID:** every exported metric is within 0.0003% of the archive.
- **PPO:** mean RMSE within ±0.01% (per axis ±0.013%) and mean ActFluct within
  ±1.64% (seed 2164 medium).
- **SAC:** mean RMSE within ±0.064% and mean ActFluct within ±0.53%, except the
  accepted seed-1 easy residual. The June 12, 2024 source reproduces that run
  byte-for-byte, so later code changes do not cause it. It accounts for nearly
  all of SAC's -0.44% shift in easy RMSE.
- **TD-MPC2:** RMSE within ±1.26% in 13 of 15 runs. The exceptions are seed
  4508 easy (-2.83%) and seed 2164 hard (-6.57%), which drives most of the
  lower hard aggregate. ActFluct is lower in 12 of 15 runs, by a seed-dependent
  amount, which also widens the ActFluct SEMs:

  | Seed | ActFluct change |
  |---|---|
  | 1 | -15% to -22% |
  | 4508, 6930 | -8% to -15% |
  | 2989 | -11% to +2% |
  | 2164 | -4% to +3% |

  The dropout explanation was tested on seed 1 easy only.

## Archive file numbering

| Training seed | 1 | 2164 | 2989 | 4508 | 6930 |
|---|---:|---:|---:|---:|---:|
| Archive file | 1 | 2 | 5 | 4 | 3 |

No configuration or commit ties the archive file numbers (`..._easyrefs_1.csv`
to `_5.csv`) to seeds. A run's distance to a file is its largest relative
roll/pitch RMSE difference; fluctuation is excluded because current TD-MPC2
inference changes it. For each algorithm, `aggregate.py` picks the one-to-one
seed-to-file assignment, shared by all three reference levels, with the
smallest summed distance. All three algorithms give the same assignment.

- **PPO and SAC:** every run is nearest its assigned file, within 0.013% and
  0.095%; any other file is at least 2.26% and 4.87% away. Only SAC seed 1 easy
  exceeds these bounds: 3.76% from its file against 28.03% from the next. The
  order matches the timestamps in the checkpoint names.
- **TD-MPC2:** the summed distance is 25.5%; the next-best assignment, which
  swaps seeds 2989 and 4508, scores 42.3%. Each reference level alone gives
  the same assignment. Seed 6930 medium lies nearer file 1 (1.13%) than its
  file 3 (1.85%): those two medium files differ by only 1.41%.
- The paper-table values do not depend on this mapping; the per-seed residuals do.

## Author decisions

Residuals are accepted as measured for the nominal, no-action-regulation
evaluations, not as a general tolerance.

| Date | Decision |
|---|---|
| Nov 3 | PPO seed 1, easy/medium/hard accepted ([evidence](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md#june-12-2024-source-and-author-acceptance)) |
| Nov 4 | SAC seed 1 easy accepted; cause unknown |
| Nov 4 | Keep current TD-MPC2 inference ([decision](./2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04)) |
| Nov 5 | SAC and TD-MPC2 seed 1, medium/hard accepted |
| Nov 5 | PPO, SAC and TD-MPC2 seeds 2164, 2989, 4508 and 6930, easy/medium/hard accepted. PPO and SAC closely match the archive; TD-MPC2 residuals are larger but preserve the method ranking and the paper's overall conclusion. |

## Evidence

- **Local, Git-ignored runs:** [`2026-11-05_phase1_nominal_five_seeds`](../runs/2026-11-05_phase1_nominal_five_seeds/)
  holds the following; PID runs are in [`2026-11-05_phase1_nominal_seed1`](../runs/2026-11-05_phase1_nominal_seed1/).
  - `run_nominal.py`: runs the legacy evaluator in a subprocess with
    `+env.jsbsim.sim_profile=icinco2024`, records provenance and compares
    against all five archive files.
  - `run_queue.py`: the two-GPU queue.
  - `aggregate.py` and its CSVs.
  - 45 run directories.
- **Checks:** every run completed 25 episodes with no out-of-bounds episode and
  50,000 × 14 observations. Each confirmed `icinco2024` and CUDA, had
  consistent averaged columns and left its input hashes unchanged. The 204
  protected inputs are unchanged.
- **Repeatability:** all nine seed-1 CSVs are byte-identical to earlier runs,
  including three that ran on the other GPU this time.
- **Source:** FlightControl `d3ac16ca6c853cc5fc95358de7d01d8026416c72` plus
  recorded uncommitted work, tdmpc2 `bce5911f4f9ced1a3d6bbc94f4f145b6fdb292b7`,
  FW-JSBGym `96e9cda0e0ca3d5eace0ca2ca9b4675e3d2b9332`.
- **Runtime:** `fw-310`: Python 3.10.13, PyTorch 2.6.0.dev20241112 (CUDA 12.4),
  NumPy 1.24.4, Gymnasium 0.29.1, JSBSim 1.2.3.
- **PID evaluator:** `pid_eval_simple.py` needed a four-line repair to run on
  current code: config path, targets directory, `env.init()` and the
  `set_target_state` signature. Gains, metrics and loop are unchanged. Its
  archives come from FW-JSBGym `d9b920e` (2024-06-03).
- **Timing:** PPO and SAC ran first, three per GPU, in 53–109 s per run.
  TD-MPC2 then ran on its own, eight and seven at once per GPU, in 60–68 min
  per run (about 20 min with a GPU to itself). The queue took 76 min, against
  about 5.5 h one run at a time. In the seed-1 batch, SAC took about 24 min per
  run when sharing a GPU with TD-MPC2.
