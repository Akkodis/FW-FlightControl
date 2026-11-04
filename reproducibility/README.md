# Reproducibility guide

**Returning to the investigation? Start with the [summary and experiment ledger](./reports/README.md).**
The [three-way results table](./reports/2026-11-03_icinco_submission_pair_evaluation.md#comparison-with-archived-seed-1-results)
compares archived seed-1 PPO, June 12 historical code, and restored `icinco2024`.
The full ICINCO and SNCS papers have **not** yet been reproduced.

## Run the validated slice

Use the existing **`fw-310`** environment and an **RTX GPU through CUDA**.
Do not upgrade the shared environment or silently fall back to CPU. JSBSim
physics still runs on CPU. Keep FW-JSBGym and FW-FlightControl as sibling
checkouts; run the commands below from the FlightControl root.

The evaluator still defaults to `current` and CPU: **pass both
`--sim-profile icinco2024` and `--device cuda` explicitly**.
Every output directory must be new; archived files are never overwritten.

### Preserved seed-1 PPO evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py \
    --sim-profile icinco2024 --device cuda --reference-set easy \
    reproducibility/runs/preserved-easy-cuda-new
```

Select `medium` or `hard` with a different output directory for each run.
Add `--checkpoint /path/to/fresh.pt` to evaluate another checkpoint against the
corresponding archived seed-1 reference. The launcher sets local-source paths
and checks device, profile, finite metrics, episode/observation counts, channels
and input hashes.
**Exit zero means the evaluation completed, not that paper scores matched.**

### Fresh seed-1 PPO training, then evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/train_ppo_icinco.py --device cuda \
    reproducibility/runs/fresh-seed1-new
```

The launcher selects `icinco2024`, disables remote tracking, trains the final
checkpoint and evaluates it on easy references. The 750,000-transition request
completes 749,568 transitions in full rollouts. Add `--smoke` for 4,096 transitions,
not a paper-score check. The full run fails explicitly if either aggregate
metric is outside the pre-recorded five-seed archive range; see
[training results and limitations](./reports/2026-11-03_icinco_ppo_training.md).

For direct Hydra entry points, the profile selector is
`+env.jsbsim.sim_profile=icinco2024`. The `+` avoids changing protected YAML.
This is not a claim that every legacy trainer works.

## Common preserved-checkpoint protocol

| Item | Value |
|---|---|
| Policy | [Archived PPO seed 1](../fw_flightcontrol/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt), no action regularization |
| Evaluation RNG seed | 10; distinct from checkpoint/training seed 1 |
| Task | `ACBohnNoVaIErr-v0`, X8, 14 observations, two actions |
| Actor | Two 64-unit Tanh hidden layers; deterministic mean |
| Episodes | 25 targets x 2,000 steps; 100 Hz agent/FDM; 20 seconds each |
| References | Fixed `simple_easy.npy`, `simple_medium.npy`, `simple_hard.npy` in [targets](../fw_flightcontrol/eval/attitude_control/targets/) |
| Simulator seeds | `jsbsim_seeds.npy`; the 26th entry serves the evaluator's final reset |
| Atmosphere | No wind, turbulence, gusts or randomized FDM; scalar severity `off` |
| Wrappers | Episode statistics and action clipping; no evaluation observation/reward normalization |
| Execution | `fw-310`, RTX A6000 CUDA, one OpenMP/MKL thread |

Observation order: roll, pitch, airspeed (kph), p/q/r, roll/pitch errors,
alpha/beta, aileron/elevator commands, roll/pitch integral errors. The last two
accumulate correctly with `icinco2024`; unchanged-target calls reset them in
the default `current` profile.

- **Mean RMSE (rad):** compute each axis's RMSE from all 50,000 raw errors
  (columns 6/7), then average the two RMSEs; not a pooled RMS.
- **Surface fluctuation (rad/sample):** mean absolute difference of successive
  actual surface positions, 1,999 differences per episode, averaged over
  episodes and aileron/elevator. Not command fluctuation or angular velocity.
  The launcher reports `paper_surface_fluct` separately from legacy exports
  that may include normalized throttle.
- A single seed/subset is **not** the paper's five-seed easy-plus-medium nominal
  aggregate. The [archive-to-paper mapping](./reports/2026-10-30_baseline_inventory.md#archived-results-really-map-to-the-paper)
  documents that distinction.

## Evidence and checks

Each launcher run retains its invocation, source commits/dirty state and diff,
package versions, input/source hashes, profile, Hydra config, log, raw metrics
and comparison. Training also retains the final model and TensorBoard curves.
`runs/` is Git-ignored: paths in reports identify **local evidence**, not files
included in a fresh clone. Reports and [`ppo_diagnosis.csv`](./ppo_diagnosis.csv)
preserve the reviewable findings. Legacy evaluation does not emit complete raw
trajectories; adding those remains harness work.

Verify the [204-file input snapshot](./reports/2026-10-30_baseline_inventory.md#frozen-starting-point):

```bash
(cd .. && sha256sum --check FW-FlightControl/reproducibility/historical_inputs.sha256)
```

Focused launcher/training and simulator-profile tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    -m unittest discover -s reproducibility -p 'test_*.py'
(cd ../FW-JSBGym && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../FW-FlightControl \
    /opt/anaconda3/envs/fw-310/bin/python -m unittest discover -s tests)
```

For unmodified June 12 source rather than today's profile, use the separate
[historical worktree procedure](./reports/2026-11-03_icinco_submission_pair_evaluation.md#evidence-and-rerunning).
