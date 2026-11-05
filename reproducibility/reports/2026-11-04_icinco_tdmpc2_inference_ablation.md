# TD-MPC2: why current inference fluctuates less

[Run guide and status](../README.md)

- On preserved seed 1 (easy references), current TD-MPC2 + `icinco2024` has
  21.7% lower ActFluct and 0.83% higher RMSE than the archive. The June 12,
  2024 source differs from the archive by only +2.51% ActFluct and +0.24% RMSE.
- **Cause:** in the June 12 code, critic dropout stayed active during
  evaluation because `model.eval()` did not reach it. Current code disables it
  correctly. Restoring it, plus three historical sampling choices, reproduces
  the June 12 metric CSV byte-for-byte; no simulator change was needed.
- **Decision (Nov 4):** [keep current TD-MPC2](#author-decision-retain-current-td-mpc2-2026-11-04)
  for training and evaluation.
- Only seed 1 easy was ablated. The other seeds show
  [seed-dependent ActFluct changes](./2026-11-05_icinco_phase1_nominal_five_seeds.md#per-seed-residuals).

## Controlled results

Preserved seed 1, evaluation seed 10, easy references, model size 5,
`icinco2024`, RTX A6000 CUDA, 25 episodes / 50,000 steps. Only the listed
inference choices change.

| Inference | Mean RMSE (rad) | ActFluct (rad/sample) | ActFluct vs June 12 |
|---|---:|---:|---:|
| Archive | 0.019974964 | 0.033656020 | -2.45% |
| June 12 source | 0.020021943 | 0.034499986 | reference |
| Current | 0.020141710 | 0.026346973 | -23.63% |
| Historical policy means only | 0.020085873 | 0.025036573 | -27.43% |
| Historical NumPy sampling only | 0.020067457 | 0.026250559 | -23.91% |
| Policy means + NumPy sampling | 0.019924864 | 0.023293171 | -32.48% |
| Historical critic dropout only | 0.020132553 | 0.034735611 | +0.68% |
| All four historical choices | 0.020021943 | 0.034499986 | 0.00% (byte-identical CSV) |

- Dropout alone restores the average fluctuation level; all four choices
  restore all six exported metrics. The effects interact through
  random-number consumption, so they are not additive.
- The unchanged current run repeats byte-for-byte on the other GPU.
- Exact values: [`tdmpc2_inference_ablation.csv`](../tdmpc2_inference_ablation.csv).

## Why the historical evaluator kept dropout on

The June 12 [`Ensemble`](../runs/submission-pair-20261103/FW-FlightControl/tdmpc2/common/layers.py)
keeps its critic inside a `vmap` closure rather than as a registered submodule,
so `model.eval()` never reaches its dropout layer. The
[current version](../../fw_flightcontrol/agents/tdmpc2/tdmpc2/common/layers.py)
registers it as `self.module`. A CUDA probe confirmed this. After
`model.eval()`, the June 12 critic's dropout (p = 0.01) is still in training
mode and repeated Q-evaluations differ; the current critic's are identical.
With dropout disabled, encodings, policy means and critic outputs match
between versions.

The old planner therefore ranked candidates with noisy critic values, which
changed the selected controls. Current TD-MPC2's smoother control is an
inference difference, not a missing flight-dynamics restoration.

The four historical choices, applied only in the diagnostic process:

1. Policy means for planner candidates and terminal-value actions, keeping the
   original random draws.
2. NumPy selection of two critics instead of torch subsampling.
3. NumPy weighted elite selection instead of the torch Gumbel sampler.
4. Critic dropout active (p = 0.01, in place, as historically).

## Author decision: retain current TD-MPC2 (2026-11-04)

Keep the current TD-MPC2 implementation for training and evaluation, with
`icinco2024` for ICINCO simulator behaviour. Do not roll back to June 12 code
or add a legacy inference mode. Lower ActFluct is better, and the author
considers it to strengthen ICINCO's claims; the +0.83% RMSE vs archive stays
on record. The historical restoration is diagnostic evidence only, and
"current" does not authorize untested upstream or dependency upgrades. The
decision rests on seed 1 easy references on CUDA; other conditions and fresh
training still need their own checks.

## Evidence and replay

- Local, Git-ignored: [`2026-11-04_tdmpc2_inference_ablation`](../runs/2026-11-04_tdmpc2_inference_ablation/)
  (six runs, logs, configs, hashes, policy traces, `probe_critic.py` and its
  outputs), plus the current and June 12 runs
  [`preserved_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_preserved_tdmpc2_seed1_easy_cuda/) and
  [`historical_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_historical_tdmpc2_seed1_easy_cuda/).
- Planner in both versions: horizon 3, six iterations, 512 samples, 64 elites,
  24 policy trajectories, discount 0.995.
- All 150 episodes completed; 482 run-input hashes and the 204 protected
  inputs validate.
- Selected baseline: the unchanged current run, FlightControl
  `1ad600b5beb46549898ac47821814694b4a1a67d`.

Diagnostic replay of the historical inference (not the selected setup), from
`FW-RL`, into a new directory:

```bash
BASE="$PWD/FW-FlightControl/reproducibility/runs/2026-11-04_tdmpc2_inference_ablation"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg \
/opt/anaconda3/envs/fw-310/bin/python "$BASE/run_dropout_ablation.py" \
    "$BASE/all-historical-replay" --mean-policy --numpy-q --numpy-elite
```
