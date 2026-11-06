# TD-MPC2: why current inference fluctuates less

[Run guide and status](../README.md)

- Preserved seed 1, easy: current TD-MPC2 + `icinco2024` has 21.7% lower
  ActFluct and 0.83% higher RMSE than the archive; the June 12, 2024 source
  differs by only +2.51% / +0.24%.
- **Cause:** June 12 critic dropout stayed active in evaluation (`model.eval()`
  missed it). Restoring it plus three sampling choices reproduces the June 12
  CSV byte-for-byte; no simulator change needed.
- **Decision (Nov 4):** [keep current TD-MPC2](#author-decision-retain-current-td-mpc2-2026-11-04).
- Combined choices across five seeds:
  [disturbances](./2026-11-05_icinco_phase1_nominal_five_seeds.md#historical-td-mpc2-inference-across-five-seeds).

## Controlled results

Evaluation seed 10, easy, model size 5, `icinco2024`, CUDA, 25 episodes.

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

- Dropout alone restores fluctuation; effects interact via RNG consumption, so
  are not additive.
- Current run repeats exactly on the other GPU.
- Exact values: [`tdmpc2_inference_ablation.csv`](../tdmpc2_inference_ablation.csv).

## Why the historical evaluator kept dropout on

- The June 12 [`Ensemble`](../runs/submission-pair-20261103/FW-FlightControl/tdmpc2/common/layers.py)
  holds its critic in a `vmap` closure, so `model.eval()` misses its dropout;
  the [current version](../../fw_flightcontrol/agents/tdmpc2/tdmpc2/common/layers.py)
  registers it as `self.module`.
- A CUDA probe confirmed this; with dropout off, both versions' outputs match.
- Noisy Q-values changed planner choices: an inference difference, not missing
  flight dynamics.

Historical choices restored:

1. Policy means for planner candidates and terminal-value actions.
2. NumPy two-critic selection instead of torch subsampling.
3. NumPy weighted elite selection instead of the Gumbel sampler.
4. Critic dropout active (p = 0.01, in place).

## Author decision: retain current TD-MPC2 (2026-11-04)

Keep the current TD-MPC2 implementation for training and evaluation, with
`icinco2024` for ICINCO simulator behaviour. Do not roll back to June 12 code
or add a legacy inference mode. Lower ActFluct is better, and the author
considers it to strengthen ICINCO's claims; the +0.83% RMSE vs archive stays
on record. The historical restoration is diagnostic evidence only, and
"current" does not authorize untested upstream or dependency upgrades. The
decision rests on seed 1 easy references on CUDA; other conditions and fresh
training still need their own checks.

**Update (Nov 6):** The author validated the measured TD-MPC2 disturbance
residuals and chose an opt-in paper-era inference mode with current inference
as the default. It affects evaluation only; historical training is separate.
See the [mode usage and limits](../README.md#paper-era-td-mpc2-inference).

## Training assessment

- Historical training also differs in RNG use, target-critic dropout, actor
  objective, critic-parameter handling and replay config (local [assessment](../runs/2026-11-06_tdmpc2_legacy_mode_check/training_assessment.json)).
- Inference restoration alone cannot reproduce training. A historical training
  mode needs a full source/config comparison, seeded rollout and update checks,
  and a fresh-training evaluation before claiming reproducibility.

## Evidence and replay

- Local: [`2026-11-04_tdmpc2_inference_ablation`](../runs/2026-11-04_tdmpc2_inference_ablation/)
  (runs, probes), plus
  [`preserved_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_preserved_tdmpc2_seed1_easy_cuda/) and
  [`historical_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_historical_tdmpc2_seed1_easy_cuda/).
- Planner (both): horizon 3, six iterations, 512 samples, 64 elites, 24 policy
  trajectories, discount 0.995.
- 150 episodes completed; 482 run-input and 204 protected hashes validate.
- Selected baseline: current run, FlightControl
  `1ad600b5beb46549898ac47821814694b4a1a67d`.

Diagnostic replay of historical inference, from `FW-RL`:

```bash
BASE="$PWD/FW-FlightControl/reproducibility/runs/2026-11-04_tdmpc2_inference_ablation"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg \
/opt/anaconda3/envs/fw-310/bin/python "$BASE/run_dropout_ablation.py" \
    "$BASE/all-historical-replay" --mean-policy --numpy-q --numpy-elite
```
