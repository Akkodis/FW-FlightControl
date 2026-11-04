# ICINCO SAC and TD-MPC2: preserved seed-1 checks

[Investigation overview](./README.md) | [Current run guide](../README.md)

**Scope:** nominal atmosphere, easy references, preserved seed-1 checkpoints,
25 episodes / 50,000 steps, evaluation seed 10, using `fw-310` and RTX A6000
CUDA. These are narrow inference checks, not full-paper reproductions or
fresh-training validations. PID is not included because it is a hand-tuned
controller, not a saved neural-network checkpoint.

## Results against archived seed-1 metrics

| Algorithm | Metric | Archived | Current + `icinco2024` | June 12 source | Current vs archive | June 12 vs archive |
|---|---|---:|---:|---:|---:|---:|
| SAC | Mean roll/pitch RMSE (rad) | 0.042395332 | 0.041027814 | 0.041027814 | -3.225633% | -3.225633% |
| SAC | Two-surface fluctuation (rad/sample) | 0.078824272540 | 0.079229578811 | 0.079229578811 | +0.514190% | +0.514190% |
| TD-MPC2 | Mean roll/pitch RMSE (rad) | 0.019974963500 | 0.020141710 | 0.020021943 | +0.834777% | +0.235192% |
| TD-MPC2 | Two-surface fluctuation (rad/sample) | 0.033656019780 | 0.026346972962 | 0.034499985838 | -21.716908% | +2.507623% |

Percent differences are relative to the archived metric. SAC's current-profile
run, a repeat on the second A6000, and the June 12 source run agree on all six
exported metrics. The historical TD-MPC2 run is much closer to the archived
fluctuation value than the current evaluator; its RMSE is also closer, though
both are close.

### Author acceptance of SAC (2026-11-04)

The author accepts the measured SAC residuals above: RMSE -3.225633% and
fluctuation +0.514190%, for the preserved seed-1 checkpoint on nominal easy
references with RTX CUDA. This closes the discrepancy investigation for that
slice, as with the earlier PPO decision. The cause remains unknown.

This is acceptance of those measurements, not a general tolerance. It does not
extend to other seeds, references, atmospheres, fresh training, or TD-MPC2.

## TD-MPC2 follow-up: inference difference explained

The [controlled follow-up](./2026-11-04_icinco_tdmpc2_inference_ablation.md)
found the dominant missing factor: historical critic dropout stayed active
despite `model.eval()`. The newer ensemble disables it correctly. Historical
policy means and NumPy critic/elite sampling also differ; restoring them alone
does not close the gap.

Restoring all four historical inference choices temporarily within the current
evaluator and `icinco2024` produces a byte-identical six-metric CSV to the
June 12 source run. This remains diagnostic evidence, not the selected setup.

### Author decision on TD-MPC2 (2026-11-04)

The author chooses to retain **current TD-MPC2 for training and evaluation**,
with `icinco2024` for ICINCO simulator behaviour: no rollback or legacy
inference mode. Lower ActFluct is better, and the author considers the observed
improvement to strengthen ICINCO's claims. The current-vs-June-12 difference is
accepted; the current row above remains the measured baseline, including its
small RMSE increase. The [full decision and scope](./2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04)
preserve the explanation and distinguish this choice from full-paper or
fresh-training validation.

## Run evidence and checks

Each local run directory is Git-ignored. Its `provenance.json` records source
commits, runtime, commands, input SHA-256 values, outputs and validation
results.

| Run | Evidence |
|---|---|
| SAC, current profile, GPU 0 | [`preserved_sac_seed1_easy_cuda`](../runs/2026-11-04_preserved_sac_seed1_easy_cuda/) |
| SAC repeat, current profile, GPU 1 | [`preserved_sac_seed1_easy_cuda_repeat`](../runs/2026-11-04_preserved_sac_seed1_easy_cuda_repeat/) |
| SAC, June 12 source, GPU 1 | [`historical_sac_seed1_easy_cuda`](../runs/2026-11-04_historical_sac_seed1_easy_cuda/) |
| TD-MPC2, current profile, model size 5, GPU 0 | [`preserved_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_preserved_tdmpc2_seed1_easy_cuda/) |
| TD-MPC2, June 12 source, model size 5, GPU 0 | [`historical_tdmpc2_seed1_easy_cuda`](../runs/2026-11-04_historical_tdmpc2_seed1_easy_cuda/) |

All five runs used CUDA and completed 25 episodes without out-of-bounds
episodes. Each produced 50,000 observations of shape `50000 x 14` and
per-episode fluctuation values of shape `25 x 2`. The SAC repeat matched its
first run exactly. All recorded input files were present and their hashes
matched the manifests at report time.

The June 12 source pair is FW-JSBGym
`733a2a8bffe9deb7e0f9ca6e202b7da132d1b168` and FW-FlightControl
`85767615f3206f92c0dd997b21071d5615da8d56`. The current pair is the pushed
`revamp/reproducibility-first` source recorded in each manifest. Both historical
runs use the source's default simulator behavior, with no profile override.

## Conclusion and next work

The preserved SAC and TD-MPC2 checkpoints run successfully under the selected
runtime. SAC is repeatable and its measured residuals are accepted. TD-MPC2's
current-vs-historical gap is explained and accepted, with the current
implementation retained rather than the historical diagnostic overrides.
Neither result changes the accepted PPO boundary or certifies a full paper.

Next, widen coverage and validate fresh training using current TD-MPC2.
See the [experiment ledger](./README.md) and
[workspace revamp plan](../../../revamp_plan.md).
