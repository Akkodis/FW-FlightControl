# TD-MPC2 fluctuation drift: historical critic dropout

[Investigation overview](./README.md) | [Baseline comparison and SAC acceptance](./2026-11-04_icinco_sac_tdmpc2_evaluation.md)

**Conclusion:** the current-vs-June-12 discrepancy is explained for this slice.
The dominant change is **critic dropout remaining active during historical
evaluation**, whereas the newer ensemble correctly disables it. Restoring that
behaviour together with historical policy means and sampling choices makes
current code + `icinco2024` produce a **byte-identical six-metric CSV** to the
June 12 source run. No further simulator change was needed.

**Author decision (2026-11-04): retain current TD-MPC2 for training and
evaluation, not the diagnostic legacy restoration.** The improved ActFluct is
an accepted behaviour difference. See the [decision and scope](#author-decision-retain-current-td-mpc2-2026-11-04).

## Controlled results

All six new evaluations use the unchanged current evaluator, preserved
checkpoint seed 1, evaluation seed 10, nominal easy references, model size 5,
and `icinco2024`. Each completes 25 episodes / 50,000 steps on RTX A6000 CUDA.
Only the indicated inference choices are overridden in the diagnostic process.

| Inference behaviour | Mean RMSE (rad) | Two-surface fluctuation (rad/sample) | Fluctuation vs June 12 |
|---|---:|---:|---:|
| June 12 source reference | 0.020021943 | 0.034499985838 | 0.000000% |
| Current, repeated unchanged | 0.020141710 | 0.026346972962 | -23.631931% |
| Historical policy means only | 0.020085873 | 0.025036572865 | -27.430194% |
| Historical NumPy sampling only | 0.020067457 | 0.026250559406 | -23.911391% |
| Policy means + NumPy sampling | 0.019924864 | 0.023293170615 | -32.483536% |
| Historical critic dropout only | 0.020132553 | 0.034735611362 | +0.682973% |
| All four historical choices | 0.020021943 | 0.034499985838 | 0.000000% |

[Exact per-axis and averaged metrics](../tdmpc2_inference_ablation.csv) retain
the full exported precision. The unchanged control on GPU 1 reproduces the
previous current-profile GPU 0 CSV byte-for-byte. The complete historical
restoration on GPU 0 reproduces the June 12 GPU 0 CSV byte-for-byte.

The original policy/sampler hypotheses were insufficient: restoring both
without dropout moved fluctuation farther from the historical result.
Dropout alone restores the average fluctuation level to within +0.683%, but
not the individual metrics. All four choices restore all six metrics exactly.
These effects are not additive: the changes interact and alter random-number
consumption. This is a fixed-seed compatibility test, not a multi-seed estimate.

## Why the historical evaluator kept dropout on

The historical [Ensemble implementation](../runs/submission-pair-20261103/FW-FlightControl/tdmpc2/common/layers.py)
stores its functional critic inside the `vmap` callable's closure. It is not a
registered child module, so `model.eval()` does not reach its dropout layer.
The [current implementation](../../fw_flightcontrol/agents/tdmpc2/tdmpc2/common/layers.py)
registers the critic as `self.module`; evaluation mode now propagates to it.

A CUDA runtime probe confirmed this, rather than relying only on source inspection:

| Check after `model.eval()` | June 12 code | Current code |
|---|---|---|
| World model / ensemble `.training` | Both `False` | Both `False` |
| Actual critic dropout `.training` | **`True`**, probability 0.01 | `False`, probability 0.01 |
| Repeated `Q(..., return_type="all")`, identical inputs | Different logits; consumes CUDA RNG | Identical logits; no CUDA RNG consumed |
| Disable the historical hidden dropout explicitly | Repeated logits become identical | Already disabled |

Using the same 64 recorded observations and checkpoint, latent encodings,
policy means and dropout-disabled critic logits match exactly between versions.
Matching active dropout also requires the historical **in-place** implementation
to reproduce its random stream; with that setting and the same CUDA seed,
the critic logits match exactly too.

Thus the old planner evaluated candidates using noisy critic values even in
evaluation mode. Those values affect elite ranking and the selected controls.
The newer evaluator's smoother controls are an inference-behaviour difference,
not evidence of another missing flight-dynamics restoration.

## What the complete diagnostic restoration changes

1. Use policy means for planner candidates and terminal-value actions, while
   retaining the original policy calls and their random draws.
2. Use NumPy selection of two critic networks rather than torch subsampling.
3. Use NumPy weighted elite selection rather than the torch Gumbel sampler.
4. Keep critic dropout active at probability 0.01, with historical in-place
   semantics. Other modules remain in evaluation mode.

Historical evaluation is still stochastic: MPPI candidates, critic selection,
elite selection and critic dropout use randomness. Policy means alone do not
make the planner deterministic.

## Evidence, checks and replay

- [Local evidence directory](../runs/2026-11-04_tdmpc2_inference_ablation/):
  six run directories named by the CSV's `case` values, logs, resolved configs,
  input hashes, provenance and full policy-input/action traces.
- [Critic probe](../runs/2026-11-04_tdmpc2_inference_ablation/probe_critic.py):
  `critic-current` / `critic-historical` JSON and NPZ outputs record the runtime
  checks above.
- Source pair and runtime are those in the baseline comparison: Python 3.10.13,
  PyTorch 2.6.0.dev20241112 / CUDA 12.4 and JSBSim 1.2.3. This is historical
  source on the selected runtime, not a reconstruction of the 2024 binaries.
- Planner settings agree: horizon 3, six iterations, 512 samples, 64 elites,
  24 policy trajectories and discount 0.995. Decoder, stochastic dynamics,
  episodic termination, compilation and the extra MPPI fluctuation term are off.
- All **150 episodes / 300,000 steps** completed without out-of-bounds episodes.
  Each run has `50000 x 14` observations and `25 x 2` fluctuation values.
  Persisted policy traces are finite and their hashes validate.
- All 482 run-manifest input-hash entries and all 204 protected baseline inputs
  validate. Checkpoints, simulator/agent source and shared dependencies are unchanged;
  only documentation, the summary CSV and local diagnostics were added/updated.

Helpers and raw evidence are local and Git-ignored. The following command
replays the historical diagnostic, **not the selected training/evaluation
setup**. From `FW-RL`, use the retained helper and a **new** output name:

```bash
BASE="$PWD/FW-FlightControl/reproducibility/runs/2026-11-04_tdmpc2_inference_ablation"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg \
/opt/anaconda3/envs/fw-310/bin/python "$BASE/run_dropout_ablation.py" \
    "$BASE/all-historical-replay" --mean-policy --numpy-q --numpy-elite
```

## Author decision: retain current TD-MPC2 (2026-11-04)

The author chooses the current TD-MPC2 implementation for the harmonized
training/evaluation setup. **Do not roll back to June 12 code or introduce the
previously proposed legacy inference mode.** Keep the current policy and
sampling behaviour and correctly disabled critic dropout during evaluation.
For ICINCO runs, retain the separate `icinco2024` simulator profile.

The rationale is that lower ActFluct is better: the measured current result is
**23.631931% lower than the June 12 rerun and 21.716908% lower than the archive**.
The author considers this improvement to strengthen ICINCO's claims rather
than undermine them. Exact agreement with the older, noisier evaluator is
therefore not the operational goal. The current result also has mean RMSE
**+0.834777% versus archive**; that measurement remains visible, not corrected
or described as an improvement in every metric.

This closes the current-vs-historical ActFluct investigation as an explained,
accepted difference. The selected measured baseline is the unchanged-current
row above (FlightControl `1ad600b5beb46549898ac47821814694b4a1a67d`), not the
all-historical ablation. "Current" does not authorize untested upstream or
dependency upgrades. No algorithm, training code or defaults changed to record
this decision.

Keep the dropout explanation, historical code references and ablation results
as diagnostic evidence, not a second supported training/evaluation mode. The
June 12 rerun's archive residuals (+0.235192% RMSE, +2.507623% fluctuation)
remain in the record but are no longer a target to pursue by reverting code.

The evidence covers preserved checkpoint seed 1, nominal easy references and
RTX CUDA only. It is not a blanket tolerance, a full-paper validation, or proof
that fresh training and every other condition improve. Historical/current full
trajectories were not compared bit-for-bit. Continue wider coverage and fresh
training with the current implementation before structural refactoring.
