# June 12 historical sources: results and acceptance

[Investigation overview](./README.md) | [Current run guide](../README.md)

**Conclusion:** unmodified historical PPO and the restored `icinco2024` profile
produce byte-identical metric CSVs on nominal easy, medium and hard references.
The small archive residuals persist; the author accepts them for this slice.
This tests a preserved checkpoint, not fresh training or the complete paper.

## Comparison with archived seed-1 results

Side-by-side results for the same preserved PPO seed-1 checkpoint in nominal
atmosphere. The archived column contains the original seed-1 results, not the
paper's multi-seed aggregates. Both reruns use `fw-310` and RTX A6000 CUDA;
the historical-code column uses the June 12, 2024 pair in the [source audit](#source-audit).

| Reference set | Metric | Archived results | June 12, 2024 code (CUDA) | Restored `icinco2024` (CUDA) | Both reruns vs archive |
|---|---|---:|---:|---:|---:|
| Easy | Mean roll/pitch RMSE (rad) | 0.029099766 | 0.029096860 | 0.029096860 | -0.009986% |
| Easy | Two-surface fluctuation (rad/sample) | 0.006079950355 | 0.006070544465 | 0.006070544465 | -0.154703% |
| Medium | Mean roll/pitch RMSE (rad) | 0.073654940 | 0.073655420 | 0.073655420 | +0.000652% |
| Medium | Two-surface fluctuation (rad/sample) | 0.043597636583 | 0.043554799400 | 0.043554799400 | -0.098256% |
| Hard | Mean roll/pitch RMSE (rad) | 0.228911640 | 0.228910620 | 0.228910620 | -0.000446% |
| Hard | Two-surface fluctuation (rad/sample) | 0.068063824562 | 0.068104233415 | 0.068104233415 | +0.059369% |

Relative differences in the last column apply to both reruns, since their
results match exactly.

All **six exported metrics per set** match exactly, not just at displayed
precision. The separate historical easy repeat is also byte-identical.
Four historical runs completed **100 episodes / 200,000 steps**, without
out-of-bounds retries. This supports fidelity of the profile for these outputs,
not bitwise equality of full trajectories or validation of every task.

## Author acceptance (2026-11-03)

The author accepts the observed archived-score residuals above for the
preserved seed-1 PPO checkpoint on easy, medium and hard references in nominal
atmosphere, evaluated on RTX CUDA. This closes the discrepancy investigation
for this slice; these residuals no longer block further reproduction work.

The author recalls updating JSBSim and/or PyTorch between the ICINCO and SNCS
papers, but does not remember the precise versions. This makes dependency
changes a plausible explanation, not a demonstrated cause.

This is an explicit acceptance of the measured results, not a new numerical
tolerance for future runs. It does not extend to CPU results, other seeds,
algorithms or atmospheres, full paper aggregates, or fresh-training differences.
The measurements and remaining provenance uncertainty are retained unchanged.

## Source audit

The author recalls these as the submission commits; the attribution is not an
independently verified release tag. Both were checked out as clean, detached
worktrees, with no source, config, aircraft asset or checkpoint patches.

| Repository | Full candidate commit | June 12, 2024 time (+02:00) / subject |
|---|---|---|
| FW-JSBGym | `733a2a8bffe9deb7e0f9ca6e202b7da132d1b168` | 16:46:49 / added model turbonly and gustsonly saves |
| FW-FlightControl | `85767615f3206f92c0dd997b21071d5615da8d56` | 17:03:13 / saved models |

| Read-only comparison | Result |
|---|---|
| ICINCO checkpoints | 70/70 byte-identical Git blobs; no deserialization |
| Attitude references/seeds | 4/4 arrays byte-identical |
| ICINCO result CSVs | 203/203 match after CRLF/LF normalization only: PID 9, PPO 70, SAC 70, TD-MPC2 54 |
| Raw CSV blobs | 66 identical; 137 differ only by line endings, not numerical values |
| Simulator semantics | Per-axis resets on changed targets and episode resets; post-FDM two-surface history; gusts at 500/1500 |
| X8 assets | Initial conditions identical to the profile; drag table identical; model differs only by `normalized-elevator-pos` to `normalized-aileron-pos` component-name correction |
| PPO protocol in source | Reference-array targets change at episode boundaries, not fixed 55/28 degrees; CUDA selected when available; training config requests 750k transitions, six environments, 2,048 steps and uniform 60/30-degree limits |

**Historical repository split:** PPO/SAC lived under `FW-JSBGym/jsbgym`;
FlightControl held TD-MPC2 under `tdmpc2`. This experiment executes only the
JSBGym PPO path. Checking out both commits does **not** validate TD-MPC2.
Source-level CUDA selection is not proof of the archive's exact GPU/runtime.
Both historical worktrees stayed clean; development branches and `paper/icinco`
pointers were not moved.

## Exact source and runtime

The [common evaluation protocol](../README.md#common-preserved-checkpoint-protocol)
applies, using the original `jsbgym/ppo_eval_simple.py` and reference/seed
arrays under `jsbgym/eval/refs/`, **without a simulator-profile override**.
Only checkpoint/reference selection, debug/severity and output isolation were
overridden; the original evaluator computed the metrics without corrections.

Checkpoint:
`jsbgym/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt`.
SHA-256: `06beda27077d75faf829ccf08c462b2f7609e2f0ff2ff2e6c130eb3aa9f2c2d9`.

| Runtime item | Value |
|---|---|
| Interpreter | `/opt/anaconda3/envs/fw-310/bin/python`, Python 3.10.13 |
| Policy device | RTX A6000, GPU 0; CUDA required, no fallback |
| PyTorch / CUDA runtime | 2.6.0.dev20241112 / 12.4 |
| NumPy / Gymnasium / JSBSim | 1.24.4 / 0.29.1 / 1.2.3 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| OpenMP / MKL threads | 1 / 1, as in the profile comparisons |

Preflight verified historical `jsbgym`, `agents.ppo` and
`jsbgym.utils.train_utils` imports; every log confirmed `Using Device: cuda`.
No dependencies changed. This is historical **source on the current runtime**,
not a reconstruction of the 2024 software environment.

## Evidence and rerunning

| Local, Git-ignored evidence | Contents |
|---|---|
| [`submission-pair-20261103`](../runs/submission-pair-20261103/) | Both detached worktrees; external `run_evaluation.py` helper; `easy-cuda`, `medium-cuda`, `hard-cuda`, `easy-repeat-cuda` logs, provenance, Hydra and metric/comparison CSVs |
| [`runtime-310-cuda`](../runs/training-checkpoint-20261103/runtime-310-cuda/) | Restored-profile easy comparison |
| [`followup-20261103`](../runs/followup-20261103/) | Restored-profile `medium-cuda` and `hard-cuda` comparisons |

The helper verified 201 input hashes before/after each run, including historical
source/config/assets, the checkpoint, archived metrics and comparison inputs.
All 204 protected workspace historical-input hashes still match.

From `FW-RL`, rerun the retained helper with a **new** output-directory name:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    FW-FlightControl/reproducibility/runs/submission-pair-20261103/run_evaluation.py \
    easy easy-cuda-new
```

For a fresh workspace, the original evaluator can be invoked directly as below.
This reproduces the evaluation command; the retained local helper additionally
records provenance and performs the checks described above. Use a new directory:

```bash
# Start in FW-RL.
BASE="$PWD/FW-FlightControl/reproducibility/runs/submission-pair-replay"
git -C FW-JSBGym worktree add --detach "$BASE/FW-JSBGym" \
    733a2a8bffe9deb7e0f9ca6e202b7da132d1b168
git -C FW-FlightControl worktree add --detach "$BASE/FW-FlightControl" \
    85767615f3206f92c0dd997b21071d5615da8d56
mkdir -p "$BASE/easy-cuda/eval"
ln -s "$BASE/FW-JSBGym/jsbgym/eval/refs" "$BASE/easy-cuda/eval/refs"
cd "$BASE/easy-cuda"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled MPLBACKEND=Agg
export PYTHONPATH="$BASE/FW-JSBGym:$BASE/FW-JSBGym/jsbgym:$BASE/FW-FlightControl"
/opt/anaconda3/envs/fw-310/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA required; do not fall back to CPU"' &&
/opt/anaconda3/envs/fw-310/bin/python "$BASE/FW-JSBGym/jsbgym/ppo_eval_simple.py" \
    "model_path=$BASE/FW-JSBGym/jsbgym/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt" \
    ref_file=simple_easy res_file=ppo_seed1_easy \
    env.jsbsim.debug=false env.jsbsim.eval_sim_options.atmosphere.severity=off \
    hydra.run.dir=hydra hydra.job.chdir=false
```

Use `simple_medium` or `simple_hard` with separate output directories for the
other reference sets. The historical evaluator itself selects CUDA when
available; retain the explicit preflight and verify its device log.
