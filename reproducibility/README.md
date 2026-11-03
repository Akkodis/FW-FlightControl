# Historical baseline: inventory and first evaluation

This records the starting point and review checkpoints for the
reproducibility-first revamp, not a claim that either paper has been reproduced.
Detailed investigations are journaled as dated reports:

| Date | Report |
|---|---|
| 2026-10-30 | [ICINCO PPO seed 1: why current code no longer reproduces the archived scores](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md) |
| 2026-11-03 | [ICINCO 2024 simulator profile: restoring the paper-era behaviour as an opt-in](./reports/2026-11-03_icinco2024_sim_profile.md) |

## Repository starting points

Both repositories were clean on local `main` before creating matching
`revamp/reproducibility-first` branches:

| Repository | Starting `main` commit |
|---|---|
| FW-FlightControl | `bffe6c33e7268d39811a652a83218e20ae2e9384` |
| FW-JSBGym | `571a52fe02ec3f7fc78968a076fa68a536be1455` |

These are local starting commits, not yet verified paper-era commits. Neither `main`
branch was moved. No training, simulator, network, metric, or configuration behavior
has been changed.

## Input checksum snapshot

[`historical_inputs.sha256`](./historical_inputs.sha256) records SHA-256 checksums
with paths relative to the enclosing `FW-RL` directory. It covers:

- all 118 paper checkpoints: 70 ICINCO and 48 SNCS;
- the four attitude reference/seed arrays and five waypoint reference/seed arrays;
- FlightControl YAML configuration;
- JSBSim aircraft, engine, system, and initial-condition XML files;
- X8 trim YAML;
- both published paper PDFs;
- both package setup files and the two JSBGym environment manifests.

Checkpoints total approximately 2332 MiB. They were hashed as files, not deserialized.
No training or full paper evaluation matrix has been executed. A ten-step PPO
smoke check and one full evaluation condition were subsequently completed below.

To verify the snapshot from this repository's root:

```bash
cd ..
sha256sum --check FW-FlightControl/reproducibility/historical_inputs.sha256
```

The starting source commits provide provenance for the remaining tracked code.
Existing result CSVs, trajectories, resolved Hydra run configs, checkpoint schemas,
and paper-table mappings still require inventory before Phase 0 is complete.

## Existing candidate environments

The following versions were read from installed distribution metadata, without
installing packages or importing the simulator/training stack:

| Conda environment | Python | PyTorch | NumPy | Gymnasium | JSBSim |
|---|---|---|---|---|---|
| `fw-310` | 3.10.13 | 2.6.0.dev20241112 | 1.24.4 | 0.29.1 | 1.2.3 |
| `fw-old` | 3.10.0 | 2.6.0.dev20241112 | 1.26.4 | 0.28.1 | 1.1.13 |
| `fw-tdmpc2-39` | 3.9.18 | 2.6.0.dev20241112 | 1.24.4 | 0.29.1 | 1.2.3.dev1 |

All three report Hydra 1.3.2 and OmegaConf 2.3.0. Two NVIDIA RTX A6000 devices
were visible through `nvidia-smi`, with driver 560.35.05. The smoke check below uses
CPU policy inference; it does not validate GPU training.

The checked-in manifests do not establish which environment produced ICINCO.
The similarly named environments differ in simulator and numerical dependencies.
On 2026-10-30 the author selected **`fw-310` first** for evaluation and fresh training,
recalling its use for most of the thesis and possible early ICINCO use of
`fw-tdmpc2-39`. Cross-Python metric compatibility is a validation goal, not yet
a demonstrated property.

The `fw-310` row was rechecked with that interpreter on 2026-10-30. NumPy now reports
1.24.4, whereas the initial inventory recorded 2.2.6. No packages were installed or
changed by this work; shared environment names are not immutable provenance.
Record actual runtime versions for every evaluation.

## First local-source inference check

Runtime probes found that `fw-old` imports the packages from a different checkout;
the other two candidates do not resolve them by default. Distribution metadata alone
is therefore insufficient to select the correct source tree.

Using process-local `PYTHONPATH` with `fw-tdmpc2-39` successfully loaded both packages
from this workspace. The seed-1 ICINCO PPO checkpoint loaded with `weights_only=True`
and strict state-dict matching. Its input has 14 observations and two actions.
Ten nominal-atmosphere simulator steps returned finite observations/rewards with
no early termination/truncation.

[`smoke_ppo.py`](./smoke_ppo.py) preserves that probe. Run it from a scratch working
directory, because the legacy environment creates a `telemetry` directory even when
logging is disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/path/to/FW-RL/FW-JSBGym:/path/to/FW-RL/FW-FlightControl \
/opt/anaconda3/envs/fw-tdmpc2-39/bin/python \
    /path/to/FW-RL/FW-FlightControl/reproducibility/smoke_ppo.py
```

The script verifies the imported source locations and fails explicitly for an
incompatible checkpoint, non-finite result, or unexpected episode end.
The initial `fw-tdmpc2-39` probe used JSBSim `1.2.3.dev1` / `JSBSim-ML v2.0`;
its influence on published scores remains unverified.

The same unchanged script also **passed in `fw-310` on 2026-10-30**, using local
source imports and JSBSim `1.2.3` (build 1561, commit
`570e8115a102df8f877b11e0e59b964ea483e3c0`, `JSBSim-ML v2.0`).
Use `/opt/anaconda3/envs/fw-310/bin/python` in the command above for the chosen
baseline. All 204 input checksums still matched. Neither smoke run establishes
full-episode metric agreement or fresh-training compatibility.

## First full evaluation: seed-1 PPO, easy references, no atmosphere

**Result: runnable and repeatable in `fw-310`, but not reproduced against the
archive.** Two independent executions of the unchanged
[`ppo_eval_simple.py`](../fw_flightcontrol/eval/attitude_control/ppo_eval_simple.py)
completed 25 episodes / 50,000 steps. Their metric CSVs were byte-identical.
No out-of-bounds retries occurred. No simulator, policy, training, or metric
implementation was changed.

The comparison uses
[`ppo_bohnmlp_noactreg_nohist_noatmo_easyrefs_1.csv`](../fw_flightcontrol/eval/attitude_control/outputs/icinco/ppo/noactreg_noatmo/ppo_bohnmlp_noactreg_nohist_noatmo_easyrefs_1.csv),
not the similarly named CSV outside the `icinco` directory.

| Metric | Archived seed 1 | Current `fw-310` | Relative change |
|---|---:|---:|---:|
| Roll RMSE (rad) | 0.034604420 | 0.049398348 | +42.75% |
| Pitch RMSE (rad) | 0.023595111 | 0.059699472 | +153.02% |
| Mean roll/pitch RMSE (rad) | 0.029099766 | 0.054548910 | +87.45% |
| Aileron fluctuation (rad/step) | 0.009466400 | 0.000496550 | -94.75% |
| Elevator fluctuation (rad/step) | 0.002693501 | 0.000306005 | -88.64% |
| Exported `avg_fcs_fluct` (see qualification below) | 0.006079950 | 0.000334999 | -94.49% |

Lower fluctuation does not establish an improvement: tracking error increased,
and matching the historical experiment is the immediate goal.
The diagnosis below subsequently found that the exported fluctuation averages
have different definitions: the current evaluator includes throttle, whereas
the archive and paper average only aileron and elevator. The original CSVs are
preserved; the diagnostic comparison reports the two-surface metric separately.

### Protocol and observation mapping

- Checkpoint: [`ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt`](../fw_flightcontrol/models/icinco/ppo/noactreg/ppo_bohnmlp_noactreg_nohist_noatmo_1_22-05_18:46:52.pt).
  Checkpoint seed 1 is distinct from the evaluator's Python/NumPy/PyTorch seed 10.
- Task: `ACBohnNoVaIErr-v0`; X8; 100 Hz agent and FDM; 20 seconds per episode.
  The existing wrappers record episode statistics and clip actions, without
  observation or reward normalization during evaluation.
- Policy: 14 inputs, two 64-wide Tanh hidden layers, two actor outputs. Use the
  deterministic actor mean, with CPU inference and one OpenMP/MKL thread.
- Current observation order: roll, pitch, airspeed in kph, angular rates p/q/r,
  roll/pitch errors, alpha/beta, aileron/elevator commands, integral roll/pitch
  errors. The last two values are reset by each existing `set_target_state` call;
  that behavior is preserved, not repaired in this probe.
- References: all 25 rows of [`simple_easy.npy`](../fw_flightcontrol/eval/attitude_control/targets/simple_easy.npy),
  in radians. Simulator seeds come from [`jsbsim_seeds.npy`](../fw_flightcontrol/eval/attitude_control/targets/jsbsim_seeds.npy);
  its 26th entry is used by the legacy evaluator's final reset.
- Atmosphere: no wind, turbulence, gusts, or FDM randomization. Override
  `env.jsbsim.eval_sim_options.atmosphere.severity=off` because the current preset
  contains a list but this older evaluator expects a scalar.
- RMSE uses all 50,000 raw observations, columns 6 and 7. The reported average is
  the mean of the two RMSEs, not a pooled roll/pitch RMS.
- The paper's fluctuation is the per-episode mean absolute difference of successive
  actual surface positions in radians, averaged over 25 episodes and then over
  aileron/elevator only. It is not angular velocity or command fluctuation. The
  code uses 1,999 differences per 2,000-sample episode. The current legacy
  aggregate also includes normalized throttle, mixing quantities with different
  units; do not interpret that exported aggregate as the paper metric.

### Connection to the paper, and remaining uncertainty

The easy subset alone is **not** the paper's complete nominal-reference score.
Combining the archived easy and medium scores within each of the five seeds gives:

- nominal mean RMSE `0.0528708032`, SEM `0.0011684226`;
- hard mean RMSE `0.1578438998`, SEM `0.0248007204`.

These round to ICINCO Table 6's PPO values, `5.29 +/- 0.12` and
`15.78 +/- 2.48`, in units of `1e-2`. The archived nominal mean fluctuation is
`0.0197599483`, consistent with Table 7's mean `1.98` in units of `1e-2`.
This links the archived CSV family to the paper; it does not establish that the
current code reproduces it.

There are 3,374 saved Hydra `config.yaml` files under the local, ignored
`fw_flightcontrol/outputs` directory. Six name this checkpoint. The 2024-10-07
16:54:55 snapshot selects this task, no history, no atmosphere, and `simple_easy`,
supporting the selected configuration. These local snapshots are not yet a
versioned, definitive paper-run manifest.

Source history also prevents attributing the drift to Python:

- JSBGym commit `97e356e` changed the X8 drag table in June 2025, after ICINCO.
- Attitude-task initialization, target handling, and airspeed PI reset behavior
  have changed since the October 2024 source.
- The evaluator committed at FlightControl `f71acbe` contains hard-coded
  55/28-degree targets despite loading a reference file. A saved Hydra config
  therefore cannot, by itself, prove which targets a historical run actually used.

These were the initial investigation leads. The controlled experiments below
establish which changes cause most of the measured drift. Do not globally roll
back dynamics or adjust tolerances merely to match the scores.

### Repeat the isolated evaluation

[`evaluate_ppo_nominal.py`](./evaluate_ppo_nominal.py) launches the existing
evaluator in a **new** directory and refuses to reuse an existing output path.
It sets local-source paths and CPU execution only for its child process.
From the FlightControl repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py \
    reproducibility/runs/ppo-seed1-easy-new-run
```

It saves the exact invocation, selected package versions, commits/dirty state,
input hashes, simulator file hashes (tracked and untracked, non-ignored) and
worktree diff, simulator profile, Hydra config, console log,
metric CSV, and comparison CSV. The separate `paper_surface_fluct` comparison
row uses only the aileron/elevator columns; the original metrics are not rewritten.
It checks episode counts, observation counts, surface-history channels, finite
metrics, and unchanged inputs. Exit zero means **evaluation completed**, not
**paper scores matched**. No numerical acceptance tolerance has been chosen.
Add `--sim-profile icinco2024` to evaluate with the ICINCO simulator profile
(see below).

The two completed runs are in the local, Git-ignored directory
[`runs/icinco-ppo-seed1-easy-fw310-20261030`](./runs/icinco-ppo-seed1-easy-fw310-20261030):
`legacy` contains the first direct invocation; `repeat` contains the provenance
and comparison produced by the launcher. These files persist locally but are
not included in commits. The table above preserves the reviewable result in Git.
Per-step raw trajectories are not emitted by this legacy evaluator and remain
work for the compatibility harness.

Focused launcher checks:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    -m unittest discover -s reproducibility -p 'test_*.py'
```

## Controlled diagnosis: causes of the PPO discrepancy

The full investigation is journaled in
[`reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md`](./reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md).
In short, three post-ICINCO FW-JSBGym changes explain the large differences:

1. `95d4b531` (2024-12-06) resets integral-error observations on every
   `set_target_state` call, i.e. every evaluation step. Restoring per-axis
   resets on target change removes the tracking regression.
2. `97e356e07` (2025-06-30) changed the X8 drag table. The earlier table is
   needed to recover the archived surface fluctuation.
3. `98580b4` (2025-02-13) added throttle to the surface history, so the legacy
   `avg_fcs_fluct` is no longer the paper's two-surface metric.

Restoring these, plus the historical initial conditions and post-step surface
sampling, gives mean RMSE `0.029106578` (+0.023% vs archive) and paper surface
fluctuation `0.006306472` (+3.726%). The remaining fluctuation gap is open.
Per-variant values are in [`ppo_diagnosis.csv`](./ppo_diagnosis.csv); the
isolated restorations are in
[`icinco_ppo_diagnostic.patch`](./icinco_ppo_diagnostic.patch), an experiment
artifact rather than a production fix.

## ICINCO 2024 simulator profile

The diagnosed restorations are now an opt-in FW-JSBGym simulator profile; see
[`reports/2026-11-03_icinco2024_sim_profile.md`](./reports/2026-11-03_icinco2024_sim_profile.md).
Select it on any FW-FlightControl command with `+env.jsbsim.sim_profile=icinco2024`,
or with `--sim-profile icinco2024` on the launcher. It loads the paper-era X8
model and initial conditions, resets integral errors only on target changes,
records only aileron/elevator positions after the FDM step, and starts gusts at
steps 500 and 1500. Without the key, behaviour is unchanged (`current`).

With the profile, the seed-1 PPO evaluation reproduces the final diagnostic row
exactly: mean RMSE `0.029106578` (+0.023% vs archive). The exported
`avg_fcs_fluct` is now the paper's two-surface metric, `0.006306472` (+3.726%).
The default profile still produces a metric file byte-identical to the
current-code baseline.

The same report records that the PPO/SAC training entry points no longer run
against current FW-JSBGym: `init()` is never called and the `sample_targets`
calls are stale. This blocks fresh-training checks until it is repaired.

## Next checkpoint

1. Repair the PPO training entry point for the ICINCO protocol: call `init()`,
   restore `sample_targets` usage, and check the target-sampling settings against
   the paper-era code. Then run fresh seed-1 training with `icinco2024` and
   evaluate it against the archive.
2. Resolve or explicitly characterize the small remaining fluctuation
   discrepancy, then extend to medium/hard references, other seeds, gust
   conditions, and the remaining algorithms. Do not assume SNCS checkpoints
   trained after these changes expect the ICINCO semantics.
3. Complete the archived-result/config inventory and definitive paper-era
   mapping. Structural refactors remain separate future gates.

Do not overwrite historical outputs, upgrade packages, or run expensive training
until the relevant baseline protocol is understood.
