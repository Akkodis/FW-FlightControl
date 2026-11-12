# Attitude-control evaluation

This Hydra-configured harness evaluates PID, PPO, SAC, and TD-MPC2 on attitude
control. Targets, checkpoint files, policy seeds, simulator seeds, atmosphere
severities, and visualization are configured independently. Waypoint tracking
is outside this harness.

Run commands from `FW-FlightControl` in an environment that can import
FW-JSBGym. The default configuration is intentionally generic and has no
targets; add target and checkpoint entries for a run.

The entry point's `run()` function orchestrates evaluation; `worker.py` provides
runtime, provenance, output helpers, and the internal checkpoint worker.

## Basic run

For example, evaluate PID on a custom array:

```bash
python -m fw_flightcontrol.eval.attitude_control algorithm=pid \
  'evaluation.targets=[{path:fw_flightcontrol/eval/attitude_control/targets/easy.npy}]' \
  evaluation.atmo_severities='[off]' \
  evaluation.output=reproducibility/runs/pid-easy
```

An attitude target file is a NumPy array with shape `(episodes, 2)` containing
roll and pitch targets in radians. Relative file paths are resolved from the
current directory first, then from the `fw_flightcontrol` package directory.

For a neural controller, provide each checkpoint explicitly. Checkpoint
selection does not depend on target names, atmosphere settings, algorithm
variants, or severity:

```bash
python -m fw_flightcontrol.eval.attitude_control algorithm=ppo checkpoints=single \
  "checkpoints.entries.0.path=/absolute/path/to/model.pt" \
  checkpoints.entries.0.training_seed=42 \
  'evaluation.targets=[{path:/absolute/path/to/targets.npy}]' \
  evaluation.policy_seeds='[10]' \
  evaluation.output=reproducibility/runs/ppo-custom
```

The `single` checkpoint group is a starter entry; use a user-owned Hydra
config group or override `evaluation.checkpoints` to define a list. A
checkpoint entry has `name`, `path`, and optional `training_seed` fields.
PID does not require a checkpoint.

To evaluate all `.pt` and `.pth` checkpoints directly inside a directory:

```bash
python -m fw_flightcontrol.eval.attitude_control algorithm=ppo \
  evaluation.checkpoint_directory=fw_flightcontrol/models/icinco/ppo/actreg/caps \
  'evaluation.targets=[{path:fw_flightcontrol/eval/attitude_control/targets/easy.npy}]' \
  evaluation.episodes.count=1 \
  evaluation.output=/absolute/path/to/new-output
```

Files are discovered in filename order; subdirectories are not searched.
Each filename is its checkpoint name. Training-seed metadata is
unknown (`null`), not inferred from filenames; use explicit entries when that
metadata is needed. A directory cannot be combined with explicit checkpoint
entries or PID. An absent directory or one without matching files is an error.
The discovered list is fixed at startup and saved in `config.yaml` and
`provenance.json`; selected files are hashed before and after the evaluation.

## Checkpoint parallelism

When multiple checkpoints are selected and visualization is disabled, each
checkpoint runs in its own subprocess. Its policy seeds, targets, severities
and episodes remain ordered within that process. Final reports retain the
configured checkpoint order, regardless of completion order.
Every worker uses the same configured target files, resolved to absolute paths
before the worker starts; targets are not regenerated per checkpoint.
Each target is identified in reports and aggregation groups by its configured
filename without the extension, such as `easy`; these names must be unique within
a run.

By default, the scheduler uses all GPUs visible to the process, with up to four
workers per GPU. It respects `CUDA_VISIBLE_DEVICES` and assigns each worker one
visible GPU. Override `evaluation.execution.workers_per_gpu` with an integer
from 1 to 4 to use fewer workers per GPU. If CUDA is unavailable, the default
is two workers; set `evaluation.execution.cpu_workers` to change this. These
limits apply to one evaluation invocation, not other concurrent invocations.
TD-MPC2 still requires CUDA internally.

Single-checkpoint evaluations and all visualized evaluations run serially.
Checkpoint names prefix worker output, and parallel runs save each checkpoint's
resolved config, assigned GPU, log, and result under `jobs/<checkpoint-name>/`.
A failed or incomplete worker fails the whole run rather than producing a
partial aggregate. Already running workers finish, while unscheduled
checkpoints are not started.

## Historical ICINCO preset

The preset selects the preserved easy/medium/hard reference arrays, simulator
seed file, the 25-episode protocol, historical PID gains, and the `icinco2024`
simulator profile. It defaults to nominal atmosphere and PID; it does not
force a model, algorithm, or weather/checkpoint pairing.

```bash
python -m fw_flightcontrol.eval.attitude_control experiment=icinco2024 \
  evaluation.output=reproducibility/runs/icinco-pid-nominal
```

To select a neural model, supply the algorithm and checkpoint as above. To
evaluate turbulence or gusts, choose the corresponding existing JSBSim config
and severity list, for example:

```bash
python -m fw_flightcontrol.eval.attitude_control experiment=icinco2024 \
  algorithm=ppo checkpoints=single \
  "checkpoints.entries.0.path=/absolute/path/to/model.pt" \
  env/jsbsim=turbonly \
  'evaluation.atmo_severities=[off,light,moderate,severe]' \
  evaluation.output=reproducibility/runs/icinco-ppo-turbulence
```

For TD-MPC2, the historical preset selects
`rl.inference_mode=icinco2024`; set it to `current` to compare with the current
inference implementation.

`experiment=icinco2024` is an overridable preset, not a closed list of
published combinations. Hydra's `--cfg job` option can preview the composed
configuration without starting the evaluation. PID uses the new reset
observation for each episode's first action, correcting the old evaluator's
behavior; its results may therefore differ from historical PID runs.

## Seeds, severity, and visualization

- `evaluation.policy_seeds` controls policy-side random state. For trained
  models, each checkpoint can separately record its training seed. Each policy
  seed (10 by default) is applied once per checkpoint; its random sequence
  continues across targets, severities, and episodes. This also seeds the global
  Python, NumPy, and PyTorch random generators. For TD-MPC2, this overrides
  `rl.seed` in the evaluation controller without changing the saved config.
- `evaluation.simulator_seeds.mode` can be `generated`, `values`, or `file`.
  Simulator seeds must be positive integers. A file or explicit seed list
  must contain one seed per selected episode,
  plus one extra when `evaluation.episodes.include_final_reset=true`.
- `evaluation.episodes.start` and `count` select a target-array slice; a null
  count evaluates all remaining targets.
- `evaluation.atmo_severities` accepts any nonempty, unique subset of
  `off`, `light`, `moderate`, and `severe`. Severity selection is independent
  of visualization.
- Set `evaluation.visualization.mode` to `matplotlib`, `flightgear`, or
  `both`. Set `evaluation.visualization.interactive=true` to pause after each
  episode; press Enter to continue or `q` to stop.
- All runs report an `ERROR` immediately when an episode goes out of bounds,
  record it, and continue with the remaining episodes.

Every Hydra option used by the evaluator has a declared default under
`fw_flightcontrol/config/`. Edit or add YAML groups for reusable target,
checkpoint, algorithm, atmosphere, and experiment choices rather than relying
on undeclared `+key=value` overrides.

## Outputs

Control-surface fluctuation metrics use aileron and elevator only; extra
recorded channels such as throttle do not contribute to the average.

`evaluation.output` must be a new or empty directory. The harness writes:

- `config.yaml` and `provenance.json` with the configuration, runtime versions,
  source revisions, and SHA-256 input hashes;
- `metrics.csv` with one row per checkpoint, target, severity, and policy seed,
  including out-of-bounds error flags and episode counts;
- `aggregates.csv` with checkpoint means/SEMs, error flags and counts, and any
  declared target groups;
- `episodes.json` with per-episode reward, length/duration, policy/simulator
  seeds, out-of-bounds status, and termination information;
- `result.json` with completion status, an out-of-bounds error flag and count,
  and remaining condition-job count;
- `telemetry/` when the selected environment or visualization records it.

The run fails rather than emitting success-shaped partial results if
configuration, target/seeds, checkpoint loading, episode execution, or
post-run input-hash validation fails.

An episode with fewer than two surface-position samples has no defined
fluctuation. Its episode record explains why, and the condition reports how
many episodes contributed to that metric. If none contribute, fluctuation
metrics are unavailable (`null` in JSON, empty in CSV); aggregate fluctuation
metrics are also unavailable if any contributing checkpoint lacks them.

## Real GPU integration test

The opt-in test runs one JSBSim episode for each PPO CAPS checkpoint and checks
parallel workers, CUDA use, the configured target file, and the final report.
It requires CUDA, FW-JSBGym, the ICINCO dependencies, and the checkpoint files:

```bash
FW_RUN_REAL_GPU_TESTS=1 python tests/test_attitude_control_evaluation.py RealGpuIntegrationTests
```

The test uses temporary output and is skipped during ordinary test runs.
