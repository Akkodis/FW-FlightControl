# Historical baseline: initial inventory

This is the first review checkpoint for the reproducibility-first revamp. It records
the untouched starting point, not a claim that either paper has been reproduced.

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
No training or full paper evaluation has been executed. A ten-step PPO inference
smoke check was subsequently completed as described below.

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

## Next checkpoint

1. Use `fw-310` and process-local source paths without altering shared Conda
   installations. Retain the uncertainty about the original ICINCO stack.
2. Inventory archived results and resolved configs; map the first nominal
   checkpoint to its observation space and evaluation protocol.
3. Attempt one preserved-checkpoint evaluation in a new output directory before
   any behavior refactor.

Do not overwrite historical outputs, upgrade packages, or run expensive training
until the relevant baseline protocol is understood.
