# ICINCO 2024 simulator profile: restoring the paper-era behaviour as an opt-in

| | |
|---|---|
| Work and validation | 2026-11-03 |
| Branches | `revamp/reproducibility-first` in both repositories |
| Base commits | FW-FlightControl `3a261d4`; FW-JSBGym `571a52f` (= original `main`) |
| Runtime | `fw-310`: Python 3.10.13, PyTorch 2.6.0.dev20241112, NumPy 1.24.4, Gymnasium 0.29.1, JSBSim 1.2.3 (build 1561), Hydra 1.3.2 |
| Scope | Seed-1 PPO checkpoint, easy references, nominal atmosphere, CPU |
| Status | Profile reproduces the diagnostic result exactly; default behaviour unchanged |

Follows [2026-10-30: ICINCO PPO seed-1 drift diagnosis](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md).

## Summary

FW-JSBGym now has **simulator profiles**: named sets of simulator behaviour that
can be selected with one configuration key. The default profile, `current`, is
the existing behaviour. The new opt-in profile, `icinco2024`, restores what the
ICINCO 2024 experiments ran on, as established by the diagnosis.

| Check | Result |
|---|---|
| Default profile vs. current baseline | metric CSV **byte-identical** |
| `icinco2024` vs. final diagnostic row | **all six per-axis and mean metrics identical** |
| `icinco2024` legacy `avg_fcs_fluct` | now **equals the paper's two-surface metric** |
| `icinco2024` repeat | byte-identical |
| Historical input checksums | 204/204 unchanged |
| FW-JSBGym profile tests | 14 passed |
| Launcher tests | 14 passed |

Seed-1 PPO, easy references, nominal atmosphere, with `icinco2024`:

| Metric | Archived | `icinco2024` | Difference |
|---|---:|---:|---:|
| Roll RMSE (rad) | 0.034604420 | 0.034630780 | +0.076% |
| Pitch RMSE (rad) | 0.023595111 | 0.023582377 | -0.054% |
| Mean roll/pitch RMSE (rad) | 0.029099766 | 0.029106578 | **+0.023%** |
| Aileron fluctuation | 0.009466400 | 0.009831173 | +3.853% |
| Elevator fluctuation | 0.002693501 | 0.002781770 | +3.277% |
| Paper surface fluctuation (= exported `avg_fcs_fluct`) | 0.006079950 | 0.006306472 | **+3.726%** |

The +3.7% surface-fluctuation residual is the open question carried over from
the diagnosis; the profile does not change it.

## What the profile restores

Defined in `FW-JSBGym/fw_jsbgym/envs/sim_profiles.py`.

| Behaviour | `current` (default) | `icinco2024` | Changed by |
|---|---|---|---|
| X8 model | `aircraft/x8/x8.xml` | `aircraft/x8_icinco2024/x8_icinco2024.xml` (drag table before the update) | `97e356e` (2025-06-30) |
| X8 initial conditions | `x8_basic_ic.xml`: 16.67 m/s, France | `x8_icinco2024_basic_ic.xml`: 16.97 m/s, lat/lon 0 | `ad64129`, `76dc60d` (2024-12) |
| Integral-error reset in `set_target_state` | both axes, on every call | only the axis whose target changed | `95d4b53` (2024-12-06) |
| Surface history (`fcs_pos_hist`) channels | aileron, elevator, throttle | aileron, elevator | `98580b4` (2025-02-13) |
| Surface history sampling instant | before applying the action | after advancing the FDM | `95d4b53` / `98580b4` |
| Gust start steps (when gusts are enabled) | 250 | 500 and 1500 | after `1072782`; paper Section 6.2.2 |

The historical asset files are exact copies of Git blobs, and the tests check
their SHA-256. Their provenance is documented next to the model in
`FW-JSBGym/fw_jsbgym/fdm_descriptions/aircraft/x8_icinco2024/README.md`. The
copied model is the one validated by the diagnosis (`97e356e^`). It differs from
the `1072782` model only by an FCS component name (`2ff5b75`) that no code reads.

The logical aircraft ID stays `x8`; only the JSBSim model that is loaded changes.
A profile that does not support the requested aircraft raises an error instead of
silently loading the current model. Unknown profile names also raise an error.

## Usage

Any FW-FlightControl command that composes the default Hydra config, for
evaluation or training:

```bash
python fw_flightcontrol/eval/attitude_control/ppo_eval_simple.py ... +env.jsbsim.sim_profile=icinco2024
```

Isolated reproduction launcher, from the FW-FlightControl root:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/fw-310/bin/python \
    reproducibility/evaluate_ppo_nominal.py --sim-profile icinco2024 \
    reproducibility/runs/<new-directory>
```

The environment prints `Simulator profile: <name>` at construction. With
`icinco2024`, the launcher requires that line and two surface channels in the
log, so a simulator without profile support cannot silently run the current
behaviour. It records the profile in `provenance.json`. Simulator file hashes
in the provenance now include untracked, non-ignored files.

The FlightControl YAML files are unchanged, because they are part of the hashed
historical inputs. The key is therefore added with `+`.

## Validation

All runs used the launcher, so each has its own provenance, log, metric CSV, and
comparison. They are stored locally in the Git-ignored
`../runs/sim-profile-20261103/` directory.

| Run | Metric CSV SHA-256 | Compared with | Result |
|---|---|---|---|
| `current` (no profile key) | `cf9d7c49…5a38e5` | `ppo-diagnosis-20261030/current-result` (`cf9d7c49…5a38e5`) | byte-identical |
| `icinco2024` | `2a8db9c0…77408c` | `ppo-diagnosis-20261030/historical-sampling-result` (`d64bb763…ffd37`) | identical except `avg_fcs_fluct` |
| `icinco2024-repeat` | `2a8db9c0…77408c` | `icinco2024` | byte-identical |

`avg_fcs_fluct` differs from the diagnostic row on purpose. The diagnostic patch
kept the throttle channel and gave `0.004255815`. The profile records two channels,
so the legacy evaluator now exports `0.006306471579796196`, the paper metric.
Logs confirm 25 episodes, 50,000 observations, `eps_fcs_fluct shape: (25, 2)`,
and no gust start, since gusts are disabled in this condition.

FW-JSBGym tests (`FW-JSBGym/tests/test_sim_profiles.py`) use real environments
for both profiles. They cover:

- profile selection, defaults, unknown names, and unsupported aircraft;
- the loaded model and initial conditions;
- integral-error resets for the same target, a changed roll target only, a changed pitch target only, and both;
- surface-history channels and sampling instant;
- gust start steps, with gusts enabled and disabled;
- historical asset hashes.

Run them from the FW-JSBGym root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:../FW-FlightControl \
    /opt/anaconda3/envs/fw-310/bin/python -m unittest discover -s tests
```

## Not restored, or not validated yet

- **Airspeed PI reset on target change.** Current `ACBohnNoVaTask.set_target_state`
  resets the airspeed PI when a target changes; `1072782` did not. In the ICINCO
  evaluation and training loops, targets only change at episode boundaries, right
  after `reset_target_state` has already reset that PI, so this is argued to be a
  no-op. It is not in the profile.
- **Gust timing** is restored from the paper and `1072782` code, and tested at
  the unit level. No gust-condition evaluation has been compared with the archive
  yet.
- **Residual +3.7% surface fluctuation.** Still open (JSBSim build, other source
  differences, numerics).
- **Other aircraft, tasks, seeds, algorithms, and reference sets** have not been
  evaluated with the profile.

## Finding: the PPO/SAC training entry points do not run on current `main`

A PPO training smoke test (`ppo_train.py`, seed 1, 2 environments, CPU, no
tracking, with and without the profile) found two **pre-existing** breakages.
They occur with an unmodified FW-JSBGym `571a52f` too, so they are not caused by
the profile:

1. **`JSBSimEnv.init()` is never called.** FW-JSBGym `2ca8d57` (2025-03-13)
   moved simulator creation from `reset()` into `init()`. Only the evaluation
   scripts, `toy_example.py`, and TD-MPC2 were updated. `ppo_train.py`,
   `ppo_norm_train.py`, `sac_train.py`, and `sac_norm_train.py` fail at the
   first `envs.reset()` with "Simulation object does not exist". (`td3_train.py`
   still uses the pre-Hydra `config_file=` environment API and is stale for
   other reasons too.)
2. **Stale `sample_targets` calls.** FW-FlightControl `abc02fe`/`943cb7c`
   (2025-03-13) changed the signature to
   `sample_targets(single_target, env_id, env, cfg_rl)` and moved the roll/pitch
   limits to `rl.*`. `ppo_train.py`, `ppo_norm_train.py`, and `sac_train.py` still
   pass three arguments.

An out-of-repo harness that only adds the missing `init()` call confirmed that
`+env.jsbsim.sim_profile=icinco2024` reaches every vectorized training
environment (2/2 logged the profile and reset), before the run stops at the
`sample_targets` call.

Fresh seed-1 training therefore needs its own slice. It must:

- repair these entry points;
- check that target sampling and other training settings match the paper-era
  protocol;
- use the wandb code snapshots in `fw_flightcontrol/wandb/` as provenance.

## Follow-up decisions

- Commit the profile (FW-JSBGym) and launcher support (FW-FlightControl)
  together. Then create `paper/icinco` in both repositories from this pair,
  labelled as a reproduction candidate. No `icinco-repro-*` tag until the
  remaining checks pass.
- Next slice: restore the PPO training entry point for the ICINCO protocol and
  run fresh seed-1 training with `icinco2024`.
- In parallel or afterwards: residual fluctuation diagnosis, wider evaluation
  coverage, and gust-condition validation.
