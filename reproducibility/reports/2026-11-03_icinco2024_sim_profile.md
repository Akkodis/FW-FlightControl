# ICINCO simulator profile: implementation and validation

[Investigation overview](./README.md) | [Diagnosis](./2026-10-30_icinco_ppo_seed1_drift_diagnosis.md) | [Run guide](../README.md)

**Conclusion:** `icinco2024` packages the diagnosed restorations behind one
opt-in selector. The default `current` profile remains unchanged.
Initial validation used preserved seed-1 PPO, easy references, nominal
atmosphere and CPU; later CUDA/historical-source results are in the
[accepted comparison](./2026-11-03_icinco_submission_pair_evaluation.md).

## Behaviour selected by the profile

Registry: [`sim_profiles.py`](../../../FW-JSBGym/fw_jsbgym/envs/sim_profiles.py).

| Behaviour | `current` (default) | `icinco2024` | Historical change |
|---|---|---|---|
| X8 model | Current `x8.xml` | Separate `x8_icinco2024.xml`, earlier drag table | `97e356e` |
| Initial conditions | 16.67 m/s, France | 16.97 m/s, lat/lon 0 | `ad64129`, `76dc60d` |
| Integral reset on target setter | Both axes on every call | Only the axis whose target changed | `95d4b53` |
| Surface-history channels | Aileron, elevator, throttle | Aileron, elevator | `98580b4` |
| Surface-history sampling | Before applying action | After advancing FDM | `95d4b53` / `98580b4` |
| Enabled gust start steps | 250 | 500 and 1500 | After `1072782`; paper Section 6.2.2 |

Both integrals also reset at episode start. The logical aircraft ID remains
`x8`. Unknown profiles or unsupported aircraft raise errors, not silent
fallbacks. The environment prints the selected profile; the evaluation launcher
requires the banner and expected channel count and records them in provenance.

Historical assets are copies of Git blobs, protected by SHA-256 tests;
[asset provenance](../../../FW-JSBGym/fw_jsbgym/fdm_descriptions/aircraft/x8_icinco2024/README.md)
records their source. The copied model (`97e356e^`) differs from `1072782` and
the June 12 candidate only by the documented aileron component-name correction
(`2ff5b75`), not its drag table or explicit output property.

Use `--sim-profile icinco2024` on the isolated evaluator, or
`+env.jsbsim.sim_profile=icinco2024` with Hydra. Protected YAML files were not
edited. See the [CUDA commands](../README.md#run-the-validated-slice).

## Validation at this checkpoint

Base sources: FlightControl `3a261d4`, JSBGym `571a52f`; `fw-310`.
Implementation commits: JSBGym `052d8b4`, FlightControl `e0d3474`.

| Check | Result |
|---|---|
| Default vs original current baseline | Metric CSV byte-identical (`cf9d7c49...5a38e5`) |
| Historical profile vs final diagnostic case | RMSE, per-axis fluctuation and paper two-surface average identical |
| Profile repeat | Byte-identical (`2a8db9c0...77408c`) |
| Runtime shape | 25 episodes / 50,000 observations; surface history `(25, 2)`; no gust in nominal conditions |
| Protected inputs | 204/204 unchanged |
| Tests | 14 simulator-profile tests and 14 launcher tests passed |
| Vectorized training smoke | Both environments selected the profile; existing trainer initialization/sampler failures were exposed |

The one intentional export difference: the diagnostic patch retained throttle,
so `avg_fcs_fluct` was `0.004255815`. The profile records two channels and exports
`0.006306471579796196`, the paper metric. This is not a dynamics discrepancy.
The CPU residual was +3.726%; later CUDA results must not be confused with it.

The real-environment [profile tests](../../../FW-JSBGym/tests/test_sim_profiles.py)
cover selection/errors, model/initial conditions, unchanged/per-axis target
resets, history channels/sampling, enabled/disabled gust timing and asset hashes.
Commands are centralized in the [run guide](../README.md#evidence-and-checks).

## Limits and follow-up

- **Airspeed PI reset:** the later base task resets its PI on a target change;
  `1072782` did not. This is not restored. In the tested ICINCO loops, changes
  occur at episode boundaries just after that PI has already been reset, so it
  is argued to be a no-op, not a general equivalence claim.
- **Gusts:** timing is unit-tested, but gust-condition scores are not validated.
- **Training:** the exposed `init()` and `sample_targets` failures were later
  repaired for standard PPO only; see [repairs and fresh training](./2026-11-03_icinco_ppo_training.md#ppo-entry-point-repairs).
- **Scope:** no global integral-reset change, SNCS certification or full-paper
  release follows from these checks. The [overview](./README.md#coverage-and-remaining-work)
  owns the current coverage status.

Local logs, provenance and comparisons:
[`sim-profile-20261103`](../runs/sim-profile-20261103/)
(`current`, `icinco2024`, `icinco2024-repeat`).
