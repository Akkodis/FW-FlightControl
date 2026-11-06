# ICINCO results: five-seed evaluations

[Run guide and status](../README.md)

> [!WARNING]
> **ICINCO disturbance reproduction needs JSBSim 1.1.13**; the historical
> profile cannot compensate for turbulence changes in 1.2.3. Nominal and
> action-regulation results used 1.2.3 and are not yet revalidated with 1.1.13.

Preserved PPO, SAC and TD-MPC2 (size 5) checkpoints of training seeds 1, 2164,
2989, 4508 and 6930, plus deterministic PID, run with current code, `icinco2024`
and the [common protocol](../README.md#common-preserved-checkpoint-protocol).

- **Nominal (Tables 6–7):** PID, PPO and SAC are within 0.01 (1e-2 units) of
  every published value; TD-MPC2 matches RMSE with lower ActFluct, ranking
  unchanged ([inference ablation](./2026-11-04_icinco_tdmpc2_inference_ablation.md)).
- **Action regulation:** PPO/SAC match closely; TD-MPC2 AVP ActFluct is higher.
- **Disturbances:** historical TD-MPC2 inference exactly reproduces gust and
  turbulence-off; stronger turbulence still deviates.
- **Paper typos:** PPO RMSE SEMs "1.20", "1.50", "2.10" should be 0.12, 0.15,
  0.21.
- All measured residuals are [accepted by the author](#author-decisions).

## Results against the paper

Five-seed mean ± SEM (1e-2): roll/pitch RMSE (rad) and ActFluct (rad/sample).
*Archive* aggregates the archived per-seed CSVs as the paper does. PID has no
SEM. The paper's "TD-MPC" is TD-MPC2.

| Agent | Metric | Paper nominal | Archive nominal | Current nominal | Paper hard | Archive hard | Current hard |
|---|---|---:|---:|---:|---:|---:|---:|
| PID | RMSE | 4.20 | 4.20 | 4.20 | 20.10 | 20.10 | 20.10 |
| PID | ActFluct | 0.07 | 0.07 | 0.07 | 0.17 | 0.17 | 0.17 |
| TD-MPC2 | RMSE | 3.81 ± 0.10 | 3.81 ± 0.10 | 3.80 ± 0.10 | 9.10 ± 0.69 | 9.10 ± 0.69 | 8.91 ± 0.54 |
| TD-MPC2 | ActFluct | 4.37 ± 0.48 | 4.37 ± 0.48 | 3.93 ± 0.58 | 6.02 ± 0.59 | 6.02 ± 0.59 | 5.63 ± 0.78 |
| PPO | RMSE | 5.29 ± 0.12 | 5.29 ± 0.12 | 5.29 ± 0.12 | 15.78 ± 2.48 | 15.78 ± 2.48 | 15.78 ± 2.48 |
| PPO | ActFluct | 1.98 ± 0.58 | 1.98 ± 0.58 | 1.97 ± 0.58 | 5.54 ± 0.93 | 5.54 ± 0.93 | 5.53 ± 0.93 |
| SAC | RMSE | 6.64 ± 0.83 | 6.64 ± 0.83 | 6.63 ± 0.84 | 11.15 ± 0.65 | 11.15 ± 0.65 | 11.15 ± 0.65 |
| SAC | ActFluct | 10.07 ± 1.12 | 10.07 ± 1.12 | 10.08 ± 1.12 | 12.00 ± 0.71 | 12.00 ± 0.71 | 12.00 ± 0.71 |

## Action-regulation variants

Archive / current, same units. CAPS does not apply to TD-MPC2.

| Variant | RMSE nominal | ActFluct nominal | RMSE hard | ActFluct hard |
|---|---:|---:|---:|---:|
| PPO CAPS | 5.05 ± 0.15 / 5.05 ± 0.15 | 1.47 ± 0.27 / 1.46 ± 0.26 | 19.17 ± 4.68 / 19.17 ± 4.68 | 4.56 ± 0.60 / 4.55 ± 0.60 |
| PPO AVP | 5.78 ± 0.21 / 5.78 ± 0.21 | 13.78 ± 3.16 / 13.78 ± 3.16 | 13.80 ± 0.82 / 13.80 ± 0.82 | 13.48 ± 2.50 / 13.48 ± 2.50 |
| SAC CAPS | 6.79 ± 0.74 / 6.79 ± 0.74 | 0.53 ± 0.27 / 0.53 ± 0.27 | 12.28 ± 1.17 / 12.28 ± 1.17 | 1.44 ± 0.55 / 1.44 ± 0.55 |
| SAC AVP | 6.54 ± 0.75 / 6.54 ± 0.75 | 0.50 ± 0.18 / 0.50 ± 0.18 | 15.14 ± 3.39 / 15.14 ± 3.39 | 1.20 ± 0.32 / 1.20 ± 0.32 |
| TD-MPC2 AVP | 3.58 ± 0.01 / 3.60 ± 0.01 | 0.54 ± 0.02 / 0.68 ± 0.01 | 8.09 ± 0.03 / 8.25 ± 0.14 | 0.60 ± 0.02 / 0.70 ± 0.03 |

All 75 runs passed validation; TD-MPC2 archive-file assignment is less clear
than PPO/SAC. Per-seed evidence:
[`2026-11-06_phase1_action_regulation`](../runs/2026-11-06_phase1_action_regulation/).

## Turbulence and gusts

Figures 2–3 use PPO CAPS, SAC CAPS and TD-MPC2 AVP checkpoints. Turbulence
has **light steady wind at every level, including `off`**; gusts have none.
Severities run sequentially in one process; all 64 runs used JSBSim 1.1.13.
The [aggregate CSV](../icinco_disturbance_aggregates.csv) holds every mean and
SEM.

### PID/PPO/SAC: largest accepted mean differences

Level with the largest nominal-reference mean change, archive → current.

| Condition | Controller | RMSE (rad) | ActFluct (rad/sample) |
|---|---|---|---|
| Turbulence | PID | \|Δ\| < 0.00000001 at all levels | \|Δ\| < 0.00000001 at all levels |
| Turbulence | PPO CAPS | Off: 0.05677053 → 0.05677004 (-0.00000049) | Off: 0.01869727 → 0.01868941 (-0.00000786) |
| Turbulence | SAC CAPS | Moderate: 0.08522816 → 0.08526340 (+0.00003524) | Severe: 0.02886086 → 0.02888208 (+0.00002122) |
| Gust | PID | \|Δ\| < 0.00000001 at all levels | \|Δ\| < 0.00000001 at all levels |
| Gust | PPO CAPS | Off: 0.05625862 → 0.05625760 (-0.00000102) | Off: 0.01470986 → 0.01471921 (+0.00000935) |
| Gust | SAC CAPS | Severe: 0.34897197 → 0.34892523 (-0.00004674) | Severe: 0.02967821 → 0.02969009 (+0.00001188) |

Largest per-axis RMSE changes: PID 0.00000001, PPO 0.00001273, SAC 0.00071496
rad; PPO seed-4508 easy/off gust ActFluct 0.000907 → 0.000838.

### TD-MPC2 AVP: archive → current (difference)

| Condition | Level | RMSE (rad) | ActFluct (rad/sample) |
|---|---|---|---|
| Turbulence | Off | 0.051344 ± 0.005189 → 0.057236 ± 0.011056 (+0.005892) | 0.006524 ± 0.000347 → 0.007233 ± 0.000301 (+0.000709) |
| Turbulence | Light | 0.061228 ± 0.006858 → 0.062488 ± 0.007758 (+0.001260) | 0.007697 ± 0.000422 → 0.008119 ± 0.000413 (+0.000422) |
| Turbulence | Moderate | 0.072314 ± 0.009109 → 0.065567 ± 0.007832 (-0.006747) | 0.008626 ± 0.000570 → 0.008754 ± 0.000493 (+0.000128) |
| Turbulence | Severe | 0.067795 ± 0.003597 → 0.070587 ± 0.005857 (+0.002792) | 0.010195 ± 0.000210 → 0.010403 ± 0.000185 (+0.000208) |
| Gust | Off | 0.051589 ± 0.002115 → 0.054246 ± 0.002979 (+0.002657) | 0.022592 ± 0.001966 → 0.018028 ± 0.002157 (-0.004564) |
| Gust | Light | 0.075541 ± 0.003264 → 0.082422 ± 0.006317 (+0.006881) | 0.027450 ± 0.002082 → 0.023302 ± 0.002253 (-0.004148) |
| Gust | Moderate | 0.116468 ± 0.003799 → 0.124647 ± 0.006088 (+0.008179) | 0.028693 ± 0.001692 → 0.024520 ± 0.001815 (-0.004173) |
| Gust | Severe | 0.238616 ± 0.015501 → 0.263331 ± 0.011311 (+0.024715) | 0.030651 ± 0.001427 → 0.026380 ± 0.001272 (-0.004271) |

At turbulence `off`, current PPO (0.056770 ± 0.002113 rad) beats TD-MPC2,
reversing the mean ranking (SEMs overlap); the other seven rankings and the
paper's conclusions hold. Seed 2164 medium/off contributes +0.005865 of the
+0.005892 rad; restoring only it gives 0.05137, below PPO's 0.05677.

### Turbulence-off reversal: targeted diagnosis

Seed 2164 medium/off RMSE (evaluation seed 10): 0.11387156 → 0.17251962
(+0.05864806 rad). June 2024 sources and current code with historical
inference both reproduce all six archive metrics exactly.

On this slice, the [production-mode replays](../runs/2026-11-06_tdmpc2_legacy_mode_check/)
match their legacy and current controls on all six metrics and 25 logged episode
rewards. Initial attempts were stopped after the protected `tdmpc2_default.yaml`
was briefly edited; it was restored exactly before these fresh retries, and all
204 protected input hashes match. These replays ran under the mode's earlier
name, `legacy_icinco2024`; it was later renamed `icinco2024` with no other code
change.

Historical choices (policy means, NumPy critic and elite selection, active
critic dropout) were restored together, not isolated. Current episode 10
(reward -912.1 vs -588.8) carries, by a conservative bound, **at least 48%**
of the squared error
([diagnostic evidence](../runs/2026-11-05_phase1_disturbances/tdmpc2_turbulence_off_investigation.json)).

### Historical TD-MPC2 inference across five seeds

Twenty sweeps with the [opt-in mode](../README.md#paper-era-td-mpc2-inference);
archive → historical:

| Condition | Level | RMSE (rad) | ActFluct (rad/sample) |
|---|---|---|---|
| Turbulence | Off | 0.051344 ± 0.005189 → 0.051344 ± 0.005189 (+0.000000) | 0.006524 ± 0.000347 → 0.006524 ± 0.000347 (+0.000000) |
| Turbulence | Light | 0.061228 ± 0.006858 → 0.061229 ± 0.006857 (+0.000001) | 0.007697 ± 0.000422 → 0.007697 ± 0.000422 (+0.000000) |
| Turbulence | Moderate | 0.072314 ± 0.009109 → 0.072284 ± 0.009120 (-0.000030) | 0.008626 ± 0.000570 → 0.008635 ± 0.000567 (+0.000009) |
| Turbulence | Severe | 0.067795 ± 0.003597 → 0.068375 ± 0.003757 (+0.000580) | 0.010195 ± 0.000210 → 0.010280 ± 0.000245 (+0.000085) |
| Gust | Off | 0.051589 ± 0.002115 → 0.051589 ± 0.002115 (+0.000000) | 0.022592 ± 0.001966 → 0.022592 ± 0.001966 (+0.000000) |
| Gust | Light | 0.075541 ± 0.003264 → 0.075541 ± 0.003264 (+0.000000) | 0.027450 ± 0.002082 → 0.027450 ± 0.002082 (+0.000000) |
| Gust | Moderate | 0.116468 ± 0.003799 → 0.116468 ± 0.003799 (+0.000000) | 0.028693 ± 0.001692 → 0.028693 ± 0.001692 (+0.000000) |
| Gust | Severe | 0.238616 ± 0.015501 → 0.238616 ± 0.015501 (+0.000000) | 0.030651 ± 0.001427 → 0.030651 ± 0.001427 (+0.000000) |

Exact matches per 10 rows, off/light/moderate/severe: turbulence `10/9/7/3`,
gust `10/10/10/10`, using the inferred mapping
`{1: 1, 2164: 2, 2989: 5, 4508: 4, 6930: 3}`. Archived rankings hold
throughout. The largest gap (turbulence/severe seed 2164 medium, +0.00385782
rad) is undiagnosed.

After a scheduler error, sweeps were rerun and independently validated. The
launcher omitted the evaluator from its input hashes; it was checked separately.
See the [historical-inference CSV](../tdmpc2_disturbance_historical_inference.csv)
and [analysis](../runs/2026-11-05_phase1_disturbances/historical_inference_five_seeds/analysis.json).

JSBSim 1.1.13 → 1.2.3 alone moves PID severe RMSE 0.05115078 → 0.04136873
(easy) and 0.09384774 → 0.08624148 (medium); this is not an accepted residual.
Running severe alone is not equivalent: PID easy pitch RMSE 0.050683774 →
0.050159547. Upstream changed turbulence history between
[1.1.13](https://github.com/JSBSim-Team/jsbsim/blob/v1.1.13/src/models/atmosphere/FGWinds.cpp#L284-L291)
and [1.2.3](https://github.com/JSBSim-Team/jsbsim/blob/v1.2.3/src/models/atmosphere/FGWinds.cpp#L130-L139);
the exact cause is not isolated.

Runs load 1.1.13 from `fw-old` into `fw-310`; results are in
[`2026-11-05_phase1_disturbances`](../runs/2026-11-05_phase1_disturbances/).

## How the paper aggregates seeds

Per level: five-seed mean, SEM = std / √5 (`compute_seed_statistics.py`).
"Nominal Refs" averages easy and medium means *and SEMs*; per-seed averaging
does not match (SAC RMSE SEM 0.74, not 0.83).

## Per-seed residuals

See [`per_seed.csv`](../runs/2026-11-05_phase1_nominal_five_seeds/per_seed.csv).
The SAC seed-1 easy residual also appears with June 2024 code.

## Archive file numbering

| Training seed | 1 | 2164 | 2989 | 4508 | 6930 |
|---|---:|---:|---:|---:|---:|
| Archive file | 1 | 2 | 5 | 4 | 3 |

Inferred from roll/pitch RMSE; only per-seed comparisons depend on it.

## Author decisions

Residuals are accepted as measured, not as a general tolerance.

| Scope | Decision |
|---|---|
| Nominal, all five seeds | Measured residuals accepted; method ranking remains unchanged. |
| Action regulation, all five seeds | Measured residuals accepted. |
| Turbulence and gusts | PID/PPO/SAC/TD-MPC2 measured residuals accepted. |
| TD-MPC2 inference | [Keep current inference as default](./2026-11-04_icinco_tdmpc2_inference_ablation.md#author-decision-retain-current-td-mpc2-2026-11-04); paper-era inference is opt-in. |

## Evidence

- Local, Git-ignored runs:
  [nominal](../runs/2026-11-05_phase1_nominal_five_seeds/),
  [action regulation](../runs/2026-11-06_phase1_action_regulation/),
  [disturbances](../runs/2026-11-05_phase1_disturbances/).
- Nominal source commits: FlightControl `d3ac16ca6c853cc5fc95358de7d01d8026416c72`
  plus recorded uncommitted work, TD-MPC2 `bce5911f4f9ced1a3d6bbc94f4f145b6fdb292b7`,
  FW-JSBGym `96e9cda0e0ca3d5eace0ca2ca9b4675e3d2b9332`.
- Older runs recorded NumPy distribution metadata, not the imported version;
  see the [run guide](../README.md) for runtime and protected-input caveats.
