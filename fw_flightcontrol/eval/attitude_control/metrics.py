"""Attitude-control metrics and configurable checkpoint/target aggregation."""

from collections import defaultdict
import math

import numpy as np


METRICS = (
    "roll_rmse", "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct",
    "avg_rmse", "avg_fcs_fluct",
)
FLUCTUATION_METRICS = ("roll_fcs_fluct", "pitch_fcs_fluct", "avg_fcs_fluct")


def compute_metrics(observations, fluctuations):
    states = np.asarray(observations)
    if states.ndim != 2 or states.shape[1] < 8 or not len(states):
        raise ValueError("Expected nonempty attitude observations with error columns 6/7")
    if not np.isfinite(states).all():
        raise ValueError("Non-finite observations")
    # Keep the paper's separate float32 reductions, not an axis-wise pooled reduction.
    rmse = np.array([
        np.sqrt(np.mean(np.square(states[:, 6]))),
        np.sqrt(np.mean(np.square(states[:, 7]))),
    ])
    result = {
        "roll_rmse": float(rmse[0]),
        "pitch_rmse": float(rmse[1]),
        "roll_fcs_fluct": None,
        "pitch_fcs_fluct": None,
        "avg_rmse": float(np.mean(rmse)),
        "avg_fcs_fluct": None,
    }
    if len(fluctuations):
        surfaces = np.asarray(fluctuations)
        if surfaces.ndim != 2 or surfaces.shape[1] < 2:
            raise ValueError("Expected completed-episode control-surface fluctuations")
        surfaces = surfaces[:, :2]
        if not np.isfinite(surfaces).all():
            raise ValueError("Non-finite control-surface fluctuations")
        fluct = np.mean(surfaces, axis=0)
        result.update({
            "roll_fcs_fluct": float(fluct[0]),
            "pitch_fcs_fluct": float(fluct[1]),
            "avg_fcs_fluct": float(np.mean(fluct)),
        })
    return result


def aggregate(rows, target_groups=()):
    """Aggregate checkpoint results while retaining target, severity and policy seed."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("algorithm", "target", "severity", "policy_seed"))
        identity = (*key, row["checkpoint"])
        if identity in seen:
            raise ValueError(f"Duplicate checkpoint result: {identity}")
        seen.add(identity)
        if not all(
            (row[m] is None and m in FLUCTUATION_METRICS)
            or (row[m] is not None and math.isfinite(float(row[m])) and float(row[m]) >= 0)
            for m in METRICS
        ):
            raise ValueError(f"Invalid metrics for {identity}")
        groups[key].append(row)
    result = []
    for key, checkpoints in sorted(groups.items()):
        item = dict(zip(
            ("algorithm", "target", "severity", "policy_seed"), key,
        ))
        item["n_checkpoints"] = len(checkpoints)
        item["episodes_attempted"] = sum(row["episodes_attempted"] for row in checkpoints)
        item["episodes_out_of_bounds"] = sum(row["episodes_out_of_bounds"] for row in checkpoints)
        item["out_of_bounds_error"] = item["episodes_out_of_bounds"] > 0
        item["episodes_fluctuation_included"] = sum(
            row["episodes_fluctuation_included"] for row in checkpoints
        )
        item["episodes_fluctuation_unavailable"] = sum(
            row["episodes_fluctuation_unavailable"] for row in checkpoints
        )
        for metric in METRICS:
            if any(checkpoint[metric] is None for checkpoint in checkpoints):
                item[metric] = None
                item[f"{metric}_sem"] = None
                continue
            values = np.array([float(checkpoint[metric]) for checkpoint in checkpoints])
            item[metric] = float(np.mean(values))
            if len(values) > 1:
                item[f"{metric}_sem"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            else:
                item[f"{metric}_sem"] = None
        result.append(item)
    if target_groups:
        by_condition = defaultdict(dict)
        for item in result:
            scope = tuple(item[key] for key in ("algorithm", "severity", "policy_seed"))
            by_condition[scope][item["target"]] = item
        for target_group in target_groups:
            names = list(target_group.targets)
            if not names or target_group.method != "mean_of_means":
                raise ValueError(f"Unsupported target aggregation: {target_group}")
            if target_group.sem != "mean_of_sems":
                raise ValueError(f"Unsupported target SEM aggregation: {target_group}")
            for scope, targets in sorted(by_condition.items()):
                if not set(names) <= targets.keys():
                    continue
                source_rows = [targets[name] for name in names]
                identity_sets = [
                    {row["checkpoint"] for row in groups[
                        (source["algorithm"], source["target"], source["severity"],
                         source["policy_seed"])
                    ]}
                    for source in source_rows
                ]
                if any(identities != identity_sets[0] for identities in identity_sets[1:]):
                    raise ValueError(
                        f"Checkpoint sets differ across target group {target_group.name}: {scope}"
                    )
                combined = {
                    **source_rows[0],
                    "target": target_group.name,
                    "n_checkpoints": len(identity_sets[0]),
                    "episodes_attempted": sum(
                        row["episodes_attempted"] for row in source_rows
                    ),
                    "episodes_out_of_bounds": sum(
                        row["episodes_out_of_bounds"] for row in source_rows
                    ),
                    "out_of_bounds_error": any(
                        row["out_of_bounds_error"] for row in source_rows
                    ),
                    "episodes_fluctuation_unavailable": sum(
                        row["episodes_fluctuation_unavailable"] for row in source_rows
                    ),
                    "episodes_fluctuation_included": sum(
                        row["episodes_fluctuation_included"] for row in source_rows
                    ),
                }
                for metric in METRICS:
                    metric_values = [row[metric] for row in source_rows]
                    if all(value is not None for value in metric_values):
                        combined[metric] = float(np.mean(metric_values))
                    else:
                        combined[metric] = None
                    sem_values = [row[f"{metric}_sem"] for row in source_rows]
                    if all(value is not None for value in sem_values):
                        combined[f"{metric}_sem"] = float(np.mean(sem_values))
                    else:
                        combined[f"{metric}_sem"] = None
                result.append(combined)
    return result
