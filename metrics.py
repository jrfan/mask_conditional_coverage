import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from project import file_hash, identity, read_json


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def mask_keys(rows):
    return ["".join(str(int(v)) for v in row) for row in rows]


def counts_by_mask(rows):
    masks, counts = np.unique(rows, axis=0, return_counts=True)
    return dict(zip(mask_keys(masks), counts.tolist()))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 and np.isfinite(values).all() else None
    return dict(mean=float(values.mean()), ci95_halfwidth=half, n=len(values))


def paper_metrics(pooled, target):
    result = {}
    for metric in ("coverage", "shortfall", "length"):
        values = np.asarray([max(target-p["coverage"], 0) if metric == "shortfall" else p[metric] for p in pooled], dtype=float)
        result[metric] = dict(mean=float(values.mean()), se_masks=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 and np.isfinite(values).all() else None)
    return result


def prediction_metrics(path, seed, protocol):
    with np.load(path, allow_pickle=False) as z:
        y, lo, hi, masks = (z[k] for k in ("y", "lower", "upper", "masks"))
        if y.ndim != 1 or lo.shape != y.shape or hi.shape != y.shape or masks.ndim != 2 or len(masks) != len(y):
            raise ValueError(f"Prediction shape mismatch: {path}")
        if not np.isfinite(y).all() or np.isnan(lo).any() or np.isnan(hi).any():
            raise ValueError(f"Invalid observations or endpoints: {path}")
        if not np.isin(masks, [0, 1]).all():
            raise ValueError(f"Invalid masks: {path}")
        covered = (lo <= y) & (y <= hi)
        with np.errstate(invalid="ignore"):
            lengths = np.maximum(hi-lo, 0)
        if np.isnan(lengths).any():
            raise ValueError(f"Undefined interval length: {path}")
        cal, train = (counts_by_mask(z[k]) for k in ("calibration_masks", "training_masks"))
        unique, inverse, counts = np.unique(masks, axis=0, return_inverse=True, return_counts=True)
        cov = np.bincount(inverse, weights=covered.astype(float))/counts
        length = np.bincount(inverse, weights=lengths)/counts
        keys = mask_keys(unique)
        patterns = [dict(mask=key, coverage=float(cov[i]), length=float(length[i]), test_count=int(counts[i]),
                         cal_count=cal.get(key, 0), train_count=train.get(key, 0)) for i, key in enumerate(keys)]
        return dict(seed=seed, patterns=patterns, protocol=protocol,
            empirical_coverage=float(covered.mean()), mask_mean_coverage=float(cov.mean()),
            mask_mean_shortfall=float(np.maximum(1-protocol["args"]["alpha"]-cov, 0).mean()),
            sample_mean_length=float(lengths.mean()), mask_mean_length=float(length.mean()),
            infinite_interval_fraction=float(np.isinf(lengths).mean()))


def summarize_rows(rows):
    by_mask = {}
    for row in rows:
        for p in row["patterns"]:
            by_mask.setdefault(p["mask"], []).append(p)
    target = 1-rows[0]["protocol"]["args"]["alpha"]
    pooled = [dict(mask=mask, coverage=float(np.mean([p["coverage"] for p in ps])),
                   length=float(np.mean([p["length"] for p in ps])), seed_count=len(ps)) for mask, ps in sorted(by_mask.items())]
    result = dict(complete=True, seeds=[r["seed"] for r in rows], pooled_masks=pooled,
                  paper_metrics=paper_metrics(pooled, target),
                  masks_below_nominal_after_seed_averaging=sum(p["coverage"] < target for p in pooled))
    for metric in ("empirical_coverage", "mask_mean_coverage", "mask_mean_shortfall", "sample_mean_length", "mask_mean_length", "infinite_interval_fraction"):
        result[metric] = mean_ci([r[metric] for r in rows])
    result["pooled_mask_shortfall"] = result["paper_metrics"]["shortfall"]["mean"]
    d = len(pooled[0]["mask"])
    coordinates = {}
    for j in range(d):
        estimates = []
        for row in rows:
            ps = [p for p in row["patterns"] if p["mask"][j] == "1"]
            weights = np.asarray([p["train_count"]+p["cal_count"] for p in ps])
            if weights.sum():
                estimates.append(float(np.average([p["coverage"] for p in ps], weights=weights)))
        coordinates[f"M{j+1}=1"] = mean_ci(estimates) if estimates else None
    result["coordinate_coverage_plugin_weighted"] = coordinates
    return result


def subset_metrics(rows, observed):
    by_mask = {}
    for row in rows:
        for p in row["patterns"]:
            if (p["cal_count"] > 0) == observed:
                by_mask.setdefault(p["mask"], []).append(dict(p, seed=row["seed"]))
    if not by_mask:
        raise ValueError("No masks in the requested calibration-presence subset")
    pooled = [dict(mask=mask, coverage=mean_ci([p["coverage"] for p in ps]),
        length=float(np.mean([p["length"] for p in ps])), cal_count=float(np.mean([p["cal_count"] for p in ps])),
        seeds=[p["seed"] for p in ps]) for mask, ps in sorted(by_mask.items())]
    plain = [dict(p, coverage=p["coverage"]["mean"]) for p in pooled]
    return dict(observed=observed, pooled_masks=pooled, paper_metrics=paper_metrics(plain, 1-rows[0]["protocol"]["args"]["alpha"]))


def collect(results, configuration):
    grouped, summaries = {}, {}
    for cid, entry in configuration.items():
        rows = []
        common = None
        for seed in range(entry["parameters"]["reps"]):
            directory = Path(results) / cid / f"seed{seed}"
            status = read_json(directory / "status.json")
            if status["status"] != "completed" or status["returncode"] != 0 or status["seed"] != seed:
                raise ValueError(f"Incomplete run: {directory}")
            attempt = directory / status["attempt"]
            request = read_json(attempt / "request.json")
            specification = {k: request[k] for k in ("id", "engine", "parameters", "data", "sources")}
            if request["id"] != cid or request["seed"] != seed or request["engine"] != entry["engine"] or request["parameters"] != entry["parameters"]:
                raise ValueError(f"Configuration mismatch: {directory}")
            if status["identity"] != request["identity"] or identity(specification) != status["identity"]:
                raise ValueError(f"Run identity mismatch: {directory}")
            if common is not None and common != specification:
                raise ValueError(f"Mixed configuration, data or source across seeds: {cid}")
            common = specification
            for name in ("predictions.npz", "protocol.json", "patterns.csv"):
                if name not in status["artifacts"]:
                    raise ValueError(f"Missing artifact: {directory / name}")
            for name, digest in status["artifacts"].items():
                if file_hash(attempt / name) != digest:
                    raise ValueError(f"Changed artifact: {attempt / name}")
            protocol = read_json(attempt / "protocol.json")
            expected_protocol = "training_only_v1" if entry["engine"] == "part2" else "training_only_v2"
            if protocol["replication"] != seed or protocol["protocol"] != expected_protocol:
                raise ValueError(f"Protocol mismatch: {directory}")
            protocol["args"] = dict(entry["parameters"])
            row = prediction_metrics(attempt / "predictions.npz", seed, protocol)
            if (attempt / "representation.csv").exists():
                frame = pd.read_csv(attempt / "representation.csv", dtype={"Mask": str})
                if frame["Mask"].duplicated().any() or (frame["n"] <= 0).any() or (frame["ss_rho"] < 0).any() or set(frame["epoch"]) != {seed}:
                    raise ValueError(f"Invalid representation diagnostics: {directory}")
                alpha = entry["parameters"]["alpha"]
                diagnostics = np.clip(1-alpha-max(alpha, 1-alpha)*np.sqrt(frame["ss_rho"]/frame["n"]), 0, 1)
                if not np.isfinite(diagnostics).all():
                    raise ValueError(f"Nonfinite representation diagnostics: {directory}")
                row["diagnostic"] = dict(zip(frame["Mask"], diagnostics.tolist()))
            rows.append(row)
        grouped[cid] = rows
        summaries[cid] = summarize_rows(rows)
    return dict(configs=summaries), grouped
