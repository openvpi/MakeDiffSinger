"""Aggregate existing alignment diagnostics without copying sample assets."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import os
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .selection import QUALITY_METRICS, current_score, rank_quality, rank_quality_groups, select
from .config import Resources
from .catalog import sample_dataset


def quality_summary(annotations, *, keep_input_order=False):
    annotations = list(annotations)
    sample_dataset(a.sample for a in annotations)
    diagnosis = [{"identifier": annotation.key} for annotation in annotations]
    metrics = {}
    for metric in QUALITY_METRICS:
        values, missing, invalid, stale = [], 0, 0, 0
        for annotation, record in zip(annotations, diagnosis):
            value, reason = current_score(annotation, metric)
            if reason == "quality_outdated":
                stale += 1
            elif reason == "quality_invalid":
                invalid += 1
            elif reason:
                missing += 1
            else:
                values.append(value)
                record[metric] = value
        summary = {"count": len(values), "missing": missing, "invalid": invalid, "stale": stale}
        if values:
            quantiles = np.quantile(values, [.05, .5, .95]).tolist()
            summary.update(minimum=min(values), maximum=max(values), mean=float(np.mean(values)),
                           p05=quantiles[0], median=quantiles[1], p95=quantiles[2])
        metrics[metric] = summary
    for annotation, record in zip(annotations, diagnosis):
        for key in ("num_frames", "num_tokens", "num_skipped_tokens"):
            value, _ = current_score(annotation, key)
            if value is not None and value >= 0 and value.is_integer():
                record[key] = int(value)
    skipped = [record["num_skipped_tokens"] for record in diagnosis if "num_skipped_tokens" in record]
    if not keep_input_order:
        ranked, unavailable = rank_quality(annotations)
        records = {record["identifier"]: record for record in diagnosis}
        diagnosis = [records[a.key] for a in ranked] + [records[item["key"]] for item in unavailable]
    return {"sample_count": len(annotations), "metrics": metrics,
            "diagnosis": diagnosis,
            "pronunciations": [{"identifier": a.key, "words": a.quality["pronunciations"]} for a in annotations
                               if a.quality.get("pronunciations") and "quality" not in a.stale],
            "skipped_tokens": {"diagnosed_samples": len(skipped),
                               "affected_samples": sum(value > 0 for value in skipped), "total": sum(skipped)}}


def quality_tail_summary(annotations, criteria, *, bands=10, preview_fraction=None):
    """Summarize rank bands and optionally preview the normal exclusion rule."""
    if bands < 1:
        raise ValueError("The number of rank bands must be positive.")
    criteria = replace(criteria, drop_bottom_fraction=None)
    scope = select(annotations, criteria)
    fields = ("num_skipped_tokens", *QUALITY_METRICS)
    preview = {}
    if preview_fraction is not None and preview_fraction != 0:
        trial = select(scope.annotations, replace(criteria, drop_bottom_fraction=preview_fraction, quality_exclusions={}))
        preview = {item["group"]: item["selected"] for item in trial.details["quality_steps"][0]["groups"]}

    groups = []
    for group, ranked, unavailable in rank_quality_groups(scope.annotations, criteria.percentile_group):
        size = len(ranked)
        band_count = min(size, bands)
        rows = []
        for index in range(band_count):
            start = index * size // band_count
            end = (index + 1) * size // band_count
            metrics = {}
            for field in fields:
                values = [item.quality[field] for item in ranked[start:end]]
                metrics[field] = {"min": float(min(values)), "median": float(np.median(values)),
                                  "max": float(max(values))}
            rows.append({"first_rank": start + 1, "last_rank": end, "metrics": metrics})
        result = {"group": group, "eligible": size,
                  "unavailable": dict(Counter(item["reason"] for item in unavailable)), "bands": rows}
        if preview_fraction is not None:
            cutoff = preview.get(group, 0)
            result["preview"] = {"ranked_excluded": cutoff,
                                 "unranked_retained": len(unavailable),
                                 "retained": size + len(unavailable) - cutoff}
            for label, index in (("last_excluded", cutoff - 1), ("first_retained", cutoff)):
                if 0 <= index < size:
                    item = ranked[index]
                    result["preview"][label] = {"key": item.key, "rank": index + 1,
                                                "metrics": {field: float(item.quality[field]) for field in fields}}
        groups.append(result)

    report = {"scope": scope.criteria, "population": len(scope.annotations), "groups": groups}
    if preview_fraction is not None:
        report["preview"] = {"fraction": preview_fraction, **{
            field: sum(group["preview"][field] for group in groups)
            for field in ("ranked_excluded", "unranked_retained", "retained")}}
    return report


def write_overview(report, output, *, plots=True, tifa_repo=None):
    """Publish TIFA's native statistics files for complete, current diagnoses."""
    output = Path(output).resolve()
    quality = report["quality"]
    required = {*QUALITY_METRICS, "num_frames", "num_tokens", "num_skipped_tokens"}
    records = [r for r in quality["diagnosis"] if required <= r.keys()]
    ids = {r["identifier"] for r in records}
    scores = [r for r in quality.get("pronunciations", []) if r["identifier"] in ids]
    names = ["diagnosis.json", "scores.json", *(f"{m}_{plot}.jpg" for m in QUALITY_METRICS
             for plot in ("histogram", "scatter"))]
    if records:
        repo = Path(tifa_repo) if tifa_repo else Resources().repo("tifa")
        worker = Path(__file__).parent / "backends/tifa_statistics.py"
        with TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(worker), tmp], cwd=repo,
                                    input=json.dumps([records, scores], allow_nan=False).encode(),
                                    env={**os.environ, "MPLBACKEND": "Agg"}, stderr=subprocess.PIPE)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            files = [Path(tmp) / name for name in names]
            if not all(path.is_file() for path in files):
                raise RuntimeError("The forced aligner's statistics bundle is incomplete.")
            output.mkdir(parents=True, exist_ok=True)
            for path in files:
                copyfile(path, output / path.name)
    else:
        for name in names:
            (output / name).unlink(missing_ok=True)
    return len(records)


