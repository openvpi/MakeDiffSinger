"""Explicit sample selection and separately defined duration summaries."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from fnmatch import fnmatchcase
import math
from numbers import Real

from .audio import audio_info
from .catalog import sample_dataset
from .types import Annotation, BatchResult

QUALITY_METRICS = ("agreement", "confidence", "determinacy", "monotonicity")


@dataclass
class QualityRule:
    bottom_fraction: float
    group_by: str | None = None

    def __post_init__(self):
        if not 0 < self.bottom_fraction <= 1:
            raise ValueError("The review fraction must be in (0, 1].")
        if self.group_by not in {None, "speaker", "subset", "language"}:
            raise ValueError("group_by must be speaker, subset, language, or null.")


@dataclass
class Selection:
    samples: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    drop_bottom_fraction: float | None = None
    percentile_group: str | None = None
    quality_exclusions: dict[str, str] = field(default_factory=dict)


@dataclass
class Selected:
    annotations: list[Annotation]
    excluded: list[dict]
    criteria: dict
    details: dict = field(default_factory=dict)
    quality_removed: list[Annotation] = field(default_factory=list, repr=False)

    @property
    def quality_filter_applied(self):
        return self.criteria.get("drop_bottom_fraction") is not None or bool(self.criteria.get("quality_exclusions"))


def current_score(annotation, metric):
    if "quality" in annotation.stale:
        return None, "quality_outdated"
    value = annotation.quality.get(metric)
    if value is None:
        return None, "quality_missing"
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None, "quality_invalid"
    return float(value), None


def rank_quality(annotations):
    """Rank records worst-first using TIFA prediction diagnostics."""
    annotations = list(annotations)
    sample_dataset(a.sample for a in annotations)
    records, unavailable = [], []
    fields = ("num_skipped_tokens", *QUALITY_METRICS)
    for annotation in annotations:
        scores = []
        for field in fields:
            value, reason = current_score(annotation, field)
            if not reason and field == "num_skipped_tokens" and (value < 0 or not value.is_integer()):
                reason = "quality_invalid"
            if reason:
                unavailable.append({"key": annotation.key, "reason": reason})
                break
            scores.append(value)
        else:
            records.append((annotation, scores))

    # Include skipped tokens in the worst-rank comparison with the self-check metrics.
    ranks = []
    for index, field in enumerate(fields):
        ordered = sorted(records, key=lambda record: record[1][index], reverse=field == "num_skipped_tokens")
        ranks.append({annotation.key: rank for rank, (annotation, _) in enumerate(ordered)})

    def sort_key(record):
        annotation, _ = record
        metric_ranks = tuple(rank[annotation.key] for rank in ranks)
        return min(metric_ranks), *metric_ranks

    records.sort(key=sort_key)
    return [annotation for annotation, _ in records], unavailable


def rank_quality_groups(annotations, group_by):
    groups = defaultdict(list)
    for annotation in annotations:
        group = getattr(annotation.sample, group_by) if group_by else None
        groups[group].append(annotation)
    for group, entries in groups.items():
        ordered, unavailable = rank_quality(entries)
        yield group, ordered, unavailable


def select_quality(annotations, rule: QualityRule) -> Selected:
    """Select the worst fraction by alignment quality rank within each group."""
    scope = annotations if isinstance(annotations, Selected) else Selected(list(annotations), [], {})
    sample_dataset(a.sample for a in scope.annotations)
    selected, excluded = [], list(scope.excluded)
    populations, selected_details = [], []
    eligible = 0
    fraction = Decimal(str(rule.bottom_fraction))
    for group, ordered, unavailable in rank_quality_groups(scope.annotations, rule.group_by):
        eligible += len(ordered)
        cutoff = int(len(ordered) * fraction)
        chosen = ordered[:cutoff]
        selected.extend(chosen)
        selected_details.extend({"key": a.key, "group": group, "rank": rank}
                                for rank, a in enumerate(chosen, 1))
        excluded.extend(unavailable)
        excluded.extend({"key": a.key, "reason": "outside_quality_rule"} for a in ordered[cutoff:])
        populations.append({"group": group, "eligible": len(ordered), "selected": len(chosen)})
    return Selected(selected, excluded, {"scope": scope.criteria, "quality": asdict(rule)}, {
        "population": len(scope.annotations), "eligible": eligible,
        "unavailable": len(scope.annotations) - eligible, "groups": populations,
        "selected": selected_details,
    })


def select(annotations, criteria: Selection | None = None) -> Selected:
    criteria = criteria or Selection()
    annotations = list(annotations)
    sample_dataset(a.sample for a in annotations)
    if criteria.drop_bottom_fraction is not None and not 0 <= criteria.drop_bottom_fraction < 1:
        raise ValueError("drop_bottom_fraction must be in [0, 1).")
    if criteria.percentile_group not in {None, "speaker", "subset", "language"}:
        raise ValueError("percentile_group must be speaker, subset, language, or null.")
    unknown = criteria.quality_exclusions.keys() - {a.key for a in annotations}
    if unknown:
        raise ValueError(f"Unknown quality exclusion keys: {', '.join(sorted(unknown))}")
    if any(not isinstance(reason, str) or not reason.strip() for reason in criteria.quality_exclusions.values()):
        raise ValueError("Each explicit quality exclusion needs an annotation defect as its reason.")
    kept, excluded, removed = [], [], []
    for annotation in annotations:
        sample, reason = annotation.sample, None
        if criteria.samples and not any(fnmatchcase(sample.key, p) or fnmatchcase(sample.id, p) for p in criteria.samples):
            reason = "sample_filter"
        for field_name, allowed in (("speaker", criteria.speakers), ("subset", criteria.subsets),
                                    ("language", criteria.languages)):
            if allowed and not any(fnmatchcase(getattr(sample, field_name) or "", pattern) for pattern in allowed):
                reason = field_name + "_filter"
        if criteria.statuses and annotation.status not in criteria.statuses:
            reason = "status_filter"
        if reason:
            excluded.append({"key": sample.key, "reason": reason})
        elif sample.key in criteria.quality_exclusions:
            excluded.append({"key": sample.key, "reason": "annotation_defect",
                             "detail": criteria.quality_exclusions[sample.key]})
            removed.append(annotation)
        else:
            kept.append(annotation)
    if not criteria.drop_bottom_fraction:
        return Selected(kept, excluded, asdict(criteria), quality_removed=removed)
    rule = QualityRule(criteria.drop_bottom_fraction, group_by=criteria.percentile_group)
    partition = select_quality(kept, rule)
    excluded.extend({"key": a.key, "reason": "bottom_quality_fraction"} for a in partition.annotations)
    tail_keys = {a.key for a in partition.annotations}
    return Selected([a for a in kept if a.key not in tail_keys], excluded, asdict(criteria),
                    {"quality_steps": [{"rule": asdict(rule), **partition.details}]}, removed + partition.annotations)


def summarize(items) -> dict:
    outcomes = items.outcomes if isinstance(items, BatchResult) else None
    annotations = [o.annotation for o in outcomes if o.annotation is not None] if outcomes is not None else list(items)
    samples = [o.sample for o in outcomes] if outcomes is not None else [a.sample for a in annotations]
    sources, clip_seconds, unavailable = {}, 0.0, []
    for sample in samples:
        try:
            info = audio_info(sample.audio)
            source = sample.metadata.get("origin_recording")
            if source:
                sources[source["path"]] = source["frames"] / source["sample_rate"]
            else:
                sources[sample.audio.path or sample.key] = info["source_seconds"]
            clip_seconds += info["clip_seconds"]
        except (OSError, ValueError, RuntimeError, ImportError):
            unavailable.append(sample.key)
    confidence = sorted(value for a in annotations if (value := current_score(a, "confidence")[0]) is not None)
    return {"sample_count": len(samples), "source_recordings": len(sources),
            "source_seconds": sum(sources.values()), "clip_seconds": clip_seconds,
            "annotation_seconds": sum(a.duration for a in annotations),
            "statuses": dict(Counter(o.status for o in outcomes)) if outcomes is not None
                        else dict(Counter(a.status for a in annotations)),
            "audio_unavailable": unavailable,
            "confidence": {"count": len(confidence), "minimum": min(confidence, default=None),
                           "maximum": max(confidence, default=None)},
            "samples_with_skipped_tokens": sum((current_score(a, "num_skipped_tokens")[0] or 0) > 0 for a in annotations)}
