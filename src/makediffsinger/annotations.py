"""Layer operations without model-specific pitch or rest overrides."""

import hashlib
import json
import math
from pathlib import Path

import textgrid

from .types import Annotation, Interval

TIERS = ("texts", "words", "phones", "notes")
TIME_EPS = 0.00002
SERIALIZATION_EPS = 1e-10


def fill_gaps(intervals: list[Interval], duration: float, label="SP") -> list[Interval]:
    result = []
    cursor = 0.0
    for item in intervals:
        if not math.isfinite(item.start) or not math.isfinite(item.end) or item.end <= item.start:
            raise ValueError(f"Invalid interval: {item}")
        if item.start < cursor - TIME_EPS:
            raise ValueError(f"Overlapping intervals at {item.start}.")
        if item.start > cursor + 1e-10:
            result.append(Interval(cursor, item.start, label))
        result.append(Interval(item.start, item.end, item.label if item.label.strip() else label))
        cursor = item.end
    if cursor > duration + TIME_EPS:
        raise ValueError("Intervals extend past the annotation domain.")
    if duration > cursor + 1e-10:
        result.append(Interval(cursor, duration, label))
    return result


def aggregate_owners(phones: list[Interval], owners: list[int], labels: list[str]) -> list[Interval]:
    result = []
    index = 0
    while index < len(phones):
        end = index + 1
        while end < len(phones) and owners[end] == owners[index]:
            end += 1
        result.append(Interval(phones[index].start, phones[end - 1].end, labels[owners[index] - 1]))
        index = end
    return result


def tier_signatures(tiers: dict[str, list[Interval]]) -> dict[str, str]:
    """Small semantic fingerprints detect edits, ignoring sub-nanosecond text round trips."""
    return {name: hashlib.sha256(json.dumps(
        [(round(i.start, 9), round(i.end, 9), i.label) for i in items],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest() for name, items in tiers.items()}


def write_textgrid(annotation: Annotation, path: str | Path) -> None:
    grid = textgrid.TextGrid(minTime=0, maxTime=annotation.duration)
    for name in TIERS:
        if name not in annotation.tiers:
            continue
        tier = textgrid.IntervalTier(name, minTime=0, maxTime=annotation.duration)
        intervals = annotation.tiers[name]
        for index, item in enumerate(intervals):
            # TextGrid requires exact domain equality; equivalent frame times
            # may differ by a few floating-point ULPs after a file round trip.
            end = annotation.duration if (
                index == len(intervals) - 1 and abs(item.end - annotation.duration) < SERIALIZATION_EPS
            ) else item.end
            tier.add(item.start, end, item.label)
        grid.append(tier)
    grid.write(str(path))


def read_textgrid(path: str | Path, tier_map: dict[str, str] | None = None) -> tuple[float, dict]:
    grid = textgrid.TextGrid()
    grid.read(str(path), round_digits=15)
    if abs(grid.minTime) > TIME_EPS:
        raise ValueError("Clip annotations must use a zero-based time domain; map source offsets explicitly.")
    tiers = {}
    for tier in grid:
        name = tier_map.get(tier.name, tier.name) if tier_map is not None else tier.name
        if name not in TIERS:
            continue
        if name in tiers:
            raise ValueError(f"Multiple source tiers map to {name!r}.")
        if not isinstance(tier, textgrid.IntervalTier):
            raise ValueError(f"Expected an IntervalTier for {name!r}.")
        tiers[name] = [Interval(float(i.minTime), float(i.maxTime), i.mark) for i in tier]
    if not tiers:
        raise ValueError("No recognized layers; provide an explicit source tier mapping.")
    return float(grid.maxTime), tiers
