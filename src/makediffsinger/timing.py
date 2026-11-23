"""Match annotation boundaries in source time and project ordered layers to model frames."""

from bisect import bisect_left, bisect_right
from dataclasses import replace
import math

import numpy as np

from .types import Interval, Issue


BOUNDARY_TOLERANCE_MS = 10
AUDIO_WARNING_MS = 10
JOIN_TOLERANCE_MS = 1
# Defaults of GAME's features schema for standalone conversion without a checkpoint.
DEFAULT_GAME_GRID = {"hop_size": 441, "sample_rate": 44100}


def milliseconds(value):
    if type(value) is not int or value < 0:
        raise ValueError("Timing tolerances must be nonnegative integer milliseconds.")
    return value


def tolerance(annotation):
    return milliseconds(annotation.provenance.get("boundary_tolerance_ms", BOUNDARY_TOLERANCE_MS))


def near(a, b, ms):
    # Include a boundary exactly at the limit after seconds-to-float conversion.
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= ms / 1000 + max(math.ulp(a), math.ulp(b))


def audio_warning(annotation, seconds):
    limit = milliseconds(annotation.provenance.get("audio_warning_ms", AUDIO_WARNING_MS))
    if not near(annotation.duration, seconds, limit):
        return Issue("audio_endpoint", f"Annotation {annotation.duration * 1000:.0f} ms, audio "
                     f"{seconds * 1000:.0f} ms (difference {abs(seconds - annotation.duration) * 1000:.0f} ms, "
                     f"warning threshold {limit} ms).", severity="warning")


def nearest_frame(seconds, timestep):
    # GAME rounds accumulated float32 boundary times to the nearest even frame.
    return int(np.rint(np.float32(seconds) / np.float32(timestep)))


def covered_frames(duration, timestep, audio_seconds=None):
    covered = duration if audio_seconds is None else min(duration, audio_seconds)
    frames = nearest_frame(covered, timestep)
    if frames < 1:
        raise ValueError("The labelled audio range has no representable model frame.")
    return frames


def tier_edges(tier, duration, endpoint_ms=JOIN_TOLERANCE_MS):
    """Return a continuous boundary sequence, absorbing serialization differences."""
    if not tier:
        raise ValueError("Empty interval tier.")
    edges = [0.]
    for interval in tier:
        if not (math.isfinite(interval.start) and math.isfinite(interval.end)
                and interval.end > interval.start and interval.end > edges[-1]):
            raise ValueError(f"Invalid interval: {interval}")
        if not near(edges[-1], interval.start, JOIN_TOLERANCE_MS):
            raise ValueError(f"Gap or overlap at {interval.start:.6f} s.")
        edges.append(interval.end)
    if not near(edges[-1], duration, endpoint_ms) or duration <= edges[-2]:
        raise ValueError(f"Tier ends at {edges[-1]:.6f} s, expected {duration:.6f} s.")
    edges[-1] = duration
    return edges


def match_boundaries(query, anchors, ms):
    """Minimum-total-distance ordered one-to-one match with fixed endpoints."""
    milliseconds(ms)
    if len(query) > len(anchors) or not near(query[0], anchors[0], ms) or not near(query[-1], anchors[-1], ms):
        raise ValueError("Boundary count or endpoints cannot match the reference tier.")
    costs = {0: 0.}
    parents = []
    for time in query[1:-1]:
        slack = ms / 1000 + math.ulp(time)
        start = bisect_left(anchors, time - slack, 1)
        stop = bisect_right(anchors, time + slack, start, len(anchors) - 1)
        previous = iter(costs)
        index = next(previous, None)
        best = None
        row, links = {}, {}
        for candidate in range(start, stop):
            while index is not None and index < candidate:
                if best is None or costs[index] < costs[best]:
                    best = index
                index = next(previous, None)
            if best is not None:
                links[candidate] = best
                row[candidate] = costs[best] + abs(time - anchors[candidate])
        if not row:
            raise ValueError(f"No ordered boundary match for {time * 1000:.2f} ms within {ms} ms.")
        parents.append(links)
        costs = row
    last = min(costs, key=costs.get)
    result = [len(anchors) - 1]
    for links in reversed(parents):
        result.append(last)
        last = links[last]
    return [0, *reversed(result)]


def frame_coordinates(layers, timestep, end_frame, tolerance_ms):
    """Reserve one frame per interval on each tier; independent tiers may share frames."""
    times = sorted({t for edges in layers.values() for t in edges})
    successors = {t: [] for t in times}
    for edges in layers.values():
        for left, right in zip(edges, edges[1:]):
            successors[left].append(right)
    slack = int(tolerance_ms / (1000 * timestep))
    targets = {t: nearest_frame(t, timestep) for t in times}
    targets[times[-1]] = end_frame
    latest = {}
    upper = end_frame
    for t in reversed(times):
        upper = min([upper, targets[t] + slack, *(latest[right] - 1 for right in successors[t])])
        if upper < max(0, targets[t] - slack):
            tiers = ", ".join(name for name, edges in layers.items() if t in edges)
            raise ValueError(f"Cannot preserve positive {tiers} intervals near {t * 1000:.2f} ms "
                             f"within {tolerance_ms} ms on the {end_frame}-frame audio grid.")
        latest[t] = upper
    lower, required, frames = 0, {}, {}
    for t in times:
        lower = max(lower, required.get(t, 0), targets[t] - slack)
        frame = min(latest[t], max(lower, targets[t]))
        frames[t] = frame * timestep
        for right in successors[t]:
            required[right] = max(required.get(right, 0), frame + 1)
        lower = frame
    frames[times[-1]] = end_frame * timestep
    return frames


def align_tiers(annotation, syllables, *, timestep=None, end_frame=None):
    """Return new intervals sharing phone/group and syllable/note boundaries."""
    ms = tolerance(annotation)
    layers = {name: tier_edges(tier, annotation.duration, ms) for name, tier in annotation.tiers.items()}
    phones = layers["phones"]
    for name in ("words", "texts"):
        if name not in layers:
            continue
        try:
            indices = match_boundaries(layers[name], phones, ms)
        except ValueError:
            # Written text and imported auxiliary word tiers may have independent segmentation.
            if name == "words" and "imported_ph_num" not in annotation.provenance:
                raise
        else:
            layers[name] = [phones[i] for i in indices]
    if "notes" in layers:
        bounds = [phones[s.first_phone] for s in syllables] + [phones[-1]]
        indices = match_boundaries(bounds, layers["notes"], ms)
        for i, time in zip(indices, bounds):
            layers["notes"][i] = time
        if any(b <= a for a, b in zip(layers["notes"], layers["notes"][1:])):
            raise ValueError("Matching syllable boundaries would collapse an intervening note.")
    if timestep is not None:
        frames = frame_coordinates(layers, timestep, end_frame, ms)
        layers = {name: [frames[t] for t in edges] for name, edges in layers.items()}
    tiers = {name: [Interval(a, b, old.label) for a, b, old in zip(edges, edges[1:], annotation.tiers[name])]
             for name, edges in layers.items()}
    return replace(annotation, tiers=tiers, duration=layers["phones"][-1])
