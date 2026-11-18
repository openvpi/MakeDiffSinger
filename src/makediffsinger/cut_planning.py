"""Preselect RMS boundaries, then optimize the sequence of retained clips."""

from typing import NamedTuple

import numpy as np


MAX_PADDING_SECONDS = 1.0
REQUIRED_PAUSE_SECONDS = 4.0
WEAK_THRESHOLD_OFFSET_DB = 10.0


class Cut(NamedTuple):
    left: int
    right: int
    kind: str
    required: bool = False


def _rms(waveform, positions, windows):
    power = np.empty(len(waveform) + 1, dtype=np.float64)
    power[0] = 0.0
    np.square(waveform, out=power[1:], dtype=np.float64)
    np.cumsum(power, out=power)
    for window in windows:
        left = np.maximum(0, positions - window // 2)
        right = np.minimum(len(waveform), positions + window - window // 2)
        yield np.sqrt((power[right] - power[left]) / window)


def _regions(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _quietest(positions, rms, begin, end, twice_midpoint):
    window = rms[begin:end]
    minima = begin + np.flatnonzero(window == window.min())
    return int(minima[np.argmin(np.abs(2 * positions[minima] - twice_midpoint))])


def _pause_cuts(positions, rms, silent, slicer, weak_threshold):
    cuts = []
    first, last = 0, len(positions) - 1
    padding = MAX_PADDING_SECONDS * slicer.sr
    required_duration = REQUIRED_PAUSE_SECONDS * slicer.sr
    weak = rms < weak_threshold
    for begin, end in _regions(silent):
        start_sample, end_sample = positions[begin], positions[min(end, len(positions) - 1)]
        required = end_sample - start_sample >= required_duration
        if not required and end - begin < slicer.min_interval:
            continue
        weak[begin:end] = False
        twice_midpoint = start_sample + end_sample
        left_end = min(end, positions.searchsorted(start_sample + padding, side="right"))
        right_start = max(begin, positions.searchsorted(end_sample - padding))
        # The windows may overlap; their shared ordering keeps the cuts ordered.
        if begin == 0:
            first = _quietest(positions, rms, right_start, end, twice_midpoint)
        elif end == len(positions):
            last = _quietest(positions, rms, begin, left_end, twice_midpoint)
        else:
            cuts.append(Cut(_quietest(positions, rms, begin, left_end, twice_midpoint),
                            _quietest(positions, rms, right_start, end, twice_midpoint), "pause", required))

    for begin, end in _regions(weak):
        begin, end = max(begin, first + 1), min(end, last)
        if begin < end:
            point = _quietest(positions, rms, begin, end, positions[begin] + positions[end - 1])
            cuts.append(Cut(point, point, "weak"))
    return [Cut(first, first, "pause"), *sorted(cuts), Cut(last, last, "pause")]


def _add_fallbacks(cuts, positions, sustained, maximum):
    added = []
    for previous, following in zip(cuts, cuts[1:]):
        left, right = previous.right, following.left
        if positions[right] - positions[left] <= maximum:
            continue
        interior = np.arange(left + 1, right)
        valleys = interior[(sustained[interior] <= sustained[interior - 1])
                           & (sustained[interior] < sustained[interior + 1])]
        added.extend(Cut(int(point), int(point), "fallback") for point in valleys)
        anchors = [left, *valleys, right]
        for a, b in zip(anchors, anchors[1:]):
            duration = positions[b] - positions[a]
            pieces = (duration + maximum - 1) // maximum
            # Flat or steadily changing audio can have no local minimum at all.
            for part in range(1, pieces):
                point = int(positions.searchsorted(positions[a] + part * duration / pieces))
                if a < point < b:
                    added.append(Cut(point, point, "fallback"))
    return sorted(set(cuts + added))


def _best_path(cuts, positions, audible, baseline, sustained, limits):
    minimum, target, maximum = limits
    starts = np.array([cut.right for cut in cuts], dtype=np.intp)
    has_sound = np.r_[0, np.cumsum(audible[:-1] | audible[1:])]
    scores = np.full((len(cuts), 5), np.inf)
    scores[0] = 0
    parents = np.full(len(cuts), -1, dtype=np.intp)
    first = 0
    for end, cut in enumerate(cuts[1:], 1):
        # Later clips cannot cross the last required cut.
        candidates = first + np.flatnonzero((has_sound[cut.left] > has_sound[starts[first:end]])
                                            & np.isfinite(scores[first:end, 0]))
        if not len(candidates):
            continue
        # Each edge is one retained clip, including any pauses left uncut.
        duration = positions[cut.left] - positions[starts[candidates]]
        cost = scores[candidates]
        cost[:, 0] += np.maximum(minimum - duration, 0) + np.maximum(duration - maximum, 0)
        if cut.kind == "fallback":
            cost[:, 1] += 1
            cost[:, 3] += sustained[cut.left]
        elif cut.kind == "weak":
            cost[:, 2] += 1
            cost[:, 3] += baseline[cut.left]
        cost[:, 4] += np.abs(duration - target)
        best = np.lexsort(cost.T[::-1])[0]
        scores[end] = cost[best]
        parents[end] = candidates[best]
        if cut.required:
            first = end

    selected = [cuts[-1]]
    index = len(cuts) - 1
    while index:
        index = parents[index]
        selected.append(cuts[index])
    return list(reversed(selected))


def recording_cuts(waveform, slicer, limits, phone_hints, *, weak_threshold):
    """Return retained sample ranges and diagnostic evidence at their boundaries."""
    hop = slicer.hop_size
    positions = np.r_[np.arange(0, len(waveform), hop), len(waveform)]
    baseline, sustained = _rms(waveform, positions, (slicer.win_size, slicer.min_interval * hop))
    silent = baseline < slicer.threshold
    audible = ~silent
    if not audible.any():
        return ([(0, len(waveform))] if len(waveform) <= limits[0] else []), {}
    cuts = _pause_cuts(positions, baseline, silent, slicer, weak_threshold)
    cuts = _add_fallbacks(cuts, positions, sustained, limits[2])
    selected = _best_path(cuts, positions, audible, baseline, sustained, limits)

    evidence = {}
    neighborhood = round(2 * slicer.sr / hop)
    boundaries = {point: cut.kind for cut in selected for point in (cut.left, cut.right)}
    for index in sorted(boundaries.keys() - {0, len(positions) - 1}):
        sample = int(positions[index])
        value = float(sustained[index])
        median = float(np.median(sustained[max(0, index - neighborhood):index + neighborhood + 1]))
        kind = boundaries[index]
        reasons = []
        if kind != "pause":
            reasons.append("rms_cut")
            if kind == "weak":
                reasons.append("weak_pause")
            if value > slicer.threshold and (median == 0 or value / median > 0.7):
                reasons.append("weak_loudness_valley")
        if any(start * slicer.sr < sample < finish * slicer.sr for start, finish in phone_hints):
            reasons.append("inside_candidate_phone")
        evidence[sample] = {"sample": sample, "rms": value, "local_median_rms": median,
                            "in_silence": kind == "pause", "cut_kind": kind, "reasons": reasons}
    return [(int(positions[previous.right]), int(positions[following.left]))
            for previous, following in zip(selected, selected[1:])], evidence
