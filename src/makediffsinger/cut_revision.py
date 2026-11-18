"""Inspect one planned boundary and prepare explicit replacement clips."""

import numpy as np

from .audio import load_audio
from .slicing import Slice, SlicePlan
from .types import AudioSource, check_identity


def _adjacent_pair(plan, left_id, right_id):
    matches = [(left, right) for left, right in zip(plan.slices, plan.slices[1:])
               if (left.id, right.id) == (left_id, right_id)]
    if len(matches) != 1:
        raise ValueError("Select two consecutive slice IDs in source order.")
    left, right = matches[0]
    if not 0 <= left.start_sample < left.end_sample == right.start_sample < right.end_sample <= plan.source["frames"]:
        raise ValueError("The selected slices must share a boundary in the source recording.")
    return left, right


def inspect_cut(plan, left_id, right_id, start, end):
    """Return source coordinates and local energy measurements without choosing a cut."""
    left, right = _adjacent_pair(plan, left_id, right_id)
    cut = left.end_sample
    if not left.start_sample <= start < cut < end <= right.end_sample:
        raise ValueError("The inspection window must straddle the cut and stay inside the two slices.")
    rate = plan.source["sample_rate"]
    hop = max(1, round(rate * 0.01))
    first, last = max(left.start_sample, start - hop), min(right.end_sample, end + hop)
    waveform, _ = load_audio(AudioSource(plan.source["path"], first, last, rate))
    centers = range(cut - ((cut - start) // hop) * hop, end, hop)
    rms = []
    for sample in centers:
        frame = waveform[max(0, sample - first - hop):sample - first + hop]
        rms.append({"sample": sample, "seconds": sample / rate, "rms": float(np.sqrt(np.mean(frame ** 2)))})
    return {"left": left.id, "right": right.id, "sample_rate": rate,
            "cut_sample": cut, "cut_seconds": cut / rate,
            "window": [start, end], "frame_samples": hop * 2, "hop_samples": hop, "rms": rms}


def _replacement(clip_id, start, end, originals):
    retained = []
    evidence_reasons, review = set(), set()
    for original in originals:
        review.update(original.review)
        for evidence in original.cut_evidence:
            evidence_reasons.update(evidence.get("reasons", ()))
            if evidence["sample"] in (start, end) and evidence not in retained:
                retained.append(evidence)
    review.difference_update(evidence_reasons)
    review.update(reason for evidence in retained for reason in evidence.get("reasons", ()))
    return Slice(clip_id, start, end, sorted(review), retained)


def revise_cut(plan, left_id, right_id, *, move_to=None, merge_as=None):
    """Make a replay plan for only the clips replacing the selected pair."""
    left, right = _adjacent_pair(plan, left_id, right_id)
    if (move_to is None) == (merge_as is None):
        raise ValueError("Choose either a new source sample or a merged clip ID.")
    start, old_cut, end = left.start_sample, left.end_sample, right.end_sample
    if move_to is not None:
        if not start < move_to < end or move_to == old_cut:
            raise ValueError("Choose a different boundary strictly inside the two slices.")
        revised = [_replacement(left.id, start, move_to, [left]),
                   _replacement(right.id, move_to, end, [right])]
    else:
        check_identity(merge_as)
        if any(item.id == merge_as for item in plan.slices):
            raise ValueError(f"Merged slice ID already exists: {merge_as}")
        revised = [_replacement(merge_as, start, end, [left, right])]
    settings = {**plan.settings, "cut_revision": {"replaces": [left.id, right.id],
                                                   "old_cut_sample": old_cut, "new_cut_sample": move_to}}
    return SlicePlan(plan.source, revised, settings, plan.dataset)
