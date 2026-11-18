"""Match independently sliced clips by phonetic retrieval and pronunciation fusion."""

from collections import defaultdict
from itertools import groupby
import math
import re

from .lyric_pronunciation import LyricUnit, PhoneGraph, lyric_units, pronunciation_distance


_CREDIT = re.compile(
    r"^\s*(?:作[词詞曲]|填[词詞]|编曲|編曲|编[写寫]|譜曲|编配|編配|原唱|演唱|歌手|"
    r"制作人|製作人|混音|母带|母帶|录音|錄音|詞|曲|"
    r"lyrics by|written by|music by|arranged by|artist|composer|producer|"
    r"vocals?|arrangement)\s*[:：-]\s*", re.IGNORECASE,
)
_TAG = re.compile(r"^\s*[\[【（(][^\n]*?[\]】）)]\s*$")
_LRC = re.compile(r"^\s*(?:\[\d{1,3}:\d{2}(?:[.:]\d+)?\]\s*)+")
_STAGE = re.compile(
    r"[\[【（(]\s*(?:前奏|间奏|間奏|尾奏|副歌|重复副歌|重複副歌|"
    r"intro|outro|interlude|instrumental|verse(?:\s+\d+)?|chorus|bridge)\s*[\]】）)]",
    re.IGNORECASE,
)


def _parse_reference(text, exclude_lines, keep_lines):
    lines = text.splitlines()
    if set(exclude_lines) & set(keep_lines):
        raise ValueError("A source line cannot be both excluded and kept.")
    if not all(1 <= n <= len(lines) for n in (*exclude_lines, *keep_lines)):
        raise ValueError("A source line number is outside the lyric file.")
    clean, reported = {}, []
    for line, raw in enumerate(lines, 1):
        raw = raw.lstrip("\ufeff") if line == 1 else raw
        lyric = _LRC.sub(lambda match: " " * len(match[0]), raw)
        if line in exclude_lines:
            reason = "explicit"
        elif line not in keep_lines and (_CREDIT.match(lyric) or _TAG.match(lyric)):
            reason = "credit_or_standalone_annotation"
        else:
            if line not in keep_lines:
                tags = [match[0] for match in _STAGE.finditer(lyric)]
                if tags:
                    reported.append({"line": line, "text": tags, "reason": "inline_stage_annotation"})
                    lyric = _STAGE.sub(lambda match: " " * len(match[0]), lyric)
            if any(char.isalnum() for char in lyric):
                clean[line] = lyric
            continue
        reported.append({"line": line, "text": raw, "reason": reason})
    if not clean:
        raise ValueError("No sung lyric text remains after source preparation.")
    return clean, reported


def _span_text(lines, first, last):
    return " ".join(" ".join(lines[n][first.start if n == first.line else 0:
                                       last.end if n == last.line else None].split())
                    for n in lines if first.line <= n <= last.line)


def _source_span(units, start, end):
    first, last = units[start], units[end - 1]
    return {"first_line": first.line, "first_column": first.start + 1,
            "last_line": last.line, "last_column": last.end}


def _reference_units(lines, converted):
    units, problems = [], []
    for number, text in lines.items():
        result = converted[text]
        line = lyric_units(text, result["words"], number)
        unknown = [word["text"] for word in result["words"] if not word["paths"]]
        if result.get("error") or not line or unknown:
            problems.append({"line": number, "error": result.get("error") or "Missing usable pronunciation.",
                             "unpronounceable_words": unknown})
        units.extend(line or [LyricUnit(number, 0, len(text), ((),))])
    regions = []
    for valid, group in groupby(enumerate(units), key=lambda item: item[1].pronounceable):
        if valid:
            indices, region = zip(*group)
            regions.append((indices[0], PhoneGraph.from_units(region)))
    return units, regions, problems


def _spans(row, offset):
    for end, (edits, matches, start) in enumerate(row):
        if start < end and matches < 0:
            yield (offset + start, offset + end), edits, -matches


def _make_candidate(lines, units, spans, edits, matches):
    return {"text": " ".join(_span_text(lines, units[a], units[b - 1]) for a, b in spans),
            "source_spans": [_source_span(units, a, b) for a, b in spans],
            "phone_edit_distance": edits, "matched_phones": matches,
            "phone_similarity": matches / (edits + matches),
            "joined": len(spans) > 1, "_spans": spans}


def _coarse_key(candidate):
    return candidate["phone_edit_distance"], -candidate["matched_phones"]


def _transcript_key(candidate, lines, units):
    # Preserve G2P word boundaries while coalescing copies that differ in line
    # wrapping or spaces between units. This never changes a phone score.
    return (candidate["joined"], tuple(
        "".join(lines[unit.line][unit.start:unit.end].split())
        for a, b in candidate["_spans"] for unit in units[a:b]
    ))


def _joined_spans(units, regions, query, forward, phrase_breaks):
    """Join two exact phone matches at a phrase boundary in the source or ASR."""
    reversed_query = query.reversed()
    backward = [pronunciation_distance(reversed_query, graph.reversed(), free_start=True)
                for _, graph in regions]
    length = len(query.boundaries) - 1
    for split in range(1, length):
        left, right = [], []
        for (offset, graph), prefixes, suffixes in zip(regions, forward, backward):
            size = len(graph.boundaries) - 1
            left.extend((span, matches) for span, edits, matches in _spans(prefixes[split], offset)
                        if edits == 0 and (split in phrase_breaks or span[1] == len(units)
                                           or units[span[1] - 1].line != units[span[1]].line))
            for (x, y), edits, matches in _spans(suffixes[length - split], 0):
                a, b = offset + size - y, offset + size - x
                if edits == 0 and (split in phrase_breaks or a == 0 or units[a - 1].line != units[a].line):
                    right.append(((a, b), matches))
        for first, left_matches in left:
            for second, right_matches in right:
                if first[1] != second[0]:
                    yield (first, second), left_matches + right_matches


def _candidate_spans(lines, units, regions, query, phrase_breaks, limit):
    forward = [pronunciation_distance(query, graph, free_start=True) for _, graph in regions]
    candidates = [_make_candidate(lines, units, (span,), edits, matches)
                  for (offset, _), rows in zip(regions, forward)
                  for span, edits, matches in _spans(rows[-1], offset)]
    if len(query.boundaries) > 2 and not any(c["phone_edit_distance"] == 0 for c in candidates):
        candidates.extend(_make_candidate(lines, units, spans, 0, matches)
                          for spans, matches in _joined_spans(units, regions, query, forward, phrase_breaks))
    unique = {}
    for candidate in sorted(candidates, key=lambda c: (*_coarse_key(c), c["_spans"])):
        kept = unique.setdefault(_transcript_key(candidate, lines, units), candidate)
        if kept is not candidate and not (kept["joined"] or candidate["joined"]):
            kept["source_spans"].extend(candidate["source_spans"])
    return list(unique.values())[:limit]


def _neighbor_anchors(samples, records):
    recordings = defaultdict(list)
    for sample in samples:
        interval = sample.metadata.get("origin_slice", {})
        recording = sample.metadata.get("origin_recording", {}).get("path")
        if recording and "start_sample" in interval and "end_sample" in interval:
            recordings[recording].append((interval["start_sample"], interval["end_sample"], sample.key))
    anchors = defaultdict(list)
    for items in recordings.values():
        items.sort()
        for (_, end, left), (start, _, right) in zip(items, items[1:]):
            if end <= start:
                for target, neighbor in ((left, right), (right, left)):
                    if candidates := records[neighbor]["candidates"]:
                        leading = candidates[0]
                        if len(leading["source_spans"]) == len(leading["_spans"]):
                            anchors[target].extend(leading["_spans"])
    return anchors


def _comparison_candidate(candidates, neighbors):
    """Use a containing retrieved excerpt if its extra words are unclaimed."""
    first = candidates[0]
    if first["joined"]:
        return 0
    left, right = first["_spans"][0]
    for i, candidate in enumerate(candidates[1:], 1):
        if candidate["joined"]:
            continue
        start, end = candidate["_spans"][0]
        if start <= left and right <= end and (start, end) != (left, right):
            additions = [(a, b) for a, b in ((start, left), (right, end)) if a < b]
            if all(max(a, x) >= min(b, y) for a, b in additions for x, y in neighbors):
                return i
    return 0


def _fuse(pipeline, samples, records, languages, chunk_size, min_score, top_k):
    neighbors = _neighbor_anchors(samples, records)
    requests = []
    for sample in samples:
        record = records[sample.key]
        if record["candidates"]:
            index = _comparison_candidate(record["candidates"], neighbors[sample.key])
            requests.append((sample, index))
    for offset in range(0, len(requests), chunk_size):
        chunk = requests[offset:offset + chunk_size]
        inputs = [(sample, records[sample.key]["candidates"][i]["text"], records[sample.key]["asr_text"])
                  for sample, i in chunk]
        outcomes = pipeline.tifa.fuse_lyrics(inputs, languages, min_score=min_score, top_k=top_k)
        for (sample, index), outcome in zip(chunk, outcomes, strict=True):
            record = records[sample.key]
            record["pronunciation_fusion"] = outcome
            if outcome["status"] == "scored":
                record["candidate_index"] = index


def match_lyrics(samples, reference, hypotheses, *, pipeline, languages=None, verify=True,
                 candidate_count=8, chunk_size=32, min_similarity=None, min_margin=None,
                 min_score=0.0, top_k=3, exclude_lines=(), keep_lines=()):
    """Retrieve clip lyrics by phone edit distance and fuse pronunciations with the forced aligner.

    The pipeline supplies the configured contextual G2P in both modes.
    min_score and top_k limit the independent source shortlists. min_similarity
    and min_margin add diagnostics to the result.
    """
    if candidate_count < 2 or chunk_size < 1:
        raise ValueError("Need at least two candidates and a positive chunk size.")
    if not math.isfinite(min_score) or type(top_k) is not int or top_k < 1:
        raise ValueError("Minimum source score must be finite and top_k must be a positive integer.")
    if min_similarity is not None and not 0 <= min_similarity <= 1:
        raise ValueError("Minimum similarity must lie in [0, 1].")
    if min_margin is not None and not math.isfinite(min_margin):
        raise ValueError("Minimum margin must be finite.")
    if min_margin is not None and min_margin < 0:
        raise ValueError("Minimum margin must be nonnegative.")
    samples = list(samples)
    if len({s.key for s in samples}) != len(samples):
        raise ValueError("Sample identities must be unique.")
    lines, excluded = _parse_reference(reference, tuple(exclude_lines), tuple(keep_lines))
    asr = hypotheses["clips"]
    queries = [asr.get(s.key, {}).get("text", "").strip() for s in samples]
    texts = list(dict.fromkeys([*lines.values(), *(text for text in queries if text)]))
    converted = dict(zip(texts, pipeline.tifa.pronunciations(texts, languages), strict=True))
    units, regions, reference_problems = _reference_units(lines, converted)

    records = {}
    for sample, query in zip(samples, queries):
        key = sample.key
        record = records[key] = {"asr_text": query, "status": "unmatched", "candidates": [], "diagnostics": []}
        if not query:
            error = asr.get(key, {}).get("error")
            record["reason"] = "asr_failed" if error else "missing_or_empty_asr"
            if error:
                record["asr_error"] = error
            continue
        result = converted[query]
        query_units = [unit for unit in lyric_units(query, result["words"], 1) if unit.pronounceable]
        if result.get("error"):
            record["g2p_error"] = result["error"]
        unknown = [word["text"] for word in result["words"] if not word["paths"]]
        if unknown:
            record["unpronounceable_asr_words"] = unknown
            record["diagnostics"].append("incomplete_asr_pronunciation")
        if not query_units:
            record["reason"] = "no_usable_asr_pronunciation"
            continue
        phrase_breaks = {
            i for i, (left, right) in enumerate(zip(query_units, query_units[1:]), 1)
            if any(c in ",.;!?，。；！？、\n" for c in query[left.end:right.start])
        }
        graph = PhoneGraph.from_units(query_units)
        record["candidates"] = _candidate_spans(lines, units, regions, graph, phrase_breaks,
                                                candidate_count)
    if verify:
        _fuse(pipeline, samples, records, languages, chunk_size, min_score, top_k)

    for record in records.values():
        if "reason" in record:
            continue
        candidates = record["candidates"]
        if not candidates:
            record["reason"] = "no_lyric_candidate"
            continue
        diagnostics = record["diagnostics"]
        fusion = record.get("pronunciation_fusion", {})
        scored = fusion.get("status") == "scored"
        index = record.get("candidate_index", 0)
        best = candidates[index]
        if verify and not scored:
            diagnostics.append("tifa_could_not_verify")
        margin = fusion.get("margin")
        distance_gap = (candidates[1]["phone_edit_distance"] - candidates[0]["phone_edit_distance"]
                        if len(candidates) > 1 else None)
        record.update(status="suggested", suggestion=" ".join(fusion["texts"]) if scored else best["text"],
                      candidate_index=index,
                      margin=margin, phone_edit_distance_gap=distance_gap,
                      reason="tifa_pronunciation_fusion" if scored else "g2p_ranked")
        if scored:
            record["transcript"] = fusion["transcript"]
            record["texts"] = fusion["texts"]
        if min_similarity is not None and candidates[0]["phone_similarity"] < min_similarity:
            diagnostics.append("weak_retrieval")
        if margin == 0 or (margin is not None and min_margin is not None and margin < min_margin) or (
                not scored and len(candidates) > 1 and _coarse_key(candidates[0]) == _coarse_key(candidates[1])):
            diagnostics.append("competing_lyrics")
        if best["joined"] and not scored:
            diagnostics.append("joined_without_acoustic_verification")
        for candidate in candidates:
            candidate.pop("_spans")
    return {"settings": {"verified": verify, "candidate_count": candidate_count,
                         "min_similarity": min_similarity, "min_margin": min_margin,
                         "min_score": min_score, "top_k": top_k,
                         "acoustic_ranking": "tifa_pronunciation_fusion" if verify else None},
            "excluded_lines": excluded, "reference_g2p_errors": reference_problems, "clips": records}
