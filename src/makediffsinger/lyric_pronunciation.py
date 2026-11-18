"""Source spans and edit distance for complete g2pflow pronunciation paths."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata


@dataclass(frozen=True)
class LyricUnit:
    line: int
    start: int
    end: int
    words: tuple

    @property
    def pronounceable(self):
        return bool(self.words) and all(self.words)


def _characters(text):
    """Project Unicode normalization back to offsets. These chars never enter the distance."""
    chars, offsets = [], []
    for index, char in enumerate(text):
        for symbol in unicodedata.normalize("NFKD", char).casefold():
            if symbol.isalnum() or unicodedata.category(symbol).startswith("M"):
                chars.append(symbol)
                offsets.append(index)
    return "".join(chars), offsets


def lyric_units(text, words, line):
    """Use G2P word boundaries where they can be projected to the source.

    If a preprocessor rewrites a span, keep its pronunciation words together
    where there is no defensible internal text boundary.
    """
    if not words:
        return []
    source, offsets = _characters(text)
    parts = [_characters(word["text"])[0] for word in words]
    converted = "".join(parts)
    if not source or not converted:
        return []
    if source == converted:
        matches = range(len(source) + 1)
    else:
        matches = {a + shift: b + shift
                   for a, b, size in SequenceMatcher(None, converted, source, autojunk=False).get_matching_blocks()
                   if size for shift in range(size + 1)}
    cuts = [(0, 0)]
    char_index = 0
    for index, part in enumerate(parts[:-1], 1):
        char_index += len(part)
        boundary = matches[char_index] if source == converted else matches.get(char_index)
        if (boundary is not None and cuts[-1][1] < boundary < len(source)
                and offsets[boundary - 1] != offsets[boundary]):
            cuts.append((index, boundary))
    cuts.append((len(words), len(source)))
    return [LyricUnit(line, offsets[left], offsets[right - 1] + 1,
                      tuple(tuple(tuple(path) for path in word["paths"] if path) for word in words[a:b]))
            for (a, left), (b, right) in zip(cuts, cuts[1:])]


@dataclass(frozen=True)
class PhoneGraph:
    incoming: tuple
    boundaries: tuple[int, ...]

    @classmethod
    def from_units(cls, units):
        incoming, boundaries = [()], [0]
        for unit in units:
            for paths in unit.words:
                prefixes = {(): len(incoming) - 1}
                terminal = set()
                for path in paths:
                    parent = prefixes[()]
                    for length, phone in enumerate(path[:-1], 1):
                        prefix = path[:length]
                        if prefix not in prefixes:
                            prefixes[prefix] = len(incoming)
                            incoming.append(((parent, phone),))
                        parent = prefixes[prefix]
                    terminal.add((parent, path[-1]))
                if not terminal:
                    raise ValueError("A pronunciation word needs at least one complete path.")
                incoming.append(tuple(sorted(terminal)))
            boundaries.append(len(incoming) - 1)
        return cls(tuple(incoming), tuple(boundaries))

    def reversed(self):
        last = len(self.incoming) - 1
        incoming = [[] for _ in self.incoming]
        for target, edges in enumerate(self.incoming):
            for source, phone in edges:
                incoming[last - source].append((last - target, phone))
        return PhoneGraph(tuple(map(tuple, incoming)), tuple(last - b for b in reversed(self.boundaries)))


def _distance_rows(query, reference, *, free_start=False):
    rows = []
    starts = {b: i for i, b in enumerate(reference.boundaries)} if free_start else {0: 0}
    for i, query_edges in enumerate(query.incoming):
        previous = [(rows[parent], phone) for parent, phone in query_edges]
        row = []
        for j, reference_edges in enumerate(reference.incoming):
            best = (0, 0, starts[j]) if i == 0 and j in starts else None
            for past, _ in previous:
                edits, matches, start = past[j]
                value = (edits + 1, matches, start)
                if best is None or value < best:
                    best = value
            for parent, ref_phone in reference_edges:
                edits, matches, start = row[parent]
                value = (edits + 1, matches, start)
                if best is None or value < best:
                    best = value
                for past, query_phone in previous:
                    edits, matches, start = past[parent]
                    same = query_phone == ref_phone
                    value = (edits + (not same), matches - same, start)
                    if value < best:
                        best = value
            row.append(best)
        rows.append(row)
    return rows


def pronunciation_distance(query, reference, *, free_start=False):
    """Compare complete graph paths by edits, matching phones, and reference start."""
    rows = _distance_rows(query, reference, free_start=free_start)
    return [[rows[i][j] for j in reference.boundaries] for i in query.boundaries]


def pronunciation_blocks(reference, recognition):
    """Trace the best phone edit path and cut where both complete words end."""
    rows = _distance_rows(reference, recognition)
    ref_ends = {node: i for i, node in enumerate(reference.boundaries)}
    asr_ends = {node: i for i, node in enumerate(recognition.boundaries)}
    i, j = len(reference.incoming) - 1, len(recognition.incoming) - 1
    cuts = []
    while i or j:
        if i in ref_ends and j in asr_ends:
            cuts.append((ref_ends[i], asr_ends[j]))
        previous = [(a, b, int(x != y), int(x == y))
                    for a, x in reference.incoming[i] for b, y in recognition.incoming[j]]
        previous.extend((a, j, 1, 0) for a, _ in reference.incoming[i])
        previous.extend((i, b, 1, 0) for b, _ in recognition.incoming[j])

        def key(step):
            a, b, edits, matches = step
            distance, negative_matches, start = rows[a][b]
            return (distance + edits, negative_matches - matches, start,
                    -int(a in ref_ends and b in asr_ends), -matches)

        i, j, _, _ = min(previous, key=key)
    cuts.append((0, 0))
    cuts.reverse()
    return [(slice(a, b), slice(c, d)) for (a, c), (b, d) in zip(cuts, cuts[1:])]
