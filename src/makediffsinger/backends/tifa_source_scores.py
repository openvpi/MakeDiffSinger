"""Project TIFA's complete source scores onto aligned word blocks."""

import math

from inference.scoring import _advance
import numpy as np


def block_scores(word_paths, spans, fragments, vocab_size):
    """Score each distinct block path with every other word free to take its best path."""
    if fragments is None:
        return [{tuple(token for paths in word_paths[span] for token in paths[0]): 0.0} for span in spans]

    descriptors, lengths, costs, tails, capacity = (item[0].detach().cpu().numpy() for item in fragments)
    normalize = math.log(vocab_size)
    num_masks = int(capacity.sum())
    prefix = [{(0, 0): 0.0}]
    transitions = []
    for i, paths in enumerate(word_paths, 1):
        pieces = np.flatnonzero(descriptors[:, 0] == i)
        scores, edges = {}, {}
        for state, prior in prefix[-1].items():
            moves = []
            for col, tokens in enumerate(paths):
                step = _advance(state, pieces, col, descriptors, lengths, costs, tails, capacity)
                if step is None:
                    continue
                target, delta = step
                moves.append((target, tokens, delta))
                scores[target] = max(scores.get(target, -math.inf), prior + delta)
            edges[state] = moves
        transitions.append(edges)
        prefix.append(scores)

    # The state carries the consumed length of the current mask segment. Closing
    # it adds TIFA's SPACE suffix once, including the final segment.
    suffix = [{} for _ in prefix]
    suffix[-1] = {state: float(tails[state]) for state in prefix[-1]}
    for i in range(len(transitions) - 1, -1, -1):
        suffix[i] = {state: max((delta + suffix[i + 1][target] for target, _, delta in edges),
                                default=-math.inf) for state, edges in transitions[i].items()}

    result = []
    for span in spans:
        active = {(state, ()): score for state, score in prefix[span.start].items()}
        for i in range(span.start, span.stop):
            following = {}
            for (state, phones), prior in active.items():
                for target, tokens, delta in transitions[i][state]:
                    key = target, phones + tokens
                    following[key] = max(following.get(key, -math.inf), prior + delta)
            active = following
        scores = {}
        for (state, phones), prior in active.items():
            score = (prior + suffix[span.stop][state]) / num_masks + normalize
            scores[phones] = max(scores.get(phones, -math.inf), score)
        result.append(scores)
    return result
