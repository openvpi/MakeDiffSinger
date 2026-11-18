"""Invoke the upstream statistics writer in the aligner's import namespace."""

import json
import math
from pathlib import Path
import sys


def pronunciation_records(words):
    """Detach TIFA's JSON-formatting dict subclass before leaving the worker."""
    records = []
    for word in words:
        alternatives = []
        for native in word["alternatives"]:
            score = native["score"]
            alternatives.append(dict(native, score=score if math.isfinite(score) else str(score)))
        records.append(dict(word, alternatives=alternatives))
    return records


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    from inference.callbacks import StatisticsCallback, _CompactDict

    metrics, scores = json.load(sys.stdin.buffer)
    for record in scores:
        for word in record["words"]:
            word["alternatives"] = [_CompactDict(row, score=float(row["score"])) for row in word["alternatives"]]
    callback = StatisticsCallback(save_dir=Path(sys.argv[1]))
    callback._metric_records = metrics
    callback._score_records = scores
    callback.on_predict_end(None, None)
