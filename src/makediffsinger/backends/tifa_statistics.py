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
    from inference.callbacks import StatisticsCallback, _CompactDict, _CompactEncoder

    metrics, scores = json.load(sys.stdin.buffer)
    for record in scores:
        for word in record["words"]:
            word["alternatives"] = [_CompactDict(row, score=float(row["score"])) for row in word["alternatives"]]
    callback = StatisticsCallback(save_dir=Path(sys.argv[1]))
    callback._metric_records = metrics
    callback._score_records = scores
    if all(row.keys() == metrics[0].keys() and "num_skipped_tokens" in row for row in metrics):
        callback.on_predict_end(None, None)
    else:
        # The native sorter assumes uniform keys. Preserve sparse measurements and
        # supply its plotting method only the observations available for each plot.
        directory = Path(sys.argv[1])
        (directory / "diagnosis.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "scores.json").write_text(_CompactEncoder(indent=2, ensure_ascii=False).encode(scores), encoding="utf-8")
        for field in ("agreement", "confidence", "determinacy", "monotonicity"):
            measured = [row for row in metrics if field in row]
            paired = [{field: row[field], "num_frames": row["num_frames"]} for row in measured if "num_frames" in row]
            callback._save_stat_plots(paired)
            if len(paired) != len(measured):
                callback._save_stat_plots([{field: row[field]} for row in measured])
