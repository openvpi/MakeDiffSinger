"""Adapt this explicit second-based JSON example to a verified source format.

Each record supplies id, audio, language, and a tiers mapping. Tier entries have
start, end, and label. Source parsing belongs here; annotation rules stay in core.
"""

import argparse
import json
from pathlib import Path

from makediffsinger.api import import_intervals, save_result
from makediffsinger.storage import write_json
from makediffsinger.types import AudioSource, BatchResult, Interval, Outcome, Sample
from makediffsinger.validation import require_valid


def map_record(record, source_root, origin, groups_verified):
    if record.get("time_unit", "seconds") != "seconds":
        raise ValueError("Map the verified source time unit to seconds before importing.")
    sample = Sample(record["id"], AudioSource(str(source_root / record["audio"])),
                    language=record.get("language"), speaker=record.get("speaker"),
                    subset=record.get("subset", ""), metadata={"source_record": origin})
    tiers = {name: [Interval(float(i["start"]), float(i["end"]), i["label"]) for i in intervals]
             for name, intervals in record["tiers"].items()}
    return import_intervals(sample, float(record["duration"]), tiers, origin=origin,
                            groups_verified=groups_verified,
                            time_resolution=float(record.get("time_resolution", 0.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-verified", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8"))
    outcomes, failures = [], []
    for index, record in enumerate(records):
        try:
            annotation = map_record(record, args.source.resolve().parent,
                                    f"{args.source.resolve()}#{index}", args.groups_verified)
            require_valid(annotation, check_audio=True)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            failures.append({"record_index": index, "source": str(args.source.resolve()),
                             "reason": str(exc)})
            continue
        outcomes.append(Outcome(annotation.sample, "imported", annotation))
    report = {"storage": save_result(BatchResult(outcomes), args.output), "mapping_failures": failures}
    write_json(args.output / "adapter-report.json", report)
    print(json.dumps(report, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
