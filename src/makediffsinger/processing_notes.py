"""Write human-readable notices about automatic processing decisions."""

from collections import Counter
from pathlib import Path


def append_processing_notes(path, stage, messages):
    if path is None or not messages:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        title = "# Processing notes\n" if stream.tell() == 0 else ""
        stream.write(title + f"\n## {stage}\n\n" + "\n".join(f"- {message}" for message in messages) + "\n")


def problem_counts(outcomes):
    return Counter((outcome.status, issue.stage, issue.code)
                   for outcome in outcomes for issue in outcome.issues)


def problem_messages(counts):
    actions = {
        "pending": "Left pending and saved the reason for recovery.",
        "failed": "Recorded the failure and excluded the failed output from subsequent use.",
        "conflict": "Kept the existing output and saved the conflict for recovery.",
    }
    return [f"{count} in {stage or 'processing'} with {code.replace('_', ' ')}. {actions[status]}"
            for (status, stage, code), count in counts.most_common() if status in actions]
