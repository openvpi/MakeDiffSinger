"""Objects shared by scripts, adapters, and isolated inference workers."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def check_identity(identifier: str, subset: str = "") -> None:
    if not identifier.strip() or identifier in {".", ".."} or any(c in identifier for c in "/\\"):
        raise ValueError(f"Sample ID must be a nonempty flat filename stem: {identifier!r}")
    path = PurePosixPath(subset.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in subset:
        raise ValueError(f"Subset must be a relative path without parent traversal: {subset!r}")


@dataclass
class AudioSource:
    path: str | None = None
    start_sample: int = 0
    end_sample: int | None = None
    sample_rate: int | None = None
    waveform: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.path is not None and self.waveform is not None:
            raise ValueError("Supply either an audio path or an in-memory waveform, not both.")
        if self.path is not None:
            self.path = str(Path(self.path).resolve())
        if self.start_sample < 0 or (self.end_sample is not None and self.end_sample <= self.start_sample):
            raise ValueError("Audio ranges must be nonempty, start-inclusive and end-exclusive.")
        if self.waveform is not None:
            self.waveform = np.asarray(self.waveform)
            if not self.sample_rate or self.waveform.ndim not in (1, 2):
                raise ValueError("In-memory audio needs a sample rate and samples or samples-by-channels array.")


@dataclass
class Transcript:
    content: str
    format: str = "text"
    path: str | None = None

    def __post_init__(self):
        if self.format not in {"text", "pfml"}:
            raise ValueError("Transcript format must be 'text' or 'pfml'.")


@dataclass
class Sample:
    id: str
    audio: AudioSource
    transcript: Transcript | None = None
    language: str | None = None
    extended_languages: list[str] = field(default_factory=list)
    speaker: str | None = None
    subset: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        check_identity(self.id, self.subset)
        self.subset = self.subset.replace("\\", "/").strip("/")

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.id}" if self.subset else self.id

    @property
    def languages(self) -> list[str] | None:
        tags = ([self.language] if self.language else []) + self.extended_languages
        return list(dict.fromkeys(tags)) or None


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Syllable:
    first_phone: int
    phone_count: int
    start: float
    end: float
    prefix: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Annotation:
    sample: Sample
    duration: float
    tiers: dict[str, list[Interval]]
    quality: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = "imported"
    stale: list[str] = field(default_factory=list)
    edited_layers: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.sample.key


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    stage: str = "validation"
    severity: str = "error"


@dataclass
class Outcome:
    sample: Sample
    status: str
    annotation: Annotation | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status not in {"pending", "failed", "conflict"} and not any(
            issue.severity == "error" for issue in self.issues
        )


@dataclass
class BatchResult:
    outcomes: list[Outcome]

    @property
    def annotations(self) -> list[Annotation]:
        return [o.annotation for o in self.outcomes if o.ok and o.annotation is not None]

    @property
    def ready(self) -> list[Sample]:
        return [o.sample for o in self.outcomes if o.ok]

    @property
    def problems(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.ok]


class ResourceError(RuntimeError):
    """A missing or incompatible shared resource prevents an entire stage."""
