"""Small semantic fixtures independent of model checkpoints."""

import numpy as np
import pytest
import soundfile as sf

from makediffsinger.phonology import PhoneProfiles
from makediffsinger.types import Annotation, AudioSource, Interval, Sample, Transcript


@pytest.fixture
def profiles():
    return PhoneProfiles({"en": {"b": "consonant", "r": "medial", "l": "medial",
                                 "aa": "vowel", "t": "consonant"},
                          "zh": {"l": "consonant", "r": "consonant", "a": "vowel"},
                          "ja": {"cl": "consonant", "a": "vowel"}})


@pytest.fixture
def annotation(tmp_path):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(48000, dtype=np.float32), 48000, subtype="PCM_24")
    sample = Sample("clip", AudioSource(str(audio)), Transcript("bright"), "en",
                    speaker="singer", subset="en/singer/control")
    return Annotation(sample, 1.0, {
        "texts": [Interval(0, 1, "bright")],
        "words": [Interval(0, .7, "b r aa"), Interval(.7, 1, "t")],
        "phones": [Interval(0, .1, "b"), Interval(.1, .2, "r"),
                   Interval(.2, .7, "aa"), Interval(.7, 1, "t")],
        "notes": [Interval(0, .1, "C3+14"), Interval(.1, .4, "D4-7"),
                  Interval(.4, .7, "E4+12"), Interval(.7, 1, "rest")],
    }, quality={"confidence": -.25, "agreement": .8},
        provenance={"groups_verified": True, "time_resolution_seconds": .01}, status="annotated")
