"""Opt-in integration checks using caller-supplied resources and clips.

MDS_RESOURCES names a resource file. MDS_INTEGRATION_SAMPLES names a JSON list
with id, audio, language, transcript, optional format (text/pfml), and optional
expected (ready/pending). MDS_DEVICE overrides the saved device. All outputs use pytest's
temporary directory; no baseline archive or persistent review copies are made.
"""

import json
from dataclasses import replace
import os
from pathlib import Path

import pytest
import soundfile as sf

from makediffsinger.api import Pipeline, export_diffsinger
from makediffsinger.config import Resources, Runtime
from makediffsinger.storage import load_annotations, save_result
from makediffsinger.types import AudioSource, Sample, Transcript
from makediffsinger.validation import validate


pytestmark = [pytest.mark.inference, pytest.mark.skipif(
    not (os.getenv("MDS_RESOURCES") and os.getenv("MDS_INTEGRATION_SAMPLES")),
    reason="Model resources and representative clips were not supplied.")]


def samples():
    rows = json.loads(Path(os.environ["MDS_INTEGRATION_SAMPLES"]).read_text(encoding="utf-8"))
    return [(Sample(row["id"], AudioSource(row["audio"]),
                    Transcript(row["transcript"], row.get("format", "text")), row["language"]),
             row.get("expected", "ready")) for row in rows]


def test_pipeline_persistence_and_notes_only_update(tmp_path):
    inputs = samples()
    with Pipeline(Resources.read(os.environ["MDS_RESOURCES"]),
                  Runtime(device=os.getenv("MDS_DEVICE"), batch_size=2)) as pipeline:
        checked = pipeline.preflight([sample for sample, _ in inputs])
        assert [o.status for o in checked.outcomes] == [expected for _, expected in inputs]
        tifa_pid = pipeline.tifa.pid
        result = pipeline.run(checked.ready)
        assert len(result.annotations) == len(checked.ready) and not result.problems
        assert pipeline.tifa.pid == tifa_pid
        game_pid = pipeline.game.pid
        assert game_pid != tifa_pid
        for annotation in result.annotations:
            assert not validate(annotation, required=("texts", "words", "phones", "notes"),
                                profiles=pipeline.profiles, check_audio=True)
        assert save_result(result, tmp_path / "annotations")["saved"] == len(result.annotations)
        restored = load_annotations(tmp_path / "annotations")
        updated = pipeline.reannotate(restored, layers=("notes",), sample_keys=[restored[0].key])
        assert len(updated.annotations) == 1 and pipeline.game.pid == game_pid
        assert all(updated.annotations[0].tiers[layer] == restored[0].tiers[layer]
                   for layer in ("texts", "words", "phones"))
        manifest = export_diffsinger(restored, tmp_path / "export", profiles=pipeline.profiles, copy_audio=False)
        assert len(manifest["selected"]) == len(restored)


def test_buffer_input_with_spawned_loader(tmp_path):
    original = next(sample for sample, expected in samples() if expected == "ready")
    waveform, rate = sf.read(original.audio.path, dtype="float32")
    original.audio = AudioSource(waveform=waveform, sample_rate=rate)
    with Pipeline(Resources.read(os.environ["MDS_RESOURCES"]),
                  Runtime(device=os.getenv("MDS_DEVICE"), num_workers=1)) as pipeline:
        result = pipeline.run([original])
        assert len(result.annotations) == 1 and not result.problems
        assert not validate(result.annotations[0], profiles=pipeline.profiles, check_audio=True)
        assert save_result(result, tmp_path)["saved"] == 1


def test_note_only_extraction_needs_no_transcript_or_phone_resources(tmp_path):
    source = next(sample for sample, expected in samples() if expected == "ready")
    sample = replace(source, transcript=None, language=None)
    resources = Resources.read(os.environ["MDS_RESOURCES"])
    with Pipeline(Resources(game_model=resources.game_model, game_repo=resources.game_repo),
                  Runtime(device=os.getenv("MDS_DEVICE"))) as pipeline:
        result = pipeline.extract_notes([sample])
        assert len(result.annotations) == 1 and not result.problems
        annotation = result.annotations[0]
        assert set(annotation.tiers) == {"notes"} and annotation.status == "partial"
        assert annotation.provenance["syllable_input"] == {"method": "unconstrained", "syllables": []}
        assert not validate(annotation, check_audio=True)
        assert save_result(result, tmp_path / "notes")["saved"] == 1
        assert set(load_annotations(tmp_path / "notes")[0].tiers) == {"notes"}
        with pytest.raises(ValueError, match="Missing phones"):
            export_diffsinger(result.annotations, tmp_path / "export", copy_audio=False)
