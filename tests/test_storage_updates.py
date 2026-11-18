from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path, PureWindowsPath
import shutil

import numpy as np
import pytest

from makediffsinger.annotations import read_textgrid, write_textgrid
from makediffsinger.api import Pipeline
from makediffsinger.reannotation import apply_annotation_updates, plan_reannotation
from makediffsinger.storage import annotation_to_dict, load_problems, read_annotations, save_result, write_annotations, write_json
from makediffsinger.types import AudioSource, BatchResult, Interval, Issue, Outcome, Sample
from makediffsinger.validation import validate
from makediffsinger.catalog import bind_dataset, dataset_root, source_for, store_origins
from makediffsinger.inputs import discover_pairs
from makediffsinger.storage import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("legacy", [False, True])
def test_roundtrip_and_manual_edit_protection(tmp_path, annotation, legacy):
    root = tmp_path / "assets"
    path = write_annotations(annotation, root)
    catalog = root / ".metadata/annotations.json"
    if legacy:
        legacy_state = annotation_to_dict(annotation, include_tiers=False)
        legacy_state["tier_signatures"] = annotation.provenance["_loaded_signatures"]
        write_json(path.with_suffix(".json"), legacy_state)
        catalog.unlink()
    assert not (path.parent / ".metadata").exists()
    saved = read_annotations(path)
    assert not saved.edited_layers
    assert saved.quality == annotation.quality
    edited = deepcopy(saved)
    edited.tiers["phones"][0] = replace(edited.tiers["phones"][0], label="t")
    write_textgrid(edited, path)
    detected = read_annotations(path)
    assert detected.edited_layers == ["phones"]
    assert set(detected.stale) == {"quality", "notes"}
    with pytest.raises(RuntimeError, match="changed since"):
        write_annotations(saved, root, overwrite=True)
    replacement = deepcopy(annotation)
    replacement.provenance.pop("_loaded_path", None)
    with pytest.raises(RuntimeError, match="edited layers"):
        write_annotations(replacement, root, overwrite=True)
    write_annotations(replacement, root, overwrite=True, replace_edited=True)
    assert catalog.is_file() and not path.with_suffix(".json").exists()


def test_notes_update_preserves_unselected_layers_and_rejects_concurrent_edits(annotation, profiles):
    original = deepcopy(annotation)
    candidate = deepcopy(annotation)
    candidate.tiers["notes"][0] = Interval(0, .1, "F4+21")
    plan = plan_reannotation([original], layers=("notes",))
    result = apply_annotation_updates(original, candidate, plan, profiles=profiles)
    assert result.ok
    for layer in ("texts", "words", "phones"):
        assert result.annotation.tiers[layer] == original.tiers[layer]
    assert result.annotation.quality == original.quality
    assert original.tiers["notes"][0].label == "C3+14"
    original.tiers["texts"][0] = Interval(0, 1, "manually changed")
    conflict = apply_annotation_updates(original, candidate, plan, profiles=profiles)
    assert conflict.status == "conflict"
    assert conflict.issues[0].code == "concurrent_edit"


def test_manual_notes_require_explicit_replacement(annotation, profiles):
    annotation.edited_layers = ["notes"]
    candidate = deepcopy(annotation)
    candidate.tiers["notes"][0] = Interval(0, .1, "D3")
    denied = apply_annotation_updates(annotation, candidate, plan_reannotation([annotation]), profiles=profiles)
    assert denied.status == "conflict"
    allowed = apply_annotation_updates(annotation, candidate,
                                       plan_reannotation([annotation], replace_edited=True), profiles=profiles)
    assert allowed.ok and not allowed.annotation.edited_layers


def test_archived_alignment_and_final_share_metrics_until_a_rerun(tmp_path, annotation):
    archived = tmp_path / ".work/fa"
    final = tmp_path / "03-annotations"
    fa_path = write_annotations(deepcopy(annotation), archived)
    final_path = write_annotations(deepcopy(annotation), final)
    path = tmp_path / "04-quality/metrics.json"
    row, = json.loads(path.read_text()).values()
    assert row == {"alignment": annotation.quality}
    updated = deepcopy(annotation)
    updated.quality["confidence"] = .5
    write_annotations(updated, archived, overwrite=True)
    assert read_annotations(fa_path).quality == updated.quality
    assert read_annotations(final_path).quality == annotation.quality
    assert list(tmp_path.rglob(".metadata")) == [tmp_path / ".metadata"]


def test_fa_only_marks_preserved_notes_stale(annotation, profiles):
    candidate = deepcopy(annotation)
    candidate.quality["confidence"] = -.7
    result = apply_annotation_updates(annotation, candidate,
                                      plan_reannotation([annotation], layers=("fa",)), profiles=profiles)
    assert result.ok
    assert result.annotation.tiers["notes"] == annotation.tiers["notes"]
    assert result.annotation.quality["confidence"] == -.7
    assert result.annotation.stale == ["notes"]
    assert "stale_result" in {i.code for i in validate(result.annotation, profiles=profiles)}


def test_partial_update_rejects_incompatible_geometry(annotation, profiles):
    candidate = deepcopy(annotation)
    candidate.tiers["words"] = [Interval(0, .3, "x"), Interval(.3, 1, "y")]
    result = apply_annotation_updates(annotation, candidate,
                                      plan_reannotation([annotation], layers=("words",)), profiles=profiles)
    assert result.status == "conflict"
    assert result.issues[0].code == "invalid_merge"


def test_full_reannotation_establishes_new_group_semantics(annotation, profiles):
    candidate = deepcopy(annotation)
    candidate.provenance["pronunciation_choices"] = [1, 2]
    annotation.provenance["groups_verified"] = False
    result = apply_annotation_updates(annotation, candidate,
                                      plan_reannotation([annotation], layers=("all",)), profiles=profiles)
    assert result.ok
    assert result.annotation.provenance["groups_verified"] is True
    assert result.annotation.provenance["pronunciation_choices"] == [1, 2]


def test_fa_domain_change_cannot_truncate_preserved_notes(annotation, profiles):
    candidate = deepcopy(annotation)
    candidate.duration = .9
    for name, tier in candidate.tiers.items():
        tier[-1] = replace(tier[-1], end=.9)
    result = apply_annotation_updates(annotation, candidate,
                                      plan_reannotation([annotation], layers=("fa",)), profiles=profiles)
    assert result.status == "conflict"
    assert annotation.tiers["notes"][-1].end == 1


def test_foreign_json_and_nonfinite_metadata_are_not_overwritten(tmp_path, annotation):
    root = tmp_path / "assets"
    path = root / annotation.sample.subset / (annotation.sample.id + ".json")
    path.parent.mkdir(parents=True)
    path.write_text('{"source": "original"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="Unrelated"):
        write_annotations(annotation, root)
    assert json.loads(path.read_text()) == {"source": "original"}
    annotation.quality["confidence"] = float("nan")
    with pytest.raises(ValueError):
        write_annotations(annotation, tmp_path / "invalid")
    assert not list((tmp_path / "invalid").rglob("*.TextGrid"))


def test_result_keeps_pending_and_success_then_retry_clears_problem(tmp_path, annotation):
    root = tmp_path / "assets"
    pending = Sample("bad", annotation.sample.audio, subset=annotation.sample.subset)
    counts = save_result(BatchResult([Outcome(annotation.sample, "annotated", annotation),
                                     Outcome(pending, "pending", issues=[Issue("g2p", "unsupported token")])]), root)
    assert counts["saved"] == counts["pending"] == 1
    assert [o.sample.id for o in load_problems(root).outcomes] == ["bad"]
    repaired = deepcopy(annotation)
    repaired.sample.id = "bad"
    save_result(BatchResult([Outcome(repaired.sample, "annotated", repaired)]), root)
    assert load_problems(root).outcomes == []


def test_unserializable_candidate_does_not_abort_other_samples(tmp_path, annotation):
    invalid = deepcopy(annotation)
    invalid.sample.id = "invalid"
    invalid.quality["confidence"] = float("nan")
    counts = save_result(BatchResult([Outcome(invalid.sample, "annotated", invalid),
                                     Outcome(annotation.sample, "annotated", annotation)]), tmp_path / "results")
    assert counts["saved"] == counts["conflict"] == 1
    problems = json.loads((tmp_path / "results/05-review-and-decisions/issues.json").read_text())
    assert "candidate_serialization_error" in problems[invalid.key + ".TextGrid"]


def test_buffer_audio_can_be_persisted_with_result(tmp_path, annotation):
    annotation.sample.audio = AudioSource(waveform=np.zeros(48000, dtype=np.float32), sample_rate=48000)
    counts = save_result(BatchResult([Outcome(annotation.sample, "annotated", annotation)]), tmp_path / "buffer")
    assert counts["saved"] == 1
    assert len(list((tmp_path / "buffer" / ".work/audio").rglob("*.wav"))) == 1


def test_pending_buffer_can_be_retried(tmp_path):
    sample = Sample("buffer", AudioSource(waveform=np.zeros(48000, dtype=np.float32), sample_rate=48000))
    result = BatchResult([Outcome(sample, "pending", issues=[Issue("missing_transcript", "Supply lyrics", "preflight")])])
    assert save_result(result, tmp_path / "pending")["pending"] == 1
    stored = load_problems(tmp_path / "pending").outcomes[0].sample
    assert stored.audio.path and stored.audio.waveform is None


def test_failed_notes_retain_the_fa_candidate_for_retry(tmp_path, annotation, profiles):
    annotation.tiers.pop("notes")
    annotation.status = "aligned"

    class FailingGame:
        def annotate(self, requests):
            return BatchResult([Outcome(a.sample, "failed", issues=[Issue("inference_failed", "interrupted", "game")])
                                for a, _ in requests])

    pipeline = Pipeline()
    pipeline._profiles, pipeline._game = profiles, FailingGame()
    result = pipeline.annotate_notes([annotation])
    save_result(result, tmp_path / "assets")
    restored = load_problems(tmp_path / "assets").problems[0].annotation
    assert restored.tiers == annotation.tiers and restored.quality == annotation.quality
    assert not list((tmp_path / "assets").rglob("*.TextGrid"))


def test_portable_originals_generated_clips_and_retry_survive_relocation(tmp_path, annotation):
    bundle = tmp_path / "before"
    dataset, delivery = bundle / "originals", bundle / "delivery"
    source = dataset / "disc 1" / "take.wav"
    source.parent.mkdir(parents=True)
    shutil.copyfile(annotation.sample.audio.path, source)
    source.with_suffix(".lab").write_text("bright")
    generated = delivery / "01-clips" / "cut.wav"
    generated.parent.mkdir(parents=True)
    shutil.copyfile(source, generated)
    store_origins(generated.parent, [(generated, {"origin_recording": {"path": str(source)},
                  "origin_slice": {"id": "cut", "start_sample": 0, "end_sample": 48000}})], source_root=dataset)
    sample, = discover_pairs(dataset, language="en")
    sample.metadata.update(audio={"path": str(source)}, source_csv=str(dataset / "labels.csv"),
                           transcript_change={"previous_path": str(source.with_suffix(".txt"))})
    annotation.sample = sample
    annotation.provenance.update(source_annotation=str(source.with_suffix(".TextGrid")),
                                 layer_origins={"phones": str(source.with_suffix(".TextGrid")), "notes": "model"})
    stage = delivery / "03-annotations"
    asset = write_annotations(annotation, stage)
    candidate = deepcopy(annotation)
    candidate.sample.id = "pending"
    save_result(BatchResult([Outcome(candidate.sample, "failed", candidate, [Issue("decode", str(source))])]), stage)
    checkpoint = delivery / ".work/checkpoint.json"
    save_checkpoint(annotation, checkpoint)
    def strings(v):
        if isinstance(v, dict):
            yield from v.keys()
            for child in v.values():
                yield from strings(child)
        elif isinstance(v, list):
            for child in v:
                yield from strings(child)
        elif isinstance(v, str):
            yield v
    for file in delivery.rglob("*.json"):
        assert not any(Path(s).is_absolute() or PureWindowsPath(s).is_absolute()
                       for s in strings(json.loads(file.read_text(encoding="utf-8")))), file
    assert list(json.loads((delivery / ".metadata/recordings.json").read_text())) == ["disc 1/take.wav"]
    moved = tmp_path / "after"
    bundle.rename(moved)
    new_source = moved / source.relative_to(bundle)
    restored = read_annotations(moved / asset.relative_to(bundle))
    assert Path(restored.sample.audio.path) == new_source
    assert Path(restored.sample.transcript.path) == new_source.with_suffix(".lab")
    assert Path(restored.sample.metadata["audio"]["path"]) == new_source
    assert Path(restored.provenance["layer_origins"]["phones"]) == new_source.with_suffix(".TextGrid")
    assert restored.provenance["layer_origins"]["notes"] == "model"
    assert source_for(moved / generated.relative_to(bundle))["origin_recording"]["path"] == str(new_source)
    issue, = load_problems(moved / stage.relative_to(bundle)).problems
    assert issue.annotation.sample.audio.path == issue.sample.audio.path == str(new_source)
    assert load_checkpoint(moved / checkpoint.relative_to(bundle)).sample.audio.path == str(new_source)


def test_separate_volume_binding_and_explicit_relocation(tmp_path, monkeypatch):
    source, delivery = tmp_path / "dataset", tmp_path / "result"
    source.mkdir()
    local = tmp_path / "machine/bindings.json"
    monkeypatch.setattr("makediffsinger.catalog._local_bindings", lambda: local)
    def separate_volumes(*args):
        raise ValueError("different volumes")
    monkeypatch.setattr("makediffsinger.catalog.os.path.relpath", separate_volumes)
    bind_dataset(delivery, source)
    assert dataset_root(delivery) == source
    assert json.loads((delivery / ".metadata/dataset.json").read_text())["relative_root"] is None
    relocated = source.rename(tmp_path / "relocated")
    with pytest.raises(ValueError, match="another dataset"):
        bind_dataset(delivery, relocated)
    bind_dataset(delivery, relocated, relocate=True)
    assert dataset_root(delivery) == relocated
