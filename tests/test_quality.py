from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path

import pytest

from scripts.quality_tail import main as tail_main
from scripts import export_diffsinger as export_script
from scripts.report import main as report_main
from scripts.review_pack import main as review_main
from makediffsinger.review import make_pending_review_pack, make_review_pack, sync_quality_review
from makediffsinger.quality import quality_summary, write_overview
from makediffsinger.backends import protocol
from makediffsinger.backends.tifa_statistics import pronunciation_records
from makediffsinger.selection import QualityRule, Selection, select, select_quality
from makediffsinger.storage import write_annotations
from makediffsinger.types import AudioSource, Issue, Outcome, Transcript


def scored(annotation, values):
    items = []
    for index, value in enumerate(values):
        item = deepcopy(annotation)
        item.sample.id = f"clip_{index}"
        item.quality.update(confidence=value, determinacy=.8, monotonicity=.8, num_skipped_tokens=0)
        items.append(item)
    return items


def test_report_review_and_exclusion_use_tifa_diagnosis_order(annotation):
    items = scored(annotation, [-.4, .9, .4, .7, None, float("nan"), -.9, .7])
    for item in items[:4]:
        score = item.quality["confidence"]
        item.quality.update(agreement=score, determinacy=score, monotonicity=score)
    items[1].quality["num_skipped_tokens"] = 2
    items[3].quality["num_skipped_tokens"] = 1
    items[6].stale = ["quality"]
    items[7].quality.pop("num_skipped_tokens")
    diagnosis = quality_summary(items)["diagnosis"]
    review = select_quality(items, QualityRule(bottom_fraction=.5))
    retained = select(items, Selection(drop_bottom_fraction=.5))
    assert [r["identifier"] for r in diagnosis[:4]] == [items[i].key for i in (1, 0, 3, 2)]
    assert [a.sample.id for a in review.annotations] == ["clip_1", "clip_0"]
    assert [a.sample.id for a in retained.annotations] == [f"clip_{i}" for i in range(2, 8)]
    assert review.details["eligible"] == 4 and review.details["unavailable"] == 4
    assert {x["reason"] for x in review.excluded} >= {"quality_missing", "quality_invalid", "quality_outdated"}
    assert {x["reason"] for x in retained.excluded} == {"bottom_quality_fraction"}


def test_grouped_fraction_has_no_minimum_quota_and_stable_ties(annotation):
    items = scored(annotation, [.5, .5, .1])
    for item, subset in zip(items, ("a", "a", "b")):
        item.sample.subset = subset
    grouped = select_quality(reversed(items), QualityRule(bottom_fraction=.5, group_by="subset"))
    assert [a.key for a in grouped.annotations] == ["a/clip_1"]
    assert next(g for g in grouped.details["groups"] if g["group"] == "b")["selected"] == 0


def test_quality_fraction_floors_the_requested_decimal_fraction(annotation):
    items = scored(annotation, [0.3] * 100)
    selected = select_quality(items, QualityRule(bottom_fraction=.29))
    assert len(selected.annotations) == 29


def test_quality_tail_preview_matches_applied_scope_and_leaves_assets_intact(tmp_path, annotation, profiles, capsys, monkeypatch):
    items = scored(annotation, [-.4, .3, .9, None, .2, -.8])
    for item, subset in zip(items, ("a", "a", "a", "a", "b", "other")):
        item.sample.subset = subset
        write_annotations(item, tmp_path / "assets")
    assets = tmp_path / "assets"
    original = {path: path.read_bytes() for path in assets.rglob("*") if path.is_file()}
    scope = [str(assets), "--subset", "a", "--subset", "b", "--percentile-group", "subset"]

    assert tail_main([*scope, "--bands", "2"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["population"] == 5 and "preview" not in stats
    group_a = stats["groups"][0]
    assert group_a["eligible"] == 3 and group_a["unavailable"] == {"quality_missing": 1}
    assert group_a["bands"][0]["metrics"]["confidence"]["median"] == -.4

    assert tail_main([*scope, "--preview-fraction", ".5"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview"] == {"fraction": .5, "ranked_excluded": 1, "unranked_retained": 1, "retained": 4}
    assert preview["groups"][0]["preview"]["last_excluded"]["key"] == "a/clip_0"
    assert preview["groups"][0]["preview"]["first_retained"]["key"] == "a/clip_1"
    assert preview["groups"][1]["preview"]["ranked_excluded"] == 0
    review_dir = tmp_path / "05-review-and-decisions"
    assert not review_dir.exists()

    assert report_main([*scope, "--drop-bottom-fraction", ".5"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert {row["identifier"] for row in applied["quality"]["diagnosis"]} == {"a/clip_1", "a/clip_2", "a/clip_3", "b/clip_4"}
    monkeypatch.setattr(export_script, "get_profiles", lambda _: profiles)
    output = tmp_path / "training"
    assert export_script.main([*scope, "--drop-bottom-fraction", ".5", "--output", str(output)]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["samples"] == 4 and handoff["quality_filter"]["eligible"] == 4
    assert handoff["quality_filter"]["excluded"] == [{"key": "a/clip_0", "reason": "bottom_quality_fraction"}]
    row, = json.loads((review_dir / "quality-review.json").read_text())["excluded"]
    assert row["key"] == "a/clip_0"
    assert (review_dir / row["audio"]).is_file() and (review_dir / row["annotation"]).is_file()
    assert not list(output.rglob("*.json"))
    assert {p.stem for p in output.rglob("*.wav")} == {"clip_1", "clip_2", "clip_3", "clip_4"}
    assert {path: path.read_bytes() for path in assets.rglob("*") if path.is_file()} == original


def test_quality_review_tracks_exact_exclusions_with_existing_pending_audio(tmp_path, annotation):
    items = scored(annotation, [-.8, .2, .8, .9])
    first, manual = items[:2]
    review = tmp_path / "05-review-and-decisions"
    pending = Outcome(first.sample, "pending", issues=[Issue("missing_layer", "Notes unavailable.")])
    published = make_pending_review_pack([pending], review)
    pending_audio = review / published["pending"][0]["audio"]
    decision = Selection(drop_bottom_fraction=.5, quality_exclusions={manual.key: "Lyrics omit a sung word."})
    selection = select(items, decision)
    sync_quality_review(selection, review)
    manifest = review / "quality-review.json"
    rows = {r["key"]: r for r in json.loads(manifest.read_text())["excluded"]}
    assert set(rows) == {first.key, manual.key}
    assert (review / rows[first.key]["audio"]) != pending_audio
    assert "Lyrics omit a sung word." in (review / "quality-review.md").read_text()
    edited = review / rows[manual.key]["annotation"]
    edited.write_text("human edit", encoding="utf8")
    sync_quality_review(selection, review)
    assert edited.read_text() == "human edit"
    # Reconsider just the first clip. Rounding this scoped tail to zero restores it.
    sync_quality_review(select(items, Selection(samples=[first.key], drop_bottom_fraction=.1)), review)
    remaining, = json.loads(manifest.read_text())["excluded"]
    assert remaining["key"] == manual.key and (review / remaining["annotation"]).is_file()
    assert not (review / rows[first.key]["audio"]).exists()
    assert pending_audio.is_file() and json.loads((review / "pending.json").read_text()) == published
    sync_quality_review(select(items, Selection(drop_bottom_fraction=0)), review)
    assert not manifest.exists() and not (review / "quality-review.md").exists()
    assert edited.read_text() == "human edit" and pending_audio.is_file()


def test_report_preserves_negative_scores_and_never_copies_samples(tmp_path, annotation, capsys):
    items = scored(annotation, [-.4, .4, None, .9])
    items[-1].stale = ["quality"]
    assets, output = tmp_path / "assets", tmp_path / "overview"
    for item in items:
        write_annotations(item, assets)
    before = set(tmp_path.rglob("*"))
    assert report_main([str(assets)]) == 0
    assert set(tmp_path.rglob("*")) == before
    report = json.loads(capsys.readouterr().out)
    metric = report["quality"]["metrics"]["confidence"]
    assert metric["count"] == 2 and metric["missing"] == metric["stale"] == 1
    assert metric["minimum"] == -.4
    assert [row["confidence"] for row in report["quality"]["diagnosis"] if "confidence" in row] == [-.4, .4]
    assert report["assets"]["source_seconds"] == 1
    assert report["assets"]["clip_seconds"] == 4
    assert report_main([str(assets), "--output", str(output)]) == 0
    diagnoses = json.loads((output / "diagnosis.json").read_text())
    assert len(diagnoses) == 3 and "confidence" not in diagnoses[-1]
    assert (output / "confidence_histogram.jpg").exists()
    assert not (output / "confidence_scatter.jpg").exists()


def test_review_pack_requires_and_exports_an_explicit_selection(tmp_path, annotation, capsys):
    assets, output = tmp_path / "assets", tmp_path / "review"
    write_annotations(annotation, assets)
    assert review_main([str(assets), "--output", str(output)]) == 1
    assert not output.exists()
    chosen = select(scored(annotation, [.2, .8]), Selection(samples=["clip_1"]))
    manifest = make_review_pack(chosen, output)
    assert manifest["samples"] == [chosen.annotations[0].key]
    assert len(list(output.rglob("*.wav"))) == len(list(output.rglob("*.TextGrid"))) == 1
    capsys.readouterr()


def test_pending_review_pack_keeps_audio_and_reason_without_a_textgrid(tmp_path, annotation):
    sample = annotation.sample
    sample.transcript = Transcript('<word text="bright" phonemes="en/b en/r en/aa en/t"/>', "pfml")
    outcome = Outcome(sample, "pending", issues=[Issue("unresolved_lyric", "Confirm the sung words.", "lyrics")])
    output = tmp_path / "05-review-and-decisions"
    missing = deepcopy(sample)
    missing.id, missing.audio, missing.transcript = "no_audio", AudioSource(), None
    blocked = Outcome(missing, "failed", issues=[Issue("invalid_input", "Missing audio.", "preflight")])
    result = make_pending_review_pack([blocked, outcome], output)
    unavailable, row = result["pending"]
    assert "audio" not in unavailable and any("Review audio unavailable" in reason for reason in unavailable["reasons"])
    assert (output / row["audio"]).is_file()
    attempted = output / row["attempted_transcript"]
    assert attempted.suffix == ".lab" and attempted.read_text(encoding="utf-8") == "bright"
    assert not list(output.rglob("*.TextGrid"))
    assert "Confirm the sung words." in (output / "review.md").read_text(encoding="utf-8")
    assert json.loads((output / "pending.json").read_text()) == result


def test_diagnosis_preserves_score_frame_pairs_and_excludes_unusable_scores(annotation):
    items = scored(annotation, [-.5, .5, .5, 1, None, float("inf"), .9])
    for index, item in enumerate(items):
        item.quality["num_frames"] = 100 + index
    items[-1].stale = ["quality"]
    summary = quality_summary(items)
    metric = summary["metrics"]["confidence"]
    assert [(row["confidence"], row["num_frames"]) for row in summary["diagnosis"] if "confidence" in row] == [
        (-.5, 100), (.5, 101), (.5, 102), (1., 103)]
    assert summary["diagnosis"][-1] == {"identifier": items[-1].key}
    assert metric["count"] == 4
    assert metric["missing"] == metric["invalid"] == metric["stale"] == 1


def test_pronunciation_scores_cross_worker_boundary_without_upstream_classes():
    class NativeAlternative(dict):
        pass

    alternatives = [NativeAlternative(index=0, scripts=["a"], phones=["a"], score=-.2),
                    NativeAlternative(index=1, scripts=["e"], phones=["e"], score=float("-inf"))]
    words = pronunciation_records([{"index": 0, "text": "a", "chosen": 0, "alternatives": alternatives}])
    stream = BytesIO()
    protocol.send(stream, words)
    stream.seek(0)
    restored = protocol.receive(stream)
    assert restored == [{"index": 0, "text": "a", "chosen": 0, "alternatives": [
        {"index": 0, "scripts": ["a"], "phones": ["a"], "score": -.2},
        {"index": 1, "scripts": ["e"], "phones": ["e"], "score": "-inf"}]}]
    assert alternatives[1]["score"] == float("-inf")


def test_overview_writes_native_tifa_histograms_and_frame_scatter(tmp_path, annotation):
    pytest.importorskip("lightning")
    pytest.importorskip("g2pflow")
    repo = Path(__file__).resolve().parents[1] / "external" / "TIFA"
    if not (repo / "lib/plot.py").is_file():
        pytest.skip("TIFA plotting source is not initialized.")
    items = scored(annotation, [-.5, 0, .5, 1] * 8)
    for index, item in enumerate(items):
        item.quality["agreement"] = 1.
        item.quality["num_frames"] = 100 + index
        item.quality["num_tokens"] = 12
    report = {"quality": quality_summary(items)}
    write_overview(report, tmp_path / "overview", plots=True, tifa_repo=repo)
    assert {path.name for path in (tmp_path / "overview").iterdir()} == {
        "diagnosis.json", "scores.json", *(m + suffix for m in ("confidence", "agreement", "determinacy", "monotonicity")
                                         for suffix in ("_histogram.jpg", "_scatter.jpg"))}
    assert len(json.loads((tmp_path / "overview/diagnosis.json").read_text())) == len(items)
