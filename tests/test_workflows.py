import csv
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from makediffsinger.exporters.diffsinger import export_diffsinger, transcription
from makediffsinger.backends.session import BackendSession
from makediffsinger.config import Resources, Runtime
from makediffsinger.importers import import_transcriptions
from makediffsinger.inputs import discover_pairs, paired_transcript, refresh_transcript
from makediffsinger import lyric_asr
from makediffsinger.slicing import apply_lyrics, prepare_lyrics
from makediffsinger.script_utils import pipeline
from scripts.review_pending import main as review_pending
from makediffsinger.storage import load_annotations, load_problems, write_annotations
from makediffsinger.types import BatchResult, Interval, Issue, Outcome, Transcript
from makediffsinger.validation import validate
from scripts.extract_notes import build_parser
from scripts import export_diffsinger as export_script
from scripts import run_pipeline
from scripts import retry as retry_script
from makediffsinger.selection import Selection, select
from scripts.prepare import debug_g2p


def test_pair_priority_and_retry_use_current_tifa_input(annotation):
    audio = Path(annotation.sample.audio.path)
    audio.with_suffix(".lab").write_text("lab", encoding="utf-8")
    audio.with_suffix(".txt").write_text("text", encoding="utf-8")
    audio.with_suffix(".pfml").write_text("", encoding="utf-8")
    selected = discover_pairs(audio, language="en")[0]
    assert selected.transcript.format == "pfml" and selected.transcript.content == ""
    audio.with_suffix(".pfml").write_text("revised", encoding="utf-8")
    revised = refresh_transcript(selected)
    assert revised.transcript.content == "revised"
    assert revised.metadata["transcript_change"]["previous_content"] == ""
    assert selected.transcript.content == ""


def test_g2p_debug_compares_candidate_without_changing_working_pair(tmp_path, annotation, monkeypatch, capsys):
    audio = Path(annotation.sample.audio.path)
    active = audio.with_suffix(".pfml")
    active.write_text("current", encoding="utf-8")
    candidate = tmp_path / "repair.lab"
    candidate.write_text("candidate", encoding="utf-8")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def preflight(self, samples):
            sample, = samples
            assert sample.languages == ["ja", "en"]
            sample = refresh_transcript(sample)
            if sample.transcript.content == "current":
                issue = Issue("G2PConversionError", "Unknown token", "g2p")
                return BatchResult([Outcome(sample, "pending", issues=[issue])])
            assert sample.transcript == Transcript("candidate", "text")
            return BatchResult([Outcome(sample, "ready")])

    monkeypatch.setattr(debug_g2p, "pipeline", lambda args: Session())
    args = [str(audio), "--language", "ja", "--extended-language", "en"]
    assert debug_g2p.main(args) == 2
    assert json.loads(capsys.readouterr().out)["current"]["status"] == "pending"
    assert debug_g2p.main(args + ["--candidate", str(candidate)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["current"]["transcript"]["path"] == str(active.resolve())
    assert report["candidate"]["transcript"]["path"] == str(candidate.resolve())
    assert active.read_text(encoding="utf-8") == "current"
    assert candidate.read_text(encoding="utf-8") == "candidate"


def test_lyric_preparation_keeps_unmatched_audio_pending(tmp_path, annotation):
    root = Path(annotation.sample.audio.path).parent
    table = tmp_path / "lyrics.csv"
    assert prepare_lyrics(root, table) == 1
    assert apply_lyrics(table, root) == {"written": 0, "pending": ["source"]}
    with table.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(content="lyrics", format="text", extension=".txt")
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert apply_lyrics(table, root)["written"] == 1
    assert paired_transcript(annotation.sample.audio.path).content == "lyrics"
    with pytest.raises(FileExistsError):
        apply_lyrics(table, root)


def test_matching_pfml_stays_in_work_and_plain_edits_take_precedence(tmp_path):
    clips = tmp_path / "01-clips" / "song"
    clips.mkdir(parents=True)
    audio = clips / "clip.wav"
    audio.touch()
    table = tmp_path / "lyrics.csv"
    content = 'a <word text="bright" phonemes="b r aa t"/> &amp; night'
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("key", "audio", "format", "extension", "content"))
        writer.writerow(("clip", "clip.wav", "pfml", ".pfml", content))

    assert apply_lyrics(table, clips) == {"written": 1, "pending": []}
    archive = tmp_path / ".work/paired-lyrics/song/clip.pfml"
    assert archive.read_text(encoding="utf-8").strip() == content
    assert not audio.with_suffix(".pfml").exists()
    plain = audio.with_suffix(".lab")
    assert plain.read_text(encoding="utf-8").strip() == "a bright & night"
    sample = discover_pairs(audio)[0]
    assert sample.transcript.path == str(archive.resolve())
    plain.write_text("a fine night", encoding="utf-8")
    current = refresh_transcript(sample)
    assert current.transcript.format == "text" and current.transcript.content == "a fine night"


def test_csv_roundtrip_preserves_fields_without_inventing_fa_layers(tmp_path, annotation, profiles):
    row = transcription(annotation, profiles)
    assert list(row) == ["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"]
    source = tmp_path / "transcriptions.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        writer.writerow({**row, "name": "invalid", "ph_dur": "-1"})
    imported = import_transcriptions(source)
    assert [o.status for o in imported.outcomes] == ["imported", "failed"]
    converted = imported.annotations[0]
    assert set(converted.tiers) == {"phones", "notes"} and converted.quality == {}
    assert transcription(converted) == row
    manifest = export_diffsinger(imported.annotations, tmp_path / "export", copy_audio=False)
    assert len(manifest["selected"]) == 1
    assert (tmp_path / "export/und/transcriptions.csv").exists()
    assert not list((tmp_path / "export").rglob("*.yaml"))
    first, second, *others = converted.tiers["phones"]
    converted.tiers["phones"] = [replace(first, end=.103), replace(second, start=.103), *others]
    converted.tiers["words"] = [Interval(0., .3, "unused")]
    edited = transcription(converted)
    assert edited["ph_num"] == row["ph_num"] and edited["note_slur"] == row["note_slur"]


def test_all_rest_asset_remains_editable_and_cannot_be_compiled(annotation, profiles):
    annotation.tiers["notes"] = [Interval(n.start, n.end, "rest") for n in annotation.tiers["notes"]]
    assert not validate(annotation, profiles=profiles)
    with pytest.raises(ValueError, match="at least one pitched"):
        transcription(annotation, profiles)


def test_export_keeps_valid_rows_beside_an_uncompilable_item(annotation, profiles, tmp_path):
    invalid = deepcopy(annotation)
    invalid.sample.id = "all-rest"
    invalid.tiers["notes"] = [replace(n, label="rest") for n in invalid.tiers["notes"]]
    manifest = export_diffsinger([invalid, annotation], tmp_path / "export", profiles=profiles)
    assert [row["key"] for row in manifest["selected"]] == [annotation.key]
    assert [row["key"] for row in manifest["failed"]] == [invalid.key]


def test_notes_retry_uses_fa_archive_and_reports_only_missing_items(tmp_path, annotation, monkeypatch):
    from contextlib import nullcontext
    from makediffsinger.storage import save_result
    root = tmp_path / "delivery"
    final = root / "03-annotations"
    missing = replace(annotation.sample, id="missing")
    write_annotations(annotation, root / ".work/fa")
    save_result(BatchResult([Outcome(s, "pending", issues=[Issue("retry", "needs notes")])
                             for s in (annotation.sample, missing)]), final)
    def reannotate(items, **kwargs):
        assert [a.key for a in items] == [annotation.key]
        return BatchResult([Outcome(a.sample, "annotated", a) for a in items])
    monkeypatch.setattr(retry_script, "pipeline", lambda args: nullcontext(SimpleNamespace(reannotate=reannotate)))
    args = retry_script.build_parser().parse_args([str(final), "--stage", "notes", "--output", str(tmp_path / "result")])
    report, code = retry_script.execute(args)
    assert code == 2 and report["storage"]["saved"] == report["storage"]["pending"] == 1


def test_diffsinger_groups_by_language_and_user_speaker_map(tmp_path, annotation, profiles):
    first, second, third = deepcopy(annotation), deepcopy(annotation), deepcopy(annotation)
    first.sample.subset, second.sample.subset, third.sample.subset = "album/one", "album/two", "album/three"
    third.sample.language = "ja"
    third.tiers["phones"] = [Interval(interval.start, interval.end, phone)
                             for interval, phone in zip(third.tiers["phones"], ("cl", "a", "cl", "a"))]
    third.tiers["words"] = [Interval(0, .2, "cl a"), Interval(.2, 1, "cl a")]
    assets = tmp_path / "assets"
    for item in (first, second, third):
        write_annotations(item, assets)

    plain = tmp_path / "plain"
    manifest = export_diffsinger(load_annotations(assets), plain, profiles=profiles)
    assert {row["csv"] for row in manifest["selected"]} == {"en/transcriptions.csv", "ja/transcriptions.csv"}
    assert {row["audio"] for row in manifest["selected"]} == {
        "en/wavs/clip.wav", "en/wavs/clip_2.wav", "ja/wavs/clip.wav",
    }
    assert not list(plain.rglob("raw"))
    with (plain / "en/transcriptions.csv").open(encoding="utf-8", newline="") as stream:
        assert [row["name"] for row in csv.DictReader(stream)] == ["clip", "clip_2"]

    speaker_map = tmp_path / "speakers.json"
    speaker_map.write_text(json.dumps({first.key: "singer-a", second.key: "singer-b", third.key: None}), encoding="utf-8")
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"phone_classes": profiles.languages}), encoding="utf-8")
    grouped = tmp_path / "grouped"
    args = [str(assets), "--resources", str(resources), "--no-audio", "--speaker-map", str(speaker_map)]
    assert export_script.main([*args, "--output", str(grouped)]) == 0
    assert {path.relative_to(grouped).as_posix() for path in grouped.rglob("*.csv")} == {
        "singer-a/en/transcriptions.csv", "singer-b/en/transcriptions.csv", "unknown/ja/transcriptions.csv",
    }
    with (grouped / "unknown/ja/transcriptions.csv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        assert rows.fieldnames == list(transcription(third, profiles))
        assert next(rows) == transcription(third, profiles)
    assert not list(grouped.rglob("*.wav"))
    filtered = tmp_path / "filtered"
    assert export_script.main([*args, "--speaker", "singer-a", "--output", str(filtered)]) == 0
    assert (filtered / "singer-a/en/transcriptions.csv").is_file()
    assert {item.sample.speaker for item in load_annotations(assets)} == {annotation.sample.speaker}


def test_resource_configuration_cannot_choose_an_environment(tmp_path, monkeypatch):
    path = tmp_path / "resources.local.json"
    content = {"game_model": "model.pt"}
    path.write_text(json.dumps(content), encoding="utf-8")
    state = tmp_path / "local/agent-setup.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"python": "unavailable-python", "device": "cuda:99"}), encoding="utf-8")
    monkeypatch.setattr("makediffsinger.config.repository_root", lambda: tmp_path)
    assert Resources.read(path).game_model == str(tmp_path / "model.pt")
    parser = build_parser()
    arguments = ["clips", "--output", "notes", "--resources", str(path)]
    with pipeline(parser.parse_args(arguments)) as session:
        assert session.runtime.trainer_options()["devices"] == [99]
    with pipeline(parser.parse_args(arguments + ["--device", "cpu"])) as session:
        assert session.runtime.device == "cpu"
        assert session.runtime.trainer_options()["accelerator"] == "cpu"
    state.unlink()
    with pytest.raises(ValueError, match="Specify a device"):
        Runtime().trainer_options()
    assert json.loads(path.read_text(encoding="utf-8")) == content
    path.write_text(json.dumps({**content, "_setup": {"python": "another-python"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown resource fields"):
        Resources.read(path)


def test_asr_retries_on_cpu_only_for_device_incompatibility(tmp_path, annotation, monkeypatch):
    state = tmp_path / "local/agent-setup.json"
    state.parent.mkdir()
    state.write_text('{"device":"cuda:1"}', encoding="utf-8")
    monkeypatch.setattr("makediffsinger.config.repository_root", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    monkeypatch.setattr(lyric_asr, "version", lambda _: "test")
    monkeypatch.setattr(lyric_asr, "load_audio", lambda *_: ([0], 16000))
    loaded = []

    def load_model(model, device, threads, language):
        loaded.append(device)

        def decode(audio):
            if device == "cuda:1":
                raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
            if device == "cuda:0":
                raise RuntimeError("CUDA out of memory")
            return "text"

        return decode

    monkeypatch.setattr(lyric_asr, "_load_qwen", load_model)
    sample = annotation.sample
    report = lyric_asr.transcribe_clips([sample], model=tmp_path)
    assert loaded == ["cuda:1", "cpu"]
    assert report["requested_device"] == "cuda:1" and report["device"] == "cpu"
    assert report["device_fallback"] and report["clips"][sample.key]["text"] == "text"
    report = lyric_asr.transcribe_clips([sample], model=tmp_path, device="cuda:0")
    assert loaded[-1] == "cuda:0" and report["device"] == "cuda:0"
    assert report["device_fallback"] is None and report["clips"][sample.key]["error"] == "CUDA out of memory"
    assert json.loads(state.read_text(encoding="utf-8"))["device"] == "cuda:1"


def test_backend_launches_with_the_calling_interpreter(tmp_path, monkeypatch):
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference/api.py").touch()
    commands = []
    def spawn(command, **kwargs):
        commands.append(command)
        return SimpleNamespace()
    monkeypatch.setattr("makediffsinger.backends.session.subprocess.Popen", spawn)
    session = BackendSession("game", Resources(game_repo=str(tmp_path)), Runtime(device="cpu"))
    monkeypatch.setattr(session, "call", lambda *args, **kwargs: {})
    session.start()
    assert commands[0][0] == sys.executable


@pytest.mark.parametrize("pfml", [False, True])
def test_full_run_keeps_fa_and_counts_problems_across_chunks(tmp_path, annotation, profiles, monkeypatch, capsys, pfml):
    good = deepcopy(annotation)
    aligned_pfml = '<word text="bright" phonemes="en/b en/r en/aa en/t"/>'
    failed = deepcopy(annotation)
    failed.sample.id = "failed"
    inputs = {a.key: a for a in (good, failed)}
    calls = []

    class Session:
        def __enter__(self):
            calls.append("open")
            return self

        def __exit__(self, *_):
            calls.append("close")

        def preflight(self, samples):
            return BatchResult([Outcome(s, "ready") for s in samples])

        def align(self, samples):
            aligned = [deepcopy(inputs[s.key]) for s in samples]
            for item in aligned:
                item.tiers.pop("notes")
                item.provenance["_aligned_pfml"] = aligned_pfml
            return BatchResult([Outcome(a.sample, "aligned", a) for a in aligned])

        def annotate_notes(self, annotations):
            outcomes = []
            for asset in annotations:
                if asset.key == failed.key:
                    outcomes.append(Outcome(asset.sample, "failed", asset, [Issue("backend_failure", "Temporary failure", "game")]))
                else:
                    asset.tiers["notes"] = good.tiers["notes"]
                    outcomes.append(Outcome(asset.sample, "annotated", asset))
            return BatchResult(outcomes)

    session = Session()
    session.profiles = profiles
    monkeypatch.setattr(run_pipeline, "pipeline", lambda args: session)
    monkeypatch.setattr(run_pipeline, "input_samples", lambda args: [good.sample, failed.sample])
    output = tmp_path / "run"
    args = [str(tmp_path), "--output", str(output), "--chunk-size", "1", "--report-output", str(output / "04-quality")]
    assert run_pipeline.main(args + (["--pfml"] if pfml else [])) == 2
    assert calls == ["open", "close"]
    assert len(load_annotations(output / ".work/fa")) == 2
    assert len(list((output / ".work/fa").rglob("*.pfml"))) == 2
    assert len(load_annotations(output / "03-annotations")) == 1
    final_pfml = list((output / "03-annotations").rglob("*.pfml"))
    assert len(final_pfml) == int(pfml)
    if pfml:
        assert final_pfml[0].read_text(encoding="utf-8").strip() == aligned_pfml
    problem = load_problems(output / "03-annotations").outcomes[0]
    assert problem.sample.key == failed.key and set(problem.annotation.tiers) == {"texts", "words", "phones"}
    summary = json.loads(capsys.readouterr().out)
    assert summary["processed"] == summary["total"] == 2
    assert summary["counts"] == {"saved": 1, "pending": 0, "failed": 1, "conflict": 0}
    assert {path.name for path in output.iterdir()} == {
        ".work", ".metadata", "03-annotations", "04-quality", "05-review-and-decisions",
    }
    assert (output / "04-quality/metrics.json").is_file()
    assert problem.issues[0].code == "backend_failure"
    assert {d for d in output.rglob(".metadata")} == {output / ".metadata"}
    review = output / "05-review-and-decisions"
    pending, = json.loads((review / "pending.json").read_text())["pending"]
    assert pending["key"] == failed.key
    assert list(output.rglob("*.wav")) == [review / pending["audio"]]
    assert (review / pending["candidate"]).is_file()
    assert pending["audio"] in (review / "review.md").read_text()
    # A second handoff uses the same copies. Repair clears the pack while
    # preserving the reviewer's independent edit.
    finalize = [str(output / "03-annotations"), "--output", str(review)]
    assert review_pending(finalize) == 0
    assert len(list(review.rglob("*.wav"))) == 1
    lyric = review / pending["attempted_transcript"]
    lyric.write_text("a reviewer's correction")
    write_annotations(failed, output / "03-annotations")
    assert review_pending(finalize) == 0
    assert lyric.read_text() == "a reviewer's correction"
    assert not any((review / name).exists() for name in ("pending.json", "review.md", pending["audio"], pending["candidate"]))
    capsys.readouterr()


def test_multi_dataset_jobs_share_only_the_worker_and_keep_tail_denominators(tmp_path, annotation, monkeypatch, capsys):
    from contextlib import nullcontext
    inputs = [tmp_path / name / "singing" for name in ("first", "second")]
    for source in inputs:
        source.mkdir(parents=True)
        (source / "clip.wav").write_bytes(Path(annotation.sample.audio.path).read_bytes())
    worker = object()
    monkeypatch.setattr(run_pipeline, "pipeline", lambda args: nullcontext(worker))
    def annotate(job, session):
        assert session is worker
        sample, = discover_pairs(job.input)
        asset = deepcopy(annotation)
        asset.sample = sample
        asset.quality.update(agreement=.5, confidence=.5, determinacy=.5, monotonicity=.5, num_skipped_tokens=0)
        write_annotations(asset, job.output / "03-annotations")
        return {}, 0
    monkeypatch.setattr(run_pipeline, "_run_dataset", annotate)
    output = tmp_path / "deliveries"
    args = [*map(str, inputs), "--output", str(output)]
    assert run_pipeline.main(args) == 0
    jobs = json.loads(capsys.readouterr().out)["datasets"]
    assert [Path(job["output"]).name for job in jobs] == ["singing", "singing_2"]
    all_assets = []
    for source, job in zip(inputs, jobs):
        assets = load_annotations(Path(job["output"]) / "03-annotations")
        assert assets[0].sample.audio.path == str(source / "clip.wav")
        assert assets[0].key == "clip"
        assert len(select(assets, Selection(drop_bottom_fraction=.5)).annotations) == 1
        all_assets.extend(assets)
    with pytest.raises(ValueError, match="separate deliveries"):
        select(all_assets, Selection(drop_bottom_fraction=.5))
    with pytest.raises(ValueError, match="separately"):
        load_annotations(output)
