from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from makediffsinger.lyric_match import match_lyrics
from makediffsinger.inputs import discover_pairs, refresh_transcript
from makediffsinger.lyric_handoff import alignment_inputs, write_candidate_table
from makediffsinger.lyric_pronunciation import LyricUnit, PhoneGraph, pronunciation_blocks
from makediffsinger.slicing import apply_lyrics
from makediffsinger.types import AudioSource, BatchResult, Issue, Outcome, Sample, Transcript
from makediffsinger.storage import save_result
from scripts.prepare.match_lyrics import main
from scripts.prepare.prepare_lyrics import main as prepare
from scripts.review_pending import main as review


def _g2p(texts, languages):
    phones = {"nite": ["night"]}
    return [{"words": [{"text": word, "paths": [phones.get(word, [word])]}
                       for word in re.findall(r"\w+", text.casefold())]} for text in texts]


def _pipeline(pronunciations=_g2p, **methods):
    return SimpleNamespace(tifa=SimpleNamespace(pronunciations=pronunciations, info={}, **methods))


def test_song_spans_allow_repeated_omitted_and_reordered_passages():
    reference = """Composer: Example
[Intro]
[00:23.00] Copper sky
(Chorus) Follow the river
None of these lines must be sung
Copper sky
Over the hill
"""
    vocalise = "啦" * 16
    reference += vocalise
    samples = [Sample(key, AudioSource()) for key in ("first", "second", "jump", "silent", "vocalise", "symbol")]
    clips = {key: {"text": text} for key, text in {
        "first": "Over the hill", "second": "Copper sky", "jump": "Over the hill copper sky",
        "silent": "", "vocalise": vocalise + "。", "symbol": "♪",
    }.items()}
    result = match_lyrics(samples, reference, {"clips": clips}, pipeline=_pipeline(), verify=False)
    records = result["clips"]
    assert records["first"]["suggestion"] == "Over the hill"
    exact = records["second"]["candidates"][0]
    assert exact["text"] == "Copper sky"
    assert [(r["first_line"], r["first_column"]) for r in exact["source_spans"]] == [(3, 12), (6, 1)]
    joined = records["jump"]["candidates"][0]
    assert joined["joined"] and joined["text"] == "Over the hill Copper sky"
    assert records["jump"]["status"] == "suggested"
    assert records["jump"]["diagnostics"] == ["joined_without_acoustic_verification"]
    assert records["silent"]["status"] == "unmatched"
    assert records["silent"]["reason"] == "missing_or_empty_asr"
    assert records["symbol"]["asr_text"] == "♪" and records["symbol"]["reason"] == "no_usable_asr_pronunciation"
    assert records["vocalise"]["suggestion"] == vocalise
    assert [line["line"] for line in result["excluded_lines"]] == [1, 2, 4]


def test_pronunciation_fusion_uses_retrieved_reference_and_preserves_transcript():
    pfml = 'bright night <word text="yes" script="yes" phonemes="en/yes"/>'

    def fuse_lyrics(requests, languages, *, min_score, top_k):
        assert languages == ["en"]
        assert (min_score, top_k) == (-0.5, 2)
        assert [(sample.id, reference, asr) for sample, reference, asr in requests] == [
            ("clip", "bright night yes", "bright nite"), ("fallback", "bright night yes", "bright nite")]
        return [{"status": "scored", "texts": ["bright", "night", "yes"],
                 "transcript": {"content": pfml, "format": "pfml"}, "margin": .02, "blocks": []},
                {"status": "unavailable", "reason": "Test inference failure"}]

    sample = Sample("clip", AudioSource(), language="en")
    fallback = Sample("fallback", AudioSource(), language="en")
    report = {"clips": {s.id: {"text": "bright nite"} for s in (sample, fallback)}}
    result = match_lyrics([sample, fallback], "bright night yes\nunder silver clouds",
                          report, languages=["en"], pipeline=_pipeline(fuse_lyrics=fuse_lyrics),
                          candidate_count=8, min_score=-0.5, top_k=2)
    record = result["clips"]["clip"]
    assert record["candidates"][0]["text"] == "bright night"
    assert record["suggestion"] == "bright night yes" and record["status"] == "suggested"
    assert record["transcript"] == {"content": pfml, "format": "pfml"}
    assert record["reason"] == "tifa_pronunciation_fusion"
    assert record["margin"] == .02
    assert result["settings"]["min_score"] == -0.5 and result["settings"]["top_k"] == 2
    assert sample.transcript is None
    fallback_record = result["clips"]["fallback"]
    assert fallback_record["suggestion"] == "bright night" and fallback_record["status"] == "suggested"
    assert fallback_record["diagnostics"] == ["tifa_could_not_verify"]


def test_programmatic_match_table_only_applies_on_request(tmp_path, annotation, monkeypatch):
    monkeypatch.setattr("scripts.prepare.match_lyrics.pipeline", lambda args: nullcontext(_pipeline()))
    audio = Path(annotation.sample.audio.path)
    missing = audio.with_name("missing.wav")
    missing.write_bytes(audio.read_bytes())
    lyrics = tmp_path / "song.txt"
    lyrics.write_text("A better place\nAn empty room", encoding="utf-8")
    asr = tmp_path / "asr"
    asr.mkdir()
    (asr / audio.with_suffix(".lab").name).write_text("a better place", encoding="utf-8")
    dest = tmp_path / "matches"
    assert main([str(audio.parent), "--lyrics", str(lyrics), "--asr", str(asr), "--g2p-only", "--language", "en",
                 "--output", str(dest)]) == 0
    assert not audio.with_suffix(".lab").exists()
    assert apply_lyrics(dest / "lyrics.csv", audio.parent) == {"written": 1, "pending": ["missing"]}
    assert audio.with_suffix(".lab").read_text(encoding="utf-8").strip() == "A better place"


def test_match_report_preserves_a_winning_asr_word_absent_from_the_reference():
    def fuse_lyrics(requests, _languages, *, min_score, top_k):
        assert (min_score, top_k) == (0.0, 3)
        assert [(sample.id, reference, asr) for sample, reference, asr in requests] == [
            ("clip", "bright", "bright now")]
        return [{"status": "scored", "texts": ["bright", "now"], "margin": None,
                 "transcript": {"content": "bright now", "format": "text"},
                 "blocks": [{"source": "reference", "texts": ["bright"]},
                            {"source": "asr", "texts": ["now"]}]}]

    result = match_lyrics([Sample("clip", AudioSource())], "bright",
                          {"clips": {"clip": {"text": "bright now"}}},
                          pipeline=_pipeline(fuse_lyrics=fuse_lyrics))
    record = result["clips"]["clip"]
    assert record["suggestion"] == record["transcript"]["content"] == "bright now"
    assert record["pronunciation_fusion"]["blocks"][-1]["source"] == "asr"


def test_missing_working_lyrics_reach_fa_with_the_selected_reading(tmp_path, annotation):
    clips = tmp_path / "01-clips"
    clips.mkdir()
    wav = clips / "voice.wav"
    wav.write_bytes(Path(annotation.sample.audio.path).read_bytes())
    sample, = discover_pairs(clips)
    pfml = '<word text="oh" phonemes="en/ow"/>'
    winner = {"content": pfml, "format": "pfml"}
    write_candidate_table([sample], {"voice": {"transcript": winner, "asr_text": "oh"}},
                          tmp_path / "02-lyrics/match.csv")
    retry_table = tmp_path / "02-lyrics/retry.csv"
    alternate = write_candidate_table([sample], {"voice": {"text": "other words"}}, retry_table)
    assert alternate["voice"].content == "other words"
    problem = Outcome(sample, "pending", issues=[Issue("missing_transcript", "No paired lyrics.", "preflight")])
    out = tmp_path / "03-annotations"
    decisions = tmp_path / "05-review-and-decisions"
    save_result(BatchResult([problem]), out)
    assert review([str(out), "--output", str(decisions)]) == 2
    manifest = json.loads((decisions / "pending.json").read_text(encoding="utf8"))
    assert not manifest["pending"] and not list(decisions.rglob("*.wav"))
    row, = manifest["return_to_alignment"]
    assert row["lyric_candidate"] == winner and row["asr_text"] == "oh"
    assert "attempted_transcript" not in row
    selected, = alignment_inputs([sample], out)
    assert refresh_transcript(selected).transcript == Transcript(pfml, "pfml")
    excerpt = replace(sample, audio=AudioSource(str(wav), start_sample=1))
    assert alignment_inputs([excerpt], out)[0].transcript is None
    write_candidate_table([excerpt], {"voice": {"text": "excerpt"}}, tmp_path / "02-lyrics/excerpt.csv")
    wav.with_suffix(".txt").write_text("edited words", encoding="utf8")
    assert refresh_transcript(selected).transcript.content == "edited words"
    assert refresh_transcript(alignment_inputs([excerpt], out)[0]).transcript.content == "excerpt"
    annotation.sample = selected
    save_result(BatchResult([Outcome(selected, "annotated", annotation)]), out)
    assert review([str(out), "--output", str(decisions)]) == 0
    assert not (decisions / "pending.json").exists()


def test_asr_only_handoff_keeps_nonempty_hypotheses_and_distinguishes_empty_asr(tmp_path, annotation):
    clips = tmp_path / "01-clips"
    clips.mkdir()
    for key in ("empty", "vocalise"):
        (clips / (key + ".wav")).write_bytes(Path(annotation.sample.audio.path).read_bytes())
    report = tmp_path / "asr.json"
    report.write_text(json.dumps({"clips": {"empty": {"text": ""}, "vocalise": {"text": "啦啦"}}}), encoding="utf8")
    assert prepare([str(clips), "--asr", str(report), "--output", str(tmp_path / "02-lyrics/song.csv")]) == 0
    empty, vocalise = alignment_inputs(discover_pairs(clips), tmp_path)
    assert empty.transcript is None and refresh_transcript(vocalise).transcript.content == "啦啦"
    blocked = Outcome(vocalise, "pending", issues=[Issue("unavailable_reading", "No encoding for this reading.", "g2p")])
    missing = Outcome(empty, "pending", issues=[Issue("missing_transcript", "No lyrics.", "preflight")])
    out = tmp_path / "03-annotations"
    decisions = tmp_path / "05-review-and-decisions"
    save_result(BatchResult([missing, blocked]), out)
    assert review([str(out), "--output", str(decisions)]) == 0
    rows = {r["key"]: r for r in json.loads((decisions / "pending.json").read_text(encoding="utf8"))["pending"]}
    assert rows["empty"]["asr_text"] == "" and "lyric_candidate" not in rows["empty"]
    assert rows["vocalise"]["stages"] == ["g2p"]
    assert (decisions / rows["vocalise"]["attempted_transcript"]).read_text(encoding="utf8") == "啦啦"


def test_retrieved_containing_excerpt_preserves_a_neighboring_clips_words():
    def sample(key, start):
        return Sample(key, AudioSource(), metadata={
            "origin_recording": {"path": "song.wav"},
            "origin_slice": {"start_sample": start, "end_sample": start + 10},
        })

    def fuse_lyrics(requests, _languages, **_shortlist):
        assert [(s.key, excerpt) for s, excerpt, _ in requests] == [("next", "yes"), ("previous", "bright night")]
        return [{"status": "unavailable", "reason": "Test inference failure"}] * len(requests)

    samples = [sample("next", 10), sample("previous", 0)]
    asr = {"clips": {"previous": {"text": "bright night"}, "next": {"text": "yes"}}}
    match_lyrics(samples, "bright night yes", asr, pipeline=_pipeline(fuse_lyrics=fuse_lyrics))


@pytest.mark.parametrize("options", [{"min_score": float("inf")}, {"top_k": 0}])
def test_source_shortlist_requires_finite_score_and_positive_cap(options):
    with pytest.raises(ValueError, match="Minimum source score"):
        match_lyrics([], "", {"clips": {}}, pipeline=_pipeline(), **options)


def test_ties_and_optional_thresholds_keep_the_automatic_result():
    sample = Sample("clip", AudioSource())
    inputs = ([sample], "sun rises\nmoon rises", {"clips": {"clip": {"text": "sky rises"}}})
    default = match_lyrics(*inputs, pipeline=_pipeline(), verify=False)["clips"]["clip"]
    flagged = match_lyrics(*inputs, pipeline=_pipeline(), verify=False, min_similarity=.9)["clips"]["clip"]
    assert flagged["phone_edit_distance_gap"] == 0
    assert flagged["status"] == default["status"] == "suggested"
    assert flagged["suggestion"] == default["suggestion"]
    assert flagged["diagnostics"] == ["weak_retrieval", "competing_lyrics"]


def test_phonetic_search_uses_alternative_paths_without_crossing_readings():
    paths = {"foo": [["p", "q"], ["r", "s"]], "baz": [["u", "v"], ["p", "s"]],
             "bar": [["x", "y"], ["p", "s"]]}

    def pronunciations(texts, languages):
        return [{"words": [{"text": text, "paths": paths[text]}]} for text in texts]

    sample = Sample("clip", AudioSource())
    result = match_lyrics([sample], "foo\nbaz", {"clips": {"clip": {"text": "bar"}}},
                          pipeline=_pipeline(pronunciations), verify=False)["clips"]["clip"]
    assert result["suggestion"] == "baz"
    distances = {item["text"]: item["phone_edit_distance"] for item in result["candidates"]}
    assert distances == {"baz": 0, "foo": 1}


def test_word_traceback_keeps_complete_paths_through_tokenization_changes():
    def graph(*words):
        return PhoneGraph.from_units([LyricUnit(1, i, i + 1, (paths,)) for i, paths in enumerate(words)])

    reference = graph((("m", "o", "cl", "t", "o"),), (("n", "i"),))
    recognition = graph((("m", "o"), ("m", "o", "u")), (("t", "o"),))
    assert pronunciation_blocks(reference, recognition) == [
        (slice(0, 1), slice(0, 2)), (slice(1, 2), slice(2, 2))]
