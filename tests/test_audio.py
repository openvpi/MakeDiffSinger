import csv
import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import soundfile as sf

from makediffsinger.audio import compare_stereo, load_audio, mix_stereo, replay_audio
from makediffsinger.backends.game_input import prepare_audio
from makediffsinger.cut_planning import Cut, _pause_cuts
from makediffsinger.cut_revision import inspect_cut, revise_cut
from makediffsinger.inputs import discover_pairs
from makediffsinger.script_utils import run
from makediffsinger.slicing import Slice, SlicePlan, plan_slices, replay_slices, write_audition_markers
from makediffsinger.types import Annotation, AudioSource, Sample, Syllable
from scripts.prepare import extract_channel, prepare_mono


def test_pcm24_replay_uses_exact_sample_ranges_and_independent_copies(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.arange(2000, dtype=np.int32).reshape(-1, 2) * 256, 48000, subtype="PCM_24")
    before = source.read_bytes()
    destination = replay_audio(AudioSource(str(source), 137, 809), tmp_path / "slice.wav")
    with wave.open(str(source)) as original, wave.open(str(destination)) as sliced:
        original.setpos(137)
        assert sliced.getsampwidth() == 3 and sliced.getnchannels() == 2
        assert sliced.getnframes() == 672
        assert sliced.readframes(672) == original.readframes(672)
    replay_audio(AudioSource(str(source)), tmp_path / "whole.wav").write_bytes(b"independent edit")
    assert source.read_bytes() == before


def test_cut_recipe_can_replay_or_keep_a_long_clip(tmp_path):
    rate = 16000
    source = tmp_path / "long.wav"
    audio = (.3 * np.sin(np.arange(rate * 32) * (2 * np.pi * 220 / rate))).astype(np.float32)
    sf.write(source, audio, rate)
    keep = plan_slices(source, keep_whole=True)
    assert len(keep.slices) == 1 and keep.slices[0].end_sample == len(audio)
    plan = plan_slices(source, phone_hints=[(0, 32)])
    assert len(plan.slices) >= 3
    assert all(0 <= s.start_sample < s.end_sample <= len(audio) for s in plan.slices)
    assert any("weak_loudness_valley" in s.review for s in plan.slices)
    assert any("inside_candidate_phone" in s.review for s in plan.slices)
    plan.write(tmp_path / "plan.json")
    restored = SlicePlan.read(tmp_path / "plan.json")
    assert restored == plan
    assert replay_slices(restored, tmp_path / "slices")["slices"] == len(plan.slices)


def test_stereo_mix_checks_active_windows_and_requires_explicit_preparation(tmp_path):
    rate = 8000
    voice = np.sin(np.arange(rate, dtype=np.float32) * .25) * .5
    stereo = np.column_stack((voice, voice * .5))
    mixed, evidence = mix_stereo(stereo, rate)
    np.testing.assert_allclose(mixed, voice * .75)
    assert evidence["method"] == "average"

    stereo[rate // 2:rate // 2 + rate // 10, 1] = -voice[rate // 2:rate // 2 + rate // 10]
    mixed, evidence = mix_stereo(stereo, rate)
    np.testing.assert_array_equal(mixed, voice)
    assert evidence["method"] == "first_channel"
    assert evidence["worst_mix_loss_db"] is None

    source = tmp_path / "stereo.wav"
    sf.write(source, stereo, rate)
    with pytest.raises(ValueError, match="Expected mono audio"):
        load_audio(AudioSource(str(source)))
    with pytest.raises(ValueError, match="Expected mono audio"):
        plan_slices(source, keep_whole=True)


def test_stereo_comparison_separates_content_from_gain_delay_and_polarity():
    rate = 8000
    time = np.arange(10 * rate) / rate
    left = .5 * np.sin(2 * np.pi * 220 * time)
    second = .5 * np.sin(2 * np.pi * 440 * time)
    half = 5 * rate
    for right, expected in ((left * .4, "average"), (-left, "first_channel"),
                            (np.zeros_like(left), "average"), (second, "split_channels"),
                            (np.r_[-left[:half], second[half:]], "regional_channels")):
        evidence = compare_stereo(np.column_stack((left, right)), rate)
        assert evidence["method"] == expected

    delayed = np.r_[np.zeros(45), left[:-45]]
    evidence = compare_stereo(np.column_stack((left, delayed)), rate)
    assert len(evidence["regions"]) == 1 and len(evidence["regions"][0]["channels"]) == 1

    evidence = compare_stereo(np.column_stack((np.r_[left[:half], np.zeros(half)],
                                               np.r_[np.zeros(half), second[half:]])), rate)
    assert [region["channels"] for region in evidence["regions"]] == [["left"], ["right"]]


def test_prepare_mono_splits_without_overwriting_and_records_actions(tmp_path, capsys):
    rate = 8000
    source = tmp_path / "source" / "subset"
    source.mkdir(parents=True)
    time = np.arange(5 * rate) / rate
    left = (.5 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    right = (.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    sf.write(source / "clip.wav", np.column_stack((left, right)), rate, subtype="FLOAT")
    sf.write(source / "clip_L.wav", right, rate, subtype="FLOAT")
    sf.write(source / "inverted.wav", np.column_stack((left, -left)), rate, subtype="FLOAT")
    (source / "clip.lab").write_text("Original lyrics", encoding="utf-8")
    output = tmp_path / "mono"
    notes = tmp_path / "delivery" / "processing-notes.md"
    original = (source / "clip.wav").read_bytes()

    assert run(prepare_mono.build_parser(), prepare_mono.execute, [str(source.parent)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["counts"] == {"split_channels": 1, "already_mono": 1, "first_channel": 1}
    assert not output.exists() and not notes.exists()

    args = [str(source.parent), "--output", str(output), "--copy-transcripts", "--processing-notes", str(notes)]
    assert run(prepare_mono.build_parser(), prepare_mono.execute, args) == 0
    report = json.loads(capsys.readouterr().out)
    split = next(item for item in report["files"] if item["method"] == "split_channels")
    assert split["outputs"] == ["subset/clip_L_2.wav", "subset/clip_R.wav"]
    for name, expected in (("clip_L_2", left), ("clip_R", right), ("clip_L", right), ("inverted", left)):
        data, output_rate = sf.read(output / "subset" / f"{name}.wav", dtype="float32")
        assert output_rate == rate
        np.testing.assert_array_equal(data, expected)
    assert not list(output.rglob("*.lab"))
    content = notes.read_text(encoding="utf-8")
    assert "clip_L_2.wav" in content and "left unassigned" in content and "Kept the left channel" in content
    assert (source / "clip.wav").read_bytes() == original


def test_stereo_only_counts_continuous_difference():
    rate = 8000
    time = np.arange(9 * rate) / rate
    shared = .5 * np.sin(2 * np.pi * 220 * time)
    distinct = .5 * np.sin(2 * np.pi * 440 * time)
    short = np.column_stack((shared, shared))
    short[rate:rate + rate // 5, 1] = distinct[rate:rate + rate // 5]
    assert compare_stereo(short, rate)["method"] == "average"
    assert compare_stereo(short, rate, min_difference_seconds=.2)["method"] == "regional_channels"

    interrupted = np.column_stack((shared, distinct))
    interrupted[3 * rate:4 * rate] = 0
    interrupted[7 * rate:, 1] = shared[7 * rate:]
    assert compare_stereo(interrupted, rate)["method"] == "average"


def test_prepare_mono_isolates_five_seconds_of_difference_and_extracts_exact_ranges(tmp_path, capsys):
    rate = 8000
    time = np.arange(12 * rate + 17) / rate
    left = (.5 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    right = left.copy()
    start, end = 25600, 25600 + 5 * rate
    right[start:end] = .5 * np.sin(2 * np.pi * 440 * time[start:end])
    source, output = tmp_path / "take.wav", tmp_path / "mono"
    notes = tmp_path / "processing-notes.md"
    sf.write(source, np.column_stack((left, right)), rate, subtype="FLOAT")
    args = [str(source), "--output", str(output), "--processing-notes", str(notes)]
    assert run(prepare_mono.build_parser(), prepare_mono.execute, args) == 0
    report = json.loads(capsys.readouterr().out)["files"][0]
    assert report["method"] == "regional_channels"
    assert [(region["start_sample"], region["end_sample"], region["channels"]) for region in report["regions"]] == [
        (0, start, ["average"]), (start, end, ["left", "right"]), (end, len(left), ["average"]),
    ]
    expected = {"take_r001": left[:start], "take_r002_L": left[start:end],
                "take_r002_R": right[start:end], "take_r003": left[end:]}
    for sample in discover_pairs(output):
        waveform, _ = load_audio(sample.audio)
        np.testing.assert_array_equal(waveform, expected.pop(sample.id))
        assert sample.metadata["origin_recording"]["path"] == str(source)
        if sample.id == "take_r002_R":
            assert sample.metadata["origin_slice"]["start_sample"] == start
            assert sample.metadata["origin_slice"]["channel"] == "right"
    assert not expected
    assert f"[{start}, {end})" in notes.read_text(encoding="utf-8")

    plan = plan_slices(output / "take_r002_R.wav", keep_whole=True)
    clips = tmp_path / "archive"
    replay_slices(plan, clips)
    markers = clips / "take.csv"
    write_audition_markers([plan], markers)
    with markers.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream, delimiter="\t")) == [{
            "Name": "take_r002_R", "Start": str(start), "Duration": str(end - start),
            "Time Format": "8000 Hz", "Type": "Cue", "Description": "take.wav (right)",
        }]
    origin = discover_pairs(clips)[0].metadata
    assert origin["origin_recording"]["path"] == str(source)
    assert origin["origin_slice"]["start_sample"] == start and origin["origin_slice"]["end_sample"] == end

    probe = tmp_path / "probe.wav"
    args = [str(source), "--range-samples", str(start + 3), str(end - 7), "--channel", "right", "--output", str(probe)]
    assert run(extract_channel.build_parser(), extract_channel.execute, args) == 0
    np.testing.assert_array_equal(sf.read(probe, dtype="float32")[0], right[start + 3:end - 7])
    assert run(extract_channel.build_parser(), extract_channel.execute, args) == 1


def test_recording_plan_prefers_a_sustained_pause_over_a_short_trough(tmp_path):
    rate = 8000
    source = tmp_path / "recording.wav"
    audio = .3 * np.sin(np.arange(round(20.26 * rate)) * (2 * np.pi * 220 / rate))
    audio[round(9.1 * rate):round(9.13 * rate)] = 0
    audio[round(11.4 * rate):round(11.6 * rate)] = 0
    sf.write(source, audio, rate)

    left, right = plan_slices(source).slices
    assert left.start_sample == 0 and right.end_sample == len(audio)
    assert left.end_sample == right.start_sample
    assert 11.4 <= left.end_sample / rate <= 11.6
    assert left.cut_evidence[0]["cut_kind"] == "weak"


def test_recording_plan_can_pass_over_the_first_game_pause(tmp_path):
    rate = 8000
    source = tmp_path / "recording.wav"
    audio = .3 * np.sin(np.arange(20 * rate) * (2 * np.pi * 220 / rate))
    audio[round(5.7 * rate):round(6.3 * rate)] = 0
    audio[round(9.6 * rate):round(10.4 * rate)] = 0
    sf.write(source, audio, rate)

    left, right = plan_slices(source).slices
    assert (left.start_sample, left.end_sample, right.start_sample, right.end_sample) == (0, 10 * rate, 10 * rate, 20 * rate)
    assert left.cut_evidence[0]["in_silence"]


@pytest.mark.parametrize(("duration", "minima", "expected", "required"), [
    (80, (10, 70), (10, 10), False),
    (160, (70, 90), (70, 70), False),
    (160, (25, 135), (25, 135), False),
    (200, (25, 175), (25, 175), False),
    (399, (25, 375), (25, 375), False),
    (400, (25, 375), (25, 375), True),
])
def test_pause_cuts_use_shared_rms_tie_order_and_mark_required_pauses(duration, minima, expected, required):
    slicer = SimpleNamespace(sr=100, min_interval=30)
    positions = np.arange(200 + duration)
    silent = (positions >= 100) & (positions < 100 + duration)
    rms = np.where(silent, .008, .1)
    rms[100 + np.array(minima)] = .002

    cuts = _pause_cuts(positions, rms, silent, slicer, weak_threshold=.03)

    assert cuts[1:-1] == [Cut(100 + expected[0], 100 + expected[1], "pause", required)]


@pytest.mark.parametrize("voice_seconds, pause_seconds, should_split", [(1.5, 3, False), (7, 3, True), (1.5, 4.2, True)])
def test_recording_plan_respects_required_pauses_and_clip_duration(tmp_path, voice_seconds, pause_seconds, should_split):
    rate = 8000
    source = tmp_path / "recording.wav"
    voice = .3 * np.sin(np.arange(round(voice_seconds * rate)) * (2 * np.pi * 220 / rate))
    sf.write(source, np.r_[voice, np.zeros(round(pause_seconds * rate)), voice], rate)

    clips = plan_slices(source).slices
    end = round((2 * voice_seconds + pause_seconds) * rate)
    assert clips[0].start_sample == 0 and clips[-1].end_sample == end
    if not should_split:
        assert len(clips) == 1
        return
    left, right = clips
    assert left.end_sample < right.start_sample
    assert left.end_sample / rate == pytest.approx(voice_seconds + 1, abs=.06)
    assert right.start_sample / rate == pytest.approx(voice_seconds + pause_seconds - 1, abs=.06)
    assert left.cut_evidence[0]["in_silence"] and right.cut_evidence[0]["in_silence"]


def test_local_cut_revision_preserves_source_pcm_and_neighboring_bounds(tmp_path):
    rate = 1000
    source = tmp_path / "source.wav"
    audio = np.full(3000, 8192, dtype=np.int16)
    audio[1050:1070] = 0
    sf.write(source, audio, rate)
    info = plan_slices(source, keep_whole=True).source
    outer = {"sample": 250, "reasons": ["outer_hint"]}
    old_cut = {"sample": 1000, "reasons": ["supplemental_rms_cut"]}
    plan = SlicePlan(info, [Slice("a", 250, 1000, ["outer_hint", "supplemental_rms_cut"], [outer, old_cut]),
                            Slice("b", 1000, 2000, ["supplemental_rms_cut"], [old_cut]),
                            Slice("c", 2000, 2800)])

    report = inspect_cut(plan, "a", "b", 950, 1100)
    rms = {row["sample"]: row["rms"] for row in report["rms"]}
    assert report["cut_sample"] == 1000 and rms[1000] == .25 and rms[1060] == 0
    trial = revise_cut(plan, "a", "b", move_to=1060)
    assert trial.slices == [Slice("a", 250, 1060, ["outer_hint"], [outer]), Slice("b", 1060, 2000)]
    assert plan.slices[0].end_sample == 1000 and plan.slices[2] == Slice("c", 2000, 2800)
    replay_slices(trial, tmp_path / "trial")
    left, _ = sf.read(tmp_path / "trial" / "a.wav", dtype="int16")
    right, _ = sf.read(tmp_path / "trial" / "b.wav", dtype="int16")
    np.testing.assert_array_equal(np.concatenate([left, right]), audio[250:2000])
    assert revise_cut(plan, "a", "b", merge_as="ab").slices == [Slice("ab", 250, 2000, ["outer_hint"], [outer])]
    with pytest.raises(ValueError, match="consecutive"):
        revise_cut(plan, "a", "c", move_to=1500)


def test_model_padding_cannot_hide_bad_timing_or_change_source():
    source = np.linspace(-.5, .5, 3280, dtype=np.float32)
    original = source.copy()
    annotation = Annotation(Sample("clip", AudioSource(waveform=source, sample_rate=16000)), .21, {},
                            provenance={"time_resolution_seconds": .01})
    config = dict(rate=16000, timestep=.01, hop_size=160, fft_size=512, win_size=512)
    waveform, padding = prepare_audio(annotation, [Syllable(0, 1, 0, .21)], **config)
    assert padding == 80 and len(waveform) == 3360
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(waveform[:len(source)], original)
    assert np.all(waveform[-padding:] == 0)
    with pytest.raises(ValueError, match="cannot be represented"):
        prepare_audio(annotation, [Syllable(0, 1, 0, .105), Syllable(1, 1, .105, .21)], **config)
    annotation.duration = .31
    with pytest.raises(ValueError, match="source mapping"):
        prepare_audio(annotation, [Syllable(0, 1, 0, .31)], **config)


def test_corrupt_audio_reports_decode_or_missing_decoder_error(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    with pytest.raises((ValueError, OSError, ImportError)):
        load_audio(AudioSource(str(broken)))


def test_unconstrained_audio_uses_only_the_native_clip_domain():
    source = np.zeros(3297, dtype=np.float32)
    annotation = Annotation(Sample("clip", AudioSource(waveform=source, sample_rate=16000)), len(source) / 16000, {})
    config = dict(rate=16000, timestep=.01, hop_size=160, fft_size=512, win_size=512)
    waveform, padding = prepare_audio(annotation, None, **config)
    assert len(waveform) == 3360 and padding == 63
    with pytest.raises(ValueError, match="at least one syllable"):
        prepare_audio(annotation, [], **config)
    annotation.duration += .1
    with pytest.raises(ValueError, match="source mapping"):
        prepare_audio(annotation, None, **config)
