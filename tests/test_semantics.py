from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import textgrid

from makediffsinger.annotations import fill_gaps, read_textgrid, write_textgrid
from makediffsinger.exporters.diffsinger import transcription
from makediffsinger.config import Resources
from makediffsinger.phonology import PhoneProfiles, derive_syllables, group_onsets
from makediffsinger.types import Interval
from makediffsinger.validation import validate
from makediffsinger.timing import align_tiers, covered_frames, frame_coordinates, match_boundaries, phone_ownership


@pytest.mark.parametrize("kinds,expected", [
    (["vowel"], [0]),
    (["consonant", "medial", "vowel"], [1]),
    (["medial", "medial", "vowel"], [1]),
    (["consonant", "consonant"], [0]),
    (["consonant", "vowel", "medial", "medial", "vowel"], [1, 3]),
])
def test_longest_match_and_no_onset_fallback(kinds, expected):
    assert group_onsets(kinds) == expected


def test_prefix_pitch_and_multiple_notes_are_preserved(annotation, profiles):
    syllables = derive_syllables(annotation, profiles)
    assert [s.phone_count for s in syllables] == [1, 2, 1]
    assert [s.prefix for s in syllables] == [True, False, False]
    assert validate(annotation, profiles=profiles, check_audio=True) == []
    row = transcription(annotation, profiles)
    assert row["ph_num"] == "1 2 1"
    assert row["note_slur"] == "0 0 1 0"
    assert row["note_seq"] == "C3+14 D4-7 E4+12 rest"
    assert sum(map(float, row["ph_dur"].split())) == pytest.approx(annotation.duration)


def test_event_intervals_are_not_merged_or_forced_to_rest(annotation, profiles):
    for layer in ("texts", "words", "phones"):
        annotation.tiers[layer] = [Interval(0, .2, "AP"), Interval(.2, .3, "AP"), Interval(.3, 1, "SP")]
    annotation.tiers["notes"] = [Interval(0, .2, "C4"), Interval(.2, .3, "rest"), Interval(.3, 1, "D4")]
    assert not validate(annotation, profiles=profiles)
    assert transcription(annotation, profiles)["note_seq"] == "C4 rest D4"
    assert [s.phone_count for s in derive_syllables(annotation, profiles)] == [1, 1, 1]


def test_phone_classes_are_language_bound(profiles):
    assert profiles.kind("l", "en") == "medial"
    assert profiles.kind("l", "zh") == "consonant"
    assert profiles.kind("ja/cl", "en") == "consonant"


def test_two_part_dictionary_requires_complete_explicit_types(tmp_path):
    dictionary = tmp_path / "custom.txt"
    dictionary.write_text("a\ty\nka\tx y\n", encoding="utf-8")
    resources = Resources(dictionaries={"zh": str(dictionary)})
    with pytest.raises(ValueError, match="Missing phone-type record"):
        PhoneProfiles.from_resources(resources)
    resources.phone_classes = {"zh": {"x": "vowel"}}
    with pytest.raises(ValueError, match="Missing phone types.*y"):
        PhoneProfiles.from_resources(resources)
    record = tmp_path / "types.json"
    record.write_text(json.dumps({"x": "vowel", "y": "consonant"}), encoding="utf-8")
    resources.phone_classes = {"zh": str(record)}
    profiles = PhoneProfiles.from_resources(resources)
    assert profiles.kind("x", "zh") == "vowel"
    assert profiles.kind("y", "zh") == "consonant"
    assert profiles.sources["zh"] == str(record)


def test_every_configured_dictionary_in_one_language_is_checked(tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("one\ta\n", encoding="utf-8")
    second.write_text("two\tx\n", encoding="utf-8")
    config = tmp_path / "g2p.json"
    config.write_text(json.dumps({"converters": [
        {"id": "dictionary", "language": "zh", "kwargs": {"dict_path": str(path)}} for path in (first, second)
    ]}), encoding="utf-8")
    resources = Resources(g2p_config=str(config), phone_classes={"zh": {"a": "vowel"}})
    with pytest.raises(ValueError, match="Missing phone types.*second.txt.*x"):
        PhoneProfiles.from_resources(resources)
    resources.phone_classes["zh"]["x"] = "medial"
    profiles = PhoneProfiles.from_resources(resources)
    assert profiles.dictionaries["zh"] == [str(first), str(second)]
    assert profiles.kind("x", "zh") == "medial"


def test_bundled_records_cover_all_upstream_dictionary_phones():
    root = Path(__file__).resolve().parents[1]
    directory = root / "external/TIFA/dictionaries"
    if not directory.is_dir():
        pytest.skip("TIFA dictionaries are not initialized.")
    catalog = json.loads((root / "src/makediffsinger/resources/phone-profiles.json").read_text(encoding="utf-8"))
    for name, entry in catalog["dictionaries"].items():
        dictionary = directory / name
        profiles = PhoneProfiles.from_resources(Resources(dictionaries={entry["language"]: str(dictionary)}))
        inventory = {phone for line in dictionary.read_text(encoding="utf-8").splitlines()
                     if "\t" in line for phone in line.partition("\t")[2].split()}
        assert set(profiles.languages[entry["language"]]) == inventory
        assert Path(profiles.sources[entry["language"]]).name == entry["classes"]
    assert PhoneProfiles().kind("AP", None) == "vowel"


def test_gap_fill_does_not_rewrite_original_boundaries():
    original = [Interval(.123456789, .25, "x"), Interval(.25, .3, "AP"),
                Interval(.3, .4, "AP"), Interval(.4, .5, "")]
    result = fill_gaps(original, .8)
    assert result[0] == Interval(0, .123456789, "SP")
    assert result[1:4] == original[:3]
    assert result[4:] == [Interval(.4, .5, "SP"), Interval(.5, .8, "SP")]


def test_read_tiers_by_name_and_keep_precision(tmp_path, annotation):
    path = tmp_path / "ordered.TextGrid"
    write_textgrid(annotation, path)
    grid = textgrid.TextGrid.fromFile(str(path))
    grid.tiers.reverse()
    grid.write(str(path))
    duration, tiers = read_textgrid(path)
    assert duration == annotation.duration
    assert tiers == annotation.tiers


def test_serialization_canonicalizes_only_a_numerically_equal_tail(tmp_path, annotation):
    last = annotation.tiers["notes"][-1]
    annotation.tiers["notes"][-1] = replace(last, end=1.0000000000000002)
    path = tmp_path / "roundtrip.TextGrid"
    write_textgrid(annotation, path)
    duration, tiers = read_textgrid(path)
    assert tiers["notes"][-1].end == duration == 1
    assert annotation.tiers["notes"][-1].end > duration
    annotation.tiers["notes"][-1] = replace(last, end=1.012)
    with pytest.raises(ValueError):
        write_textgrid(annotation, tmp_path / "invalid.TextGrid")


def test_audio_tolerance_does_not_relax_layer_alignment(annotation, profiles):
    shifted = deepcopy(annotation)
    shifted.duration = 1.005
    for name, tier in shifted.tiers.items():
        tier[-1] = Interval(tier[-1].start, shifted.duration, tier[-1].label)
    assert not validate(shifted, profiles=profiles, check_audio=True)
    shifted.tiers["notes"][-1] = Interval(.7, .990, "rest")
    assert "layer_endpoint" in {i.code for i in validate(shifted, profiles=profiles, check_audio=True)}
    shifted.duration = 1.02
    assert "audio_endpoint" in {i.code for i in validate(shifted, check_audio=True)}


def test_unverified_groups_and_stale_imported_counts_need_review(annotation, profiles):
    annotation.provenance["groups_verified"] = False
    with pytest.raises(ValueError, match="not been identified"):
        derive_syllables(annotation, profiles)
    annotation.provenance["imported_ph_num"] = [1, 2, 1]
    annotation.provenance["imported_phones_signature"] = "obsolete"
    with pytest.raises(ValueError, match="stale"):
        derive_syllables(annotation)


def test_ordered_matching_chooses_closest_feasible_boundary_assignment():
    anchors = [0., .095, .100, .106, .4]
    assert match_boundaries([0., .1035, .107, .4], anchors, 10) == [0, 2, 3, 4]
    assert match_boundaries([0., .104, .4], anchors, 10) == [0, 3, 4]
    with pytest.raises(ValueError, match="within 1 ms"):
        match_boundaries([0., .1035, .107, .4], anchors, 1)


def test_word_ownership_uses_nearest_phone_edge_regardless_of_distance(annotation, profiles):
    annotation.provenance["boundary_tolerance_ms"] = 0
    original = annotation.tiers["phones"].copy()
    expected = derive_syllables(annotation, profiles)
    for layer in ("words", "texts"):
        annotation.tiers[layer] = [Interval(0, .62, "first"), Interval(.62, 1., "second")]
    assert derive_syllables(annotation, profiles) == expected
    assert not validate(annotation, profiles=profiles)
    aligned = align_tiers(annotation, expected)
    assert aligned.tiers["words"][0].end == aligned.tiers["texts"][0].end == .7
    assert aligned.tiers["phones"] == original
    assert phone_ownership([0., .375, 1.], [0., .25, .5, 1.]) == [0, 1, 3]


def test_frame_collisions_respect_local_budget_and_are_idempotent():
    assert covered_frames(.575, .01) == 58
    layers = {"phones": [0., .101, .104, .2], "words": [0., .104, .2]}
    frames = frame_coordinates(layers, .01, 20, 10)
    assert [frames[t] for t in layers["phones"]] == [0., .1, .11, .2]
    projected = {k: [frames[t] for t in times] for k, times in layers.items()}
    assert all(t == u for t, u in frame_coordinates(projected, .01, 20, 10).items())
    with pytest.raises(ValueError, match="phones"):
        frame_coordinates(layers, .01, 20, 0)
