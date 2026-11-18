# Python API

The API takes objects as well as paired files. Backend-specific classes are imported only inside their dedicated worker process.

~~~python
from makediffsinger.api import Pipeline, discover_pairs, save_result
from makediffsinger.config import Resources, Runtime

samples = discover_pairs("clips", language="zh", speaker="singer")
with Pipeline(Resources.read("configs/resources.local.json"),
              Runtime(device="cuda:0", batch_size=4)) as pipeline:
    result = pipeline.run(samples)
    save_result(result, "outputs/03-annotations")
~~~

The same `Pipeline` keeps the forced aligner and note extractor workers alive across calls. Each worker loads its model once. Use a context manager or close the pipeline when finished.

For a full dataset delivery, use the [numbered layout](workflow.md#delivery-layout) and the independent stages below to save FA before adding notes.

## Independent stages

- preflight(samples) returns ready and pending input outcomes.
- align(samples) returns FA/AP annotations and raw diagnoses.
- annotate_notes(annotations) runs the note extractor using shared syllable derivation.
- extract_notes(samples) produces independent note-only assets without syllable constraints. It takes audio samples without requiring a transcript or phone profile.
- reannotate(annotations, layers=..., sample_keys=..., replace_edited=...) computes and applies a scoped update.
- run(samples) combines alignment and note annotation.

BatchResult exposes outcomes, ready, annotations, and problems. Failed or pending items retain their Sample and Issue records. A missing shared resource raises ResourceError instead of pretending every lyric failed.

`Runtime` controls device, precision, batch size, worker count, and seed. Both workers use `sys.executable` from the calling process. `AlignmentOptions` controls selected VED events, OOV behavior, pronunciation scoring, and skip penalty. `NoteOptions` controls the note extractor's language, thresholds, radius, and sampling schedule.

Leave `NoteOptions.language` at `None` for standard note extraction. This selects unconditioned GAME (`language_id=0`). Explicit GAME language conditioning is independent of `Sample.language` and `extended_languages`, which control the forced aligner and phoneme interpretation.

Runtime.allow_tf32 defaults to False and controls CUDA matrix/cuDNN TF32 operations.

Use Resources.read for model and pronunciation resource references. Pass Runtime for device and inference settings.

## In-memory input and exact slices

~~~python
from makediffsinger import AudioSource, Sample, Transcript

sample = Sample(
    "clip_001",
    AudioSource(path="recording.wav", start_sample=48000, end_sample=528000,
                sample_rate=48000),
    Transcript("ordinary lyrics", format="text"),
    language="en",
)
~~~

Use AudioSource(waveform=array, sample_rate=rate) for mono arrays shaped as samples or samples-by-one-channel. They are transferred to the worker directly, without a temporary audio file. Transcript(format="pfml") uses TIFA's PFML parser; format is retained across persistence.

## Existing labels

import_annotations handles canonical TextGrid and DiffSinger CSV. For an unfamiliar format, map its records to Sample and Interval objects and call import_intervals. Use groups_verified=True only after establishing pronunciation-group semantics.

Importing CSV with ph_num preserves declared syllable grouping separately. It does not invent a texts or words layer. An AudioSource without a path is allowed for label-only conversion, but inference and audio export require actual audio.

Pass samples to `extract_notes` for unconstrained extraction. The returned assets contain only notes, have partial status, and record `syllable_input.method` as `unconstrained`. Their duration follows the note extractor's frame domain. For DiffSinger compilation, provide the phone timings and syllable grouping required by the [CSV contract](annotations.md#diffsinger-compilation).

The core validation, derive_syllables, select, and export_diffsinger functions operate on those same objects. See the [import adapter](../.agents/skills/dataset-creator/assets/import_adapter.py) for an example.

Keep custom source metadata as ordinary JSON data: string-keyed dictionaries, lists, and scalar values. Normalize library-specific containers in the source adapter before storing them in Sample.metadata.

## Quality and requested review copies

`quality_summary(annotations)` aggregates current self-check metrics and orders per-sample records by the [quality ranking](workflow.md#quality-overview). Original score/frame-count pairs and finite negative values are preserved. Missing, invalid, and stale scores are omitted from those records and counted separately in the summary.

`write_overview(report, output)` invokes the native statistics writer on current complete diagnoses and collected pronunciation scores. It returns the number of measured samples. Use an initialized TIFA checkout and the inference environment. Pass `tifa_repo` for a configured checkout.

~~~python
from makediffsinger.api import quality_summary, summarize, write_overview

report = {"assets": summarize(annotations), "quality": quality_summary(annotations)}
write_overview(report, "outputs/04-quality")
~~~

Use `select` for an explicit sample scope. `select_quality(scope, QualityRule(bottom_fraction=review_fraction))` returns the requested fraction in worst-first order. Use `group_by` to rank existing groups separately. `Selection(drop_bottom_fraction=drop_fraction, percentile_group=group)` applies the same partition for exclusion. Supply a fraction explicitly when a quality rule is requested.

make_review_pack(selected, output) accepts a Selected result and exports exactly those items.

For an applied quality filter, use `Selection.quality_exclusions` to map individual sample keys to reasons, alongside any ranked fraction. `export_diffsinger` publishes the exact removed samples for review. A custom operation applying a `Selected` result calls `makediffsinger.review.sync_quality_review(selected, review_dir)`. Statistical inspection and preview use the pure selection functions.

## Persistence and updates

write_annotations(annotation, output_root) writes an editable asset. read_annotations reads it and detects edited layers. save_checkpoint/load_checkpoint provide optional complete JSON snapshots.

plan_reannotation records selected layers and expected layer signatures. apply_annotation_updates verifies the plan, preserves other layers, and reports an invalid partial merge instead of widening its write scope.

Saving to an existing path requires overwrite=True. Replacing human edits also requires replace_edited=True. A failed update leaves the existing TextGrid intact and can save the candidate for review.

## Preparation

plan_slices and replay_slices separate cut planning from source PCM replay. SlicePlan.write/read expose an editable JSON recipe. prepare_lyrics and apply_lyrics provide an optional CSV editing step before the paired-input workflow.

`makediffsinger.lyric_asr.transcribe_clips` loads Qwen3-ASR from the supplied local model directory and returns an independent ASR report keyed by `Sample.key`. `read_hypotheses` loads that report or a separate LAB/TXT directory. `makediffsinger.lyric_match.match_lyrics` takes samples from one song, reference text, ASR evidence, and a `Pipeline`. See the [matching guide](lyric-matching.md#interpret-the-evidence) for search, scoring, and diagnostic options. Pass `verify=False` to run the phone search alone. Each scored match provides selected `texts` and a `transcript` object with `content` and `format`, suitable for constructing a `Transcript`. Pass the match or ASR report's `clips` to `makediffsinger.lyric_handoff.write_candidate_table(samples, records, output)` and apply the table with `apply_lyrics`. A custom adapter can use `alignment_inputs(samples, delivery)` from the same module to recover generated candidates for missing working pairs before calling `Pipeline.align`.
