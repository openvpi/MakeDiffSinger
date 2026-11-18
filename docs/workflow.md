# Step-by-step workflow

Run the Python scripts from the checkout with the chosen environment activated, or invoke that environment's interpreter directly. Each script accepts explicit paths. Outputs are separate from source recordings. Sample filters match IDs or subset/ID keys; subset paths use forward slashes.

## Delivery layout

For a shared delivery destination, use these stage directories. Create only the ones needed for the request. Use the requested destination when a format has its own path. The stage numbers apply to directories at the delivery root. Preserve the existing relative dataset layout and sample names inside them.

Each input directory supplied by the user gets its own numbered delivery. For multiple paired datasets, `python scripts/run_pipeline.py input_a input_b --output deliveries --language zh --resources configs/resources.local.json` reuses the inference workers and writes `deliveries/input_a/` and `deliveries/input_b/`. For each delivery, run quality-tail selection and compilation on its own `03-annotations/`. Each has its own ranking denominator and cutoff even when the requested fraction is the same.

~~~text
delivery/
  00-recordings/      requested copies of original recordings
  01-clips/           cut audio with source-recording marker files beside the clips
  02-lyrics/          song lyrics or editable clip-to-lyric tables
  03-annotations/     final editable TextGrids
  04-quality/         per-asset metrics and native alignment statistics
  05-review-and-decisions/  Agent decisions, unresolved samples, quality exclusions and requested review pairs
  06-diffsinger/      requested training CSV and optional matching WAV files
  .metadata/          shared recording, clip, language, and editable asset tables
  .work/              intermediate alignment and pronunciation files
~~~

For a task that prepares or annotates data and compiles training data, deliver the maintained clip/lyric pairs and final four-tier TextGrids with their companion state alongside the training derivative. Write task-generated pairs under `01-clips` before annotation, with transcripts beside their audio and [accepted cut markers](#archive-slice-markers) linked to the original tracks. Link existing pairs directly if they already have a durable location. Use `02-lyrics` for a separate matching table when needed. Keep the three-tier FA TextGrids and aligned PFML in `.work/fa`. Apply training exclusions only to the derivative. Use `.work/` for trials, ASR, and run logs. The scripts group companion state under `.metadata/` for resuming, editing, and auditing an artifact.

### Processing notes

Keep blocking issues and consequential judgment calls under `05-review-and-decisions/`. Scripts return routine progress and counts on the console. Use `--processing-notes` for a requested log of processing exceptions.

## Prepare mono audio

Slicing, ASR, alignment, note extraction, and audio export require mono input. Check the channel layout before those stages. Run the first command to inspect only. Run the second to produce a mono working tree, with `--copy-transcripts` when the paired sidecars are authorized inputs.

~~~bash
python scripts/prepare/prepare_mono.py source_audio
~~~

~~~bash
python scripts/prepare/prepare_mono.py source_audio --output work/mono --copy-transcripts --processing-notes outputs/05-review-and-decisions/processing-notes.md
~~~

The tool compares left and right over 100 ms windows and reports contiguous regions in original sample coordinates. Channel separation requires at least 5 continuous seconds of distinct content by default. Use `--min-difference-seconds` to change that minimum. A qualifying region with different material on both channels produces `_L` and `_R` mono samples. Within a qualifying region, a stretch with activity on only one side produces just that channel. Shared regions use the average or keep the left channel when the mix loss reaches 12 dB in an active window. Adjust `--cancellation-db` for that loss threshold. A null worst mix loss means complete cancellation.

Content comparison ignores activity more than 40 dB below the loudest channel window. For windows with activity on both channels, it checks polarity-insensitive waveform similarity within a 20 ms lag and spectral similarity. Quiet and shared windows break a run of differences. Quiet windows can be attached to neighboring output after the duration check. Isolated shorter differences go through the shared-region mix and cancellation check.

Working copies keep relative directories and sample rates. WAV and FLAC keep their extensions. Stereo inputs in other formats become WAV. Regional outputs retain the original stem with suffixes such as `_r001`, `_r002_L`, and `_r002_R`. A whole-recording split uses `_L` and `_R` alone. Name collisions receive a further suffix. The report maps source ranges to output files, and companion metadata preserves the original bounds and channel. A split leaves the source-level transcript unassigned, even with `--copy-transcripts`. Establish lyrics for each output from its audio and source context. Use an explicit channel selection or a mono source for files with more than two channels.

### Extract an exact stereo range

Use the inspection report to locate a region, then extract one left, right, or average probe without preparing the rest of the recording. `START` is inclusive and `END` is exclusive. Use a new WAV or FLAC path for each probe or selected output.

~~~bash
python scripts/prepare/extract_channel.py song.wav --range-samples START END --channel right --output work/right-probe.wav
~~~

## Paired input

Arrange flat samples within each leaf subset. Files paired directly with working clips use the priority PFML, TXT, then LAB. In a numbered delivery, `apply_lyrics.py` writes plain lyrics beside the clips in `01-clips` and keeps matching PFML in `.work/paired-lyrics` with the same relative paths. The aligner reuses the working PFML when its written text matches the current plain transcript. Editing the plain text makes it the input for the next alignment.

Language and an explicitly supplied speaker label apply to the call. Speaker identity is unset by default. Sample IDs use the original filename stems, and the output preserves relative input directories. If an authorized layout or format change causes a collision, add a unique suffix to the conflicting outputs and record the mapping.

See [existing labels and format conversion](#existing-labels-and-format-conversion) for imported inputs.

## Preflight

The [preflight command](../README.md#start-with-paired-clips) checks audio, the selected transcript, G2P, vocabulary coverage, and shared resources. Sample failures are retained as problem records. Missing models or shared dictionaries fail the stage.

Exit code 0 means the requested operation succeeded, 2 means sample-level problems remain, and 1 means the operation could not complete. Inspect the JSON summary and problem records rather than treating every nonzero exit as the same condition.

## Forced alignment and diagnosis

Run the [forced alignment command](../README.md#run-forced-alignment), then inspect the resulting FA assets if needed:

~~~bash
python scripts/report.py outputs/.work/fa
python scripts/check.py outputs/.work/fa --audio
~~~

The FA stage writes three named layers and per-item metadata. It performs G2P pronunciation selection, FA, selected VED insertion, and second alignment. Zero-width skipped phones are omitted from editable intervals and retained in diagnostics. Scores, including negative confidence values, are preserved.

To compare a proposed lyric or pronunciation with an existing FA, pass `--candidate FILE` to `align.py` for the same clip. Supply the original alignment settings and a separate working output directory. The candidate can be TXT, LAB or PFML.

Edit the TextGrid if needed. Readers detect layer edits and retain the original model scores, marked stale when their input layers change.

## Notes

Use the [notes command](../README.md#annotate-notes) with FA output, then [check the finished assets](../README.md#check-quality-and-select-samples). The [shared syllable rules](annotations.md#shared-syllable-rules) derive syllable boundaries for the note extractor from pronunciation groups and phone classes.

### Full pipeline

Run preflight, FA, notes, and final validation in one persistent session.

~~~bash
python scripts/run_pipeline.py clips --language zh --resources configs/resources.local.json --output outputs/run
~~~

The command follows the [delivery layout](#delivery-layout). Aligned PFML stays in `.work/fa` by default. Add `--pfml` to deliver it beside the final TextGrids. `--chunk-size` controls how many clips are processed before saving. Progress goes to stderr and the final counts to stdout. Native statistics go to `04-quality`; `--report-output` selects another destination for that bundle.

### Notes without syllable boundaries

Use this command to extract independent notes from audio without syllable constraints.

~~~bash
python scripts/extract_notes.py clips --resources configs/resources.local.json --output outputs/03-annotations
~~~

The note extractor uses the whole clip as one input region. Audio is sufficient input. The command writes a notes tier and metadata with an unconstrained-input marker. The result has no phoneme alignment or syllable relationships. Use a separate destination for these assets.

## Partial reannotation

The [partial reannotation command](../README.md#reannotate-selected-layers) replaces only selected samples and requested layers. Use `--layers fa` for the three coupled FA layers, `--layers all` for full reannotation, or pass explicit layer names. A partial merge must satisfy the retained layers' boundaries. If FA changes while notes are retained, those notes become stale until updated.

Edited layers are protected unless `--replace-edited` is supplied. A conflicting candidate and its reason are retained in a problem record.

## Recovery

For G2P failures, follow the [transcript repair guide](../.agents/skills/dataset-creator/references/pronunciation.md). After correcting the transcript or resource issue, use the [retry command](../README.md#retry-failed-items) with `--sample` to select affected items.

Use stage notes when prior FA assets or saved FA candidates are available, or fa to stop after alignment. Retry reads the current transcript using the [paired input rules](#paired-input). Successful samples outside the retry selection are untouched.

The [full pipeline](#full-pipeline) saves each completed chunk. Resume by selecting failed or missing samples or by running `annotate_notes.py` on retained FA. Confirm source, transcript, and resource compatibility before reusing cached work.

## Select and compile

Use the [export command](../README.md#deliver-annotations-or-export-diffsinger-data) to compile saved annotations to a separate destination.

Filters include sample, speaker, subset, language, and status. Add `--drop-bottom-fraction FRACTION` to exclude the worst fraction of rankable samples by the [annotation quality ranking](#quality-overview). Add `--percentile-group speaker`, `subset`, or `language` to rank each group separately. Unranked samples remain subject to ordinary annotation checks. The same ranking options apply when adding notes, reannotating, checking, or reporting on existing annotations. A filtered export prints the eligible population, group counts and excluded IDs with reasons. Use the subset CSVs to look up retained rows.

See the [DiffSinger compilation contract](annotations.md#diffsinger-compilation) for required CSV fields and subset layout. Use `--speaker-map FILE` to override existing labels at export. Supply a JSON object mapping stored sample keys to speaker labels, with `null` for unassigned items. Use `--no-audio` to write the same CSV fields without copying WAVs.

## Quality overview

The [quality report command](../README.md#check-quality-and-select-samples) summarizes alignment metrics and their distributions, with missing, invalid, and stale diagnostics counted separately.

With `--output`, the command uses the installed inference environment and the upstream TIFA statistics writer. It publishes `diagnosis.json`, `scores.json`, and a histogram and frame-count scatter plot for each of agreement, confidence, determinacy and monotonicity. Complete current measurements are required. If the same population already has TIFA `--stat` outputs, copy its `statistics/` contents directly. The terminal report counts incomplete or stale measurements. Use `--resources` or `--tifa-repo` to select the checkout.

The delivered diagnosis uses the upstream writer's order. Quality-tail selection ranks samples by their worst rank across five metrics. Ties compare the ranks for skipped-phone count, agreement, confidence, determinacy and monotonicity in sequence. More skipped phones and lower alignment scores rank worse. Equal metric values retain input order.

### Inspect and preview quality-tail exclusion

`quality_tail.py` prints JSON with eligible counts, unavailable diagnoses by reason, and the minimum, median, and maximum of each diagnostic by rank band. Use `--bands` to change display resolution. Supply the same sample filters and `--percentile-group` as the intended exclusion.

~~~bash
python scripts/quality_tail.py outputs/03-annotations
python scripts/quality_tail.py outputs/03-annotations --percentile-group subset --preview-fraction FRACTION
~~~

Preview reports the ranked exclusions, unranked samples retained by the tail rule, retained counts, and boundary samples. It creates no output files. To apply the fraction, pass `--drop-bottom-fraction FRACTION` to the relevant operation with the same scope and grouping. Each group excludes `floor(eligible_count * fraction)` ranked samples. An explicit zero clears prior ranked exclusions in the current scope.

Use repeatable `--exclude-quality KEY REASON` for individual completed annotations with a known defect. These exact keys are excluded before ranking. When a quality filter is applied, each excluded clip's WAV and TextGrid are delivered with its reason in `05-review-and-decisions/quality-review.md`, including for CSV-only training exports. Use `--review-output DIR` for a separate review destination. Unresolved clips and their available evidence remain in `review.md`.

## Manual review selection

Supply exact IDs or patterns, or select the worst fraction over a scoped population. Replace FRACTION with the chosen value.

~~~bash
python scripts/review_pack.py outputs/03-annotations --sample "singer_a/control/clip_001" --output outputs/05-review-and-decisions
python scripts/review_pack.py outputs/03-annotations --speaker singer_a --bottom-fraction FRACTION --output outputs/05-review-and-decisions
python scripts/review_pack.py outputs/03-annotations --bottom-fraction FRACTION --group-by subset --output outputs/05-review-and-decisions
~~~

Fractions use the same quality ranking as training exclusion. Ranking runs within the selected pool or separately for each `--group-by` group. Each group selects `floor(eligible_count * fraction)` samples, starting with the worst. The review list records the selected IDs and rule.

The selected pack contains a WAV/TextGrid pair for each requested item. For unresolved annotation problems the full pipeline delivers a separate pending list, listening links, reasons, and any available attempted lyrics or candidate TextGrids in stage 05. To synchronize those files after a custom run, use:

~~~bash
python scripts/review_pending.py outputs/03-annotations --output outputs/05-review-and-decisions
~~~

## Existing labels and format conversion

~~~bash
python scripts/prepare/inspect_sources.py source_dataset --output work/inventory.json
python scripts/import_annotations.py source_labels --audio-root source_audio --language zh --groups pronunciation --time-resolution 0.01 --output work/imported
python scripts/export_diffsinger.py work/imported --resources configs/resources.local.json --output outputs/06-diffsinger
~~~

Declare groups pronunciation only after verifying that the source words tier means pronunciation groups. An explicit tier-map option maps source names to canonical names; it does not prove their semantics. Existing DiffSinger CSV can be imported directly and retains its ph_num as declared syllable grouping without inventing FA words or texts layers.

For other formats, use a source adapter calling `import_intervals`. Import can preserve partial labels. Compile only once the source or authorized annotation supplies the [required fields](annotations.md#diffsinger-compilation).

## Long recordings and lyric preparation

### Slice a recording

~~~bash
python scripts/prepare/plan_slices.py song.wav --recording-id song --output work/song-slices.json
python scripts/prepare/replay_slices.py work/song-slices.json --output outputs/01-clips/song
~~~

The duration preferences default to `--minimum-seconds 5 --target-seconds 10 --maximum-target-seconds 15`. Keep `0 < minimum <= target <= maximum`. The maximum also governs fallback candidate generation. The target breaks ties after duration penalties and cut quality.

The planner finds pauses on GAME's RMS grid and searches up to one second inward from each pause edge for its lowest RMS point. The windows may overlap. Both searches break RMS ties by distance to the pause midpoint, then by earlier time. When the selected points coincide, the clips share a boundary. Otherwise the planner can skip the quiet gap between them. At the start or end of a recording, it uses the search on the side adjacent to the audio being kept.

The second RMS threshold defaults to 10 dB above the silence threshold, or −30 dBFS with the default −40 dBFS silence threshold. It provides additional single-cut candidates. Set `--weak-threshold-db` to change it. Long stretches with insufficient candidates can use local RMS troughs as fallback cuts.

An internal pause of at least four seconds below the silence threshold always splits the recording at its fixed cut points, even if this produces a short clip. The one-second search window and four-second split threshold are fixed in the CLI. The dynamic program selects the remaining cuts by minimizing the sum of the time each clip falls short of the preferred minimum or exceeds the preferred maximum. An unselected pause counts toward the duration of the clip that contains it. Ties are resolved in order by fewer fallback cuts, fewer second-threshold cuts, lower RMS at those supplemental cuts, and smaller total distance from the target duration. Python callers can supply trusted phone intervals as diagnostic evidence when timing reuse is in scope. Replay the saved sample ranges directly, or edit the plan as needed. The keep-whole option preserves an existing clip.

The planner uses the source filename stem by default. It appends a slice suffix only when creating multiple clips from the recording.

### Archive slice markers

Write one Audition marker file per original recording, using the recording's filename stem. Save it with a `.csv` extension so Audition recognizes it on import. Keep the fields separated by tabs to match the Audition marker format.

~~~bash
python scripts/prepare/export_slice_markers.py work/song-slices.json --output outputs/01-clips/song/song.csv
~~~

This places `song.csv` beside the WAV clips. `Name` identifies the clip. `Start`, `Duration`, and `Time Format` identify the sample range and rate on the original recording's grid, and `Type` is `Cue`. `Description` identifies the original audio relative to its dataset root and includes the source channel when applicable. An explicitly linked delivery copy can instead be referenced relative to the marker file.

Use `--recording` to link to a kept copy with the same frame count and sample rate.

### Pair lyrics with the cuts

For whole-song lyrics, use [clip lyric matching](lyric-matching.md). To supply your own clip transcripts using a table, create the table, fill the lyric rows, then apply it.

~~~bash
python scripts/prepare/prepare_lyrics.py outputs/01-clips --output outputs/02-lyrics/lyrics.csv
~~~

~~~bash
python scripts/prepare/apply_lyrics.py outputs/02-lyrics/lyrics.csv --audio-root outputs/01-clips
~~~
