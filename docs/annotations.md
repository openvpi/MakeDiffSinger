# Annotation and persistence contract

## Layers and identities

Complete assets have these named interval layers in order:

| Layer | Meaning |
| --- | --- |
| texts | Aligned lyric text |
| words | Pronunciation groups from forced alignment |
| phones | Phonemes |
| notes | ASCII note names with cents, or `rest` |

Pronunciation groups and note-extraction syllables can have different boundaries. Source adapters must identify a foreign layer's meaning before mapping it. Imported assets may be partial, and their provenance identifies the actual source rather than claiming model generation.

Sample IDs are flat filename stems. Subset paths are stored separately and may be hierarchical. A sample key is subset/ID, or ID for an ungrouped sample. Keys must be unique within a batch.

## Shared syllable rules

Within each pronunciation group, longest-match scanning selects the vowel in V, the medial in C M V, and the second medial in M M V. A group without any match uses its first phone. Phones before the first onset in the complete sequence form a separate prefix syllable.

AP/SP provide onset positions. They do not create a pitch or rest override. Adjacent AP intervals remain separate. English l, r, _r, y, and w are medials; the profile is bound to English and its dictionary. Japanese cl is a consonant.

All phone classes come from explicit records, including two-part dictionaries. The resource catalog binds dictionary names and languages to those records and declares global event classes. Runtime dictionary inspection checks inventory coverage only; it does not assign types by position or apply language-specific classification exceptions.

Each syllable continues to the next onset. Multiple notes inside it form a slur. Its leading note has note_slur 0 and later notes have 1. Pitched prefixes and pitched event positions are not rejected by an additional MakeDiffSinger rule.

## Time

Source slices use original-rate sample indices [start_sample, end_sample). Resampling is a model-input operation and does not replace source recordings.

The forced aligner's configuration determines alignment frame times. The note extractor uses integer frame counts to avoid accumulating float32 duration drift. The annotation also retains raw model durations, presence, and MIDI scores. Integer cents in editable note labels do not replace the raw numeric pitch record.

Hand-edited TextGrids can use arbitrary timestamps. The cross-tier tolerance defaults to 10 ms. Each pronunciation-group boundary matches a distinct phone boundary in order, minimizing total distance among matches within tolerance. The same rule matches syllables to notes. Each tier joins consecutive intervals within 1 ms. Audio duration differences above 10 ms produce a warning with both durations. Imports can override these thresholds with integer `boundary_tolerance_ms` and `audio_warning_ms`.

Constrained note extraction uses the labelled duration covered by the WAV, rounded to the nearest model frame. It projects cumulative boundaries once and reserves a positive interval on each tier. Shared boundaries retain a common frame. If a cluster cannot fit within the available local frames, the item diagnostic identifies the timing conflict. Only the input buffer receives padding for frame rounding and feature extraction. The saved frame clock uses the model's exact hop size and sample rate, including fractional-millisecond steps. Source audio and slicing markers retain their original sample coordinates.

## Files

Each asset has an editable `ID.TextGrid` in its subset directory. One delivery-root `.metadata/` sits beside `.work/`. Its `recordings.json` and `sources.json` map clips to original recordings, sample ranges and channels. Its `annotations.json` associates each TextGrid path with its sample, languages, speaker, grouping and edit state. Readers also recognize older per-item JSON records.

### Source paths

One delivery belongs to one input dataset. Original recording keys and their `path` values are relative to the original directory supplied by the user, for example `disc-1/take.wav`. In fields that can reference either tree, `dataset:disc-1/take.wav` identifies an original input and `01-clips/song/clip.wav` identifies a file under the delivery root. Paths use forward slashes. Legacy absolute references remain readable.

The standard pipeline binds the input automatically. Before processing separate branches of a dataset with an adapter, call `bind_dataset(delivery, source_directory)` from `makediffsinger.catalog` or run `python scripts/prepare/bind_dataset.py original_dataset --output delivery`. Pass `--dataset-root original_dataset` when scanning only a source subdirectory. The catalog stores one locator relative to the delivery when the filesystem supports it. Across Windows volumes, the location is recorded in the checkout's ignored `local/dataset-roots.json`. An auxiliary file on a third volume can use a `local-file:` reference in that registry. After independently moving either tree, use the binding command with `--relocate` to reconnect the same dataset.

Saved grouping records retain the method, verified grouping state, and imported phone counts and signatures when applicable.

Explicit unconstrained extraction produces a separate notes-only asset. Its `syllable_input` method is `unconstrained`, and its domain follows the audio duration rounded to the note extractor's frame grid.

A blocking issue is indexed by asset path in `05-review-and-decisions/issues.json` with the input, stage, and reason. It can retain a candidate for retry. A successful save clears the corresponding issue.

Per-sample measurements are in `04-quality/metrics.json`. The alignment archive and final annotation share a measurement when their values agree. A different final measurement is stored separately in the same row.

Optional JSON checkpoints include all layers. Persist in-memory audio explicitly before writing a standalone checkpoint. Saving a normal asset can materialize a buffer to a WAV under `.work/audio`.

## Edits and quality

Small layer signatures detect edits after TextGrid round trips. A loaded asset also carries a concurrency guard so a later write cannot silently overwrite intervening edits.

Phone/group edits invalidate related notes and the applicability of original quality scores. A notes update projects the supplied labels to the output frame grid. Other samples retain their original assets.

Quality metrics are model diagnostics, not calibrated probabilities. Negative finite values are valid. Padding columns in pronunciation score tensors are omitted; non-finite candidate scores use explicit strings such as -inf instead of pretending to be zero.

Imported `ph_num` follows the ordered phone labels across timing edits. After sequence changes, derive new counts from verified pronunciation groups or supply a new mapping. Older assets with only a timing-sensitive signature can establish a label baseline while the signature still matches.

## DiffSinger compilation

Every training CSV has `name`, `ph_seq`, `ph_dur`, `ph_num`, `note_seq`, `note_dur`, and `note_slur`. Compilation requires usable phone timings, syllable grouping, and note labels. Durations use full floating-point text precision.

Each subset contains `transcriptions.csv` and, for an audio export, `wavs/`. Subsets are named for the primary language, with `und` for annotations lacking language metadata. With multiple identified speakers, all subsets use `SPEAKER/LANGUAGE/`, including `unknown/LANGUAGE/` for unassigned speakers. The export records any filename suffixes needed to resolve collisions after regrouping.

Each subset CSV identifies its training rows and the matching WAV stems. Names containing DiffSinger's reserved # or : need an explicit ID mapping.

The export uses the configured note model's grid or the clock saved on the annotation. A standalone conversion with neither uses GAME's default frame clock (441 samples at 44.1 kHz, or 10 ms). Uncompilable items, including all-rest pitch rows, appear in the export's failed list. The remaining training rows are written and the CLI returns partial status.
