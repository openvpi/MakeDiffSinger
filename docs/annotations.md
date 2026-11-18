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

Audio endpoint checking permits the recorded annotation time resolution plus one source sample. Layer coverage and syllable matching use a separate 20-microsecond numerical tolerance. Off-grid model constraints are reported, not repaired by shifting exported note endpoints.

If the declared alignment domain needs slightly more audio than the source supplies, only the model input may receive zero padding. The amount is bounded by the recorded resolution, a source sample, and the feature extractor's required context; it is recorded per sample. A larger mismatch is rejected. Source audio and annotation boundaries remain unchanged. TextGrid serialization may canonicalize only a final endpoint within 1e-10 seconds of the domain to satisfy exact floating-point equality.

## Files

Each asset has an editable `ID.TextGrid` in its subset directory. One delivery-root `.metadata/` sits beside `.work/`. Its `recordings.json` and `sources.json` map clips to original recordings, sample ranges and channels. Its `annotations.json` associates each TextGrid path with its sample, languages, speaker, grouping and edit state. Readers also recognize older per-item JSON records.

### Source paths

One delivery belongs to one input dataset. Original recording keys and their `path` values are relative to the original directory supplied by the user, for example `disc-1/take.wav`. In fields that can reference either tree, `dataset:disc-1/take.wav` identifies an original input and `01-clips/song/clip.wav` identifies a file under the delivery root. Paths use forward slashes. Legacy absolute references remain readable.

The standard pipeline binds the input automatically. Before processing separate branches of a dataset with an adapter, call `bind_dataset(delivery, source_directory)` from `makediffsinger.catalog` or run `python scripts/prepare/bind_dataset.py original_dataset --output delivery`. Pass `--dataset-root original_dataset` when scanning only a source subdirectory. The catalog stores one locator relative to the delivery when the filesystem supports it. Across Windows volumes, the location is recorded in the checkout's ignored `local/dataset-roots.json`. An auxiliary file on a third volume can use a `local-file:` reference in that registry. After independently moving either tree, use the binding command with `--relocate` to reconnect the same dataset.

Saved grouping records retain the method, verified grouping state, and imported phone counts and signatures when applicable.

Explicit unconstrained extraction produces a separate notes-only asset. Its `syllable_input` method is `unconstrained`, and its domain follows the note extractor's frame duration. Source-audio endpoint checking uses the recorded frame resolution.

A blocking issue is indexed by asset path in `05-review-and-decisions/issues.json` with the input, stage, and reason. It can retain a candidate for retry. A successful save clears the corresponding issue.

Per-sample measurements are in `04-quality/metrics.json`. The alignment archive and final annotation share a measurement when their values agree. A different final measurement is stored separately in the same row.

Optional JSON checkpoints include all layers. Persist in-memory audio explicitly before writing a standalone checkpoint. Saving a normal asset can materialize a buffer to a WAV under `.work/audio`.

## Edits and quality

Small layer signatures detect edits after TextGrid round trips. A loaded asset also carries a concurrency guard so a later write cannot silently overwrite intervening edits.

Phone/group edits invalidate related notes and the applicability of original quality scores. A model rerun can compute more context than it is allowed to replace. Partial updates preserve all unselected layers and samples.

Quality metrics are model diagnostics, not calibrated probabilities. Negative finite values are valid. Padding columns in pronunciation score tensors are omitted; non-finite candidate scores use explicit strings such as -inf instead of pretending to be zero.

Do not reuse source grouping counts after editing their phones: imported ph_num is tied to the imported phone sequence. Either provide a new verified mapping or use pronunciation groups and language-bound profiles.

## DiffSinger compilation

Every training CSV has `name`, `ph_seq`, `ph_dur`, `ph_num`, `note_seq`, `note_dur`, and `note_slur`. Compilation requires usable phone timings, syllable grouping, and note labels. Durations use full floating-point text precision.

Each subset contains `transcriptions.csv` and, for an audio export, `wavs/`. Subsets are named for the primary language, with `und` for annotations lacking language metadata. With multiple identified speakers, all subsets use `SPEAKER/LANGUAGE/`, including `unknown/LANGUAGE/` for unassigned speakers. The export records any filename suffixes needed to resolve collisions after regrouping.

Each subset CSV identifies its training rows and the matching WAV stems. Names containing DiffSinger's reserved # or : need an explicit ID mapping.

The DiffSinger pitch reader rejects all-rest samples. Keep their editable annotations and exclude them from the training selection.
