# Annotation and reannotation

Choose the smallest processing scope that fulfills the request.

## New annotation

Start with [paired clips](prepare-inputs.md#paired-clips) or Sample objects. Record which fields will be generated. When using existing labels, verify the [source fields in scope](source-fields.md).

Run `scripts/run_pipeline.py` for [full annotation](../../../../docs/workflow.md#full-pipeline) or `scripts/align.py` for [forced alignment](../../../../docs/workflow.md#forced-alignment-and-diagnosis) only. Check a representative pilot before scaling. Use [failure handling](setup-and-recovery.md#classify-failures-before-retrying) for failed items or broad anomalies. Before handoff, [validate the requested layers](review-and-deliver.md#validate-outputs) and [check the alignment statistics](review-and-deliver.md#quality-overview).

## Notes inputs

- **With verified syllable boundaries.** Use `scripts/annotate_notes.py` with annotations that provide verified pronunciation groups and phones, or declared syllable counts and phone durations. Verify that these fields may be reused. When deriving syllables from phones, require [explicit phone-type records](../../../../docs/installation.md#resource-file) covering every selected dictionary, including two-part dictionaries. Validate the resulting note boundaries against the supplied syllables.
- **Without usable syllable boundaries.** Offer deriving them by authorized alignment, or extract notes independently with `scripts/extract_notes.py` once the user accepts that phoneme alignment cannot be guaranteed. Write separate note-only assets.

If a constrained run fails, use [failure handling](setup-and-recovery.md#classify-failures-before-retrying).

## Adjust alignment and note inference

| User preference or symptom | Option or input to adjust |
| --- | --- |
| The forced aligner skips phones from a verified sung word | Check the clip, text, and pronunciation first. On a small pilot, raising `--skip-penalty` makes a zero-frame phone costlier. Check for forced alignments to unsung material. `--score-unit` controls pronunciation scoring templates and should stay fixed when comparing skip penalties. |
| Automatic breath or event insertion is unsuitable | If VED is configured, use `--ved-events` to select its comma-separated event labels, or `--no-ved` to disable it for the run. Check the resulting labels and timings. |
| Notes split too often or neighboring notes are missed | Check the supplied syllable boundaries first. Lower `--seg-threshold` or use a wider `--seg-radius` in seconds to test fewer model-predicted boundaries. Try the opposite for missed boundaries. Change one option and check both split and merged notes. |
| Too many rests or notes reported in rests | Lower `--est-threshold` to accept weaker note-presence scores, or raise it to suppress weak detections. Compare the suspected regions with the audio. |

Keep `--oov-handling raise` when diagnosing lexicon problems. Use the [G2P guide](pronunciation.md) to check the written lyrics before supplying a pronunciation override. Leave `--game-language` unset for standard note extraction, even when the singing language is known. The option is available for explicitly requested GAME language conditioning. `--steps`, `--t0`, and `--timesteps` control GAME's sampling schedule. Use a fixed runtime and schedule when evaluating a decoder adjustment.

## Partial annotation

Use `scripts/reannotate.py` for the standard partial update. Use `plan_reannotation` and `Pipeline.reannotate` for a task-local adapter that needs in-memory composition. The plan selects samples and layers and records the expected current layers.

Use `--sample` with explicit IDs or quoted patterns to limit the update and `--layers` to name the layers or use `fa` or `all` when those scopes are requested. For `retry.py`, use `--stage fa`, `--stage notes`, or `--stage all` to limit what is rerun on selected problem records.

| Request | Processing and preservation |
| --- | --- |
| Regenerate only notes | Reuse existing verified groups and phones or declared syllable counts. Run the note extractor and preserve the first three layers. |
| Retry failed samples | Select problem records; refresh their selected input transcripts. Leave successful samples outside the selection untouched. |
| Correct lyrics and redo FA | Retain the previous input and reason. Recompute the requested FA layers; identify related notes as stale unless their update is also requested. |
| Replace only one FA layer | Compute the necessary FA context, then merge only the requested layer. A boundary conflict requires a wider authorized update or manual resolution. |
| Repair a time interval | Check whether a candidate can be merged within the requested boundaries. The current core applies whole selected layers; prepare an explicitly reviewed local candidate rather than pretending the worker performs arbitrary exact window replacement. |

Separate the extent of computation from the permitted extent of replacement. Whole-clip model inference does not authorize overwriting unrelated layers or samples.

Human edits are protected by default. An authorized in-place update uses `--overwrite`. Use `--replace-edited` only when replacing human-edited layers is also authorized. Otherwise retain the current asset and surface the candidate or conflict. Source changes, layer edits, and reruns must not leave an old diagnostic score labeled as a score for the new version.

When a task generates full annotations, follow the [FA archive requirement](review-and-deliver.md#deliver). For a partial update, save scratch checkpoints as needed for inspection or recovery.
