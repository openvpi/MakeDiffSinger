# Prepare clips and transcripts

## Mono audio

Before slicing, ASR, or annotation, check channels with `python scripts/prepare/prepare_mono.py INPUT`. Read the per-region sample ranges and channel decisions. For stereo, write working audio with `--output` and direct `--processing-notes` to `05-review-and-decisions/processing-notes.md` for applied exceptions. The tool requires five continuous seconds of channel difference by default and prepares qualifying regions independently. Adjust `--min-difference-seconds` when the task calls for a different minimum. Shared regions use the average or keep the left channel when the average has severe cancellation. See [commands and diagnostics](../../../../docs/workflow.md#prepare-mono-audio).

Keep source files intact. Use the prepared audio tree for downstream stages. Add `--copy-transcripts` only when those paired sidecars are trusted inputs. For split outputs, establish lyrics separately and verify each song identity if needed. A shared source transcript stays unassigned until its channel correspondence is verified. Channel splitting preserves the existing subset and supplies no new speaker identity. For sources with more than two channels, obtain an explicit channel selection or a mono source.

If the stereo average loses audible material, inspect the mix-loss report and compare the affected section. Lower `--cancellation-db` to make first-channel fallback more sensitive, or raise it if the average sounds intact and the fallback is too eager. The default is 12 dB of loss in an active window.

If a reported difference starts or ends mid-phrase, use [`extract_channel.py`](../../../../docs/workflow.md#extract-an-exact-stereo-range) to compare short left, right, or average probes around the boundary. The reported ranges come from 100 ms windows. Use the confirmed source sample bounds for a local refinement and record the applied ranges and channels in the [decision log](review-and-deliver.md#record-agent-decisions). Continue treating other regions according to their own evidence.

## Paired clips

See the [paired input rules](../../../../docs/workflow.md#paired-input) for same-basename sidecars and working PFML. For maintained clips, use plain UTF-8 lyrics beside the audio and keep generated PFML in `.work/paired-lyrics`. For reused material, check [which source fields are in scope](source-fields.md). When the user is supplying paired lyrics, report missing inputs and ask how to handle them if that decision is still open. When generating lyrics, [send available candidates to FA](match-lyrics.md#hand-candidates-to-forced-alignment) and continue with all usable clips.

Measure clip durations. Recommend 5–15 seconds, with some shorter clips acceptable. Flag unusually long clips, especially above 30 seconds, with affected counts and any processing implications. If the user chooses to continue, proceed on the supplied cuts. Recut or discard only within the task's scope.

Run `scripts/preflight.py` when the task needs a separate forced-alignment input check. The [full annotation command](reannotate.md#new-annotation) includes preflight.

## Long recordings

Map recording IDs to their audio. Keep any permitted supplied lyrics associated with the recording. When the request is to recut and reuse only lyrics, plan from audio without importing source cut points or phone timing.

Run `scripts/prepare/plan_slices.py` and `scripts/prepare/replay_slices.py` per recording. Keep the plan and generated source-sample coordinates with the working cuts. See the [slice commands](../../../../docs/workflow.md#slice-a-recording).

When creating new clips for an annotation delivery, replay the accepted ranges into the maintained `01-clips` tree. Apply the accepted lyrics there before annotation. Keep trial cuts in scratch space. The training export uses the maintained pairs and annotations as its source.

Export the [Audition slice markers](../../../../docs/workflow.md#archive-slice-markers) as tab-separated `.csv` files in the same directories as the corresponding clips. Verify their recording links before handoff.

Use the recording's original filename stem for the plan. When one recording yields several new clips, retain the stem and append the generated slice suffix. Keep the source's relative directories when preparing multiple recordings.

### Adjust slice length and boundaries

See the [duration defaults and planner rules](../../../../docs/workflow.md#slice-a-recording) before changing the slice plan.

| User preference or symptom | What to try on the slice plan |
| --- | --- |
| Shorter clips overall | Lower `--target-seconds`. If the long tail persists, lower `--maximum-target-seconds` as well. Lower `--minimum-seconds` only if shorter clips are acceptable. |
| Cuts are too fragmented | Raise `--minimum-seconds` if optional cuts produce too many short clips. Raise `--target-seconds` when typical clips should be longer. If the maximum is too low for the requested lengths, relax `--maximum-target-seconds` within the user's limits. |
| Quiet vocals are treated as pauses, or actual gaps fail to yield candidates | Inspect the affected RMS valleys. `--threshold-db` sets the main silence cutoff and `--weak-threshold-db` sets the higher cutoff for additional single-cut candidates. Lower the relevant cutoff for quiet vocals classified as silence. Raise it only for verified quiet gaps that lack a candidate. |
| Keep supplied clips as they are | Use `--keep-whole` for an input that should be emitted as one unchanged range. |

Duration preferences do not override mandatory cuts. When a detected pause is left inside a clip, inspect the split's effect on clip lengths. For a few unwanted boundaries after matching, use [local cut refinement](refine-cuts.md) when the clips share a source boundary and cut changes are in scope.

Compare the new plan with the original using clip counts, the duration distribution, and boundary evidence on affected clips. Check any explicit maximum on the final ranges. Replay the chosen plan to the working output. Changed clip ranges need fresh ASR and lyric matching for downstream use.

Deliver cuts if that is the requested result. To prepare clip transcripts, use supplied pairs verified for those cuts, [match a whole-song reference](match-lyrics.md), or [find the original lyrics](find-lyrics.md) if lookup is in scope.
