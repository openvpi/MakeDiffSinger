---
name: dataset-creator
description: Inspect, convert, create, or selectively reannotate singing datasets with MakeDiffSinger. Use for unfamiliar source layouts, song identification and lyric discovery, clip-to-song lyric matching, existing annotations, and DiffSinger data exports.
---

# Dataset Creator

Follow the checkout's [AGENTS.md](../../../AGENTS.md). Preserve the source and use the user's existing decisions about inputs, trusted fields, scope, and delivery.

Treat each input directory named by the user as one dataset. Give each its own delivery, even when filenames, languages or speakers coincide. Workers may be shared for inference. Keep sample identities, quality statistics, tail cutoffs, review, and training export scoped to the individual dataset. Subdirectories within one supplied root retain their source layout.

Keep original filenames and the existing relative directory structure. Obtain user authorization before bulk renumbering or regrouping maintained data. Use the user's speaker mapping first, then the [source evidence rules for DiffSinger export](references/review-and-deliver.md#deliver). Before processing audio, [check its channels and prepare mono input](references/prepare-inputs.md#mono-audio).

In dataset creation, “quality” means the quality of the annotations. A recording issue counts only through an observed defect in the labels, such as an omitted lyric or a misplaced boundary. Carry decodable, prepared clips through lyric matching and annotation regardless of loudness, clipping or noise. For a request to remove poor-quality samples, apply the [annotation quality filter](references/review-and-deliver.md#quality-tail-filtering).

## Choose the next operation

Open the relevant reference or linked section when you reach it. Once the requested result is ready, deliver it. Continue to another stage only when the task calls for it.

| Input or task | Reference |
| --- | --- |
| Unfamiliar data or format investigation | [Investigate the source](references/inspect-datasets.md). |
| Long recordings or new cuts | [Slice recordings](references/prepare-inputs.md#long-recordings). |
| Missing or ambiguous song identity or original lyrics, with lookup in scope | [Search using phrases from different clips and verify the sung version](references/find-lyrics.md). |
| Full annotation from recordings with no paired lyrics | [Resolve the song reference and pair the clips](references/match-lyrics.md) before starting annotation. |
| Existing clips and whole-song lyrics | [Match and apply lyrics](references/match-lyrics.md). |
| Doubt about generated lyrics for a clip | [Run forced alignment on the candidate](references/match-lyrics.md#hand-candidates-to-forced-alignment) before deciding its disposition. |
| A clip's ASR uses an unexpected writing system or the sung language seems misidentified | Preserve the hypothesis for matching. If a different recognition language is warranted, [carry the retry through matching and alignment](references/match-lyrics.md#suspected-asr-language-error). |
| A suspected split word after lyric matching, with cut changes in scope | [Inspect and optionally refine the local cut](references/refine-cuts.md). |
| Paired clips that need checking or annotation | [Check paired inputs](references/prepare-inputs.md#paired-clips) or [run annotation](references/reannotate.md#new-annotation). |
| Notes from existing labels or from clips without syllable boundaries | [Choose the notes input](references/reannotate.md#notes-inputs). |
| Existing labels to import, preserve, or convert | [Convert or reuse annotations](references/convert-annotations.md). |
| Selected samples or layers to replace | [Reannotate only the requested scope](references/reannotate.md#partial-annotation). |
| Missing or unusable capability | [Set up or repair the required capability](references/setup-and-recovery.md#set-up-or-repair-a-required-capability). |
| Failed stage or anomalies across clips | [Identify the shared failing input or resource](references/setup-and-recovery.md#classify-failures-before-retrying). Continue clips that can run independently. |
| Interrupted run | [Resume from current stage outputs](references/setup-and-recovery.md#resume-completed-work). |
| Completed annotations | [Validate](references/review-and-deliver.md#validate-outputs), [deliver the native alignment statistics](references/review-and-deliver.md#quality-overview), and [verify the handoff](references/review-and-deliver.md#deliver). |
| User requests quality-based exclusion or a preview | [Inspect the quality tail](references/review-and-deliver.md#quality-tail-filtering). When applying exclusions, deliver every excluded clip in stage 05 for review. |
| Agent applies a recovery or changes how inputs or outputs are handled | [Record the decision and its result](references/review-and-deliver.md#record-agent-decisions). |
| Unresolved annotation problems after recovery, or an explicit selection for listening | [Deliver the pending or selected review material](references/review-and-deliver.md#review-samples). |

## Run standard stages

Follow the documented SOP without asking for confirmation when it already determines the next action and the user's requirements allow it to proceed. Ask only when missing information or authorization blocks the next action, or when the user's requirements leave a material decision unresolved. Continue any unaffected work while waiting.

For full annotation, run the available candidates through FA across the requested dataset. Investigate disputed passages locally while other clips continue. Before final filtering and export, [account for every requested clip](references/review-and-deliver.md#validate-outputs) and finish any runnable work.

Use the [saved runtime](references/setup-and-recovery.md#use-the-saved-runtime) when executing project tools. Use the documented scripts and their `--help` for routine work. For an unfamiliar format, write only the task-local adapter needed to map the source into the [public API](references/api-map.md). Start troubleshooting with the public contract and diagnostics. Read implementation only for a specific unresolved behavior.

For preferences such as shorter clips, fewer cuts, or stricter selection, use the same stage reference to choose the relevant option. Check the current results to identify what needs to change. When a preference has no exact value, compare a small trial with the baseline, keep unrelated settings fixed, and use the results to choose the setting for this run. Report the setting and its effect. Treat an explicit limit as a requirement to verify on the output.
