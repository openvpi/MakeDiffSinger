# Review, select, and deliver

## Validate outputs

Use `scripts/check.py` for the requested layer, timing, grouping, and audio correspondence checks. Use this checkout's [annotation and compilation contract](../../../../docs/annotations.md). For partial updates, check preserved layers and unselected items. For conversions, verify the trusted fields and timing against the source.

Add `--complete` when all four maintained layers are required and `--audio` when checking audio correspondence. Use `--sample`, `--subset`, or the available metadata filters to validate the requested scope. Use the relevant subset or speaker filter only when those labels are already known.

For a full run, compare the input clip IDs with the maintained annotations and the recorded blockers for the requested stages. Validation of existing TextGrids cannot detect an omitted clip. Resume every unaccounted ID that can run, including clips that have only a trial FA. Perform the requested quality selection over the full eligible population once this reconciliation is complete. A partial export is a checkpoint while runnable IDs remain.

## Quality overview

For a run that includes forced alignment, deliver the aligner's native statistics in `04-quality/`. The full pipeline writes them there. If using TIFA directly, enable `--stat` and copy the contents of that run's `statistics/` directory. For a custom adapter, use the [statistics writer](../../../../docs/workflow.md#quality-overview) with the current run's measurements.

At handoff, check the sample IDs in `diagnosis.json` against the clips with complete, current measurements. Check for `scores.json` and both the histogram and frame-count scatter plot for agreement, confidence, determinacy and monotonicity. Report any gaps. A notes-only run or an import can have no alignment measurements.

## Quality tail filtering

Enter this step only when the user asks to exclude samples by quality or explicitly asks for a preview. A quality overview or a request for review copies alone does not authorize exclusion.

Use blocking defects in the requested annotations and the ranked tail of alignment diagnostics. If a recording issue affects the labels, name the observed defect, such as skipped phones or incorrect note boundaries. A waveform statistic alone supplies no exclusion reason. Channel preparation and audio correspondence checks still apply. Report an unreadable input as blocked. Audio restoration or duplicate-content cleanup requires a separate request. Keep usable annotations without a current alignment rank outside the ranked denominator.

1. Run `scripts/quality_tail.py ANNOTATIONS` with the same sample and metadata scope and `--percentile-group` as the intended operation. Read the eligible population, unavailable diagnoses, and five diagnostic distributions by rank band. Increase `--bands` for finer resolution. Use `report.py` for full per-sample evidence when needed.
2. Use a user-specified fraction directly. If exclusion is requested without a fraction, choose it from the current evidence. Look for a clear difference between the worst-ranked tail and the adjacent ranks, such as an increase in skipped phones or a drop in the alignment metrics. Prefer the smallest worst-ranked prefix supported by the diagnostic change. Treat a dataset-wide quality problem as a [stage failure](setup-and-recovery.md#classify-failures-before-retrying). If the diagnostics support no distinct tail, choose zero and report the reason. The supplied request to exclude poor samples is enough to act on the evidence-based fraction.
3. Preview the chosen positive fraction with `quality_tail.py --preview-fraction FRACTION`. Check the boundary diagnoses, ranked exclusions, and retained unranked samples. Use `--drop-bottom-fraction FRACTION` for the intended operation. Keep the scoped population and grouping consistent. Reuse the retained IDs from the filtered report when several stages share one selection.
4. Deliver every quality exclusion in `05-review-and-decisions/` with its clip WAV, available TextGrid and exclusion reason. This includes both the ranked tail and individual clips excluded for a specific annotation defect. Pass each such completed clip as `--exclude-quality KEY REASON` so the same decision controls the operation and review copies. For unresolved clips use the [pending pack](#review-samples) with the available evidence. Keep the full maintained annotations.
5. Check the actual excluded IDs against the listening links in `quality-review.md` and any pending entries in `review.md`. Report the scope, denominator, rule and counts, including the evidence for an automatically chosen fraction. Apply the training decision immediately; listening can happen after delivery.

## Record Agent decisions

Use `05-review-and-decisions/decisions.md` to tell the user which material decisions the Agent made during the run. Record applied repairs, overrides, fallbacks, and other choices that change the inputs, stage behavior, or final output. Examples include choosing a different sung version, switching to clip ASR after a reference lookup fails, rerunning with an explicit recognition language, revising a cut, and repairing a transcript or pronunciation.

Add an entry when the action is taken. Update its result after verification, including when the repair succeeds and its pending record is cleared. Group samples in one row when the evidence, action, and result are the same. Name the clip IDs, song, or subset. Link the review entry when the result still needs human follow-up.

| Stage | Affected inputs | Reason and evidence | Action taken | Result |
| --- | --- | --- | --- | --- |

The scripts collect blocking sample errors in `issues.json` and uncertain or unresolved lyric matches in `lyric-matching.json` at the top of stage 05. If a shared failure stops a stage before it writes sample records, report the affected stage, scope and reason there. Link existing records when documenting an intervention. Keep routine success messages, retries that had no effect on the result, and progress counts out of the delivery.

## Review samples

For an explicit review request, choose the requested IDs or a fraction of the worst-ranked alignment results in a defined population. Use the [review commands and selection semantics](../../../../docs/workflow.md#manual-review-selection).

Use `review_pack.py --sample` for explicit clips or `--bottom-fraction FRACTION` for the worst-ranked tail. Add `--group-by subset`, `speaker`, or `language` to rank separately within each existing group. For a representative review, choose IDs across the relevant groups or diagnostic ranges.

After repairs and retries, deliver **every still-unresolved sample** in `05-review-and-decisions/`. The full pipeline synchronizes the pack automatically. After a custom run using `save_result`, run:

```bash
python scripts/review_pending.py outputs/03-annotations --output outputs/05-review-and-decisions
```

Reconcile the pending IDs with the current annotation problems. Resolve entries under **Return to forced alignment** before final handoff. For full annotation, rerun `run_pipeline.py` on those `--sample` IDs in the same delivery so the FA archive and final annotations are both saved. For a custom adapter, check that every blank working lyric with a nonempty ASR or matching candidate has an FA result or a specific upstream blocker, then synchronize the pack. Open `review.md` and verify the stage, the concrete issue, and each available WAV, candidate text and TextGrid. A reviewer should be able to locate the unresolved word or passage. Include the reason when audio or lyrics are unavailable. Deliver listening material for unresolved clips even when excluded from training. Use `review_pack.py` for an explicit selection of finished annotations.

## Deliver

Use the [numbered delivery layout](../../../../docs/workflow.md#delivery-layout) for a shared destination. Link the maintained assets alongside any training derivative. Standalone export from existing annotations uses those annotations as the maintained source.

Bind each delivery to the original directory supplied for that dataset before processing individual songs or subsets. Use [dataset-relative references](../../../../docs/annotations.md#source-paths) for original recordings and labels in tables, reports, and task-local adapters. Use delivery-relative paths for generated clips, TextGrids, and review files. Compute every requested tail fraction against that dataset's own eligible population, including when several datasets share one worker session.

For a task-specific adapter, use the same destinations as the scripts. Collect source recording, clip range and channel, language and known speaker associations in shared tables under one delivery-root `.metadata/` beside `.work/`. Key rows by relative sample path so equal stems in different subdirectories stay distinct. Keep per-sample quality measurements together in `04-quality/`. Put review lists, blocking issues and material judgment calls directly in `05-review-and-decisions/`. For other structured reports use one JSON, JSONL or CSV per purpose instead of distributing files among sample directories.

For full annotation, archive forced-alignment results before note extraction. In task-local API adapters, call `save_result(aligned, fa_root, archive_fa=True)` using the intermediate FA directory. Keep matching PFML in `.work/paired-lyrics` and pair the maintained clips with plain lyrics. Use `run_pipeline.py --pfml` only when the user requests aligned PFML from the forced aligner beside the final TextGrids. Keep training exclusions out of the maintained assets.

Compile a single DiffSinger dataset with the [complete CSV schema](../../../../docs/annotations.md#diffsinger-compilation) in every subset. Deliver just the subset CSVs and the requested audio. Verify row names against the corresponding WAV stems when audio is included. The subset tree and CSV rows are the training inventory. Populate the primary singing language in `Sample.language`. Set `Sample.speaker` from the user's mapping first. Otherwise use clear source evidence such as declared labels, consistent filename prefixes or suffixes, or speaker directories. A song, take, harmony role, or channel suffix alone supplies no speaker identity. Leave ambiguous identities unset. Use the [export options](../../../../docs/workflow.md#select-and-compile) to apply any export-only speaker corrections or request CSV only.

For a requested quality filter, link the [review material for the excluded samples](#quality-tail-filtering). The subset CSVs already list all retained rows. Do not create a second inventory of included samples, output paths, or cut-to-training mappings anywhere in the delivery, including stage 05. Link the shared source tables for tracing clips.

Before handing off a full run, verify the cut markers, four-tier TextGrids, root source and language tables, [native statistics](#quality-overview), and any decision or review records. Verify the intermediate FA archive separately. Check the compiled tree against the [DiffSinger format](../../../../docs/annotations.md#diffsinger-compilation).

Report destinations, counts and unresolved items with reasons. Distinguish source-recording, clip and annotation durations. Link the native diagnosis and figures when available. Use the [delivery template](../assets/delivery_report.md) for a requested handoff document.
