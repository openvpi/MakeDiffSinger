# Match song lyrics to clips

Group clips by song. Verify the language and lyric version, and use the [source field definitions](source-fields.md) when preparing the reference. If lookup is in scope and the reference is missing, [find the original lyrics](find-lyrics.md). Check ambiguous credit or section lines for actual singing before excluding them.

For full annotation without paired lyrics, look for the original lyrics as part of preparing the pairs, subject to any source or lookup limits the user set. Resolve the reference per song before writing final paired transcripts. Mark each song as using a verified whole-song reference or an ASR fallback. A song title or search snippet alone is insufficient evidence for a verified reference. When a reliable version cannot be established or lookup is outside the task's scope, use the per-clip ASR fallback and record the choice in the [decision log](review-and-deliver.md#record-agent-decisions). Keep unclear recordings moving on the fallback while matching the songs that do have a reference.

Use [Qwen3-ASR on each selected clip](../../../../docs/lyric-matching.md#transcribe-clips) even when the full lyrics are available. Keep the reference text out of recognition. Reuse independent ASR if it covers the same audio ranges and was produced without the reference. This can include independent hypotheses supplied by the user. If a required model is missing, follow [setup](setup-and-recovery.md#set-up-or-repair-a-required-capability).

For each song with a verified reference, pass the whole-song text and the nonempty clip hypotheses to the [matching command or its public API](../../../../docs/lyric-matching.md#match-and-apply). Preserve short vocalizations and mixed-script passages. Never drop, blank, or classify a hypothesis as missing based on its alphabet, character counts, or proportions. The matcher retrieves excerpts and selects text and pronunciations from both sources. Repeated, reordered, or omitted lyrics can occur. Use source-sample coordinates when comparing adjacent clips.

## Hand candidates to forced alignment

For annotation, carry the selected text and pronunciation (including PFML) into forced alignment on the actual clip. Without a matched reference excerpt, carry the nonempty clip ASR instead. For an ASR-only song, use `prepare_lyrics.py CLIPS --asr REPORT --output TABLE`. Use the [working lyric table](../../../../docs/lyric-matching.md#match-and-apply) even for sparse harmonies, repeated syllables and partial lines. A matching flag is a diagnostic; keep the candidate in the table.

Forced alignment (FA) is the deciding acoustic evidence for disputed lyrics. Run the original candidate through FA before rejecting or rewriting it. If the recording or another authorized source supports a materially different version, align that candidate separately on the same audio and settings. Check the word and phone intervals at the disputed passage, skipped phones and the aligner's diagnostics. Preflight only checks encodability; pronunciation selection during matching does not supply these timings. When FA supports the current reading and there is no localized contradiction, finish annotation and deliver it. Do not introduce a score cutoff or make listening a prerequisite.

For a comparison, use `python scripts/align.py CLIP --candidate PROPOSAL.pfml --language LANG --resources RESOURCES --output DELIVERY/.work/compare/alternative`. Use a separate output for the original or reuse its current FA. The candidate may also be TXT or LAB. This runs the aligner without changing the paired lyrics.

If G2P blocks the proposed reading, try a supported alternative already present in the ASR or matching evidence and apply the [G2P repairs](pronunciation.md) where applicable. If FA still cannot run, record the actual blocker and the attempted text. After FA, send an item to review only for a specific remaining annotation defect. Retain its candidate TextGrid and identify the disputed words or range and the alignment evidence. A request to remove poor samples uses the [quality filter](review-and-deliver.md#quality-tail-filtering) on the completed results.

## Suspected ASR language error

1. If the audio or verified lyric version suggests a different recognition language, rerun `transcribe_clips.py` on the affected `--sample` IDs with `--asr-language`. Keep the original report and write the retry to a fresh file under `.work/`. Use the same audio ranges and sample keys.
2. Feed the retry report to `match_lyrics.py --asr RETRY_REPORT --evidence RETRY_EVIDENCE` with the same verified song reference and the affected IDs. Use the normal acoustic matching pass. Compare both distinct transcripts or pronunciation paths by FA as above. When no verified reference is available, use the retry text as a candidate in the clip-ASR fallback.
3. Force-align the transcript produced by the retry pass to the same clip audio. The matching CLI preserves an existing paired transcript by default. In that case, take the retry's suggested transcript (including any PFML) from the per-clip `--evidence` record or the matching API and stage it with the clip. Check the resulting word and phone intervals against the sung passage before adopting the repair. Update the working pair and lyric table, rerun the requested dependent stages, and reconcile the pending record. If the retry is empty or matching, G2P, or alignment cannot produce a usable result, record what happened to that candidate and retain any usable original result.

## Adjust recognition and matching

| User preference or symptom | Option or input to adjust |
| --- | --- |
| Select a recognition or alignment language | Omit recognition language options for automatic ASR detection. `--asr-language` overrides the recognizer's `--language` hint. Set the aligner's `--language` for matching and add `--extended-language` for an evidenced additional sung language. |
| Credits appear as lyrics, or a sung line is missing from the reference candidates | Verify the lyric version and the exclusions in the match report. Use repeatable `--exclude-line N` or `--keep-line N` for the affected reference lines. |
| The right reference excerpt is missing from the shortlist | Try a larger `--candidate-count` on the affected clips after verifying the reference and ASR. It controls the coarse shortlist for choosing an excerpt. |
| A plausible path from ASR or the reference disappears before joint scoring | Look at both source shortlists. Increase `--top-k` if the candidate is capped by rank, or lower `--min-score` if its independent pronunciation score is below the cutoff. Compare the winning blocks on the same audio and reference. See [scoring and defaults](../../../../docs/lyric-matching.md#interpret-the-evidence). |
| Flag more uncertain matches for later inspection | Raise `--min-similarity` for coarse phone similarity or `--min-margin` for the final pronunciation score gap. These change report flags only. |

Use `--sample` to scope a comparison to affected clips. For an ASR resource limit, adjust `--threads` on CPU. For a matching resource limit, try a smaller `--chunk-size`. Use `--g2p-only` for a requested phone-only run or to isolate coarse retrieval during diagnosis. It omits acoustic pronunciation scoring. Apply changes to working sidecars with `--audio-root`. Use `apply_lyrics.py --overwrite` only when replacement of those sidecars is in scope.

Follow [failure handling](setup-and-recovery.md#classify-failures-before-retrying) if mismatches point to a shared problem. If the task includes changing cuts and evidence points to a split word, [inspect the local cut](refine-cuts.md) after matching.

Apply nonempty rows to the working clips with `apply_lyrics.py` when clip sidecars are requested or needed. Use [minimal PFML](pronunciation.md) where pronunciation instructions are needed, with the matching table format and file extension. Report counts and unmatched reasons. For annotation, [check the paired inputs](prepare-inputs.md#paired-clips) and run the requested stages on the resulting clip/transcript pairs.
