# Match clips to whole-song lyrics

Each matching call takes one song's cuts, a UTF-8 whole-song reference, and independent ASR hypotheses for those clips. The hypotheses retrieve reference excerpts and provide candidate words and pronunciations. Clip names and order can be arbitrary.

## Run the steps

Use the [README command block](../README.md#match-lyrics-to-clips) to transcribe, match, and apply the result for one song.

### Transcribe clips

See [ASR installation](installation.md#optional-asr-for-lyric-matching) for dependencies and model bundles.

Qwen3-ASR receives the clip audio and language hint. It accepts supported language codes or English language names. Set `--asr-language` when the recognizer and forced aligner need different identifiers. The report records raw hypotheses, per-clip errors, elapsed times, and recognition settings.

### Match and apply

Matching fills an editable lyric table with the selected transcript, including its pronunciation markup. For an unmatched excerpt with nonempty ASR, the clip hypothesis becomes the forced-alignment input. For a song without a reference, use `python scripts/prepare/prepare_lyrics.py CLIPS --asr REPORT --output DELIVERY/02-lyrics/lyrics.csv`. The apply command protects existing sidecars. Under `01-clips`, it writes the plain lyrics next to the audio and stores pronunciation overrides in `.work/paired-lyrics`. See the [paired input rules](workflow.md#paired-input) for how those overrides are reused.

Keep the generated table inside the same delivery as annotation. The first nonempty candidate for each unchanged clip and audio range is recorded in `.work/lyric-candidates.json`. Alignment, the full pipeline and retry can recover an omitted working lyric from this record. A nonempty selected transcript takes precedence. On a new matching pass, save `--evidence FILE` to retain the proposed transcript and its pronunciation even where an existing pair is protected. Use forced alignment on the original and on any proposed replacement before applying the decision.

To reuse a different recognizer, pass a separate directory of per-clip LAB/TXT hypotheses with the matching relative directory structure, or pass JSON of the form `{"clips":{"clip_id":{"text":"hypothesis"}}}`. For nested clips the key includes the relative parent path. An ASR report that records an audio path is checked against the selected clip. Use `--g2p-only` to write matches by phone edit distance. It reads the forced aligner's G2P configuration and skips acoustic weight loading.

## Interpret the evidence

- The matcher removes LRC time prefixes, recognizable credits and known stage cues. Other bracketed passages remain eligible as sung lyrics. Use `--exclude-line N` or `--keep-line N` to correct the classification. When no reference lyrics remain, nonempty clip hypotheses continue to forced alignment.
- **Coarse search.** The forced aligner's g2pflow pipeline supplies complete pronunciation alternatives for both the ASR hypothesis and the lyric reference. Each alternative occupies a separate path through its word. Dynamic programming finds the minimum number of phone insertions, deletions, and substitutions across pairs of legal paths, with more matched phones breaking ties. Source text identifies the excerpt and its offsets. The search can cross line breaks, and repeated text can retain multiple source locations. Two exact phone matches can be joined at source line boundaries or an ASR phrase break to handle a skip, jump, or repetition.
- **Source shortlists.** The matcher pairs the ASR and retrieved reference words by tracing Levenshtein alignment through their g2pflow paths. A block can include several words on either side. The pronunciation scorer first scores the reference and ASR in separate full-clip templates. A path's block score is the best full-clip score compatible with that path. On each side of each block, retain paths at or above `--min-score` (default `0`), up to `--top-k` paths (default `3`). If all scores on one side fall below the threshold, retain that side's best path.
- **Lyric and pronunciation fusion.** The pronunciation scorer evaluates the retained paths from both sources together in one template. Each winning block contributes its own g2pflow words and pronunciations. Identical phone paths use the reference spelling when available. The template represents a word present on only one side as a choice between its phones and an empty path. Neighboring clips with source coordinates help select a containing reference excerpt when one is available.
- Flagged or unmatched clips appear in `05-review-and-decisions/lyric-matching.json`. For detailed debugging, pass `--evidence FILE` to save candidates, source spans, selected words and pronunciation scores. `--candidate-count` controls the coarse shortlist consulted for a containing excerpt. `--min-similarity` flags low coarse phone similarity. `--min-margin` flags a small gap between final paths. A clip with unavailable pronunciation scoring keeps the coarse result and a diagnostic.
- The lyric table uses g2pflow's selected words from the reference, ASR, or both. Plain text is used where it preserves those words and pronunciations on reprocessing. PFML binds the words that need explicit groups and phones. Run preflight and forced alignment on the paired clips to obtain timings and alignment quality metrics.

Recognition and matching return success once their output is written, including runs with per-clip failures or unmatched items. Check their JSON summaries for those counts. A missing model, incompatible resource, or invalid input configuration is a stage error. These steps write lyric sidecars and reports without copying audio.
