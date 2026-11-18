# Debug G2P and repair clip transcripts

Start with the [adopted source fields](source-fields.md). Verify the sung words and language against the authorized lyrics and audio. Correct a clear typo or an unintended character form in the working transcript first. Use minimal PFML when the verified text still needs pronunciation control or the source spelling must be preserved.

Repair straightforward cases before preparing the review list. Send encodable candidates to [forced alignment](match-lyrics.md#hand-candidates-to-forced-alignment), including supported alternative readings. If G2P still blocks alignment, record the exact word and conversion error and continue with the other clips.

## Diagnose one failed clip

Read the saved problem and use the same forced-alignment resources, language scope, and VED settings as the failed run. The debug command shows the selected transcript, the stage and issue, and any encodable phone paths for optional plain-text probes. It runs preflight without acoustic inference. Use the [saved runtime](setup-and-recovery.md#use-the-saved-runtime).

~~~bash
python scripts/prepare/debug_g2p.py clips/clip.wav --language ja --resources configs/resources.local.json --probe '爱' --probe '愛'
~~~

| Finding | Action |
| --- | --- |
| Wrong transcript or language scope | Check the selected sidecar, written lyrics, and clip audio. Correct the language scope when the singing supports it. Use `--extended-language` for an evidenced additional language. |
| Unintended character form or spelling | Check the affected span against the trusted lyrics and the conventions of the sung language. Examples include a simplified Chinese character in Japanese text and a traditional or variant character unsupported by the selected Chinese dictionary. Prefer a local plain-text correction when it preserves the sung word and intended reading. Test the full candidate transcript. |
| Verified spelling still fails G2P or must be kept verbatim | Use minimal PFML for a known reading. Probe an equivalent spelling if it helps recover the model's phone symbols. Probe paths list phones in sequence. Verify the group boundaries before writing PFML for a word with several groups. |
| G2P misses one of several defensible pronunciations | Keep the written text and supply the evidence-backed readings in PFML for the forced aligner to choose acoustically. |
| Malformed PFML or out-of-vocabulary phones | Check XML escaping, language, group boundaries, and the configured vocabulary. Keep strict OOV handling while verifying the repair. |
| ASR words disagree with the sung passage or verified lyric version | Return to [lyric matching](match-lyrics.md#suspected-asr-language-error) for the affected clip. Verify a proposed replacement against the audio. |
| Missing input, unreadable audio, or repeated software failure | Follow [failure handling](setup-and-recovery.md#classify-failures-before-retrying) for the reported stage. |

Limit text corrections to spans supported by the source or language context. Preserve intentional spellings and any source orthography required by the task.

## Validate and apply a repair

Stage a complete candidate transcript in `.work`. Test it with the same command and `--candidate PATH`. The report compares the selected transcript with the candidate. Exit code 0 means the candidate passed preflight, or the selected transcript passed when no candidate was supplied. Inspect any warnings. Preflight checks encoding. Run FA on the candidate to verify the disputed passage before applying a text correction.

~~~bash
python scripts/prepare/debug_g2p.py clips/clip.wav --language ja --resources configs/resources.local.json --candidate .work/clip.txt
~~~

When the evidence supports the repair, apply the checked candidate to the single working pair. Preserve the previous transcript and record the repair in the [decision log](review-and-deliver.md#record-agent-decisions). In the numbered delivery layout, update the plain transcript beside the clip and put any PFML override in the [working transcript directory](../../../../docs/workflow.md#paired-input). Preserve source files.

Rerun only the affected sample through the requested stages. Use `scripts/retry.py --sample ID` for saved problem records, or the same stage scripts with the corrected pair for new work. Verify the repaired result and keep the aligner's candidate and selection diagnostics. Reconcile the review list with the rerun outcomes.

## Minimal markup

Plain text alone is valid PFML. Start with the original transcript and add inline tags only around spans that need pronunciation control, whether specifying one reading or a bounded set of candidates. Leave the surrounding text unchanged for normal G2P. A whole transcript needs no tags when ordinary G2P already handles it correctly.

## One known pronunciation

When only one word needs an override, keep the surrounding lyrics as plain text. The marked word carries its original text and complete final phonemes:

~~~xml
心头<word text="重" language="zh" script="zhong" phonemes="zh ong"/>担
~~~

## A bounded set of possible readings

Keep alternatives inside the same word. Insert this fragment only where needed in the otherwise plain-text transcript:

~~~xml
<word text="重" language="zh">
  <reading><group script="chong" phonemes="ch ong"/></reading>
  <reading><group script="zhong" phonemes="zh ong"/></reading>
</word>
~~~

Each reading is an alternative, not an extra sung word. Multiple path elements inside one reading can represent alternative complete realizations of that reading. Do not flatten alternatives into a single phone sequence or select one arbitrarily.

These examples illustrate PFML structure. Use the phone symbols and language tags accepted by the actual model vocabulary. Include final phonemes in every direct candidate. Escape XML text and attribute characters when constructing PFML. Python callers can use `Transcript(content, format="pfml")`.
