# MakeDiffSinger

Build accurate, informative and maintainable singing datasets in agentic or programmatic ways. Finish your labeling tasks with one line of prompt, or step-by-step commands with full control.

## Annotate a dataset in one run

With [setup](#install) complete, use **Prompt** for an unfamiliar dataset or recordings with song-level lyrics. Use **Command** for clips with matching transcripts. Both examples deliver editable annotations and a quality overview in a fresh output directory. See the [full pipeline and its outputs](docs/workflow.md#full-pipeline).

When supplying multiple dataset directories, each gets its own [delivery and quality ranking](docs/workflow.md#delivery-layout).

<details name="full" open>
<summary>Prompt</summary>

~~~text
Use dataset-creator to annotate the Mandarin singing dataset in ./incoming.

- Recut the original recordings.
- Reuse only the written lyrics. Look up the original lyrics when they are missing.
- Use CPU only. Proceed even if some clips are unusually long.
- Deliver editable annotations and a quality overview under ./outputs/run.
~~~

</details>

<details name="full">
<summary>Command</summary>

~~~bash
python scripts/run_pipeline.py clips --language zh --resources configs/resources.local.json --output outputs/run
~~~

</details>

## Install

Use `dataset-creator` from `.agents/skills`. If your agent needs an explicit entry point, load [SKILL.md](.agents/skills/dataset-creator/SKILL.md) or install the complete Skill folder. See [responsibilities and handoffs](docs/agent-workflow.md).

For command-line setup, use Python 3.12 and one environment for both inference backends. Run the first block to initialize the submodules and create the environment. Activate it and install a PyTorch build suitable for your hardware before running the second block. For inspection and conversion alone, omit the inference extra.

For inference, copy [configs/resources.example.json](configs/resources.example.json) to `configs/resources.local.json` and set the model paths. Save a device in `local/agent-setup.json` or pass `--device` to the inference command. See [installation and resources](docs/installation.md).

<details name="install" open>
<summary>Prompt</summary>

~~~text
Set up this checkout in my Python environment named ENV_NAME for Mandarin annotation on my GPU. Save the working setup for later tasks.
~~~

</details>

<details name="install">
<summary>Command</summary>

~~~bash
git submodule update --init --recursive
python -m venv .venv
~~~

~~~bash
python -m pip install -e ".[inference]"
~~~

</details>

## Inspect an unfamiliar dataset

Use the inventory to locate representative source files and verify their fields before mapping them. See [existing labels and format conversion](docs/workflow.md#existing-labels-and-format-conversion).

<details name="inspect" open>
<summary>Prompt</summary>

~~~text
Inspect ./incoming. Tell me whether the audio is already sliced, which annotation fields we can reuse, and how the source is organized.
~~~

</details>

<details name="inspect">
<summary>Command</summary>

~~~bash
python scripts/prepare/inspect_sources.py incoming --output work/inventory.json
~~~

</details>

## Prepare mono audio

Audio stages require mono input. Distinct stereo regions become independent channel samples. Shared regions use the average or fall back to the left channel when the average has severe cancellation. See [channel checks and conversion](docs/workflow.md#prepare-mono-audio) and [exact range extraction](docs/workflow.md#extract-an-exact-stereo-range).

<details name="mono" open>
<summary>Prompt</summary>

~~~text
Check the channels in ./incoming and prepare a mono working copy under ./work/mono.
~~~

</details>

<details name="mono">
<summary>Command</summary>

~~~bash
python scripts/prepare/prepare_mono.py incoming
~~~

~~~bash
python scripts/prepare/prepare_mono.py incoming --output work/mono
~~~

</details>

## Find songs and original lyrics

The Skill can search from metadata or independently recognized lyrics and [verify the reference](.agents/skills/dataset-creator/references/find-lyrics.md).

<details open>
<summary>Prompt</summary>

~~~text
Identify the songs in ./recordings and find their original lyrics. Give me the sources and say which recordings remain unresolved.
~~~

</details>

## Slice long recordings

Save the slice plan and export a tab-separated [Audition cut-marker `.csv`](docs/workflow.md#archive-slice-markers) beside the clips.

<details name="slice" open>
<summary>Prompt</summary>

~~~text
Slice ./recordings/song.wav from the audio. Save the cuts and the slice plan under ./work/song.
~~~

</details>

<details name="slice">
<summary>Command</summary>

~~~bash
python scripts/prepare/plan_slices.py recordings/song.wav --recording-id song --output work/song/slices.json
python scripts/prepare/replay_slices.py work/song/slices.json --output work/song/clips
python scripts/prepare/export_slice_markers.py work/song/slices.json --output work/song/clips/song.csv
~~~

</details>

## Match lyrics to clips

Match one song at a time. The script workflow uses [Qwen3-ASR](docs/installation.md#optional-asr-for-lyric-matching). Replace `ASR_BUNDLE` with its folder name under `local/models/`. See [lyric matching](docs/lyric-matching.md) for the editable table and match report.

<details name="match" open>
<summary>Prompt</summary>

~~~text
Match the clips in ./work/song/clips to the whole-song lyrics in ./sources/song.txt. Write the paired transcripts and report unmatched clips.
~~~

</details>

<details name="match">
<summary>Command</summary>

~~~bash
python scripts/prepare/transcribe_clips.py work/song/clips --model local/models/ASR_BUNDLE --language zh --output work/song/asr.json
python scripts/prepare/match_lyrics.py work/song/clips --lyrics sources/song.txt --asr work/song/asr.json --language zh --resources configs/resources.local.json --output work/song/match
python scripts/prepare/apply_lyrics.py work/song/match/lyrics.csv --audio-root work/song/clips
~~~

</details>

## Refine a cut after matching

Use this step when revising cuts is in scope. See [local cut refinement](.agents/skills/dataset-creator/references/refine-cuts.md).

<details open>
<summary>Prompt</summary>

~~~text
Check whether these two adjacent clips split a sung word. If a nearby cut gives better lyrics for both clips, test and apply it.
~~~

</details>

## Start with paired clips

Pair each clip with a same-basename UTF-8 transcript. Deliver plain lyrics with `01-clips`. The [paired input rules](docs/workflow.md#paired-input) explain how the aligner reuses working PFML.

<details name="paired" open>
<summary>Prompt</summary>

~~~text
Check the paired Mandarin clips in ./clips. Report missing transcripts, pronunciation failures, and unusual clip durations.
~~~

</details>

<details name="paired">
<summary>Command</summary>

~~~bash
python scripts/preflight.py clips --language zh --resources configs/resources.local.json --output work/preflight
~~~

</details>

## Run forced alignment

See [forced alignment and diagnosis](docs/workflow.md#forced-alignment-and-diagnosis).

<details name="align" open>
<summary>Prompt</summary>

~~~text
Align the paired Mandarin clips in ./clips. Archive the alignment TextGrids and aligned PFML under ./outputs/.work/fa.
~~~

</details>

<details name="align">
<summary>Command</summary>

~~~bash
python scripts/align.py clips --language zh --resources configs/resources.local.json --output outputs/.work/fa --pfml
~~~

</details>

## Annotate notes

Use forced-alignment output or existing annotations with usable syllable boundaries. If those boundaries are unavailable, [extract independent notes](docs/workflow.md#notes-without-syllable-boundaries). When requesting independent notes, say whether you accept notes without phoneme alignment.

<details name="notes" open>
<summary>Prompt</summary>

~~~text
Add notes to ./outputs/.work/fa using its existing phonemes and pronunciation groups. Write the annotations to ./outputs/03-annotations.
~~~

</details>

<details name="notes">
<summary>Command</summary>

~~~bash
python scripts/annotate_notes.py outputs/.work/fa --resources configs/resources.local.json --output outputs/03-annotations
~~~

</details>

## Import or convert existing annotations

The built-in importer accepts canonical TextGrid and DiffSinger CSV. For other layouts, map verified fields into `import_intervals`. See [import and conversion](docs/workflow.md#existing-labels-and-format-conversion) and the [Python import API](docs/api.md#existing-labels).

<details name="convert" open>
<summary>Prompt</summary>

~~~text
Convert ./source_labels to DiffSinger CSV under ./outputs/06-diffsinger. Reuse the verified phone timings, pronunciation groups, and notes. Deliver CSV only.
~~~

</details>

<details name="convert">
<summary>Command</summary>

~~~bash
python scripts/import_annotations.py source_labels --audio-root source_audio --language zh --groups pronunciation --output work/imported
python scripts/export_diffsinger.py work/imported --resources configs/resources.local.json --no-audio --output outputs/06-diffsinger
~~~

</details>

## Reannotate selected layers

See [partial reannotation](docs/workflow.md#partial-reannotation) for layer selection and edit protection.

<details name="reannotate" open>
<summary>Prompt</summary>

~~~text
Regenerate only the notes layer for singer_a/control in ./outputs/03-annotations. Reuse the current phoneme timings and pronunciation groups.
~~~

</details>

<details name="reannotate">
<summary>Command</summary>

~~~bash
python scripts/reannotate.py outputs/03-annotations --layers notes --sample "singer_a/control/*" --resources configs/resources.local.json --output outputs/03-annotations --overwrite
~~~

</details>

## Retry failed items

Resolve the input or resource issue before retrying the affected stage. See [stage-specific recovery](docs/workflow.md#recovery).

<details name="retry" open>
<summary>Prompt</summary>

~~~text
Diagnose the failed items in ./outputs/03-annotations and retry the ones whose causes can be resolved. Report any remaining failures.
~~~

</details>

<details name="retry">
<summary>Command</summary>

~~~bash
python scripts/retry.py outputs/03-annotations --stage all --resources configs/resources.local.json --output outputs/03-annotations --overwrite
~~~

</details>

## Check quality and select samples

The quality report describes the annotations. Applying a quality filter places every excluded clip's WAV, available TextGrid and reason in `05-review-and-decisions/`. The full pipeline also delivers native alignment statistics and listening files for unresolved items. Use the [inference environment](docs/installation.md#optional-tifa-quality-plots) to regenerate statistics, or [select completed samples](docs/workflow.md#manual-review-selection) for additional listening.

<details name="quality" open>
<summary>Prompt</summary>

~~~text
Check ./outputs/03-annotations, show the quality distributions, and recommend a retention rule for training.
~~~

</details>

<details name="quality">
<summary>Command</summary>

~~~bash
python scripts/check.py outputs/03-annotations --complete --audio --resources configs/resources.local.json
python scripts/report.py outputs/03-annotations --output outputs/04-quality
~~~

</details>

## Deliver annotations or export DiffSinger data

Keep editable annotations and paired inputs when compiling a training dataset. See the [delivery layout](docs/workflow.md#delivery-layout), [DiffSinger format](docs/annotations.md#diffsinger-compilation), and [export options](docs/workflow.md#select-and-compile).

<details name="deliver" open>
<summary>Prompt</summary>

~~~text
Keep the editable annotations in ./outputs/03-annotations. Export a training-ready DiffSinger dataset to ./outputs/06-diffsinger.
~~~

</details>

<details name="deliver">
<summary>Command</summary>

~~~bash
python scripts/export_diffsinger.py outputs/03-annotations --resources configs/resources.local.json --output outputs/06-diffsinger
~~~

</details>

## Reference

[Script workflow](docs/workflow.md) · [Python API](docs/api.md) · [Annotation format](docs/annotations.md) · [Testing and compatibility](docs/validation.md)
