# MakeDiffSinger

Build accurate, informative and maintainable singing datasets in agentic or programmatic ways. Finish your labeling tasks with one line of prompt, or step-by-step commands with full control.

## Annotate a dataset in one run

With [setup](#install) complete, use **Prompt** for an unfamiliar dataset or recordings with song-level lyrics. Use **Command** for clips with matching transcripts. Both examples deliver editable annotations and a quality overview in a fresh output directory. See the [full pipeline and its outputs](docs/workflow.md#full-pipeline).

Replace angle-bracketed placeholders such as `<DATASET_DIR>` and `<DELIVERY_DIR>` with your own values. Run commands from the repository root.

When supplying multiple dataset directories, each gets its own [delivery and quality ranking](docs/workflow.md#delivery-layout).

<details name="full" open>
<summary>Prompt</summary>

~~~text
Annotate the singing dataset in <DATASET_DIR> and save the results under <DELIVERY_DIR>.

- Recut the original recordings.
- Reuse only the written lyrics from the source annotations.
- Exclude clips with poor annotation quality.
~~~

</details>

<details name="full">
<summary>Command</summary>

~~~bash
python scripts/run_pipeline.py "<PAIRED_CLIPS_DIR>" --language zh --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>"
~~~

</details>

## Install

Use `dataset-creator` from `.agents/skills`. If your agent needs an explicit entry point, load [SKILL.md](.agents/skills/dataset-creator/SKILL.md) or install the complete Skill folder. See [responsibilities and handoffs](docs/agent-workflow.md).

For command-line setup, use Python 3.12 and one environment for both inference backends. Run the first block to initialize the submodules and create the environment. Activate it and install a PyTorch build suitable for your hardware before running the second block. For inspection and conversion alone, omit the inference extra.

For inference, copy [configs/resources.example.json](configs/resources.example.json) to `configs/resources.local.json`, set the model paths, and use this file for `<RESOURCES_JSON>`. Save a device in `local/agent-setup.json` or pass `--device` to the inference command. See [installation and resources](docs/installation.md).

<details name="install" open>
<summary>Prompt</summary>

~~~text
Set up my Python environment <ENV_NAME> for Mandarin annotation on my GPU.
~~~

</details>

<details name="install">
<summary>Command</summary>

~~~bash
git submodule update --init --recursive
python -m venv "<VENV_DIR>"
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
Inspect <DATASET_DIR> and summarize what can be reused for annotation.
~~~

</details>

<details name="inspect">
<summary>Command</summary>

~~~bash
python scripts/prepare/inspect_sources.py "<DATASET_DIR>" --output "<WORK_DIR>/inventory.json"
~~~

</details>

## Prepare mono audio

Audio stages require mono input. Distinct stereo regions become independent channel samples. Shared regions use the average or fall back to the left channel when the average has severe cancellation. See [channel checks and conversion](docs/workflow.md#prepare-mono-audio) and [exact range extraction](docs/workflow.md#extract-an-exact-stereo-range).

<details name="mono" open>
<summary>Prompt</summary>

~~~text
Prepare a mono copy of <DATASET_DIR> under <WORK_DIR>/mono.
~~~

</details>

<details name="mono">
<summary>Command</summary>

~~~bash
python scripts/prepare/prepare_mono.py "<DATASET_DIR>"
~~~

~~~bash
python scripts/prepare/prepare_mono.py "<DATASET_DIR>" --output "<WORK_DIR>/mono"
~~~

</details>

## Find songs and original lyrics

The Skill can search from metadata or independently recognized lyrics and [verify the reference](.agents/skills/dataset-creator/references/find-lyrics.md).

<details open>
<summary>Prompt</summary>

~~~text
Identify the songs in <RECORDINGS_DIR> and find their original lyrics.
~~~

</details>

## Slice long recordings

Save the slice plan and export a tab-separated [Audition cut-marker `.csv`](docs/workflow.md#archive-slice-markers) beside the clips.

<details name="slice" open>
<summary>Prompt</summary>

~~~text
Slice <RECORDING_WAV> into clips under <SONG_WORK_DIR>.
~~~

</details>

<details name="slice">
<summary>Command</summary>

~~~bash
python scripts/prepare/plan_slices.py "<RECORDING_WAV>" --recording-id "<RECORDING_ID>" --output "<SONG_WORK_DIR>/slices.json"
python scripts/prepare/replay_slices.py "<SONG_WORK_DIR>/slices.json" --output "<SONG_WORK_DIR>/clips"
python scripts/prepare/export_slice_markers.py "<SONG_WORK_DIR>/slices.json" --output "<SONG_WORK_DIR>/clips/<RECORDING_ID>.csv"
~~~

</details>

## Match lyrics to clips

Match one song at a time. Set `<ASR_MODEL_DIR>` to the local [Qwen3-ASR](docs/installation.md#optional-asr-for-lyric-matching) model directory under `local/models/`. See [lyric matching](docs/lyric-matching.md) for the editable table and match report.

<details name="match" open>
<summary>Prompt</summary>

~~~text
Write a transcript for each clip in <SONG_WORK_DIR>/clips using the song lyrics in <LYRICS_TXT>.
~~~

</details>

<details name="match">
<summary>Command</summary>

~~~bash
python scripts/prepare/transcribe_clips.py "<SONG_WORK_DIR>/clips" --model "<ASR_MODEL_DIR>" --language zh --output "<SONG_WORK_DIR>/asr.json"
python scripts/prepare/match_lyrics.py "<SONG_WORK_DIR>/clips" --lyrics "<LYRICS_TXT>" --asr "<SONG_WORK_DIR>/asr.json" --language zh --resources "<RESOURCES_JSON>" --output "<SONG_WORK_DIR>/match"
python scripts/prepare/apply_lyrics.py "<SONG_WORK_DIR>/match/lyrics.csv" --audio-root "<SONG_WORK_DIR>/clips"
~~~

</details>

## Refine a cut after matching

Use this step when revising cuts is in scope. See [local cut refinement](.agents/skills/dataset-creator/references/refine-cuts.md).

<details open>
<summary>Prompt</summary>

~~~text
Adjust the cut between <LEFT_CLIP> and <RIGHT_CLIP> if it splits a sung word.
~~~

</details>

## Start with paired clips

Pair each clip with a same-basename UTF-8 transcript. Deliver plain lyrics with `01-clips`. The [paired input rules](docs/workflow.md#paired-input) explain how the aligner reuses working PFML.

<details name="paired" open>
<summary>Prompt</summary>

~~~text
Check whether the paired Mandarin clips in <PAIRED_CLIPS_DIR> are ready for alignment.
~~~

</details>

<details name="paired">
<summary>Command</summary>

~~~bash
python scripts/preflight.py "<PAIRED_CLIPS_DIR>" --language zh --resources "<RESOURCES_JSON>" --output "<WORK_DIR>/preflight"
~~~

</details>

## Run forced alignment

See [forced alignment and diagnosis](docs/workflow.md#forced-alignment-and-diagnosis).

<details name="align" open>
<summary>Prompt</summary>

~~~text
Force-align the Mandarin clips in <PAIRED_CLIPS_DIR> and save the results under <DELIVERY_DIR>, including aligned PFML.
~~~

</details>

<details name="align">
<summary>Command</summary>

~~~bash
python scripts/align.py "<PAIRED_CLIPS_DIR>" --language zh --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>/.work/fa" --pfml
~~~

</details>

## Annotate notes

Use forced-alignment output or existing annotations with usable syllable boundaries. If those boundaries are unavailable, [extract independent notes](docs/workflow.md#notes-without-syllable-boundaries). When requesting independent notes, say whether you accept notes without phoneme alignment.

<details name="notes" open>
<summary>Prompt</summary>

~~~text
Add notes to the existing alignments in <DELIVERY_DIR>.
~~~

</details>

<details name="notes">
<summary>Command</summary>

~~~bash
python scripts/annotate_notes.py "<DELIVERY_DIR>/.work/fa" --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>/03-annotations"
~~~

</details>

## Import or convert existing annotations

The built-in importer accepts canonical TextGrid and DiffSinger CSV. For other layouts, map verified fields into `import_intervals`. See [import and conversion](docs/workflow.md#existing-labels-and-format-conversion) and the [Python import API](docs/api.md#existing-labels).

<details name="convert" open>
<summary>Prompt</summary>

~~~text
Export DiffSinger CSV only to <DELIVERY_DIR> using the phone timings, pronunciation groups, and notes in <SOURCE_LABELS_PATH>.
~~~

</details>

<details name="convert">
<summary>Command</summary>

~~~bash
python scripts/import_annotations.py "<SOURCE_LABELS_PATH>" --audio-root "<SOURCE_AUDIO_DIR>" --language zh --groups pronunciation --output "<WORK_DIR>/imported"
python scripts/export_diffsinger.py "<WORK_DIR>/imported" --resources "<RESOURCES_JSON>" --no-audio --output "<DELIVERY_DIR>/06-diffsinger"
~~~

</details>

## Reannotate selected layers

See [partial reannotation](docs/workflow.md#partial-reannotation) for layer selection and edit protection.

<details name="reannotate" open>
<summary>Prompt</summary>

~~~text
Regenerate only the notes layer for <SUBSET_PATH> in <DELIVERY_DIR>.
~~~

</details>

<details name="reannotate">
<summary>Command</summary>

~~~bash
python scripts/reannotate.py "<DELIVERY_DIR>/03-annotations" --layers notes --sample "<SUBSET_PATH>/*" --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>/03-annotations" --overwrite
~~~

</details>

## Retry failed items

Resolve the input or resource issue before retrying the affected stage. See [stage-specific recovery](docs/workflow.md#recovery).

<details name="retry" open>
<summary>Prompt</summary>

~~~text
Retry the failed annotations in <DELIVERY_DIR>.
~~~

</details>

<details name="retry">
<summary>Command</summary>

~~~bash
python scripts/retry.py "<DELIVERY_DIR>/03-annotations" --stage all --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>/03-annotations" --overwrite
~~~

</details>

## Check quality and select samples

The quality report describes the annotations. Applying a quality filter places every excluded clip's WAV, available TextGrid and reason in `05-review-and-decisions/`. The full pipeline also delivers native alignment statistics and listening files for unresolved items. Use the [inference environment](docs/installation.md#optional-tifa-quality-plots) to regenerate statistics, or [select completed samples](docs/workflow.md#manual-review-selection) for additional listening.

<details name="quality" open>
<summary>Prompt</summary>

~~~text
Review annotation quality in <DELIVERY_DIR> and recommend a retention rule for training.
~~~

</details>

<details name="quality">
<summary>Command</summary>

~~~bash
python scripts/check.py "<DELIVERY_DIR>/03-annotations" --complete --audio --resources "<RESOURCES_JSON>"
python scripts/report.py "<DELIVERY_DIR>/03-annotations" --output "<DELIVERY_DIR>/04-quality"
~~~

</details>

## Deliver annotations or export DiffSinger data

Keep editable annotations and paired inputs when compiling a training dataset. See the [delivery layout](docs/workflow.md#delivery-layout), [DiffSinger format](docs/annotations.md#diffsinger-compilation), and [export options](docs/workflow.md#select-and-compile).

<details name="deliver" open>
<summary>Prompt</summary>

~~~text
Compile the annotations in <DELIVERY_DIR> into a DiffSinger training dataset.
~~~

</details>

<details name="deliver">
<summary>Command</summary>

~~~bash
python scripts/export_diffsinger.py "<DELIVERY_DIR>/03-annotations" --resources "<RESOURCES_JSON>" --output "<DELIVERY_DIR>/06-diffsinger"
~~~

</details>

## Reference

[Script workflow](docs/workflow.md) · [Python API](docs/api.md) · [Annotation format](docs/annotations.md) · [Testing and compatibility](docs/validation.md)
