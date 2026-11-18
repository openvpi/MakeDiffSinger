# Installation and resource preparation

The current forced aligner is TIFA and the current note extractor is GAME. Follow the [manual installation commands](../README.md#install) to check out the pinned submodule revisions and install both backends in one Python 3.12 environment. [pyproject.toml](../pyproject.toml) declares the package requirements for each extra. The workers use the interpreter that launched the script. For an accelerator, verify that the selected PyTorch build can execute a tensor operation on it.

Choose model bundles that support the pinned APIs and the requested language and task. The submodules' READMEs list supported languages and distribution links. Store each local bundle under `local/models/` using its original folder name. Keep checkpoints with their matching configurations, vocabularies or language maps, and associated G2P assets.

The inference extra installs g2pflow with Japanese and LSTM backends. Prepare the full UniDic dictionary before an offline Japanese run:

~~~bash
python -m unidic download
~~~

An existing UniDic data directory can instead be referenced by a custom G2P configuration's unidic_dir option. An independent copy of dictionary data is also suitable. Do not mix site-packages from different environments.

The English G2P model requires its ONNX encoder/decoder and character/phone metadata. Dictionary and asset references in the aligner configuration must resolve. Paths beginning with `@` follow TIFA's G2P root rules. The forced aligner receives the custom G2P configuration and extended language filters.

## Optional ASR for lyric matching

Install `asr-qwen` in the shared environment to run Qwen3-ASR with a local Transformers-format bundle. The CUDA path uses BF16. See [Qwen3-ASR releases and API](https://github.com/QwenLM/Qwen3-ASR).

Run `python -m pip install -e ".[inference,asr-qwen]"` after installing the selected PyTorch build. Pass the local model directory using `--model`. For model inputs, existing ASR reports, and applying the results, see [whole-song lyric matching](lyric-matching.md).

## Resource file

Copy configs/resources.example.json to configs/resources.local.json. Replace the bundle placeholders with the actual folder names. Resource-file paths are relative to that file. Absolute model paths are also accepted.

Use this file for model and pronunciation resources. Pass device and other run settings as script arguments or `Runtime` fields.

Optional fields include tifa_repo, game_repo, g2p_config, dictionaries, and phone_classes. Repository paths default to the uppercase submodule directories in this checkout.

Each selected dictionary requires an explicit phone-type record. Resource mappings are organized by language:

~~~json
{
  "game_model": "../local/models/GAME_BUNDLE/model.pt",
  "dictionaries": {"zh": "../resources/mandarin-dictionary.txt"},
  "phone_classes": {"zh": "../resources/mandarin-phone-types.json"}
}
~~~

Without dictionary overrides, paths are discovered from the forced aligner's G2P configuration. All discovered dictionaries for a language are checked. Type coverage is checked against each dictionary's phone inventory. Phone types come from the explicit records, including for two-part pronunciations.

The [profile catalog](../src/makediffsinger/resources/phone-profiles.json) associates supported dictionaries with checked-in JSON records:

| Dictionary | Language | Phone-type record |
| --- | --- | --- |
| ds-zh-pinyin-lite.txt | zh | [zh-pinyin-lite.json](../src/makediffsinger/resources/zh-pinyin-lite.json) |
| ds_cmudict-07b.txt | en | [en-cmudict-07b.json](../src/makediffsinger/resources/en-cmudict-07b.json) |
| japanese_dict_full.txt | ja | [ja-dictionary-full.json](../src/makediffsinger/resources/ja-dictionary-full.json) |
| jyutping_dict.txt | yue | [yue-jyutping.json](../src/makediffsinger/resources/yue-jyutping.json) |

Each record maps phone strings to consonant, medial, or vowel. An unregistered dictionary requires phone_classes as an explicit JSON file or inline map, including for two-part dictionaries. The record must cover every phone in the selected dictionary. Missing records, missing types, invalid types, and conflicting records fail explicitly; there is no classification fallback. Multiple dictionaries may share a complete explicit record when their symbol meanings agree.

The catalog also records the global AP/SP grouping classes. They mark onset-bearing events and do not impose pitch/rest decisions. The type records define dictionary-specific classes such as Japanese `cl` and English medials. A dictionary override selects the phonology inventory. To change the forced aligner's pronunciation dictionary, update its G2P configuration.

## Runtime controls

Set `--device` to `cpu`, `cuda`, or `cuda:N` to override the device saved in `local/agent-setup.json` for one run. Inference requires a device from one of these sources. A model with an incompatible CUDA kernel or precision can retry on CPU. The stage reports that fallback and leaves the saved choice intact.

Inference scripts also accept precision, batch size, data-loader worker count, and seed. The default precision is 32-true. The seed is reset at each inference request; reproduce the input ordering and batch settings as well.

TF32 is disabled by default for CUDA matrix operations and cuDNN. Enable it explicitly with allow-tf32 if that performance/precision tradeoff is appropriate. This setting is recorded with the runtime parameters.

The initial VED selection is AP. Use `--no-ved` to disable detection for a run. Omit `--game-language` for standard note extraction. This advanced option sets an explicit GAME language condition independently of the input language.

Audio decoding depends on the available decoder resources. The input scanner accepts WAV, FLAC, Opus, MP3, AAC, and Ogg. Opus uses the optional opuscodec package:

~~~bash
python -m pip install -e ".[opus]"
~~~

Other compressed formats may require the decoder used by librosa/audioread. A missing decoder is reported rather than silently dropping a format. Exact PCM replay is supported for PCM WAV source slices; compressed sources are decoded into independent WAV exports.

## Optional TIFA quality plots

To regenerate TIFA's statistics from measured scores, use the initialized TIFA checkout and its inference dependencies. The writer runs without loading model weights.

~~~bash
python -m pip install -e ".[inference]"
~~~

See [testing and compatibility](validation.md) for development and model integration checks.
