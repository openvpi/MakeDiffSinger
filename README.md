# MakeDiffSinger
Pipelines and tools to build your own DiffSinger dataset.

For the recommended standard dataset making pipelines, see:

- acoustic-forced-alignment: make dataset from scratch with MFA for acoustic model training
- variance-temp-solution: temporary solution to extend acoustic datasets into variance datasets

For other useful pipelines and tools for making a dataset, welcome to raise issues or submit PRs.

## DiffSinger dataset structure

- dataset1/
  - raw/
    - wavs/
      - recording1.wav
      - recording2.wav
      - ...
    - transcriptions.csv
- dataset2/
  - raw/
    - wavs/
      - ...
    - transcriptions.csv
- ...

