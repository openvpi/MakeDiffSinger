# Making variance models (temporary solution)

## 1. Clone the repository

```bash
git clone https://github.com/openvpi/DiffSinger.git
cd DiffSinger
git checkout -b variance origin/variance
```

## 2. Convert transcriptions

Assume you have a dataset in your data/ folder as follows:

- MyDataset/
  - raw/
    - wavs/
      - 1.wav
      - 2.wav
      - ...
    - transcriptions.txt

Run:

```bash
python scripts/migrate.py txt data/MyDataset/raw/transcriptions.txt
```

This will generate transcriptions.csv in the same folder as transcriptions.txt, which has three attributes: `name`, `ph_seq` and `ph_dur`.

## 3. Add `ph_num` attribute

The attribute `ph_num` is needed for training the variance models especially if you need to train the phoneme duration predictor. This attribute represents the number of phones that each word contains.