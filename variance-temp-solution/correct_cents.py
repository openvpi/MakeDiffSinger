import math
from collections import OrderedDict

import librosa
import numpy as np
import tqdm
import csv
import pathlib

import click

from get_pitch import get_pitch_parselmouth


@click.command(help='Apply cents correction to note sequences')
@click.argument('transcriptions', metavar='TRANSCRIPTIONS')
@click.argument('waveforms', metavar='WAVS')
@click.option('--error_ratio', metavar='RATIO', type=float, default=0.4,
              help='If the percentage of pitch points within a deviation of 50 cents compared to the note label '
                   'is lower than this value, a warning will be raised.')
def correct_cents(
        transcriptions,
        waveforms,
        error_ratio
):
    transcriptions = pathlib.Path(transcriptions).resolve()
    waveforms = pathlib.Path(waveforms).resolve()
    with open(transcriptions, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        items: list[OrderedDict] = []
        for item in reader:
            items.append(OrderedDict(item))

    timestep = 512 / 44100
    for item in tqdm.tqdm(items):
        item: OrderedDict
        note_seq = item['note_seq'].split()
        note_dur = [float(d) for d in item['note_dur'].split()]
        assert len(note_seq) == len(note_dur)

        total_secs = sum(note_dur)
        waveform, _ = librosa.load(waveforms / (item['name'] + '.wav'), sr=44100, mono=True)
        _, f0, _ = get_pitch_parselmouth(waveform, 512, 44100)
        pitch = librosa.hz_to_midi(f0)
        if pitch.shape[0] < total_secs / timestep:
            pad = math.ceil(total_secs / timestep) - pitch.shape[0]
            pitch = np.pad(pitch, [0, pad], mode='constant', constant_values=[0, pitch[-1]])

        start = 0.
        note_seq_correct = []
        for i, (note, dur) in enumerate(zip(note_seq, note_dur)):
            end = start + dur
            if note == 'rest':
                start = end
                note_seq_correct.append('rest')
                continue

            midi = librosa.note_to_midi(note, round_midi=False)
            start_idx = math.floor(start / timestep)
            end_idx = math.ceil(end / timestep)
            note_pitch = pitch[start_idx: end_idx]
            note_pitch_close = note_pitch[(note_pitch >= midi - 0.5) & (note_pitch < midi + 0.5)]
            if len(note_pitch_close) < len(note_pitch) * error_ratio:
                print(f'[{item["name"]}] WARN: possible labeling error in note #{i}.')
                if len(note_pitch_close) == 0:
                    start = end
                    note_seq_correct.append(note)
                    continue
            midi_correct = np.mean(note_pitch_close)
            note_seq_correct.append(librosa.midi_to_note(midi_correct, cents=True, unicode=False))

            start = end

        item['note_seq'] = ' '.join(note_seq_correct)

    with open(transcriptions, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num', 'note_seq', 'note_dur'])
        writer.writeheader()
        writer.writerows(items)


if __name__ == '__main__':
    correct_cents()
