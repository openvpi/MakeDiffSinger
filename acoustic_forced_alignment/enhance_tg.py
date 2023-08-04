import pathlib
from collections import defaultdict

import click
import librosa
import numpy as np
import parselmouth as pm
import textgrid as tg
import tqdm


# Given a list of words and a list of phones, regroup the phones that match the words.
# Return a list of lists of phones.
# Example: ['ge', 'chang'], ['g', 'e', 'ch', 'ang'] -> [['g', 'e'], ['ch', 'ang']]
def reconstruct_word_phone(dictionary, words_tier, phones_tier):
    words = [word.mark for word in words_tier if word.mark is not None and word.mark != '']
    phones = [ph.mark for ph in phones_tier if ph.mark is not None and ph.mark != '']
    def helper(i, j, tmp):
        if i >= len(words):
            return True
        candidates = dictionary[words[i]]
        for cand in candidates:
            tj = j
            correct = True
            for ph in cand:
                if phones[tj] != ph:
                    correct = False
                    break
                tj += 1
            if correct:
                tmp.append(cand)
                next_result = helper(i + 1, j + len(cand), tmp)
                if next_result:
                    return next_result
                else:
                    tmp.pop(-1)
        return False
    ret = []
    result = helper(0, 0, ret)
    if not result:
        raise Exception("Cannot match\n" + str(words) + "\n" + str(phones))
    return ret

@click.command(help='Enhance and finish the TextGrids')
@click.option('--wavs', required=True, help='Path to the segments directory')
@click.option('--dictionary', required=True, help='Path to the dictionary file')
@click.option('--src', required=True, help='Path to the raw TextGrids directory')
@click.option('--dst', required=True, help='Path to the final TextGrids directory')
@click.option('--f0_min', type=float, default=40., show_default=True, help='Minimum value of pitch')
@click.option('--f0_max', type=float, default=1100., show_default=True, help='Maximum value of pitch')
@click.option('--br_len', type=float, default=0.1, show_default=True,
              help='Minimum length of breath in seconds')
@click.option('--br_db', type=float, default=-60., show_default=True,
              help='Threshold of RMS in dB for detecting breath')
@click.option('--br_centroid', type=float, default=2000., show_default=True,
              help='Threshold of spectral centroid in Hz for detecting breath')
@click.option('--time_step', type=float, default=0.005, show_default=True,
              help='Time step for feature extraction')
@click.option('--min_space', type=float, default=0.04, show_default=True,
              help='Minimum length of space in seconds')
@click.option('--voicing_thresh_vowel', type=float, default=0.45, show_default=True,
              help='Threshold of voicing for fixing long utterances')
@click.option('--voicing_thresh_breath', type=float, default=0.6, show_default=True,
              help='Threshold of voicing for detecting breath')
@click.option('--br_win_sz', type=float, default=0.05, show_default=True,
              help='Size of sliding window in seconds for detecting breath')
def enhance_tg(
        wavs, dictionary, src, dst,
        f0_min, f0_max, br_len, br_db, br_centroid,
        time_step, min_space, voicing_thresh_vowel, voicing_thresh_breath, br_win_sz
):
    wavs = pathlib.Path(wavs)
    dict_path = pathlib.Path(dictionary)
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = defaultdict(list)
    phoneme_set = set()
    for r in rules:
        phonemes = r[-1].split()
        dictionary[r[0]].append(phonemes)
        phoneme_set.update(phonemes)

    filelist = list(wavs.glob('*.wav'))
    for wavfile in tqdm.tqdm(filelist):
        tgfile = src / wavfile.with_suffix('.TextGrid').name
        textgrid = tg.TextGrid()
        textgrid.read(str(tgfile))
        words = textgrid[0]
        phones = textgrid[1]
        sound = pm.Sound(str(wavfile))
        f0_voicing_breath = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=voicing_thresh_breath,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        ).selected_array['frequency']
        f0_voicing_vowel = sound.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=voicing_thresh_vowel,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        ).selected_array['frequency']
        y, sr = librosa.load(wavfile, sr=24000, mono=True)
        hop_size = int(time_step * sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop_size).squeeze(0)

        try:
            word_phone_map = reconstruct_word_phone(dictionary, words, phones)
        except Exception as e:
            raise Exception(f'Error in {wavfile}: {e}')

        # Fix long utterances
        i = j = word_i = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != '':
                i += 1
                j += len(word_phone_map[word_i])
                # preserve both word and phoneme groups
                word.mark = word.mark + ':' + '_'.join(word_phone_map[word_i])
                word_i += 1
                continue
            if i == 0:
                i += 1
                j += 1
                continue
            prev_word = words[i - 1]
            prev_phone = phones[j - 1]
            # Extend length of long utterances
            while word.minTime < word.maxTime - time_step:
                pos = min(f0_voicing_vowel.shape[0] - 1, int(word.minTime / time_step))
                if f0_voicing_vowel[pos] < f0_min:
                    break
                prev_word.maxTime += time_step
                prev_phone.maxTime += time_step
                word.minTime += time_step
                phone.minTime += time_step
            i += 1
            j += 1

        # Detect aspiration
        i = j = word_i = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != '':
                i += 1
                j += len(word_phone_map[word_i])
                word_i += 1
                continue
            if word.maxTime - word.minTime < br_len:
                i += 1
                j += 1
                continue
            ap_ranges = []
            br_start = None
            win_pos = word.minTime
            while win_pos + br_win_sz <= word.maxTime:
                all_noisy = (f0_voicing_breath[
                             int(win_pos / time_step): int((win_pos + br_win_sz) / time_step)] < f0_min).all()
                rms_db = 20 * np.log10(
                    np.clip(sound.get_rms(from_time=win_pos, to_time=win_pos + br_win_sz), a_min=1e-12, a_max=1))
                # print(win_pos, win_pos + br_win_sz, all_noisy, rms_db)
                if all_noisy and rms_db >= br_db:
                    if br_start is None:
                        br_start = win_pos
                else:
                    if br_start is not None:
                        br_end = win_pos + br_win_sz - time_step
                        if br_end - br_start >= br_len:
                            centroid = spectral_centroid[int(br_start / time_step): int(br_end / time_step)].mean()
                            if centroid >= br_centroid:
                                ap_ranges.append((br_start, br_end))
                        br_start = None
                        win_pos = br_end
                win_pos += time_step
            if br_start is not None:
                br_end = win_pos + br_win_sz - time_step
                if br_end - br_start >= br_len:
                    centroid = spectral_centroid[int(br_start / time_step): int(br_end / time_step)].mean()
                    if centroid >= br_centroid:
                        ap_ranges.append((br_start, br_end))
            # print(ap_ranges)
            if len(ap_ranges) == 0:
                i += 1
                j += 1
                continue
            words.removeInterval(word)
            phones.removeInterval(phone)
            if word.minTime < ap_ranges[0][0]:
                words.add(minTime=word.minTime, maxTime=ap_ranges[0][0], mark=None)
                phones.add(minTime=phone.minTime, maxTime=ap_ranges[0][0], mark=None)
                i += 1
                j += 1
            for k, ap in enumerate(ap_ranges):
                if k > 0:
                    words.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                    phones.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                    i += 1
                    j += 1
                words.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark='AP')
                phones.add(minTime=ap[0], maxTime=min(word.maxTime, ap[1]), mark='AP')
                i += 1
                j += 1
            if ap_ranges[-1][1] < word.maxTime:
                words.add(minTime=ap_ranges[-1][1], maxTime=word.maxTime, mark=None)
                phones.add(minTime=ap_ranges[-1][1], maxTime=phone.maxTime, mark=None)
                i += 1
                j += 1

        # Remove short spaces
        i = j = word_i = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != '':
                i += 1
                j += (1 if word.mark == 'AP' else len(word_phone_map[word_i]))
                if word.mark != 'AP' and word.mark != 'SP':
                    word_i += 1
                continue
            if word.maxTime - word.minTime >= min_space:
                word.mark = 'SP'
                phone.mark = 'SP'
                i += 1
                j += 1
                continue
            if i == 0:
                if len(words) >= 2:
                    words[i + 1].minTime = word.minTime
                    phones[j + 1].minTime = phone.minTime
                    words.removeInterval(word)
                    phones.removeInterval(phone)
                else:
                    break
            elif i == len(words) - 1:
                if len(words) >= 2:
                    words[i - 1].maxTime = word.maxTime
                    phones[j - 1].maxTime = phone.maxTime
                    words.removeInterval(word)
                    phones.removeInterval(phone)
                else:
                    break
            else:
                words[i - 1].maxTime = words[i + 1].minTime = (word.minTime + word.maxTime) / 2
                phones[j - 1].maxTime = phones[j + 1].minTime = (phone.minTime + phone.maxTime) / 2
                words.removeInterval(word)
                phones.removeInterval(phone)
        textgrid.write(str(dst / tgfile.name))


if __name__ == '__main__':
    enhance_tg()
