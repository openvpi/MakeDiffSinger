import pathlib

import click
import textgrid
import tqdm


@click.command(help='Align words tiers in TextGrids to phones tiers')
@click.option('--tg', required=True, help='Path to TextGrids (2-tier or 3-tier format)')
@click.option('--dictionary', required=True, help='Path to the dictionary file')
@click.option(
    '--out', required=False,
    help='Path to save the aligned TextGrids (defaults to the input directory)'
)
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
def align_tg_words(tg, dictionary, out, overwrite):
    tg_path_in = pathlib.Path(tg)
    dict_path = pathlib.Path(dictionary)
    tg_path_out = pathlib.Path(out) if out is not None else tg_path_in

    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = {
        'SP': ['SP'],
        'AP': ['AP']
    }
    phoneme_set = {'SP', 'AP'}
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
        phoneme_set.update(phonemes)

    for tgfile in tqdm.tqdm(tg_path_in.glob('*.TextGrid')):
        tg = textgrid.TextGrid()
        tg.read(tgfile)
        old_words_tier: textgrid.IntervalTier = tg[-2]
        if old_words_tier.name != 'words':
            raise ValueError(
                f'Invalid tier name or order in \'{tgfile}\'. '
                f'The words tier should be the 1st tier of a 2-tier TextGrid, '
                f'or the 2nd tier of a 3-tier TextGrid.'
            )
        phones_tier: textgrid.IntervalTier = tg[-1]
        new_words_tier = textgrid.IntervalTier(name='words')
        word_seq = [i.mark for i in old_words_tier]
        word_div = [len(dictionary[w]) for w in word_seq]
        ph_dur = [i.maxTime - i.minTime for i in phones_tier]
        assert sum(word_div) == len(ph_dur), tgfile
        assert all(i.mark in phoneme_set for i in phones_tier), tgfile
        start = 0.
        idx = 0
        for j in range(len(word_seq)):
            end = start + sum(ph_dur[idx: idx + word_div[j]])
            new_words_tier.add(minTime=start, maxTime=end, mark=word_seq[j])
            start = end
            idx += word_div[j]
        tg.tiers[-2] = new_words_tier
        tg_file_out = tg_path_out / tgfile.name
        if tg_file_out.exists() and not overwrite:
            raise FileExistsError(str(tg_file_out))
        tg.write(tgfile)


if __name__ == '__main__':
    align_tg_words()
