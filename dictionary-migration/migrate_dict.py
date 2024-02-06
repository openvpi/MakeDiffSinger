import csv
import json
import pathlib
from typing import Dict, List

import click
import textgrid
import tqdm

from trie import Trie


def read_dictionary(path: pathlib.Path) -> Dict[str, List[str]]:
    dictionary = {}
    with open(path, 'r', encoding='utf8') as fr:
        for line in fr:
            syllable, phones = line.strip().split('\t')
            dictionary[syllable] = phones.split()
    return dictionary


def verify_dictionaries(old_dict: Dict[str, List[str]], new_dict: Dict[str, List[str]]):
    for syllable, old_phone_split in old_dict:
        assert syllable in new_dict, (
            f"The new dictionary does not contain syllable "
            f"\'{syllable}\' in the old dictionary."
        )
        new_phone_split = new_dict[syllable]
        assert len(old_phone_split) == len(new_phone_split), (
            f"The new dictionary does not map the syllable \'{syllable}\' "
            f"to the same number of phonemes as the old dictionary "
            f"({old_phone_split} vs. {new_phone_split})."
        )


def migrate_phones(
        phones: List[str],
        old_dict_trie: Trie,
        new_dict: Dict[str, List[str]],
) -> List[str]:
    new_phones = []
    leaf = old_dict_trie
    cur_path = []
    for ph in phones:
        leaf = leaf.forward(ph, ignore=True)
        cur_path.append(ph)
        if leaf is None:
            new_phones.extend(cur_path)
            leaf = old_dict_trie
            cur_path.clear()
        else:
            syllable = leaf.value()
            if syllable is not None:
                new_phone_split = new_dict[syllable]
                new_phones.extend(new_phone_split)
                leaf = old_dict_trie
                cur_path.clear()
    new_phones.extend(cur_path)
    return new_phones


@click.group(help="Migrate labels from a old dictionary to a new dictionary.")
def main():
    pass


@main.command(name='csv', help="Migrate single transcriptions.csv.")
@click.argument(
    'old_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'new_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'input_csv',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'output_csv',
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=pathlib.Path)
)
@click.option(
    '--overwrite',
    is_flag=True,
    help="Overwrite existing file."
)
def migrate_dict_csv(
        old_dict: pathlib.Path,
        new_dict: pathlib.Path,
        input_csv: pathlib.Path,
        output_csv: pathlib.Path,
        overwrite=False
):
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"The output CSV file \'{output_csv}\' already exists. Try --overwrite to overwrite it.")
    old_dict = read_dictionary(old_dict)
    new_dict = read_dictionary(new_dict)
    trie = Trie()
    for key, value in old_dict.items():
        trie.store(value, key)
    with open(input_csv, 'r', encoding='utf8') as fr:
        reader = csv.DictReader(fr)
        with open(output_csv, 'w', encoding='utf8', newline='') as fw:
            writer = csv.DictWriter(fw, fieldnames=reader.fieldnames)
            writer.writeheader()
            for label in reader:
                label: dict
                old_phones = label['ph_seq'].split()
                new_phones = migrate_phones(old_phones, trie, new_dict)
                label['ph_seq'] = ' '.join(new_phones)
                writer.writerow(label)


@main.command(name='tg', help="Migrate all TextGrid files in the given directory.")
@click.argument(
    'old_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'new_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'input_tg',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'output_tg',
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=pathlib.Path)
)
@click.option(
    '--overwrite',
    is_flag=True,
    help="Overwrite existing files."
)
def migrate_dict_tg(
        old_dict: pathlib.Path,
        new_dict: pathlib.Path,
        input_tg: pathlib.Path,
        output_tg: pathlib.Path,
        overwrite=False
):
    if output_tg.exists():
        for file in input_tg.glob('*.TextGrid'):
            new_file = output_tg / file.name
            if new_file.exists() and not overwrite:
                raise FileExistsError(
                    f"The output TextGrid file \'{new_file}\' already exists. Try --overwrite to overwrite it.")
    else:
        output_tg.mkdir(parents=True, exist_ok=True)

    old_dict = read_dictionary(old_dict)
    new_dict = read_dictionary(new_dict)
    trie = Trie()
    for key, value in old_dict.items():
        trie.store(value, key)
    for file in tqdm.tqdm(input_tg.glob('*.TextGrid')):
        tg = textgrid.TextGrid()
        tg.read(file, encoding='utf8')
        phones_tier = None
        for tier in tg.tiers:
            if isinstance(tier, textgrid.IntervalTier) and tier.name == 'phones':
                phones_tier = tier
                continue
        assert phones_tier is not None, f"There are no phones tier found in \'{file}\'."
        old_phones = [i.mark for i in phones_tier]
        new_phones = migrate_phones(old_phones, trie, new_dict)
        for interval, phone in zip(phones_tier, new_phones):
            interval.mark = phone
        with open(output_tg / file.name, 'w', encoding='utf8') as fw:
            tg.write(fw)


@main.command(name='ds', help="Migrate all DS files in the given directory.")
@click.argument(
    'old_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'new_dict',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'input_ds',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=pathlib.Path)
)
@click.argument(
    'output_ds',
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=pathlib.Path)
)
@click.option(
    '--overwrite',
    is_flag=True,
    help="Overwrite existing files."
)
def migrate_dict_ds(
        old_dict: pathlib.Path,
        new_dict: pathlib.Path,
        input_ds: pathlib.Path,
        output_ds: pathlib.Path,
        overwrite=False
):
    if output_ds.exists():
        for file in input_ds.glob('*.ds'):
            new_file = output_ds / file.name
            if new_file.exists() and not overwrite:
                raise FileExistsError(
                    f"The output DS file \'{new_file}\' already exists. Try --overwrite to overwrite it.")
    else:
        output_ds.mkdir(parents=True, exist_ok=True)

    old_dict = read_dictionary(old_dict)
    new_dict = read_dictionary(new_dict)
    trie = Trie()
    for key, value in old_dict.items():
        trie.store(value, key)
    for file in tqdm.tqdm(input_ds.glob('*.ds')):
        with open(file, 'r', encoding='utf8') as fr:
            ds = json.load(fr)
            if not isinstance(ds, list):
                ds = [ds]
        for seg in ds:
            old_phones = seg['ph_seq'].split()
            new_phones = migrate_phones(old_phones, trie, new_dict)
            seg['ph_seq'] = ' '.join(new_phones)
        with open(output_ds / file.name, 'w', encoding='utf8') as fw:
            json.dump(ds, fw, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
