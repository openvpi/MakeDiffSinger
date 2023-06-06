import os
import csv
from praatio import textgrid
import click
import pathlib

# This assumes you used https://github.com/openvpi/DiffSinger/blob/main/pipelines/no_midi_preparation.ipynb to build your transcriptions
# if you have textgrid files from montreal alignment or labelling that has words and phones in them, this **should** work
# The no_midi_preperation notebook pipeline adds in random SP at start and end of the textgrid data, this is taken into account but I recommend a manual sanity check after running this

# Usage: python textgrid_add_ph_num.py {PATH TO YOUR transcriptions.csv} --textgrid_dir {textgrids DIRECTORY} --split_phones_file {file PATH}

# You can choose if you want to split on gaps by adding htem into your split_phones_file, eg include SP, pau, AP etc.. 

@click.command(help='Add ph_num attribute into transcriptions.csv')
@click.argument('transcription_path', metavar='TRANSCRIPTIONS')
@click.option('--split_phones_file', metavar='FILE', help="Supply a file with phonemes to split phone counts")
@click.option('--textgrid_dir', metavar='FILE')
def add_ph_num(
        transcription_path: str = "transcriptions.csv",
        textgrid_dir: str = "textgrid",
        split_phones_file: str = "vowels.txt",
):

    transcription_new_path = transcription_path + '.new.csv'

    #open csv and textgrid files
    items: list[dict] = []
    with open(transcription_path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            items.append(item)

    split_on_vowel = False
    split_phones = []

    if split_phones_file and len(split_phones_file) > 0 : 
        try:
            split_phones_path = pathlib.Path(split_phones_file).resolve()
        except FileNotFoundError:
            # doesn't exist
            print(f'Split phoneme file {split_phones_file} not found')
        else:
        # exists
            with open(split_phones_path, 'r', encoding='utf8') as f:
                #vowels.update(f.read().split())
                split_phones = f.read().splitlines()
            split_using_file = True

    out_data: list[dict] = []

    for item in items:

        if all_silence_check(item['ph_seq'].split()):

            print(f'ERROR: {item["name"]} has all pauses and silences [{item["ph_seq"]}] please manually check - excluding row from data')

        else:
            item: dict

            if item['name'] and len(item['name']) > 0 :
                textgrid_file = os.path.join(textgrid_dir, item['name'] + '.TextGrid')
                ph_num = []

                if os.path.exists(textgrid_file) :
                    textgrid_data = textgrid.openTextgrid(textgrid_file, True)
                    print('| Processing ' + item['name'])
                    ph_num = generate_ph(textgrid_data, item, split_using_file, split_phones)
                    item['ph_num'] = ' '.join(str(x) for x in ph_num)
                else:
                    print(f'| ERROR: No textgrid {item["name"]}')

                #final check
                ph_dur = [float(d) for d in item['ph_dur'].split()]
                if sum(ph_num) != len(ph_dur):
                    print( f'ph_num {sum(ph_num)} does not sum to number of phones {len(ph_dur)} in {item["name"]}.' )

            out_data.append(item)


    with open(transcription_new_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num'])
        writer.writeheader()
        writer.writerows(out_data)

def is_joinable_silence(phone_str):
    if phone_str == 'pau' or phone_str == 'SP' or phone_str == 'AS':
        return True
    else:
        return False

         
# NOTE: The no_midi_preperation notebook pipeline adds in random SP at start and end of the textgrid data
# This returns a list (ph_num) containing the number of phonemes
def generate_ph(textgrid_data, csv_data, split_using_file, split_phones):
    words = textgrid_data.getTier('words').entries
    phones = textgrid_data.getTier('phones').entries
    csv_phones = csv_data['ph_seq'].split()

    start_diff = 0
    end_diff = 0

    if len(phones) != len(csv_phones) :
        #assuming the missmatch is due to SP at start or end
        num_diff = len(csv_phones) - len(phones)
        if len(csv_phones) > len(phones) and num_diff <= 2:

            if csv_phones[0] != phones[0].label and is_joinable_silence(csv_phones[0]) :
                start_diff = 1
            if csv_phones[-1] != phones[-1].label and is_joinable_silence(csv_phones[-1]) :
                end_diff = 1
        else:
            print(f'ERROR: {csv_data["name"]} has {len(csv_phones)} phones, but {len(phones)} in the TextGrid file, please manually fix')

    if split_using_file:

        ph_num = generate_ph_vowel_num(words,phones,split_phones)
    else:
        ph_num = generate_ph_word_num(words,phones)

    #add in any differneces between the csv and the TextGrid data
    if start_diff > 0:
        ph_num[0] += start_diff
    if end_diff > 0:
        ph_num[-1] += end_diff

    return ph_num

# NOTE: The no_midi_preperation notebook pipeline adds in random SP at start and end of the textgrid data
# This returns a list (ph_num) containing the number of phonemes , split by vowel and word starts 

# In singing, vowels, instead of consonants, are used to align with the beginnings of notes. 
# For this reason, each word should start with a vowel/AP/SP, and end with leading consonant(s) of the next word (if there are any). 
# See the example below:
# 
# text      |   AP   |     shi     |        zhe       |  => word transcriptions (pinyin, romaji, etc.)
# ph_seq    |   AP   |  sh  |  ir  | zh |      e      |  => phoneme sequence
# ph_num    |       2       |     2     |      1      |  => word-level phoneme division
# 
# where sh and zh are consonants, AP, ir and e can be regarded as vowels. There are one special case that a word can start with a consonants: isolated consonants. In this case, all phones in the word are consonants.
def generate_ph_vowel_num(words, phones, split_phones):

    ph_num = []
    phone_index = 0
    phone_count = 0
    ph_labels = []
    vowel_index = []
    skip_next = False

    # each "syllable" gets split by phonemes in split_phones
    for phone in phones:
        phone_count += 1
        ph_labels.append(phone.label)
        if phone.label in split_phones:
            ph_num.append(phone_count)
            phone_count = 0
    
    return ph_num


def generate_ph_word_num(words, phones):
    ph_num = []
    # force_increment is used to track pau,SP,AS

    force_increment = 0 
    # assumption: the order of entries never changes and it the same sequence as in the textgrid file
    for word in words:

        num_phones_in_word = 0
        # force_increment is set by the last word if it was a silence
        for phone in phones:
            if word.start <= phone.start < word.end :
                num_phones_in_word += 1

        num_recorded = num_phones_in_word
        
        #reset for each word, unless its a pause - if the word is a pause we will "add" +1 to the next word's phone count
        if not word.label or word.label == '' or word.label == '' :
            force_increment += 1
        else:
            if force_increment > 0 :
                num_recorded += force_increment
                force_increment = 0
            ph_num.append(num_recorded)
    
    #if we exit the loop on a pause we need to add it in
    if force_increment > 0:
        ph_num.append(force_increment)
    
    return ph_num

def all_silence_check(phones):
    silence_markers = ['pau','spn','SP','AS']
    silence_num = 0
    phones_num = 0
    for phone in phones:
        phones_num += 1
        if phone in silence_markers:
            silence_num += 1
    if phones_num == silence_num:
        return True
    
    return False

if __name__ == '__main__':
    add_ph_num()