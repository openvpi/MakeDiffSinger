import os
import csv
from praatio import textgrid
import click

# This assumes you used https://github.com/openvpi/DiffSinger/blob/main/pipelines/no_midi_preparation.ipynb to build your transcriptions
# if you have textgrid files from montreal alignment or labelling that has words and phones in them, this **should** work
# The no_midi_preperation notebook pipeline adds in random SP at start and end of the textgrid data, this is taken into account but I recommend a manual sanity check after running this


@click.command(help='Add ph_num attribute into transcriptions.csv')
@click.argument('transcription_path', metavar='TRANSCRIPTIONS')
@click.option('--textgrid_dir', metavar='FILE')
def add_ph_num(
        transcription_path: str = "transcriptions.csv",
        textgrid_dir: str = "textgrids",
):
def add_ph_num():

    transcription_new_path = transcription_path + '.new.csv'

    #open csv and textgrid files
    items: list[dict] = []
    with open(transcription_path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            items.append(item)

    for item in items:
        item: dict
        if item['name'] and len(item['name']) > 0 :
            textgrid_file = os.path.join(textgrid_dir, item['name'] + '.TextGrid')
            ph_num = []
            if os.path.exists(textgrid_file) :
                textgrid_data = textgrid.openTextgrid(textgrid_file, True)
                ph_num = generate_ph_word_num(textgrid_data,item)
                item['ph_num'] = ' '.join(str(x) for x in ph_num)
            else:
                print('>>> NO textgrid',item['name'])

            #final check
            ph_dur = [float(d) for d in item['ph_dur'].split()]
            if sum(ph_num) != len(ph_dur):
                print( f'ph_num {sum(ph_num)} does not sum to number of phones {len(ph_dur)} in {item["name"]}.' )


    with open(transcription_new_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur', 'ph_num'])
        writer.writeheader()
        writer.writerows(items)

def is_joinable_silence(phone_str):
    if phone_str == 'pau' or phone_str == 'SP' or phone_str == 'AS':
        return True
    else:
        return False


          
# NOTE: The no_midi_preperation notebook pipeline adds in random SP at start and end of the textgrid data
def generate_ph_word_num(textgrid_data, csv_data):
    words = textgrid_data.getTier('words').entries
    phones = textgrid_data.getTier('phones').entries
    csv_phones = csv_data['ph_seq'].split()

    start_diff = 0
    end_diff = 0

    if len(phones) != len(csv_phones) :
        #assuming the missmatch is due to SP at start or end
        num_diff = len(csv_phones) - len(phones)
        if len(csv_phones) > len(phones) and num_diff <= 2:
            #if num_diff == 1
            if csv_phones[0] != phones[0].label and is_joinable_silence(csv_phones[0]) :
                start_diff = 1
            if csv_phones[-1] != phones[-1].label and is_joinable_silence(csv_phones[-1]) :
                end_diff = 1
        else:
            print(f'ERROR: {csv_data["name"]} has {len(csv_phones)} phones, but {len(phones)} in the TextGrid file, please manually fix')

    word_ph_nums = []
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

    #add in any differneces between the csv and the TextGrid data
    if start_diff > 0:
        ph_num[0] += start_diff
    if end_diff > 0:
        ph_num[-1] += end_diff
    return ph_num

if __name__ == '__main__':
    add_ph_num()