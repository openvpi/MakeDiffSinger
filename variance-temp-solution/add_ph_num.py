import click
import csv

def find_ph_num(i, phonemes_split, dict):
    ph_tmp = []
    left = i
    right = i
    for j in range(i, len(phonemes_split)):
        ph_tmp.append(phonemes_split[j])
        if ph_tmp in dict.values():
            right = j
    return left, right



@click.command()
@click.option('--csv_path',required = True, help='Path to CSV file')
@click.option('--dictionary',required = True, help='Path to dictionary file')
def add_ph_num(csv_path,dictionary):
    ph_seq_index = 1
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        phonemes_tmp = []
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            phonemes_tmp.append(row[ph_seq_index])
    
    dict = {}
    with open(dictionary, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            key = line.split('\t')[0]
            values = (line.split('\t')[1]).split(' ')
            dict.update({key: values})

    ph_num = []
    for phonemes in phonemes_tmp:
        tmp = []
        phonemes_split = phonemes.split(' ')
        i=0
        while i < len(phonemes_split):
            if phonemes_split[i] == "AP" or phonemes_split[i] == "SP":
                tmp.append("1")
                i+=1
            else:
                left,right = find_ph_num(i,phonemes_split,dict)
                tmp.append(str(right-left+1))
                i = right+1

        ph_num.append(tmp)

    ph_num_str = []
    for i in ph_num:
        string = ' '.join(i)
        ph_num_str.append(string)

    ph_num_str[0] = "ph_num"

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    for i, value in enumerate(ph_num_str):
        rows[i].append(value)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

if __name__ == '__main__':
    add_ph_num()
