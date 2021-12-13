# -*- coding: utf-8 -*-

"""
    @file   ksponspeech_preparation.py
    @date   2021-11-23
    @author Hyunsoo Son
    @brief  data preparation for ksponspeech

    @usage  python3 ksponspeech_preparation.py [RAW_DATA_DIRECTORY] [TRAIN_DIRECTORY] [EVAL_CLEAN_DIRECTORY] [EVAL_OTHER_DIRECTORY]
"""

import sys
import re

transcription_rule = 'b/ |l/ |o/ |n/ |b/|l/|o/|n/|\+|\?|\.|,|\*|\!|/|u'

train = sys.argv[1] + '/KsponSpeech_scripts/train.trn'
eval_clean = sys.argv[1] + '/KsponSpeech_scripts/eval_clean.trn'
eval_other = sys.argv[1] + '/KsponSpeech_scripts/eval_other.trn'

train_dir = sys.argv[2]
eval_clean_dir = sys.argv[3]
eval_other_dir = sys.argv[4]

train_dict = {}
eval_clean_dict = {}
eval_other_dict = {}

# make wav.scp and text file
def data_prep(trn: str, path: str, d: dict) -> None:

    with open(trn, 'r', encoding='utf-8') as raw, \
            open(path + '/wav.scp', 'w', encoding='utf-8') as wav, \
            open(path + '/text', 'w', encoding='utf-8') as text:
        lines = raw.readlines()

        for line in lines:
            key, value = line.strip('\n').split(' :: ')

            # pre-processing by ETRI transcription rule for korean speech.
            # sentences containing English are not used for training.
            value = re.sub('\((.+?)\)/\((.+?)\)', '\\2', value)
            value = re.sub(transcription_rule, '', value)

            if (re.search('[^ 가-힣]', value) == None):
                d[key.split('/')[-1][:-4]] = ('sox -t raw -r 16000 -e signed -b 16 -c 1 ' + sys.argv[1] +\
                     '/' + key + ' -t wav - |', value)

        for key in sorted(list(d.keys())):
            wav.write(key + ' ' + d[key][0] + '\n')
            text.write(key + ' ' + d[key][1] + '\n')

        print(sys.argv[0] + ": %d of %d utterances are used from %s" \
              %(len(d), len(lines), trn))

def main():

    print(sys.argv[0] + ": preparation for train set")
    data_prep(train, train_dir, train_dict)

    print(sys.argv[0] + ": preparation for eval-clean")
    data_prep(eval_clean, eval_clean_dir, eval_clean_dict)

    print(sys.argv[0] + ": preparation for eval-other")
    data_prep(eval_other, eval_other_dir, eval_other_dict)

if __name__ == "__main__":
    main()
