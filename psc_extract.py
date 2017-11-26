#! python3
# encoding: utf-8

# 形態素データと特徴量定義データを入力として、行ごとの特徴ベクトルを出力するプログラム。

import sys
import re
import json
from psc import psc


# すべてのファイルを閉じる関数。
def close_all():
    if 'fout' in globals():
        fout.close()
    if 'ffeat' in globals():
        ffeat.close()
    if 'fin' in globals():
        fin.close()


class Extractor:
    def __init__(self, lines, ftels):
        self.lines = lines
        self.ftels = ftels
    def extract(self, fout):
        for lnum in range(len(self.lines)):
            fout.write(ex.extract_line(lnum) + '\n')
    def extract_line(self, lnum):
        feature_vec = []
        line = self.lines[lnum]
        for ftname in [ftel[0] for ftel in ftels]:
            if ftname == 'sc_count_of_lines':
                ft = str(len(self.lines))
            elif ftname == 'sc_count_of_lines_with_bracket':
                ft = str(self.get_count_of_lines_with_bracket())
            elif ftname == 'ln_count_of_words':
                ft = str(len(line['tokenized_words']))
            elif ftname == 'ln_count_of_brackets':
                ft = str(sum(word['base_form'] == '「' \
                    for word in line['tokenized_words']))
            feature_vec.append(ft)
        return ",".join(feature_vec)
    
    def get_count_of_lines_with_bracket(self):
        if hasattr(self, 'count_of_lines_with_bracket'):
            return self.count_of_lines_with_bracket
        count = 0
        for line in self.lines:
            words = [w['surface'] for w in line['tokenized_words']]
            for w in words:
                if w in psc.brackets:
                    count += 1
                    break
        self.count_of_lines_with_bracket = count
        return count


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_extract.py in-filename feature-elements-filename ' + \
        'out-filename')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8') # Morphologically analyzed script
    ffeat = open(args[2]) # Feature vector definition
    fout = open(args[3], 'w') # Output of features
except IOError as err:
    print(err)
    close_all()
    sys.exit()

# Load feature elements to be used
ftels = psc.make_feature_elements(ffeat)

# print ('Feature elements : {0}'.format(ftels))

ex = Extractor(json.load(fin), ftels)
ex.extract(fout)
close_all()
