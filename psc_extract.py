#! python3
# encoding: utf-8

# 形態素データと特徴量設定データを入力として、行ごとの特徴ベクトルを出力するプログラム。

import sys
import re
import json
import psclib.psc as psc
from psclib.extract import Extractor

# すべてのファイルを閉じる関数。
def close_all():
    if 'fout' in globals():
        fout.close()
    if 'fin' in globals():
        fin.close()


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_extract.py in-filename feature-elements-filename ' + \
        'out-filename')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8') # Morphologically analyzed script.
    fout = open(args[3], 'w')             # Output of features.
except IOError as err:
    print(err)
    close_all()
    sys.exit()

# Load feature elements to be used.
ftels = psc.make_feature_elements(args[2])

# print ('Feature elements : {0}'.format(ftels))

ex = Extractor(json.load(fin), ftels)
ex.extract(fout)
close_all()
