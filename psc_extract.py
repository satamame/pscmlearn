#! python3
# encoding: utf-8

# 形態素データと特徴量定義データを入力として、行ごとの特徴ベクトルを出力するプログラム。

import sys
import re
import json
import psclib.psc as psc
import psclib.extractor as pscext

# すべてのファイルを閉じる関数。
def close_all():
    if 'fout' in globals():
        fout.close()
    if 'ffeat' in globals():
        ffeat.close()
    if 'fin' in globals():
        fin.close()


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

ex = pscext.Extractor(json.load(fin), ftels)
ex.extract(fout)
close_all()
