#! python3
# encoding: utf-8

# 特徴量データ + 教師ラベルのリストと、特徴量設定データを入力として、モデル + 特徴量設定データを出力するプログラム。

import sys
import csv
import psclib.psc as psc


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_train.py in-filename feature-elements-filename ' + \
        'out-filename')
    sys.exit()

in_filename = args[1]

# 「特徴量データ, 教師ラベル」 のファイル名リストを読み込む
ft_files = []
lbl_files = []
try:
    for line in open(in_filename, 'r'):
        ft, lbl = line.split(',')
        ft_files.append(ft.strip())
        lbl_files.append(lbl.strip())
except IOError as err:
    print(err)
    sys.exit()
except ValueError as err:
    print(err)
    sys.exit()

# 学習用の特徴量データのリストを作成
in_fts = []
for ft_file in ft_files:
    for line in open(ft_file, 'r'):
        fts = [float(f) for f in line.split(',')]
        in_fts.append(fts)

# 教師ラベル (数値に変換したもの) のリストを作成
in_lbls = []
try:
    for lbl_file in lbl_files:
        for line in open(lbl_file, 'r'):
            in_lbls.append(psc.classes.index(line.strip()))

# 特徴量データと教師ラベルを対にしたままシャッフル
z = list(zip(in_fts, in_lbls))
random.shuffle(z)
in_fts, in_lbls = list(zip(*z))

