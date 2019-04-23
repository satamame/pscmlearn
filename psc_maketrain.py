#! python3
# encoding: utf-8

# モデルごとの学習用中間データ (特徴量ファイルと教師ラベルファイル)
# および評価用データ (特徴量ファイルと教師ラベルファイル) を作るプログラム。
# モデル名は、モデルフォルダ (models フォルダのサブフォルダ) の名前。
# 出力される特徴量データは csv 形式。

import sys
import os
import shutil
import csv

from psclib.extract import extract_file


args = sys.argv
if len(args) < 2:
    print('Usage: python psc_maketrain.py model_name')
    sys.exit(1)

# モデル名と特徴量設定ファイル
model_name = args[1]
fts_path = "models/" + model_name + "/mdl_fts.txt"

if not os.path.isfile(fts_path):
    print('{} doesn\'t exist. Terminated.'.format(fts_path))
    sys.exit(1)

for basename in ['train', 'eval']: # 学習用と評価用

    # 番号リストファイル
    list_path = "models/" + model_name + "/ds_" + basename + "_list.txt"

    # ファイルの存在確認
    if not os.path.isfile(list_path):
        print('{} doesn\'t exist. Skipped.'.format(list_path))
        continue

    # 保存先ディレクトリ
    dest_dir = "models/" + model_name + "/" + basename

    # ds_train_list.txt (または ds_eval_list.txt) を読み込み、
    # ファイル名（拡張子なし）リストを作る
    filenames = []
    with open(list_path, 'r', encoding='utf_8_sig') as f:
        for i, line in enumerate(f):
            try:
                num = int(line.strip())
                filenames.append("{:0>6}".format(num))
            except ValueError:
                print('Warning: {}: Line {} is not number. Ignored.'.format(list_path, i+1))

    for fn in filenames:
        # 台本ファイル
        in_file = "dataset/{}.txt".format(fn)

        # ファイルの存在確認
        if not os.path.isfile(in_file):
            print('{} doesn\'t exist. Skipped.'.format(in_file))
            continue

        # 特徴量ファイルを作る
        ft_list = extract_file(in_file, fts_path)
        dest_path = dest_dir + "/{}_ft.csv".format(fn)
        with open(dest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(ft_list)
        
        print('{} created.'.format(dest_path))
        
        # 教師ラベルファイル
        lbl_file = "dataset/{}_lbl.txt".format(fn)

        # ファイルの存在確認
        if not os.path.isfile(lbl_file):
            print('{} doesn\'t exist. Skipped.'.format(lbl_file))
            continue

        # 教師ラベルをコピー
        dest_path = dest_dir + "/{}_lbl.txt".format(fn)
        shutil.copyfile(lbl_file, dest_path)
        
        print('{} created.'.format(dest_path))
