#! python3
# encoding: utf-8

# モデルごとの検証用データ (特徴量ファイルと教師ラベルファイル) を作るプログラム。
# モデル名は、モデルフォルダ (models フォルダのサブフォルダ) の名前。
# 
# 出力される特徴量データは csv 形式。

import sys
import shutil
import csv

from psclib.extract import extract_file


args = sys.argv
if len(args) < 2:
    print('Usage: python psc_makeeval.py model_folder')
    sys.exit()

list_path = "models/" + args[1] + "/ds_eval_list.txt"
fts_path = "models/" + args[1] + "/mdl_fts.txt"
dest_dir = "models/" + args[1] + "/eval"


# ds_eval_list.txt を読み込み、ファイル名（拡張子なし）リストを作る
filenames = []
with open(list_path, 'r', encoding='utf_8_sig') as f:
    for i, line in enumerate(f):
        try:
            num = int(line.strip())
            filenames.append("{:0>6}".format(num))
        except ValueError:
            print('Warning: {}: Line {} is not number. Ignored.'.format(list_path, i+1))

for fn in filenames:
    # 特徴量ファイルを作る
    in_file = "dataset/{}.txt".format(fn)
    ft_list = extract_file(in_file, fts_path)
    dest_path = dest_dir + "/{}_ft.csv".format(fn)
    with open(dest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ft_list)
    
    print('{} created.'.format(dest_path))
    
    # 教師ラベルをコピー
    lbl_file = "dataset/{}_lbl.txt".format(fn)
    dest_path = dest_dir + "/{}_lbl.txt".format(fn)
    shutil.copyfile(lbl_file, dest_path)
    
    print('{} created.'.format(dest_path))
