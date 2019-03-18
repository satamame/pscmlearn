#! python3
# encoding: utf-8

# 学習の前処理として、中間データ（特徴量と教師ラベル）を作るプログラム。
# 特徴量データは、csv 形式。

import sys
import csv

import psclib.psc as psc
from psclib.extract import Extractor


# In[ ]:


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_extract.py script_file feature_setting_file feature_save_file [label_save_file]')
    sys.exit()
"""
args = [
    "psc_extract.py",
    "train/tr_0000_sc.txt",
    "model/mdl_0000_ft.txt",
    "train/tr_0000_ft.csv",
    "train/tr_0000_lbl.txt"
]
"""


# In[ ]:


# パラメタを再定義
sc_file = args[1]
ft_setting_file = args[2]
ft_save_file = args[3]
if len(args) >= 5:
    lbl_save_file = args[4]
else:
    lbl_save_file = ""


# In[ ]:


# 入力データをリストに
with open(sc_file, 'r', encoding='utf_8_sig') as f:
    lines = [l.rstrip() for l in f.readlines()]

# 教師ラベルを分離・保存
if lbl_save_file != "":
    data, labels = psc.isolate_labels_from_lines(lines)
    with open(lbl_save_file, 'w') as f:
        for l in labels:
            f.write(l + '\n')
    print("Training labels are saved as {}.".format(lbl_save_file))
else:
    data = lines


# In[ ]:


# 形態素解析
token_lines = psc.tokenize_lines(data)

# 特徴量の設定を読み込む
ftels = psc.read_feature_elements(ft_setting_file)

# 特徴抽出・保存
ex = Extractor(token_lines, ftels)
ft_list = ex.extract()

with open(ft_save_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ft_list)

print("Feature data is saved as {}.".format(ft_save_file))


# In[ ]:




