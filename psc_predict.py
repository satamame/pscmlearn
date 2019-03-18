#! python3
# encoding: utf-8

# 予測モデル, 特徴量データを入力として、予測結果 (ラベル) を出力するプログラム。

import sys
import pickle

import psclib.psc as psc
from psclib.extract import Extractor

import numpy as np
import cupy
from chainer import Chain
import pandas as pd


# In[ ]:


args = sys.argv
if len(args) < 5:
    print('Usage: python psc_predict.py model_file, feature_setting_file, script_file, result_save_file')
    sys.exit()
"""
args = [
    "psc_predict.py",
    "model/mdl_0000.pkl",
    "model/mdl_0000_ft.txt",
    "predict/prd_0000_sc.txt",
    "predict/prd_0000_lbl.txt"
]
"""


# In[ ]:


# パラメタを再定義
mdl_file = args[1]
ft_setting_file = args[2]
sc_file = args[3]
lbl_file = args[4]


# In[ ]:


# 入力データをリストに
with open(sc_file, 'r', encoding='utf_8_sig') as f:
    lines = [l.rstrip() for l in f.readlines()]

# 形態素解析
token_lines = psc.tokenize_lines(lines)


# In[ ]:


# 特徴量設定を読み込む
ftels = psc.read_feature_elements(ft_setting_file)

# 特徴量抽出
ex = Extractor(token_lines, ftels)
ft_list = ex.extract()


# In[ ]:


# モデルを読み込む
with open(mdl_file, mode='rb') as f:
    model = pickle.load(f)

# GPU を使うかのフラグ (使うなら 0, 使わないなら -1)
gpu_id = 0

if gpu_id >= 0:
    model.to_gpu(gpu_id)


# In[ ]:


# 特徴量データの ndarray を作成
if gpu_id >= 0:
    in_dataset = cupy.cupy.array(ft_list, dtype=np.float32)
else:
    in_dataset = np.array(ft_list, dtype=np.float32)


# In[ ]:


# 順伝播
out = model(in_dataset)

# 個々の出力の最大値のインデックスをリストにする
out_idx = out.data.argmax(axis = 1).tolist()

# ラベルを表す文字列にする
lbls = [psc.classes[i] for i in out_idx]


# In[ ]:


# 表示用に DataFrame にする
df = pd.DataFrame({"line": [l.rstrip() for l in open(sc_file, 'r', encoding='utf-8')],
                            "label": lbls})


# In[ ]:


print(df)


# In[ ]:




