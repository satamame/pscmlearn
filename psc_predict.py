#! python3
# encoding: utf-8

# 予測モデル, 特徴量データを入力として、予測結果 (ラベル) を出力するプログラム。

import sys
import pickle

import psclib.psc as psc

import numpy as np
from chainer import Chain
import chainer.links as L
import chainer.functions as F
import cupy


# In[ ]:


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_predict.py model_file, feature_file, result_file')
    sys.exit()
"""
args = [
    "psc_predict.py",
    "model/mdl_0000.pkl",
    "predict/prd_0000_ft.txt",
    "predict/prd_0000_lbl.txt"
]
"""


# In[ ]:


mdl_file = args[1]
ft_file = args[2]
lbl_file = args[3]


# In[ ]:


# モデルを読み込む
with open(mdl_file, mode='rb') as f:
    model = pickle.load(f)

# GPU を使う (使うなら 0, 使わないなら -1)
gpu_id = -1

if gpu_id >= 0:
    model.to_gpu(gpu_id)


# In[ ]:


# 特徴量データのリストを作成
in_fts = []
for line in open(ft_file, 'r'):
    fts = [float(f) for f in line.split(',')]
    in_fts.append(fts)

if gpu_id >= 0:
    in_dataset = cupy.cupy.array(in_fts, dtype=np.float32)
else:
    in_dataset = np.array(in_fts, dtype=np.float32)


# In[ ]:


print(type(in_dataset))
print(in_dataset.shape)


# In[ ]:


# 順伝播の結果を得て、ラベルとして保存
out = model(in_dataset)
# 個々の出力の最大値のインデックスをリストにする
out_idx = out.data.argmax(axis = 1).tolist()
# ラベルを表す文字列にする
lbls = [psc.classes[i] for i in out_idx]


# In[ ]:


print(lbls)


# In[ ]:




