#! python3
# encoding: utf-8

# モデルを指定して、学習した予測モデルを保存するプログラム。
# モデル名は、モデルフォルダ (models フォルダのサブフォルダ) の名前。
# 標準出力に、評価用データで評価した結果を出力する。

import sys
import pickle
import numpy as np

import psclib.psc as psc
from psclib.chain import PscChain

from chainer import optimizers
from chainer.cuda import to_cpu, to_gpu

args = sys.argv
if len(args) < 2:
    print('Usage: python psc_train.py model_folder')
    sys.exit(1)
"""
args = [
    "psc_train.py",
    "mdl000"
]
"""

# モデル名
model_name = args[1]

# 学習用番号リストファイル
train_list_path = "models/" + model_name + "/ds_train_list.txt"
# 学習用データセット
ds_train = psc.make_dataset(model_name, 'train')

# モデル定義
hid_dim = 20                # 隠れ層のノード数 : いい塩梅に決める
out_dim = len(psc.classes)  # 出力層のノード数 : 定義されているラベルの数
model = PscChain(hid_dim, out_dim)

# Optimizer の setup
optimizer = optimizers.SGD(lr=0.01).setup(model)
# GPU を使う
gpu_id=0

print("Learning with dataset from list: {}".format(train_list_path))

# 学習
model.train(dataset=ds_train, batch_size=100, max_epoch=20, optimizer=optimizer, gpu_id=gpu_id)

# 評価用番号リストファイル
eval_list_path = "models/" + model_name + "/ds_eval_list.txt"
# 評価用データセット
ds_eval = psc.make_dataset(model_name, 'eval')

print("Evaluating with dataset from list: {}".format(eval_list_path))

# 評価して精度を得る
accuracies = model.evaluate(dataset=ds_eval, batch_size=100, gpu_id=gpu_id)

print('accuracy:{:.04f}'.format(np.mean(accuracies)))

# モデルを保存する (保存するときは CPU 版とする)
model_save_file = "models/" + model_name + "/mdl.pkl"
with open(model_save_file, 'wb') as f:
    pickle.dump(model.to_cpu(), f)
    
print("Model is saved as {}.".format(model_save_file))
