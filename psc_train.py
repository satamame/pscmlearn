#! python3
# encoding: utf-8

# 学習用データ, 評価用データを入力として、学習した予測モデルを出力するプログラム。

import sys
import csv
# import random # 不要か？
import pickle

import psclib.psc as psc

import numpy as np
import chainer
from chainer import Link, ChainList, Variable
# from chainer import Chain
from chainer import iterators, optimizers
# import chainer.links as L
import chainer.functions as F
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu


# In[ ]:


args = sys.argv
if len(args) < 4:
    print('Usage: python psc_train.py train_list_file, eval_list_file, model_file')
    sys.exit()
"""
args = [
    "psc_train.py",
    "ds_train_list.txt",
    "ds_eval_list.txt",
    "model/mdl_0000.pkl"
]
"""


# In[ ]:


train_list_file = args[1]
eval_list_file = args[2]
model_save_file = args[3]


# In[ ]:


def make_dataset(list_file):
    """
    特徴量データと教師ラベルが対になったデータセットを作成

    Parameters
    ----------
    list_file : string
        入力データのリストのファイル名
        各行が、"特徴量データのファイル名, 教師ラベルのファイル名" の形式
    """

    # 「特徴量データ, 教師ラベル」 のファイル名リストを読み込む
    ft_files = []
    lbl_files = []
    try:
        for line in open(list_file, 'r'):
            ft, lbl = line.split(',')
            ft_files.append(ft.strip())
            lbl_files.append(lbl.strip())
    except IOError as err:
        print(err)
        sys.exit()
    except ValueError as err:
        print(err)
        sys.exit()

    # 特徴量データのリストを作成
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
    except ValueError as err:
        print(err)
        sys.exit()

    dataset = list(zip(in_fts, in_lbls))
    return dataset


# In[ ]:


ds_train = make_dataset(train_list_file)
ds_eval = make_dataset(eval_list_file)

# __TODO__ BOM があるとエラーになるので何とかする。


# In[ ]:


print(len(ds_train))
print(len(ds_eval))


# In[ ]:


# 特徴量データと教師ラベルを対にして、学習用と検証用に分ける
train_count = int(len(ds_train) * 0.8) # 8割のデータを学習用に
ds_train, ds_valid = split_dataset_random(ds_train, train_count, seed=0)


# In[ ]:


print(len(ds_train))
print(len(ds_valid))


# In[ ]:


# モデル定義
hid_dim = 20                # 隠れ層のノード数 : いい塩梅に決める
out_dim = len(psc.classes)  # 出力層のノード数 : 定義されているラベルの数

model = psc.PscChain(hid_dim, out_dim)


# In[ ]:


# バッチサイズ
batch_size = 100

# 学習用イテレータ
train_iter = iterators.SerialIterator(ds_train, batch_size)
# 検証用イテレータ
valid_iter = iterators.SerialIterator(ds_valid, batch_size, repeat=False, shuffle=False)

# Optimizer の setup
optimizer = optimizers.SGD(lr=0.01).setup(model)

# GPU を使う (使うなら 0, 使わないなら -1)
gpu_id = 0

if gpu_id >= 0:
    model.to_gpu(gpu_id)


# In[ ]:


# エポック数
max_epoch = 40

while train_iter.epoch < max_epoch:
    # イテレーション
    train_batch = train_iter.next()
    
    # 1 イテレーション分の、入力データの array と、教師ラベルの array
    x, t = concat_examples(train_batch, gpu_id)
    x = x.astype(np.float32)
    
    print(type(x))
    print(x.shape)
    
    # 順伝播の結果を得る
    y = model(x)
    
    # ロスの計算
    loss = F.softmax_cross_entropy(y, t)
    
    # 勾配の計算
    model.cleargrads()
    loss.backward()
    
    # パラメータの更新
    optimizer.update()
    
    if train_iter.is_new_epoch:  # 1 epochが終わったら

        # ロスの表示
        # print('epoch:{:02d} train_loss:{:.04f} '.format(
        #    train_iter.epoch, float(to_cpu(loss.data))), end='')
        
        valid_losses = []
        valid_accuracies = []
        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = concat_examples(valid_batch, gpu_id)
            x_valid = x_valid.astype(np.float32)

            # Validationデータをforward
            with chainer.using_config('train', False),                     chainer.using_config('enable_backprop', False):
                y_valid = model(x_valid)
                
            # print(y_valid)

            # ロスを計算
            loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))

            # 精度を計算
            accuracy = F.accuracy(y_valid, t_valid)
            accuracy.to_cpu()
            valid_accuracies.append(accuracy.array)

            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break

        print('{:0=2} val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            train_iter.epoch, np.mean(valid_losses), np.mean(valid_accuracies)))


# In[ ]:


# 評価用イテレータ
eval_iter = iterators.SerialIterator(ds_eval, batch_size, repeat=False, shuffle=False)


# In[ ]:


# 評価用データでの評価
eval_accuracies = []
while True:
    eval_batch = eval_iter.next()
    x_eval, t_eval = concat_examples(eval_batch, gpu_id)
    x_eval = x_eval.astype(np.float32)
    
    # print(x_eval)

    # 評価用データをforward
    with chainer.using_config('train', False),             chainer.using_config('enable_backprop', False):
        y_eval = model(x_eval)
    
    # print(y_eval)
    
    """
    for i, a in enumerate(y_eval):
        print(a, t_eval[i])
    """
    # 精度を計算
    accuracy = F.accuracy(y_eval, t_eval)
    accuracy.to_cpu()
    eval_accuracies.append(accuracy.array)

    if eval_iter.is_new_epoch:
        eval_iter.reset()
        break

print('eval_accuracy:{:.04f}'.format(np.mean(eval_accuracies)))


# In[ ]:


# モデルを保存する (保存するときは CPU 版とする)
with open(model_save_file, 'wb') as f:
    pickle.dump(model.to_cpu(), f)
print("Model is saved as {}.".format(model_save_file))


# In[ ]:




