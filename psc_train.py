#! python3
# encoding: utf-8

# In[ ]:


# 特徴量データ + 教師ラベルのリストを入力として、予測モデルを出力するプログラム。

import sys
import csv
import random

import psclib.psc as psc

import numpy as np
import chainer
from chainer import Link, Chain, ChainList, Variable
from chainer import iterators, optimizers
import chainer.links as L
import chainer.functions as F
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu


# In[ ]:



args = sys.argv
if len(args) < 4:
    print('Usage: python psc_train.py train_list_file, test_list_file, model_file')
    sys.exit()

"""
args = [
    "psc_train.py",
    "ds_train_list.txt",
    "ds_test_list.txt",
    "model/mdl_0000.pkl"
]
"""

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


ds_train = make_dataset(args[1])
ds_test = make_dataset(args[2])

# __TODO__ BOM があるとエラーになるので何とかする。


# In[ ]:


print(len(ds_train))
print(len(ds_test))


# In[ ]:


# クラスは、あとで psc パッケージに移す
class PscChain(Chain):
    def __init__(self, hid_dim, out_dim):
        """
        初期化メソッド

        Parameters
        ----------
        hid_dim : integer
            隠れ層のノード数
        out_dim : integer
            出力層のノード数
        """
        super().__init__(
            l1=L.Linear(None, hid_dim),
            l2=L.Linear(hid_dim, hid_dim),
            l3=L.Linear(hid_dim, out_dim)
        )
    
    def __call__(self, x):
        """
        順伝播して、出力層 (Variable) を返す

        Parameters
        ----------
        x : Variable
            (バッチサイズ x 特徴ベクトルの次元数) の、入力データ
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

        # ここで softmax して、確率にして返すこと。


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

model = PscChain(hid_dim, out_dim)


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
max_epoch = 100

while train_iter.epoch < max_epoch:
    # イテレーション
    train_batch = train_iter.next()
    
    # 1 イテレーション分の、入力データの array と、教師ラベルの array
    x, t = concat_examples(train_batch, gpu_id)
    x = x.astype(np.float32)
    
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

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(valid_losses), np.mean(valid_accuracies)))


# In[ ]:


# 評価用イテレータ
test_iter = iterators.SerialIterator(ds_test, batch_size, repeat=False, shuffle=False)


# In[ ]:


# テストデータでの評価
test_accuracies = []
while True:
    test_batch = test_iter.next()
    x_test, t_test = concat_examples(test_batch, gpu_id)
    x_test = x_test.astype(np.float32)
    
    # print(x_test)

    # テストデータをforward
    with chainer.using_config('train', False),             chainer.using_config('enable_backprop', False):
        y_test = model(x_test)
    
    # print(y_test)
    
    """
    for i, a in enumerate(y_test):
        print(a, t_test[i])
    """
    # 精度を計算
    accuracy = F.accuracy(y_test, t_test)
    accuracy.to_cpu()
    test_accuracies.append(accuracy.array)

    if test_iter.is_new_epoch:
        test_iter.reset()
        break

print('test_accuracy:{:.04f}'.format(np.mean(test_accuracies)))


# In[ ]:




