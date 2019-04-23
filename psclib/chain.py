import numpy as np

import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer import iterators
from chainer.datasets import split_dataset_random
from chainer.dataset import concat_examples

from chainer.cuda import to_cpu, to_gpu


class PscChain(Chain):
    """
    Chainer の Chain クラスを使った予測モデル
    """
    def __init__(self, hid_dim, out_dim):
        """
        コンストラクタ

        Parameters
        ----------
        hid_dim : int
            隠れ層のノード数
        out_dim : int
            出力層のノード数
        """
        super().__init__(
            l1=L.Linear(None, hid_dim),
            l2=L.Linear(hid_dim, hid_dim),
            l3=L.Linear(hid_dim, out_dim)
        )
    
    def __call__(self, x):
        """
        順伝播して、出力層 (Variable) を返すメソッド

        Parameters
        ----------
        x : chainer.variable.Variable
            (バッチサイズ x 特徴ベクトルの次元数) の、入力データ
        
        Returns
        -------
        output : chainer.variable.Variable
            softmax をかける前の出力
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
    
    def train(self, *, dataset=None, batch_size=100, max_epoch=20, optimizer=None,
        gpu_id=-1, verbose=True):
        """
        学習用データを使って学習する
        """
        if dataset is None:
            raise ValueError("dataset is required.")

        if optimizer is None:
            raise ValueError("optimizer is required.")
        
        # 特徴量データと教師ラベルを対にして、学習用と検証用に分ける
        train_count = int(len(dataset) * 0.8) # 8割のデータを学習用に
        ds_train, ds_valid = split_dataset_random(dataset, train_count, seed=0)

        # 学習用イテレータ
        train_iter = iterators.SerialIterator(ds_train, batch_size)
        # 検証用イテレータ
        valid_iter = iterators.SerialIterator(ds_valid, batch_size, repeat=False, shuffle=False)

        # GPU を使うなら 0, 使わないなら -1
        if gpu_id >= 0:
            self.to_gpu(gpu_id)
        else:
            self.to_cpu()
        
        while train_iter.epoch < max_epoch:
            # イテレーション
            train_batch = train_iter.next()
            
            # バッチサイズ分の、入力データの array と、教師ラベルの array
            x, t = concat_examples(train_batch, gpu_id)
            x = x.astype(np.float32)
            
            # 順伝播の結果を得る
            y = self(x)
            
            # ロスの計算
            loss = F.softmax_cross_entropy(y, t)
            
            # 勾配の計算
            self.cleargrads()
            loss.backward()
            
            # パラメータの更新
            optimizer.update()
            
            # 学習用のデータセットを一周したタイミングでのみ検証を実行
            if train_iter.is_new_epoch:

                # ロスの表示
                # print('epoch:{:02d} train_loss:{:.04f} '.format(
                #    train_iter.epoch, float(to_cpu(loss.data))), end='')
                
                valid_losses = []
                valid_accuracies = []
                while True:
                    valid_batch = valid_iter.next()
                    x_valid, t_valid = concat_examples(valid_batch, gpu_id)
                    x_valid = x_valid.astype(np.float32)

                    # Validation データを forward
                    with chainer.using_config('train', False), \
                            chainer.using_config('enable_backprop', False):
                        y_valid = self(x_valid)
                        
                    # ロスを計算
                    loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
                    valid_losses.append(to_cpu(loss_valid.array))

                    # 精度を計算
                    accuracy = F.accuracy(y_valid, t_valid)
                    accuracy.to_cpu()
                    valid_accuracies.append(accuracy.array)

                    # 検証用のデータセットを一周したら終了
                    if valid_iter.is_new_epoch:
                        # 次回の検証に備えてイテレータをリセット
                        valid_iter.reset()
                        break

                # ロスと精度の表示
                if (verbose):
                    print('{:0=2} val_loss:{:.04f} val_accuracy:{:.04f}'.format(
                        train_iter.epoch, np.mean(valid_losses), np.mean(valid_accuracies)))

    def evaluate(self, *, dataset=None, batch_size=100, gpu_id=-1):
        """
        評価用データを使って評価する

        Returns
        -------
        accuracies : list
            バッチごとの精度のリスト
        """
        if dataset is None:
            raise ValueError("dataset is required.")

        # 評価用イテレータ
        iter = iterators.SerialIterator(dataset, batch_size, repeat=False, shuffle=False)

        # 評価用データでの評価
        accuracies = []
        while True:
            batch = iter.next()
            x, t = concat_examples(batch, gpu_id)
            x = x.astype(np.float32)
            
            # 評価用データをforward
            with chainer.using_config('train', False), \
                    chainer.using_config('enable_backprop', False):
                y = self(x)
            
            # 精度を計算
            accuracy = F.accuracy(y, t)
            accuracy.to_cpu()
            accuracies.append(accuracy.array)

            if iter.is_new_epoch:
                iter.reset()
                break
        
        return accuracies
