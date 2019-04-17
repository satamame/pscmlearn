# Command samples

## 学習用データの作成

python psc_maketrain.py model_folder

- model_folder : モデルのフォルダ名

例
```
> python psc_maketrain.py mdl000
```

## 検証用データの作成

python psc_makeeval.py model_folder

- model_folder : モデルのフォルダ名

例
```
> python psc_makeeval.py mdl000
```

## モデルの学習

python psc_train.py train_list_file eval_list_file model_save_file

- train_list_file: 学習に使うファイルのリスト
- eval_list_file: 評価に使うファイルのリスト
- model_save_file: モデルの保存先

例
```
> python psc_train.py models/mdl000/ds_train_list.txt models/mdl000/ds_eval_list.txt models/mdl000/mdl.pkl
```

## 予測

python psc_predict.py model_file feature_setting_file script_file result_save_file

- model_file: 予測モデル
- feature_setting_file: モデルの特徴量設定
- script_file: 台本
- result_save_file: 予測結果の保存先

例
```
> python psc_predict.py models/mdl000/mdl.pkl, models/mdl000/mdl_fts.txt predict/undercontrol.txt predict/undercontrol_lbl.txt
```
