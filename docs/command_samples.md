# Command samples

## 特徴量抽出

python psc_extract.py script_file feature_setting_file feature_save_file [label_save_file]

- script_file: 台本
- feature_setting_file: 特徴量設定
- feature_save_file: 特徴量出力先
- label_save_file: 教師ラベル出力先 (台本に含まれている場合)

例（入力に教師ラベルがない場合）
```
> python psc_extract.py dataset/000001.txt models/mdl000/mdl_fts.txt models/mdl000/train/000001_ft.csv
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
