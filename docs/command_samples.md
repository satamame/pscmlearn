# Command samples

## 特徴量抽出

python psc_extract.py script_file feature_setting_file feature_save_file [label_save_file]

例（入力に教師ラベルがある場合）
```
> python psc_extract.py train/tr_0000_sc.txt model/mdl_0000_ft.txt train/tr_0000_ft.csv train/tr_0000_lbl.txt
```

## 学習

python psc_train.py train_list_file, eval_list_file, model_save_file

例
```
> python psc_train.py ds_train_list.txt ds_eval_list.txt model/mdl_0000.pkl
```

## 予測

python psc_predict.py model_file, feature_setting_file, script_file, result_save_file

例
```
> python psc_predict.py model/mdl_0000.pkl, model/mdl_0000_ft.txt, predict/prd_0000_sc.txt, predict/prd_0000_lbl.txt
```
