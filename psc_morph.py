#! python3
# encoding: utf-8

# 台本データ（テキスト）を入力として、形態素データを出力するプログラム。
# 出力ファイル名を2個指定すると、2個めはラベルの出力となる（入力に教師ラベルがある場合）

import sys
from janome.tokenizer import Tokenizer
import re
import json
from psclib import psc


# すべてのファイルを閉じる関数。
def close_all():
    if 'flbl' in globals():
        flbl.close()
    if 'fout' in globals():
        fout.close()
    if 'fin' in globals():
        fin.close()


# ここからメインルーチン。
# パラメタの取得。
args = sys.argv
if len(args) < 3:
    # パラメタが少なすぎる。使い方を表示して終了する。
    print('Usage: python psc_morph.py in-filename out-filename [out-filename-for-labels]')
    sys.exit()
try:
    # パラメタに従って、入力ファイルと出力用ファイルをオープンする。
    fin = open(args[1], encoding='utf-8')
    fout = open(args[2], 'w', encoding='utf-8')
    if len(args) > 3:
        flbl = open(args[3], 'w')
except IOError as err:
    # いずれかのファイルのオープンに失敗したら、すべて閉じて終了する。
    print(err)
    close_all()
    sys.exit()


t = Tokenizer()
# token_lines は、行ごとの解析結果（辞書型）を要素とするリストになる。
token_lines = []
# 以下の2つは、正規表現マッチングに使うパターン。
class_pattern = re.compile(r"([A-Z0-9_]+),(.*)")
space_pattern = re.compile(r"[/s　]+")


# 入力の行ごとの処理。
for line in fin.readlines():
    # 教師ラベルと台本データに分ける。
    # Isolate class name and data
    matched = re.match(class_pattern, line)
    # 入力文字列に教師ラベルが含まれていた場合。
    if matched and matched.group(1) in psc.classes:
        # 教師ラベルをファイルに書出し、残りを台本データとする。
        data = matched.group(2)
        if 'flbl' in globals():
            flbl.write(matched.group(1) + '\n')
    # 入力文字列に教師ラベルが含まれていない場合。
    else:
        # 入力文字列をまるごと台本データとする。
        data = line
    
    # 行頭の空白文字列を、形態素解析とは別に取っておく。
    # Space characters in indent
    matched = re.match(space_pattern, data)
    if matched:
        indent_chars = matched.group()
    else:
        indent_chars = ""

    # この行の単語リストを作る。
    # List words in the line
    tokenized_words = []
    for token in t.tokenize(data):
        token_info = {
            'surface': token.surface,
            'part_of_speech': token.part_of_speech,
            'infl_type': token.infl_type,
            'infl_form': token.infl_form,
            'base_form': token.base_form,
            'reading': token.reading,
            'phonetic': token.phonetic
        }
        tokenized_words.append(token_info)

    # 行頭の空白文字列と、解析した単語のリストを、辞書型にしてリストに追加する。
    token_lines.append({
        'indent_chars': indent_chars,
        'tokenized_words': tokenized_words
    })

# リストを JSON にしてファイルに出力する（形態素データ）。
json.dump(token_lines, fp=fout, ensure_ascii=False, indent=4)

# すべてのファイルを閉じて終了。
close_all()
