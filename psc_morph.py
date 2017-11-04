#! python3
# encoding: utf-8

import sys
from janome.tokenizer import Tokenizer
import re
import json
import psc


def close_all():
    if 'flbl' in globals():
        flbl.close()
    fout.close()
    fin.close()


args = sys.argv
if len(args) < 3:
    print('Usage: python psc_morph.py in-filename out-filename [out-filename-for-labels]')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8')
    fout = open(args[2], 'w', encoding='utf-8')
    if len(args) > 3:
        flbl = open(args[3], 'w')
except IOError as err:
    print(err)
    close_all()
    sys.exit()

t = Tokenizer()
token_lines = []
class_pattern = re.compile(r"([A-Z0-9_]+),(.*)")
space_pattern = re.compile(r"[/s　]+")

for line in fin.readlines():
    # Isolate class name and data
    matched = re.match(class_pattern, line)
    if matched and matched.group(1) in psc.classes:
        data = matched.group(2)
        if 'flbl' in globals():
            flbl.write(matched.group(1) + '\n')
    else:
        data = line
    
    # Space characters in indent
    matched = re.match(space_pattern, data)
    if matched:
        indent_chars = matched.group()
    else:
        indent_chars = ""

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

    token_lines.append({
        'indent_chars': indent_chars,
        'tokenized_words': tokenized_words
    })

json.dump(token_lines, fp=fout, ensure_ascii=False, indent=4)

close_all()
