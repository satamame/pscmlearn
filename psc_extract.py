#! python3
# encoding: utf-8

import sys
import re

import psc


def close_all():
    foutln.close()
    foutsc.close()
    ffeat.close()
    fin.close()


args = sys.argv
if len(args) < 5:
    print('Usage: python psc_extract.py in-filename feature-vector-filename ' + \
        'out-filename-for-script out-filename-for-lines')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8') # Morphologically analyzed script
    ffeat = open(args[2]) # Definition of features to be extracted
    foutsc = open(args[3], 'w') # Features extracted from script
    foutln = open(args[4], 'w') # Features extracted from lines
except IOError as err:
    print(err)
    close_all()
    sys.exit()

# Check feature vector
features = []
for line in ffeat.readlines():
    ft = re.sub(r"#.*", "", line).strip()
    if ft in psc.features:
        features.append(ft)
    elif ft:
        print('Warning: ' + ft + ' is in feature vector but not defined.')

for ft in psc.features:
    if features.count(ft) > 1:
        print('Warning: ' + ft + ' appears in feature vector twice or more.')

feature_vec = []
for ft in features:
    if ft not in feature_vec:
        feature_vec.append(ft)

print (feature_vec)
