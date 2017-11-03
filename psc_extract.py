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


def make_feature_elements(ffeat):

    # Load feature elements to be used
    ft_sc = [] # Feature elements of script
    ft_ln = [] # Feature elements of line
    for line in ffeat.readlines():
        ftline = re.sub(r"#.*", "", line).strip()
        if not ftline:
            continue
        ftel = [ft.strip() for ft in ftline.split(",")]
        if len(ftel) < 2:
            ftel.append('1')
        if ftel[0] in psc.features_sc:
            ft_sc.append(ftel)
        elif ftel[0] in psc.features_ln:
            ft_ln.append(ftel)
        elif ftel[0]:
            print('Warning: ' + ftel[0] + ' is in feature elements but not defined.')
    
    ftnames_sc = [ftel[0] for ftel in ft_sc]
    ftnames_ln = [ftel[0] for ftel in ft_ln]

    # Check duplicates
    for ft in psc.features_sc:
        if ftnames_sc.count(ft) > 1:
            print('Warning: ' + ft + ' is duplicated in feature elements.')
    for ft in psc.features_ln:
        if ftnames_ln.count(ft) > 1:
            print('Warning: ' + ft + ' is duplicated in feature elements.')

    # Make feature elements definition
    ftel_sc = []
    for ftel in ft_sc:
        if not ftel[0] in [ft[0] for ft in ftel_sc]:
            try:
                v = float(ftel[1])
            except:
                v = 1.
            ftel_sc.append((ftel[0], v))
    ftel_ln = []
    for ftel in ft_ln:
        if not ftel[0] in [ft[0] for ft in ftel_ln]:
            try:
                v = float(ftel[1])
            except:
                v = 1.
            ftel_ln.append((ftel[0], v))

    return ftel_sc, ftel_ln


# def extract_script_features(fin, ftel_sc):


args = sys.argv
if len(args) < 5:
    print('Usage: python psc_extract.py in-filename feature-vector-filename ' + \
        'out-filename-for-script out-filename-for-lines')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8') # Morphologically analyzed script
    ffeat = open(args[2]) # Feature vector definition
    foutsc = open(args[3], 'w') # Output of features extracted from script
    foutln = open(args[4], 'w') # Output of features extracted from lines
except IOError as err:
    print(err)
    close_all()
    sys.exit()

# Load feature elements to be used
ftel_sc, ftel_ln = make_feature_elements(ffeat)

print ('Features of script : {0}'.format(ftel_sc))
print ('Features of line : {0}'.format(ftel_ln))

