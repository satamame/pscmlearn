#! python3
# encoding: utf-8

import sys
import re
import json
import psc


def close_all():
    fout.close()
    ffeat.close()
    fin.close()


def make_feature_elements(ffeat):

    # Load feature elements to be used
    ftin = [] # Feature elements input
    for line in ffeat.readlines():
        ftline = re.sub(r"#.*", "", line).strip()
        if not ftline:
            continue
        ftel = [ft.strip() for ft in ftline.split(",")]
        if len(ftel) < 2:
            ftel.append('1')
        if ftel[0] in psc.features:
            ftin.append(ftel)
        elif ftel[0]:
            print('Warning: ' + ftel[0] + ' is in feature elements but not defined.')
    
    ftnames = [ftel[0] for ftel in ftin]

    # Check duplicates
    for ft in psc.features:
        if ftnames.count(ft) > 1:
            print('Warning: ' + ft + ' is duplicated in feature elements.')

    # Make feature elements definition
    ftels = []
    for ftel in ftin:
        if not ftel[0] in [ft[0] for ft in ftels]:
            try:
                v = float(ftel[1])
            except:
                v = 1.
            ftels.append((ftel[0], v))

    return ftels


class Extractor:
    def __init__(self, lines, ftels):
        self.lines = lines
        self.ftels = ftels
    def extract(self, fout):
        for lnum in range(len(self.lines)):
            fout.write(ex.extract_line(lnum) + '\n')
    def extract_line(self, lnum):
        feature_vec = []
        for ftname in [ftel[0] for ftel in ftels]:
            if ftname == 'sc_count_of_lines':
                ft = str(len(self.lines))
            elif ftname == 'sc_count_of_lines_with_bracket':
                ft = str(self.count_of_lines_with_bracket())
            elif ftname == 'ln_count_of_characters':
                #
                #
                #
                ft = "1"
            elif ftname == 'ln_count_of_brackets':
                #
                #
                #
                ft = "1"
            feature_vec.append(ft)
        return ",".join(feature_vec)
    def count_of_lines_with_bracket(self):
        #
        #   
        #
        return 1

args = sys.argv
if len(args) < 4:
    print('Usage: python psc_extract.py in-filename feature-elements-filename ' + \
        'out-filename')
    sys.exit()
try:
    fin = open(args[1], encoding='utf-8') # Morphologically analyzed script
    ffeat = open(args[2]) # Feature vector definition
    fout = open(args[3], 'w') # Output of features
except IOError as err:
    print(err)
    close_all()
    sys.exit()

# Load feature elements to be used
ftels = make_feature_elements(ffeat)

# print ('Feature elements : {0}'.format(ftels))

ex = Extractor(json.load(fin), ftels)
ex.extract(fout)
close_all()
