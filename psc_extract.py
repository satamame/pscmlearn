#! python3
# encoding: utf-8

args = sys.argv
if len(args) < 5:
    print('Usage: python psc_extract.py in-filename feature-def-filename out-filename-for-script out-filename-for-lines')
    sys.exit()