import psclib.psc as psc


class Extractor:
    def __init__(self, lines, ftels):
        self.lines = lines
        self.ftels = ftels
    def extract(self, fout):
        for lnum in range(len(self.lines)):
            fout.write(ex.extract_line(lnum) + '\n')
    def extract_line(self, lnum):
        feature_vec = []
        line = self.lines[lnum]
        for ftname in [ftel[0] for ftel in ftels]:
            if ftname == 'sc_count_of_lines':
                ft = str(len(self.lines))
            elif ftname == 'sc_count_of_lines_with_bracket':
                ft = str(self.get_count_of_lines_with_bracket())
            elif ftname == 'ln_count_of_words':
                ft = str(len(line['tokenized_words']))
            elif ftname == 'ln_count_of_brackets':
                ft = str(sum(word['base_form'] == '「' \
                    for word in line['tokenized_words']))
            feature_vec.append(ft)
        return ",".join(feature_vec)
    
    def get_count_of_lines_with_bracket(self):
        if hasattr(self, 'count_of_lines_with_bracket'):
            return self.count_of_lines_with_bracket
        count = 0
        for line in self.lines:
            words = [w['surface'] for w in line['tokenized_words']]
            for w in words:
                if w in psc.brackets:
                    count += 1
                    break
        self.count_of_lines_with_bracket = count
        return count