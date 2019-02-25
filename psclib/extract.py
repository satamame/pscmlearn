import psclib.psc as psc
import copy
import numpy as np


class Extractor:
    def __init__(self, lines, ftels):
        self.lines = lines # 形態素解析された行のリスト (台本1冊分)
        self.ftels = ftels # (特徴名, ハイパーパラメータ) のタプルのリスト
        self.common_heads = [] # (よくある行頭, 出現行数) のタプルのリスト
    
    def extract(self, fout):
        """
        台本のすべての行の特徴を抽出してファイルに書き出すメソッド。
        """
        for lnum in range(len(self.lines)):
            # print('Line {}'.format(lnum))
            fout.write(self.extract_line(lnum) + '\n')
    
    def extract_line(self, lnum):
        """
        lnum 行目の特徴を抽出し、カンマ区切りにして返すメソッド。
        """
        feature_vec = []
        line = self.lines[lnum]
        for ftname in [ftel[0] for ftel in self.ftels]:
            """
            ftels の各要素の特徴名にもとづいて、特徴量を取得する。
            """
            if ftname == 'sc_count_of_lines': # 台本内の行数
                ft = len(self.lines)

            elif ftname == 'sc_count_of_lines_with_bracket': # 台本内の括弧を含む行の数
                ft = self.get_count_of_lines_with_bracket()

            elif ftname == 'ln_count_of_words': # 行内の語数
                ft = len(line['tokenized_words'])

            elif ftname == 'ln_count_of_brackets': # 行内の括弧の数
                lw = [word['surface'] for word in line['tokenized_words']]
                lb = [x for x in lw if x in psc.brackets]
                ft = len(lb)

            elif ftname == 'ln_length_of_common_head': # 行頭の何単語まで他の行と共通するか
                ft = len(self.get_length_of_common_head(lnum))
                # とりあえず、共通の行頭が存在する最大の単語数を特徴として使ってみる。
                # "単語数 * 存在する行数" などを特徴として使う手もある。

            elif ftname == 'ln_first_open_bracket_pos': # 開き括弧がどれくらい早く出現するか
                ft = self.get_first_pos_in_line(lnum, psc.open_brackets)

            elif ftname == 'ln_first_close_bracket_pos': # 閉じ括弧がどれくらい早く出現するか
                ft = self.get_first_pos_in_line(lnum, psc.close_brackets)

            elif ftname == 'ln_first_space_pos': # 空白がどれくらい早く出現するか
                ft = self.get_first_pos_in_line(lnum, psc.spaces)

            elif ftname == 'ln_first_comma_pos': # 読点がどれくらい早く出現するか
                ft = self.get_first_pos_in_line(lnum, psc.commas)

            elif ftname == 'ln_first_period_pos': # 句点がどれくらい早く出現するか
                ft = self.get_first_pos_in_line(lnum, psc.periods)

            elif ftname == 'ln_length_of_indent': # インデントの長さ
                ft = len(line['indent_chars'])

            elif ftname == 'ln_begins_with_name': # 最初の単語が名詞/固有名詞/人名か
                tw = line['tokenized_words']
                ft = 0
                if len(tw) > 0:
                    pos = tw[0]['part_of_speech'].split(',')
                    ft += (pos[0] == '名詞')        # 名詞なら 1
                    ft += (pos[1] == '固有名詞')    # さらに固有名詞なら 2
                    ft += (pos[2] == '人名') * 2    # さらに人名なら 4
            
            elif ftname == 'ln_ends_with_close_bracket': # 行の最後が閉じ括弧か
                tw = line['tokenized_words']
                if len(tw) > 0:
                    ft = int(tw[-1]['surface'] in psc.close_brackets)
                else:
                    ft = 0

            feature_vec.append(ft)
        return ",".join([str(ft) for ft in feature_vec])
    
    def get_count_of_lines_with_bracket(self):
        """
        台本内の括弧を含む行の数を返すメソッド。
        """
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

    def get_count_of_lines_with_head(self, head, lnum = 0):
        """
        lnum 行目以降の、行頭が head である行の数を返すメソッド。
        head は {surface, part_of_speech} の辞書のリスト。
        get_length_of_common_head() から呼ばれるためのメソッド。
        """
        count = 0
        for line in self.lines[lnum:]:
            match = True
            for i in range(len(head)):
                if len(line['tokenized_words']) < i + 1:
                    match = False
                    break
                tokenized_words = line['tokenized_words'][i]
                surface = head[i]['surface']
                part_of_speech = head[i]['part_of_speech']
                if tokenized_words['surface'] != surface \
                    or tokenized_words['part_of_speech'] != part_of_speech:
                    match = False
                    break
            if match:
                count += 1
        return count

    def get_length_of_common_head(self, lnum):
        """
        lnum 行目にて、共通の行頭を持つ行の数のリストを返す。
        返り値は1単語目～n単語目までの、同じ行頭を持つ行数のリスト（自分自身もカウントする）。
        """
        line = self.lines[lnum]
        length = 0
        head = []
        count = []
        for w in line['tokenized_words']: # 行頭からの各単語について
            # 「行頭」の定義に追加
            head.append({'surface': w['surface'], 'part_of_speech': w['part_of_speech']})
            # その「行頭」がすでにカウントされていれば、それ (common_heads の要素) を使う。
            counted = [x for x in self.common_heads if x[0] == head]
            if len(counted) > 0:
                count.append(counted[0][1])
            else:
                # まだ common_heads としてカウントされてなければ lnum 行目以降でカウント。
                line_count = self.get_count_of_lines_with_head(head, lnum)
                if line_count > 1:
                    # lnum 行目以外にも、行頭が head のものがあったなら、common_heads に追加。
                    # ※ タプルとして追加する。
                    self.common_heads.append((copy.deepcopy(head), line_count))
                    count.append(line_count)
                else:
                    break
        return count

    def get_first_pos_in_line(self, lnum, words):
        """
        lnum 行目にて、words 内のいずれかの語が最初に出現する位置。
        返り値は、位置 (0, 1, ...) を x とし、e^(-x/4) の値。なければ 0。
        """
        line = self.lines[lnum]
        lw = [word['surface'] for word in line['tokenized_words']]
        li = [i for i, x in enumerate(lw) if x in words]
        if len(li) > 0:
            x = li[0]
            return np.exp(-x/4)
        else:
            return 0
