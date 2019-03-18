import re

from chainer import Chain
import chainer.links as L
import chainer.functions as F

from janome.tokenizer import Tokenizer


classes = (
    "TITLE",                # 0
    "AUTHOR",               # 1
    "CHARSHEADLINE",        # 2
    "CHARACTER",            # 3
    "H1",                   # 4
    "H2",                   # 5
    "H3",                   # 6
    "DIRECTION",            # 7
    "DIALOGUE",             # 8
    "ENDMARK",              # 9
    "COMMENT",              # 10
    "EMPTY",                # 11
    "CHARACTER_CONTINUED",  # 12
    "DIRECTION_CONTINUED",  # 13
    "DIALOGUE_CONTINUED",   # 14
    "COMMENT_CONTINUED"     # 15
)

features = (
    # Feature elements of the script
    "sc_count_of_lines",
    "sc_count_of_lines_with_bracket",
    # Feature elements of the line
    "ln_count_of_words",
    "ln_count_of_brackets",
    "ln_length_of_common_head",
    "ln_first_open_bracket_pos",
    "ln_first_close_bracket_pos",
    "ln_first_space_pos",
    "ln_first_comma_pos",
    "ln_first_period_pos",
    "ln_length_of_indent",
    "ln_begins_with_name",
    "ln_ends_with_close_bracket"
)

open_brackets = ('「', '『')
close_brackets = ('」', '』')
brackets = open_brackets + close_brackets
spaces = (' ', '　', '\t')
commas = (',', '、')
periods = ('.', '。')

def isolate_labels_from_lines(lines):
    """
    教師データの各行から教師ラベルとデータを分離する関数
    
    各行にラベルがある前提とし、ラベルがない行については空文字列を出力する。
    入力ファイルにおいてラベルがない事を明示するには、行の先頭にカンマをつける。
    ラベルが正しいかどうかの厳密なチェックは行わない。

    Parameters
    ----------
    lines : list
        行 (str) のリスト
    
    Returns
    -------
    data : list
        教師ラベルを除いたデータ (str) のリスト
    labels : list
        教師ラベル（str）のリスト
    """
    
    # 正規表現マッチングパターン
    class_pattern = re.compile(r"([A-Z0-9_]*),(.*)")
    
    data = []
    labels = []
    for line in lines:
        matched = re.match(class_pattern, line)
        if matched:
            data.append(matched.group(2))
            labels.append(matched.group(1))
        else:
            data.append(line)
            labels.append("")
    
    return data, labels


def tokenize_lines(lines):
    """
    複数の行を形態素解析する関数

    Parameters
    ----------
    lines : list
        行 (str) のリスト
    
    Returns
    -------
    token_lines : list
        解析結果（dict）のリスト
    """
    t = Tokenizer()
    # token_lines は、行ごとの解析結果（辞書型）を要素とするリストになる。
    token_lines = []
    # 正規表現マッチングに使うパターン
    space_pattern = re.compile(r"[\s　]+")
    
    for data in lines:
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
    return token_lines


def read_feature_elements(ftset_file):
    """
    特徴量設定ファイルの内容からリストを作る関数
    
    Parameters
    ----------
    ftset_file : str
        特徴量設定ファイルの名前。ファイルの存在が保証されていること。
    
    Returns
    -------
    ftels : list
        (特徴名, ハイパーパラメータ) のタプルのリスト
    """

    # ファイル ftset_file から、使用する特徴量の設定を取得。
    # Load feature elements to be used.
    ftin = [] # Feature elements input.
    for line in open(ftset_file, 'r'):
        # 読み込んだ行からコメント部分を削除。
        ftline = re.sub(r"#.*", "", line).strip()
        # 空行ならスキップ。
        if not ftline:
            continue
        # 読み込んだ行をカンマで区切ったリスト。
        ftel = [ft.strip() for ft in ftline.split(",")]
        # 2個目の要素（ハイパーパラメータ）が無ければ 1（デフォルト値）とする。
        if len(ftel) < 2:
            ftel.append('1')
        # 定義された特徴名なら、リストに追加。
        if ftel[0] in features:
            ftin.append(ftel)
        elif ftel[0]:
            print('Warning: Feature \'' + ftel[0] + '\' not defined.')
    
    # 特徴量の名前のリスト
    ftnames = [ftel[0] for ftel in ftin]

    # Check duplicates.
    for ft in features:
        if ftnames.count(ft) > 1:
            print('Warning: \'' + ft + '\' is duplicated in feature elements.')

    # Make feature elements definition.
    ftels = []
    for ftel in ftin:
        # 重複がないように、初めて現れた特徴名だけを集める。
        if not ftel[0] in [ft[0] for ft in ftels]:
            try:
                v = float(ftel[1])
            except:
                v = 1.
            # ハイパーパラメータと合体させてタプルにしてリストに追加。
            ftels.append((ftel[0], v))

    return ftels


class PscChain(Chain):
    """
    予測モデル
    """
    def __init__(self, hid_dim, out_dim):
        """
        初期化メソッド

        Parameters
        ----------
        hid_dim : int
            隠れ層のノード数
        out_dim : int
            出力層のノード数
        """
        super().__init__(
            l1=L.Linear(None, hid_dim),
            l2=L.Linear(hid_dim, hid_dim),
            l3=L.Linear(hid_dim, out_dim)
        )
    
    def __call__(self, x):
        """
        順伝播して、出力層 (Variable) を返すメソッド

        Parameters
        ----------
        x : chainer.variable.Variable
            (バッチサイズ x 特徴ベクトルの次元数) の、入力データ
        
        Returns
        -------
        output : chainer.variable.Variable
            softmax をかける前の出力
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
