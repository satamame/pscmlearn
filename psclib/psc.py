import re


classes = (
    "TITLE",
    "AUTHOR",
    "CHARSHEADLINE",
    "CHARACTER",
    "H1",
    "H2",
    "H3",
    "DIRECTION",
    "DIALOGUE",
    "ENDMARK",
    "COMMENT",
    "EMPTY",
    "CHARACTER_CONTINUED",
    "DIRECTION_CONTINUED",
    "DIALOGUE_CONTINUED",
    "COMMENT_CONTINUED"
)

features = (
    # Feature elements of the script
    "sc_count_of_lines",
    "sc_count_of_lines_with_bracket",
    # Feature elements of the line
    "ln_count_of_words",
    "ln_count_of_brackets",
    "ln_length_of_common_head",
    "ln_first_bracket_pos",
    "ln_first_space_pos",
    "ln_length_of_indent"
)

brackets = ('「', '」', '『', '』')
spaces = (' ', '　', '\t')

# 特徴量設定データから、(特徴名, ハイパーパラメータ) のタプルのリストを作る関数。
# ffeat は、存在が保証されているファイルの名前。
def make_feature_elements(ffeat):

    # ファイル ffeat から、使用する特徴量の設定を取得。
    # Load feature elements to be used.
    ftin = [] # Feature elements input.
    for line in ffeat.readlines():
        # 読み込んだ行からコメント部分を削除。
        ftline = re.sub(r"#.*", "", line).strip()
        # 空行ならスキップ。
        if not ftline:
            continue
        # 読み込んだ行をカンマで区切ったリスト。
        ftel = [ft.strip() for ft in ftline.split(",")]
        # 2個目の要素（ハイパーパラメータ）が無ければ1（デフォルト値）とする。
        if len(ftel) < 2:
            ftel.append('1')
        # 定義された特徴名なら、リストに追加。
        if ftel[0] in features:
            ftin.append(ftel)
        elif ftel[0]:
            print('Warning: ' + ftel[0] + ' is in feature elements but not defined.')
    
    ftnames = [ftel[0] for ftel in ftin]

    # Check duplicates.
    for ft in features:
        if ftnames.count(ft) > 1:
            print('Warning: ' + ft + ' is duplicated in feature elements.')

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
