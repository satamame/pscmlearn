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
    "ln_count_of_brackets"
)

brackets = ('「', '」', '『', '』')


# 特徴量定義データから、(特徴名, ハイパーパラメータ) のタプルのリストを作る関数。
# ffeat は、存在が保証されているファイルの名前。
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
        if ftel[0] in features:
            ftin.append(ftel)
        elif ftel[0]:
            print('Warning: ' + ftel[0] + ' is in feature elements but not defined.')
    
    ftnames = [ftel[0] for ftel in ftin]

    # Check duplicates
    for ft in features:
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
