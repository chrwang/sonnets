import re
import numpy as np


def read_text(path):
    """
    Reads the text line by line, creating an array of length n_sonnets, where each element of the array is an array
    of strings containing the lines in that sonnet. Note that the sonnet indexing is Shakespeare specific.

    :param path: Path to read the input file from.
    :return: An array of sonnets.
    """
    result = []
    cnt = 0
    with open(path) as f:
        tmp = []
        for line in f:
            cnt += 1
            if re.match(r'^\s*1\s*$', line):
                continue
            else:
                if re.match(r'^\s*\d+\s*$', line):
                    result.append(tmp)
                    tmp = []
                else:
                    # Lower case and strip punctuation, strip newlines
                    l = line.lower().translate(str.maketrans("", "", '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~'''))
                    if len(l) != 0:
                        tmp.append(l)
    result.append(tmp)
    return result


def build_chars(path, max_len, skip, char_window=1):
    text_array = read_text(path)
    sentences = []
    next_chars = []
    for sonnet in text_array:
        text = "".join(line for line in sonnet)
        for i in range(0, len(text) - max_len - char_window + 1, skip):
            sentences.append(text[i: i + max_len])
            next_chars.append(text[i + max_len:i + max_len + char_window])
        # print('nb sequences:', len(sentences))

    chars = sorted(list(set("".join(s for s in sentences))))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # print('Vectorization...')
    x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
    y = [("c{0}_output".format(i+1), np.zeros((len(sentences), len(chars)), dtype=np.bool)) for i in range(char_window)]
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        for w in range(char_window):
            y[w][1][i, char_indices[next_chars[i][w]]] += 1
    return x, dict(y), char_indices, indices_char
