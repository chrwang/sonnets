import re


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
