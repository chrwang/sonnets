import numpy as np
import re
import random
import itertools

def parse_sonnet(text):
    # Split each sonnet into lines
    lines = re.split(r'\n', text)[1:]

    # Dump sonnets of incorrect length
    if len(lines) != 14:
        return None

    # Remove all punctuation besides hyphens and apostrophes
    filtered_lines = [re.sub(r'[^\w\s\-\']', '', line) for line in lines]
    # Remove multiple spaces from sonnets
    filtered_lines = [re.sub(r'\s{2,}', '', line) for line in filtered_lines]
    tokenized = [re.split(r'\s+', line) for line in filtered_lines]
    return tokenized

def parse_sonnets(text):
    # Split raw text into sonnets
    sonnets = re.split(r'\s+[0-9]+', text.lower())[1:]
    return [parse_sonnet(sonnet) for sonnet in sonnets if parse_sonnet(sonnet) is not None]

def build_vocabulary(lines):
    # Build a dictionary mapping each word to an index
    index = 0
    vocab = {}
    for word in sum(lines, []):
        if word not in vocab:
            vocab[word] = index
            index += 1

    # Invert the vocab to get a mapping from index to word
    inv_vocab = {v: k for k, v in vocab.items()}

    return vocab, inv_vocab

def get_observations(text):
    sonnets = parse_sonnets(text)

    # Combine all lines of sonnets together
    lines = sum(sonnets, [])
    vocab, inv_vocab = build_vocabulary(lines)

    obs = [[vocab[word] for word in line] for line in lines]

    return obs, vocab, inv_vocab

def get_lengths(obs):
    lengths = {}
    for line in obs:
        lengths[len(line)] = lengths.get(len(line), 0) + 1

    total = sum(lengths.values())
    return {k: v / total for k, v in lengths.items()}

def get_punctuation(text):
    sonnets = re.split(r'\s+[0-9]+', text.lower())[1:]
    punctuation = {}
    for sonnet in sonnets:
        for line in re.split(r'\n', sonnet)[1:]:
            token = line[-1] if not line[-1].isalpha() else " "
            punctuation[token] = punctuation.get(token, 0) + 1

    total = sum(punctuation.values())
    return {k: v / total for k, v in punctuation.items()}

def add_to_rhyme_dict(couplet, rhyme_dict):
    for rhyme in rhyme_dict:
        if not couplet.isdisjoint(rhyme):
            rhyme.update(couplet)
            return

    rhyme_dict.append(couplet)

def build_rhyme_dict(text, vocab):
    sonnets = parse_sonnets(text)
    rhymes = []
    for sonnet in sonnets:
        for quatrain in range(3):
            for line in range(2):
                couplet = {vocab[sonnet[quatrain * 4 + line][-1]], vocab[sonnet[quatrain * 4 + line + 2][-1]]}
                add_to_rhyme_dict(couplet, rhymes)

        couplet = {vocab[sonnet[12][-1]], vocab[sonnet[13][-1]]}
        add_to_rhyme_dict(couplet, rhymes)

    return rhymes

def build_syllable_dict(vocab):
    syllable_dict = {}
    with open('data/Syllable_dictionary.txt') as fp:
        for line in fp:
            tokens = re.split(r'\s', line)
            entry = {'normal': [], 'end': []}
            for count in tokens[1:-1]:
                if count[0] == 'E':
                    entry['end'].append(int(count[1:]))
                else:
                    entry['normal'].append(int(count))
            syllable_dict[tokens[0]] = entry

    return {vocab[k]: v for k, v in syllable_dict.items()}
