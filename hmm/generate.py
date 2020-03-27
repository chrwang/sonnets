import preprocess
import train
import numpy as np
import random

def generate_emission(hmm, length):
    emission = []
    state = np.random.choice(hmm.L, p=hmm.A_start)

    for _ in range(length):
        emission.append(np.random.choice(hmm.D, p=hmm.O[state]))
        state = np.random.choice(hmm.L, p=hmm.A[state])

    return emission

def generate_sonnet(num_states):
    hmm = train.load(num_states)
    text = open('data/shakespeare.txt').read()

    obs, vocab, inv_vocab = preprocess.get_observations(text)
    lengths = preprocess.get_lengths(obs)
    punctuation = preprocess.get_punctuation(text)

    samples = []
    for _ in range(14):
        length = np.random.choice(list(lengths.keys()), p=list(lengths.values()))
        samples.append(generate_emission(hmm, length))

    return format_sonnet(samples, inv_vocab, punctuation)

def format_sonnet(samples, inv_vocab, punctuation):
    lines = []
    for sample in samples:
        line = ""
        for i, ob in enumerate(sample):
            if i == 0:
                line += inv_vocab[ob].capitalize()
            else:
                line += " " + ('I' if inv_vocab[ob] == 'i' else inv_vocab[ob])
        punc = np.random.choice(list(punctuation.keys()), p=list(punctuation.values()))
        lines.append(line + punc)

    lines[-1] = lines[-1][:-1] + '.'
    return '\n'.join(lines)

def reweigh_observations(curr_O, valid_obs):
    reweighed_O = [0 for _ in range(len(curr_O))]

    for obs in range(len(curr_O)):
        reweighed_O[obs] = curr_O[obs] if obs in valid_obs else 0

    # Normalize probabilities
    return [x / sum(reweighed_O) for x in reweighed_O]

def reweigh_transitions(curr_A, O, valid_obs):
    reweighed_A = [0 for _ in range(len(curr_A))]

    for state in range(len(O)):
        num, denom = 0, 0
        for obs in range(len(O[state])):
            num += O[state][obs] if obs in valid_obs else 0
            denom += O[state][obs]

        reweighed_A[state] = curr_A[state] * num / denom

    # Normalize probabilities
    return [x / sum(reweighed_A) for x in reweighed_A]

def generate_emission_seeded(hmm, length, seed):
    emission = []
    state = np.random.choice(hmm.L, p=reweigh_transitions(hmm.A_start, hmm.O, {seed}))
    emission.append(np.random.choice(hmm.D, p=reweigh_observations(hmm.O[state], {seed})))

    for _ in range(length - 1):
        state = np.random.choice(hmm.L, p=hmm.A[state])
        emission.append(np.random.choice(hmm.D, p=hmm.O[state]))

    return emission

def generate_sonnet_rhyme(num_states):
    hmm = train.load(num_states, is_reversed=True)
    text = open('data/shakespeare.txt').read()

    obs, vocab, inv_vocab = preprocess.get_observations(text)
    rhyme_dict = preprocess.build_rhyme_dict(text, vocab)
    lengths = preprocess.get_lengths(obs)
    punctuation = preprocess.get_punctuation(text)

    samples = [None] * 14
    for quatrain in range(3):
        for line in range(2):
            couplet = random.sample(random.choice(rhyme_dict), 2)
            length = np.random.choice(list(lengths.keys()), p=list(lengths.values()))

            samples[quatrain * 4 + line] = generate_emission_seeded(hmm, length, couplet[0])
            samples[quatrain * 4 + line + 2] = generate_emission_seeded(hmm, length, couplet[1])

    couplet = random.sample(random.choice(rhyme_dict), 2)
    samples[12] = generate_emission_seeded(hmm, length, couplet[0])
    samples[13] = generate_emission_seeded(hmm, length, couplet[1])

    for i in range(len(samples)):
        samples[i].reverse()

    return format_sonnet(samples, inv_vocab, punctuation)

print(generate_sonnet_rhyme(30))
