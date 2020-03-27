import preprocess
import HMM

def train(num_states, is_reversed=False):
    text = open('data/shakespeare.txt').read()
    obs, vocab, inv_vocab = preprocess.get_observations(text)

    if is_reversed:
        for ob in obs:
            ob.reverse()

    filename = f"models/hmm{num_states}" + ('_rev' if is_reversed else '') + ".txt"
    hmm = HMM.unsupervised_HMM(obs, num_states, 100)
    hmm.save(filename)

    return hmm

def load(num_states, is_reversed=False):
    filename = f"models/hmm{num_states}" + ('_rev' if is_reversed else '') + ".txt"
    hmm = HMM.load(filename)
    return hmm

# train(30, is_reversed=True)

# for state in [1, 5, 10, 20, 30, 50]:
#     train(state)
