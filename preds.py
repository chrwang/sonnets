import sys
import curses
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from preprocess import build_chars

_, _, ctoi, itoc = build_chars("data/shakespeare.txt", 40, 3)
path = "m_final_window5_small"
window = 5
model = load_model(path)
seed = "shall i compare thee to a summer's day\n"


def generate(seed):
    result = seed
    seq = [ctoi[c] for c in result]
    for _ in range(int(1000/window)):
        enc = pad_sequences([seq], maxlen=40, truncating='pre')

        enc = to_categorical(enc, num_classes=len(ctoi))
        enc = enc.reshape(1, 40, 30)

        p_prob = model.predict(enc)
        p_ind = np.argmax(p_prob, axis=2)
        out = ""
        for i in range(window):
            out += itoc[p_ind[i, 0]]
            seq.append(p_ind[i, 0])
        result += out
    return result


print("Generating...")
result = generate(seed)
print(result)
with open("{0}/assets/sonnet_window{1}.txt".format(path, window), 'w') as f:
    f.write(result)
