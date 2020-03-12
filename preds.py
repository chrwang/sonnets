import numpy as np
from keras.layers import Dense, LSTM, Lambda
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from rnn import build_data

_, _,ctoi,  itoc = build_data("data/shakespeare.txt", 40, 3)
model = load_model("weights-improvement-89-0.0021.hdf5")

seed = "shall i compare thee to a summer's day\n"

result = ""
for _ in range(1000):
    seq = [ctoi[c] for c in seed]
    enc = pad_sequences([seq], maxlen=40, truncating='pre')

    enc = to_categorical(enc, num_classes=len(ctoi))
    enc = enc.reshape(1, 40, 30)

    p_ind = model.predict_classes(enc)
    c = itoc[p_ind[0]]
    seed += c
    # seed = seed[1:len(seed)]
print(seed)
