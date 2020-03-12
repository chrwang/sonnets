import numpy as np
from keras.layers import Dense, LSTM, Lambda
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from preprocess import read_text
import time

def build_data(path, max_len, skip):
    text_array = read_text(path)
    sentences = []
    next_chars = []
    for sonnet in text_array:
        text = "".join(line for line in sonnet)
        for i in range(0, len(text) - max_len, skip):
            sentences.append(text[i: i + max_len])
            next_chars.append(text[i + max_len])
        print('nb sequences:', len(sentences))

    chars = sorted(list(set("".join(s for s in sentences))))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('Vectorization...')
    x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y


def build_model(input_shape, lstm_size=200, temperature=1):
    model = Sequential()
    model.add(LSTM(lstm_size, input_shape=input_shape))
    model.add(Dense(input_shape[1], activation='softmax'))
    model.add(Lambda(lambda x: x / temperature))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


if __name__ == "__main__":
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    X, Y = build_data("data/shakespeare.txt", 40, 3)
    m = build_model((40, len(X[0][0])), temperature=1)
    m.fit(X, Y, 64, 150, callbacks=[checkpoint])
    m.save("m_final_"+str(time.time()))

