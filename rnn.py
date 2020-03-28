import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Concatenate, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import build_chars
import time


def build_model(input_shape, lstm_size=200, char_window=1, temperature=1):
    inputs = Input(shape=input_shape)
    ref = inputs
    outputs = []
    for w in range(char_window):
        if w is 0:
            x = inputs
        else:
            x = Reshape(target_shape=(1, input_shape[1]))(outputs[w-1])
            ref = x = Concatenate(axis=1)([ref, x])
        x = LSTM(lstm_size, input_shape=(input_shape[0]+w, input_shape[1]))(x)
        x = Dense(100, activation="relu")(x)
        x = Dense(input_shape[1], activation='softmax')(x)
        x = Lambda(lambda a: a / temperature, name="c{0}_output".format(w+1))(x)
        outputs.append(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def build_model_1(input_shape, lstm_size=300, char_window=1, temperature=1):
    inputs = Input(shape=input_shape)
    outputs = []

    ref = inputs
    rec = LSTM(lstm_size)
    for w in range(char_window):
        if w is 0:
            x = inputs
        else:
            x = Reshape(target_shape=(1, input_shape[1]))(outputs[w-1])
            ref = x = Concatenate(axis=1)([ref, x])
        x = rec(x)
        x = Dense(100, activation="relu")(x)
        x = Dense(input_shape[1], activation='softmax')(x)
        x = Lambda(lambda a: a / temperature, name="c{0}_output".format(w+1))(x)
        outputs.append(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


if __name__ == "__main__":
    WINDOW_SIZE = 5
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    X, Y, _, _ = build_chars("data/shakespeare.txt", 40, 3, char_window=WINDOW_SIZE)
    m = build_model((40, len(X[0][0])), temperature=1, char_window=WINDOW_SIZE)
    # m = load_model("weights-improvement-14-40.8626.hdf5")
    print("Fitting model...")
    hist = m.fit(X, Y, 64, 100, callbacks=[checkpoint])
    name = "m_final_window{0}_{1}".format(WINDOW_SIZE, time.time())
    print("Saving to: {0}...".format(name))
    m.save(name)

    # Plotting
    if True:
        names = []
        for w in range(WINDOW_SIZE):
            names.append("c{0}_output".format(w+1))

        # Plot total loss values
        plt.plot(hist.history['loss'])
        plt.title('Model loss (total)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("{0}/assets/loss_total.png".format(name))
        plt.show()

        # Plot individual loss values
        for n in names:
            plt.plot(hist.history['{0}_loss'.format(n)])
        plt.title('Model loss (individual)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(names, loc='best')
        plt.savefig("{0}/assets/loss_ind.png".format(name))
        plt.show()

        # Plot training & validation accuracy values
        for n in names:
            plt.plot(hist.history['{0}_accuracy'.format(n)])
        plt.title('Model accuracy (individual)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(names, loc='best')
        plt.savefig("{0}/assets/accuracy_ind.png".format(name))
        plt.show()

        # Plot first character accuracy values
        plt.plot(hist.history['{0}_accuracy'.format(names[0])])
        plt.title('Model accuracy (first char)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig("{0}/assets/accuracy_first.png".format(name))
        plt.show()
