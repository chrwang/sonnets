import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from rnn import build_data, build_model

# Build dataset for mapping dicts and shapes.
X, _,ctoi,  itoc = build_data("data/shakespeare.txt", 40, 3)
# Load the trained model from the model file.
trained_model = load_model("trained_checkpoints/weights-improvement-68-0.0014.hdf5")
# For the set of temperatures specified
for t in [0.25, 0.75, 1.5]:
    # Initial seed
    seed = "shall i compare thee to a summer's day\n"
    # Build model with specified temp
    model = build_model((40, len(X[0][0])), temperature=t)
    # Get trained weights
    model.set_weights(trained_model.get_weights())
    print("500 character prediction for temperature {}.".format(t))
    # Predict 500 Characters
    for _ in range(500):
        # Convert to ints
        seq = [ctoi[c] for c in seed]
        # Pad to 40 characters
        enc = pad_sequences([seq], maxlen=40, truncating='pre')
        # Convert to categorical encoding
        enc = to_categorical(enc, num_classes=len(ctoi))
        # Reshape to fit the network
        enc = enc.reshape(1, 40, 30)
    
        # Predict an integer
        p_ind = model.predict_classes(enc)
        # Convert back to character
        c = itoc[p_ind[0]]
        # Append to seed
        seed += c
    print(seed)
