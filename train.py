# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import pickle
import os


train_df = pd.read_csv("data/train.txt", sep=";", names=["text", "label"])
test_df = pd.read_csv("data/test.txt", sep=";", names=["text", "label"])

print("Sample training data:")
print(train_df.head())


texts = train_df["text"].values
labels = train_df["label"].values


label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {i: l for l, i in label2id.items()}
y = np.array([label2id[label] for label in labels])

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=50)

 #Build Model

model = Sequential([
    Embedding(10000, 128, input_length=50),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(label2id), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


#  Train Model

history = model.fit(X, y, epochs=5, batch_size=64, validation_split=0.2, verbose=1)


# Save Model + Tokenizer + Labels

os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.h5")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/labels.pkl", "wb") as f:
    pickle.dump(id2label, f)

print(" Training complete! Model and tokenizer saved in 'models/' folder.")
