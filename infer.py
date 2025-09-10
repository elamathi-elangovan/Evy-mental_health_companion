# infer.py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# =========================
# Load Model + Tokenizer + Labels
# =========================
model = tf.keras.models.load_model("models/emotion_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    id2label = pickle.load(f)

# =========================
# Prediction Function
# =========================
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)
    prediction = model.predict(padded)[0]
    label_id = prediction.argmax()
    confidence = prediction[label_id]
    return id2label[label_id], float(confidence)

# =========================
# Test
# =========================
if __name__ == "__main__":
    samples = [
        "I am feeling very lonely and hopeless",
        "You make me so happy",
        "I am scared of the exam tomorrow",
        "I love spending time with my friends",
        "This makes me so angry!"
    ]

    for s in samples:
        emotion, conf = predict_emotion(s)
        print(f"Text: {s}\n → Predicted: {emotion} (confidence: {conf:.2f})\n")
