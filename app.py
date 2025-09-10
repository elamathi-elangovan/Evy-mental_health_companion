from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import os
from groq import Groq

app = Flask(__name__)

# ===== Load Model & Tokenizer =====
model = tf.keras.models.load_model("models/emotion_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    id2label = pickle.load(f)

max_len = 50  # use same as training

# ===== Groq API Setup =====
# safer way: set your key in environment variables first
# in PowerShell: setx GROQ_API_KEY "gsk_your_real_key_here"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_supportive_reply(user_message, emotion, conversation_history=None):
    """Generate more nuanced and context-aware empathetic replies."""
    try:
        # Build conversation context
        history_context = ""
        if conversation_history and len(conversation_history) > 1:
            # Include last few exchanges for context
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_context = "Previous conversation:\n"
            for msg in recent_history:
                history_context += f"{msg['role']}: {msg['content']}\n"
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
            messages=[
                {"role": "system",
                    "content": (
                         "You are EVY, a compassionate mental health companion. "
                     "Respond naturally and empathetically. Use emojis and casual language where appropriate. "
                     "Rules:\n"
                     "1. Don't always end with a question.\n"
                     "2. Match user's tone.\n"
                     "3. Vary responses.\n"
                     "4. Sometimes just validate feelings.\n"
                     "5. Accept silence and don't pressure user.\n"
                     "6. Use human emotions to reply and emojis when needed. \n "
                        
                    )
                },
                {
                    "role": "user", 
                    "content": f"{history_context}Current emotion detection: {emotion}\nUser's latest message: {user_message}"
                }
            ],
            max_tokens=80,
            temperature=0.85,  # Slightly higher for more varied responses
            presence_penalty=0.3,  # Discourage repetition
        )

        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        # More nuanced fallbacks based on emotion
        fallbacks = {
            "happy": [
                "That's lovely to hear! 😊",
                "I'm glad you're feeling good today!",
                "Joy looks good on you! What's been nice about your day?"
            ],
            "sad": [
                "I'm here with you through this 💙",
                "That sounds really tough. I'm listening.",
                "Your feelings are valid. I'm here when you want to share more."
            ],
            "neutral": [
                "Thanks for checking in 👋",
                "I'm here whenever you want to talk.",
                "How's everything going with you?"
            ],
            "angry": [
                "That sounds really frustrating. Want to talk about it?",
                "I hear your frustration. What's been bothering you?",
                "That sounds upsetting. I'm here to listen."
            ]
        }
        
        import random
        return random.choice(fallbacks.get(emotion, ["I'm here with you 💙", "Thanks for sharing that."]))      

# ===== Routes =====
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' field in JSON"}), 400

        text = data["message"]

        # Preprocess input
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

        # Predict emotion
        prediction = model.predict(padded)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        emotion = id2label.get(predicted_class, "neutral")
        reply = generate_supportive_reply(text, emotion)

        return jsonify({
            "input": text,
            "predicted_emotion": emotion,
            "confidence": confidence,
            "chatbot_reply": reply
        })

    except Exception as e:
        print(f"[Server Error] {e}")
        return jsonify({"error": str(e)}), 500

# ===== Run Server =====
if __name__ == "__main__":
    app.run(debug=True)
