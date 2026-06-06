"""
app.py — Flask Web Server for Marathi NLP Chatbot
"""

import json
import os
from flask import Flask, render_template, request, jsonify
from marathi_nlp import MarathiIntentRecognizer, extract_entities

app = Flask(__name__)

# ── Load & Train Model ────────────────────────────────────────────────────────

MODEL_PATH = 'marathi_model.pkl'
INTENTS_PATH = 'intents.json'

recognizer = MarathiIntentRecognizer(confidence_threshold=0.60)

def load_model():
    if os.path.exists(MODEL_PATH):
        recognizer.load(MODEL_PATH)
        print("[App] Loaded pre-trained model.")
    else:
        print("[App] Training model from intents.json ...")
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        recognizer.train(intents_data)
        recognizer.save(MODEL_PATH)

load_model()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # NLP Pipeline
    response, intent, confidence = recognizer.get_response(user_message)
    entities = extract_entities(user_message)

    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence * 100, 1),
        'entities': entities,
        'is_fallback': intent == 'fallback'
    })


@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to retrain model (useful during development)."""
    with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    recognizer.train(intents_data)
    recognizer.save(MODEL_PATH)
    return jsonify({'status': 'Model retrained successfully'})


@app.route('/intents', methods=['GET'])
def get_intents():
    """Return all available intent tags."""
    tags = [t for t in recognizer.intents.keys() if t != 'fallback']
    return jsonify({'intents': tags, 'count': len(tags)})


if __name__ == '__main__':
    print("\n🟠 मराठी NLP Chatbot starting on http://localhost:5000\n")
    app.run(debug=True, port=5000)
