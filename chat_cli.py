"""
chat_cli.py — Command-line Marathi Chatbot
Run: python chat_cli.py
"""

import json
import os
from marathi_nlp import MarathiIntentRecognizer, extract_entities

MODEL_PATH = 'marathi_model.pkl'
INTENTS_PATH = 'intents.json'

def main():
    print("=" * 55)
    print("  🟠 मराठी NLP चॅटबॉट  (Terminal Mode)")
    print("  बाहेर पडण्यासाठी 'quit' किंवा 'exit' टाइप करा")
    print("=" * 55)
    print()

    recognizer = MarathiIntentRecognizer(confidence_threshold=0.60)

    if os.path.exists(MODEL_PATH):
        recognizer.load(MODEL_PATH)
    else:
        print("Training model...")
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        recognizer.train(intents_data)
        recognizer.save(MODEL_PATH)

    print("\nबॉट: नमस्कार! मी मराठी NLP चॅटबॉट आहे. मला काय विचारायचे आहे?\n")

    while True:
        try:
            user_input = input("तुम्ही: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nबॉट: निरोप! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'बाय', 'निरोप'):
            print("बॉट: निरोप! पुन्हा भेटू. 👋")
            break

        response, intent, confidence = recognizer.get_response(user_input)
        entities = extract_entities(user_input)

        print(f"\nबॉट: {response}")
        print(f"     [Intent: {intent} | Confidence: {confidence*100:.1f}%", end="")
        if entities['cities']:
            print(f" | Cities: {entities['cities']}", end="")
        if entities['question_type']:
            print(f" | Q-type: {entities['question_type']}", end="")
        print("]\n")

if __name__ == '__main__':
    main()
