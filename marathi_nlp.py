"""
marathi_nlp.py — NLP Utilities for Marathi Chatbot
-----------------------------------------------------
Pipeline:
  1. Exact match lookup  (handles short words like बाय, मदत)
  2. TF-IDF vectorization (char n-grams 2-4 + word n-grams 1-2)
  3. Cosine Similarity ensemble (55% char + 45% word)
  4. Confidence threshold gate → fallback if too weak
  5. Entity extraction (cities, numbers, question type)
"""

import re
import json
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────

MARATHI_PUNCT = r'[।,!?\.\'\"؟،\-\(\)\[\]{}:;]'

def clean_text(text: str) -> str:
    text = re.sub(MARATHI_PUNCT, ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> list:
    return clean_text(text).split()


# ─────────────────────────────────────────────
#  Intent Recognizer
# ─────────────────────────────────────────────

class MarathiIntentRecognizer:
    """
    NLP intent recognizer using:
      1. Exact-match shortcut for known patterns
      2. TF-IDF (char n-grams + word n-grams) + cosine similarity
      3. Confidence threshold → fallback on weak matches
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        )
        self.patterns = []
        self.tags = []
        self.intents = {}
        self._exact_map = {}
        self.char_matrix = None
        self.word_matrix = None
        self.is_fitted = False

    # ── Training ──────────────────────────────

    def train(self, intents_data: dict):
        self.patterns = []
        self.tags = []
        self.intents = {}
        self._exact_map = {}

        for intent in intents_data['intents']:
            tag = intent['tag']
            self.intents[tag] = intent['responses']
            if tag == 'fallback':
                continue
            for pattern in intent['patterns']:
                cleaned = clean_text(pattern)
                if cleaned:
                    self.patterns.append(cleaned)
                    self.tags.append(tag)
                    # Build exact-match lookup for every pattern
                    self._exact_map[cleaned.lower()] = tag

        self.char_matrix = self.vectorizer.fit_transform(self.patterns)
        self.word_matrix = self.word_vectorizer.fit_transform(self.patterns)
        self.is_fitted = True

        print(f"[NLP] Trained on {len(self.patterns)} patterns, {len(self.intents)} intents")
        print(f"[NLP] Exact-match entries: {len(self._exact_map)}")
        print(f"[NLP] Char-ngram vocab: {len(self.vectorizer.vocabulary_)}")
        print(f"[NLP] Word vocab: {len(self.word_vectorizer.vocabulary_)}")

    # ── Prediction ────────────────────────────

    def predict(self, user_input: str) -> tuple:
        """
        Predict intent. Returns (tag, confidence).
        Pipeline:
          1. Exact match  → 1.0 confidence (handles: बाय, मदत, नमस्कार...)
          2. TF-IDF cosine similarity
          3. Threshold gate → fallback
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        cleaned = clean_text(user_input)
        if not cleaned:
            return 'fallback', 0.0

        # ── Step 1: Exact match ──────────────────────────────────────────
        if cleaned.lower() in self._exact_map:
            return self._exact_map[cleaned.lower()], 1.0

        # ── Step 2: TF-IDF + Cosine Similarity ──────────────────────────
        char_vec = self.vectorizer.transform([cleaned])
        word_vec = self.word_vectorizer.transform([cleaned])

        char_sims = cosine_similarity(char_vec, self.char_matrix).flatten()
        word_sims = cosine_similarity(word_vec, self.word_matrix).flatten()

        # Weighted ensemble (char n-grams capture Marathi morphology better)
        combined = 0.55 * char_sims + 0.45 * word_sims

        best_idx = int(np.argmax(combined))
        best_score = float(combined[best_idx])

        # ── Step 3: Confidence threshold gate ───────────────────────────
        if best_score < self.threshold:
            return 'fallback', best_score

        return self.tags[best_idx], best_score

    def get_response(self, user_input: str) -> tuple:
        """Full pipeline → (response_text, intent_tag, confidence)."""
        tag, confidence = self.predict(user_input)
        responses = self.intents.get(tag, self.intents.get('fallback', ['माफ करा!']))
        return random.choice(responses), tag, confidence

    # ── Persistence ───────────────────────────

    def save(self, filepath: str):
        data = {
            'vectorizer': self.vectorizer,
            'word_vectorizer': self.word_vectorizer,
            'patterns': self.patterns,
            'tags': self.tags,
            'intents': self.intents,
            'char_matrix': self.char_matrix,
            'word_matrix': self.word_matrix,
            'threshold': self.threshold,
            'exact_map': self._exact_map,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[NLP] Model saved → {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.word_vectorizer = data['word_vectorizer']
        self.patterns = data['patterns']
        self.tags = data['tags']
        self.intents = data['intents']
        self.char_matrix = data['char_matrix']
        self.word_matrix = data['word_matrix']
        self.threshold = data['threshold']
        self._exact_map = data.get('exact_map', {})
        self.is_fitted = True
        print(f"[NLP] Model loaded ← {filepath}")


# ─────────────────────────────────────────────
#  Entity Extractor
# ─────────────────────────────────────────────

MARATHI_NUMBERS = {
    'एक': 1, 'दोन': 2, 'तीन': 3, 'चार': 4, 'पाच': 5,
    'सहा': 6, 'सात': 7, 'आठ': 8, 'नऊ': 9, 'दहा': 10
}

CITY_NAMES = [
    'मुंबई', 'पुणे', 'नागपूर', 'नाशिक', 'औरंगाबाद',
    'कोल्हापूर', 'सोलापूर', 'ठाणे', 'अमरावती', 'लातूर'
]

def extract_entities(text: str) -> dict:
    entities = {'cities': [], 'numbers': [], 'question_type': None}
    for city in CITY_NAMES:
        if city in text:
            entities['cities'].append(city)
    for word, num in MARATHI_NUMBERS.items():
        if word in text:
            entities['numbers'].append(num)
    question_words = {
        'काय': 'WHAT', 'कोण': 'WHO', 'कुठे': 'WHERE',
        'कधी': 'WHEN', 'कसे': 'HOW', 'का': 'WHY', 'किती': 'HOW_MUCH'
    }
    for word, qtype in question_words.items():
        if word in text:
            entities['question_type'] = qtype
            break
    return entities
