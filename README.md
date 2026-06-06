# Marathi-Chatbot
🟠 मराठी NLP Chatbot — Intent detection using TF-IDF + Cosine Similarity | Flask web app | 20 Marathi topics | Built with Python &amp; scikit-learn
# 🟠 मराठी NLP चॅटबॉट
A rule-based Marathi-language chatbot built with NLP — **no external AI API needed**. Uses TF-IDF vectorization and cosine similarity to detect user intent from Devanagari text, with a polished Flask web interface.

---

## ✨ Features

- 🧠 **Pure NLP** — TF-IDF + Cosine Similarity, zero API calls
- 🔤 **Devanagari-aware** — Character n-grams handle Marathi morphology
- ⚡ **Exact-match shortcut** — Single words (बाय, मदत) match instantly at 100%
- 💬 **20 intent categories** — Maharashtra, Shivaji, food, festivals, Python, cricket & more
- ⚠️ **Smart fallback** — Unknown topics show a polite error card with topic suggestions
- 🌐 **REST API** — `/chat`, `/intents`, `/retrain` endpoints
- 💻 **CLI mode** — Run without Flask for quick testing

---

## 📸 Demo

```
User: शिवाजी महाराज कोण होते?
Bot:  छत्रपती शिवाजी महाराज (१६३०-१६८०) हे मराठा साम्राज्याचे संस्थापक होते...
      [Intent: shivaji | Confidence: 100%]

User: पायथन म्हणजे काय?
Bot:  Python एक लोकप्रिय programming language आहे! 🐍
      [Intent: python_programming | Confidence: 100%]

User: Einstein kaun the?
Bot:  ⚠️ हे मला माहीत नाही!
      [Shows clickable topic suggestions]
```

---

## 🧠 NLP Pipeline

```
User Input (Marathi text)
        │
        ▼
┌─────────────────────────┐
│  1. Text Cleaning        │  Regex: remove ।,!?. normalize spaces
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  2. Tokenization         │  Whitespace split (works natively with Devanagari)
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  3. Exact Match Lookup   │  Dictionary check → 100% confidence, skips TF-IDF
└─────────────────────────┘
        │ (no exact match)
        ▼
┌──────────────────────────────────────────┐
│  4. TF-IDF Vectorization                  │
│     • Char n-grams (2–4) — 55% weight    │  ← handles morphological variants
│     • Word n-grams (1–2) — 45% weight    │  ← handles full phrases
│     • Ensemble cosine similarity          │
└──────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  5. Confidence Gate      │  score < 0.60 → fallback error card
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  6. Entity Extraction    │  Cities · Numbers · Question type (WHO/WHAT/WHERE...)
└─────────────────────────┘
        │
        ▼
      Response
```

### Why character n-grams for Marathi?

Marathi is morphologically rich — the same root appears in many forms:

| Root | Variants |
|------|---------|
| जा (go) | जातो, जाते, जाणार, गेला, जाऊ |
| सांग (tell) | सांगा, सांगतो, सांगितले |

Character n-grams (e.g. `जात`, `ाते`, `णार`) capture these shared substrings, making intent matching robust **without needing a Marathi stemmer or lemmatizer**.

---

## 📁 Project Structure

```
marathi_nlp_chatbot/
├── app.py                ← Flask web server + REST API
├── chat_cli.py           ← Terminal chatbot (no Flask needed)
├── marathi_nlp.py        ← Core NLP engine
│                            TF-IDF vectorizer, cosine similarity,
│                            exact-match lookup, entity extractor
├── intents.json          ← 20 intent categories, 252 Marathi patterns
├── marathi_model.pkl     ← Pre-trained model (auto-generated)
├── requirements.txt
└── templates/
    └── index.html        ← Chat UI with live NLP stats panel
```

---

## 🚀 Setup & Run

### 1. Clone & Install

```bash
git clone https://github.com/your-username/marathi-nlp-chatbot.git
cd marathi-nlp-chatbot

pip install -r requirements.txt
```

### 2a. Web App

```bash
python app.py
```
Open **http://localhost:5000** in your browser.

The model trains automatically on first run and saves to `marathi_model.pkl`.

### 2b. Terminal Mode

```bash
python chat_cli.py
```

```
══════════════════════════════════════════════════════
  🟠 मराठी NLP चॅटबॉट  (Terminal Mode)
══════════════════════════════════════════════════════

बॉट: नमस्कार! मी मराठी NLP चॅटबॉट आहे. मला काय विचारायचे आहे?

तुम्ही: शिवाजी महाराज
बॉट: छत्रपती शिवाजी महाराज (१६३०-१६८०) हे मराठा साम्राज्याचे...
     [Intent: shivaji | Confidence: 100.0%]
```

---

## 💬 Supported Intents (20 categories)

| Tag | Sample Patterns | Sample Response |
|-----|----------------|-----------------|
| `namaskaar` | नमस्कार, हॅलो, कसे आहात | नमस्कार! मी तुमच्या सेवेत आहे 😊 |
| `nirop` | बाय, निरोप, जातो | निरोप! पुन्हा भेटू 👋 |
| `dhanyavaad` | धन्यवाद, थँक्स | आपले स्वागत आहे! |
| `maharashtra` | महाराष्ट्राची राजधानी, जिल्हे | मुंबई ही राजधानी आहे... |
| `shivaji` | शिवाजी महाराज कोण होते | छत्रपती शिवाजी महाराज (१६३०-१६८०)... |
| `jevan` | मराठी जेवण, वडापाव | वडापाव हा महाराष्ट्राचा 'बर्गर' आहे! |
| `san_utsav` | गणेश चतुर्थी, दिवाळी | महाराष्ट्रातील प्रमुख सण... |
| `marathi_bhaasha` | मराठी भाषा, देवनागरी | मराठी ही ९०० वर्षांपूर्वीची भाषा... |
| `itihas` | मराठा इतिहास, पेशवे | मराठा साम्राज्याचा इतिहास... |
| `kavita` | मराठी कविता सांगा | येरे येरे पावसा... 🌧️ |
| `vinod` | एक विनोद सांगा | एक शिक्षक वर्गात... 😂 |
| `python_programming` | पायथन म्हणजे काय | Python एक लोकप्रिय language आहे! 🐍 |
| `sports` | क्रिकेट, IPL, सचिन | क्रिकेट हा भारताचा लोकप्रिय खेळ! 🏏 |
| `havaman` | हवामान, पाऊस | imd.gov.in वर तपासा 🌦️ |
| `vel` | किती वाजले, तारीख | फोनवर पाहा ⏰ |
| `nav` | तुमचे नाव काय | मी मराठी NLP बॉट आहे 🤖 |
| `vay` | तुमचे वय किती | AI ला वय नसतं! 😄 |
| `madad` | मदत, help | मी या topics वर मदत करतो... |
| `technology` | coding, software, AI | तंत्रज्ञानाच्या क्षेत्रात... |
| `fallback` | anything unknown | ⚠️ Polite error card with topic chips |

---

## ⚠️ Fallback Behavior

When a user asks something outside the known intents (confidence < 60%), the chatbot shows a polite error card instead of a wrong answer:

```
⚠️  हे मला माहीत नाही!
─────────────────────────────────────
माफ करा, "Einstein kaun the" बद्दल
माझ्याकडे माहिती नाही.

मी फक्त या topics वर बोलू शकतो:
[नमस्कार] [शिवाजी महाराज] [महाराष्ट्र]
[मराठी जेवण] [Python] [Cricket] ...
─────────────────────────────────────
```

Clicking any topic chip auto-fills the input box.

---

## 📡 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message, get intent + response |
| `GET` | `/intents` | List all available intent tags |
| `POST` | `/retrain` | Retrain model from updated intents.json |

### POST `/chat`

**Request**
```json
{ "message": "पायथन म्हणजे काय" }
```

**Response**
```json
{
  "response": "Python एक लोकप्रिय programming language आहे! 🐍...",
  "intent": "python_programming",
  "confidence": 100.0,
  "is_fallback": false,
  "entities": {
    "cities": [],
    "question_type": "WHAT",
    "numbers": []
  }
}
```

**Fallback Response**
```json
{
  "response": "माफ करा, हा विषय मला नक्की माहीत नाही...",
  "intent": "fallback",
  "confidence": 18.2,
  "is_fallback": true,
  "entities": {}
}
```

---

## 🔧 Adding New Intents

Edit `intents.json` and add a new object to the `intents` array:

```json
{
  "tag": "cricket",
  "patterns": [
    "क्रिकेट बद्दल सांगा",
    "IPL कधी आहे",
    "सचिन तेंडुलकर"
  ],
  "responses": [
    "क्रिकेट हा भारताचा सर्वात लोकप्रिय खेळ आहे! 🏏"
  ]
}
```

Then retrain — two options:

```bash
# Option A: Restart the server (auto-retrains if no .pkl found)
rm marathi_model.pkl
python app.py

# Option B: API call (no restart needed)
curl -X POST http://localhost:5000/retrain
```

**Tips for good patterns:**
- Add at least 8–10 patterns per intent
- Use different sentence structures (question form, statement form)
- Include common typo variants
- Avoid patterns that overlap with other intents

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| Flask | 3.0 | Web server & REST API |
| scikit-learn | 1.4 | TF-IDF vectorizer, cosine similarity |
| NumPy | 1.26 | Vector math |

No external NLP APIs. No internet connection needed at runtime.

---

## 📊 Model Stats

| Metric | Value |
|--------|-------|
| Training patterns | 252 |
| Intent categories | 20 |
| Char n-gram vocab | 2,870 features |
| Word vocab | 321 features |
| Confidence threshold | 60% |
| Exact-match entries | 251 |

---

## 📄 License

MIT License — free to use, modify, and distribute.
