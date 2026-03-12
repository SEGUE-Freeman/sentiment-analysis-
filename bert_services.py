from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect, LangDetectException

vader = SentimentIntensityAnalyzer()

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang  # "fr", "en", "es", etc.
    except LangDetectException:
        return "en"  # fallback anglais

def analyze_english(text: str) -> dict:
    scores = vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positif"
        score = round((compound + 1) / 2, 4)
    elif compound <= -0.05:
        label = "Négatif"
        score = round((1 - compound) / 2, 4)
    else:
        label = "Neutre"
        score = round(0.5 + compound, 4)

    return {
        "label": label,
        "score": score,
        "all": [
            {"label": "Positif", "score": round(scores["pos"], 4)},
            {"label": "Neutre",  "score": round(scores["neu"], 4)},
            {"label": "Négatif", "score": round(scores["neg"], 4)},
        ],
        "model": "VADER (anglais)",
        "langue": "Anglais 🇬🇧",
    }

def analyze_french(text: str) -> dict:
    blob = TextBlob(text)
    # polarity entre -1.0 et 1.0
    polarity = blob.sentiment.polarity

    if polarity > 0.05:
        label = "Positif"
        score = round((polarity + 1) / 2, 4)
    elif polarity < -0.05:
        label = "Négatif"
        score = round((1 - abs(polarity)) / 2, 4)
    else:
        label = "Neutre"
        score = 0.5

    # Construire une distribution approximative
    pos = round(max(0, polarity), 4)
    neg = round(max(0, -polarity), 4)
    neu = round(1 - pos - neg, 4)

    return {
        "label": label,
        "score": score,
        "all": [
            {"label": "Positif", "score": pos},
            {"label": "Neutre",  "score": neu},
            {"label": "Négatif", "score": neg},
        ],
        "model": "TextBlob (français)",
        "langue": "Français 🇫🇷",
    }

def analyze_bert(text: str) -> dict:
    lang = detect_language(text)

    if lang == "fr":
        return analyze_french(text)
    else:
        return analyze_english(text)