import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """Analyse le sentiment du texte suivant et réponds UNIQUEMENT en JSON valide, sans markdown.

Texte : "{text}"

Format attendu :
{{
  "label": "Positif" | "Négatif" | "Neutre" | "Mitigé",
  "score": 0.0-1.0,
  "confidence": "Faible" | "Moyen" | "Élevé",
  "emotions": ["émotion1", "émotion2"],
  "nuances": "explication courte",
  "intensite": "Faible" | "Modérée" | "Forte"
}}"""

def analyze_llm(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # rapide et peu coûteux, change en gpt-4o si tu veux
        messages=[
            {"role": "system", "content": "Tu es un expert en analyse de sentiment. Réponds uniquement en JSON."},
            {"role": "user",   "content": PROMPT.format(text=text)},
        ],
        temperature=0,         # résultats déterministes
    )
    raw = response.choices[0].message.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    result["model"] = "GPT-4o mini"
    return result