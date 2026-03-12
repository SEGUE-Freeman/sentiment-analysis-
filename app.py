import streamlit as st
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

from bert_services import analyze_bert
from llm_services import analyze_llm

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse de Sentiment",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Analyse de Sentiment — BERT vs LLM")
st.caption("Comparez une approche classique (DistilBERT) et une approche moderne (GPT-4o mini)")

# ── Exemples ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Ce produit est absolument fantastique, je l'adore !",
    "Service client horrible, jamais je ne recommanderai cette marque.",
    "C'est correct, ni trop bien ni trop mal.",
    "Je suis partagé : l'écran est magnifique mais la batterie dure 3h...",
]

st.markdown("**Exemples rapides :**")
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"Exemple {i+1}", use_container_width=True):
        st.session_state["input_text"] = ex
        st.rerun()

# ── Zone de saisie ────────────────────────────────────────────────────────────
# Initialiser la clé si elle n'existe pas encore
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

text = st.text_area(
    "Entrez votre texte :",
    height=120,
    placeholder="Tapez ou collez un texte à analyser…",
    key="input_text",
)

analyze_btn = st.button("🔍 Analyser", type="primary", disabled=not text.strip())

# ── Analyse ───────────────────────────────────────────────────────────────────
if analyze_btn and text.strip():
    with st.spinner("Analyse en cours (BERT + GPT en parallèle)…"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_bert = executor.submit(analyze_bert, text.strip())
            future_llm  = executor.submit(analyze_llm,  text.strip())

            bert_result = bert_error = None
            llm_result  = llm_error  = None

            try:
                bert_result = future_bert.result(timeout=15)
            except Exception as e:
                bert_error = str(e)

            try:
                llm_result = future_llm.result(timeout=30)
            except Exception as e:
                llm_error = str(e)

    st.divider()

    col_bert, col_llm = st.columns(2)

    # ── Colonne BERT ──────────────────────────────────────────────────────────
    with col_bert:
        st.subheader("⬡ BERT — DistilBERT SST-2")
        st.caption("Classification supervisée · Binaire")

        if bert_error:
            st.error(f"Erreur : {bert_error}")
        else:
            label = bert_result["label"]
            score = bert_result["score"]

            color = "green" if label == "Positif" else "red"
            st.markdown(f"### :{color}[{label}]")
            st.metric("Score de confiance", f"{round(score * 100, 1)}%")

            st.markdown("**Distribution des scores :**")
            for item in bert_result["all"]:
                st.markdown(f"`{item['label']}`")
                st.progress(item["score"])

            st.caption(f"Modèle : {bert_result['model']}")

    # ── Colonne LLM ───────────────────────────────────────────────────────────
    with col_llm:
        st.subheader("◈ LLM — GPT-4o mini")
        st.caption("Raisonnement génératif · Contextuel")

        if llm_error:
            st.error(f"Erreur : {llm_error}")
        else:
            label     = llm_result["label"]
            score     = llm_result["score"]
            emotions  = llm_result.get("emotions", [])
            nuances   = llm_result.get("nuances", "")
            confidence = llm_result.get("confidence", "")
            intensite  = llm_result.get("intensite", "")

            color_map = {
                "Positif": "green", "Négatif": "red",
                "Neutre": "blue",   "Mitigé": "orange",
            }
            color = color_map.get(label, "gray")
            st.markdown(f"### :{color}[{label}]")

            m1, m2, m3 = st.columns(3)
            m1.metric("Score",      f"{round(score * 100, 1)}%")
            m2.metric("Confiance",  confidence)
            m3.metric("Intensité",  intensite)

            if emotions:
                st.markdown("**Émotions détectées :**")
                st.write(" · ".join(f"`{e}`" for e in emotions))

            if nuances:
                st.info(f"💬 {nuances}")

            st.caption(f"Modèle : {llm_result['model']}")