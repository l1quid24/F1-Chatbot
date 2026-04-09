"""
app.py  —  F1 Strategy Chatbot (RAG + Gemini + SpaCy NER)
Usage: python -m streamlit run app.py
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import spacy

from track_vis import render_sidebar_controls, render_track_output, detect_vis_intent

# ── Configuration ──────────────────────────────────────────────────────────

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
N_RESULTS      = 4

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ── Load SpaCy ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ── Track name mapping (SpaCy extracts raw text, we map to KB keys) ────────

TRACK_ALIASES = {
    "monaco":           "Monaco Grand Prix",
    "monte carlo":      "Monaco Grand Prix",
    "monza":            "Italian Grand Prix",
    "italy":            "Italian Grand Prix",
    "italian":          "Italian Grand Prix",
    "silverstone":      "British Grand Prix",
    "britain":          "British Grand Prix",
    "british":          "British Grand Prix",
    "uk":               "British Grand Prix",
    "spa":              "Belgian Grand Prix",
    "belgium":          "Belgian Grand Prix",
    "belgian":          "Belgian Grand Prix",
    "suzuka":           "Japanese Grand Prix",
    "suzuka circuit":   "Japanese Grand Prix",
    "japan":            "Japanese Grand Prix",
    "japanese":         "Japanese Grand Prix",
    "singapore":        "Singapore Grand Prix",
    "bahrain":          "Bahrain Grand Prix",
    "jeddah":           "Saudi Arabian Grand Prix",
    "saudi":            "Saudi Arabian Grand Prix",
    "saudi arabia":     "Saudi Arabian Grand Prix",
    "melbourne":        "Australian Grand Prix",
    "australia":        "Australian Grand Prix",
    "australian":       "Australian Grand Prix",
    "albert park":      "Australian Grand Prix",
    "baku":             "Azerbaijan Grand Prix",
    "azerbaijan":       "Azerbaijan Grand Prix",
    "miami":            "Miami Grand Prix",
    "barcelona":        "Spanish Grand Prix",
    "spain":            "Spanish Grand Prix",
    "spanish":          "Spanish Grand Prix",
    "canada":           "Canadian Grand Prix",
    "montreal":         "Canadian Grand Prix",
    "canadian":         "Canadian Grand Prix",
    "spielberg":        "Austrian Grand Prix",
    "austria":          "Austrian Grand Prix",
    "austrian":         "Austrian Grand Prix",
    "red bull ring":    "Austrian Grand Prix",
    "budapest":         "Hungarian Grand Prix",
    "hungary":          "Hungarian Grand Prix",
    "hungarian":        "Hungarian Grand Prix",
    "zandvoort":        "Dutch Grand Prix",
    "netherlands":      "Dutch Grand Prix",
    "dutch":            "Dutch Grand Prix",
    "holland":          "Dutch Grand Prix",
    "interlagos":       "São Paulo Grand Prix",
    "brazil":           "São Paulo Grand Prix",
    "são paulo":        "São Paulo Grand Prix",
    "sao paulo":        "São Paulo Grand Prix",
    "las vegas":        "Las Vegas Grand Prix",
    "vegas":            "Las Vegas Grand Prix",
    "lusail":           "Qatar Grand Prix",
    "qatar":            "Qatar Grand Prix",
    "abu dhabi":        "Abu Dhabi Grand Prix",
    "yas marina":       "Abu Dhabi Grand Prix",
    "austin":           "United States Grand Prix",
    "cota":             "United States Grand Prix",
    "united states":    "United States Grand Prix",
    "usa":              "United States Grand Prix",
    "mexico":           "Mexico City Grand Prix",
    "mexico city":      "Mexico City Grand Prix",
}

DRIVER_ALIASES = {
    "verstappen":   "Max Verstappen",
    "max":          "Max Verstappen",
    "hamilton":     "Lewis Hamilton",
    "lewis":        "Lewis Hamilton",
    "leclerc":      "Charles Leclerc",
    "charles":      "Charles Leclerc",
    "norris":       "Lando Norris",
    "lando":        "Lando Norris",
    "russell":      "George Russell",
    "george":       "George Russell",
    "alonso":       "Fernando Alonso",
    "fernando":     "Fernando Alonso",
}

# ── NER extraction ─────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict:
    """
    Use SpaCy NER to extract tracks and drivers from user input.
    Returns dict with keys: 'tracks' (list), 'drivers' (list), 'raw_entities' (list)
    """
    doc = nlp(text)
    text_lower = text.lower()

    raw_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract tracks
    found_tracks = []

    # Method 1: SpaCy detected GPE / LOC / FAC entities
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC", "ORG"):
            key = ent.text.lower()
            if key in TRACK_ALIASES:
                gp = TRACK_ALIASES[key]
                if gp not in found_tracks:
                    found_tracks.append(gp)

    # Method 2: direct substring scan (catches multi-word aliases SpaCy may miss)
    for alias, gp_name in TRACK_ALIASES.items():
        if alias in text_lower and gp_name not in found_tracks:
            found_tracks.append(gp_name)

    # Extract drivers
    found_drivers = []
    for alias, driver_name in DRIVER_ALIASES.items():
        if alias in text_lower and driver_name not in found_drivers:
            found_drivers.append(driver_name)

    return {
        "tracks":       found_tracks,
        "drivers":      found_drivers,
        "raw_entities": raw_entities,
    }

# ── Load ChromaDB ──────────────────────────────────────────────────────────

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(
        name="f1_knowledge",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

collection = load_collection()

# ── RAG core ───────────────────────────────────────────────────────────────

def retrieve(query: str, entities: dict, n: int = N_RESULTS) -> list[str]:
    """
    Retrieve relevant chunks.
    If NER found specific tracks, do targeted retrieval for those tracks first,
    then fill remaining slots with semantic search.
    """
    chunks   = []
    seen_ids = set()

    # Targeted retrieval for detected tracks
    if entities["tracks"]:
        # Detect if question is safety-car related
        sc_keywords = ["safety car", "safety-car", "sc ", "deployed", "neutralised"]
        is_sc_query = any(kw in query.lower() for kw in sc_keywords)
        query_text_template = "{track} safety car probability advice" if is_sc_query else "{track} tyre strategy"

        for track in entities["tracks"][:2]:
            results = collection.query(
                query_texts=[query_text_template.format(track=track)],
                n_results=2,
            )

            
            for doc, mid in zip(results["documents"][0], results["ids"][0]):
                if mid not in seen_ids:
                    chunks.append(doc)
                    seen_ids.add(mid)

    # Fill remaining slots with general semantic search
    remaining = max(1, n - len(chunks))
    results = collection.query(query_texts=[query], n_results=remaining + 2)
    for doc, mid in zip(results["documents"][0], results["ids"][0]):
        if mid not in seen_ids and len(chunks) < n:
            chunks.append(doc)
            seen_ids.add(mid)

    return chunks


def build_prompt(query: str, chunks: list[str], entities: dict, history: list[dict] = []) -> str:
    context = "\n\n---\n\n".join(chunks)

    entity_hint = ""
    if entities["tracks"]:
        entity_hint += f"The user is asking about: {', '.join(entities['tracks'])}.\n"
    if entities["drivers"]:
        entity_hint += f"Drivers mentioned: {', '.join(entities['drivers'])}.\n"

    # Format recent conversation history (last 3 exchanges)
    history_text = ""
    if history:
        recent = history[-6:]  # 3 user + 3 assistant messages
        history_text = "Recent conversation:\n"
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    return f"""You are an F1 strategy assistant helping fans and enthusiasts understand
Formula 1 race strategy. Use the context below to answer the question.

Rules:
- Be clear and friendly, avoid overly technical jargon.
- If you use a technical term (like "pit stop window"), briefly explain it.
- Base your answer on the context provided. If the context does not contain
  enough information, say so honestly rather than making things up.
- Keep answers concise but informative (3-5 sentences is usually enough).

{entity_hint}
Context:
{context}

{history_text}Question: {query}

Answer:"""


def ask(query: str, history: list[dict]) -> tuple[str, dict]:
    """Full RAG pipeline: NER → retrieve → build prompt → generate."""
    entities = extract_entities(query)
    chunks   = retrieve(query, entities)
    prompt   = build_prompt(query, chunks, entities, history)
    response = model.generate_content(prompt)
    return response.text, entities


# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Strategy Chatbot",
    page_icon="🏎️",
    layout="centered"
)

st.title("🏎️ F1 Strategy Chatbot")
st.caption(
    "Ask me anything about Formula 1 race strategy — tyre choices, "
    "safety car decisions, circuit characteristics, and more."
)

# Suggested questions
st.markdown("**Try asking:**")
cols = st.columns(2)
suggestions = [
    "What tyre strategy should I use at Monaco?",
    "What should a team do when the safety car comes out at Singapore?",
    "What makes Monza different from Monaco?",
    "Why do teams use Medium tyres at the start?",
]
for i, suggestion in enumerate(suggestions):
    if cols[i % 2].button(suggestion, use_container_width=True):
        st.session_state["pending_input"] = suggestion
        st.rerun()



# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("entities") and (
            msg["entities"]["tracks"] or msg["entities"]["drivers"]
        ):
            with st.expander("Entities detected by SpaCy NER", expanded=False):
                if msg["entities"]["tracks"]:
                    st.write("**Tracks:**", ", ".join(msg["entities"]["tracks"]))
                if msg["entities"]["drivers"]:
                    st.write("**Drivers:**", ", ".join(msg["entities"]["drivers"]))
                if msg["entities"]["raw_entities"]:
                    st.write("**Raw SpaCy output:**",
                             str(msg["entities"]["raw_entities"]))

# Handle suggested question click
user_input = st.chat_input("Ask about F1 strategy...")
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# Process input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vis_gp = detect_vis_intent(user_input, extract_entities(user_input)["tracks"])
                if vis_gp:
                    st.session_state["vis_track"] = vis_gp
                    confirm_msg = f"Sure! Loading the track map for **{vis_gp}** 🗺️"
                    st.markdown(confirm_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": confirm_msg,
                        "entities": {"tracks": [vis_gp], "drivers": [], "raw_entities": []},
                    })
                    st.rerun()

                answer, entities = ask(user_input, st.session_state.messages)
                st.markdown(answer)

                # Show NER results in expander (useful for demo / marking)
                if entities["tracks"] or entities["drivers"]:
                    with st.expander(
                        "Entities detected by SpaCy NER", expanded=False
                    ):
                        if entities["tracks"]:
                            st.write("**Tracks:**", ", ".join(entities["tracks"]))
                        if entities["drivers"]:
                            st.write("**Drivers:**",
                                     ", ".join(entities["drivers"]))
                        if entities["raw_entities"]:
                            st.write("**Raw SpaCy output:**",
                                     str(entities["raw_entities"]))

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  answer,
                    "entities": entities,
                })
            except Exception as e:
                st.error(f"Sorry, something went wrong: {str(e)}")

# ── Track visualisation output (main area) ────────────────────────────────
render_track_output()

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.write(
        "1. **SpaCy NER** extracts track and driver names from your question\n"
        "2. Relevant facts are retrieved from the knowledge base\n"
        "3. **Gemini** generates a natural language answer based on those facts"
    )
    st.divider()
    st.write("**Data sources:**")
    st.write("- Tyre strategy records (top 10 finishers per race)")
    st.write("- Circuit telemetry clusters (high-speed / mixed / street)")
    st.write("- F1 domain knowledge ontology")
    st.divider()
    render_sidebar_controls()
    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()