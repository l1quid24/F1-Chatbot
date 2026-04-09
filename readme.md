# F1 Strategy Chatbot

An intelligent Formula 1 race strategy assistant powered by RAG, Google Gemini, and SpaCy NER.

---

## Project Structure

```
├── app.py                      # Main application
├── Final Project.py            # Knowledge base builder
├── track_vis.py                # Track visualizer
├── strategies_kb.json          # Tyre strategy data (per race)
├── track_info.json             # Circuit cluster and speed data
├── cluster_safetycar_rate.csv  # Safety car rate per cluster
├── F1Strategist_Ontology.owl   # F1 domain ontology
├── chroma_db/                  # ChromaDB vector store (auto-generated)
```

---

## Core Files

### 1. `Final Project.py`

Converts raw data files into a searchable vector knowledge base stored in ChromaDB. Run this once before starting the app.

**What it does:**
- Reads `strategies_kb.json`, `track_info.json`, and `cluster_safetycar_rate.csv`
- Builds four types of text chunks:
  - **Per-track strategy summaries** — tyre compound sequences used by top 10 finishers at each circuit
  - **Per-track safety car advice** — circuit-specific safety car guidance including historical appearance rates (e.g. High-speed circuits: 75%, Mixed/Street circuits: 50%)
  - **Cluster overviews** — characteristics of High-speed, Mixed, and Low-speed circuit types derived from telemetry clustering
  - **General F1 knowledge** — tyre compound explanations, pit stop basics, safety car mechanics
- Stores all chunks as vector embeddings in ChromaDB using the default embedding function

**Run once with:**
```bash
python build_knowledge_base.py
```

---

### 2. `App.py`

The main Streamlit application. Handles the full RAG pipeline and chat interface.

**What it does:**

**SpaCy NER (Named Entity Recognition)**
- Loads `en_core_web_sm` to extract location and person entities from user input
- Uses two-pass detection: SpaCy's GPE/LOC labels + direct alias substring scan
- Maps extracted text to knowledge base keys (e.g. "Suzuka" → "Japanese Grand Prix", "Hamilton" → "Lewis Hamilton")
- Displays detected entities in an expandable panel below each response

**RAG**
- `extract_entities()` — identifies tracks and drivers in the user query
- `retrieve()` — if a track is detected, performs targeted retrieval for that track first; fills remaining slots with semantic search. Detects safety car queries and adjusts the retrieval query accordingly to prioritise safety car chunks
- `build_prompt()` — assembles context chunks, entity hints, and recent conversation history into a structured prompt
- `ask()` — runs the full pipeline and returns the generated answer

**Conversation Memory**
- Maintains chat history in `st.session_state`
- Passes the last 3 exchanges (6 messages) to Gemini as context, allowing follow-up questions

**Track Visualization Trigger**
- Detects visualization intent in user messages (keywords: "load", "show", "draw", "circuit", "track map", etc.)
- If intent and a track name are both detected, automatically triggers the track visualizer instead of generating a text response

---

### 3. `track_vis.py`

Generates interactive track layout and speed heatmap visualizations using real F1 telemetry data from FastF1.

**What it does:**
- `load_track_data()` — fetches the fastest qualifying lap telemetry and circuit corner data for a given GP from FastF1 (2024 season), with Streamlit caching to avoid repeated downloads
- `plot_track()` — generates a two-panel matplotlib figure:
  - **Left panel**: track outline with corner number annotations (T1, T2, ...) and start/finish marker
  - **Right panel**: speed heatmap along the track using a Red→Yellow→Green colormap, with minimum and maximum speed markers
- `render_sidebar_controls()` — renders the circuit dropdown and Load Track button in the sidebar; stores the selected GP in `st.session_state["viz_track"]`
- `render_track_output()` — renders the figure and speed statistics (top speed, min speed, corner count) in the main area when `viz_track` is set
- `detect_viz_intent()` — checks if a user message contains visualization keywords alongside a detected track name

---

## Setup

Please access the chatbot by using this link: https://f1-chatbot-etbhiwhqdedwg2388kgzd2.streamlit.app/

## Data Sources

All of the data comes from fastf1 API and Kaggle.
| File | Description |
|---|---|
| `strategies_kb.json` | Top tyre strategies per race (top 10 finishers, 2024 season) |
| `track_info.json` | Circuit cluster classification and average race speed |
| `cluster_safetycar_rate.csv` | Historical safety car appearance rate per circuit cluster |
| `F1Strategist_Ontology.owl` | OWL ontology defining relationships between tracks, strategies, tyres, drivers, and teams |