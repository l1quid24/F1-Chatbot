"""
build_knowledge_base.py
Run this once to convert your data files into a ChromaDB vector store.
Usage: python build_knowledge_base.py
"""

import csv
import json
import chromadb
from chromadb.utils import embedding_functions

# ── 1. Load raw data ───────────────────────────────────────────────────────

with open("strategies_kb.json") as f:
    strategies_kb = json.load(f)

with open("track_info.json") as f:
    track_info = json.load(f)

with open("cluster_safetycar_rate.csv") as f:
    reader = csv.DictReader(f)
    safetycar_rates = {row["cluster_name"]: float(row["safety_car_rate"]) for row in reader}

# ── 2. Build text chunks ───────────────────────────────────────────────────

CLUSTER_CONTEXT = {
    "High-speed": (
        "This is a high-speed circuit where cars spend most of the time at full throttle. "
        "Tyre degradation is driven by high downforce and sustained speed rather than heavy braking. "
        "A safety car on a high-speed track is a good chance to pit without losing much time, "
        "since overtaking is generally possible on these layouts."
    ),
    "Mixed": (
        "This is a mixed-character circuit with a combination of high-speed sections and technical corners. "
        "Tyre wear is moderate and spread across the whole lap. "
        "Under a safety car, pitting can be a strong option if you have not yet made your planned stop."
    ),
    "Low-speed / street-like": (
        "This is a low-speed or street circuit with tight corners and narrow roads. "
        "Overtaking is very difficult, so track position is extremely important. "
        "Under a safety car, pitting is tempting because pit-lane time loss is reduced — "
        "but you risk rejoining in heavy traffic, which is hard to pass on this type of track."
    ),
}

SAFETY_CAR_GENERAL = (
    "When a safety car is deployed in Formula 1, the pit-lane time loss is reduced because "
    "all cars slow down behind the safety car. This makes it a good opportunity to make a "
    "planned tyre change earlier than originally scheduled. However, the decision also depends "
    "on track position: if you pit, you may lose places to cars that stay out. "
    "Teams always weigh up fresh tyres versus where they rejoin in the queue."
)

chunks = []

# ── Chunk type 1: per-track strategy summary ──────────────────────────────
for gp_name, strategies in strategies_kb.items():
    info = track_info.get(gp_name, {})
    cluster = info.get("cluster", "Mixed")
    avg_speed = info.get("avg_speed", "N/A")

    # Format strategy list
    strat_lines = []
    total_drivers = sum(s[1] for s in strategies)
    for strat, count in strategies:
        strat_lines.append(f"  - {strat} ({count} out of {total_drivers} top finishers)")
    strat_text = "\n".join(strat_lines)

    # Determine number of stops for top strategy
    top_strat = strategies[0][0]
    n_stints = len(top_strat.split(" -> "))
    n_stops = n_stints - 1

    chunk = (
        f"Track: {gp_name}\n"
        f"Circuit type: {cluster}\n"
        f"Average race speed: {avg_speed} km/h\n"
        f"Most common tyre strategies used by top 10 finishers:\n{strat_text}\n"
        f"The most popular strategy was {top_strat}, used by {strategies[0][1]} drivers. "
        f"This involves {n_stops} pit stop(s) during the race.\n"
        f"{CLUSTER_CONTEXT.get(cluster, '')}"
    )
    chunks.append({
        "id": f"strategy_{gp_name.replace(' ', '_')}",
        "text": chunk,
        "metadata": {"type": "strategy", "track": gp_name, "cluster": cluster}
    })

# ── Chunk type 2: per-track safety car advice ─────────────────────────────
for gp_name, info in track_info.items():
    cluster = info.get("cluster", "Mixed")
    avg_speed = info.get("avg_speed", "N/A")

    # Get top strategy if available
    strat_note = ""
    if gp_name in strategies_kb:
        top_strat = strategies_kb[gp_name][0][0]
        count = strategies_kb[gp_name][0][1]
        total = sum(s[1] for s in strategies_kb[gp_name])
        strat_note = (
            f"The most common dry-weather strategy at {gp_name} is {top_strat} "
            f"({count}/{total} top finishers). "
            f"A safety car may give you a chance to make this stop earlier at lower cost."
        )


    sc_rate = safetycar_rates.get(cluster, 0)
    sc_pct  = int(sc_rate * 100)

    chunk = (
        f"Safety car strategy advice for {gp_name}:\n"
        f"Circuit type: {cluster} | Average speed: {avg_speed} km/h\n"
        f"Historical safety car appearance rate for {cluster} circuits: {sc_pct}%\n"
        f"{SAFETY_CAR_GENERAL}\n"
        f"{CLUSTER_CONTEXT.get(cluster, '')}\n"
        f"{strat_note}"
    )
    
    chunks.append({
        "id": f"safetycar_{gp_name.replace(' ', '_')}",
        "text": chunk,
        "metadata": {"type": "safety_car", "track": gp_name, "cluster": cluster}
    })

# ── Chunk type 3: cluster overview ────────────────────────────────────────
for cluster_name, context in CLUSTER_CONTEXT.items():
    tracks_in_cluster = [
        gp for gp, info in track_info.items()
        if info.get("cluster") == cluster_name
    ]
    chunk = (
        f"Circuit cluster: {cluster_name}\n"
        f"Tracks in this cluster: {', '.join(tracks_in_cluster)}\n"
        f"{context}"
    )
    chunks.append({
        "id": f"cluster_{cluster_name.replace('/', '_').replace(' ', '_')}",
        "text": chunk,
        "metadata": {"type": "cluster", "cluster": cluster_name}
    })

# ── Chunk type 4: general F1 tyre knowledge ───────────────────────────────
general_chunks = [
    {
        "id": "tyre_compounds",
        "text": (
            "In Formula 1, there are five tyre compounds:\n"
            "- SOFT (red): fastest but wears out quickest, used for short stints or qualifying.\n"
            "- MEDIUM (yellow): balanced performance and durability, the most versatile compound.\n"
            "- HARD (white): slowest but most durable, used for long stints.\n"
            "- INTERMEDIATE (green): used in light rain or damp conditions with no standing water.\n"
            "- WET (blue): used in heavy rain with standing water on track.\n"
            "In a dry race, teams must use at least two different dry compounds (Soft, Medium, or Hard)."
        ),
        "metadata": {"type": "general", "topic": "tyre_compounds"}
    },
    {
        "id": "safety_car_general",
        "text": (
            "General safety car strategy in Formula 1:\n"
            + SAFETY_CAR_GENERAL +
            "\nThe virtual safety car (VSC) also slows down cars but gives less pit-lane time benefit "
            "compared to a full safety car. Teams typically prefer to pit under a full safety car."
        ),
        "metadata": {"type": "general", "topic": "safety_car"}
    },
    {
        "id": "pit_stop_basics",
        "text": (
            "A pit stop in Formula 1 typically takes 2 to 3 seconds of stationary time. "
            "The total time lost during a pit stop (including the pit lane entry and exit) "
            "is usually around 18 to 25 seconds depending on the circuit. "
            "A 1-stop strategy means one pit stop during the race, using two different tyre compounds. "
            "A 2-stop strategy means two pit stops, using three stints of tyres. "
            "More pit stops give fresher tyres but cost more total time in the pit lane."
        ),
        "metadata": {"type": "general", "topic": "pit_stop"}
    },
]
chunks.extend(general_chunks)

# ── 3. Store in ChromaDB ───────────────────────────────────────────────────

print(f"Building knowledge base with {len(chunks)} chunks...")

# Use default embedding function (no API key needed)
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if rebuilding
try:
    client.delete_collection("f1_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="f1_knowledge",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

collection.add(
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],
    metadatas=[c["metadata"] for c in chunks]
)

print(f"✅ Done! {len(chunks)} chunks stored in ./chroma_db")
print("\nChunk types created:")
print(f"  - Per-track strategy summaries: {len(strategies_kb)}")
print(f"  - Per-track safety car advice:  {len(track_info)}")
print(f"  - Cluster overviews:            {len(CLUSTER_CONTEXT)}")
print(f"  - General F1 knowledge:         {len(general_chunks)}")