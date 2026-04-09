"""
F1 Track Visualizer
Renders track layout with corner annotations and speed heatmap.
"""

import os
import fastf1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import streamlit as st

os.makedirs("./fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("./fastf1_cache")

# ── Track name → (year, FastF1 event name) ────────────────────────────────

TRACK_SESSION_MAP = {
    "Monaco Grand Prix":          (2024, "Monaco"),
    "Italian Grand Prix":         (2024, "Italian"),
    "Singapore Grand Prix":       (2024, "Singapore"),
    "British Grand Prix":         (2024, "British"),
    "Belgian Grand Prix":         (2024, "Belgian"),
    "Japanese Grand Prix":        (2024, "Japanese"),
    "Bahrain Grand Prix":         (2024, "Bahrain"),
    "Saudi Arabian Grand Prix":   (2024, "Saudi Arabian"),
    "Australian Grand Prix":      (2024, "Australian"),
    "Azerbaijan Grand Prix":      (2024, "Azerbaijan"),
    "Miami Grand Prix":           (2024, "Miami"),
    "Spanish Grand Prix":         (2024, "Spanish"),
    "Canadian Grand Prix":        (2024, "Canadian"),
    "Austrian Grand Prix":        (2024, "Austrian"),
    "Hungarian Grand Prix":       (2024, "Hungarian"),
    "Dutch Grand Prix":           (2024, "Dutch"),
    "United States Grand Prix":   (2024, "United States"),
    "Mexico City Grand Prix":     (2024, "Mexico City"),
    "São Paulo Grand Prix":       (2024, "São Paulo"),
    "Las Vegas Grand Prix":       (2024, "Las Vegas"),
    "Qatar Grand Prix":           (2024, "Qatar"),
    "Abu Dhabi Grand Prix":       (2024, "Abu Dhabi"),
}

# Keywords that suggest the user wants a track visualization
VIS_KEYWORDS = [
    "load", "show", "draw", "display", "visuali",
    "circuit", "track map", "layout", "heatmap", "heat map"
]


def detect_vis_intent(text: str, detected_tracks: list[str]) -> str | None:
    """
    Return a GP name if the user message suggests they want a track visualization,
    otherwise return None.
    """
    text_lower = text.lower()
    has_vis_keyword = any(kw in text_lower for kw in VIS_KEYWORDS)
    if has_vis_keyword and detected_tracks:
        return detected_tracks[0]
    return None


@st.cache_data(show_spinner=False)
def load_track_data(gp_name: str):
    """Load fastest qualifying lap telemetry and circuit info."""
    if gp_name not in TRACK_SESSION_MAP:
        return None, None
    year, event_name = TRACK_SESSION_MAP[gp_name]
    try:
        session = fastf1.get_session(year, event_name, "Q")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry().add_distance()
        circuit_info = session.get_circuit_info()
        return tel, circuit_info
    except Exception as e:
        st.warning(f"Could not load data for {gp_name}: {e}")
        return None, None


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def plot_track(gp_name: str) -> plt.Figure:
    """Generate side-by-side track layout + speed heatmap figure."""
    tel, circuit_info = load_track_data(gp_name)

    if tel is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.text(0.5, 0.5, f"Data unavailable for\n{gp_name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="white")
        ax.axis("off")
        return fig

    x   = tel["X"].values
    y   = tel["Y"].values
    spd = tel["Speed"].values

    # Auto-scale figure height based on track aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    aspect  = y_range / max(x_range, 1)
    fig_h   = max(4.5, min(7.0, 5.5 * aspect))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, fig_h),
        gridspec_kw={"wspace": 0.02}   # ← eliminates the gap between panels
    )
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: track layout + corners ──────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    ax1.plot(x, y, color="#3a3a5c", linewidth=10, zorder=1,
             solid_capstyle="round")
    ax1.plot(x, y, color="#e8e8f0", linewidth=3,  zorder=2,
             solid_capstyle="round")

    # Start/finish marker
    ax1.plot(x[0], y[0], "o", color="#00ff88", markersize=9, zorder=5)
    ax1.annotate("S/F", (x[0], y[0]),
                 textcoords="offset points", xytext=(7, 7),
                 color="#00ff88", fontsize=8, fontweight="bold")

    # Corner annotations
    if circuit_info is not None:
        for _, corner in circuit_info.corners.iterrows():
            cx, cy = corner["X"], corner["Y"]
            label  = f"T{int(corner['Number'])}"
            if str(corner.get("Letter", "")).strip():
                label += str(corner["Letter"])
            ax1.plot(cx, cy, "o", color="#ff6b6b", markersize=4, zorder=4)
            ax1.annotate(label, (cx, cy),
                         textcoords="offset points", xytext=(4, 4),
                         color="#ffcc44", fontsize=7, fontweight="bold", zorder=6)

    ax1.set_title(f"{gp_name}\nTrack Layout & Corners",
                  color="white", fontsize=10, pad=8)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ── Right: speed heatmap ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")

    segments = make_segments(x, y)
    norm     = plt.Normalize(spd.min(), spd.max())
    lc       = LineCollection(segments, cmap=plt.cm.RdYlGn,
                               norm=norm, linewidth=4, zorder=2)
    lc.set_array(spd)
    ax2.add_collection(lc)
    ax2.set_xlim(x.min() - 50, x.max() + 50)
    ax2.set_ylim(y.min() - 50, y.max() + 50)

    cbar = fig.colorbar(lc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("Speed (km/h)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    slow_idx = int(np.argmin(spd))
    fast_idx = int(np.argmax(spd))
    ax2.plot(x[slow_idx], y[slow_idx], "v", color="#ff4444",
             markersize=9, zorder=5, label=f"Min {spd[slow_idx]:.0f} km/h")
    ax2.plot(x[fast_idx], y[fast_idx], "^", color="#44ff88",
             markersize=9, zorder=5, label=f"Max {spd[fast_idx]:.0f} km/h")
    ax2.legend(loc="lower right", facecolor="#2a2a3e",
               labelcolor="white", fontsize=8)

    ax2.set_title(f"{gp_name}\nSpeed Heatmap (Fastest Qualifying Lap)",
                  color="white", fontsize=10, pad=8)
    ax2.set_aspect("equal")
    ax2.axis("off")

    return fig


# ── Streamlit UI helpers ───────────────────────────────────────────────────

def render_sidebar_controls():
    """
    Sidebar widget: dropdown + button.
    Stores the selected GP in st.session_state['vis_track']
    when the user clicks Load Track.
    """
    st.subheader("Track Explorer")
    selected = st.selectbox(
        "Select a circuit:",
        options=list(TRACK_SESSION_MAP.keys()),
        index=list(TRACK_SESSION_MAP.keys()).index("Monaco Grand Prix"),
        key="sidebar_track_select",
    )
    if st.button("Load Track", use_container_width=True, key="sidebar_load_btn"):
        st.session_state["vis_track"] = selected
        st.session_state["vis_source"] = "sidebar"


def render_track_output():
    """
    Main-area widget: renders the figure and stats if vis_track is set.
    Call this once in the main body of app.py.
    """
    gp_name = st.session_state.get("vis_track")
    if not gp_name:
        return

    st.markdown(f"### {gp_name}")
    with st.spinner(f"Loading {gp_name} data…"):
        fig = plot_track(gp_name)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    tel, circuit_info = load_track_data(gp_name)
    if tel is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Top Speed",  f"{tel['Speed'].max():.0f} km/h")
        c2.metric("Min Speed",  f"{tel['Speed'].min():.0f} km/h")
        if circuit_info is not None:
            c3.metric("Corners", len(circuit_info.corners))

    st.divider()