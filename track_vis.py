import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
import streamlit as st
import os

# Cache FastF1 data to avoid re-downloading
fastf1.Cache.enable_cache("./fastf1_cache")

# We use 2024 season data where available

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


@st.cache_data(show_spinner=False)
def load_track_data(gp_name: str):
    """
    Load fastest lap telemetry and circuit info for a given GP.
    Returns (telemetry_df, circuit_info) or (None, None) on error.
    """
    if gp_name not in TRACK_SESSION_MAP:
        return None, None

    year, event_name = TRACK_SESSION_MAP[gp_name]
    try:
        session = fastf1.get_session(year, event_name, "Q")  # use qualifying for clean lap
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry().add_distance()
        circuit_info = session.get_circuit_info()
        return tel, circuit_info
    except Exception as e:
        st.warning(f"Could not load data for {gp_name}: {e}")
        return None, None


def make_segments(x, y):
    """Convert x/y arrays into line segments for LineCollection."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def plot_track(gp_name: str) -> plt.Figure:
    """
    Generate a two-panel figure:
      Left:  track layout with corner annotations
      Right: speed heatmap along the track
    """
    tel, circuit_info = load_track_data(gp_name)
    if tel is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Data unavailable for\n{gp_name}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig

    x   = tel["X"].values
    y   = tel["Y"].values
    spd = tel["Speed"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left panel: track layout + corners ────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")

    # Draw track outline (thick background line)
    ax1.plot(x, y, color="#444466", linewidth=10, zorder=1, solid_capstyle="round")
    # Draw track centerline
    ax1.plot(x, y, color="#e8e8f0", linewidth=3, zorder=2, solid_capstyle="round")

    # Mark start/finish
    ax1.plot(x[0], y[0], "o", color="#00ff88", markersize=10, zorder=5)
    ax1.annotate("S/F", (x[0], y[0]), textcoords="offset points",
                 xytext=(8, 8), color="#00ff88", fontsize=8, fontweight="bold")

    # Annotate corners from circuit_info
    if circuit_info is not None:
        corners = circuit_info.corners
        for _, corner in corners.iterrows():
            # corner columns: Number, Letter, X, Y, Angle, Distance
            cx, cy = corner["X"], corner["Y"]
            label  = f"T{int(corner['Number'])}"
            if corner.get("Letter", ""):
                label += str(corner["Letter"])

            ax1.plot(cx, cy, "o", color="#ff6b6b", markersize=5, zorder=4)
            ax1.annotate(
                label, (cx, cy),
                textcoords="offset points",
                xytext=(5, 5),
                color="#ffcc44",
                fontsize=7,
                fontweight="bold",
                zorder=6,
            )

    ax1.set_title(f"{gp_name}\nTrack Layout & Corners",
                  color="white", fontsize=11, pad=10)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ── Right panel: speed heatmap ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")

    segments = make_segments(x, y)
    norm     = plt.Normalize(spd.min(), spd.max())
    cmap     = plt.cm.RdYlGn        # red = slow, green = fast
    lc       = LineCollection(segments, cmap=cmap, norm=norm,
                               linewidth=4, zorder=2)
    lc.set_array(spd)
    ax2.add_collection(lc)
    ax2.set_xlim(x.min() - 50, x.max() + 50)
    ax2.set_ylim(y.min() - 50, y.max() + 50)

    # Colorbar
    cbar = fig.colorbar(lc, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_label("Speed (km/h)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Mark slowest and fastest points
    slow_idx = np.argmin(spd)
    fast_idx = np.argmax(spd)
    ax2.plot(x[slow_idx], y[slow_idx], "v", color="#ff4444",
             markersize=10, zorder=5, label=f"Min {spd[slow_idx]:.0f} km/h")
    ax2.plot(x[fast_idx], y[fast_idx], "^", color="#44ff88",
             markersize=10, zorder=5, label=f"Max {spd[fast_idx]:.0f} km/h")
    ax2.legend(loc="lower right", facecolor="#2a2a3e",
               labelcolor="white", fontsize=8)

    ax2.set_title(f"{gp_name}\nSpeed Heatmap (Fastest Qualifying Lap)",
                  color="white", fontsize=11, pad=10)
    ax2.set_aspect("equal")
    ax2.axis("off")

    plt.tight_layout()
    return fig


def render_track_explorer():
    """
    Streamlit UI block for track visualization.
    Call this from app.py sidebar or main area.
    """
    st.subheader("🗺️ Track Explorer")

    selected = st.selectbox(
        "Select a circuit:",
        options=list(TRACK_SESSION_MAP.keys()),
        index=list(TRACK_SESSION_MAP.keys()).index("Monaco Grand Prix"),
    )

    if st.button("Load Track", use_container_width=True):
        with st.spinner(f"Loading {selected} data from FastF1..."):
            fig = plot_track(selected)
            st.pyplot(fig)
            plt.close(fig)

        # Show quick stats
        tel, circuit_info = load_track_data(selected)
        if tel is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Speed", f"{tel['Speed'].max():.0f} km/h")
            col2.metric("Min Speed", f"{tel['Speed'].min():.0f} km/h")
            if circuit_info is not None:
                col3.metric("Corners", len(circuit_info.corners))