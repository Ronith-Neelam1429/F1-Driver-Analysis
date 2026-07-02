"""Streamlit dashboard for the Australia 2025 F1 qualifying telemetry analysis.

Run from the project root:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable no matter where streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (  # noqa: E402
    best_laps,
    corner_summary,
    driver_color,
    driver_telemetry_summary,
    fastest_lap_telemetry,
    parse_laptime_seconds,
)
from src.cluster_plots import (  # noqa: E402
    build_cluster_radar,
    build_cluster_scatter,
    build_driver_cluster_bars,
)
from src.data_io import read_csv  # noqa: E402
from src.paths import PROJECT_ROOT, QUALIFYING_PROCESSED, RACE_RAW  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="F1 Australia 2025 Telemetry", page_icon="🏎️", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading telemetry…")
def load_quali() -> pd.DataFrame:
    return read_csv(QUALIFYING_PROCESSED)


@st.cache_data(show_spinner=False)
def load_race() -> pd.DataFrame | None:
    try:
        return read_csv(RACE_RAW)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cached_best_laps(df: pd.DataFrame) -> pd.DataFrame:
    return best_laps(df)


@st.cache_data(show_spinner=False)
def cached_summary(df: pd.DataFrame) -> pd.DataFrame:
    return driver_telemetry_summary(df)


@st.cache_data(show_spinner=False)
def cached_corners(df: pd.DataFrame) -> pd.DataFrame:
    return corner_summary(df)


@st.cache_data(show_spinner=False)
def cached_fastest_lap(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    return fastest_lap_telemetry(df, driver)


@st.cache_data(show_spinner="Loading cluster results…")
def load_cluster_results() -> dict:
    """Load precomputed clustering outputs from results/."""
    labels_path = RESULTS_DIR / "driver_clusters.csv"
    pca_path = RESULTS_DIR / "driver_clusters_pca.csv"
    profiles_path = RESULTS_DIR / "cluster_profiles.csv"
    summary_path = RESULTS_DIR / "clustering_summary.json"

    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "labels": pd.read_csv(labels_path) if labels_path.exists() else None,
        "pca": pd.read_csv(pca_path) if pca_path.exists() else None,
        "profiles": pd.read_csv(profiles_path) if profiles_path.exists() else None,
        "summary": summary,
    }


def fmt_laptime(seconds: float) -> str:
    """Render seconds as M:SS.mmm."""
    if pd.isna(seconds):
        return "—"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
try:
    quali = load_quali()
except FileNotFoundError:
    st.error(
        "Processed qualifying data not found.\n\n"
        "Generate it first with `python process_telemetry.py` "
        "(it reads the raw FastF1 telemetry and writes the processed `.csv.gz`)."
    )
    st.stop()

race = load_race()
drivers = sorted(quali["Driver"].dropna().unique().tolist())

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🏎️ Australia 2025")
st.sidebar.caption("Qualifying telemetry dashboard")
st.sidebar.metric("Telemetry rows", f"{len(quali):,}")
st.sidebar.metric("Drivers", quali["Driver"].nunique())
st.sidebar.metric("Corners detected", int(quali["Corner"].nunique()))
st.sidebar.info(
    "Data: FastF1 · Round 1, Albert Park.\n\n"
    "Note: corner detection currently resolves a subset of turns, and "
    "acceleration is clipped to ±15 m/s²."
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Australian Grand Prix 2025 — Qualifying Telemetry")

tab_overview, tab_driver, tab_track, tab_compare, tab_corners, tab_clusters = st.tabs(
    ["📊 Overview", "👤 Driver", "🗺️ Track Map", "⚔️ Compare", "🌀 Corners", "🎯 Clusters"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    bl = cached_best_laps(quali)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drivers", len(drivers))
    c2.metric("Total laps", int(quali["LapNumber"].nunique()) if "LapNumber" in quali else 0)
    if not bl.empty:
        c3.metric("Pole lap", fmt_laptime(bl["LapTime_seconds"].iloc[0]), bl["Driver"].iloc[0])
        c4.metric("Top-10 spread", f"+{bl['Gap'].iloc[min(9, len(bl) - 1)]:.3f}s")

    st.subheader("Qualifying classification (fastest lap per driver)")
    if bl.empty:
        st.warning("No valid lap times found in the dataset.")
    else:
        table = bl.copy()
        table.insert(0, "Pos", range(1, len(table) + 1))
        table["Lap time"] = table["LapTime_seconds"].apply(fmt_laptime)
        table["Gap"] = table["Gap"].apply(lambda g: "—" if g == 0 else f"+{g:.3f}")
        st.dataframe(
            table[["Pos", "Driver", "Lap time", "Gap", "Compound"]],
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            bl,
            x="Gap",
            y="Driver",
            orientation="h",
            color="Driver",
            color_discrete_map={d: driver_color(d) for d in bl["Driver"]},
            labels={"Gap": "Gap to pole (s)"},
            title="Gap to pole",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total descending"},
            showlegend=False,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Single driver
# --------------------------------------------------------------------------- #
with tab_driver:
    driver = st.selectbox("Driver", drivers, key="driver_select")
    summary = cached_summary(quali)
    row = summary[summary["Driver"] == driver]

    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max speed", f"{r['Max_Speed']:.0f} km/h")
        c2.metric("Avg throttle", f"{r['Avg_Throttle']:.0f}%")
        c3.metric("Time braking", f"{r['Brake_Pct']:.0f}%")
        c4.metric("Max RPM", f"{r['Max_RPM']:.0f}")

    lap = cached_fastest_lap(quali, driver)
    if lap.empty:
        st.warning("No valid fastest lap telemetry for this driver.")
    else:
        st.subheader(f"Fastest lap trace — {driver}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lap["TimeInLap"], y=lap["Speed"], name="Speed (km/h)",
                                 line=dict(color=driver_color(driver))))
        fig.add_trace(go.Scatter(x=lap["TimeInLap"], y=lap["Throttle"], name="Throttle (%)",
                                 yaxis="y2", line=dict(color="#2ca02c")))
        fig.update_layout(
            xaxis_title="Time into lap (s)",
            yaxis=dict(title="Speed (km/h)"),
            yaxis2=dict(title="Throttle (%)", overlaying="y", side="right", range=[0, 105]),
            height=420,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Channel distributions")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.histogram(lap, x="Speed", nbins=40, title="Speed distribution",
                             color_discrete_sequence=[driver_color(driver)]),
                use_container_width=True,
            )
        with col2:
            if "Gear" in lap.columns:
                st.plotly_chart(
                    px.histogram(lap, x="Gear", title="Gear usage",
                                 color_discrete_sequence=[driver_color(driver)]),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    px.histogram(lap, x="RPM", nbins=40, title="RPM distribution",
                                 color_discrete_sequence=[driver_color(driver)]),
                    use_container_width=True,
                )

# --------------------------------------------------------------------------- #
# Track map
# --------------------------------------------------------------------------- #
with tab_track:
    driver = st.selectbox("Driver", drivers, key="track_driver")
    metric = st.radio("Colour by", ["Speed", "Throttle", "RPM"], horizontal=True)
    lap = cached_fastest_lap(quali, driver)

    if lap.empty or lap[["X", "Y"]].isna().all().any():
        st.warning("No position data available for this driver's fastest lap.")
    else:
        fig = px.scatter(
            lap, x="X", y="Y", color=metric,
            color_continuous_scale="Turbo",
            title=f"{driver} — fastest lap racing line (coloured by {metric})",
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # keep the track shape correct
        fig.update_layout(height=650, xaxis_title="", yaxis_title="",
                          xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Position data from FastF1 (X/Y in track coordinate units).")

# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
with tab_compare:
    picks = st.multiselect(
        "Drivers to compare", drivers,
        default=drivers[:3] if len(drivers) >= 3 else drivers,
    )
    if not picks:
        st.info("Pick at least one driver.")
    else:
        summary = cached_summary(quali)
        sub = summary[summary["Driver"].isin(picks)]
        cmap = {d: driver_color(d) for d in picks}

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.bar(sub, x="Driver", y="Max_Speed", color="Driver",
                       color_discrete_map=cmap, title="Max speed (km/h)").update_layout(showlegend=False),
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(sub, x="Driver", y="Avg_Throttle", color="Driver",
                       color_discrete_map=cmap, title="Avg throttle (%)").update_layout(showlegend=False),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                px.bar(sub, x="Driver", y="Brake_Pct", color="Driver",
                       color_discrete_map=cmap, title="Time on brakes (%)").update_layout(showlegend=False),
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(sub, x="Driver", y="Avg_RPM", color="Driver",
                       color_discrete_map=cmap, title="Avg RPM").update_layout(showlegend=False),
                use_container_width=True,
            )

        st.subheader("Fastest-lap speed traces")
        fig = go.Figure()
        for d in picks:
            lap = cached_fastest_lap(quali, d)
            if not lap.empty:
                fig.add_trace(go.Scatter(x=lap["TimeInLap"], y=lap["Speed"],
                                         name=d, line=dict(color=driver_color(d))))
        fig.update_layout(xaxis_title="Time into lap (s)", yaxis_title="Speed (km/h)", height=450)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Corners
# --------------------------------------------------------------------------- #
with tab_corners:
    corners = cached_corners(quali)
    if corners.empty:
        st.warning("No corner-tagged telemetry available.")
    else:
        corner_names = sorted(corners["Corner"].unique().tolist())
        st.caption(
            f"Detected corners: {', '.join(corner_names)}. "
            "(Corner detection is approximate — see the sidebar note.)"
        )
        corner = st.selectbox("Corner", corner_names)
        sub = corners[corners["Corner"] == corner].copy()

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.bar(sub.sort_values("Min_Speed"), x="Min_Speed", y="Driver",
                       orientation="h", color="Driver",
                       color_discrete_map={d: driver_color(d) for d in sub["Driver"]},
                       title=f"{corner} — minimum (apex) speed (km/h)")
                .update_layout(showlegend=False, height=500),
                use_container_width=True,
            )
        with col2:
            decel = sub.dropna(subset=["Max_Deceleration"]).sort_values("Max_Deceleration")
            st.plotly_chart(
                px.bar(decel, x="Max_Deceleration", y="Driver",
                       orientation="h", color="Driver",
                       color_discrete_map={d: driver_color(d) for d in decel["Driver"]},
                       title=f"{corner} — peak deceleration (m/s², clipped at -15)")
                .update_layout(showlegend=False, height=500),
                use_container_width=True,
            )

        st.dataframe(sub.round(2), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# Clusters (2025 season driving style)
# --------------------------------------------------------------------------- #
with tab_clusters:
    st.subheader("2025 Driving Style Clusters")
    st.caption(
        "Clusters from qualifying telemetry across the full 2025 season. "
        "Features include braking zones, entry/exit speed, lap profiles, and sector ratios."
    )

    cluster_data = load_cluster_results()
    labels = cluster_data["labels"]
    pca = cluster_data["pca"]
    profiles = cluster_data["profiles"]
    summary = cluster_data["summary"]

    if labels is None or pca is None:
        st.warning(
            "Cluster results not found. Generate them from the project root:\n\n"
            "```\npython generate_2025_season.py\npython train_driver_clusters.py\n```"
        )
    else:
        if summary is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Drivers clustered", int(summary.get("n_drivers", len(labels))))
            c2.metric("Clusters", int(summary.get("n_clusters", labels["Cluster"].nunique())))
            c3.metric("Silhouette score", f"{summary.get('silhouette_score', 0):.3f}")

        st.plotly_chart(
            build_cluster_scatter(labels, pca),
            use_container_width=True,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            if profiles is not None:
                st.plotly_chart(build_cluster_radar(profiles), use_container_width=True)
            else:
                st.info("Cluster profiles file not found.")
        with col_right:
            st.plotly_chart(build_driver_cluster_bars(labels), use_container_width=True)

        st.subheader("Cluster assignments")
        table = labels.merge(pca, on="Driver", how="left").sort_values(["Cluster", "Driver"])
        table["Cluster"] = table["Cluster"].apply(lambda c: f"Cluster {c}")
        st.dataframe(table, hide_index=True, use_container_width=True)

        if profiles is not None:
            with st.expander("Cluster feature centroids"):
                st.dataframe(profiles.round(2), hide_index=True, use_container_width=True)

        plot_file = RESULTS_DIR / "driver_clusters_plot.html"
        if plot_file.exists():
            st.caption(f"Full combined plot also saved to `{plot_file.relative_to(PROJECT_ROOT)}`.")
