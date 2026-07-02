"""Plotly figures for driver driving-style clusters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis import driver_color
from src.clustering import ClusteringResult

CLUSTER_PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
]

RADAR_FEATURES = [
    "Brake_Pct",
    "Mean_Entry_Speed",
    "Mean_Apex_Speed",
    "Mean_Exit_Speed",
    "Mean_Brake_Duration",
    "Mean_Peak_Decel",
    "Avg_Throttle",
    "Brake_Time_Pct",
]


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_PALETTE[int(cluster_id) % len(CLUSTER_PALETTE)]


def build_cluster_scatter(
    labels: pd.DataFrame,
    pca_coords: pd.DataFrame,
    *,
    title: str = "2025 Driver Style Clusters (PCA)",
) -> go.Figure:
    """Scatter plot of drivers in PCA space, coloured by cluster."""
    plot_df = pca_coords.merge(labels, on="Driver", how="inner")
    plot_df["ClusterLabel"] = plot_df["Cluster"].apply(lambda c: f"Cluster {c}")

    if "PC2" not in plot_df.columns:
        plot_df["PC2"] = 0.0

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="ClusterLabel",
        text="Driver",
        color_discrete_map={
            f"Cluster {c}": cluster_color(c) for c in sorted(plot_df["Cluster"].unique())
        },
        title=title,
        labels={"PC1": "Principal component 1", "PC2": "Principal component 2"},
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=1, color="white")),
    )
    fig.update_layout(
        height=620,
        legend=dict(title="Cluster", orientation="h", y=-0.12),
    )
    return fig


def build_driver_cluster_bars(labels: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart listing drivers grouped by cluster."""
    plot_df = labels.sort_values(["Cluster", "Driver"]).copy()
    plot_df["ClusterLabel"] = plot_df["Cluster"].apply(lambda c: f"Cluster {c}")
    plot_df["BarColor"] = plot_df["Cluster"].apply(cluster_color)

    fig = go.Figure(
        go.Bar(
            y=plot_df["Driver"],
            x=[1] * len(plot_df),
            orientation="h",
            marker_color=plot_df["BarColor"],
            text=plot_df["ClusterLabel"],
            textposition="outside",
            hovertext=plot_df.apply(
                lambda r: f"{r['Driver']} — Cluster {r['Cluster']}", axis=1
            ),
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Drivers by cluster",
        xaxis_visible=False,
        height=max(400, 28 * len(plot_df)),
        margin=dict(l=20, r=120),
        showlegend=False,
    )
    return fig


def _normalize_profiles(profiles: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Scale feature columns to 0–1 for comparable radar spokes."""
    work = profiles.copy()
    for col in features:
        if col not in work.columns:
            continue
        lo = work[col].min()
        hi = work[col].max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            work[col] = 0.5
        else:
            work[col] = (work[col] - lo) / (hi - lo)
    return work


def build_cluster_radar(profiles: pd.DataFrame) -> go.Figure:
    """Radar chart comparing cluster centroids on key style features."""
    features = [f for f in RADAR_FEATURES if f in profiles.columns]
    if not features:
        return go.Figure().update_layout(title="No cluster profile features available")

    norm = _normalize_profiles(profiles, features)
    fig = go.Figure()

    for _, row in norm.iterrows():
        cluster_id = int(row["Cluster"])
        values = [row[f] for f in features]
        values.append(values[0])
        theta = features + [features[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                name=f"Cluster {cluster_id}",
                line=dict(color=cluster_color(cluster_id)),
                fill="toself",
                opacity=0.55,
            )
        )

    fig.update_layout(
        title="Cluster style profiles (normalized)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=520,
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


def build_cluster_dashboard_figure(result: ClusteringResult) -> go.Figure:
    """Combined figure: PCA scatter + radar profiles."""
    scatter = build_cluster_scatter(result.labels, result.pca_coords)
    radar = build_cluster_radar(result.cluster_profiles)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=("Driver clusters (PCA)", "Cluster style profiles"),
        horizontal_spacing=0.08,
    )

    for trace in scatter.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in radar.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        height=620,
        title_text=f"2025 Driving Style Clusters — {result.n_clusters} groups "
        f"(silhouette {result.silhouette:.2f})",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    return fig


def save_cluster_plots(result: ClusteringResult, output_dir: Path) -> dict[str, Path]:
    """Write cluster visualizations to HTML files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    combined = build_cluster_dashboard_figure(result)
    combined_path = output_dir / "driver_clusters_plot.html"
    combined.write_html(combined_path, include_plotlyjs="cdn")
    paths["combined"] = combined_path

    scatter = build_cluster_scatter(result.labels, result.pca_coords)
    scatter_path = output_dir / "driver_clusters_scatter.html"
    scatter.write_html(scatter_path, include_plotlyjs="cdn")
    paths["scatter"] = scatter_path

    radar = build_cluster_radar(result.cluster_profiles)
    radar_path = output_dir / "cluster_profiles_radar.html"
    radar.write_html(radar_path, include_plotlyjs="cdn")
    paths["radar"] = radar_path

    return paths
