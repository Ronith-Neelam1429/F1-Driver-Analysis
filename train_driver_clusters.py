#!/usr/bin/env python3
"""Train driver driving-style clusters from 2025 qualifying telemetry."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import fit_driver_clusters, save_clustering_model
from src.cluster_plots import save_cluster_plots
from src.data_io import read_csv, resolve_data_path
from src.features import extract_round_driver_features, build_driver_feature_matrix
from src.paths import iter_processed_qualifying_2025, PROJECT_ROOT as ROOT, _resolved_csv


def load_2025_qualifying_rounds() -> list[tuple[int, str, pd.DataFrame]]:
    rounds: list[tuple[int, str, pd.DataFrame]] = []
    using_samples = False

    for round_num, country_slug, path in iter_processed_qualifying_2025():
        resolved = resolve_data_path(path)
        if not resolved.exists():
            continue
        if path.name.endswith("_sample.csv") and _resolved_csv(path.with_name(path.name.replace("_sample", ""))) is not None:
            continue
        if path.name.endswith("_sample.csv"):
            using_samples = True
        df = read_csv(path)
        rounds.append((round_num, country_slug, df))

    if using_samples:
        print(
            "WARNING: Using _sample.csv previews. Run generate_2025_season.py for full telemetry "
            "before trusting cluster assignments."
        )

    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster 2025 F1 drivers by driving style")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed number of clusters (default: auto via silhouette)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for cluster outputs",
    )
    args = parser.parse_args()

    round_entries = load_2025_qualifying_rounds()
    if not round_entries:
        print("No 2025 processed qualifying data found in data/processed/")
        sys.exit(1)

    print(f"Loading {len(round_entries)} qualifying sessions...")

    per_round_rows: list[pd.DataFrame] = []
    for round_num, country_slug, df in round_entries:
        feats = extract_round_driver_features(df)
        if feats.empty:
            print(f"  Round {round_num:02d} ({country_slug}): no driver features extracted")
            continue
        print(f"  Round {round_num:02d} ({country_slug}): {len(feats)} drivers")
        per_round_rows.append(feats)

    if not per_round_rows:
        print("No features extracted from any round.")
        sys.exit(1)

    per_round = pd.concat(per_round_rows, ignore_index=True)
    feature_matrix = build_driver_feature_matrix(per_round)

    if len(feature_matrix) < 3:
        print(f"Only {len(feature_matrix)} drivers with features — need at least 3 for clustering.")
        sys.exit(1)

    print(f"\nFeature matrix: {len(feature_matrix)} drivers × {len(feature_matrix.columns) - 1} features")

    result = fit_driver_clusters(feature_matrix, n_clusters=args.n_clusters)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_round.to_csv(args.output_dir / "driver_features_per_round.csv", index=False)
    feature_matrix.to_csv(args.output_dir / "driver_feature_matrix.csv", index=False)
    result.labels.to_csv(args.output_dir / "driver_clusters.csv", index=False)
    result.pca_coords.to_csv(args.output_dir / "driver_clusters_pca.csv", index=False)
    result.cluster_profiles.to_csv(args.output_dir / "cluster_profiles.csv", index=False)

    summary = {
        "n_rounds": len(per_round_rows),
        "n_drivers": len(feature_matrix),
        "n_clusters": result.n_clusters,
        "silhouette_score": round(result.silhouette, 4),
    }
    with open(args.output_dir / "clustering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_path = save_clustering_model(result, args.output_dir / "driver_cluster_model.joblib")
    print(f"  Model saved: {model_path.name}")

    plot_paths = save_cluster_plots(result, args.output_dir)
    for name, path in plot_paths.items():
        print(f"  Plot saved: {path.name} ({name})")

    print("\nClustering complete")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Silhouette: {result.silhouette:.3f}")
    print("\nDriver assignments:")
    for _, row in result.labels.sort_values(["Cluster", "Driver"]).iterrows():
        print(f"  Cluster {row['Cluster']}: {row['Driver']}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
