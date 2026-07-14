# Project Overview

This repository analyzes Formula 1 telemetry, builds driver-style features, clusters drivers, and serves the results in a Streamlit dashboard.

## Top-Level Layout

### `README.md`

Project entry documentation with setup, usage, and dashboard launch instructions.

### `requirements.txt`

Python dependencies for the analysis pipeline, notebooks, and dashboard.

### `PROJECT_OVERVIEW.md`

This file. Explains what each folder and major file does.

### `dashboard/`

Streamlit application for exploring telemetry and clustering results.

- `dashboard/app.py`: main dashboard entry point.

### `scripts/`

Runnable Python entry points organized by purpose.

#### `scripts/data/`

Data extraction, processing, compression, and recovery jobs.

- `scripts/data/generate_2025_season.py`: extracts FastF1 season data, processes qualifying telemetry, and writes season summary files.
- `scripts/data/generate_2025_openf1.py`: OpenF1 fallback pipeline for qualifying telemetry.
- `scripts/data/process_telemetry.py`: processes the Australia 2025 qualifying session only.
- `scripts/data/compress_processed_data.py`: compresses processed CSV files into `.csv.gz` companions.
- `scripts/data/decompress_processed_data.py`: decompresses processed `.csv.gz` files for local inspection.
- `scripts/data/extract_australia_telemetry.py`: legacy Australia race telemetry extractor.
- `scripts/data/__init__.py`: marks the folder as a Python package.

#### `scripts/modeling/`

Model training and clustering jobs.

- `scripts/modeling/train_driver_clusters.py`: builds the driver-style clustering model and writes result artifacts.
- `scripts/modeling/__init__.py`: marks the folder as a Python package.

#### `scripts/__init__.py`

Marks `scripts/` as a Python package so the data scripts can import each other cleanly.

### `src/`

Reusable analysis, feature engineering, clustering, and path helpers.

- `src/__init__.py`: package marker.
- `src/analysis.py`: reusable telemetry analytics helpers used by notebooks, scripts, and the dashboard.
- `src/cluster_plots.py`: Plotly visualizations for cluster output.
- `src/clustering.py`: clustering model fitting, prediction, and serialization helpers.
- `src/data_extraction_fastf1.py`: FastF1 season/session extraction helpers.
- `src/data_extraction_openf1.py`: OpenF1 extraction helpers.
- `src/data_io.py`: CSV read/write and sample-file helpers.
- `src/features.py`: driver feature engineering from qualifying telemetry.
- `src/openf1_telemetry.py`: OpenF1 telemetry normalization into the project’s data format.
- `src/paths.py`: project root and data-path helpers.
- `src/telemetry_processing.py`: telemetry post-processing helpers such as acceleration and corner detection.

### `data/`

Input and generated data.

- `data/raw_data/`: raw telemetry and season files written by extraction scripts.
- `data/processed/`: processed qualifying CSVs, `.csv.gz` companions, and `_sample.csv` previews.
- `data/processed/r*_2025_quali_telemetry_processed.csv`: full processed qualifying telemetry for each round.
- `data/processed/r*_2025_quali_telemetry_processed.csv.gz`: compressed copies kept for git-friendly storage.
- `data/processed/r*_2025_quali_telemetry_processed_sample.csv`: trimmed previews for quick inspection.

### `notebooks/`

Exploratory notebooks.

- `notebooks/Australia_2025_Telemetry_Analysis.ipynb`: exploratory analysis for the Australia 2025 session.

### `results/`

Cluster-training outputs and visualization artifacts.

- `results/cluster_profiles.csv`: per-cluster feature centroids.
- `results/clustering_summary.json`: clustering summary metrics.
- `results/driver_cluster_model.joblib`: saved clustering artifact.
- `results/driver_clusters.csv`: driver-to-cluster assignments.
- `results/driver_clusters_pca.csv`: PCA coordinates used in the cluster plot.
- `results/driver_feature_matrix.csv`: driver-level feature matrix used for clustering.
- `results/driver_features_per_round.csv`: round-by-round feature extraction output.
- `results/driver_clusters_plot.html`: combined dashboard-style cluster visualization.
- `results/driver_clusters_scatter.html`: PCA scatter plot HTML.
- `results/cluster_profiles_radar.html`: radar chart of cluster profiles.

### `tests/`

Automated checks for the clustering pipeline.

- `tests/test_clustering.py`: validates clustering helpers and feature extraction behavior.

## How The Pieces Fit Together

1. `scripts/data/generate_2025_season.py` or `scripts/data/generate_2025_openf1.py` produces raw telemetry and processed qualifying CSVs.
2. `src/features.py` converts telemetry into driver-level feature matrices.
3. `scripts/modeling/train_driver_clusters.py` trains the clustering model and writes files in `results/`.
4. `dashboard/app.py` loads `data/processed/` and `results/` to render the dashboard.

## Typical Commands

```bash
python scripts/data/generate_2025_season.py
python scripts/modeling/train_driver_clusters.py
streamlit run dashboard/app.py
```
