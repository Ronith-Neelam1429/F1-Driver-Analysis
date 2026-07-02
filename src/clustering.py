"""Driver driving-style clustering using scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features import STYLE_FEATURE_COLUMNS


@dataclass
class ClusteringResult:
    """Output of fitting a driver-style clustering model."""

    labels: pd.DataFrame
    feature_matrix: pd.DataFrame
    scaled_features: np.ndarray
    n_clusters: int
    silhouette: float
    pca_coords: pd.DataFrame
    cluster_profiles: pd.DataFrame
    kmeans: KMeans
    scaler: StandardScaler
    pca: PCA


def _prepare_matrix(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    drivers = feature_matrix["Driver"].tolist()
    cols = [c for c in STYLE_FEATURE_COLUMNS if c in feature_matrix.columns]
    X = feature_matrix[cols].copy()
    X = X.fillna(X.mean())
    return feature_matrix[["Driver"]].copy(), X.values, cols


def find_best_k(
    X_scaled: np.ndarray,
    k_min: int = 3,
    k_max: int = 8,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Pick cluster count with the highest silhouette score."""
    scores: dict[int, float] = {}
    n_samples = X_scaled.shape[0]
    upper = min(k_max, n_samples - 1)
    lower = min(k_min, upper)

    for k in range(lower, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(X_scaled, labels)

    if not scores:
        return 2, {2: 0.0}

    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_driver_clusters(
    feature_matrix: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> ClusteringResult:
    """Cluster drivers from a driver-level style feature matrix.

    Args:
        feature_matrix: Columns ``Driver`` plus style features from ``features.py``.
        n_clusters: Fixed cluster count; when ``None``, chosen via silhouette score.
    """
    drivers_df, X, feature_cols = _prepare_matrix(feature_matrix)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_clusters is None:
        n_clusters, _ = find_best_k(X_scaled, random_state=random_state)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0

    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"Driver": drivers_df["Driver"]})
    for i in range(n_components):
        pca_df[f"PC{i + 1}"] = coords[:, i]

    labels_df = pd.DataFrame(
        {
            "Driver": drivers_df["Driver"],
            "Cluster": cluster_labels,
        }
    )

    profiles = (
        feature_matrix.assign(Cluster=cluster_labels)
        .groupby("Cluster")[feature_cols]
        .mean()
        .reset_index()
    )

    out_features = feature_matrix.copy()
    out_features["Cluster"] = cluster_labels

    return ClusteringResult(
        labels=labels_df,
        feature_matrix=out_features,
        scaled_features=X_scaled,
        n_clusters=n_clusters,
        silhouette=sil,
        pca_coords=pca_df,
        cluster_profiles=profiles,
        kmeans=kmeans,
        scaler=scaler,
        pca=pca,
    )


def save_clustering_model(result: ClusteringResult, path: Path) -> Path:
    """Persist scaler, KMeans model, and metadata for later inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kmeans": result.kmeans,
        "scaler": result.scaler,
        "pca": result.pca,
        "feature_columns": [c for c in STYLE_FEATURE_COLUMNS if c in result.feature_matrix.columns],
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette,
    }
    joblib.dump(artifact, path)
    return path


def load_clustering_model(path: Path) -> dict:
    """Load a saved clustering artifact."""
    return joblib.load(path)


def predict_driver_clusters(feature_matrix: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Assign cluster labels to a driver feature matrix using a saved model."""
    cols = artifact["feature_columns"]
    X = feature_matrix[cols].copy().fillna(feature_matrix[cols].mean())
    X_scaled = artifact["scaler"].transform(X)
    labels = artifact["kmeans"].predict(X_scaled)
    return pd.DataFrame({"Driver": feature_matrix["Driver"], "Cluster": labels})
