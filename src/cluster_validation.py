"""Cluster validation for driver-style K-Means models.

Combines internal quality metrics, external checks against team/teammate
proxies, feature ablation (pace vs style), and leave-one-round stability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.analysis import DRIVER_TO_TEAM
from src.clustering import ClusteringResult, fit_driver_clusters
from src.features import STYLE_FEATURE_COLUMNS, build_driver_feature_matrix

# Outcome / pace features that can dominate "style" clusters.
PACE_FEATURES = ["Sector1_Ratio", "Sector2_Ratio", "Sector3_Ratio"]
SPEED_PROFILE_FEATURES = [f"Speed_Pct_{p}" for p in range(10, 100, 10)]
PACE_AND_SPEED_FEATURES = PACE_FEATURES + SPEED_PROFILE_FEATURES


def _feature_cols(feature_matrix: pd.DataFrame) -> list[str]:
    return [c for c in STYLE_FEATURE_COLUMNS if c in feature_matrix.columns]


def _scaled_matrix(feature_matrix: pd.DataFrame, cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cols = cols or _feature_cols(feature_matrix)
    X = feature_matrix[cols].copy().fillna(feature_matrix[cols].mean())
    return StandardScaler().fit_transform(X.values), cols


def compute_internal_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette, Davies–Bouldin, and Calinski–Harabasz for a labeling."""
    unique = set(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {
            "silhouette": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }
    return {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }


def evaluate_k_range(
    feature_matrix: pd.DataFrame,
    *,
    k_min: int = 3,
    k_max: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score candidate k values with silhouette, Davies–Bouldin, and Calinski–Harabasz.

    Higher silhouette / Calinski–Harabasz is better; lower Davies–Bouldin is better.
    """
    X_scaled, _ = _scaled_matrix(feature_matrix)
    n_samples = X_scaled.shape[0]
    upper = min(k_max, n_samples - 1)
    lower = min(k_min, upper)

    rows: list[dict[str, float | int]] = []
    for k in range(lower, upper + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        metrics = compute_internal_metrics(X_scaled, labels)
        rows.append({"k": k, "inertia": float(model.inertia_), **metrics})

    return pd.DataFrame(rows)


def team_labels_for_drivers(drivers: pd.Series | list[str]) -> pd.Series:
    """Map driver codes to constructor names; unknown drivers become 'Unknown'."""
    return pd.Series([DRIVER_TO_TEAM.get(str(d).upper(), "Unknown") for d in drivers], index=range(len(list(drivers))))


def teammate_pairs(drivers: list[str]) -> list[tuple[str, str]]:
    """Unique teammate pairs present in ``drivers``."""
    by_team: dict[str, list[str]] = {}
    for d in drivers:
        team = DRIVER_TO_TEAM.get(str(d).upper())
        if team is None:
            continue
        by_team.setdefault(team, []).append(str(d).upper())

    pairs: list[tuple[str, str]] = []
    for team, members in sorted(by_team.items()):
        members = sorted(set(members))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


def external_team_validation(labels: pd.DataFrame) -> dict[str, Any]:
    """Compare clusters to constructor labels (ARI, NMI, purity, crosstab)."""
    work = labels.copy()
    work["Driver"] = work["Driver"].astype(str).str.upper()
    work["Team"] = work["Driver"].map(DRIVER_TO_TEAM).fillna("Unknown")
    known = work[work["Team"] != "Unknown"]
    if known.empty or known["Cluster"].nunique() < 2 or known["Team"].nunique() < 2:
        return {
            "adjusted_rand_index": 0.0,
            "normalized_mutual_info": 0.0,
            "cluster_purity": 0.0,
            "crosstab": pd.DataFrame(),
        }

    y_true = known["Team"].values
    y_pred = known["Cluster"].values
    crosstab = pd.crosstab(known["Team"], known["Cluster"])

    # Purity: for each cluster, fraction belonging to the modal team.
    purity_num = 0
    for _, group in known.groupby("Cluster"):
        purity_num += int(group["Team"].value_counts().iloc[0])
    purity = purity_num / len(known)

    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, y_pred)),
        "cluster_purity": float(purity),
        "crosstab": crosstab,
    }


def teammate_agreement(labels: pd.DataFrame) -> dict[str, Any]:
    """How often teammates land in the same cluster (car-effect check)."""
    work = labels.copy()
    work["Driver"] = work["Driver"].astype(str).str.upper()
    cluster_map = dict(zip(work["Driver"], work["Cluster"]))
    drivers = list(cluster_map.keys())
    pairs = teammate_pairs(drivers)

    rows: list[dict[str, Any]] = []
    same = 0
    for a, b in pairs:
        if a not in cluster_map or b not in cluster_map:
            continue
        ca, cb = int(cluster_map[a]), int(cluster_map[b])
        agree = ca == cb
        same += int(agree)
        team = DRIVER_TO_TEAM.get(a, "Unknown")
        rows.append(
            {
                "Team": team,
                "Driver_A": a,
                "Driver_B": b,
                "Cluster_A": ca,
                "Cluster_B": cb,
                "Same_Cluster": agree,
            }
        )

    detail = pd.DataFrame(rows)
    rate = float(same / len(rows)) if rows else 0.0
    return {
        "n_teammate_pairs": len(rows),
        "pairs_same_cluster": same,
        "teammate_agreement_rate": rate,
        "pairs": detail,
    }


def _align_label_vectors(
    base: pd.DataFrame,
    other: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    merged = base.merge(other, on="Driver", suffixes=("_base", "_other"))
    if len(merged) < 3:
        return None
    if merged["Cluster_base"].nunique() < 2 or merged["Cluster_other"].nunique() < 2:
        return None
    return merged["Cluster_base"].values, merged["Cluster_other"].values


def ablation_validation(
    feature_matrix: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Refit clusters after dropping pace / speed-profile features; compare via ARI.

    High ARI vs the full model means assignments are robust to removing pace cues
    (more style-driven). Low ARI means clusters were largely pace/car performance.
    """
    full = fit_driver_clusters(feature_matrix, n_clusters=n_clusters, random_state=random_state)

    variants = {
        "no_sector_ratios": [c for c in PACE_FEATURES if c in feature_matrix.columns],
        "no_pace_or_speed_profile": [
            c for c in PACE_AND_SPEED_FEATURES if c in feature_matrix.columns
        ],
    }

    out: dict[str, Any] = {
        "full_silhouette": float(full.silhouette),
        "variants": {},
    }

    for name, drop_cols in variants.items():
        if not drop_cols:
            continue
        ablated = feature_matrix.drop(columns=drop_cols)
        if len(_feature_cols(ablated)) < 2:
            continue
        result = fit_driver_clusters(ablated, n_clusters=n_clusters, random_state=random_state)
        aligned = _align_label_vectors(full.labels, result.labels)
        ari = float(adjusted_rand_score(*aligned)) if aligned else 0.0
        out["variants"][name] = {
            "dropped_features": drop_cols,
            "n_features": len(_feature_cols(ablated)),
            "silhouette": float(result.silhouette),
            "ari_vs_full": ari,
            "labels": result.labels,
        }

    return out


def leave_one_round_stability(
    per_round: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Leave-one-round-out: ARI between full-season labels and each held-out refit."""
    full_matrix = build_driver_feature_matrix(per_round)
    if len(full_matrix) < 3:
        return {"mean_ari": 0.0, "std_ari": 0.0, "folds": pd.DataFrame()}

    full = fit_driver_clusters(full_matrix, n_clusters=n_clusters, random_state=random_state)
    rounds = sorted(per_round["Round"].dropna().unique())
    fold_rows: list[dict[str, Any]] = []

    for round_num in rounds:
        subset = per_round[per_round["Round"] != round_num]
        if subset["Round"].nunique() < 1:
            continue
        matrix = build_driver_feature_matrix(subset)
        if len(matrix) < max(3, n_clusters + 1):
            continue
        result = fit_driver_clusters(matrix, n_clusters=n_clusters, random_state=random_state)
        aligned = _align_label_vectors(full.labels, result.labels)
        if aligned is None:
            continue
        ari = float(adjusted_rand_score(*aligned))
        fold_rows.append({"held_out_round": int(round_num), "n_drivers": len(matrix), "ari_vs_full": ari})

    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        return {"mean_ari": 0.0, "std_ari": 0.0, "folds": folds}

    return {
        "mean_ari": float(folds["ari_vs_full"].mean()),
        "std_ari": float(folds["ari_vs_full"].std(ddof=0)),
        "folds": folds,
    }


def run_cluster_validation(
    result: ClusteringResult,
    feature_matrix: pd.DataFrame,
    per_round: pd.DataFrame | None = None,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full validation suite for a fitted clustering result."""
    X_scaled = result.scaled_features
    labels_arr = result.labels["Cluster"].values

    internal = compute_internal_metrics(X_scaled, labels_arr)
    k_scores = evaluate_k_range(feature_matrix, random_state=random_state)
    team = external_team_validation(result.labels)
    teammates = teammate_agreement(result.labels)
    ablation = ablation_validation(
        feature_matrix,
        n_clusters=result.n_clusters,
        random_state=random_state,
    )

    stability: dict[str, Any] | None = None
    if per_round is not None and not per_round.empty and "Round" in per_round.columns:
        stability = leave_one_round_stability(
            per_round,
            n_clusters=result.n_clusters,
            random_state=random_state,
        )

    # Drop heavy label frames from ablation before JSON serialization.
    ablation_summary = {
        "full_silhouette": round(float(ablation["full_silhouette"]), 4),
        "variants": {
            name: {
                "dropped_features": meta["dropped_features"],
                "n_features": meta["n_features"],
                "silhouette": round(float(meta["silhouette"]), 4),
                "ari_vs_full": round(float(meta["ari_vs_full"]), 4),
            }
            for name, meta in ablation["variants"].items()
        },
    }

    return {
        "internal": internal,
        "k_selection": k_scores,
        "team": {
            "adjusted_rand_index": team["adjusted_rand_index"],
            "normalized_mutual_info": team["normalized_mutual_info"],
            "cluster_purity": team["cluster_purity"],
            "crosstab": team["crosstab"],
        },
        "teammates": teammates,
        "ablation": ablation_summary,
        "stability": {
            "mean_ari": stability["mean_ari"] if stability else None,
            "std_ari": stability["std_ari"] if stability else None,
            "folds": stability["folds"] if stability else pd.DataFrame(),
        },
    }


def validation_summary_dict(validation: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable summary for ``clustering_summary.json`` / dashboard."""
    team = validation["team"]
    teammates = validation["teammates"]
    ablation = validation["ablation"]
    stability = validation["stability"]
    k_sel = validation["k_selection"]

    best_sil_k = int(k_sel.loc[k_sel["silhouette"].idxmax(), "k"]) if not k_sel.empty else None
    best_ch_k = int(k_sel.loc[k_sel["calinski_harabasz"].idxmax(), "k"]) if not k_sel.empty else None
    best_db_k = int(k_sel.loc[k_sel["davies_bouldin"].idxmin(), "k"]) if not k_sel.empty else None

    return {
        "internal_metrics": {
            "silhouette": round(validation["internal"]["silhouette"], 4),
            "davies_bouldin": round(validation["internal"]["davies_bouldin"], 4),
            "calinski_harabasz": round(validation["internal"]["calinski_harabasz"], 4),
        },
        "k_selection": {
            "scores": k_sel.round(4).to_dict(orient="records") if not k_sel.empty else [],
            "best_k_silhouette": best_sil_k,
            "best_k_calinski_harabasz": best_ch_k,
            "best_k_davies_bouldin": best_db_k,
        },
        "team_validation": {
            "adjusted_rand_index": round(team["adjusted_rand_index"], 4),
            "normalized_mutual_info": round(team["normalized_mutual_info"], 4),
            "cluster_purity": round(team["cluster_purity"], 4),
        },
        "teammate_agreement": {
            "n_teammate_pairs": teammates["n_teammate_pairs"],
            "pairs_same_cluster": teammates["pairs_same_cluster"],
            "teammate_agreement_rate": round(teammates["teammate_agreement_rate"], 4),
        },
        "ablation": ablation,
        "stability": {
            "leave_one_round_mean_ari": (
                round(stability["mean_ari"], 4) if stability["mean_ari"] is not None else None
            ),
            "leave_one_round_std_ari": (
                round(stability["std_ari"], 4) if stability["std_ari"] is not None else None
            ),
        },
    }
