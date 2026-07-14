"""Tests for driver style clustering pipeline."""

import numpy as np
import pandas as pd

from src.clustering import fit_driver_clusters, load_clustering_model, predict_driver_clusters, save_clustering_model
from src.cluster_validation import (
    ablation_validation,
    compute_internal_metrics,
    evaluate_k_range,
    external_team_validation,
    leave_one_round_stability,
    run_cluster_validation,
    teammate_agreement,
    validation_summary_dict,
)
from src.features import build_driver_feature_matrix, extract_round_driver_features


def _synthetic_lap(
    driver: str,
    lap_number: int,
    round_num: int,
    brake_bias: float,
    speed_scale: float,
    n_points: int = 400,
) -> pd.DataFrame:
    """Build a minimal lap telemetry frame for testing."""
    t = np.linspace(0, 90, n_points)
    speed = speed_scale * (80 + 120 * np.sin(t / 14) ** 2)
    brake = (t % 20 < 2 + brake_bias * 3).astype(bool)

    rows = []
    for i in range(n_points):
        accel = float(np.gradient(speed / 3.6, t)[i]) if i > 0 else 0.0
        accel = float(np.clip(accel, -15, 15))
        rows.append(
            {
                "Year": 2025,
                "Round": round_num,
                "SessionType": "Q",
                "DriverNumber": hash(driver) % 100,
                "Driver": driver,
                "LapNumber": lap_number,
                "TimeInLap": t[i],
                "Throttle": 100 - brake[i] * 80,
                "Brake": brake[i],
                "DRS": 0,
                "RPM": 9000 + speed[i] * 10,
                "Speed": speed[i],
                "X": np.cos(t[i] / 10) * 1000 + ord(driver[0]),
                "Y": np.sin(t[i] / 10) * 1000,
                "Z": 0.0,
                "LapTime": "0 days 00:01:30.000000",
                "Sector1Time": "0 days 00:00:30.000000",
                "Sector2Time": "0 days 00:00:30.000000",
                "Sector3Time": "0 days 00:00:30.000000",
                "Compound": "SOFT",
                "FreshTyre": True,
                "Acceleration": accel,
                "Accel_Type": "Deceleration" if accel < -0.5 else ("Acceleration" if accel > 0.5 else "Neutral"),
            }
        )
    return pd.DataFrame(rows)


def _synthetic_round(drivers: list[tuple[str, float, float]], round_num: int) -> pd.DataFrame:
    laps = []
    for driver, brake_bias, speed_scale in drivers:
        laps.append(_synthetic_lap(driver, 1, round_num, brake_bias, speed_scale))
    return pd.concat(laps, ignore_index=True)


def _synthetic_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two rounds of synthetic telemetry → per-round features + season matrix."""
    roster = [
        ("VER", 0.2, 1.1),
        ("NOR", 0.2, 1.08),
        ("PIA", 0.25, 1.07),
        ("HAM", 0.6, 0.95),
        ("LEC", 0.55, 0.97),
        ("ALO", 0.7, 0.9),
        ("RUS", 0.22, 1.05),
        ("ANT", 0.28, 1.02),
    ]
    rounds = []
    for round_num in (1, 2, 3):
        roster_jitter = [
            (d, b + 0.01 * (round_num - 1), s - 0.005 * (round_num - 1)) for d, b, s in roster
        ]
        rounds.append(extract_round_driver_features(_synthetic_round(roster_jitter, round_num)))
    per_round = pd.concat(rounds, ignore_index=True)
    return per_round, build_driver_feature_matrix(per_round)


def test_clustering_pipeline(tmp_path):
    round1 = _synthetic_round(
        [
            ("VER", 0.2, 1.1),
            ("NOR", 0.2, 1.08),
            ("PIA", 0.25, 1.07),
            ("HAM", 0.6, 0.95),
            ("LEC", 0.55, 0.97),
            ("ALO", 0.7, 0.9),
        ],
        round_num=1,
    )
    round2 = _synthetic_round(
        [
            ("VER", 0.22, 1.09),
            ("NOR", 0.21, 1.07),
            ("PIA", 0.24, 1.06),
            ("HAM", 0.58, 0.94),
            ("LEC", 0.57, 0.96),
            ("ALO", 0.72, 0.89),
        ],
        round_num=2,
    )

    per_round = pd.concat(
        [extract_round_driver_features(round1), extract_round_driver_features(round2)],
        ignore_index=True,
    )
    matrix = build_driver_feature_matrix(per_round)
    assert len(matrix) == 6

    result = fit_driver_clusters(matrix, n_clusters=2)
    assert result.n_clusters == 2
    assert len(result.labels) == 6
    assert result.silhouette > 0

    model_path = save_clustering_model(result, tmp_path / "model.joblib")
    artifact = load_clustering_model(model_path)
    preds = predict_driver_clusters(matrix, artifact)
    assert list(preds["Cluster"]) == list(result.labels["Cluster"])


def test_internal_metrics_and_k_range():
    _, matrix = _synthetic_matrix()
    result = fit_driver_clusters(matrix, n_clusters=2)
    metrics = compute_internal_metrics(result.scaled_features, result.labels["Cluster"].values)
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics
    assert "calinski_harabasz" in metrics
    assert metrics["silhouette"] > 0
    assert metrics["davies_bouldin"] > 0
    assert metrics["calinski_harabasz"] > 0

    k_scores = evaluate_k_range(matrix, k_min=2, k_max=3)
    assert set(k_scores.columns) >= {"k", "silhouette", "davies_bouldin", "calinski_harabasz", "inertia"}
    assert len(k_scores) >= 1


def test_team_and_teammate_validation():
    labels = pd.DataFrame(
        {
            "Driver": ["NOR", "PIA", "VER", "TSU", "HAM", "LEC"],
            "Cluster": [0, 0, 1, 1, 0, 1],
        }
    )
    team = external_team_validation(labels)
    assert 0.0 <= team["normalized_mutual_info"] <= 1.0
    assert 0.0 <= team["cluster_purity"] <= 1.0
    assert not team["crosstab"].empty

    mates = teammate_agreement(labels)
    assert mates["n_teammate_pairs"] == 3  # McLaren, Red Bull, Ferrari
    assert mates["pairs_same_cluster"] == 2  # NOR-PIA and VER-TSU
    assert mates["teammate_agreement_rate"] == 2 / 3


def test_ablation_and_stability():
    per_round, matrix = _synthetic_matrix()
    ablation = ablation_validation(matrix, n_clusters=2)
    assert "variants" in ablation
    assert "no_sector_ratios" in ablation["variants"]
    assert "ari_vs_full" in ablation["variants"]["no_sector_ratios"]

    stability = leave_one_round_stability(per_round, n_clusters=2)
    assert "mean_ari" in stability
    assert not stability["folds"].empty


def test_run_cluster_validation_summary():
    per_round, matrix = _synthetic_matrix()
    result = fit_driver_clusters(matrix, n_clusters=2)
    validation = run_cluster_validation(result, matrix, per_round)
    summary = validation_summary_dict(validation)
    assert "internal_metrics" in summary
    assert "team_validation" in summary
    assert "teammate_agreement" in summary
    assert "ablation" in summary
    assert "stability" in summary
    assert summary["stability"]["leave_one_round_mean_ari"] is not None
