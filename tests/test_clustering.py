"""Tests for driver style clustering pipeline."""

import numpy as np
import pandas as pd

from src.clustering import fit_driver_clusters, load_clustering_model, predict_driver_clusters, save_clustering_model
from src.features import STYLE_FEATURE_COLUMNS, build_driver_feature_matrix, extract_round_driver_features


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
