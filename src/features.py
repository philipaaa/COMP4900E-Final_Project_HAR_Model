import numpy as np
import pandas as pd


def magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)


def extract_features(window: pd.DataFrame) -> list[float]:
    ax = window["ax"].to_numpy()
    ay = window["ay"].to_numpy()
    az = window["az"].to_numpy()

    mag = magnitude(ax, ay, az)

    features = [
        np.mean(ax), np.std(ax), np.min(ax), np.max(ax),
        np.mean(ay), np.std(ay), np.min(ay), np.max(ay),
        np.mean(az), np.std(az), np.min(az), np.max(az),
        np.mean(mag), np.std(mag), np.min(mag), np.max(mag),
        np.mean(np.abs(ax)),
        np.mean(np.abs(ay)),
        np.mean(np.abs(az)),
        np.mean(np.abs(np.diff(mag))) if len(mag) > 1 else 0.0,
    ]

    return [float(x) for x in features]


def windows_to_feature_matrix(windows):
    return np.array([extract_features(w) for w in windows], dtype=float)