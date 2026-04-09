from pathlib import Path
import pandas as pd
import numpy as np


WINDOW_SIZE = 30   # 1 second at 30 Hz
STEP_SIZE = 15     # 50% overlap


def infer_label(filename: str) -> str:
    name = filename.lower()

    if "burpee" in name:
        return "burpees"
    if "jog" in name:
        return "jogging"
    if "jump" in name:
        return "jumping_jacks"
    if "pushup" in name:
        return "pushups"
    if "situp" in name:
        return "situps"
    if "squat" in name:
        return "squats"
    if "stand" in name:
        return "standing"
    if "walk" in name:
        return "walking"

    raise ValueError(f"Unknown label in filename: {filename}")


def infer_person(filename: str) -> str:
    # assuming filenames end with C.txt / P.txt / Z.txt
    stem = Path(filename).stem
    return stem[-1].upper()


def load_sensor_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={
        "accelerometerTimestamp_sinceReboot(s)": "t",
        "accelerometerAccelerationX(G)": "ax",
        "accelerometerAccelerationY(G)": "ay",
        "accelerometerAccelerationZ(G)": "az",
    })

    needed = ["t", "ax", "ay", "az"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"{path.name} missing required column: {col}")

    df = df[needed].dropna().reset_index(drop=True)
    return df


def simple_clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Very simple cleaning:
    - for standing: keep the middle 60%
    - for other activities: keep the middle 80%
    """
    n = len(df)
    if n < WINDOW_SIZE:
        return df

    if label == "standing":
        start = int(n * 0.2)
        end = int(n * 0.8)
    else:
        start = int(n * 0.1)
        end = int(n * 0.9)

    return df.iloc[start:end].reset_index(drop=True)


def make_windows(df: pd.DataFrame):
    windows = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        windows.append(df.iloc[start:end].reset_index(drop=True))
    return windows


def build_dataset(data_dir="data/raw"):
    X_windows = []
    y = []
    groups = []   # file/person grouping for better splitting
    file_ids = []

    for path in sorted(Path(data_dir).glob("*.txt")):
        if path.stat().st_size == 0:
            print(f"Skipping empty file: {path.name}")
            continue

        label = infer_label(path.name)
        person = infer_person(path.name)

        try:
            df = load_sensor_file(path)
            df = simple_clean(df, label)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

        windows = make_windows(df)
        if not windows:
            print(f"Skipping {path.name}: no valid windows")
            continue

        for w in windows:
            X_windows.append(w)
            y.append(label)
            groups.append(person)
            file_ids.append(path.name)

        print(f"{path.name:25s} -> {label:15s} | person={person} | windows={len(windows)}")

    return X_windows, np.array(y), np.array(groups), np.array(file_ids)