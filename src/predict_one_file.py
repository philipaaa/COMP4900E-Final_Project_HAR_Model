from pathlib import Path
from collections import Counter
import joblib

from data_utils import load_sensor_file, simple_clean, make_windows, infer_label
from features import windows_to_feature_matrix


def main():
    model = joblib.load("simple_har_model.pkl")

    path = Path("data/raw/sampleWalkingP.txt")
    label = infer_label(path.name)

    df = load_sensor_file(path)
    df = simple_clean(df, label)
    windows = make_windows(df)

    X = windows_to_feature_matrix(windows)
    preds = model.predict(X)

    print("Window predictions:")
    print(preds)

    majority = Counter(preds).most_common(1)[0][0]
    print("\nMajority vote prediction:", majority)


if __name__ == "__main__":
    main()