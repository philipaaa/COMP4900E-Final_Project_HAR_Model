import joblib
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

from data_utils import build_dataset
from features import windows_to_feature_matrix

def export_model(model, path="model.json"):
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    export = {
        "classes": clf.classes_.tolist(),
        "weights": clf.coef_.tolist(),
        "bias": clf.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

    print("Exported model to", path)


def split_by_person(X, y, groups, test_person="Z"):
    train_idx = groups != test_person
    test_idx = groups == test_person

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    # load windows + labels
    windows, y, groups, file_ids = build_dataset("data/raw")

    #convert windows to feature vectors (this is feature extraction)
    X = windows_to_feature_matrix(windows)

    print("\nTotal windows:", len(X))
    print("Classes:", sorted(set(y)))
    print("People:", sorted(set(groups)))

    # train/test split by person
    #  in this case, we train on C and P, test on Z
    X_train, X_test, y_train, y_test = split_by_person(X, y, groups, test_person="Z")

    if len(X_test) == 0:
        raise ValueError("No test samples found for person Z. Change test_person or use another split.")

    print("\nTrain windows:", len(X_train))
    print("Test windows:", len(X_test))

    #build simple model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    #train
    model.fit(X_train, y_train)


    #validate/test and measure inference time
    start = time.time()
    preds = model.predict(X_test)
    end = time.time()

    avg_time = (end - start) / len(X_test)
    print("\nAvg inference time per window:", avg_time, "seconds")

    print("\nAccuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    #save model
    joblib.dump(model, "simple_har_model.pkl")
    print("\nSaved model to simple_har_model.pkl")

    export_model(model)

    ##to plot figure 1, confusion matrix


    disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)

    # Rotate x-axis labels (bottom labels)
    plt.xticks(rotation=90)

    plt.title("Confusion Matrix")
    plt.tight_layout()  # prevents labels from being cut off
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()