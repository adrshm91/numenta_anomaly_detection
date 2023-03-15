import pandas as pd
import json
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


def main():
    # Load processed data
    df = pd.read_csv(
        "data/processed/temperature_data.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    # Split data into train and test
    train_data, test_data = train_test_split(
        df, test_size=0.6, random_state=42, shuffle=False
    )

    # Train an isolation forest model
    model = IsolationForest(n_estimators=100, contamination=0.01)
    clf = Pipeline([("scaler", StandardScaler()), ("model", model)])
    clf.fit(train_data["value"].values.reshape(-1, 1))

    test_data.loc[:, "prediction"] = clf.predict(
        test_data["value"].values.reshape(-1, 1)
    )

    # Plot results
    with open("data/labels/combined_windows.json") as window_file:
        windows = json.loads(window_file.read())

    windows = windows["realKnownCause/machine_temperature_system_failure.csv"]

    ax = df["value"].plot(figsize=(10, 5))
    for window in windows:
        ax.axvspan(xmin=window[0], xmax=window[1], alpha=0.2, color="red")
        df_slice = df[(df.index >= window[0]) & (df.index <= window[1])]
        idx = df_slice.shape[0] // 2
        ax.scatter(x=df_slice.index[idx], y=df_slice.iloc[idx]["value"])
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature sensor data with ground truth anomalies")
    plt.show()

    ax = train_data["value"].plot(figsize=(10, 5), label="training", color="blue")
    ax = test_data["value"].plot(figsize=(10, 5), label="testing", color="green")
    for window in windows:
        ax.axvspan(xmin=window[0], xmax=window[1], alpha=0.2, color="red")
        df_slice = df[(df.index >= window[0]) & (df.index <= window[1])]
    predictions = test_data[test_data["prediction"] == -1]
    ax.scatter(
        predictions.index,
        predictions["value"],
        color="red",
        label="predicted anomaly",
        marker="x",
    )
    plt.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature sensor data with predictions")
    plt.show()

    y_pred = clf.predict(test_data[["value"]])

    # Compute the F1 score
    f1 = f1_score(test_data["anomaly"], y_pred, pos_label=-1)

    print(f"F1 score: {f1}")

    # Compute the confusion matrix
    cm = confusion_matrix(test_data["anomaly"], y_pred, labels=[1, -1])

    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="g",
        xticklabels=["1", "-1"],
        yticklabels=["1", "-1"],
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Save the model
    filename = "models/isolation_forest.joblib"
    joblib.dump(clf, filename)


if __name__ == "__main__":
    sys.exit(main())
