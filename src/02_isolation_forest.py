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
from clearml import Task, TaskTypes, Logger, Dataset

task = Task.init(
    project_name="numenta_anomaly_detection",
    task_name="training",
    reuse_last_task_id=False,
    task_type=TaskTypes.training,
)


def main():

    dataset = Dataset.get(
        dataset_name="numenta", dataset_project="numenta_anomaly_detection"
    )
    local_folder = dataset.get_local_copy()

    # Load processed data
    df = pd.read_csv(
        f"{local_folder}/processed/temperature_data_processed.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    # Split data into train and test
    train_data, test_data = train_test_split(
        df, test_size=0.6, random_state=42, shuffle=False
    )

    # Train an isolation forest model
    hyperparameters = {"n_estimators": 100, "contamination": 0.01}
    task.connect(hyperparameters)

    model = IsolationForest(
        n_estimators=hyperparameters["n_estimators"],
        contamination=hyperparameters["contamination"],
    )
    clf = Pipeline([("scaler", StandardScaler()), ("model", model)])
    clf.fit(train_data["value"].values.reshape(-1, 1))

    # Predict using test data
    test_data.loc[:, "prediction"] = clf.predict(
        test_data["value"].values.reshape(-1, 1)
    )

    # Plot results
    with open("data/labels/combined_windows.json") as window_file:
        windows = json.loads(window_file.read())

    windows = windows["realKnownCause/machine_temperature_system_failure.csv"]

    ax = train_data["value"].plot(figsize=(10, 5), label="training", color="blue")
    ax = test_data["value"].plot(figsize=(10, 5), label="testing", color="green")
    for window in windows:
        ax.axvspan(xmin=window[0], xmax=window[1], alpha=0.2, color="red")
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
    plt.savefig("results/temperature_sensor_data_predictions.png")

    # Compute the F1 score
    y_pred = clf.predict(test_data[["value"]])
    f1 = f1_score(test_data["anomaly"], y_pred, pos_label=-1)

    print(f"F1 score: {f1}")

    Logger.current_logger().report_scalar(
        "Model Performance", "F1 Score", iteration=0, value=f1
    )

    # Save test predictions
    test_data["predictions"] = y_pred
    test_data.to_csv("results/test_predictions.csv")
    task.upload_artifact(
        artifact_object="results/test_predictions.csv", name="predictions"
    )
    Logger.current_logger().report_table(
        "Test Predictions", "PD with index", iteration=0, table_plot=test_data
    )

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
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")

    Logger.current_logger().report_matrix(
        "Confusion Matrix",
        "ignored",
        iteration=0,
        matrix=cm,
        xaxis="Predicted Labels",
        yaxis="True Labels",
    )

    # I would not use this. Instead i can plot histogram using countplot
    Logger.current_logger().report_histogram(
        "Count of anomaly",
        "anomaly count",
        iteration=0,
        values=df["anomaly"].value_counts().to_list(),
        xaxis="Anomaly",
        yaxis="Count",
        xlabels=[1, -1],
    )

    # Save the model
    filename = "models/isolation_forest.joblib"
    joblib.dump(clf, filename)


if __name__ == "__main__":
    sys.exit(main())
