import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from clearml import Task, TaskTypes, Dataset

task = Task.init(
    project_name="numenta_anomaly_detection",
    task_name="preprocess",
    task_type=TaskTypes.data_processing,
    reuse_last_task_id=False,
)


def main(dataset_name="realKnownCause/machine_temperature_system_failure"):
    # Load input data
    df = pd.read_csv(f"data/input/{dataset_name}.csv")

    with open("data/labels/combined_labels.json") as label_file:
        labels = json.loads(label_file.read())
    labels = labels[f"{dataset_name}.csv"]

    with open("data/labels/combined_windows.json") as window_file:
        windows = json.loads(window_file.read())
    windows = windows[f"{dataset_name}.csv"]

    # Set anomaly label according to windows
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # -1 for outliers and 1 for inliers.
    df["anomaly"] = 1

    # Set timestamps overlapping with windows as anomaly
    for window in windows:
        mask = (df.index >= window[0]) & (df.index <= window[1])
        df.loc[mask, "anomaly"] = -1

    # Plot data and labels
    ax = df["value"].plot(figsize=(10, 5))
    for window in windows:
        ax.axvspan(xmin=window[0], xmax=window[1], alpha=0.2, color="red")
        df_slice = df[(df.index >= window[0]) & (df.index <= window[1])]
        idx = df_slice.shape[0] // 2
        ax.scatter(x=df_slice.index[idx], y=df_slice.iloc[idx]["value"])
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Sensor data with anomalies")
    plt.savefig("results/sensor_data.png")

    # Save processed data
    df.to_csv("data/processed/data_processed.csv")

    dataset = Dataset.create(
        dataset_name="numenta", dataset_project="numenta_anomaly_detection"
    )

    dataset.add_files(path="data")

    dataset.finalize(auto_upload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    sys.exit(main(dataset_name))
