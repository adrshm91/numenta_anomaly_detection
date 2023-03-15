import sys
import json
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load machine temperature system failure data
    df = pd.read_csv("data/input/realKnownCause/machine_temperature_system_failure.csv")

    with open("data/labels/combined_labels.json") as label_file:
        labels = json.loads(label_file.read())
    labels = labels["realKnownCause/machine_temperature_system_failure.csv"]

    with open("data/labels/combined_windows.json") as window_file:
        windows = json.loads(window_file.read())
    windows = windows["realKnownCause/machine_temperature_system_failure.csv"]

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
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature sensor data with anomalies")
    plt.show()

    # Save processed data
    df.to_csv("data/processed/temperature_data.csv")


if __name__ == "__main__":
    sys.exit(main())
