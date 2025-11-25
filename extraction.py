import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wearable_acceleration_extract_0900_0.csv", sep="\t")
# print(df)

# dataset_low = df[df["label"] == "low_cupboard"]

# plt.plot(dataset_low["timestamp"], dataset_low["x"])
# plt.show()

labels = list(set(df["label"].tolist()))
# print(labels)


def activities(label=("low_cupboard","high_cupboard"),dataset=df):
    start_label, end_label = label[0], label[1]
    try:
        start_idx = dataset.index[dataset["label"] == start_label][0]
    except IndexError:
        return f"Start label {start_label} not found"
    try:
        end_idx = dataset.index[dataset["label"] == end_label][-1]
    except IndexError:
        return f"End label {end_label} not found"
    # Ensure proper ordering
    if start_idx > end_idx:
        return "Start occurs after end — slice not possible."

    
 
    dataset_low = dataset.loc[start_idx:end_idx].copy()
    # Sort timestamps to prevent backward lines
    dataset_low = dataset_low.sort_values("ts").reset_index(drop=True)
    # Convert timestamp → datetime
    dataset_low["ts"] = pd.to_datetime(dataset_low["ts"])
    # Convert numeric columns to float (or int if you really want)
    dataset_low["x"] = dataset_low["x"].astype(float)
    dataset_low["y"] = dataset_low["y"].astype(float)
    dataset_low["z"] = dataset_low["z"].astype(float)
    
    axis=["x","y","z"]
    fig,axes=plt.subplots(3,1,figsize=(16,10))
    for ax, dim in zip(axes,axis):
        ax.plot(dataset_low["ts"],dataset_low[dim],label=f"{dim} acceleration")
        ax.set_xlabel("Time in time stamps")
        ax.set_ylabel("in meters")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    plt.suptitle(f"The activity here is {label[0]} to {label[1]}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    activities(("low_cupboard","table_rear"))