import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

df = pd.read_csv("fps_log.csv")
df["datetime"] = pd.to_datetime(df["datetime"])


df = df.sort_values(["segment_name", "frame_index"])

df["t_rel"] = df.groupby("segment_name")["datetime"].transform(
    lambda x: (x - x.iloc[0]).dt.total_seconds()
)


W = 20  # smoothing window
df["ma_lane"] = (
    df.groupby(["segment_name","num_lanes"])["total_fps"]
      .rolling(window=W, min_periods=1)
      .mean()
      .reset_index(level=[0,1], drop=True)
)


df["ma_total"] = (
    df.groupby("segment_name")["total_fps"]
      .rolling(window=W, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)


segments = df["segment_name"].unique()
idx = 0


cmap = get_cmap("tab10")

fig, ax = plt.subplots(figsize=(10,4))

def plot_segment(i):
    ax.clear()
    seg = segments[i]
    g = df[df.segment_name == seg]
    

    for lane_i, (lanes, h) in enumerate(g.groupby("num_lanes")):
        ax.plot(
            h["t_rel"],
            h["ma_lane"],
            linewidth=2,
            color=cmap(lane_i % 10),
            label=f"{lanes} lanes"
        )


    ax.plot(
        g["t_rel"],
        g["ma_total"],
        linewidth=3,
        color="black",
        linestyle="--",
        label="MA total"
    )

    ax.set_title(f"Segment: {seg}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FPS")
    ax.legend()
    fig.canvas.draw_idle()


def on_key(event):
    global idx
    if event.key == "right":
        idx = (idx + 1) % len(segments)
        plot_segment(idx)
    elif event.key == "left":
        idx = (idx - 1) % len(segments)
        plot_segment(idx)


fig.canvas.mpl_connect("key_press_event", on_key)

plot_segment(idx)
plt.tight_layout()
plt.show()
