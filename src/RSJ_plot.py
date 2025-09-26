import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

data_dir = "/home/user/Genesis/data/eval_tmp/csv/"

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Define datasets and start indices
datasets = {
    "grab": {"path": "bottle_drop2/Rigid/none/bottle_drop2_Rigid_none.csv", "start": 0},
    "place": {"path": "cube_slip/Rigid/none/cube_slip_Rigid_none.csv", "start": 2440},
    "hold": {"path": "cube_slip/Rigid/none/cube_slip_Rigid_none.csv", "start": 2150},
    "no_contact": {"path": "bottle_drop2/Rigid/none/bottle_drop2_Rigid_none.csv", "start": 1800},
}

modes = ["normal", "insight"]

for mode in modes:
    for data_type, info in datasets.items():
        path = data_dir + info["path"]
        start = info["start"]

        # path ="/home/user/Genesis/data/eval_tmp/csv/001_chips_can/Rigid/none/001_chips_can_Rigid_none.csv"
        df = pd.read_csv(path)
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

        normal_ylims = [(-1, 1), (-10, 30), (-50, 10)]
        hold_nc_ylims = [(-0.05, 0.05), (2.8, 3.1), (-0.1, 0.6)]
        # 上段：left_fx, left_fy, left_fz
        for ax, col in zip(axs[0], ["left_fx", "left_fy", "left_fz"]):
            ax.plot(df["step"][0:80], df[col][start:start+80])
            ax.set_ylabel(col)
            # Limit number of y-axis ticks to avoid overly dense labels
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            if mode == "normal":
                ax.set_ylim(normal_ylims[["left_fx", "left_fy", "left_fz"].index(col)])
            elif data_type == "hold" or data_type == "no_contact":
                ax.set_ylim(hold_nc_ylims[["left_fx", "left_fy", "left_fz"].index(col)])

        # 下段：right_fx, right_fy, right_fz
        for ax, col in zip(axs[1], ["right_fx", "right_fy", "right_fz"]):
            ax.plot(df["step"][0:80], df[col][start:start+80])
            ax.set_xlabel("step")
            ax.set_ylabel(col)
            # Limit number of y-axis ticks to avoid overly dense labels
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            if mode == "normal":
                ax.set_ylim(normal_ylims[["right_fx", "right_fy", "right_fz"].index(col)])
            elif data_type == "hold" or data_type == "no_contact":
                ax.set_ylim(hold_nc_ylims[["right_fx", "right_fy", "right_fz"].index(col)])

        fig.tight_layout()
        # 画像として保存
        fig.savefig(f"{mode}_{data_type}.png", dpi=300)
        plt.close(fig)
