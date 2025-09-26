import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# path = "/home/ghoti/sim/Genesis/data/csv/004_sugar_box/004_sugar_box_aluminium_150.csv"
# path  = "/home/user/Genesis/data/eval_tmp/csv/002_master_chef_can/Rigid/none/002_master_chef_can_Rigid_none.csv"
path ="/home/user/Genesis/data/eval_tmp/csv/bottle_drop/Rigid/none/bottle_drop_Rigid_none.csv"

df = pd.read_csv(path)
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

# 上段：left_fx, left_fy, left_fz
for ax, col in zip(axs[0], ["left_fx", "left_fy", "left_fz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    ax.set_xlim(300, 500)
    # ax.set_ylim(-20, 80)

# 下段：right_fx, right_fy, right_fz
for ax, col in zip(axs[1], ["right_fx", "right_fy", "right_fz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    ax.set_xlim(300, 500)
    # ax.set_ylim(-20, 80)

fig.tight_layout()
fig.savefig(path.replace(".csv", "_force.png"), dpi=300)
