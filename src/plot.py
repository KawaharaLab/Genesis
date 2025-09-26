import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# item = "soft_cube_strong"  # Change this to "soft_cube" for the other example
# df = pd.read_csv(f"/home/ghoti/sim/Genesis/grasp_{item}.csv")

# path = "/home/ghoti/sim/Genesis/data/csv/004_sugar_box/004_sugar_box_aluminium_150.csv"
# path  = "/home/ghoti/sim/Genesis/grasp_bottle_world.csv"
# path ="/home/user/Genesis/data/eval_tmp/csv/bottle_drop2/Rigid/none/bottle_drop2_Rigid_none.csv"
path = "/home/user/Genesis/data/eval_video/csv/002_master_chef_can/Rigid/none/002_master_chef_can_Rigid_none.csv"
start = 520
df = pd.read_csv(path).iloc[start:start+80].reset_index(drop=True)
# lines =  [430, 530, 780, 880]
lines = []
# lines =  [330, 430, 530, 630]
# 2行×3列のサブプロットを作成
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
times = [i for i in range(80)]

# 上段：left_fx, left_fy, left_fz
f_lims = [(-0.3, 0.3), (2, 4), (0, 1)]
for ax, col in zip(axs[0], ["fx", "fy", "fz"]):
    ax.plot(times, df[f"left_{col}"])
    # ax.set_title(col[1], fontsize=14)
    # ax.set_xlabel("step")
    if col == "fx":
        ax.set_ylabel("Force [N]", fontsize=14)
    # Format y-axis ticks to 2 significant digits
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
    # ax.set_ylim(f_lims[["fx", "fy", "fz"].index(col)])
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(timestep)
    # ax.set_ylim(0, 10)

# 下段：left_tx, left_ty, left_tz
# t_lims = [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
t_lims = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
for ax, col in zip(axs[1], ["tx", "ty", "tz"]):
    ax.plot(times, df[f"left_{col}"])
    # ax.set_title(col)
    ax.set_xlabel("time [ms]", fontsize=14)
    if col == "tx":
        ax.set_ylabel("Torque [N*m]", fontsize=14)
    # Format y-axis ticks to 2 significant digits
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
    # ax.set_ylim(t_lims[["tx", "ty", "tz"].index(col)])
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(timestep)
    # ax.set_ylim(-500, 500)

fig.tight_layout()
# 画像として保存
# fig.savefig(f"/home/ghoti/sim/Genesis/plot_{item}.png", dpi=300)
fig.savefig(path.replace(".csv", "_left.jpg"), dpi=300)

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
# 上段：right_fx, right_fy, right_fz
for ax, col in zip(axs[0], ["fx", "fy", "fz"]):
    ax.plot(times, df[f"right_{col}"])
    # ax.set_title(col[1], fontsize=14)
    # ax.set_xlabel("step")
    if col == "fx":
        ax.set_ylabel("Force [N]", fontsize=14)
    # Format y-axis ticks to 2 significant digits
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
    # ax.set_ylim(f_lims[["fx", "fy", "fz"].index(col)])
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(timestep)
    # ax.set_ylim(-500, 500)
# 下段：right_tx, right_ty, right_tz
for ax, col in zip(axs[1], ["tx", "ty", "tz"]):
    ax.plot(times, df[f"right_{col}"])
    # ax.set_title(col)
    ax.set_xlabel("time [ms]", fontsize=14)
    if col == "tx":
        ax.set_ylabel("Torque [N*m]", fontsize=14)
    # Format y-axis ticks to 2 significant digits
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
    # ax.set_ylim(t_lims[["tx", "ty", "tz"].index(col)])
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(timestep)
    # ax.set_ylim(-500, 500)

fig.tight_layout()
# 画像として保存
fig.savefig(path.replace(".csv", "_right.jpg"), dpi=300)