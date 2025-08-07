import matplotlib.pyplot as plt
import pandas as pd

# item = "soft_cube_strong"  # Change this to "soft_cube" for the other example
# df = pd.read_csv(f"/home/ghoti/sim/Genesis/grasp_{item}.csv")

# path = "/home/ghoti/sim/Genesis/data/csv/004_sugar_box/004_sugar_box_aluminium_150.csv"
# path  = "/home/ghoti/sim/Genesis/grasp_bottle_world.csv"
path ="data/csv/001_chips_can/001_chips_can_rubber_500.csv"
df = pd.read_csv(path)
# lines =  [430, 530, 780, 880]
lines = []
# lines =  [330, 430, 530, 630]
# 2行×3列のサブプロットを作成
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

# 上段：left_fx, left_fy, left_fz
for ax, col in zip(axs[0], ["left_fx", "left_fy", "left_fz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(0, 14000)
    # ax.set_ylim(-500, 500)

# 下段：left_tx, left_ty, left_tz
for ax, col in zip(axs[1], ["left_tx", "left_ty", "left_tz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(0, 14000)
    # ax.set_ylim(-500, 500)

fig.tight_layout()
# 画像として保存
# fig.savefig(f"/home/ghoti/sim/Genesis/plot_{item}.png", dpi=300)
fig.savefig(path.replace(".csv", "_left.png"), dpi=300)

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
# 上段：right_fx, right_fy, right_fz
for ax, col in zip(axs[0], ["right_fx", "right_fy", "right_fz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(0, 14000)
    # ax.set_ylim(-500, 500)
# 下段：right_tx, right_ty, right_tz
for ax, col in zip(axs[1], ["right_tx", "right_ty", "right_tz"]):
    ax.plot(df["step"], df[col])
    ax.set_title(col)
    ax.set_xlabel("step")
    ax.set_ylabel(col)
    for line in lines:
        ax.axvline(x=line, color='r', linestyle='--')
    # ax.set_xlim(0, 14000)
    # ax.set_ylim(-500, 500)

fig.tight_layout()
# 画像として保存
fig.savefig(path.replace(".csv", "_right.png"), dpi=300)