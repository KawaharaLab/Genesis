import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
import numpy as np
import pandas as pd

# CSV path and read
path = "/home/user/Genesis/data/eval_video/csv/010_potted_meat_can/Rigid/none/010_potted_meat_can_Rigid_none.csv"
df = pd.read_csv(path).iloc[20:1600].reset_index(drop=True)
# start = 520
# df = pd.read_csv(path).iloc[start:start+80].reset_index(drop=True)

# determine figure size to make output tall and narrow based on number of samples
num_samples = len(df)
# set height small and scale width with num_samples so long timeseries become wide
height = 2
# scale width such that ~200 samples -> width 6, cap to reasonable range
# width = min(8, min(120, num_samples / 30))
width = 8

fig, ax = plt.subplots(figsize=(width, height))
lr = "left"
f_type = "fy" 
# data for animation
x = df["step"].values
y = df[f"{lr}_{f_type}"].values
# ax.set_title("Left Finger, vertical", fontsize=14)
# prepare axis
ax.set_ylabel("Force [N]", fontsize=12)
# ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
# ax.set_xlabel('time [ms]', fontsize=10)

ax.set_xlim(x.min(), x.max())
yy_min, yy_max = y.min(), y.max()
margin = (yy_max - yy_min) * 0.05 if (yy_max - yy_min) != 0 else 1.0
ax.set_ylim(yy_min - margin, yy_max + margin)

# Static plot: draw the whole series with thin line and small markers
ax.plot(x, y, '-', linewidth=0.5, markersize=1, marker='.', markerfacecolor='none', markeredgewidth=0.6)

# vertical lines every 80 steps (visible)
step_start = int(x.min())
step_end = int(x.max())
v_positions = np.arange(step_start, step_end + 1, 80)
# for pos in v_positions:
	# ax.axvline(x=pos, color='r', linestyle='--', linewidth=1)

fig.tight_layout(pad=0.2)

# save as JPEG
save_dpi = 200 if width < 60 else 150
out_path = path.replace('.csv', f'_{lr}_{f_type}.jpg')
fig.savefig(out_path, dpi=save_dpi, bbox_inches='tight')
print(f"Saved static plot to {out_path}")

plt.close(fig)