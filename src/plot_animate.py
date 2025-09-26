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
height = 3
# scale width such that ~200 samples -> width 6, cap to reasonable range
width = min(8, min(120, num_samples / 30))

fig, ax = plt.subplots(figsize=(width, height))

# data for animation
x = df["step"].values
y = df["left_fy"].values
ax.set_title("Left Finger, vertical", fontsize=14)
# prepare axis
ax.set_ylabel("Force [N]", fontsize=12)
# ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
ax.set_xlabel('time [ms]', fontsize=10)

ax.set_xlim(x.min(), x.max())
yy_min, yy_max = y.min(), y.max()
margin = (yy_max - yy_min) * 0.05 if (yy_max - yy_min) != 0 else 1.0
ax.set_ylim(yy_min - margin, yy_max + margin)

# line object that will be updated
# line object that will be updated: thinner line, smaller hollow markers
line, = ax.plot([], [], '-', linewidth=0.5, markersize=1, marker='.', markerfacecolor='none', markeredgewidth=0.6)

# prepare vertical lines every 80 steps (they start hidden and become visible when passed)
step_start = int(x.min())
step_end = int(x.max())
v_positions = np.arange(step_start, step_end + 1, 80)
# create axvline artists but keep them invisible initially
vlines = [ax.axvline(x=pos, color='r', linestyle='--', linewidth=1, visible=False) for pos in v_positions]

def init():
	line.set_data([], [])
	return (line,)

def update(frame):
	i = frame + 1
	xi = x[:i]
	yi = y[:i]
	line.set_data(xi, yi)
	# reveal vertical lines whose position is <= current x
	current_x = xi[-1]
	for vline, pos in zip(vlines, v_positions):
		if (not vline.get_visible()) and (current_x >= pos):
			vline.set_visible(True)
	return (line,)

# animation parameters
fps = 100
frames = len(x)
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
							  blit=False, interval=1000/fps)

fig.tight_layout()
# save with tight bbox; adapt dpi for width
save_dpi = 200 if width < 60 else 150
out_path = path.replace('.csv', '_narrow.mp4')

# try ffmpeg writer first, then pillow GIF fallback; finally save a PNG as last resort
FFWriter = None
try:
	# Some matplotlib versions support dict-like access
	FFWriter = animation.writers['ffmpeg'] if 'ffmpeg' in animation.writers.list() else None
except Exception:
	# Fallback: try accessing by attribute or other registry behaviors
	try:
		FFWriter = animation.writers['ffmpeg']
	except Exception:
		FFWriter = None

if FFWriter is not None:
	try:
		writer = FFWriter(fps=fps, metadata=dict(artist='genesis'), bitrate=1800)
		ani.save(out_path, writer=writer, dpi=save_dpi)
		print(f"Saved animation to {out_path}")
	except Exception as e:
		print(f"Failed to save with ffmpeg writer: {e}")
		FFWriter = None

if FFWriter is None:
	try:
		gif_path = path.replace('.csv', '_narrow.gif')
		ani.save(gif_path, writer='pillow', fps=fps, dpi=save_dpi)
		print(f"Saved GIF to {gif_path}")
	except Exception as e:
		# last-resort: save final frame as PNG (use thin line style)
		last_png = path.replace('.csv', '_narrow_last.png')
		ax.plot(x, y, '-', linewidth=0.2, markersize=1, marker='.', markerfacecolor='none', markeredgewidth=0.6)
		fig.tight_layout(pad=0.2)
		fig.savefig(last_png, dpi=save_dpi, bbox_inches='tight')
		print(f"Could not save animation (error: {e}). Saved final frame to {last_png}")

plt.close(fig)