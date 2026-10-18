import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=60)
rng = np.random.default_rng(11)
K = 10
vals = rng.random(K)
idx = np.arange(K)
bars = ax.bar(idx, vals, color="tab:blue")
ax.set_ylim(0, 1.2)


def upd(i):
    global vals
    vals = vals + rng.normal(0, 0.04, size=K)
    vals = np.clip(vals, 0.02, 1.15)
    for b, v in zip(bars, vals):
        b.set_height(v)
    ax.set_title("frame %d" % i)
    return bars


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
