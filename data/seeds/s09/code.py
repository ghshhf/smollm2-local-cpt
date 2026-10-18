import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=60)
rng = np.random.default_rng(7)
N = 4 * 100
pos = rng.normal(0, 0.3, size=(N, 2))
sc = ax.scatter(pos[:, 0], pos[:, 1], s=6, alpha=0.6)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)


def upd(i):
    global pos
    pos = np.clip(pos + rng.normal(0, 0.05, size=(N, 2)), -2.4, 2.4)
    sc.set_offsets(pos)
    return sc,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
