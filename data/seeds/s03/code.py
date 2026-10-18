import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, 4 * np.pi, 400)
sq = np.sign(np.sin(x))
tri = (2.0 / np.pi) * np.arcsin(np.sin(x))
l1, = ax.plot(x, np.sin(x), lw=1.3, label="sin")
l2, = ax.plot(x, sq, lw=1.0, label="square")
l3, = ax.plot(x, tri, lw=1.0, label="triangle")
ax.set_ylim(-1.5, 1.5)
ax.legend(loc="upper right", fontsize=7)


def upd(i):
    s = i / 14.0
    l1.set_ydata(np.sin(x + s))
    l2.set_ydata(np.sign(np.sin(x + s)))
    l3.set_ydata((2.0 / np.pi) * np.arcsin(np.sin(x + s)))
    return l1, l2, l3


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
