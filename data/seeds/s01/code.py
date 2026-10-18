import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, 4 * np.pi, 300)
l1, = ax.plot(x, np.sin(x), lw=1.4, label="sin")
l2, = ax.plot(x, np.cos(x), lw=1.4, label="cos")
ax.set_ylim(-1.4, 1.4)
ax.legend(loc="upper right", fontsize=7)


def upd(i):
    l1.set_ydata(np.sin(x + i / 12.0))
    l2.set_ydata(np.cos(x + i / 12.0))
    return l1, l2


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
