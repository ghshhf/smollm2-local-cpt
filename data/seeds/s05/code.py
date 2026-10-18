import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=60)
theta = 2 * np.pi * np.linspace(0, 2, 800)
r = np.linspace(0, 1, 800)
xs = r * np.cos(theta)
ys = r * np.sin(theta)
line, = ax.plot(xs, ys, lw=1.2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    k = i / 25.0
    line.set_xdata(r * np.cos(theta + k))
    line.set_ydata(r * np.sin(theta + k))
    return line,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
