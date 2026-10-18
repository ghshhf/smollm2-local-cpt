import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=60)
t = np.linspace(0, 2.0 * np.pi, 800)
m, n = 3, 4
xs = np.sin(m * t)
ys = np.sin(n * t)
line, = ax.plot(xs, ys, lw=1.2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    k = i / 30.0
    line.set_xdata(np.sin(m * t + k))
    line.set_ydata(np.sin(n * t))
    return line,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
