import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, 4 * np.pi, 300)
line, = ax.plot(x, np.cos(x), lw=1.6)
ax.set_ylim(-1.4, 1.4)


def upd(i):
    line.set_ydata(np.cos(x + i / 10.0))
    ax.set_title("cos phase %d" % i)
    return line,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
