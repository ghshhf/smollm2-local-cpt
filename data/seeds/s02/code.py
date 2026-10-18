import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(0, 4 * np.pi, 300)
line, = ax.plot(x, np.sin(x), lw=1.6)
ax.set_ylim(-1.2, 1.2)


def upd(i):
    amp = np.exp(-i / 30.0)
    line.set_ydata(amp * np.sin(x - i / 10.0))
    ax.set_title("damped  amp=%.3f" % amp)
    return line,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
