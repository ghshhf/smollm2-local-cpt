import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(4, 3), dpi=60)
x = np.linspace(-np.pi, np.pi, 500)
target = np.sign(np.sin(x))
line, = ax.plot(x, np.zeros_like(x), lw=1.5)
ref, = ax.plot(x, target, lw=1.0, ls="--", color="gray")
ax.set_ylim(-1.6, 1.6)
NT = 4


def partial(n_terms):
    y = np.zeros_like(x)
    for k in range(1, 2 * n_terms, 2):
        y += np.sin(k * x) / k
    return y * 4.0 / np.pi


def upd(i):
    n = min(i + 1, NT)
    line.set_ydata(partial(n))
    ax.set_title("fourier n=%d" % n)
    return line, ref


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
