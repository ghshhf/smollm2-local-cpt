import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=60)
N = 32
rng = np.random.default_rng(5)
grid = (rng.random((N, N)) < 0.3).astype(np.int8)
im = ax.imshow(grid, cmap="binary", animated=True)
ax.set_xticks([])
ax.set_yticks([])


def step(g):
    nb = sum(np.roll(np.roll(g, i, 0), j, 1)
             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    alive = (nb == 3) | ((g == 1) & (nb == 2))
    return alive.astype(np.int8)


def upd(i):
    global grid
    grid = step(grid)
    im.set_array(grid)
    ax.set_title("gen %d" % i)
    return im,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
