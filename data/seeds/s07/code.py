import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=60)
N = 48
Z = np.zeros((N, N))
X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
im = ax.imshow(Z, cmap="viridis", animated=True, vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])


def upd(i):
    t = i / 12.0
    z = np.sin(np.sqrt(X ** 2 + Y ** 2) * 3 - t) * np.exp(-0.3 * np.sqrt(X ** 2 + Y ** 2))
    im.set_array(z)
    return im,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
