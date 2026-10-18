import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(3.8, 3.4), dpi=60)
ax = fig.add_subplot(111, projection="3d")
X, Y = np.meshgrid(np.linspace(0, 4 * np.pi, 50), np.linspace(0, 4 * np.pi, 50))
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)


def upd(i):
    ax.view_init(elev=28, azim=i * 5)
    return ax,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
