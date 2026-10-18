import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=60)
N = 90
c = complex(-0.7, 0.27)


def julia(zoom):
    xs = np.linspace(-1.6 / zoom, 1.6 / zoom, N)
    ys = np.linspace(-1.6 / zoom, 1.6 / zoom, N)
    Z = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            z = complex(xs[b], ys[a])
            k = 0
            while k < 28 and abs(z) < 2.0:
                z = z * z + c
                k += 1
            Z[a, b] = k
    return Z


img = ax.imshow(julia(1.0), cmap="hot", animated=True)
ax.set_xticks([])
ax.set_yticks([])


def upd(i):
    img.set_array(julia(1.0 + i * 0.03))
    return img,


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
