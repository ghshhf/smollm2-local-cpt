import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=60)
rng = np.random.default_rng(3)
M = 4 * 10
arr0 = rng.random(M)
states = [arr0.copy()]
cur = arr0.copy()
for a in range(M):
    for b in range(M - 1 - a):
        if cur[b] > cur[b + 1]:
            cur[b], cur[b + 1] = cur[b + 1], cur[b]
        states.append(cur.copy())
step = max(1, len(states) // max(1, 100))
picked = states[::step][:100]
bars = ax.bar(np.arange(M), picked[0], color="tab:orange")
ax.set_ylim(0, 1.1)


def upd(i):
    snap = picked[min(i, len(picked) - 1)]
    for b, v in zip(bars, snap):
        b.set_height(v)
    ax.set_title("bubble step %d" % i)
    return bars


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
