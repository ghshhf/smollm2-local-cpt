import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=60)
g = 9.8
x, y = 0.0, 0.0
vx, vy = 4.0, 9.0
trail_x, trail_y = [x], [y]
line, = ax.plot(trail_x, trail_y, lw=1.2)
dot, = ax.plot([x], [y], "o", ms=6, color="tab:red")
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)


def upd(i):
    global x, y, vx, vy
    dt = 0.06
    x += vx * dt
    y += vy * dt
    vy -= g * dt
    if y <= 0:
        y = 0.0
        vy = -vy * 0.8
    if x > 11.5:
        x, y, vx, vy = 0.0, 0.0, 4.0, 9.0
        trail_x.clear()
        trail_y.clear()
    trail_x.append(x)
    trail_y.append(y)
    line.set_xdata(trail_x)
    line.set_ydata(trail_y)
    dot.set_xdata([x])
    dot.set_ydata([y])
    return line, dot


ani = FuncAnimation(fig, upd, frames=100, blit=False)
ani.save("out.mp4", writer="ffmpeg", fps=10)
