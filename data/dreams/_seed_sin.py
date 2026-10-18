import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 4 * np.pi, 200)
line, = ax.plot(x, np.sin(x))
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, 4 * np.pi)


def upd(i):
    line.set_ydata(np.sin(x + i / 8.0))
    ax.set_title("sin wave phase %d" % i)
    return line,


ani = FuncAnimation(fig, upd, frames=40, blit=False)
ani.save('out.mp4', writer='ffmpeg', fps=10)
print('DONE')
