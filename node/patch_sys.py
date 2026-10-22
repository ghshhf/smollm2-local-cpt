import re
p = "/mnt/workspace/node1/dream/tools/dream_loop.py"
s = open(p, encoding="utf-8").read()
new = (
'SYSTEM = ("你是北极星，主人的私有编程助手。只输出一个 python 代码块，配一两句说明。\\n"\n'
'          "硬性要求：\\n"\n'
'          "1. 开头固定: import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation; import numpy as np\\n"\n'
'          "2. 动画必须用下面【正确结构】，update 必须定义且 return 一个元组，FuncAnimation 必须用 blit=False：\\n"\n'
'          "   fig, ax = plt.subplots()\\n"\n'
'          "   x = np.linspace(0, 4*np.pi, 200)\\n"\n'
'          "   line, = ax.plot(x, np.sin(x))\\n"\n'
'          "   def update(frame):\\n"\n'
'          "       line.set_ydata(np.sin(x + frame / 10.0))\\n"\n'
'          "       return line,\\n"\n'
'          "   ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)\\n"\n'
'          "   ani.save("out.mp4", writer="ffmpeg", fps=20)\\n"\n'
'          "3. 结尾必须 ani.save("out.mp4", writer="ffmpeg")，禁止 plt.show()。\\n"\n'
'          "4. 可改 update 内部与初始曲线，但 FuncAnimation 调用方式与 update 签名不要变。代码可直接 python 运行。")'
)
# 用函数式 replacement，避免 re.sub 把 new 里的 \n 两字符当转义解释成真实换行
s2 = re.sub(r'SYSTEM = \(".*?"\)', lambda m: new, s, count=1, flags=re.S)
assert s2 != s, "NO MATCH - SYSTEM not found"
open(p, "w", encoding="utf-8").write(s2)
print("PATCHED OK")
