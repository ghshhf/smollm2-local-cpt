# -*- coding: utf-8 -*-
"""代码种子 第 2 批：热力学（51-72） + 物理工程与算法（73-100）。

每条 = (instruction, code, note)，硬约束同 seed_code_p1.py。
"""

S = []


def add(instruction, code, note):
    S.append((instruction, code.strip('\n'), note))


# ---------- L4 热力学（51-72） ----------
add('写一段 Python 代码，定义函数用理想气体状态方程求压强，n=2 mol、T=300 K、V=0.05 m^3，打印压强（Pa）。', '''
R = 8.314  # 理想气体常数 J/(mol*K)


def ideal_pressure(n, T, V):
    return n * R * T / V


print(round(ideal_pressure(2, 300, 0.05), 1))
''', 'PV = nRT，所以压强 P 等于 nRT 除以体积 V。')

add('写一段 Python 代码，定义三个函数在摄氏、华氏、开尔文之间互转，打印 25 摄氏度对应的华氏度和开尔文。', '''
def c2f(c):
    return c * 9 / 5 + 32


def c2k(c):
    return c + 273.15


def f2c(f):
    return (f - 32) * 5 / 9


print(round(c2f(25), 2), round(c2k(25), 2))
''', '三个温标线性互转，开尔文只是摄氏度加 273.15。')

add('写一段 Python 代码，定义函数计算显热 Q=mcΔT，2 kg 水升温 30 度（c=4186）需要多少焦耳，打印结果。', '''
def sensible_heat(m, c, dT):
    return m * c * dT


print(round(sensible_heat(2, 4186, 30), 1))
''', '显热等于质量乘比热容乘温差，不涉及相变。')

add('写一段 Python 代码，定义函数求两杯水混合后的热平衡终温（1 kg 80 度与 2 kg 20 度，c 相同），打印终温。', '''
def equilibrium_T(m1, T1, m2, T2, c=4186):
    # 放热等于吸热：m1*c*(T1-T) = m2*c*(T-T2)
    return (m1 * T1 + m2 * T2) / (m1 + m2)


print(round(equilibrium_T(1, 80, 2, 20), 2))
''', '比热容相同时终温就是按质量加权平均。')

add('写一段 Python 代码，定义函数计算卡诺热机效率，高温热源 600 K、低温热源 300 K，打印效率。', '''
def carnot_eff(Tc, Th):
    return 1 - Tc / Th


print(round(carnot_eff(300, 600), 4))
''', '卡诺效率只由两个热源温度决定，是同温区热机的上限。')

add('写一段 Python 代码，定义函数由热机效率和吸热量算输出功，再按每秒 2 次循环算功率，效率 0.35、吸热 1000 J，打印功和功率。', '''
def engine_work(eff, Qh):
    return eff * Qh


def power(eff, Qh, cycles_per_sec):
    return engine_work(eff, Qh) * cycles_per_sec


print(round(engine_work(0.35, 1000), 1), round(power(0.35, 1000, 2), 1))
''', '输出功等于效率乘吸热量，功率再乘以单位时间循环次数。')

add('写一段 Python 代码，定义函数计算理想气体等温膨胀的熵变，n=2 mol、体积从 1 到 3 立方米，打印熵变（J/K）。', '''
import math


def isothermal_entropy(n, V1, V2, R=8.314):
    return n * R * math.log(V2 / V1)


print(round(isothermal_entropy(2, 1, 3), 3))
''', '等温过程熵变等于 nR 乘体积比的自然对数。')

add('写一段 Python 代码，定义函数计算把 0.5 kg 的冰从 -10 度加热成 50 度的水所需总热量，分三段计算并打印。', '''
def heat_ice_to_water(m):
    c_ice, c_water, Lf = 2100, 4186, 334000
    q1 = m * c_ice * 10        # 冰从 -10 升到 0
    q2 = m * Lf                # 0 度融化
    q3 = m * c_water * 50      # 水从 0 升到 50
    return round(q1, 1), round(q2, 1), round(q3, 1), round(q1 + q2 + q3, 1)


print(heat_ice_to_water(0.5))
''', '升温段用显热公式，融化段用潜热，总热量是三段之和。')

add('写一段 Python 代码，用傅里叶定律定义函数计算导热热量，k=0.5、面积 2 平米、温差 20 度、厚 0.1 米、持续 60 秒，打印热量。', '''
def conduction(k, A, dT, d, t):
    return k * A * dT * t / d


print(round(conduction(0.5, 2, 20, 0.1, 60), 1))
''', '导热速率与温差和面积成正比、与厚度成反比。')

add('写一段 Python 代码，定义函数计算金属棒的热膨胀伸长量，原长 10 米、线膨胀系数 1.2e-5、升温 50 度，打印伸长量和新长度。', '''
def linear_expand(L0, alpha, dT):
    dL = alpha * L0 * dT
    return round(dL, 6), round(L0 + dL, 6)


print(linear_expand(10, 1.2e-5, 50))
''', '伸长量等于线膨胀系数乘原长乘温差。')

add('写一段 Python 代码，定义函数用范德瓦尔斯方程求真实气体压强，CO2 取 a=0.364、b=4.27e-5，n=1、V=0.024、T=300，打印压强并与理想气体对比。', '''
R = 8.314


def vdw_pressure(n, V, T, a=0.364, b=4.27e-5):
    return n * R * T / (V - n * b) - a * n * n / (V * V)


def ideal_pressure(n, V, T):
    return n * R * T / V


print(round(vdw_pressure(1, 0.024, 300), 1), round(ideal_pressure(1, 0.024, 300), 1))
''', '范德瓦尔斯方程修正了分子体积和引力，高压下偏离理想气体。')

add('写一段 Python 代码，定义函数用斯特凡-玻尔兹曼定律计算辐射功率，发射率 0.9、面积 1 平米、温度 500 K，打印功率（W）。', '''
SIGMA = 5.67e-8


def radiation(eps, A, T):
    return eps * SIGMA * A * T ** 4


print(round(radiation(0.9, 1, 500), 1))
''', '辐射功率与热力学温度的四次方成正比，高温下增长极快。')

add('写一段 Python 代码，定义函数计算 1 mol 单原子理想气体等压膨胀的做功、内能变化和吸热量，P=101325 Pa，体积 0.01 到 0.02 立方米，打印三项。', '''
R = 8.314


def isobaric(n, P, V1, V2, Cv=1.5 * R):
    W = P * (V2 - V1)
    dT = P * (V2 - V1) / (n * R)
    dU = n * Cv * dT
    return round(W, 1), round(dU, 1), round(dU + W, 1)


print(isobaric(1, 101325, 0.01, 0.02))
''', '等压做功是 PΔV，吸热分成内能变化和对外做功两部分。')

add('写一段 Python 代码，定义函数计算理想气体绝热膨胀后的温度，T1=300 K、体积比 1:2、γ=1.4，打印终温。', '''
def adiabatic_T(T1, V1, V2, gamma=1.4):
    return T1 * (V1 / V2) ** (gamma - 1)


print(round(adiabatic_T(300, 1, 2), 2))
''', '绝热过程满足 TV^(γ-1) 为常数，膨胀必然降温。')

add('写一段 Python 代码，定义函数用生成焓按 Hess 定律算甲烷燃烧的反应焓（kJ/mol），打印结果。', '''
def reaction_enthalpy(products, reactants):
    return sum(products.values()) - sum(reactants.values())


hf = {"CO2": -393.5, "H2O": -285.8, "CH4": -74.8}
# CH4 + 2 O2 -> CO2 + 2 H2O(l)，O2 生成焓为 0
prod = {"CO2": hf["CO2"], "H2O": 2 * hf["H2O"]}
reac = {"CH4": hf["CH4"]}
print(round(reaction_enthalpy(prod, reac), 1))
''', '反应焓等于生成物生成焓之和减反应物之和，负值表示放热。')

add('写一段 Python 代码，定义函数计算理想气体内能，n=2 mol 双原子气体（Cv=5/2 R）在 350 K 时的内能，打印结果。', '''
R = 8.314


def internal_energy(n, T, Cv=2.5 * R):
    return n * Cv * T


print(round(internal_energy(2, 350), 1))
''', '理想气体内能只与温度和自由度有关，与体积压强无关。')

add('写一段 Python 代码，定义函数按热阻串联计算复合墙体的热流，三层厚度 [0.1,0.2,0.05]、导热系数 [0.5,0.04,0.8]、面积 1 平米、温差 30 度，打印热阻和热流。', '''
def composite_wall(layers, ks, A, dT):
    Rsum = sum(L / (k * A) for L, k in zip(layers, ks))
    return round(Rsum, 4), round(dT / Rsum, 3)


print(composite_wall([0.1, 0.2, 0.05], [0.5, 0.04, 0.8], 1, 30))
''', '每层热阻是 L/(kA)，串联相加，热流等于总温差除以总热阻。')

add('写一段 Python 代码，定义函数计算制冷机的理论制冷系数 COP，冷库 270 K、环境 310 K，打印 COP 和移走 1000 J 热量所需的最小功。', '''
def cop_carnot(Tc, Th):
    return Tc / (Th - Tc)


def min_work(Tc, Th, Qc):
    return Qc / cop_carnot(Tc, Th)


c = cop_carnot(270, 310)
print(round(c, 3), round(min_work(270, 310, 1000), 2))
''', 'COP 等于移热量除以耗功，温差越小 COP 越高。')

add('写一段 Python 代码，用 Magnus 公式定义函数由相对湿度和气温算露点，湿度 60%、气温 25 度，打印露点温度。', '''
import math


def dew_point(rh, T):
    a, b = 17.62, 243.12
    gamma = math.log(rh) + a * T / (b + T)
    return b * gamma / (a - gamma)


print(round(dew_point(0.6, 25), 2))
''', '露点是水汽开始凝结的温度，Magnus 公式是工程上常用的近似。')

add('写一段 Python 代码，定义函数用克劳修斯不等式判断热机过程是否可行：从 600 K 吸热 1000 J，分别向 300 K 放热 500 J、600 J、400 J，打印三种情况的熵变与结论。', '''
def entropy_balance(Qh, Th, Qc, Tc):
    dS = -Qh / Th + Qc / Tc
    if dS > 1e-9:
        verdict = "irreversible"
    elif dS < -1e-9:
        verdict = "impossible"
    else:
        verdict = "reversible"
    return round(dS, 5), verdict


for Qc in (500, 600, 400):
    print(Qc, entropy_balance(1000, 600, Qc, 300))
''', '总熵变大于 0 是不可逆、等于 0 是可逆、小于 0 违反第二定律不可能发生。')

add('写一段 Python 代码，定义函数计算燃料燃烧的有效输出，1 kg 汽油热值 44 MJ/kg、热机效率 30%，打印总放热和有效功（MJ）。', '''
def fuel_output(m, heat_value, eff):
    total = m * heat_value
    return round(total, 2), round(total * eff, 2)


print(fuel_output(1, 44.0, 0.3))
''', '有效功等于燃料总热值乘热机效率，剩下的都排给环境。')

add('写一段 Python 代码，用显式有限差分定义函数模拟一维非稳态热传导：棒长 1 米、热扩散率 1e-4、左端恒温 100 度、初始 0 度，跑 100 步后打印中点温度。', '''
def rod_heat(nx=11, L=1.0, alpha=1e-4, TL=100.0, steps=100):
    dx = L / (nx - 1)
    dt = 0.5 * dx * dx / alpha       # 满足显式格式稳定条件
    T = [0.0] * nx
    T[0] = TL
    for _ in range(steps):
        new = T[:]
        for i in range(1, nx - 1):
            new[i] = T[i] + alpha * dt / (dx * dx) * (T[i - 1] - 2 * T[i] + T[i + 1])
        T = new
    return round(T[nx // 2], 4)


print(rod_heat())
''', '显式差分用上一时刻的温度推进下一时刻，时间步必须满足稳定条件。')

# ---------- L5 物理工程与算法（73-100） ----------
add('写一段 Python 代码，定义函数计算自由落体，从 20 米高处落下（g=9.8），打印落地时间和落地速度。', '''
import math


def free_fall(h, g=9.8):
    t = math.sqrt(2 * h / g)
    return round(t, 3), round(g * t, 3)


print(free_fall(20))
''', '由 h = gt²/2 解出时间，再乘 g 得末速度。')

add('写一段 Python 代码，定义函数计算斜抛运动的射程和最大高度，初速 30 m/s、仰角 45 度、g=9.8，打印结果。', '''
import math


def projectile(v0, deg, g=9.8):
    a = math.radians(deg)
    rng = v0 * v0 * math.sin(2 * a) / g
    hmax = (v0 * math.sin(a)) ** 2 / (2 * g)
    return round(rng, 2), round(hmax, 2)


print(projectile(30, 45))
''', '射程用 sin2θ、最大高度用竖直分速度算，45 度射程最远。')

add('写一段 Python 代码，定义函数计算水平拉力下带摩擦滑块的加速度，质量 5 kg、拉力 30 N、动摩擦系数 0.2，打印加速度。', '''
def accel_with_friction(m, F, mu, g=9.8):
    return (F - mu * m * g) / m


print(round(accel_with_friction(5, 30, 0.2), 3))
''', '合力是拉力减动摩擦力，再除以质量得加速度。')

add('写一段 Python 代码，定义函数计算弹簧振子的周期和角频率，劲度系数 200 N/m、质量 0.5 kg，打印结果。', '''
import math


def spring_period(k, m):
    w = math.sqrt(k / m)
    return round(w, 4), round(2 * math.pi / w, 4)


print(spring_period(200, 0.5))
''', '角频率是根号下 k/m，周期等于 2π 除以角频率。')

add('写一段 Python 代码，用数值积分定义函数求大角度单摆的周期，摆长 1 米、初始摆角 90 度，打印数值周期并与小角度公式对比。', '''
import math


def pendulum_period(L, deg, g=9.8, n=20000):
    k = math.sin(math.radians(deg) / 2)
    hi = math.pi / 2
    h = hi / n
    s = 0.5 * (1.0 + 1.0 / math.sqrt(1 - k * k * math.sin(hi) ** 2))
    for i in range(1, n):
        phi = i * h
        s += 1.0 / math.sqrt(1 - k * k * math.sin(phi) ** 2)
    T = 4 * math.sqrt(L / g) * s * h
    return round(T, 4), round(2 * math.pi * math.sqrt(L / g), 4)


print(pendulum_period(1, 90))
''', '大角度单摆周期要用椭圆积分数值求解，小角度公式会低估。')

add('写一段 Python 代码，定义函数分别计算电阻串联和并联的总阻值，对 [10,20,30] 欧姆求解并打印。', '''
def series(rs):
    return sum(rs)


def parallel(rs):
    return 1 / sum(1 / r for r in rs)


print(series([10, 20, 30]), round(parallel([10, 20, 30]), 4))
''', '串联直接相加，并联是各支路倒数之和再取倒数。')

add('写一段 Python 代码，定义函数计算 RC 电路电容充电电压，R=1000 欧、C=1e-3 法、电源 5 V，打印 0、1、2、3 秒时的电压。', '''
import math


def rc_voltage(t, R=1000, C=1e-3, V0=5):
    return V0 * (1 - math.exp(-t / (R * C)))


print([round(rc_voltage(t), 4) for t in (0, 1, 2, 3)])
''', '充电按指数逼近电源电压，一个时间常数处约到 63%。')

add('写一段 Python 代码，定义函数计算 LC 谐振频率和 RLC 电路的品质因数，L=0.1 H、C=1e-6 F、R=10 欧，打印结果。', '''
import math


def resonant(L, C, R=10):
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))
    Q = math.sqrt(L / C) / R
    return round(f0, 2), round(Q, 3)


print(resonant(0.1, 1e-6))
''', '谐振频率由 L 和 C 决定，品质因数反映谐振峰的尖锐程度。')

add('写一段 Python 代码，用类定义一个栈，实现入栈、出栈和查看栈顶，演示后打印出栈顺序。', '''
class Stack:
    def __init__(self):
        self.data = []

    def push(self, x):
        self.data.append(x)

    def pop(self):
        return self.data.pop()

    def top(self):
        return self.data[-1] if self.data else None


s = Stack()
for x in (1, 2, 3):
    s.push(x)
print(s.top(), s.pop(), s.pop())
''', '栈是后进先出，用列表尾端做入栈出栈最快。')

add('写一段 Python 代码，用 collections.deque 定义函数模拟队列，依次入队 a/b/c 后出队两个并打印剩余队列。', '''
from collections import deque


def queue_demo(items):
    q = deque()
    q.extend(items)
    q.popleft()
    q.popleft()
    return list(q)


print(queue_demo(["a", "b", "c"]))
''', '队列是先进先出，deque 的 popleft 是 O(1)。')

add('写一段 Python 代码，定义单链表节点类并完成头插法建表和遍历，打印链表元素顺序。', '''
class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.nxt = nxt


def build(vals):
    head = None
    for v in reversed(vals):
        head = Node(v, head)
    return head


def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.nxt
    return out


print(to_list(build([1, 2, 3])))
''', '头插法逆序插入，遍历时顺着 nxt 指针一路走到底。')

add('写一段 Python 代码，定义二叉树节点并实现前序、中序、后序三种遍历，打印三种结果。', '''
class Tree:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder(t):
    return [] if t is None else [t.val] + preorder(t.left) + preorder(t.right)


def inorder(t):
    return [] if t is None else inorder(t.left) + [t.val] + inorder(t.right)


def postorder(t):
    return [] if t is None else postorder(t.left) + postorder(t.right) + [t.val]


root = Tree(1, Tree(2, Tree(4), Tree(5)), Tree(3))
print(preorder(root), inorder(root), postorder(root))
''', '三种遍历只是访问根节点的时机不同，递归写起来最直观。')

add('写一段 Python 代码，定义快速排序函数（原地分区），打印 [8,3,1,7,0,10,2] 的排序结果。', '''
def quicksort(a):
    if len(a) <= 1:
        return a
    pivot = a[len(a) // 2]
    left = [x for x in a if x < pivot]
    mid = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quicksort(left) + mid + quicksort(right)


print(quicksort([8, 3, 1, 7, 0, 10, 2]))
''', '选基准分成三段再递归，平均复杂度是 O(n log n)。')

add('写一段 Python 代码，定义归并排序函数并对 [5,2,9,1,5,6] 排序，打印结果。', '''
def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    return out + left[i:] + right[j:]


print(merge_sort([5, 2, 9, 1, 5, 6]))
''', '归并排序稳定，靠两个有序序列双指针合并。')

add('写一段 Python 代码，用 heapq 定义函数取列表中最大的 3 个数，对 [4,9,1,7,3,8,2] 调用并打印。', '''
import heapq


def top_k(xs, k=3):
    return heapq.nlargest(k, xs)


print(top_k([4, 9, 1, 7, 3, 8, 2]))
''', 'nlargest 内部用堆，比整体排序再切片更省。')

add('写一段 Python 代码，定义函数对图做广度优先搜索，从 A 出发遍历下面的邻接表，打印访问顺序。', '''
from collections import deque


def bfs(graph, start):
    seen = {start}
    q = deque([start])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order


g = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": ["F"], "F": []}
print(bfs(g, "A"))
''', 'BFS 用队列按层扩散，第一次到达就是无权图的最短路。')

add('写一段 Python 代码，定义函数对图做深度优先搜索（递归版），从 A 出发遍历邻接表，打印访问顺序。', '''
def dfs(graph, start, seen=None):
    if seen is None:
        seen = set()
    seen.add(start)
    order = [start]
    for v in graph.get(start, []):
        if v not in seen:
            order += dfs(graph, v, seen)
    return order


g = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": ["F"], "F": []}
print(dfs(g, "A"))
''', 'DFS 一条路走到底再回溯，用集合防止重复访问。')

add('写一段 Python 代码，用 Dijkstra 算法定义函数求带权图的最短距离，打印从 A 到各点的距离。', '''
import heapq


def dijkstra(graph, start):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


g = {"A": {"B": 4, "C": 2}, "B": {"D": 5}, "C": {"D": 1}, "D": {}}
print(dijkstra(g, "A"))
''', '每次取出当前距离最小的点松弛邻边，边权非负时结果最优。')

add('写一段 Python 代码，定义函数解 0-1 背包问题，重量 [2,3,4,5]、价值 [3,4,5,6]、容量 8，打印最大价值。', '''
def knapsack(w, v, cap):
    dp = [0] * (cap + 1)
    for wi, vi in zip(w, v):
        for c in range(cap, wi - 1, -1):
            dp[c] = max(dp[c], dp[c - wi] + vi)
    return dp[cap]


print(knapsack([2, 3, 4, 5], [3, 4, 5, 6], 8))
''', '一维 DP 倒序遍历容量，保证每个物品只用一次。')

add('写一段 Python 代码，定义函数计算两个字符串的编辑距离，打印 kitten 到 sitting 的距离。', '''
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


print(edit_distance("kitten", "sitting"))
''', '增删改三种操作取最小代价，二维 DP 递推即可。')

add('写一段 Python 代码，定义函数把字典写入 JSON 字符串再解析回来，打印解析出的类型和姓名。', '''
import json


def roundtrip(d):
    s = json.dumps(d, ensure_ascii=False)
    back = json.loads(s)
    return type(back).__name__, back["name"]


print(roundtrip({"name": "polaris", "version": 3}))
''', 'dumps 序列化、loads 反序列化，ensure_ascii=False 保留中文。')

add('写一段 Python 代码，定义函数解析一段 CSV 文本并统计每行的字段数和总行数，打印结果。', '''
import csv
import io


def csv_stat(text):
    rows = list(csv.reader(io.StringIO(text)))
    return len(rows), [len(r) for r in rows]


print(csv_stat("name,age" + chr(10) + "tom,18" + chr(10) + "amy,20"))
''', 'csv.reader 按逗号切分并正确处理引号，比手动 split 稳。')

add('写一段 Python 代码，定义函数计算两个日期之间相差的天数，打印 2026-01-01 到 2026-10-19 的天数。', '''
from datetime import date


def days_between(d1, d2):
    return (date(*map(int, d2.split("-"))) - date(*map(int, d1.split("-")))).days


print(days_between("2026-01-01", "2026-10-19"))
''', 'date 对象相减得到 timedelta，取 days 就是相差天数。')

add('写一段 Python 代码，用正则表达式定义函数从文本里提取所有邮箱和所有数字，打印提取结果。', '''
import re


def extract(text):
    return re.findall(r"[\\w.]+@[\\w.]+", text), re.findall(r"\\d+", text)


print(extract("联系 a.b@mail.com 或 root@polaris.ai，电话 12345"))
''', 'findall 一次把所有匹配抓出来，邮箱和数字各写一条模式。')

add('写一段 Python 代码，定义函数统计日志文本里各级别出现的次数，打印统计结果。', '''
from collections import Counter


def level_count(lines):
    return dict(Counter(line.split()[0] for line in lines if line.strip()))


logs = ["INFO start", "WARN slow", "INFO ok", "ERROR fail", "WARN retry"]
print(level_count(logs))
''', '取每行第一个字段当级别，Counter 直接出频次。')

add('写一段 Python 代码，定义函数做长度、质量、时间三类单位换算，打印 1 英里对应的公里和 1 磅对应的千克。', '''
def convert(value, from_u, to_u):
    table = {
        ("mile", "km"): 1.609344,
        ("km", "mile"): 1 / 1.609344,
        ("lb", "kg"): 0.45359237,
        ("kg", "lb"): 1 / 0.45359237,
        ("min", "s"): 60.0,
        ("s", "min"): 1 / 60.0,
    }
    return round(value * table[(from_u, to_u)], 6)


print(convert(1, "mile", "km"), convert(1, "lb", "kg"))
''', '换算表用元组做键，查表乘系数最不容易写错。')

add('写一段 Python 代码，定义函数计算麦克斯韦速率分布的最概然速率并用数值积分验证概率归一化，氮气 T=300 K，打印最概然速率和积分值。', '''
import math


def most_probable_speed(T, M=0.028, R=8.314):
    return math.sqrt(2 * R * T / M)


def speed_pdf(v, T=300, M=0.028, R=8.314):
    a = M / (2 * R * T)
    return 4 * math.pi * (a / math.pi) ** 1.5 * v * v * math.exp(-a * v * v)


def normalize(T=300, vmax=3000, n=20000):
    h = vmax / n
    s = 0.5 * (speed_pdf(0, T) + speed_pdf(vmax, T))
    for i in range(1, n):
        s += speed_pdf(i * h, T)
    return s * h


print(round(most_probable_speed(300), 2), round(normalize(), 6))
''', '最概然速率是根号下 2RT/M，全区间积分应接近 1。')

add('写一段 Python 代码，用四阶龙格-库塔法定义函数解牛顿冷却方程，物体初温 90 度、环境 20 度、k=0.05，打印 60 秒后的数值解并与解析解对比。', '''
import math


def rk4_cooling(T0, Tenv, k, t_end, dt=0.5):
    f = lambda T: -k * (T - Tenv)
    T, t = T0, 0.0
    while t < t_end - 1e-9:
        k1 = f(T)
        k2 = f(T + dt / 2 * k1)
        k3 = f(T + dt / 2 * k2)
        k4 = f(T + dt * k3)
        T += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return T


def analytic(T0, Tenv, k, t):
    return Tenv + (T0 - Tenv) * math.exp(-k * t)


print(round(rk4_cooling(90, 20, 0.05, 60), 4), round(analytic(90, 20, 0.05, 60), 4))
''', 'RK4 每步取四个斜率加权平均，和解析解几乎重合。')
