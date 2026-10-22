# -*- coding: utf-8 -*-
"""代码种子 第 1 批：基础语法 -> 数据结构 -> 数学统计（1-50，由易到难）。

每条 = (instruction, code, note)
硬约束：
  1. code 必须独立可执行、无外部依赖（只用标准库）、有 stdout 输出、结果确定
  2. 不得与 data/code_holdout_v1.jsonl 的 20 题同题（泄题自查交给 build 脚本）
  3. 统一风格：定义函数 -> 调用 -> print，带一两句中文注释
"""

S = []


def add(instruction, code, note):
    S.append((instruction, code.strip('\n'), note))


# ---------- L1 基础语法与算术（1-15） ----------
add('写一段 Python 代码，定义一个函数打印问候语 Hello Polaris。', '''
def greet(name):
    # 拼接问候语并返回
    return "Hello " + name

print(greet("Polaris"))
''', '用 f-string 之外的字符串拼接也能拼出问候语，函数方便复用。')

add('写一段 Python 代码，定义函数计算两个数的和并打印 3 加 5 的结果。', '''
def add(a, b):
    return a + b

print(add(3, 5))
''', '加法封装成函数，调用时传入实参即可。')

add('写一段 Python 代码，定义函数计算矩形面积，打印长 6 宽 4 的矩形面积。', '''
def rect_area(w, h):
    return w * h

print(rect_area(6, 4))
''', '面积等于长乘宽，函数把公式固定下来。')

add('写一段 Python 代码，定义函数把摄氏温度转成华氏温度，打印 100 摄氏度对应的华氏度。', '''
def c2f(c):
    # F = C * 9/5 + 32
    return c * 9 / 5 + 32

print(c2f(100))
''', '华氏度 = 摄氏度 × 9/5 + 32，水的沸点正好是 212。')

add('写一段 Python 代码，定义函数判断一个整数是奇数还是偶数，打印 17 的判断结果。', '''
def parity(n):
    return "even" if n % 2 == 0 else "odd"

print(parity(17))
''', '对 2 取余为 0 是偶数，否则是奇数。')

add('写一段 Python 代码，不用内置 abs，定义函数求一个数的绝对值，打印 -8.5 的绝对值。', '''
def my_abs(x):
    return x if x >= 0 else -x

print(my_abs(-8.5))
''', '负数取相反数、正数保持不变，就是绝对值的定义。')

add('写一段 Python 代码，定义函数返回字符串的长度和大写形式，对 polaris 调用并打印。', '''
def info(s):
    return len(s), s.upper()

n, up = info("polaris")
print(n, up)
''', 'len 取长度，upper 转大写，函数可以一次返回多个值。')

add('写一段 Python 代码，不用内置 sum，定义函数求列表所有元素之和，打印 [2,4,6,8] 的和。', '''
def total(xs):
    acc = 0
    for x in xs:
        acc += x
    return acc

print(total([2, 4, 6, 8]))
''', '累加器从 0 开始遍历累加，这是最朴素的求和。')

add('写一段 Python 代码，不用内置 max，定义函数求列表最大值，打印 [3,9,2,7] 的最大值。', '''
def my_max(xs):
    m = xs[0]
    for x in xs[1:]:
        if x > m:
            m = x
    return m

print(my_max([3, 9, 2, 7]))
''', '先假设第一个最大，后面遇到更大的就替换。')

add('写一段 Python 代码，用循环定义函数求阶乘，打印 6 的阶乘。', '''
def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

print(factorial(6))
''', '阶乘从 1 连乘到 n，6! 等于 720。')

add('写一段 Python 代码，定义函数判断一个年份是不是闰年，打印 2024 和 2100 的判断结果。', '''
def is_leap(y):
    # 能被 400 整除，或能被 4 整除但不能被 100 整除
    return y % 400 == 0 or (y % 4 == 0 and y % 100 != 0)

print(is_leap(2024), is_leap(2100))
''', '整百年必须能被 400 整除才是闰年，所以 2100 不是。')

add('写一段 Python 代码，定义函数把整数 12345 的各位反转，打印结果。', '''
def reverse_int(n):
    r = 0
    while n > 0:
        r = r * 10 + n % 10
        n //= 10
    return r

print(reverse_int(12345))
''', '每次取个位拼到结果末尾，同时把原数整除 10 去掉个位。')

add('写一段 Python 代码，定义函数统计一句话里的单词个数，对 "to be or not to be" 调用并打印。', '''
def word_count(s):
    return len(s.split())

print(word_count("to be or not to be"))
''', 'split 默认按空白切分，切出来的列表长度就是单词数。')

add('写一段 Python 代码，定义函数把华氏温度转成摄氏温度，打印 98.6 华氏度对应的摄氏度。', '''
def f2c(f):
    return (f - 32) * 5 / 9

print(round(f2c(98.6), 2))
''', '摄氏度 = (华氏度 - 32) × 5/9，人体体温约 37 度。')

add('写一段 Python 代码，定义函数计算圆的面积和周长，打印半径为 3 的结果。', '''
import math


def circle(r):
    return math.pi * r * r, 2 * math.pi * r


area, peri = circle(3)
print(round(area, 3), round(peri, 3))
''', '面积 πr²、周长 2πr，用 math.pi 取圆周率。')

# ---------- L2 数据结构与控制流（16-35） ----------
add('写一段 Python 代码，用冒泡排序定义函数给列表升序排序，打印 [5,2,9,1,6] 的排序结果。', '''
def bubble(xs):
    a = list(xs)
    for i in range(len(a)):
        for j in range(len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

print(bubble([5, 2, 9, 1, 6]))
''', '相邻元素两两比较交换，每轮把最大的顶到末尾。')

add('写一段 Python 代码，定义二分查找函数，在有序列表 [1,3,5,7,9,11] 里找 7，打印其下标。', '''
def bisect_index(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

print(bisect_index([1, 3, 5, 7, 9, 11], 7))
''', '每次把区间对半砍，找得到返回下标，找不到返回 -1。')

add('写一段 Python 代码，定义函数给列表去重并保持原有顺序，打印 [3,1,3,2,1,4] 的结果。', '''
def dedup(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

print(dedup([3, 1, 3, 2, 1, 4]))
''', '用集合记录见过的元素，顺序靠列表保持。')

add('写一段 Python 代码，定义函数统计一段文本里每个单词出现的次数，打印 "a b a c b a" 的结果。', '''
def word_freq(s):
    d = {}
    for w in s.split():
        d[w] = d.get(w, 0) + 1
    return d

print(word_freq("a b a c b a"))
''', '字典 get 带默认值累加，是计数最常用的写法。')

add('写一段 Python 代码，用带记忆化的递归定义函数求斐波那契第 20 项，打印结果。', '''
def fib(n, memo={0: 0, 1: 1}):
    if n not in memo:
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]

print(fib(20))
''', '记忆化把重复子问题缓存下来，递归就不会指数级爆炸。')

add('写一段 Python 代码，定义函数判断字符串是不是回文（忽略大小写和空格），打印 "Never odd or even" 的判断结果。', '''
def is_palindrome(s):
    t = "".join(ch.lower() for ch in s if ch.isalnum())
    return t == t[::-1]

print(is_palindrome("Never odd or even"))
''', '先清洗成只留字母数字并小写，再和反转结果比较。')

add('写一段 Python 代码，用埃氏筛定义函数求 30 以内的所有素数，打印结果。', '''
def sieve(n):
    flag = [True] * (n + 1)
    flag[0] = flag[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if flag[i]:
            for j in range(i * i, n + 1, i):
                flag[j] = False
    return [i for i in range(n + 1) if flag[i]]

print(sieve(30))
''', '把每个素数的倍数划掉，剩下的就是素数。')

add('写一段 Python 代码，用欧几里得算法定义函数求最大公约数，打印 48 和 18 的结果。', '''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

print(gcd(48, 18))
''', '反复取余直到余数为 0，最后的除数就是最大公约数。')

add('写一段 Python 代码，定义函数求两个数的最小公倍数，打印 12 和 15 的结果。', '''
def lcm(a, b):
    def g(x, y):
        while y:
            x, y = y, x % y
        return x
    return a * b // g(a, b)

print(lcm(12, 15))
''', '最小公倍数等于两数之积除以最大公约数。')

add('写一段 Python 代码，定义函数做矩阵转置，打印 [[1,2,3],[4,5,6]] 的转置结果。', '''
def transpose(m):
    return [list(row) for row in zip(*m)]

print(transpose([[1, 2, 3], [4, 5, 6]]))
''', 'zip 加星号把每行拆开重新按列组合，就是转置。')

add('写一段 Python 代码，定义递归函数把嵌套列表扁平化，打印 [1,[2,[3,4]],5] 的结果。', '''
def flatten(xs):
    out = []
    for x in xs:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out

print(flatten([1, [2, [3, 4]], 5]))
''', '遇到列表就递归展开，遇到普通元素就收进结果。')

add('写一段 Python 代码，定义函数统计一段多行文本的行数、单词数和字符数，打印结果。', '''
def wc(text):
    lines = text.strip().split(chr(10))
    return len(lines), sum(len(l.split()) for l in lines), len(text)

print(wc("one two" + chr(10) + "three four five" + chr(10) + "six"))
''', '按换行切分得行数，再对每行 split 得单词数。')

add('写一段 Python 代码，定义函数把字典按值从大到小排序，打印 {"a":3,"b":9,"c":1} 的结果。', '''
def sort_by_value(d):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)

print(sort_by_value({"a": 3, "b": 9, "c": 1}))
''', 'sorted 的 key 指定取元组第二项，reverse 控制降序。')

add('写一段 Python 代码，定义函数把字符串按字符出现频率从高到低排序输出，对 "mississippi" 调用并打印。', '''
from collections import Counter


def freq_order(s):
    return [ch for ch, _ in Counter(s).most_common()]

print(freq_order("mississippi"))
''', 'Counter 统计完用 most_common 直接按频次降序取出。')

add('写一段 Python 代码，定义函数把罗马数字转成整数，打印 XIV 和 MCMXCIV 的结果。', '''
def roman2int(s):
    val = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = val[ch]
        if i + 1 < len(s) and v < val[s[i + 1]]:
            total -= v
        else:
            total += v
    return total

print(roman2int("XIV"), roman2int("MCMXCIV"))
''', '小数字在大数字左边要减，其余相加。')

add('写一段 Python 代码，定义函数检查括号字符串是否配对，打印 "([]{})" 和 "([)]" 的检查结果。', '''
def balanced(s):
    pair = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
    return not stack

print(balanced("([]{})"), balanced("([)]"))
''', '左括号入栈，右括号必须和栈顶配对，最后栈要空。')

add('写一段 Python 代码，定义函数实现凯撒密码加密，把 "hello" 偏移 3 位后打印。', '''
def caesar(s, k):
    out = ""
    for ch in s:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            out += chr((ord(ch) - base + k) % 26 + base)
        else:
            out += ch
    return out

print(caesar("hello", 3))
''', '字母按偏移量循环移动，非字母原样保留。')

add('写一段 Python 代码，定义函数把列表按指定大小分块，打印 [1,2,3,4,5,6,7] 按 3 分块的结果。', '''
def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]

print(chunks([1, 2, 3, 4, 5, 6, 7], 3))
''', '利用切片步长 n 一次取一段，最后一段不满也照收。')

add('写一段 Python 代码，定义函数把两个列表合并成字典，打印 ["a","b"] 和 [1,2] 的合并结果。', '''
def to_dict(keys, values):
    return dict(zip(keys, values))

print(to_dict(["a", "b"], [1, 2]))
''', 'zip 把两个列表逐位配成对，dict 直接转成字典。')

add('写一段 Python 代码，定义函数计算列表的平均数和样本方差，打印 [2,4,4,4,5,5,7,9] 的结果。', '''
def mean_var(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, round(var, 4)

print(mean_var([2, 4, 4, 4, 5, 5, 7, 9]))
''', '样本方差除以 n-1 才是无偏估计。')

# ---------- L3 数学与数值方法（36-50） ----------
add('写一段 Python 代码，用梯形法定义函数做数值积分，计算 sin(x) 在 0 到 pi 上的积分并打印。', '''
import math


def trapz(f, a, b, n=1000):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


print(round(trapz(math.sin, 0, math.pi), 6))
''', '把区间切成小梯形求和，sin 在 0 到 π 的积分正好约等于 2。')

add('写一段 Python 代码，用牛顿迭代法定义函数求平方根，打印 2 的平方根（保留 6 位）。', '''
def newton_sqrt(x, eps=1e-10):
    g = x
    while abs(g * g - x) > eps:
        g = (g + x / g) / 2
    return g

print(round(newton_sqrt(2), 6))
''', '牛顿法每次用切线逼近零点，迭代几步就收敛。')

add('写一段 Python 代码，定义函数用最小二乘法做一元线性回归，打印拟合出的斜率和截距。', '''
def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    k = num / den
    return round(k, 4), round(my - k * mx, 4)

print(linfit([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 8.1, 9.8]))
''', '斜率由协方差除以方差得到，截距用过均值点反推。')

add('写一段 Python 代码，定义函数计算总体标准差，打印 [10,12,23,23,16,23,21,16] 的结果（保留 4 位）。', '''
import math


def pstdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


print(round(pstdev([10, 12, 23, 23, 16, 23, 21, 16]), 4))
''', '总体标准差除以 n，样本标准差才除以 n-1。')

add('写一段 Python 代码，定义函数计算组合数 C(n,k)，打印 C(10,3) 的结果。', '''
import math


def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


print(comb(10, 3))
''', '组合数等于 n! 除以 k! 与 (n-k)! 的乘积。')

add('写一段 Python 代码，定义快速幂函数（二分 exponentiation），打印 3 的 40 次方。', '''
def fast_pow(a, n):
    r = 1
    while n:
        if n & 1:
            r *= a
        a *= a
        n >>= 1
    return r

print(fast_pow(3, 40))
''', '把指数按二进制拆开，复杂度从 O(n) 降到 O(log n)。')

add('写一段 Python 代码，用复数定义函数计算 RL 串联电路的阻抗模和相位角，R=30、L=0.1、f=50，打印结果。', '''
import math


def rl_impedance(R, L, f):
    w = 2 * math.pi * f
    z = complex(R, w * L)
    return round(abs(z), 4), round(math.degrees(math.atan2(z.imag, z.real)), 4)


print(rl_impedance(30, 0.1, 50))
''', '感抗是 ωL，阻抗是 R + jωL 的复数，模和辐角都要算。')

add('写一段 Python 代码，定义函数求一元二次方程 ax^2+bx+c=0 的实根，打印 x^2-5x+6=0 的两个根。', '''
import math


def quad(a, b, c):
    d = b * b - 4 * a * c
    if d < 0:
        return None
    r = math.sqrt(d)
    return (-b - r) / (2 * a), (-b + r) / (2 * a)


print(quad(1, -5, 6))
''', '判别式小于 0 无实根，否则用求根公式算出两根。')

add('写一段 Python 代码，定义函数用克拉默法则解二元一次方程组，打印 2x+3y=8 与 x-y=1 的解。', '''
def solve2(a1, b1, c1, a2, b2, c2):
    d = a1 * b2 - a2 * b1
    x = (c1 * b2 - c2 * b1) / d
    y = (a1 * c2 - a2 * c1) / d
    return round(x, 4), round(y, 4)

print(solve2(2, 3, 8, 1, -1, 1))
''', '系数行列式不为 0 时有唯一解，两个行列式分别替换列得到 x 和 y。')

add('写一段 Python 代码，用蒙特卡洛方法定义函数估算圆周率，固定随机种子，打印估算值。', '''
import random


def mc_pi(n=200000, seed=42):
    rnd = random.Random(seed)
    inside = 0
    for _ in range(n):
        x, y = rnd.random(), rnd.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / n


print(round(mc_pi(), 4))
''', '四分之一圆内的点数占比乘以 4 就是 π 的估计。')

add('写一段 Python 代码，定义函数用泰勒级数计算自然常数 e，取前 20 项并打印。', '''
import math


def euler_e(terms=20):
    s = 0.0
    for k in range(terms):
        s += 1.0 / math.factorial(k)
    return s


print(round(euler_e(), 10))
''', 'e 是 1/k! 的无穷求和，取前 20 项精度已经很高。')

add('写一段 Python 代码，定义函数用 Kadane 算法求最大连续子数组和，打印 [-2,1,-3,4,-1,2,1,-5,4] 的结果。', '''
def kadane(xs):
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

print(kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
''', '当前和与自身加当前值取大者，全局最优同步更新。')

add('写一段 Python 代码，定义函数计算皮尔逊相关系数，打印两组数据的相关系数（保留 4 位）。', '''
def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 4)

print(pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5]))
''', '相关系数是协方差除以两个标准差的乘积。')

add('写一段 Python 代码，定义函数计算窗口为 3 的移动平均，打印 [1,2,3,4,5,6] 的结果。', '''
def moving_avg(xs, w=3):
    out = []
    for i in range(len(xs) - w + 1):
        out.append(round(sum(xs[i:i + w]) / w, 3))
    return out

print(moving_avg([1, 2, 3, 4, 5, 6]))
''', '滑动窗口逐段取平均，可以抹平短期波动。')

add('写一段 Python 代码，定义函数计算正态分布的概率密度并用梯形法求 -1 到 1 的概率，打印结果。', '''
import math


def norm_pdf(x, mu=0, sigma=1):
    return math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi))


def prob(a, b, n=2000):
    h = (b - a) / n
    s = 0.5 * (norm_pdf(a) + norm_pdf(b))
    for i in range(1, n):
        s += norm_pdf(a + i * h)
    return s * h


print(round(prob(-1, 1), 4))
''', '标准正态在正负一个标准差内的概率约 68.27%。')
