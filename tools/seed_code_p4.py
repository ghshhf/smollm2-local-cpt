# -*- coding: utf-8 -*-
"""代码种子 第 4 批（151-200）：热力学进阶 -> 工程物理 -> 概率统计 -> 文件与文本。

每条 = (instruction, code, note)
硬约束同 p1-p3：独立可执行、只用标准库、有 stdout、结果确定、不与 holdout 同题。
"""

S = []


def add(instruction, code, note):
    S.append((instruction, code.strip('\n'), note))


# ---------- 热力学进阶（151-170） ----------
add('写一段 Python 代码，计算 1000 瓦热量从 500K 传到 300K 时的熵产。', '''
def entropy_gen(q, t_hot, t_cold):
    # 热流 q 由高温流向低温，熵产 = q*(1/Tc - 1/Th)
    return q * (1 / t_cold - 1 / t_hot)

print(round(entropy_gen(1000, 500, 300), 4))
''', '熵产等于热流乘以两温度倒数的差，恒为正。')

add('写一段 Python 代码，计算压比 10 的布雷顿循环热效率。', '''
def brayton_eff(pr, gamma=1.4):
    return 1 - 1 / pr ** ((gamma - 1) / gamma)

print(round(brayton_eff(10), 4))
''', '布雷顿循环效率只取决于压比和工质比热比。')

add('写一段 Python 代码，计算压缩比 8 的奥托循环热效率。', '''
def otto_eff(cr, gamma=1.4):
    return 1 - 1 / cr ** (gamma - 1)

print(round(otto_eff(8), 4))
''', '奥托循环是定容加热，效率随压缩比升高。')

add('写一段 Python 代码，计算压缩比 16、截断比 2 的狄塞尔循环热效率。', '''
def diesel_eff(cr, cut, gamma=1.4):
    num = cut ** gamma - 1
    den = gamma * (cut - 1) * cr ** (gamma - 1)
    return 1 - num / den

print(round(diesel_eff(16, 2), 4))
''', '狄塞尔循环是定压加热，截断比越大效率越低。')

add('写一段 Python 代码，用四个焓值计算朗肯循环的热效率。', '''
def rankine(h1, h2, h3, h4):
    # h1 泵入口, h2 泵出口, h3 透平入口, h4 透平出口
    w_turb = h3 - h4
    w_pump = h2 - h1
    q_in = h3 - h2
    return round((w_turb - w_pump) / q_in, 4)

print(rankine(191, 201, 3000, 2200))
''', '净功等于透平功减泵功，再除以吸热量。')

add('写一段 Python 代码，用对数平均温差法计算换热器的平均温差。', '''
import math

def lmtd(t1_in, t1_out, t2_in, t2_out):
    d1 = t1_in - t2_out
    d2 = t1_out - t2_in
    if abs(d1 - d2) < 1e-9:
        return d1
    return (d1 - d2) / math.log(d1 / d2)

print(round(lmtd(150, 90, 20, 60), 3))
''', '两端温差取对数平均，逆流换热更高效。')

add('写一段 Python 代码，用集总热容法估算物体从 90 度冷到 40 度所需时间。', '''
import math

def cooling_time(t0, t_env, t_end, tau):
    return round(tau * math.log((t0 - t_env) / (t_end - t_env)), 1)

print(cooling_time(90, 25, 40, 600))
''', '牛顿冷却按指数衰减，时间常数决定快慢。')

add('写一段 Python 代码，计算对流换热系数 50、特征长度 0.02、导热系数 0.6 时的毕渥数。', '''
def biot(h, lc, k):
    return round(h * lc / k, 4)

print(biot(50, 0.02, 0.6))
''', '毕渥数小于 0.1 时集总热容法才适用。')

add('写一段 Python 代码，用 Dittus-Boelter 公式算雷诺数 20000、普朗特数 0.7 的努塞尔数。', '''
def nusselt(re, pr, heating=True):
    n = 0.4 if heating else 0.3
    return round(0.023 * re ** 0.8 * pr ** n, 2)

print(nusselt(20000, 0.7))
''', '湍流强制对流经验式，加热取 0.4 次方。')

add('写一段 Python 代码，计算两块平行平板之间的辐射换热热流。', '''
SIGMA = 5.67e-8

def parallel_plates(t1, t2, eps):
    return round(SIGMA * (t1 ** 4 - t2 ** 4) / (2 / eps - 1), 2)

print(parallel_plates(500, 300, 0.8))
''', '两平板多次反射，等效发射率按 2/eps-1 折算。')

add('写一段 Python 代码，计算两摩尔气体等温等压混合的熵变。', '''
import math

def mix_entropy(n1, n2, r=8.314):
    n = n1 + n2
    return round(-r * (n1 * math.log(n1 / n) + n2 * math.log(n2 / n)), 3)

print(mix_entropy(1, 1))
''', '混合熵只与摩尔分数有关，与气体种类无关。')

add('写一段 Python 代码，用范德瓦尔斯方程计算真实气体的压缩因子。', '''
def z_factor(p, v, r, t, a, b):
    return round((p + a / v ** 2) * (v - b) / (r * t), 4)

print(z_factor(1e6, 0.0025, 8.314, 300, 0.364, 4.27e-5))
''', '压缩因子等于实际摩尔体积与理想值之比，1 附近接近理想气体。')

add('写一段 Python 代码，由节流后的焓和饱和参数计算蒸汽干度。', '''
def throttle_quality(h, hf, hg):
    return round((h - hf) / (hg - hf), 4)

print(throttle_quality(1200, 400, 2600))
''', '干度是汽相质量占比，节流前后焓相等。')

add('写一段 Python 代码，计算导热系数 0.6、密度 1000、比热 4180 的材料的导温系数。', '''
def alpha(k, rho, cp):
    return k / (rho * cp)

print("%.2e" % alpha(0.6, 1000, 4180))
''', '导温系数等于导热系数除以密度与比热之积。')

add('写一段 Python 代码，计算三层平壁在总温差 80 度下的热流密度。', '''
def wall_flux(dt, layers):
    r = sum(t / k for t, k in layers)
    return round(dt / r, 2)

print(wall_flux(80, [(0.2, 0.8), (0.1, 0.04), (0.15, 0.6)]))
''', '串联热阻直接相加，热流等于温差除以总热阻。')

add('写一段 Python 代码，计算冷源 273K、热源 300K 时卡诺制冷机的制冷系数。', '''
def carnot_cop_cool(t_cold, t_hot):
    return round(t_cold / (t_hot - t_cold), 3)

print(carnot_cop_cool(273, 300))
''', '制冷系数等于冷源温度除以温差，是可逆上限。')

add('写一段 Python 代码，按卡诺效率的一半估算热泵的制热系数。', '''
def cop_heat(t_cold, t_hot, eta=0.5):
    carnot = t_hot / (t_hot - t_cold)
    return round(carnot * eta, 3)

print(cop_heat(273, 300))
''', '实际热泵达不到卡诺，用效率系数折算。')

add('写一段 Python 代码，用 Magnus 公式算 30 度、露点 20 度时的相对湿度。', '''
import math

def rh(t, dew):
    def es(x):
        return 611.2 * math.exp(17.67 * x / (x + 243.5))
    return round(es(dew) / es(t) * 100, 1)

print(rh(30, 20))
''', '相对湿度是实际水汽压与饱和水汽压之比。')

add('写一段 Python 代码，用 Dulong 公式估算煤的高位发热量。', '''
def hhv_estimate(c, h, o):
    # 质量分数口径，结果单位 MJ/kg
    return round(33.8 * c + 144.2 * (h - o / 8), 2)

print(hhv_estimate(0.52, 0.13, 0.35))
''', '碳氢氧三元素经验式，氧已结合部分氢要扣掉。')

add('写一段 Python 代码，用显式差分算非稳态导热节点一步后的温度。', '''
def step(t, t_left, t_right, dx, alpha, dt):
    return round(t + alpha * dt / dx ** 2 * (t_left - 2 * t + t_right), 3)

print(step(100, 150, 0, 0.01, 2.5e-6, 20))
''', '显式格式稳定要求时间步足够小，否则会震荡。')

# ---------- 工程物理（171-184） ----------
add('写一段 Python 代码，由主动轮转速和齿数计算从动轮转速。', '''
def gear_out(rpm_in, z_in, z_out):
    return round(rpm_in * z_in / z_out, 1)

print(gear_out(1500, 15, 45))
''', '齿数比与转速成反比，减速同时放大扭矩。')

add('写一段 Python 代码，计算简支梁中点受集中力时的最大挠度。', '''
def beam_deflection(p, l, e, i):
    return round(p * l ** 3 / (48 * e * i), 6)

print(beam_deflection(1000, 2, 2e11, 8e-6))
''', '挠度与力乘跨度三次方成正比，与刚度成反比。')

add('写一段 Python 代码，由管径和流速计算管道体积流量。', '''
import math

def flow(d, v):
    return round(math.pi * (d / 2) ** 2 * v, 4)

print(flow(0.1, 2))
''', '流量等于截面积乘流速。')

add('写一段 Python 代码，计算流量 0.02、扬程 20 米的水泵轴功率。', '''
def pump_power(q, head, rho=1000, g=9.81, eta=0.7):
    return round(rho * g * q * head / eta, 1)

print(pump_power(0.02, 20))
''', '轴功率等于水力功率除以泵效率。')

add('写一段 Python 代码，由功率和转速计算电机输出扭矩。', '''
def torque(power_kw, rpm):
    return round(9550 * power_kw / rpm, 2)

print(torque(15, 1450))
''', '工程上扭矩等于 9550 乘千瓦除以转速。')

add('写一段 Python 代码，计算电池恒流放电时的端电压。', '''
def terminal(emf, current, r_int):
    return round(emf - current * r_int, 3)

print(terminal(12.6, 10, 0.05))
''', '端电压等于电动势减去内阻压降。')

add('写一段 Python 代码，由面积、辐照度和效率计算光伏板输出功率。', '''
def pv_power(area, irradiance, eff):
    return round(area * irradiance * eff, 1)

print(pv_power(20, 1000, 0.2))
''', '输出功率等于面积乘辐照度再乘组件效率。')

add('写一段 Python 代码，计算叶轮面积 50 平方米、风速 10 米每秒的风机功率。', '''
def wind_power(area, speed, rho=1.225, cp=0.45):
    return round(0.5 * rho * area * speed ** 3 * cp, 1)

print(wind_power(50, 10))
''', '风功率与风速三次方成正比，贝茨极限约 0.593。')

add('写一段 Python 代码，计算流量 30、水头 40 米的水轮机出力。', '''
def hydro_power(q, head, eta=0.9, rho=1000, g=9.81):
    return round(rho * g * q * head * eta / 1000, 1)

print(hydro_power(30, 40))
''', '出力等于密度乘重力乘流量乘水头再乘效率。')

add('写一段 Python 代码，由一次侧电压和匝数计算变压器二次侧电压。', '''
def secondary(v1, n1, n2):
    return round(v1 * n2 / n1, 2)

print(secondary(220, 1100, 200))
''', '电压比等于匝数比，降压时二次匝数更少。')

add('写一段 Python 代码，计算光功率从 1 毫瓦衰减到 0.5 毫瓦的损耗分贝数。', '''
import math

def loss_db(p_in, p_out):
    return round(10 * math.log10(p_in / p_out), 2)

print(loss_db(1.0, 0.5))
''', '功率减半约等于 3 分贝损耗。')

add('写一段 Python 代码，由声压 0.2 帕计算声压级分贝数。', '''
import math

def spl(p, p0=2e-5):
    return round(20 * math.log10(p / p0), 1)

print(spl(0.2))
''', '声压级用 20 微帕作基准，每 20 分贝声压涨十倍。')

add('写一段 Python 代码，计算声源以 30 米每秒接近时 500 赫兹的观测频率。', '''
def doppler(f, v, vs):
    return round(f * 340 / (340 - vs), 1)

print(doppler(500, 340, 30))
''', '声源靠近时波长被压缩，观测频率升高。')

add('写一段 Python 代码，计算水在半径 0.5 毫米毛细管中的上升高度。', '''
import math

def capillary(sigma, theta_deg, rho, r, g=9.81):
    return round(2 * sigma * math.cos(math.radians(theta_deg)) / (rho * g * r), 4)

print(capillary(0.072, 0, 1000, 0.0005))
''', '表面张力拉液体上升，管越细升得越高。')

# ---------- 概率统计（185-192） ----------
add('写一段 Python 代码，计算 10 次试验成功 3 次、单次概率 0.5 的二项分布概率。', '''
import math

def binom(n, k, p):
    return round(math.comb(n, k) * p ** k * (1 - p) ** (n - k), 4)

print(binom(10, 3, 0.5))
''', '组合数乘成功与失败概率的幂。')

add('写一段 Python 代码，计算泊松分布中平均 2 次时恰好发生 3 次的概率。', '''
import math

def poisson(k, lam):
    return round(math.exp(-lam) * lam ** k / math.factorial(k), 4)

print(poisson(3, 2))
''', '泊松分布描述单位时间内稀有事件的发生次数。')

add('写一段 Python 代码，计算标准正态分布在 0 处的概率密度。', '''
import math

def normal_pdf(x, mu, sigma):
    return round(math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi)), 4)

print(normal_pdf(0, 0, 1))
''', '标准正态在均值处的密度约 0.3989。')

add('写一段 Python 代码，由取值和概率计算离散随机变量的期望与方差。', '''
def mean_var(vals, probs):
    mean = sum(v * p for v, p in zip(vals, probs))
    var = sum(p * (v - mean) ** 2 for v, p in zip(vals, probs))
    return round(mean, 3), round(var, 3)

print(mean_var([0, 1, 2], [0.2, 0.5, 0.3]))
''', '期望按概率加权，方差是离差平方的加权和。')

add('写一段 Python 代码，计算两组样本的协方差。', '''
def cov(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n, 3)

print(cov([1, 2, 3], [2, 4, 7]))
''', '协方差反映两组数据同向变化的程度。')

add('写一段 Python 代码，用贝叶斯公式算患病概率 1%、灵敏度 99%、误报 5% 时的阳性后验。', '''
def bayes(prior, sens, fpr):
    p_pos = sens * prior + fpr * (1 - prior)
    return round(sens * prior / p_pos, 4)

print(bayes(0.01, 0.99, 0.05))
''', '先验低时即使检测很准，阳性后验也不高。')

add('写一段 Python 代码，用蒙特卡洛法撒两万个点估算圆周率。', '''
import random

def mc_pi(n, seed=7):
    rnd = random.Random(seed)
    inside = 0
    for _ in range(n):
        x, y = rnd.random(), rnd.random()
        if x * x + y * y <= 1:
            inside += 1
    return round(4 * inside / n, 3)

print(mc_pi(20000))
''', '四分之一圆面积比上正方形面积等于四分之派。')

add('写一段 Python 代码，模拟一百步随机游走并打印终点位置。', '''
import random

def walk(steps, seed=3):
    rnd = random.Random(seed)
    pos = 0
    for _ in range(steps):
        pos += rnd.choice([-1, 1])
    return pos

print(walk(100))
''', '每步等概率加一或减一，固定随机种子结果可复现。')

# ---------- 文件与文本（193-200） ----------
add('写一段 Python 代码，解析一段订单 JSON 并算出总金额。', '''
import json

def parse_order(text):
    data = json.loads(text)
    total = sum(item["price"] * item["qty"] for item in data["items"])
    return data["order_id"], round(total, 2)

print(parse_order('{"order_id": "A7", "items": [{"price": 12.5, "qty": 2}, {"price": 3, "qty": 4}]}'))
''', 'json.loads 把文本转字典，再按条目累加金额。')

add('写一段 Python 代码，解析一段 CSV 文本并统计金额合计与行数。', '''
import csv
import io

def csv_sum(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    return sum(int(r["amount"]) for r in rows), len(rows)

print(csv_sum("name,amount\\n甲,10\\n乙,25\\n丙,5\\n"))
''', 'DictReader 按表头取列，内存里用 StringIO 当文件。')

add('写一段 Python 代码，从日志列表里筛出所有 ERROR 级别的行。', '''
def filter_level(lines, level="ERROR"):
    return [ln for ln in lines if ("[%s]" % level) in ln]

logs = ["[INFO] start", "[ERROR] disk full", "[WARN] slow", "[ERROR] timeout"]
print(filter_level(logs))
''', '按级别标签过滤，日志分析最常用的一步。')

add('写一段 Python 代码，计算两个日期之间相差多少天。', '''
from datetime import date

def days_between(a, b):
    y1, m1, d1 = a
    y2, m2, d2 = b
    return (date(y2, m2, d2) - date(y1, m1, d1)).days

print(days_between((2026, 1, 1), (2026, 10, 19)))
''', '两个日期相减得到时间差对象，取 days 即可。')

add('写一段 Python 代码，把 Unix 时间戳转成东八区的日期时间字符串。', '''
from datetime import datetime, timezone, timedelta

def ts_to_str(ts, offset=8):
    tz = timezone(timedelta(hours=offset))
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M")

print(ts_to_str(1700000000))
''', '先带时区再格式化，避免机器本地时区干扰。')

add('写一段 Python 代码，用正则从一句话里提取所有数字和邮箱地址。', '''
import re

def extract(text):
    nums = re.findall(r"\\d+", text)
    mails = re.findall(r"[\\w.+-]+@[\\w-]+\\.[\\w.]+", text)
    return nums, mails

print(extract("订单 42 金额 1280 元，联系 a.b@mail.com 或 ops@site.cn"))
''', 'findall 返回所有匹配串，数字和邮箱各用一条模式。')

add('写一段 Python 代码，把一段文字写进临时文件并统计字节数与行数。', '''
import os
import tempfile

def count_bytes(text):
    fd, path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        size = os.path.getsize(path)
        with open(path, encoding="utf-8") as f:
            lines = len(f.readlines())
        return size, lines
    finally:
        os.unlink(path)

print(count_bytes("北极星\\n主人\\n"))
''', '中文按 UTF-8 每字三字节，换行符也算一个字节。')

add('写一段 Python 代码，用 argparse 解析训练脚本的三个命令行参数。', '''
import argparse

def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tag", default="run")
    return p.parse_args(argv)

a = parse(["--epochs", "2", "--lr", "0.0005"])
print(a.epochs, a.lr, a.tag)
''', 'argparse 自动处理类型转换和默认值，脚本标配。')
