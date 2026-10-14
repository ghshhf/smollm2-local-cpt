# -*- coding: utf-8 -*-
"""
逐 0.1B 阶梯定量核算
====================
输入(已实测/已提供):
  t_ref  = 214M 模型在 64 核 CPU 上的单步耗时 18.7~29.1 s/step
输出:
  (1) 压缩口径下的最优容量点 N*(D, bits/param)
  (2) 逐级阶梯的 delta_N / 所需 delta_D / 验收
  (3) 单步耗时与单遍墙钟, CPU->GPU 拐点等值线
所有推导均为解析式, 常数来自 Hoffmann et al. 2022 (arXiv:2203.15556) Eq.(10).
"""
import numpy as np, json, math

LN2 = math.log(2.0)
E_, A_, ALPHA, B_, BETA = 1.69, 406.4, 0.34, 410.7, 0.28   # Hoffmann 2022 Eq.(10), nats/token

def loss_nd(N, D):
    return E_ + A_/N**ALPHA + B_/D**BETA

# ---- (1) 压缩口径最优容量点 -------------------------------------------------
# 两支码长: 模型体积(b*N 比特) + 数据编码((D/ln2)*L 纳特->比特)
# 最小化 f(N) = b*N + (D/ln2)*L(N,D)   =>   N* = [ D*A*alpha/(b*ln2) ]^(1/(1+alpha))
def N_star(D, b):
    return (D*A_*ALPHA/(b*LN2))**(1.0/(1.0+ALPHA))

def D_needed(N, b):          # 反解: 要 N 成为最优点所需的数据量
    return (N**(1.0+ALPHA))*b*LN2/(A_*ALPHA)

def adj_rate(N, D, b, bits_per_raw_token=32.0):
    """adjusted rate = (模型比特 + 数据编码比特) / 原始数据比特"""
    model_bits  = b*N
    data_bits   = (D/LN2)*loss_nd(N, D)
    raw_bits    = D*bits_per_raw_token
    return (model_bits+data_bits)/raw_bits, loss_nd(N,D)

# ---- (2) 三值记账下的有效 bits/param (含 fp16 embedding) ---------------------
def b_eff(d, L, V=49152, b_w=2.0, b_emb=16.0):
    """Llama 结构: 非嵌入参数 ~ 12*L*d^2 ; 共享 embedding 参数量 V*d"""
    n_nonemb = 12.0*L*d*d
    n_emb    = float(V)*d
    n_tot    = n_nonemb + n_emb
    return (b_w*n_nonemb + b_emb*n_emb)/n_tot, n_tot, n_nonemb, n_emb

# ---- (3) 墙钟 ---------------------------------------------------------------
T_REF_LO, T_REF_HI, N_REF = 18.7, 29.1, 2.14e8
def t_step(N, expo, t_ref=23.9):
    return t_ref*(N/N_REF)**expo

D_CORPUS = 3.5e9    # 用户已提供: 35 亿 token (现有 12 亿 + 需补 23 亿)

out = {}
print("="*78)
print("A. 压缩口径最优容量点 N*  (D = %.1e token)" % D_CORPUS)
print("="*78)
print(f"{'记账口径':<34}{'bits/param':>11}{'N*(D=3.5e9)':>15}")
for name,b in [("fp32 全精度",32.0),("fp16 全精度",16.0),
               ("三值名义 1.58bit",1.58),("三值取整 2bit",2.0),
               ("三值权重+fp16 embedding (d=1024,L=12)",None)]:
    if b is None:
        b,v = b_eff(1024,12)[0], None
    else:
        v = None
    ns = N_star(D_CORPUS, b)
    print(f"{name:<34}{b:>11.3f}{ns/1e6:>13.1f} M")
    out.setdefault('Nstar',{})[name] = {'bits_per_param':round(b,3),'N_star':ns}

print()
print("="*78)
print("B. 各级别'成为最优'所需的数据量  D_needed(N, b=2.0)   [N* ∝ D^%.3f]" % (1/(1+ALPHA)))
print("="*78)
levels = [0.1,0.135,0.2,0.3,0.4,0.5,0.6,0.75,1.0,1.5,2.0,3.0,5.0,8.0]
print(f"{'N':>6}{'D_needed(B token)':>19}{'相对现有 3.5B':>14}")
for l in levels:
    dn = D_needed(l*1e9, 2.0)
    print(f"{l:>6.3f}{dn/1e9:>19.1f}{dn/D_CORPUS:>13.1f}x")

print()
print("="*78)
print("C. 逐 0.1B 阶梯: delta_N / delta_D / 单步耗时 / 单遍墙钟")
print("  t_step(N)=24.0*(N/214M)^x ; tokens/step = 32768 (缺配置, 敏感性见 D 节)")
print("="*78)
B_TOK = 32768
hdr = f"{'N(B)':>7}{'ΔN(M)':>8}{'ΔD(B tok)':>11}{'t_step(s) x=1':>15}{'x=0.7':>9}{'单遍天数 D=3.5B':>17}{'M-DRIFT 门':>11}"
print(hdr); print("-"*len(hdr))
rows=[]
prev_N, prev_D = None, 0.0
for l in levels:
    N = l*1e9
    Dn = D_needed(N, 2.0)
    dN = 0.0 if prev_N is None else (N-prev_N)/1e6
    dD = Dn-prev_D
    t1 = t_step(N,1.0); t07 = t_step(N,0.7)
    days = (D_CORPUS/B_TOK)*t1/86400.0
    gate = "Gate0/1" if l<=0.5 else "Gate9"
    print(f"{l:>7.3f}{dN:>8.0f}{dD/1e9:>11.2f}{t1:>15.1f}{t07:>9.1f}{days:>17.1f}{gate:>11}")
    rows.append(dict(N=N,dN_M=dN,dD_B=dD/1e9,t_step_x1=t1,t_step_x07=t07,one_epoch_days=days))
    prev_N, prev_D = N, Dn

print()
print("="*78)
print("D. CPU->GPU 拐点等值线: 单遍墙钟(天) = (D/B)*t_step(N),  D=3.5e9")
print("="*78)
print(f"{'N(B)':>7}" + "".join(f"{('B=%dk'%(b//1000)):>12}" for b in [4096,8192,16384,32768,65536,131072]))
for l in [0.1,0.2,0.3,0.5,0.75,1.0,1.5,3.0,8.0]:
    N=l*1e9; t=t_step(N,1.0)
    cells=""
    for b in [4096,8192,16384,32768,65536,131072]:
        d=(D_CORPUS/b)*t/86400.0
        cells+=f"{d:>12.1f}"
    print(f"{l:>7.3f}"+cells)
print("  (x=1 计算受限口径; 若带宽受限 x=0.7, 上表数值约降至 %d%%~%d%%)" %
      (100*t_step(8e9,0.7)/t_step(8e9,1.0), 100*t_step(0.1e9,0.7)/t_step(0.1e9,1.0)))

print()
print("="*78)
print("E. adjusted rate 随 N 变化 (D=3.5e9, b=2.0, 原始 32 bit/token)")
print("="*78)
best=(1e9,9e9,None)
for l in [0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0,2.0,4.0,8.0]:
    N=l*1e9; r,L=adj_rate(N,D_CORPUS,2.0)
    if r<best[0]: best=(r,l,L)
    print(f"  N={l:>5.2f}B  L={L:.3f} nat  model_bits={2*N/1e9:.2f}e9  data_bits={(D_CORPUS/LN2)*L/1e9:.2f}e9  adj_rate={r*100:6.2f}%")
print(f"  -> 极小点在 N≈{best[1]:.2f}B (adj_rate={best[0]*100:.2f}%)  [与 N*={N_star(D_CORPUS,2.0)/1e9:.2f}B 一致]")

print()
print("="*78)
print("F. 三值记账下 embedding 未量化带来的有效 bits/param")
print("="*78)
print(f"{'d_model':>9}{'L':>4}{'总参数(M)':>11}{'embedding 占比':>16}{'b_eff':>8}{'b_eff=2 时 N* 的偏差':>22}")
for d,L in [(768,12),(1024,12),(1024,18),(1280,14),(1280,18),(2048,24),(4096,32)]:
    b,n_tot,n_ne,n_e = b_eff(d,L)
    print(f"{d:>9}{L:>4}{n_tot/1e6:>11.1f}{n_e/n_tot:>15.1%}{b:>8.2f}{N_star(D_CORPUS,b)/N_star(D_CORPUS,2.0):>21.2f}x")

json.dump({'rows':rows,'Nstar':{k:{kk:(vv if not isinstance(vv,dict) else vv) for kk,vv in v.items()} for k,v in out.get('Nstar',{}).items()}},
          open('/home/user/ClawsGO/project-20261014-1/analysis/ladder_numbers.json','w'),
          ensure_ascii=False, indent=1, default=str)
print("\n[已写出 analysis/ladder_numbers.json]")
