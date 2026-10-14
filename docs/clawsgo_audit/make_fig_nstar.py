import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2.0)
A_, ALPHA = 406.4, 0.34

def N_star(D, b):
    return (D * A_ * ALPHA / (b * LN2)) ** (1.0 / (1.0 + ALPHA))

def D_needed(N, b):
    return (N ** (1.0 + ALPHA)) * b * LN2 / (A_ * ALPHA)

# b_eff for ternary weights + fp16 embedding at d=1024, L=12, V=49152
n_nonemb = 12.0 * 12 * 1024 * 1024   # d=1024, L=12
n_emb = 49152.0 * 1024
B_MIX = (1.58 * n_nonemb + 16.0 * n_emb) / (n_nonemb + n_emb)

REG = [("fp32 (32b)", 32.0, "#444444", "-"),
       ("fp16 (16b)", 16.0, "#1f77b4", "--"),
       ("2-bit", 2.0, "#2ca02c", "-."),
       ("1.58-bit", 1.58, "#d62728", ":"),
       (f"ternary+fp16 emb (b_eff={B_MIX:.2f})", B_MIX, "#9467bd", "-")]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- (a) N*(D,b) curves ----
D = np.logspace(math.log10(1e9), math.log10(2e11), 300)
for name, b, c, ls in REG:
    ax[0].plot(D, N_star(D, b) / 1e6, color=c, ls=ls, lw=2, label=name)
ax[0].axvline(3.5e9, color="k", lw=1.2, alpha=.7)
ax[0].text(3.6e9, 6.0, "our corpus\n3.5e9 tok", fontsize=9, va="bottom")
for N, lab in [(135e6, "R2 135M"), (200e6, "0.2B"), (400e6, "0.4B"), (1e9, "1B"), (3e9, "3B"), (8e9, "8B")]:
    ax[0].axhline(N / 1e6, color="grey", lw=.6, alpha=.45)
    ax[0].text(1.05e9, N / 1e6 * 1.03, lab, fontsize=8, color="grey")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlim(1e9, 2e11); ax[0].set_ylim(30, 1.2e4)
ax[0].set_xlabel("语料规模 D (token)")
ax[0].set_ylabel(r"压缩最优容量点 $N^*$ (百万参数)")
ax[0].set_title("(a) $N^*(D,b)$：固定语料下的最优参数量")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=.25, which="both")

# ---- (b) corpus needed for a given N ----
Ns = [1e9, 3e9, 8e9]
labels = ["1B", "3B", "8B"]
x = np.arange(len(Ns)); wdt = 0.16
for i, (name, b, c, ls) in enumerate(REG):
    vals = [D_needed(N, b) / 1e9 for N in Ns]
    bar = ax[1].bar(x + (i - 2) * wdt, vals, wdt, color=c, label=name)
    ax[1].bar_label(bar, fmt="%.0f", fontsize=7, padding=1)
ax[1].axhline(3.5, color="k", lw=1.6)
ax[1].text(-0.44, 3.9, "现有语料 3.5e9", fontsize=9, ha="left")
ax[1].set_yscale("log")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"目标规模 {l}" for l in labels])
ax[1].set_ylabel("使该规模成为最优所需的语料 (十亿 token)")
ax[1].set_title("(b) 各档规模所需的语料量（对数轴）")
ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].grid(alpha=.25, axis="y", which="both")

fig.suptitle("逐级扩容的定量边界：压缩最优容量点与语料需求（Hoffmann 2022 常数 + Delétang 判据）", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_nstar_budget.png", dpi=170)
print("saved fig_nstar_budget.png")
