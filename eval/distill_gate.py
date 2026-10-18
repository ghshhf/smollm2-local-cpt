# -*- coding: utf-8 -*-
"""
distill_gate.py — 0.5B 蒸馏管线的分歧 token 门控（P3 防御性实现）
================================================================
背景（CONSTITUTION 附录 A4 / E1 前置防御）：
  subliminal learning（Cloud et al., Nature 2026）表明同族教师-学生蒸馏会把教师
  隐性属性经生成数据无声传给学生，输出层不可检测。防御手段之一（报告 §8.2）：
  在教师"分歧 token"位置降权/mask——两代教师（当前教师 vs 更早/更小的教师）
  分布不一致的位置，最可能携带隐性特质信号。

用法（训练循环内）：
    from eval.distill_gate import GateScheduler, divergence_gate
    sched = GateScheduler(total_steps=2000)          # 三段式：冻结→放开→衰减
    for step in range(2000):
        w = divergence_gate(logits_teacher_now, logits_teacher_prev,
                            tau=sched.tau(step), kl_thresh=sched.kl_thresh(step))
        loss = gated_kl_loss(student_logits, logits_teacher_now, weights=w, tau=sched.tau(step))
        ...

门控语义：w=1 正常蒸馏；w=0 该 token 退化为纯硬标签 CE（不吸收教师分布细节）。
参数来源：Grok Q-10 建议（τ 分段 2–4 / 1.5–2 / →1；KL 阈值 0.3–0.5，后期收紧），
配 ClawsGO 执行手册蒸馏三段式（冻结 30–40% / 放开 40–50% / 衰减 20%）。
"""
import torch
import torch.nn.functional as F


class GateScheduler:
    """三段式调度：tau 与 kl_thresh 随训练进度变化。"""

    def __init__(self, total_steps, freeze_frac=0.35, open_frac=0.45):
        # 段1 冻结主干(只训新组件): tau 高(2~4) 保软分布信息; 门控宽松
        # 段2 放开: tau 中(1.5~2); 门控收紧
        # 段3 衰减: tau→1; 门控最严
        self.total = total_steps
        self.p1 = int(total_steps * freeze_frac)
        self.p2 = int(total_steps * (freeze_frac + open_frac))

    def phase(self, step):
        return 0 if step < self.p1 else (1 if step < self.p2 else 2)

    def tau(self, step):
        return [3.0, 2.0, 1.0][self.phase(step)]

    def kl_thresh(self, step):
        return [0.15, 0.20, 0.30][self.phase(step)]    # JSD 阈值（有界 ln2），后期收紧

    def freeze_backbone(self, step):
        return self.phase(step) == 0


@torch.no_grad()
def divergence_gate(logits_now, logits_prev, jsd_thresh=0.2, topk_guard=8):
    """
    逐 token 计算两教师分布的 Jensen-Shannon 散度（有界 [0, ln2]，尺度无关——
    KL 阈值随词表大小/温度/分布峰度漂移，V=1000 平坦分布实测 KL 只有 ~0.27，
    固定 KL 阈值不可移植，故换 JSD）。返回 weights: [B,T]，1=可蒸馏, 0=分歧(退硬标签)。
    jsd_thresh 默认 0.2 ≈ 29% of ln2：轻度分歧放行，明显分歧关闸。
    """
    ln2 = math.log(2.0)
    pn = F.log_softmax(logits_now, dim=-1)          # log P
    pp = F.log_softmax(logits_prev, dim=-1)         # log Q
    m_log = torch.logaddexp(pn, pp) - ln2           # log M,  M=(P+Q)/2
    kl_pm = (pn.exp() * (pn - m_log)).sum(-1)
    kl_qm = (pp.exp() * (pp - m_log)).sum(-1)
    jsd = 0.5 * kl_pm + 0.5 * kl_qm                 # [B,T], 上界 ln2
    w = (jsd < jsd_thresh).float()
    # top-k 保险：两教师 top-k 集合完全不相交的 token 一律视为分歧
    kn = logits_now.topk(topk_guard, dim=-1).indices
    kp = logits_prev.topk(topk_guard, dim=-1).indices
    intersect = (kn.unsqueeze(-1) == kp.unsqueeze(-2)).any(-1).any(-1)   # [B,T]（torch.isin 是逐元素比较，不能做集合求交）
    w = w * intersect.float()
    return w


def gated_distill_loss(student_logits, teacher_logits, weights, tau=2.0, hard_labels=None,
                       hard_weight=1.0):
    """
    加权蒸馏损失 = w·KL(学生‖教师@tau)·τ² + (1−w)·CE(硬标签)。
    hard_labels: [B,T] 教师生成的 token id（数据本身的下一 token）。
    """
    B, T, V = student_logits.shape
    sl = F.log_softmax(student_logits / tau, dim=-1)
    tl = F.log_softmax(teacher_logits / tau, dim=-1)
    kl_tok = F.kl_div(tl, sl.exp(), reduction="none").sum(-1) * (tau * tau)   # [B,T]
    loss_soft = (weights * kl_tok).sum() / weights.sum().clamp(min=1.0)
    loss = loss_soft
    if hard_labels is not None:
        hard = F.cross_entropy(student_logits.reshape(-1, V), hard_labels.reshape(-1),
                               reduction="none").reshape(B, T)
        loss_inv = ((1 - weights) * hard).sum() / (1 - weights).sum().clamp(min=1.0)
        loss = loss + hard_weight * loss_inv
    return loss


if __name__ == "__main__":
    # 自检：近一致分布应几乎全放行；人为制造峰化分歧位置应被关掉
    import math
    torch.manual_seed(0)
    B, T, V = 2, 16, 1000
    a = torch.randn(B, T, V) * 8.0                # 峰化分布（贴近真实模型 logits）
    b = a + 0.01 * torch.randn(B, T, V)          # 几乎一致
    w = divergence_gate(a, b)
    assert (w == 1).float().mean() > 0.95, "一致分布被误杀"
    b2 = a.clone()
    b2[:, 3] = torch.randn(B, V) * 8.0           # 人为分歧 token（独立峰化分布）
    w2 = divergence_gate(a, b2)
    assert (w2[:, 3] == 0).all(), "分歧位置未被关掉"
    assert w2.sum() >= B * T - 4, "误杀过多"
    # 平坦分布下也不应整片误杀（教训用例：V=1000, randn, KL 只有 ~0.27）
    a0, b0 = torch.randn(B, T, V), torch.randn(B, T, V)
    w0 = divergence_gate(a0, b0)
    assert 0 < w0.float().mean() < 1, "平坦分布下门控行为异常"
    print("distill_gate self-test OK")
