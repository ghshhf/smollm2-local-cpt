#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantic_probe.py — 视觉语义信号的离线验证（不上闭环，只答一个问题）

  SigLIP 能不能区分「画对了」和「画错了」？

现在 vid_verify 的 8 项全是技术项（可解码/时长/音轨/黑场/静音/期望），
零语义 —— 画成圆的「三角波动画」照样 PASS。本脚本验证补视觉是否真有增量。

两组对照（都不需要人工标注）：
  A 组 = 入库样本（过了 vid_verify + task_fit，task 与画面应匹配）
  B 组 = rejects 里 task_mismatch 的（技术合规，但任务要素缺失/画错）
  ⇒ score(A) 显著高于 score(B) 则信号有效

附加检索式验证 R@1（更严格，防"两组都高分"的假阳性）：
  N 条视频 × N 个 task 描述算相似度矩阵，正确 task 是否排第一
  随机基线 = 1/N，显著高于基线才算真信号

用法(node1):
  python3 -u tools/semantic_probe.py --n 30 --out logs/semantic_probe.json
退出码: 0=跑完  1=环境/数据错误
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)                      # dream/
DB = os.path.join(D, "data", "triple_bank.db")
MODEL_DEFAULT = "google/siglip2-base-patch16-224"
FF = "ffmpeg"


def frames(video, prefix, n_max=3):
    """均匀抽帧：fps=1 每秒 1 帧，最多 n_max 张。返回 png 路径列表。"""
    out = []
    for i in range(n_max):
        p = "%s_%02d.png" % (prefix, i)
        if os.path.exists(p):
            os.remove(p)
    cmd = [FF, "-y", "-loglevel", "error", "-i", video,
           "-vf", "fps=1", "-frames:v", str(n_max), prefix + "_%02d.png"]
    subprocess.run(cmd, capture_output=True, timeout=120)
    for i in range(n_max):
        p = "%s_%02d.png" % (prefix, i)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--n", type=int, default=30, help="每组样本数")
    ap.add_argument("--threads", type=int, default=4, help="别抢自训的核")
    ap.add_argument("--out", default=os.path.join(D, "logs", "semantic_probe.json"))
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from PIL import Image
    from transformers import AutoProcessor, AutoModel

    t0 = time.time()
    print("[probe] 加载模型 %s ..." % a.model, flush=True)
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model)
    model.eval()
    print("[probe] 模型就绪 %.1fs" % (time.time() - t0), flush=True)

    c = sqlite3.connect(a.db, timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    # A 组：入库样本，按唯一定码取（去重口径与 evolve_train 一致）
    rows_a = c.execute(
        "SELECT id, task, code_text, video_path FROM triples "
        "GROUP BY code_text ORDER BY id DESC LIMIT ?", (a.n,)).fetchall()
    # B 组：rejects 里 task_mismatch（技术合规但要素缺失/画错）
    rows_b = c.execute(
        "SELECT id, task, video_path FROM rejects "
        "WHERE reason='task_mismatch' ORDER BY id DESC LIMIT ?", (a.n,)).fetchall()
    c.close()
    print("[probe] A组(入库)=%d  B组(task_mismatch)=%d" % (len(rows_a), len(rows_b)), flush=True)
    if not rows_a or not rows_b:
        print("[probe] 样本不足，退出")
        return 1

    tmpd = tempfile.mkdtemp(prefix="semprobe_")

    def score_one(video, task):
        """返回 视频各帧 × 该 task 文本 的相似度均值；无帧返回 None"""
        fs = frames(video, os.path.join(tmpd, "f%d" % (abs(hash(video)) % 10 ** 8)))
        if not fs:
            return None
        ims = []
        for p in fs:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if not ims:
            return None
        with torch.no_grad():
            inputs = proc(text=[task] * len(ims), images=ims,
                          return_tensors="pt", padding=True)
            out = model(**inputs)
            # SigLIP: logits_per_image [n_img, n_text]
            lg = out.logits_per_image
            if lg is None:
                return None
            v = float(lg[:, 0].mean())
        for p in fs:
            try:
                os.remove(p)
            except Exception:
                pass
        return v

    res_a, res_b = [], []
    for tag, rows, bucket in (("A", rows_a, res_a), ("B", rows_b, res_b)):
        for r in rows:
            rid, task, video = r[0], r[1], r[-1]
            if not video or not os.path.isfile(video):
                continue
            s = score_one(video, task)
            if s is not None:
                bucket.append({"id": rid, "task": task[:60], "score": round(s, 4)})
        print("[probe] %s 组完成 %d 条 (%.0fs)" % (tag, len(bucket), time.time() - t0), flush=True)

    def stats(b):
        vs = sorted(x["score"] for x in b)
        if not vs:
            return {"n": 0}
        n = len(vs)
        return {"n": n, "mean": round(sum(vs) / n, 4),
                "median": round(vs[n // 2], 4),
                "min": round(vs[0], 4), "max": round(vs[-1], 4)}

    sa, sb = stats(res_a), stats(res_b)

    # 检索式 R@1：A 组视频 × A 组 task，正确 task 是否排第一
    r1 = None
    if len(res_a) >= 5:
        ims_all, tasks_all, owner = [], [], []
        for r in rows_a:
            video, task = r[-1], r[1]
            if not video or not os.path.isfile(video):
                continue
            fs = frames(video, os.path.join(tmpd, "r%d" % (abs(hash(video)) % 10 ** 8)), n_max=3)
            if not fs:
                continue
            try:
                ims_all.append(Image.open(fs[0]).convert("RGB"))
                tasks_all.append(task)
                owner.append(len(tasks_all) - 1)
            except Exception:
                continue
        if len(ims_all) >= 5:
            with torch.no_grad():
                inputs = proc(text=tasks_all, images=ims_all,
                              return_tensors="pt", padding=True)
                lg = model(**inputs).logits_per_image   # [n_img, n_text]
            top1 = 0
            for i in range(len(ims_all)):
                if int(lg[i].argmax()) == owner[i]:
                    top1 += 1
            r1 = {"n": len(ims_all), "top1": top1,
                  "r1": round(top1 / len(ims_all), 4),
                  "random_baseline": round(1.0 / len(ims_all), 4)}

    verdict = "unknown"
    if sa.get("n", 0) >= 5 and sb.get("n", 0) >= 5:
        verdict = "signal_ok" if sa["mean"] > sb["mean"] else "no_signal"

    report = {"model": a.model, "ts": time.strftime("%F %T"),
              "group_A_inbank": sa, "group_B_task_mismatch": sb,
              "retrieval_r1": r1, "verdict": verdict,
              "samples_A": res_a[:10], "samples_B": res_b[:10]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n===== 语义信号验证结果 =====")
    print("A组(入库)        : %s" % sa)
    print("B组(task_mismatch): %s" % sb)
    print("检索 R@1         : %s" % r1)
    print("判定             : %s" % verdict)
    print("报告             : %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
