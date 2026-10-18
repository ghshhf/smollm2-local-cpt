#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audio_attach.py — 给北极星 0.3B 装上「听觉」：Whisper 编码器 + 投影层 -> LLM。

架构（照 vlm_attach.py 的已验证配方）：
    wav -> log-mel(80,3000) -> WhisperEncoder(frozen) -> (1500, 768)
        -> 线性池化到 n_tok -> MLPProjector(768 -> llm_hidden) -> (n_tok, d)
        -> 与文本 embedding 做 early fusion -> 0.3B LLM -> 输出转录文本

三阶段：
    align  只训投影层（Whisper 与 LLM 全冻），学"音频特征 -> 语言空间"
    joint  放开 LLM（lr*0.1）与 Whisper（lr*0.05），学"读音频 token"
    asr    同 joint，但训练目标固定为转录（本项目当前只用到这一档）

验收（禁信 loss）：字符错误率 CER + 盲测样本对照。

用法：
  # 0) 先做数据
  python tools/build_audio_pairs.py

  # 1) 预计算音频特征（Whisper 冻结 => 可缓存，训练提速 10x）
  python tools/audio_attach.py --precompute --aud-tokens 150

  # 2) 训练投影层
  python tools/audio_attach.py --llm backups/0.3B/polaris-03b_20261018 \
      --stage align --steps 2000 --use-cache --save models/audio_03b_align

  # 3) 推理验收
  python tools/audio_attach.py --infer --llm backups/0.3B/polaris-03b_20261018 \
      --proj models/audio_03b_align/projector.pt --wav data/audio/wav/xxx.wav
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAIRS = os.path.join(ROOT, "data", "audio", "aishell_pairs.jsonl")
CACHE = os.path.join(ROOT, "data", "audio", "aud_feats.pt")
WHISPER_DEFAULT = os.path.join(ROOT, "..", "models", "whisper-small")
SR = 16000
PROMPT = "请把这段语音转成文字。"


# ============================================================
# 组件
# ============================================================
class MLPProjector(nn.Module):
    """2层 MLP: audio_hidden -> llm_hidden

    ⚠️ fc2 必须微小随机初始化(std=0.01)，绝不能用零初始化：
    零输出会让 Llama RMSNorm 反向除 sqrt(eps)≈316，深堆叠后 fp32 梯度溢出
    => loss 有限但梯度全 nan（2026-10-17 在视觉线上实测复现，同源坑）。
    """
    def __init__(self, a_dim, l_dim):
        super().__init__()
        self.fc1 = nn.Linear(a_dim, l_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(l_dim, l_dim, bias=True)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def load_wav(path, target_sr=SR):
    """wave 读 16k 单声道 float32；非 16k 线性重采样。"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) / 128.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    if sr != target_sr:
        m = int(round(len(a) * target_sr / sr))
        if m > 1:
            idx = np.linspace(0, len(a) - 1, m)
            a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    return a


def pool_audio(x, n_tok):
    """(B,T,D) -> (B,n_tok,D) 线性插值池化（Whisper 输出 1500 帧太长）。"""
    if n_tok <= 0 or n_tok >= x.size(1):
        return x
    idx = torch.linspace(0, x.size(1) - 1, n_tok).round().long()
    return x[:, idx, :]


def speaker_of(utt):
    """AISHELL utt 形如 BAC009S0002W0122 -> S0002"""
    i = utt.find("S")
    return utt[i:i + 5] if i >= 0 else utt[:5]


def split_by_speaker(rows, val_speakers):
    """按说话人切分 train/val —— 真盲测（验证集的人声训练时从没见过）。"""
    vs = set(s.strip() for s in val_speakers.split(",") if s.strip())
    if not vs:
        return rows, []
    tr = [r for r in rows if speaker_of(r["utt"]) not in vs]
    va = [r for r in rows if speaker_of(r["utt"]) in vs]
    return tr, va


def load_pairs(limit=None):
    if not os.path.exists(PAIRS):
        raise FileNotFoundError("先跑 tools/build_audio_pairs.py 生成 %s" % PAIRS)
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref = "".join(ref.split())
    hyp = "".join(hyp.split())
    return edit_distance(ref, hyp) / max(1, len(ref))


def norm_text(s):
    """去掉 AISHELL 文本里的标点空格，只留中文和字母数字。"""
    keep = []
    for ch in s:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            keep.append(ch)
    return "".join(keep)


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=os.path.join(ROOT, "backups", "0.3B", "polaris-03b_20261018"))
    ap.add_argument("--whisper", default=WHISPER_DEFAULT)
    ap.add_argument("--stage", default="align", choices=["align", "joint", "asr"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--aud-tokens", type=int, default=150)
    ap.add_argument("--max-sec", type=float, default=8.0, help="超过这个时长截断(控算力)")
    ap.add_argument("--save", default=os.path.join(ROOT, "models", "audio_03b_align"))
    ap.add_argument("--init", default=None)
    ap.add_argument("--proj", default=None, help="推理用的 projector.pt")
    ap.add_argument("--limit", type=int, default=0, help="限制样本数(0=全量)")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--rept-penalty", type=float, default=1.3)
    ap.add_argument("--eval-n", type=int, default=30, help="训练后自动盲测条数")
    ap.add_argument("--val-speakers", default="S0024,S0025",
                    help="留作验证的说话人(逗号分隔)，训练时不使用；空串=不切分")
    a = ap.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    os.makedirs(a.save, exist_ok=True)

    # ---- 1. Whisper 编码器（冻结）----
    from transformers import WhisperFeatureExtractor, WhisperModel
    if not os.path.isdir(a.whisper):
        print("[fatal] 找不到 Whisper: %s" % a.whisper)
        return 2
    t0 = time.time()
    fe = WhisperFeatureExtractor.from_pretrained(a.whisper)
    wm = WhisperModel.from_pretrained(a.whisper, dtype=torch.float32)
    whisper = wm.encoder.eval()
    for p in whisper.parameters():
        p.requires_grad = False
    a_dim = whisper.config.d_model
    print("[whisper] encoder 就绪 d_model=%d  %.1fs" % (a_dim, time.time() - t0))

    def audio_feat(wav_path):
        """wav -> (T,768) whisper 帧特征（先池化到 aud_tokens，控内存/算力）"""
        arr = load_wav(wav_path)
        max_n = int(a.max_sec * SR)
        if len(arr) > max_n:
            arr = arr[:max_n]
        if len(arr) < 400:
            arr = np.pad(arr, (0, 400 - len(arr)))
        inp = fe(arr, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            enc = whisper(input_features=inp.input_features)
        h = enc.last_hidden_state[0]                     # (1500, d)
        return pool_audio(h.unsqueeze(0), a.aud_tokens)[0]   # (n_tok, d)

    # ---- 2. 预计算 ----
    if a.precompute:
        rows = load_pairs()
        # 必须覆盖留出的验证说话人，否则训练完无法做真盲测
        tr, va = split_by_speaker(rows, a.val_speakers)
        if a.limit:
            tr = tr[:a.limit]
            va = va[:max(60, a.limit // 10)]
        rows = tr + va
        print("[pre] 计划: 训练集 %d + 验证集 %d = %d 条" % (len(tr), len(va), len(rows)))
        feats, miss = {}, 0
        if os.path.exists(a.cache):
            try:
                feats = torch.load(a.cache, map_location="cpu")
                print("[pre] 已有缓存 %d 条，续算" % len(feats))
            except Exception as e:
                print("[pre] 缓存损坏，重建: %s" % e)
                feats = {}
        todo = [r for r in rows if r["utt"] not in feats]
        print("[pre] 待计算 %d 条 (共 %d)" % (len(todo), len(rows)), flush=True)
        tt = time.time()
        for i, r in enumerate(todo):
            try:
                feats[r["utt"]] = audio_feat(r["audio"]).to(torch.float16)
            except Exception as e:
                miss += 1
                if miss <= 3:
                    print("  [skip] %s %s" % (r["audio"], e))
                continue
            if (i + 1) % 50 == 0:
                el = time.time() - tt
                print("  [pre] %d/%d  %.0fs  (%.2fs/条)" % (
                    i + 1, len(todo), el, el / (i + 1)), flush=True)
        torch.save(feats, a.cache)
        print("[pre] 完成 %d 条 -> %s (%.0fs, %.1fMB)" % (
            len(feats), a.cache, time.time() - tt,
            os.path.getsize(a.cache) / 1e6 if os.path.exists(a.cache) else 0))
        return 0

    # ---- 3. LLM（0.3B）----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.llm)
    llm = AutoModelForCausalLM.from_pretrained(a.llm, dtype=torch.float32).eval()
    l_dim = llm.config.hidden_size
    print("[LLM] hidden=%d params=%.1fM" % (l_dim, sum(p.numel() for p in llm.parameters()) / 1e6))

    proj = MLPProjector(a_dim, l_dim)
    if a.init and os.path.exists(a.init):
        proj.load_state_dict(torch.load(a.init, map_location="cpu"))
        print("[init] projector <- %s" % a.init)

    # ---- 4. 冻结策略 ----
    if a.stage == "align":
        for p in llm.parameters():
            p.requires_grad = False
    else:
        for p in llm.parameters():
            p.requires_grad = True
    for p in proj.parameters():
        p.requires_grad = True
    params = [{"params": proj.parameters(), "lr": a.lr}]
    if a.stage != "align":
        params.append({"params": llm.parameters(), "lr": a.lr * 0.1})
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))
    n_train = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    if a.stage != "align":
        n_train += sum(p.numel() for p in llm.parameters() if p.requires_grad)
    print("[TRAIN] stage=%s 可训练=%.2fM" % (a.stage, n_train / 1e6))

    cache = None
    if a.use_cache and os.path.exists(a.cache):
        print("[cache] 加载音频特征 %s" % a.cache)
        cache = torch.load(a.cache, map_location="cpu")   # {utt: (n_tok,d) fp16}
        print("[cache] %d 条就绪" % len(cache))

    def get_atok(wav_path, utt=None):
        if cache is not None and utt and utt in cache:
            f = cache[utt].float().unsqueeze(0)
        else:
            f = audio_feat(wav_path).unsqueeze(0).float()
        return proj(f)                                     # (1,n_tok,l_dim)

    def make_prompt():
        return "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % PROMPT

    def forward_step(wav_path, text, utt=None):
        atok = get_atok(wav_path, utt)
        prompt = make_prompt()
        pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        tgt = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        eos = tok("<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
        tgt = torch.cat([tgt, eos], dim=1)
        temb = llm.get_input_embeddings()(torch.cat([pids, tgt], dim=1))
        inputs = torch.cat([atok, temb], dim=1)             # early fusion
        out = llm(inputs_embeds=inputs)
        np_ = atok.size(1) + pids.size(1)
        logits = out.logits[:, np_ - 1:-1, :]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               tgt.reshape(-1))

    # ---- 5. 推理 ----
    def generate(wav_path, utt=None, proj_path=None):
        atok = get_atok(wav_path, utt)
        prompt = make_prompt()
        pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        temb = llm.get_input_embeddings()(pids)
        inputs = torch.cat([atok, temb], dim=1)
        with torch.no_grad():
            out = llm.generate(inputs_embeds=inputs, max_new_tokens=a.max_new,
                               do_sample=False, repetition_penalty=a.rept_penalty,
                               eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True)

    if a.infer:
        if a.proj:
            proj.load_state_dict(torch.load(a.proj, map_location="cpu"))
            print("[infer] projector <- %s" % a.proj)
        proj.eval()
        if not a.wav:
            print("[fatal] --infer 需要 --wav")
            return 2
        t = time.time()
        txt = generate(a.wav)
        print("[infer] %.1fs" % (time.time() - t))
        print("[out] %s" % txt.strip())
        return 0

    # ---- 6. 训练 ----
    rows = load_pairs(a.limit or None)
    rows = [r for r in rows if (not cache) or (r["utt"] in cache)]
    if not rows:
        print("[fatal] 没有可用训练样本")
        return 2
    train_rows, val_rows = split_by_speaker(rows, a.val_speakers)
    if not train_rows:
        train_rows, val_rows = rows, []
        print("[warn] val-speakers 把全部样本都划走了，退回单集")
    print("[data] 训练 %d 条 | 验证(没见过的说话人) %d 条 | val=%s"
          % (len(train_rows), len(val_rows), a.val_speakers or "无"))
    proj.train()
    if a.stage != "align":
        llm.train()

    step, loss_acc, t0 = 0, 0.0, time.time()
    import random
    rng = random.Random(7)
    order = list(range(len(train_rows)))
    while step < a.steps:
        rng.shuffle(order)
        for i in order:
            r = train_rows[i]
            try:
                loss = forward_step(r["audio"], norm_text(r["text"]), r["utt"])
            except Exception as e:
                print("  [skip] %s %s" % (r["audio"], e))
                continue
            (loss / a.accum).backward()
            loss_acc += float(loss)
            if (i + 1) % a.accum == 0 or i == order[-1]:
                torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                if a.stage != "align":
                    torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = time.time() - t0
                    print("[step %4d/%d] loss=%.4f  %.0fs (%.1fs/步)"
                          % (step, a.steps, loss_acc / 10, el, el / step), flush=True)
                    loss_acc = 0.0
                if step % 250 == 0:
                    torch.save(proj.state_dict(),
                               os.path.join(a.save, "projector_%d.pt" % step))
                if step >= a.steps:
                    break

    torch.save(proj.state_dict(), os.path.join(a.save, "projector.pt"))
    print("[save] %s" % os.path.join(a.save, "projector.pt"))

    # ---- 7. 训练后盲测（禁信 loss：直接用 CER 说话）----
    if a.eval_n > 0:
        proj.eval()
        llm.eval()
        pool_ = val_rows if val_rows else train_rows
        blind = bool(val_rows)
        sample = rng.sample(pool_, min(a.eval_n, len(pool_)))
        errs = []
        print("\n[EVAL] %s %d 条（字符错误率）"
              % ("真盲测(未见说话人)" if blind else "训练集内(非盲测!)", len(sample)))
        for r in sample:
            hyp = generate(r["audio"], r["utt"])
            ref = norm_text(r["text"])
            e = cer(ref, hyp)
            errs.append(e)
            if len(errs) <= 5:
                print("  参考: %s" % ref[:40])
                print("  输出: %s  (CER=%.2f)" % (hyp.strip()[:40], e))
        mean = sum(errs) / len(errs)
        print("[EVAL] 平均 CER = %.3f  中位 = %.3f  完全正确 = %d/%d"
              % (mean, sorted(errs)[len(errs) // 2],
                 sum(1 for x in errs if x < 0.05), len(errs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
