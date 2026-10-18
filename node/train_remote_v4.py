# -*- coding: utf-8 -*-
"""魔搭节点端训练脚本 v4 (64 核 CPU, 续训 R4 214M) —— fp32 修复版。

v4 相对 v3 的关键改动:
  - 彻底移除 torch.autocast(cpu, bf16): CPU bf16 在 Linux 服务器上数值崩溃,
    导致初始 loss 飙到 ~31 且全程平走不降(本机同栈 fp32/bf16 初始 loss 均 16.31,
    但 node1 是 Linux + 训练态, bf16 反向不稳定, 把权重训坏)。改回 fp32 全精度训练。
  - AdamW betas=(0.9, 0.95) 与本地已验证有效的 train_smollm2_pt.py 对齐。
  - warmup 提到 50。
  - 其余(缓存/随机切片/续训/上传)与 v3 一致。
"""
import argparse, os, sys, json, time, glob, math
import urllib.parse, urllib.request, urllib.error

def upload_file(url, local_path, name=None, retries=4):
    """把本地文件 POST 到 upload_url(本机隧道 /upload 或 ECS /upload 端点)。
    大文件(856MB)经 ECS 单线程 mailserver 易 Broken pipe/403, 带重试+退避。"""
    name = name or os.path.basename(local_path)
    for attempt in range(1, retries + 1):
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            req = urllib.request.Request(
                f"{url}?name={urllib.parse.quote(name)}",
                data=data, method="POST",
                headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                print(f"[upload] {name} -> {r.status} {r.read(200).decode('utf-8','ignore')}")
            return True
        except Exception as e:
            print(f"[upload] FAIL {name} (try {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(20 * attempt)
    return False

def _safe_load_tokenizer(base_dir):
    """加载 tokenizer, 规避 transformers 新版本把 extra_special_tokens(list) 当 dict 的崩溃。
    现象: AttributeError: 'list' object has no attribute 'keys' (tokenization_gpt2_fast._set_model_specific_special_tokens)。
    修复: 把 tokenizer_config.json 里的 list 归一化为 {}, 失败再回退 use_fast=False。"""
    from transformers import AutoTokenizer
    import json as _json
    tc = os.path.join(base_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        try:
            with open(tc, "r", encoding="utf-8") as f:
                d = _json.load(f)
            if isinstance(d.get("extra_special_tokens"), list):
                d["extra_special_tokens"] = {}
                with open(tc, "w", encoding="utf-8") as f:
                    _json.dump(d, f, ensure_ascii=False, indent=2)
                print("[tok] 归一化 tokenizer_config.json: extra_special_tokens list->{}")
        except Exception as e:
            print(f"[tok] 归一化警告(忽略): {e}")
    try:
        return AutoTokenizer.from_pretrained(base_dir)
    except Exception as e:
        print(f"[tok] 快速 tokenizer 加载失败({e}), 回退 use_fast=False")
        return AutoTokenizer.from_pretrained(base_dir, use_fast=False)


def _save_ckpt(model, ckpt):
    """存权重。优先 safetensors(规避 torch<2.6 的 CVE 拒绝 pickle 加载),
    tied weights 触发共享内存报错时回退 pickle bin。"""
    import os as _os
    _os.makedirs(ckpt, exist_ok=True)
    try:
        model.save_pretrained(ckpt, safe_serialization=True)
        if not _os.path.exists(_os.path.join(ckpt, "model.safetensors")):
            raise RuntimeError("no model.safetensors produced")
    except Exception as e:
        print(f"[ckpt] safetensors failed ({repr(e)[:120]}) -> fallback bin")
        model.save_pretrained(ckpt, safe_serialization=False)
        # 关键: 必须清掉可能残留的旧 model.safetensors,
        # 否则下一轮 from_pretrained 会优先读 safetensors -> 加载旧权重 -> 白训。
        old = _os.path.join(ckpt, "model.safetensors")
        if _os.path.exists(old):
            _os.remove(old)
            print("[ckpt] removed stale model.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/mnt/workspace/node1/r4_base")
    ap.add_argument("--run_dir", default="/mnt/workspace/node1/r4_run")
    ap.add_argument("--corpus_dir", default="/mnt/workspace/node1/corpus")
    ap.add_argument("--cache", default="/mnt/workspace/node1/tok_cache.npy")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=512)
    # 2026-10-14 修复: 原 batch_size 硬编码为 1 -> 每步仅 accum*512=4096 token,
    # GEMM 太小导致 64 核只跑出 4.4 GFLOP/s/核(约峰值 5~10%)。
    # 社区小模型建议 0.5M~1M token/batch; 我们算力有限, 取 16*512=8192/微批 起步。
    # 🔴 07:26 实测推翻上面这段推理: max_len=512 时序列维本身就是 batch 维,
    # GEMM 已是 (512x576)x(576x576), 本来就够大。batch=16 反而把中间激活撑到
    # 18.9MB/层/张量, 30 层直接撞内存带宽墙: 8线程下每步 18.7s -> >140s(吞吐 -47%),
    # 64线程下 -> >75s(吞吐持平)。=> batch 回到 1, 每步 4096 token。
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    # workers=0: RandDS.__getitem__ 只是一次 randint + mmap 切片, 开销可忽略;
    # 用 worker 进程反而要 fork 281M token 的 arr, 还会触发 tokenizers 的
    # "process just got forked" 警告(有死锁风险)。主进程取数最稳。
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=300_000_000)
    # 线程数显式可调: 容器里 sched_getaffinity 报 64 但实际超售,
    # 64 线程跑 214M 模型每核仅 ~3 GFLOP/s(8 线程时是 35), 纯竞争无收益。
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--upload_url", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import Dataset, DataLoader
    import torch
    try:
        n_th = args.threads if args.threads > 0 else len(os.sched_getaffinity(0))
    except Exception:
        n_th = args.threads if args.threads > 0 else (os.cpu_count() or 8)
    torch.set_num_threads(n_th)
    print(f"[threads] torch.set_num_threads({n_th})")
    import numpy as np
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    os.makedirs(args.run_dir, exist_ok=True)

    print(f"[load] model + tokenizer from {args.base_dir}")
    tok = _safe_load_tokenizer(args.base_dir)
    # fp32 全精度(关键修复: 去掉 bf16 autocast)
    model = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.float32)
    model.train(); device = "cpu"
    print(f"[load] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={device} (fp32)")

    # ---- 语料 -> 有上限 token cache ----
    corpus_dir = args.corpus_dir
    base_dir = args.base_dir
    cache_path = args.cache
    max_tokens = args.max_tokens
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        print(f"[cache] 复用已有 token cache: {arr.shape[0]} tokens")
    else:
        tk = _safe_load_tokenizer(base_dir)
        files = sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl")) +
                       glob.glob(os.path.join(corpus_dir, "*.txt")))
        print(f"[cache] 流式 tokenize {len(files)} 个文件, 上限 {max_tokens//1_000_000}M tokens")
        chunks, total = [], 0
        for fp in files:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                buf = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line); txt = obj.get("text", "")
                    except Exception:
                        txt = line
                    if not txt or len(txt) < 20:
                        continue
                    buf.append(txt)
                    if len(buf) >= 400:
                        ids = tk("\n".join(buf), return_tensors=None, truncation=False).input_ids
                        buf = []
                        if ids:
                            chunks.append(np.array(ids, dtype=np.int32)); total += len(ids)
                        if total >= max_tokens:
                            break
            if total >= max_tokens:
                break
        arr = np.concatenate(chunks) if chunks else np.array([0], dtype=np.int32)
        np.save(cache_path, arr)
        print(f"[cache] 完成: {arr.shape[0]} tokens -> {cache_path}")

    N = arr.shape[0]
    if N < args.max_len + 1:
        print("[ERR] token 太少, 退出"); sys.exit(1)

    class RandDS(Dataset):
        def __len__(self): return 1_000_000
        def __getitem__(self, _):
            s = int(np.random.randint(0, N - args.max_len - 1))
            return torch.from_numpy(arr[s:s + args.max_len + 1].astype(np.int64)).clone()

    dl = DataLoader(RandDS(), batch_size=args.batch, num_workers=args.workers)
    it = iter(dl)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, total_iters=args.warmup),
        CosineAnnealingLR(opt, T_max=max(1, args.steps - args.warmup))],
        milestones=[args.warmup])

    tok_per_step = args.batch * args.accum * args.max_len
    print(f"[train] steps={args.steps} lr={args.lr} batch={args.batch} accum={args.accum} "
          f"max_len={args.max_len} => {tok_per_step} token/步 (fp32)")
    # 注意: 不要在此处加"token 数偏小"类告警。曾误判 batch=1 是瓶颈,
    # 实测 batch=16 反而更慢(中间激活 18.9MB/层/张量 x 30 层撞内存带宽墙,
    # 8线程每步 18.7s -> >140s)。4096 token/步在 max_len=512+batch=1 下已是极限。
    t0 = time.time(); opt_step = 0; loss_acc = 0
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            # fp32 全精度前向(无 autocast)
            # 标签口径: input_ids == labels(完整 chunk), 由 HF CausalLM 内部做 shift。
            # 写成 labels=input_ids[:,1:] 会被再移一次 -> 目标错位一格, loss 虚高到 ~16/80。
            # (本地 train_smollm2_pt.py 注释里明确记过这个坑)
            x = batch[:, :-1]
            out = model(input_ids=x, labels=x)
            loss = out.loss / args.accum
            loss.backward()
            loss_acc += loss.item() * args.accum
        opt.step(); sched.step(); opt_step += 1
        if step % 10 == 0:
            # loss_acc 累计的是 accum×10 个微批的真实 loss, 必须除以微批数才是均值。
            # 之前除 10 -> 打印值被放大 accum(=8) 倍, 曾把 2.09 显示成 16.7。
            el = time.time() - t0
            tps = step * tok_per_step / max(el, 1e-9)
            print(f"[train] step {step}/{args.steps} loss={loss_acc/(10*args.accum):.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={el:.0f}s tok/s={tps:.0f}")
            sys.stdout.flush(); loss_acc = 0
        if step % 100 == 0:
            ckpt = os.path.join(args.run_dir, "checkpoint-latest")
            _save_ckpt(model, ckpt)
            open(os.path.join(ckpt, "step.txt"), "w").write(str(step))
            print(f"[ckpt] saved @ step {step}")
            sys.stdout.flush()

    ckpt = os.path.join(args.run_dir, "checkpoint-latest")
    _save_ckpt(model, ckpt)
    open(os.path.join(ckpt, "step.txt"), "w").write(str(args.steps))
    print(f"[train] DONE steps={args.steps} total_t={time.time()-t0:.0f}s -> {ckpt}")
    sys.stdout.flush()

    if args.upload_url:
        print("[upload] 推回权重 ...")
        _st = os.path.join(ckpt, "model.safetensors")
        _bin = os.path.join(ckpt, "pytorch_model.bin")
        if os.path.exists(_st):
            upload_file(args.upload_url, _st, "r4_run_model.safetensors")
        elif os.path.exists(_bin):
            upload_file(args.upload_url, _bin, "r4_run_pytorch_model.bin")
        upload_file(args.upload_url, os.path.join(ckpt, "step.txt"), "r4_run_step.txt")

if __name__ == "__main__":
    main()
