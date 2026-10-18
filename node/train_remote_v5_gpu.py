# -*- coding: utf-8 -*-
"""魔搭节点端训练脚本 v5 —— GPU 版（2026-10-18 新增）

v5 相对 v4(CPU fp32) 的关键改动:
  1. 设备: 自动优先 cuda; --device 可强制。
  2. 精度: --precision bf16amp(默认) = 主权重 fp32 + autocast(bf16) 计算。
     为什么不直接 bf16 权重: CPU 上 bf16 曾把权重训坏(v4 注释有记), GPU 上
     主流安全做法就是 fp32 主权重 + bf16 autocast, bf16 不需要 GradScaler。
     保守可 --precision fp32 全精度(慢约 2x, 数值最稳)。
  3. 批大小: GPU 上 batch=1x512 严重喂不饱算力 -> 默认 batch=8, accum=4
     (16384 token/步)。显存不够可下调 --batch。
  4. attention: 优先 sdpa。
  5. --tok_dir: 权重目录(NAS ckpt)通常没有 tokenizer, 从 base_tok 兜底。
  6. --bench N: 只测吞吐, 跑 N 步前反向但不 opt.step()、不存盘, 用于上卡验证。

算力预算(0.3B/346.9M, fp32 主权重):
  参数1.4G + 梯度1.4G + AdamW状态2.8G ≈ 5.6G, 加激活约 8~12G => 24G 显存够。
"""
import argparse, os, sys, json, time, glob
import urllib.parse, urllib.request


def upload_file(url, local_path, name=None, retries=4):
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


def _safe_load_tokenizer(tok_dir):
    from transformers import AutoTokenizer
    import json as _json
    tc = os.path.join(tok_dir, "tokenizer_config.json")
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
        return AutoTokenizer.from_pretrained(tok_dir)
    except Exception as e:
        print(f"[tok] 快速 tokenizer 加载失败({e}), 回退 use_fast=False")
        return AutoTokenizer.from_pretrained(tok_dir, use_fast=False)


def _save_ckpt(model, ckpt):
    os.makedirs(ckpt, exist_ok=True)
    try:
        model.save_pretrained(ckpt, safe_serialization=True)
        if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
            raise RuntimeError("no model.safetensors produced")
    except Exception as e:
        print(f"[ckpt] safetensors failed ({repr(e)[:120]}) -> fallback bin")
        model.save_pretrained(ckpt, safe_serialization=False)
        old = os.path.join(ckpt, "model.safetensors")
        if os.path.exists(old):
            os.remove(old)
            print("[ckpt] removed stale model.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/root/r4work/base_next")
    ap.add_argument("--tok_dir", default="")
    ap.add_argument("--run_dir", default="/root/r4work/run_gpu")
    ap.add_argument("--corpus_dir", default="/mnt/workspace/node1/chain/corpus_R7")
    ap.add_argument("--cache", default="/root/c_cache_R7.npy")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=300_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="bf16amp", choices=["bf16amp", "fp32"])
    ap.add_argument("--bench", type=int, default=0, help=">0: 只测吞吐 N 步, 不 step 不存盘")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--upload_url", default="")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device}" + (f" = {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    if device == "cuda":
        print(f"[device] 显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"[precision] {args.precision}")
    print(f"[threads] cpu threads={args.threads}")

    from transformers import AutoModelForCausalLM
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    os.makedirs(args.run_dir, exist_ok=True)

    tok_dir = args.tok_dir or args.base_dir
    print(f"[load] tokenizer from {tok_dir}")
    tok = _safe_load_tokenizer(tok_dir)
    print(f"[load] model from {args.base_dir}")
    kw = {}
    try:
        m = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.float32,
                                                 attn_implementation="sdpa")
    except Exception as e:
        print(f"[load] sdpa 不可用({repr(e)[:80]}) -> 默认 attention")
        m = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.float32)
    m.train()
    m.to(device)
    print(f"[load] params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    cache_path = args.cache
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        print(f"[cache] 复用已有 token cache: {arr.shape[0]} tokens")
    else:
        print(f"[cache] 流式 tokenize, 上限 {args.max_tokens//1_000_000}M tokens")
        files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.jsonl")) +
                       glob.glob(os.path.join(args.corpus_dir, "*.txt")))
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
                        ids = tok("\n".join(buf), return_tensors=None, truncation=False).input_ids
                        buf = []
                        if ids:
                            chunks.append(np.array(ids, dtype=np.int32)); total += len(ids)
                        if total >= args.max_tokens:
                            break
            if total >= args.max_tokens:
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

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps_eff = args.bench if args.bench > 0 else args.steps
    sched = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, total_iters=args.warmup),
        CosineAnnealingLR(opt, T_max=max(1, steps_eff - args.warmup))],
        milestones=[args.warmup])

    tok_per_step = args.batch * args.accum * args.max_len
    print(f"[train] steps={steps_eff} lr={args.lr} batch={args.batch} accum={args.accum} "
          f"max_len={args.max_len} => {tok_per_step} token/步"
          + ("  (BENCH: 不更新权重)" if args.bench > 0 else ""))

    amp = (args.precision == "bf16amp" and device == "cuda")
    t0 = time.time(); loss_acc = 0
    for step in range(1, steps_eff + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            x = batch[:, :-1].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                out = m(input_ids=x, labels=x)
                loss = out.loss / args.accum
            loss.backward()
            loss_acc += loss.item() * args.accum
        if args.bench == 0:
            opt.step()
        sched.step()
        if device == "cuda" and step % 10 == 0:
            torch.cuda.synchronize()
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            tps = step * tok_per_step / max(el, 1e-9)
            mem = f" mem={torch.cuda.max_memory_allocated()/1e9:.1f}G" if device == "cuda" else ""
            print(f"[train] step {step}/{steps_eff} loss={loss_acc/(10*args.accum):.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={el:.0f}s tok/s={tps:.0f}{mem}")
            sys.stdout.flush(); loss_acc = 0
        if args.bench == 0 and step % args.save_every == 0:
            ckpt = os.path.join(args.run_dir, "checkpoint-latest")
            _save_ckpt(m, ckpt)
            open(os.path.join(ckpt, "step.txt"), "w").write(str(step))
            print(f"[ckpt] saved @ step {step}")
            sys.stdout.flush()

    el = time.time() - t0
    tps = steps_eff * tok_per_step / max(el, 1e-9)
    print(f"[SUMMARY] steps={steps_eff} total_t={el:.0f}s tok/s={tps:.0f} "
          f"token={steps_eff*tok_per_step}")
    sys.stdout.flush()

    if args.bench == 0:
        ckpt = os.path.join(args.run_dir, "checkpoint-latest")
        _save_ckpt(m, ckpt)
        open(os.path.join(ckpt, "step.txt"), "w").write(str(steps_eff))
        print(f"[train] DONE -> {ckpt}")
        sys.stdout.flush()
        if args.upload_url:
            _st = os.path.join(ckpt, "model.safetensors")
            _bin = os.path.join(ckpt, "pytorch_model.bin")
            if os.path.exists(_st):
                upload_file(args.upload_url, _st, "gpu_run_model.safetensors")
            elif os.path.exists(_bin):
                upload_file(args.upload_url, _bin, "gpu_run_pytorch_model.bin")
            upload_file(args.upload_url, os.path.join(ckpt, "step.txt"), "gpu_run_step.txt")


if __name__ == "__main__":
    main()
