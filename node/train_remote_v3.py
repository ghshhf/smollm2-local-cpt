# -*- coding: utf-8 -*-
"""魔搭节点端训练脚本 v3 (64 核 CPU, 续训 R4 214M)。

与 v2 的区别:
  - 彻底移除硬编码 OSS AK/SK(明文凭证违规, 且已弃用 OSS)
  - 权重/语料由 shell 侧从 --pull_url(本机公网隧道 / GitHub raw) 预先拉好到本地目录
  - 训完权重通过 --upload_url(POST) 推回(本机隧道 /upload 端点), 不再依赖 OSS

用法(node 上, 权重已就位):
  python node/train_remote_v2.py --base_dir r4_base --corpus_dir corpus \
      --run_dir r4_run --steps 400 --lr 5e-4 [--upload_url https://<tunnel>/upload]
单机无 GPU, 靠 64 核 + bf16 加速。
"""
import argparse, os, sys, json, time, glob, math
import urllib.parse, urllib.request, urllib.error

def upload_file(url, local_path, name=None):
    """把本地文件 POST 到 upload_url(本机隧道 /upload?name=)。"""
    name = name or os.path.basename(local_path)
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            f"{url}?name={urllib.parse.quote(name)}",
            data=data, method="POST",
            headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            print(f"[upload] {name} -> {r.status} {r.read(200).decode('utf-8','ignore')}")
    except Exception as e:
        print(f"[upload] FAIL {name}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/mnt/workspace/node1/r4_base")
    ap.add_argument("--run_dir", default="/mnt/workspace/node1/r4_run")
    ap.add_argument("--corpus_dir", default="/mnt/workspace/node1/corpus")
    ap.add_argument("--cache", default="/mnt/workspace/node1/tok_cache.npy")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=300_000_000)
    ap.add_argument("--upload_url", default="", help="训完推回权重的端点(本机隧道 /upload)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import Dataset, DataLoader
    import torch, torch.nn.functional as F
    n_th = int(os.environ.get("TORCH_THREADS", "0")) or len(os.sched_getaffinity(0)) or os.cpu_count() or 8
    torch.set_num_threads(n_th)
    print(f"[threads] torch.set_num_threads({n_th})")
    import numpy as np
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    os.makedirs(args.run_dir, exist_ok=True)

    print(f"[load] model + tokenizer from {args.base_dir}")
    tok = AutoTokenizer.from_pretrained(args.base_dir)
    model = AutoModelForCausalLM.from_pretrained(args.base_dir)
    model.train(); device = "cpu"
    print(f"[load] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M device={device}")

    # ---- 语料 -> 有上限 memmap ----
    corpus_dir = args.corpus_dir
    base_dir = args.base_dir
    cache_path = args.cache
    max_tokens = args.max_tokens
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        print(f"[cache] 复用已有 token cache: {arr.shape[0]} tokens")
    else:
        from transformers import AutoTokenizer as _AT
        tk = _AT.from_pretrained(base_dir)
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

    dl = DataLoader(RandDS(), batch_size=args.batch_size, num_workers=2)
    it = iter(dl)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, total_iters=args.warmup),
        CosineAnnealingLR(opt, T_max=max(1, args.steps - args.warmup))],
        milestones=[args.warmup])

    print(f"[train] steps={args.steps} lr={args.lr} accum={args.accum} bs={args.batch_size} max_len={args.max_len}")
    t0 = time.time(); opt_step = 0; loss_acc = 0
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
                loss = out.loss / args.accum
            loss.backward()
            loss_acc += loss.item() * args.accum
        opt.step(); sched.step(); opt_step += 1
        if step % 10 == 0:
            print(f"[train] step {step}/{args.steps} loss={loss_acc/10:.4f} lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s")
            sys.stdout.flush(); loss_acc = 0
        if step % 100 == 0:
            ckpt = os.path.join(args.run_dir, "checkpoint-latest")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt, safe_serialization=False)
            open(os.path.join(ckpt, "step.txt"), "w").write(str(step))
            print(f"[ckpt] saved @ step {step}")
            sys.stdout.flush()

    ckpt = os.path.join(args.run_dir, "checkpoint-latest")
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt, safe_serialization=False)
    open(os.path.join(ckpt, "step.txt"), "w").write(str(args.steps))
    print(f"[train] DONE steps={args.steps} total_t={time.time()-t0:.0f}s -> {ckpt}")
    sys.stdout.flush()

    if args.upload_url:
        print("[upload] 推回权重 ...")
        upload_file(args.upload_url, os.path.join(ckpt, "pytorch_model.bin"), "r4_run_pytorch_model.bin")
        upload_file(args.upload_url, os.path.join(ckpt, "step.txt"), "r4_run_step.txt")

if __name__ == "__main__":
    main()
