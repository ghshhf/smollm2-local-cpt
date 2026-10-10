#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SmolLM2-135M 全量持续预训练 (continual pretraining, 动底座权重, 非 LoRA)
- 目标: 用本地中文语料 (pretrain_corpus/zh_text.jsonl) 持续训练 0.135B 底座
- 硬件: 本机 CPU (CUDA 不可用), fp32 全精度训练
- 设计: 文档拼接(packing) -> 定长切块 -> next-token CLM
- 可续训: 检测到 output_dir 内有 checkpoint 自动 resume
"""
import argparse, json, math, os, sys, time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------- 数据集: 文档流 packing ----------
class PackedPretrainDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, max_docs=0):
        self.seq_len = seq_len
        self.examples = []
        log("读取语料:", corpus_path)
        # 逐行读文档
        docs = []
        used = 0
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 兼容两种格式: 纯文本行 / jsonl {"text":...} / {"instruction/output":...}
                text = None
                try:
                    d = json.loads(line)
                    if isinstance(d, dict):
                        if "text" in d:
                            text = d["text"]
                        elif "output" in d:
                            text = (d.get("instruction", "") + "\n" + d.get("input", "") + "\n" + d.get("output", ""))
                        elif "content" in d:
                            text = d["content"]
                except Exception:
                    text = line
                if not text:
                    text = line
                if len(text.strip()) >= 8:
                    docs.append(text.strip())
                    used += 1
                    if max_docs and used >= max_docs:
                        break
        if max_docs:
            log(f"  (限制 max_docs={max_docs}) 实际文档数: {len(docs)}")
        else:
            log(f"  文档数: {len(docs)}")
        # 拼接 + 切块
        sep = tokenizer.eos_token or "\n"
        all_ids = []
        for doc in docs:
            ids = tokenizer.encode(doc, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
        log(f"  总 token 数: {len(all_ids):,}")
        n = len(all_ids)
        step = seq_len + 1
        cnt = 0
        for i in range(0, n - seq_len, seq_len):
            chunk = all_ids[i:i + step]
            if len(chunk) < step:
                break
            self.examples.append(torch.tensor(chunk, dtype=torch.long))
            cnt += 1
        self.n = cnt
        log(f"  切块数 (seq_len={seq_len}): {cnt:,}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # 关键: labels 必须 = 完整 chunk, 让 HF CausalLM 内部自己做 shift
        # (若这里先左移一格, HF 会再移一次 -> 目标错位一格, loss 失真)
        chunk = self.examples[idx]
        return {"input_ids": chunk.clone(), "labels": chunk.clone()}

def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="D:/work/_local_ai/models/SmolLM2-135M")
    ap.add_argument("--corpus", default="D:/work/_local_ai/train_15b/pretrain_corpus/zh_text.jsonl")
    ap.add_argument("--output_dir", default="D:/work/_local_ai/train_15b/out_smollm2_135m_pt")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0)  # 0=按 epoch
    ap.add_argument("--max_docs", type=int, default=0)    # 0=全部(验证用, 限制文档数)
    ap.add_argument("--compile", action="store_true", default=False, help="torch.compile 加速(需 MSVC cl, 本机不可用)")
    ap.add_argument("--no_compile", dest="compile", action="store_false")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sched", choices=["cosine", "wsd"], default="cosine",
                    help="学习率调度: cosine(旧) / wsd(warmup-stable-decay)")
    ap.add_argument("--decay_frac", type=float, default=0.2,
                    help="wsd 末期退火占总步数比例")
    ap.add_argument("--min_lr_ratio", type=float, default=0.05,
                    help="wsd 退火终点 lr 相对峰值比例(0=降到0)")
    ap.add_argument("--chunk_seconds", type=int, default=0,
                    help="单轮运行时长上限(秒); 到时存检查点并优雅退出, 配合自动化看门狗实现被杀可续(0=不限)")
    args = ap.parse_args()

    # 写 pidfile 供看门狗检测进程存活
    os.makedirs(args.output_dir, exist_ok=True)
    open(os.path.join(args.output_dir, "train.pid"), "w").write(str(os.getpid()))

    # CPU 优化: 用满物理核心
    try:
        import os as _os
        ncpu = _os.cpu_count() or 4
        torch.set_num_threads(ncpu)
        log(f"CPU 线程数: {ncpu} / {ncpu}")
    except Exception as e:
        log("set_num_threads 跳过:", e)

    device = torch.device("cpu")
    log("设备:", device, "(CUDA 不可用, CPU 训练)")

    # 加载底座
    log("加载 tokenizer / 模型:", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.train()

    # 可选 torch.compile (CPU 上通常显著加速小模型; 失败则降级 eager)
    if args.compile:
        try:
            log("torch.compile 中(首次较慢)...")
            model = torch.compile(model, backend="inductor", dynamic=True)
            log("torch.compile OK")
        except Exception as e:
            log("torch.compile 失败, 降级 eager:", str(e)[:120])

    # 参数量
    nparams = sum(p.numel() for p in model.parameters())
    log(f"底座参数量: {nparams/1e6:.1f}M")

    ds = PackedPretrainDataset(args.corpus, tokenizer, args.seq_len, max_docs=args.max_docs)
    total_steps_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    log(f"每 epoch 优化步数: {total_steps_epoch} (batch={args.batch_size} x accum={args.grad_accum})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # 续训
    start_step = 0
    ckpt_dir = os.path.join(args.output_dir, "checkpoint-latest")
    if args.resume and os.path.isdir(ckpt_dir):
        log("续训: 加载", ckpt_dir)
        sd = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(sd)
        try:
            start_step = int(open(os.path.join(ckpt_dir, "step.txt")).read().strip())
            log("  从 step", start_step, "继续")
        except Exception:
            pass
        # 优化器/调度器状态(分块续训必须恢复, 否则 Adam 动量清零、warmup 重来)
        try:
            opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu")
            log("  已恢复优化器状态")
        except Exception:
            opt_sd = None
        try:
            sch_sd = torch.load(os.path.join(ckpt_dir, "scheduler.pt"), map_location="cpu")
        except Exception:
            sch_sd = None
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        opt_sd = None
        sch_sd = None

    model.to(device)
    scaler = None  # CPU fp32, 不需 autocast

    # 学习率调度: cosine(旧) / wsd(warmup-stable-decay, 现代大模型主流)
    #   wsd 的好处: ① 稳定期可随时中断续训, lr 不因总步数变化而错位
    #               ② 最后 decay_frac 比例步数做退火, 该阶段用最高质量数据收益最大
    max_steps = args.max_steps if args.max_steps > 0 else (total_steps_epoch * args.epochs)
    decay_steps = max(1, int(max_steps * args.decay_frac))
    stable_end = max(args.warmup_steps + 1, max_steps - decay_steps)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.sched == "wsd":
            if step < stable_end:
                return 1.0
            prog = (step - stable_end) / max(1, decay_steps)
            prog = min(1.0, max(0.0, prog))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * (1.0 - prog)
        prog = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # 恢复优化器/调度器状态(必须在 scheduler 创建之后)
    if opt_sd is not None:
        try:
            optimizer.load_state_dict(opt_sd)
        except Exception as e:
            log("优化器状态恢复失败(忽略):", str(e)[:120])
    if sch_sd is not None:
        try:
            scheduler.load_state_dict(sch_sd)
        except Exception as e:
            log("调度器状态恢复失败(忽略):", str(e)[:120])

    log(f"开始训练: max_opt_steps={max_steps}, lr={args.lr}, seq_len={args.seq_len}, accum={args.grad_accum}")
    log_file = os.path.join(args.output_dir, "train_log.txt")
    lf = open(log_file, "a", encoding="utf-8")
    window_loss = 0.0        # 最近一个日志窗口累计的(未除权)loss
    window_micro = 0         # 窗口内 micro-batch 数
    opt_step = start_step     # optimizer 更新次数
    micro = 0                 # 已处理 micro-batch 数
    t0 = time.time()
    chunk_start = time.time()
    model.zero_grad()
    stop = False
    chunk_exit = False
    while not stop:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum   # 归一化, 累加 grad 等价大 batch
            loss.backward()
            micro += 1
            window_loss += outputs.loss.item()
            window_micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                # 短验证时前几步也记, 便于看 loss 曲线; 长训练每 10 步
                if opt_step <= 5 or opt_step % 5 == 0 or opt_step == max_steps:
                    avg = window_loss / max(1, window_micro)
                    el = time.time() - t0
                    lr_now = scheduler.get_last_lr()[0]
                    msg = (f"step {opt_step:>5d}/{max_steps}  loss={avg:.4f}  "
                           f"lr={lr_now:.2e}  {el:.0f}s")
                    log(msg); lf.write(msg + "\n"); lf.flush()
                    window_loss = 0.0; window_micro = 0; t0 = time.time()
                # 定期存检查点
                if opt_step % args.save_steps == 0:
                    save_ckpt(model, tokenizer, args.output_dir, opt_step, args,
                              optimizer=optimizer, scheduler=scheduler)
                # 单轮时长上限: 存盘并退出, 交给看门狗续训(避免被平台回收丢进度)
                if args.chunk_seconds > 0 and (time.time() - chunk_start) >= args.chunk_seconds:
                    save_ckpt(model, tokenizer, args.output_dir, opt_step, args,
                              optimizer=optimizer, scheduler=scheduler)
                    log(f"chunk 时长到({args.chunk_seconds}s), 存检查点 step={opt_step}, 退出等待续训")
                    chunk_exit = True; stop = True; break
                if opt_step >= max_steps:
                    stop = True
                    break
            if opt_step >= max_steps:
                stop = True
                break
        if stop:
            break

    if chunk_exit:
        lf.close()
        log("chunk 结束, 进程退出(自动化看门狗会续训)")
        return

    # 末次不足 grad_accum 的残余梯度也更新一次
    if micro % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        opt_step += 1
    save_ckpt(model, tokenizer, args.output_dir, opt_step, args, final=True,
              optimizer=optimizer, scheduler=scheduler)
    open(os.path.join(args.output_dir, "done.flag"), "w").write(str(opt_step))
    lf.close()
    log("训练完成, 权重保存在:", args.output_dir)

def save_ckpt(model, tokenizer, out_dir, step, args, final=False, optimizer=None, scheduler=None):
    ckpt_dir = os.path.join(out_dir, "checkpoint-latest")
    os.makedirs(ckpt_dir, exist_ok=True)
    # 仅存 fp32 权重(CPU 训练)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    open(os.path.join(ckpt_dir, "step.txt"), "w").write(str(step))
    # 优化器/调度器状态(分块续训保 Adam 动量与 LR 轨迹)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
    if final:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        log("已保存最终权重到", out_dir)
    else:
        log(f"已存检查点 step={step} -> {ckpt_dir}")

if __name__ == "__main__":
    main()
