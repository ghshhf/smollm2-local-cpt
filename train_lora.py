"""
1.5B 模型 LoRA 微调（CPU 版，本机无 CUDA）
============================================
环境：本机 i5-12450H / 16GB / 仅 Intel 核显（无 CUDA）。
策略：bf16 + PEFT LoRA + 梯度检查点，纯 CPU 训练。1.5B 在 16GB 内可跑。

适配 TRL 1.14+ 新 API：配置走 SFTConfig，tokenizer 用 processing_class 传入。

用法（冒烟测试，几十步）：
  python train_lora.py --model_path model/Qwen2.5-1.5B-Instruct \
      --data_path data/sample.jsonl --output_dir out_lora --max_steps 30

正式训练：
  python train_lora.py --model_path model/Qwen2.5-1.5B-Instruct \
      --data_path data/sample.jsonl --output_dir out_lora \
      --num_train_epochs 3 --lora_r 16 --lora_alpha 32 --max_seq_length 1024
"""
import argparse
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="out_lora")
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accumulation", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--logging_steps", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=None,
                   help="覆盖自动计算的保存间隔；不传则按 max_steps//2")
    p.add_argument("--init_adapter", type=str, default=None,
                   help="以某张已有 LoRA 卡的权重为**起点**继续训练（沿用已学到的东西，只喂新数据）。"
                        "区别于 --resume_from_checkpoint：后者连同 optimizer/step 一起复原、会跳过前 N 条样本；"
                        "这里是从基座出发不跳数据，适合「换一份新数据集但不丢已训能力」的场景")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="续训：传入 checkpoint 目录路径（如 out_lora_role/checkpoint-15）")
    return p.parse_args()


def build_text(example):
    if "text" in example and example["text"]:
        return {"text": example["text"]}
    instr = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    if inp:
        text = f"下面是一项任务。\n{instr}\n输入：{inp}\n输出：{out}"
    else:
        text = f"下面是一项任务。\n{instr}\n输出：{out}"
    return {"text": text}


def main():
    args = parse_args()
    t0 = time.time()
    print(f"[info] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  device=cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] 加载模型(bf16, cpu): {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,  # CPU 训练用 fp32：本机 i5-12450H 无 AVX512 BF16，bf16 在 CPU 上是标量回退、单核慢到不可用
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # CPU + 梯度检查点需关闭

    if args.init_adapter:
        # 已有卡片当起点：保留它已学到的全部权重，接下来只在新数据上继续走。
        # 注意 is_trainable=True，否则 peft 会把 adapter 冻住导致什么都学不到。
        from peft import PeftModel as _PeftModel

        print(f"[info] 续练起点(已有卡): {args.init_adapter}")
        model = _PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        print("[info] 配置 LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

    print(f"[info] 读取数据集: {args.data_path}")
    raw = load_dataset("json", data_files=args.data_path, split="train")
    data = raw.map(build_text, remove_columns=raw.column_names)

    # 保存间隔按 1/4 程：颗粒度越细，「随时暂停」的代价越低（之前是 1/2 程，中途杀一次最多白丢半程）
    save_steps = args.save_steps if args.save_steps else (max(args.max_steps // 4, 5) if args.max_steps else 100)
    # epoch 与 max_steps 互斥：传了 epoch 就由 epoch 驱动、max_steps=-1；否则 epoch 给保底值、由 max_steps 控制
    num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else 1
    max_steps = -1 if args.num_train_epochs is not None else args.max_steps
    training_args = SFTConfig(
        output_dir=args.output_dir,
        use_cpu=True,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        bf16=False,  # CPU 路径禁用 bf16（无 AVX512 支持时极慢）
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        gradient_checkpointing=False,  # 恢复跑出 step 7 的原始设置(LoRA 仅训适配器, fp32 内存充足)
        max_length=args.max_seq_length,
        packing=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=lambda ex: ex["text"],
    )

    resume = args.resume_from_checkpoint
    print(f"[info] 开始训练 ...{' 续训自 ' + resume if resume else ''}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] 训练完成，LoRA 已保存到 {args.output_dir}，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
