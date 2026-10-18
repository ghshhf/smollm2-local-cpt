# zen.deepseek (vendor=deepseek, served=deepseek-v4-flash-free, 73.2s)

The user is asking for a decision on embedding quantization strategy for a 0.2B model (hidden=576) in the context of a Net2Net scaling path (0.1B -> 0.4B -> 0.5B) targeting BitNet b1.58 native ternary weights.
Key constraints:
- Embedding currently kept at fp16 (previous decision).
- 0.2B is near a "flip point" where embedding params % is highest (17.4% at 135M, will be higher at 0.2B? Actually usually embedding % *decreases* as model scales up. The user says "non-embedding ratio monotonically increases with scale, small models have highest embedding %". This is correct. At 135M, emb is 17.4%. At 0.2B (hidden=576), vocab likely same (Chinese vocab added), so emb params fixed ~28M. Total params ~200M. Emb % ~14%. Wait, user says 0.2B is near flip point and "embedding keep fp16 cost highest relatively". This implies memory bandwidth/storage pressure from embedding is significant at this scale because the model is small enough that embedding isn't negligible, but large enough that total size matters for CPU inference.
- Target: CPU <= 15ms/token @ 0.5B. 0.2B is intermediate.
- BitNet b1.58 native ternary QAT.
- No model-generated data pre-0.5B.
- Owner corpus local only.

Questions:
1. Ternarize embedding at 0.2B (accept low-freq precision loss for bandwidth) vs Keep fp16?
2. Partial quantization (mixed precision embedding) implementation traps -> mature solution? LoRA-style: base_fp16 + delta_ternary?

Analysis:
- **Embedding % Trend**: User says 135M: 17.4% emb. 0.2B (hidden=576, L~24? d=576 -> 12*L*d^2. If L=24, 12*24*576^2 ≈ 95M. Emb=28M. Total ~123M. Emb% = 22%. Wait. If hidden=576, usually L=12-16 for "deep narrow". Let's assume L=24 (deep). Non-emb = 12*24*576^2 ≈ 95M. Total = 123M. Emb% = 28/123 = 22.7%. At 0.4B (hidden=1024, L=24? Non-emb=12*24*1024^2=300M. Emb=28M. Emb% = 8.5%. At 0.5B (BitNet, deeper?), Emb% drops further.
- **Bandwidth vs Precision**: On CPU, memory bandwidth is king. 28M fp16 = 56 MB. Ternary (2-bit) = 7 MB. Saving 49 MB weight loading per token generation (embedding lookup is read-only for inference, but training needs write).
- **Inference**: Embedding lookup is gather operation. Bandwidth bound. Ternary helps bandwidth massively.
- **Training (QAT)**: Straight-Through Estimator (STE) for ternary embedding works but low freq tokens get noisy gradients. User mentions "gradient pulls low freq to high freq" issue in mixed precision.
- **LoRA-style (Base fp16 + Delta ternary)**:
    - Forward: `emb = base_fp16[idx] + delta_ternary[idx] * scale`.
    - Backward: `grad_base = grad_output`, `grad_delta = grad_output * scale` (STE).
    - This isolates high-freq stability (base) from low-freq compression (delta).
    - But: Base is fp16 (56MB). Delta is ternary (7MB). Total 63MB > 56MB. Increases model size! Defeats purpose of compression unless base is frozen/quantized too.
    - If base is *frozen* fp16 (pre-trained), only delta trained: Good for fine-tuning, bad for native QAT from scratch/continue training.
    - If base is trained fp16 + delta ternary: Parameter count increases.

Decision Logic:
1.  **0.2B Verdict**: **Ternarize Embedding (Native BitNet style)**.
    - Reason: 0.2B is a *stepping stone* to 0.4B/0.5B. The 0.5B target is native BitNet b1.58 (all ternary weights). Carrying fp16 embedding as a special case breaks the "native" pipeline, adds kernel complexity (mixed precision gather), and creates a distillation gap (0.4B fp16 emb -> 0.5B ternary emb).
    - Bandwidth gain (49MB) is critical for CPU 15ms/token target at 0.5B, must validate kernel at 0.2B.
    - Low-freq token degradation is acceptable for *pre-training/continue-training* phase; tokenizer quality (Chinese vocab added) handles OOV/rare. Distillation to 0.5B will recover.
    - "Flip point" argument: Precisely because emb% is high (22%), bandwidth pressure is highest here. Ternary helps most here.

2.  **Partial Quantization (Mixed) / LoRA-style**: **Reject for Mainline**.
    - Trap 1: Size increase (Base + Delta).
    - Trap 2: Two gather kernels / mixed precision kernel (slow on CPU).
    - Trap 3: Gradient interference requires complex optimizer state separation.
    - Mature Solution: **Standard BitNet b1.58 Embedding QAT (Full Ternary) + Learned Scale per row/block**.
    - Alternative for "Safety Net": **Embedding Distillation**. Train 0.2B ternary. If specific rare tokens fail eval, fix at 0.4B/0.5B via distillation from fp16 teacher (which we have from 0.1B->0.4B path? No, 0.4B is also ternary target? User said 0.4B "Net2Net expand + Chinese vocab + own data". Did not explicitly say 0.4B is ternary. "0.5B = BitNet b1.58 native". Implies 0.4B might be fp16/bf16?).
    - *Correction*: User says "0.1B→0.4B 逐级 Net2Net 保函数扩容... 