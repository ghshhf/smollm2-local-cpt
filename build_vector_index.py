"""用 EmbeddingGemma 2 给训练数据建向量索引，为将来的「语义检索式喂数据」做准备。

背景与动机
----------
现在的训练是**随机顺序**喂 jsonl 的。但同一张卡里的样本难度/主题并不均匀
（比如 card_code_lang 748 条里既有"LRU 缓存"也有"0/1 背包"），随机喂会让
梯度分布平均化。2026 年的主流做法是**检索式喂数据**：先用向量模型把训练集
和一批「任务类型探针」都编码成向量，训练时按探针检索出最相关的 K 条样本喂进去。

EmbeddingGemma 2（Google DeepMind, 2026-10-06, Apache 2.0）提供 768 维统一向量空间，
支持 MRL 截断（768/512/256/128）。本项目只用它的**文本编码器**（270M，
用 config_kwargs 关掉 vision/audio），因为我们的数据全是纯文本。

关键用法（来自官方 model card，别改）
------------------------------------
- 文本必须带 task prompt，用 prompt_name 指定；检索类任务 query 与 document 用不同 prompt。
- 截断维度后**必须重新 L2 归一化**，否则相似度算出来是错的（且不报错，静默降质）。
- 本机 CPU 无 AVX512，**不能用 float16**（激活范围超出 fp16 动态范围）；官方明确要求
  bfloat16 或 float32。本机 bf16 在 CPU 上是标量回退，故用 **float32**。
- CPU 上embedding 比 fp16 快，且与训练同精度无冲突。

用法
----
  python build_vector_index.py --data data/card_reason.jsonl --out vec_reason
  python build_vector_index.py --data data/card_code_lang.jsonl --out vec_lang --truncate 256
  python build_vector_index.py --probe "怎么用 LRU 缓存避免重复计算"
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_FILE = os.path.join(ROOT, "embedding_model_path.txt")
# 各卡的探针(query)：建索引时用它验证检索是否真的召回相关内容
PROBES = {
    "reason": [
        "怎么判断算法的时间复杂度是否够用",
        "什么时候该用单调栈而不是普通栈",
    ],
    "code_lang": [
        "怎么用 LRU 缓存避免重复计算",
        "二分查找的边界条件怎么写才不出错",
    ],
}


def load_model(path, truncate_dim=0):
    from sentence_transformers import SentenceTransformer
    kw = {"vision_config": None, "audio_config": None}   # 只要文本 270M
    if truncate_dim:
        kw["truncate_dim"] = truncate_dim
    m = SentenceTransformer(path, config_kwargs=kw)
    m.eval()
    return m


def encode_docs(model, texts, batch_size=8):
    """文档侧用 Document prompt（title: none | text:）。"""
    return model.encode(texts, prompt_name="Document", batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False)


def encode_queries(model, texts, batch_size=4):
    """查询侧用 SearchQuery prompt（task: search result | query:）。"""
    return model.encode(texts, prompt_name="SearchQuery", batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="训练 jsonl（instruction 字段做文档）")
    ap.add_argument("--out", required=True, help="输出前缀，产出 <out>.npy + <out>_meta.json")
    ap.add_argument("--truncate", type=int, default=256,
                    help="MRL 截断维度；256 官方称近乎无损，768 为全维")
    ap.add_argument("--probe", default=None, help="给一句查询，看它召回了什么")
    ap.add_argument("--limit", type=int, default=0, help="0=全部；调试可限条数")
    args = ap.parse_args()

    if not os.path.isfile(PATH_FILE):
        raise SystemExit("找不到 embedding_model_path.txt，请先跑 fetch_embedding_model.py")
    mpath = open(PATH_FILE, encoding="utf-8").read().strip()
    print("模型路径:", mpath)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    texts = [r["instruction"] for r in rows]
    print("待编码 %d 条（truncate=%d）" % (len(texts), args.truncate))

    model = load_model(mpath, args.truncate)
    vecs = encode_docs(model, texts)

    import numpy as np
    out_np = os.path.join(ROOT, f"{args.out}.npy")
    np.save(out_np, vecs.astype("float32"))
    meta = {"data": args.data, "n": len(texts), "dim": int(vecs.shape[1]),
            "truncate": args.truncate, "model": mpath}
    with open(os.path.join(ROOT, f"{args.out}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("向量已存:", out_np, "形状:", vecs.shape)

    # 探针检索：验证语义召回是否有效
    key = None
    for k in PROBES:
        if k in args.data:
            key = k
            break
    qs = [args.probe] if args.probe else (PROBES.get(key, ["怎么写代码"]) if key else ["怎么写代码"])
    qv = encode_queries(model, qs)
    sims = qv @ vecs.T          # 两侧都已归一化 ⇒ 内积即余弦
    for qi, q in enumerate(qs):
        top = sims[qi].argsort()[::-1][:5]
        print(f"\n=== 查询: {q}")
        for t in top:
            print(f"  {sims[qi][t]:.4f}  {texts[t][:70]}")


if __name__ == "__main__":
    main()