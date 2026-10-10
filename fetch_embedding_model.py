"""下载 EmbeddingGemma 2（文本模态，270M）到本地，供训练数据语义检索使用。

为什么用它：2026-10-06 Google DeepMind 发布，Apache 2.0，把文本/代码/图片/视频/音频
映射到同一768 维向量空间。我们只用它的**文本编码能力**——给4062 条训练数据建向量，
将来喂数据时按语义相关性检索，而不是随机采样。

为什么走 ModelScope：HuggingFace 直连实测 20 分钟只下了 667MB 就卡死（~550KB/s 且停滞）；
ModelScope 是国内源（记忆里记过：国内拉 HF 走 ModelScope 最快）。

文本模态只需 270M，用 config_kwargs 关闭 vision/audio 编码器即可（740M→270M）。
"""
import os
import sys

os.environ.setdefault("HF_HOME", r"D:/work/_local_ai/hf_cache")
MODEL_ID = "google/embeddinggemma-2"


def main():
    # 优先 ModelScope（国内源），失败再回退 HuggingFace
    use_ms = os.environ.get("USE_HF", "") != "1"
    if use_ms:
        try:
            from modelscope import snapshot_download as ms
            p = ms(MODEL_ID, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"])
            print("[OK] ModelScope 下载完成:", p)
        except Exception as e:
            print("[WARN] ModelScope 失败(%s)，回退 HuggingFace" % e)
            from huggingface_hub import snapshot_download as hf
            p = hf(MODEL_ID, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"])
            print("[OK] HuggingFace 下载完成:", p)
    else:
        from huggingface_hub import snapshot_download as hf
        p = hf(MODEL_ID, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"])
        print("[OK] HuggingFace 下载完成:", p)

    tot = 0
    for root, _, fs in os.walk(p):
        for f in fs:
            if f.endswith(".safetensors"):
                tot += os.path.getsize(os.path.join(root, f))
    print("权重总计: %.2f GB" % (tot / 1e9))
    print("路径:", p)
    # 记下路径供后续脚本读取
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "embedding_model_path.txt"), "w", encoding="utf-8") as fh:
        fh.write(p)


if __name__ == "__main__":
    sys.exit(main())