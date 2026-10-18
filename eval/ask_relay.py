#!/usr/bin/env python3
# 可复用 · 跨模型并行问答（流式 + 重试 + 多轮）。
# 用法: python ask_relay.py <question_md> <out_dir> [model1,model2,...]
import json, time, threading, urllib.request, os, sys

BASE = "https://api.krisphyn.cc/v1/chat/completions"
KEY = "sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U"

DEFAULT_MODELS = [
    ("gemini-3.8-flash-high", "gemini"),
    ("deepseek-v4-pro-0813", "deepseek"),
    ("Qwen3.8-27B", "qwen"),
]

def main():
    if len(sys.argv) < 3:
        print("usage: ask_relay.py <question_md> <out_dir> [model,model,...]")
        sys.exit(1)
    qfile, out_dir = sys.argv[1], sys.argv[2]
    QUESTION = open(qfile, encoding="utf-8").read().strip()
    if len(sys.argv) > 3:
        models = [(m, m.split("/")[-1].split("-")[0].lower()) for m in sys.argv[3].split(",")]
    else:
        models = DEFAULT_MODELS
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    lock = threading.Lock()

    def stream_call(model, tag, attempt=1):
        payload = json.dumps({
            "model": model, "messages": [{"role": "user", "content": QUESTION}],
            "temperature": 0.7, "max_tokens": 3500, "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(BASE, data=payload, headers={
            "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
            "Accept": "text/event-stream"})
        t0 = time.time()
        try:
            chunks, served = [], model
            with urllib.request.urlopen(req, timeout=600) as r:
                buf = b""
                while True:
                    piece = r.read(4096)
                    if not piece:
                        break
                    buf += piece
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(b"data:"):
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                raise StopIteration
                            try:
                                obj = json.loads(data)
                                if obj.get("model"):
                                    served = obj["model"]
                                for c in obj.get("choices", []):
                                    d = c.get("delta", {})
                                    for k in ("content", "reasoning_content", "reasoning"):
                                        if d.get(k):
                                            chunks.append(d[k])
                            except Exception:
                                pass
            content = "".join(chunks)
            if not content.strip():
                raise ValueError("empty content")
            with lock:
                results[tag] = {"ok": True, "served": served, "content": content, "sec": round(time.time()-t0, 1)}
            print(f"[OK] {tag} served={served} {round(time.time()-t0,1)}s len={len(content)}")
        except StopIteration:
            content = "".join(chunks)
            with lock:
                results[tag] = {"ok": True, "served": served, "content": content, "sec": round(time.time()-t0, 1)}
            print(f"[OK] {tag}(DONE) {round(time.time()-t0,1)}s len={len(content)}")
        except Exception as e:
            with lock:
                results[tag] = {"ok": False, "error": str(e)[:600], "sec": round(time.time()-t0, 1)}
            print(f"[ERR] {tag} attempt{attempt}: {e}")

    def with_retry(model, tag, tries=3):
        for i in range(1, tries+1):
            if results.get(tag, {}).get("ok"):
                return
            stream_call(model, tag, i)
            if results.get(tag, {}).get("ok"):
                return
            if i < tries:
                time.sleep(4)

    ths = [threading.Thread(target=with_retry, args=(m, t)) for m, t in models]
    for t in ths: t.start()
    for t in ths: t.join()

    raw = os.path.join(out_dir, "raw_answers.json")
    json.dump(results, open(raw, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for tag, v in results.items():
        if v.get("ok"):
            open(os.path.join(out_dir, f"ans_{tag}.md"), "w", encoding="utf-8").write(
                f"# {tag} (served: {v.get('served')})\n\n{v['content']}")
    print("\n=== SAVED ===", raw)
    for tag, v in results.items():
        print(tag, "OK" if v.get("ok") else "FAIL", v.get("served",""), v.get("sec"), "s len", len(v.get("content","")))

if __name__ == "__main__":
    main()
