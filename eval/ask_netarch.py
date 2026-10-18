#!/usr/bin/env python3
# 并行向公益API多个模型询问"本地模型联网组件架构"，流式读取 + 捕获所有文本字段 + 重试。
import json, time, threading, urllib.request, os

BASE = "https://api.krisphyn.cc/v1/chat/completions"
KEY = "sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U"

QUESTION = """我们想给一个**本地端侧个人模型**（约0.5B参数、CPU优先、三值BitNet 1.58量化、单机常驻）构建一个"联网自学组件"。核心目标：
- 模型未来要能**自主决定何时上网抓取信息**，把抓取内容消化吸收进自身知识，从而打破"固定本地语料的信息天花板"（让RSI自举循环获得流动信息源）。
- **个人机密数据（owner_corpus、私人记忆、适配权重）绝不外泄**——本地是信任边界。
- 我们已经有一个"信箱机制"（本地信标 node_v5.py，决定何时触发训练/同步/自检），联网组件要和它集成。

请给出一个**模块图级别、可落地的联网组件架构方案**，必须覆盖以下5点，并针对"CPU端可行性"和"本地小模型能否胜任"给出明确判断：

1) **抓取决策层**：模型如何 autotelic（自驱）地决定"我现在该上网查什么"？本地0.5B三值小模型能否直接胜任这个决策，还是需要一个外部强模型当"决策大脑/编排器"？如需外部大脑，如何设计本地↔云的混合协议（机密不出本地的前提下）？

2) **数据管道分层**：抓取 → 清洗 → 去污染/漂移检测(D1) → 入训练池，每一层的职责、用什么技术、CPU端开销如何。特别关心：如何防止外部网页的恶意/隐性注入（subliminal/提示注入）。

3) **机密隔离边界**：明确哪些数据绝对留在本地、哪些可以出本地、用何种机制（如本地差分/仅发embedding/仅发匿名查询）保证 owner_corpus 和个人记忆不泄漏。

4) **安全闸门（防污染）**：外部数据进入训练池前，需要哪些可验证的安全闸门（格式沙箱、语义异常检测、分歧token门控、holdout回测）？给出可操作的判据。

5) **与信箱机制的集成点**：联网组件的触发/暂停/预算控制如何挂到现有信箱信标上？

请给出模块图（ASCII或文字分层），并指出每个模块用什么技术实现、CPU端是否可行。最后用一段话总结"最小可落地MVP"应该先做哪3个模块。"""

MODELS = [
    ("grok-4.7", "grok-4.7"),
    ("gemini-3.8-flash-high", "gemini-3.8-flash-high"),
    ("deepseek-v4-pro-0813", "deepseek-v4-pro-0813"),
    ("Qwen3.8-27B", "qwen3.8-27b"),
]

results = {}
lock = threading.Lock()

def stream_call(model, tag, attempt=1):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": QUESTION}],
        "temperature": 0.7,
        "max_tokens": 3500,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(BASE, data=payload, headers={
        "Authorization": "Bearer " + KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    t0 = time.time()
    try:
        chunks = []
        served = model
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
            raise ValueError("empty content (fields captured: see raw)")
        with lock:
            results[tag] = {"ok": True, "served": served, "content": content, "sec": round(time.time()-t0, 1)}
        print(f"[OK] {tag} served={served} {round(time.time()-t0,1)}s len={len(content)}")
    except StopIteration:
        content = "".join(chunks)
        with lock:
            results[tag] = {"ok": True, "served": served, "content": content, "sec": round(time.time()-t0, 1)}
        print(f"[OK] {tag}(DONE) served={served} {round(time.time()-t0,1)}s len={len(content)}")
    except Exception as e:
        with lock:
            results[tag] = {"ok": False, "error": str(e)[:600], "sec": round(time.time()-t0, 1)}
        print(f"[ERR] {tag} attempt{attempt}: {e}")

def call_with_retry(model, tag, tries=3):
    for i in range(1, tries+1):
        if results.get(tag, {}).get("ok"):
            return
        stream_call(model, tag, i)
        if results.get(tag, {}).get("ok"):
            return
        if i < tries:
            time.sleep(4)

ths = []
for model, tag in MODELS:
    t = threading.Thread(target=call_with_retry, args=(model, tag))
    ths.append(t)
    t.start()
for t in ths:
    t.join()

out_dir = "D:/work/_local_ai/train_15b/eval/netarch_answers"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "raw_answers.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n=== SAVED ===")
for tag, v in results.items():
    print(tag, "OK" if v.get("ok") else "FAIL", v.get("served",""), v.get("sec"), "s len", len(v.get("content","")))
