#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网关模型探活：对 /v1/models 全量发一次极短请求，标出真正能用的。

用户 2026-10-19：面板显示有问题，实际约 28 个免费模型可用 —— 以实测为准。
用法: python probe_models.py [--conc 6] [--out probe_models.json]
"""
import json, os, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8787/v1/models"
CHAT = "http://127.0.0.1:8787/v1/chat/completions"
TOKEN = os.environ.get("ZEN_GATE_TOKEN", "")
SKIP_SUFFIX = (" (light)", " (deep)")


def list_models():
    req = urllib.request.Request(URL, headers={"Authorization": "Bearer " + TOKEN})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
    return [m["id"] for m in d.get("data", []) if not m["id"].endswith(SKIP_SUFFIX)]


def probe(mid, timeout=45):
    body = json.dumps({
        "model": mid,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(CHAT, data=body, headers={
        "Authorization": "Bearer " + TOKEN,
        "Content-Type": "application/json",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            dt = time.time() - t0
            try:
                d = json.loads(raw)
                ch = (d.get("choices") or [{}])[0]
                msg = (ch.get("message") or {})
                txt = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                return (mid, True, round(dt, 1), txt[:20].replace("\n", " "))
            except Exception:
                return (mid, True, round(dt, 1), raw[:40])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:120]
        return (mid, False, round(time.time() - t0, 1), f"HTTP{e.code} {body}")
    except Exception as e:
        return (mid, False, round(time.time() - t0, 1), type(e).__name__)


def main():
    conc = 6
    out = "probe_models.json"
    if "--conc" in sys.argv:
        conc = int(sys.argv[sys.argv.index("--conc") + 1])
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    ids = list_models()
    print(f"探活 {len(ids)} 个模型，并发 {conc}", flush=True)
    res = {}
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(probe, m): m for m in ids}
        for f in as_completed(futs):
            mid, ok, dt, note = f.result()
            res[mid] = {"ok": ok, "sec": dt, "note": note}
            done += 1
            if ok:
                print(f"[{done}/{len(ids)}] OK   {mid} {dt}s {note!r}", flush=True)
            elif done % 10 == 0:
                print(f"[{done}/{len(ids)}] ...", flush=True)
    good = [m for m, v in res.items() if v["ok"]]
    print(f"\n可用 {len(good)} / {len(ids)}")
    for m in good:
        print("  " + m)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> " + out)


if __name__ == "__main__":
    main()
