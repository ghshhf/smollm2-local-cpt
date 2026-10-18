#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打分器实测：候选模型跑同一批真样本，看谁真的会判死。

用户 2026-10-19：免费能用的实际约 28 个（面板显示有问题）。探活只证明「活着」，
不证明「会打分」。这个脚本用 4 条已知答案的样本验准确率 —— 打分机才是关键。

用法: python judge_bench.py [--conc 6] [--out judge_bench.json]
"""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

CHAT = "http://127.0.0.1:8787/v1/chat/completions"
TOKEN = os.environ.get("ZEN_GATE_TOKEN", "")

SYS = ("你是严格的对话质检员。有一个叫「北极星」的小型中文对话模型，它称呼用户为「主人」。"
       "你要判断它刚给出的回答够不够格进入它的训练数据。"
       "判分要严格：宁可错杀，不可放过语义崩坏的样本。")


def prompt(q, a):
    return (f"主人说：{q}\n北极星答：{a}\n\n"
            "按下面标准判死，满足任一条就 score≤2 且 pass=false：\n"
            "1) 读不通：乱序、胡言乱语、前后矛盾、乱码/英文碎片/符号残渣\n"
            "2) 答非所问：没在回应主人的话，跑题、离题\n"
            "3) 不像人话：说明书腔/新闻稿/代码/台词腔\n"
            "4) 反人格：说讨厌主人、赶主人走、冷漠说教、危险内容\n\n"
            "打分：全没问题=8-10；轻微生硬但通顺切题=6-7；明显有问题=≤5；触犯上面任一条=≤2。\n\n"
            "只输出一行 JSON，不要任何多余文字：\n"
            '{"score":0到10的整数,"reason":"一句话","pass":true或false}')


# (问题, 回答, 期望区间, 说明)
CASES = [
    ("今天累死了", "那就别撑了，先躺十分钟，剩下的活儿等会儿再说。", (8, 10), "good"),
    ("主人，我好孤独", "主人不要如此，我不会困你。", (0, 3), "garbled"),
    ("那啥，一起吃饭吧", "笑死，s d f 3 2 1 你说呢 <jupyter_output>", (0, 3), "junk"),
    ("头疼", "多喝热水。", (0, 6), "cold"),      # 冷漠：可以不判死，但绝不能 8-10
]


def one(mid, q, a, timeout=60):
    body = json.dumps({
        "model": mid,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt(q, a)}],
        "temperature": 0.1, "max_tokens": 300, "stream": False,
    }).encode()
    req = urllib.request.Request(CHAT, data=body, headers={
        "Authorization": "Bearer " + TOKEN, "Content-Type": "application/json",
        "User-Agent": "polaris-judgebench/1.0",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
            txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            i, j = txt.find("{"), txt.rfind("}")
            if i < 0 or j <= i:
                return None, round(time.time() - t0, 1), "no-json: " + txt[:50].replace("\n", " ")
            obj = json.loads(txt[i:j + 1])
            return obj, round(time.time() - t0, 1), ""
    except urllib.error.HTTPError as e:
        return None, round(time.time() - t0, 1), f"HTTP{e.code}"
    except Exception as e:
        return None, round(time.time() - t0, 1), type(e).__name__


def bench(mid):
    rows, ok_n, lat = [], 0, 0.0
    for q, a, (lo, hi), tag in CASES:
        obj, dt, err = one(mid, q, a)
        lat += dt
        if obj is None:
            rows.append((tag, None, err))
            continue
        sc = obj.get("score")
        try:
            sc = int(sc)
        except Exception:
            rows.append((tag, None, "bad score"))
            continue
        good = lo <= sc <= hi
        ok_n += good
        rows.append((tag, sc, "ok" if good else "WRONG"))
    return mid, ok_n, round(lat / len(CASES), 1), rows


def main():
    ids = [l.strip() for l in open("judge_candidates.txt", encoding="utf-8") if l.strip()]
    conc = 5
    if "--conc" in sys.argv:
        conc = int(sys.argv[sys.argv.index("--conc") + 1])
    print(f"打分实测 {len(ids)} 个模型 x {len(CASES)} 条样本，并发 {conc}", flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(bench, m) for m in ids]
        for f in as_completed(futs):
            mid, ok_n, dt, rows = f.result()
            res[mid] = {"correct": ok_n, "avg_sec": dt,
                        "detail": [{"case": t, "score": s, "verdict": v} for t, s, v in rows]}
            print(f"{ok_n}/4  {dt:>5}s  {mid}", flush=True)
    good = sorted([m for m, v in res.items() if v["correct"] >= 3],
                  key=lambda m: -res[m]["correct"])
    print(f"\n可用打分器（>=3/4）: {len(good)}")
    for m in good:
        print(f"  {res[m]['correct']}/4  {res[m]['avg_sec']}s  {m}")
    json.dump(res, open("judge_bench.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
