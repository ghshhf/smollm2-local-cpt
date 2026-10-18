# -*- coding: utf-8 -*-
"""标定「画面在动」阈值：拿历史真实产物算出帧间差分布，再定规则。

用法: PYTHONIOENCODING=utf-8 venv/Scripts/python.exe tools/calib_framedelta.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import vid_verify as vv  # noqa: E402

FF = vv.DEFAULT_FF
print("ffmpeg:", FF, "存在" if os.path.isfile(FF) else "缺失")

vids = []
for d in sorted(os.listdir(os.path.join(ROOT, "data", "dreams"))):
    p = os.path.join(ROOT, "data", "dreams", d, "out.mp4")
    if os.path.isfile(p):
        vids.append(p)
# 外加一个必须被判死的目标：同一视频抽单帧做成静态视频（如果 ffmpeg 在）
print("找到 %d 个视频\n" % len(vids))

print("%-46s %7s %8s %8s" % ("视频", "帧数", "均值差", "最大差"))
rows = []
for p in vids:
    r = vv.frame_delta(FF, p)
    if not r:
        print("%-46s   解码失败" % os.path.basename(os.path.dirname(p)))
        continue
    rows.append(r["mean"])
    print("%-46s %7d %8.3f %8.3f"
          % (os.path.basename(os.path.dirname(p)), r["frames"], r["mean"], r["max"]))

if rows:
    print("\n均值差分布: min=%.3f p50=%.3f max=%.3f"
          % (min(rows), sorted(rows)[len(rows) // 2], max(rows)))

# 造一个静态对照：取第一个视频的首帧生成 4s 静止 mp4，它的帧间差应当是 0
if vids:
    import subprocess
    static = os.path.join(ROOT, "tmp", "static_probe.mp4")
    os.makedirs(os.path.dirname(static), exist_ok=True)
    cmd = [FF, "-y", "-v", "error", "-i", vids[0],
           "-vf", "select=eq(n\\,0),loop=loop=40:size=1,setpts=N/10/TB",
           "-t", "4", "-r", "10", static]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.isfile(static):
        r = vv.frame_delta(FF, static)
        print("\n静态对照(首帧循环4s): frames=%s mean=%.3f  ← 这一项必须被打死"
              % (r["frames"], r["mean"]))
    else:
        print("\n静态对照生成失败:", (p.stderr or "")[:200])
