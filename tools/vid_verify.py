#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vid_verify.py — 剪辑成片脚本验收（B 线「极简验证」的核心）+ 细粒度质量评分

依据: Timeline-Bench(arXiv 2609.35143) 发现 —— agent 失败 72.9% 只败在质量测试，
      靠静态帧+转录稿感知素材、只查技术缺陷。⇒ 我们反过来: 把技术缺陷做成确定性断言。

v2 升级（做梦 P0a，2026-10-18）:
  - 每项检查除 pass/fail 外，额外给 0~1 的 partial-credit score（MRTPO/SCOPE 式可定位贡献）
  - 汇总 quality 分（0~1），让做梦自产任务对 fast weights 的贡献可定位、可排序
  - 向后兼容：--json 仍返回 pass；退出码 0=PASS 1=FAIL 不变
  - 新增 --min-quality：quality 低于阈值也判 FAIL（做梦态用它挡低质自产任务）

检查项（全部由 ffmpeg 输出解析，无需 ffprobe）:
  1. 可解码 / 时长 / 分辨率 / 帧率
  2. 音轨存在性
  3. 黑场段 (blackdetect)  —— partial: 黑场占比越高分越低
  4. 静音段 (silencedetect) —— partial: 静音占比越高分越低
  5. 可选期望: 期望时长/分辨率（--expect-*）—— 期望项给距离 partial

用法:
  python tools/vid_verify.py <video> [--expect-duration 10] [--ffmpeg path] [--min-quality 0.8]
退出码: 0=PASS  1=FAIL
"""
import os, sys, re, subprocess, argparse, json

DEFAULT_FF = r"D:\work\_local_ai\vidcut\node_modules\ffmpeg-static\ffmpeg.exe"

# 「画面在动」阈值（tools/calib_framedelta.py 实测标定，2026-10-18）：
#   真实 matplotlib 动画 帧间平均灰度差 2.588~2.629
#   首帧循环 4s 的静态对照            0.000
# 取 0.5 作判定线（静态 0 vs 动画 2.6，中间空档足够宽，不会误杀慢速动画）；
# 取 3.0 作满分区（达到即 1.0，低于则线性给分，让"动得更明显"的产物占优）。
MOVE_MIN = 0.5
MOVE_FULL = 2.6


def run(ff, args):
    p = subprocess.run([ff] + args, capture_output=True, text=True,
                       errors="replace", timeout=600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def probe(ff, path):
    rc, out = run(ff, ["-i", path])
    info = {"duration": None, "width": None, "height": None, "fps": None,
            "has_audio": False, "vcodec": None, "acodec": None}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\d+:\d+.*?Video:\s*(\w+).*?,\s*(\d+)x(\d+)", out)
    if m:
        info["vcodec"] = m.group(1); info["width"] = int(m.group(2)); info["height"] = int(m.group(3))
    m = re.search(r"Stream #\d+:\d+.*?Audio:\s*(\w+)", out)
    if m:
        info["has_audio"] = True; info["acodec"] = m.group(1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
    if m:
        info["fps"] = float(m.group(1))
    return info, out


def detect(ff, path, kind):
    """kind: black | silence -> 返回 [(start, end, dur)]"""
    if kind == "black":
        args = ["-i", path, "-vf", "blackdetect=d=0.3:pix_th=0.10", "-f", "null", "-"]
        pat = r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)"
    else:
        args = ["-i", path, "-af", "silencedetect=n=-35dB:d=0.4", "-f", "null", "-"]
        pat = r"silence_start:\s*([\d.-]+).*?silence_end:\s*([\d.-]+).*?silence_duration:\s*([\d.]+)"
    rc, out = run(ff, args)
    hits = [(float(a), float(b), float(c)) for a, b, c in re.findall(pat, out, re.S)]
    return hits


def frame_delta(ff, path, fps=4, size=64):
    """抽帧算相邻帧的平均绝对灰度差（0~255），衡量「画面是不是真的在动」。

    🔴 为什么必须有这一项：现有检查全是「能不能播」（可解码/分辨率/音轨/黑场/静音），
    没有一项看「画面对不对」。而「有音轨」对 matplotlib 产物恒为 0 ⇒
    quality 恒定 = 4/5 = 0.80 ⇒ 精英档案按质量排序等于随机（实测 7 条全 0.80）。
    更糟的是：一张静态 PNG 挂 4 秒也能满分通过那 5 项 —— 做梦的退货闸门是漏的。

    返回 {"frames": n, "mean": 相邻帧差均值, "max": 最大帧差}；解码失败返回 None。
    """
    args = ["-v", "error", "-i", path,
            "-vf", "fps=%d,scale=%d:%d,format=gray" % (fps, size, size),
            "-f", "rawvideo", "-"]
    try:
        p = subprocess.run([ff] + args, capture_output=True, timeout=120)
    except Exception:
        return None
    buf = p.stdout or b""
    fs = size * size
    n = len(buf) // fs
    if n < 2:
        return {"frames": n, "mean": 0.0, "max": 0.0}
    prev, diffs = buf[:fs], []
    for i in range(1, n):
        cur = buf[i * fs:(i + 1) * fs]
        diffs.append(sum(abs(x - y) for x, y in zip(prev, cur)) / float(fs))
        prev = cur
    return {"frames": n, "mean": sum(diffs) / len(diffs), "max": max(diffs)}


def verify(ff, path, a):
    """主验收：返回 (result_dict)。result 含 checks(每项 pass+score+detail) + quality + pass。"""
    info, raw = probe(ff, path)
    blacks = detect(ff, path, "black")
    sils = detect(ff, path, "silence")
    dur = info["duration"] or 0.0

    checks = []
    def chk(name, ok, detail, score, hard=True):
        checks.append({"name": name, "pass": bool(ok), "detail": detail,
                       "score": round(float(score), 4), "hard": bool(hard)})

    # 硬检查（score 0/1，fail 即整段不通过）
    chk("可解码", info["duration"] is not None, "duration=%s" % info["duration"],
        1.0 if info["duration"] is not None else 0.0)
    chk("分辨率有效", bool(info["width"] and info["height"] > 0),
        "%sx%s" % (info["width"], info["height"]),
        1.0 if (info["width"] and info["height"] > 0) else 0.0)
    # 🔧 有音轨默认改为「软检查」：代码生成的可视化(matplotlib)本就无声，不应被硬杀；
    #    剪辑成片场景可加 --require-audio 恢复硬要求。黑场/空片仍由「无黑场」硬检查拦截。
    chk("有音轨", info["has_audio"], "acodec=%s" % info["acodec"],
        1.0 if info["has_audio"] else 0.0, hard=a.require_audio)

    # 期望项：距离 partial（硬 pass 仍看容差，但 score 按偏离比例衰减）
    if a.expect_duration is not None:
        d = info["duration"] or 0
        hard_ok = abs(d - a.expect_duration) <= a.tol
        # partial: 在 4×容差外归零，向内线性衰减
        dev = abs(d - a.expect_duration)
        score = max(0.0, 1.0 - dev / max(1e-6, 4 * a.tol))
        chk("时长符合期望", hard_ok,
            "实际 %.2fs / 期望 %.2fs ±%.2f" % (d, a.expect_duration, a.tol), score)
    if a.expect_width:
        ok = info["width"] == a.expect_width
        chk("宽度符合", ok, "%s vs %s" % (info["width"], a.expect_width),
            1.0 if ok else 0.0)
    if a.expect_height:
        ok = info["height"] == a.expect_height
        chk("高度符合", ok, "%s vs %s" % (info["height"], a.expect_height),
            1.0 if ok else 0.0)

    # 🔴 时间帧验证（2026-10-19 补）：上面只查「时长」，不查「帧率/帧数」。
    #    任务要求「100 帧 10fps」，代码静态写了 frames=100，但 ffmpeg 导出时
    #    fps 设错（如默认 5fps），视频实际 20s/100帧，旧版查不出来。
    #    帧数 = 时长 × 帧率（ffmpeg 解析值，VFR 下为近似），容差放宽避免误杀。
    if a.expect_fps is not None:
        got = info["fps"]
        tol = max(0.5, a.expect_fps * 0.05)
        ok = got is not None and abs(got - a.expect_fps) <= tol
        chk("帧率符合", ok,
            "实际 %.2f fps / 期望 %.2f (±%.2f)" % (got, a.expect_fps, tol),
            1.0 if ok else 0.0)
    if a.expect_frames is not None:
        got = round((info["duration"] or 0) * (info["fps"] or 0))
        tol = max(2, int(a.expect_frames * 0.10))
        ok = abs(got - a.expect_frames) <= tol
        chk("帧数符合", ok,
            "实际约 %d 帧 / 期望 %d (±%d)" % (got, a.expect_frames, tol),
            1.0 if ok else 0.0)

    # 软检查（partial credit，默认不硬 fail；--allow-* 时直接满分不计入缺陷）
    black_dur = sum(x[2] for x in blacks)
    black_score = 1.0 if a.allow_black else max(0.0, 1.0 - (black_dur / dur if dur > 0 else 0.0))
    chk("无黑场", len(blacks) == 0, "%d 段 共%.2fs" % (len(blacks), black_dur),
        black_score, hard=not a.allow_black)
    sil_dur = sum(x[2] for x in sils)
    sil_score = 1.0 if a.allow_silence else max(0.0, 1.0 - (sil_dur / dur if dur > 0 else 0.0))
    chk("无静音", len(sils) == 0, "%d 段 共%.2fs" % (len(sils), sil_dur),
        sil_score, hard=not a.allow_silence)

    # 🔴 画面动态性（软检查，partial credit）：上面 5 项都只看「能不能播」，
    #    一张静态 PNG 挂 4 秒也能满分 —— 做梦的退货闸门在这里是漏的。
    #    阈值实测标定（tools/calib_framedelta.py，用 data/dreams 里 7 个真实产物）：
    #      真实 matplotlib 动画 帧间差 2.588~2.629；首帧循环 4s 的静态对照 = 0.000。
    fd = frame_delta(ff, path)
    if fd is not None:
        if fd["frames"] < 2:
            chk("画面在动", False, "只解出 %d 帧" % fd["frames"], 0.0, hard=False)
        else:
            m = fd["mean"]
            # hard=True：静止画面直接判 FAIL。理由——「没动」不是"质量低一点"，
            # 而是"这不是动画"，让它在 quality 上扣点分再放行毫无意义（默认
            # min-quality=0，扣分根本拦不住）。实测余量安全：真实动画最低 2.588，
            # 判定线 0.5，差 5 倍。
            chk("画面在动", m >= MOVE_MIN,
                "帧间差=%.2f (%d帧)" % (m, fd["frames"]),
                min(1.0, m / MOVE_FULL) if MOVE_FULL > 0 else 1.0, hard=True)

    # 汇总
    quality = sum(c["score"] for c in checks) / len(checks) if checks else 0.0
    hard_fail = any(c["hard"] and not c["pass"] for c in checks)
    pass_ = (not hard_fail) and (quality >= a.min_quality)
    return {"file": path, "info": info,
            "black_segments": blacks, "silence_segments": sils,
            "checks": checks, "quality": round(quality, 4), "pass": bool(pass_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ffmpeg", default=DEFAULT_FF)
    ap.add_argument("--expect-duration", type=float, default=None)
    ap.add_argument("--expect-width", type=int, default=None)
    ap.add_argument("--expect-height", type=int, default=None)
    ap.add_argument("--expect-fps", type=float, default=None,
                    help="期望帧率（时间帧验证，默认不查）")
    ap.add_argument("--expect-frames", type=int, default=None,
                    help="期望总帧数（时间帧验证，默认不查）")
    ap.add_argument("--tol", type=float, default=0.5, help="时长容差(秒)")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="quality 低于此值判 FAIL（做梦态挡低质自产任务，默认0=只看硬检查）")
    ap.add_argument("--allow-black", action="store_true", help="允许黑场(不计缺陷)")
    ap.add_argument("--allow-silence", action="store_true", help="允许静音(不计缺陷)")
    ap.add_argument("--require-audio", action="store_true",
                    help="把「有音轨」恢复为硬检查(剪辑成片场景；代码生成可视化默认不要求)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print("FAIL 文件不存在: %s" % a.video); sys.exit(1)
    ff = a.ffmpeg if os.path.exists(a.ffmpeg) else "ffmpeg"

    result = verify(ff, a.video, a)

    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        i = result["info"]
        print("=" * 66)
        print("验收:", a.video)
        print("=" * 66)
        print("  时长 %.2fs | %sx%s | %s fps | 视频 %s | 音频 %s"
              % (i["duration"] or -1, i["width"], i["height"], i["fps"],
                 i["vcodec"], i["acodec"] or "无"))
        print("  黑场段 %d | 静音段 %d" % (len(result["black_segments"]), len(result["silence_segments"])))
        print("  ---")
        for c in result["checks"]:
            mark = "PASS" if c["pass"] else ("FAIL" if c["hard"] else "WARN")
            print("  [%-4s] %-14s score=%.2f  %s" % (mark, c["name"], c["score"], c["detail"]))
        print("  ---")
        print("  质量分 quality=%.3f  (min=%.2f)" % (result["quality"], a.min_quality))
        print("  结论: %s" % ("PASS ✅" if result["pass"] else "FAIL ❌"))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
