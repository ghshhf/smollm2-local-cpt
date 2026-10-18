#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r6_finish2.py — 接力: 写拼回器+README -> commit -> push (权重已校验通过)"""
import os, subprocess, sys, time

REPO = "D:/work/_local_ai/train_15b"
DEST = os.path.join(REPO, "backups", "0.2B", "r6_final_20261016")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    sys.path.insert(0, os.path.join(REPO, "node"))
    import r6_fetch_bg as bg  # if __name__ 保护, import 不跑主流程

    with open(os.path.join(DEST, "reassemble.py"), "w", encoding="utf-8") as f:
        f.write(bg.REASSEMBLE_TEMPLATE)
    readme = bg.README_TEMPLATE.replace(
        "- sha256: `4c91e350",
        "- 注意: ckpt_old/step.txt 内容为 `400`（链上局部步数计数），全局进度以 chain.log `R6 complete @ 7100` 为准\n- sha256: `4c91e350")
    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    log("拼回器 + README 写好")

    # 完整性自检: 拼回器真能拼回(输出到临时目录)
    r = subprocess.run([sys.executable, os.path.join(DEST, "reassemble.py"), "/tmp/r6_rt_test"],
                       capture_output=True, text=True)
    log("拼回器自检: " + (r.stdout.strip() or r.stderr.strip()[-100:]))
    if r.returncode != 0:
        log("FATAL: 拼回器自检失败, 停止入账")
        return 2

    r = subprocess.run(["git", "-C", REPO, "add", "backups/0.2B/r6_final_20261016/"],
                       capture_output=True, text=True)
    r = subprocess.run(["git", "-C", REPO, "commit", "-m",
                        "backup(0.2B): R6 封顶正式版权重(done=7100)回捞入账 + 9分片 + 拼回器 + provenance"],
                       capture_output=True, text=True)
    log("commit: " + ((r.stdout or r.stderr).strip()[-150:]))

    pushlog = os.path.join(REPO, "node", "git_push_r6final.log")
    subprocess.Popen(f'(nohup git -C {REPO} push --progress origin HEAD > "{pushlog}" 2>&1 </dev/null &)',
                     shell=True)
    log("push 后台启动 (git_push_r6final.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
