# -*- coding: utf-8 -*-
"""把音频外设模块推到魔搭（ModelScope）仓库。

不传 Whisper 权重（967MB）：魔搭上 openai/whisper-small 官方仓库已有，
重复传既浪费配额又没有意义。模块仓库只带代码、契约、说明，权重靠 README 指路。
"""
import os
import sys

REPO = "hhhlhuk/polaris-audio-io"
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "modules", "polaris-audio-io")
PRIVATE = 5   # 1=公开, 5=私有


TOKEN = "ms-d4692c9b-86ad-4dc4-9aaa-62189a40b38d"


def token():
    """🔴 不要调用 api.login()：实测它会成功返回，然后把 git_token 文件覆盖成旧值
    （2026-10-19 18:37，写回 zEx6... 20 字节），导致下一次调用报 E3001。
    正确姿势：自己写好文件，直接调 create/upload，SDK 会自己读文件认证。"""
    p = os.path.expanduser("~/.modelscope/credentials/git_token")
    with open(p, "w", encoding="utf-8") as f:
        f.write(TOKEN)
    return TOKEN


def main():
    from modelscope.hub.api import HubApi

    token()
    api = HubApi()
    print("[ms] 已就绪（不走 login）")

    try:
        api.create_model(REPO, visibility=PRIVATE,
                         license="Apache License 2.0",
                         chinese_name="北极星音频外设")
        print("[ms] 仓库已创建：%s" % REPO)
    except Exception as e:
        print("[ms] 创建跳过（多半已存在）：%s" % str(e)[:160])

    files = sorted(os.listdir(FOLDER))
    print("[ms] 待传 %d 个文件：%s" % (len(files), ", ".join(files)))

    api.upload_folder(
        repo_id=REPO,
        folder_path=FOLDER,
        commit_message="polaris-audio-io v0.1.0：Whisper 完整版听觉 + edge-tts/SAPI 双引擎说话，闭环 CER 0.125",
    )
    print("[ms] 上传完成：https://www.modelscope.cn/models/%s" % REPO)


if __name__ == "__main__":
    sys.exit(main())
