# -*- coding: utf-8 -*-
"""把 0.4B 权重与训练语料推到魔搭，供 GPU 实例内网拉取。

🔴 不要调 api.login()——实测它会把 git_token 覆盖成旧值。
"""
import os
import sys
import time

TOKEN = "ms-d4692c9b-86ad-4dc4-9aaa-62189a40b38d"
MODEL_REPO = "hhhlhuk/polaris-04b-depth42"
DATA_REPO = "hhhlhuk/polaris-relay"


def prep():
    os.makedirs(os.path.expanduser("~/.modelscope/credentials"), exist_ok=True)
    with open(os.path.expanduser("~/.modelscope/credentials/git_token"), "w") as f:
        f.write(TOKEN)
    from modelscope.hub.api import HubApi
    return HubApi()


def up_model(api, src):
    try:
        api.create_model(MODEL_REPO, visibility=5, license="Apache License 2.0",
                         chinese_name="北极星 0.4B（42 层，深度扩容保函数）")
        print("[ms] 模型仓库已建：%s" % MODEL_REPO)
    except Exception as e:
        print("[ms] 建仓跳过：%s" % str(e)[:120])

    files = [f for f in os.listdir(src) if not f.startswith("model.safetensors.part")]
    t0 = time.time()
    for fn in sorted(files):
        p = os.path.join(src, fn)
        if not os.path.isfile(p):
            continue
        sz = os.path.getsize(p) / 1e6
        print("[ms] 传 %-32s %8.1f MB ..." % (fn, sz), flush=True)
        t1 = time.time()
        api.upload_file(path_or_fileobj=p, path_in_repo=fn, repo_id=MODEL_REPO)
        el = time.time() - t1
        print("[ms]   ok %.1fs  %.2f MB/s" % (el, sz / max(el, 0.1)), flush=True)
    print("[ms] 模型上传完毕，用时 %.1fs" % (time.time() - t0))


def up_data(api, files):
    try:
        api.create_dataset(DATA_REPO, visibility=5)
        print("[ms] 数据集已建：%s" % DATA_REPO)
    except Exception as e:
        print("[ms] 数据集跳过：%s" % str(e)[:120])

    for p in files:
        if not os.path.exists(p):
            print("[ms] 缺 %s" % p)
            continue
        fn = "corpus_04b/" + os.path.basename(p)
        sz = os.path.getsize(p) / 1e6
        print("[ms] 传 %-38s %8.1f MB ..." % (fn, sz), flush=True)
        t1 = time.time()
        api.upload_file(path_or_fileobj=p, path_in_repo=fn, repo_id=DATA_REPO,
                        repo_type="dataset")
        el = time.time() - t1
        print("[ms]   ok %.1fs  %.2f MB/s" % (el, sz / max(el, 0.1)), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "model"
    api = prep()
    if what == "model":
        up_model(api, "models/polaris-04b-depth42")
    else:
        up_data(api, ["corpus_04b/ufw_zh.jsonl",
                      "corpus_04b/creative_articles_novel.jsonl",
                      "corpus_04b/poetry_zh.jsonl"])
