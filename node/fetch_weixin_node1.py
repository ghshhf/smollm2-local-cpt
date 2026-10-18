import json, time, urllib.request
ECS = "http://47.99.131.12:8081"
# node1 直连 GitHub 用 curl 拉分卷 zip(不走 git clone), 解压->切片->成品 jsonl
# 源/流程全定死, node1 不搜索
SCRIPT = r'''
set -e
D=/mnt/workspace/node1/chain/corpus_reserve
T=/tmp/weixin_dl
mkdir -p $T
cd $T
echo "=== 下载分卷 zip (github raw, curl 直拉, 不走 git) ==="
BASE=https://raw.githubusercontent.com/nonamestreet/weixin_public_corpus/master
for f in articles.z01 articles.z02 articles.z03 articles.z04 articles.z05 articles.z06 articles.z07 articles.z08 articles.z09 articles.z10 articles.z11 articles.z12 articles.z13 articles.z14 articles.zip; do
  echo "  -> $f"
  curl -s -L -m600 -o $f "$BASE/$f" || { echo "FAIL $f"; exit 1; }
done
echo "=== 合并 + 解压 ==="
cat articles.z01 articles.z02 articles.z03 articles.z04 articles.z05 articles.z06 articles.z07 articles.z08 articles.z09 articles.z10 articles.z11 articles.z12 articles.z13 articles.z14 articles.zip > combined.zip
ls -la combined.zip
which unzip || apt-get install -y unzip >/dev/null 2>&1 || true
unzip -o -q combined.zip -d $T/ext || { echo "unzip failed"; exit 1; }
ls -la $T/ext | head
# 找 articles.jsonl / articles.txt
SRC=""
for c in articles.jsonl articles.txt; do
  [ -f "$T/ext/$c" ] && SRC="$T/ext/$c" && break
done
if [ -z "$SRC" ]; then
  SRC=$(find $T/ext -name 'articles*' -type f | head -1)
fi
echo "SRC=$SRC"
echo "=== 切片处理 -> zh_r_weixin.jsonl (上限 ~250MB) ==="
python3 - "$SRC" "$D/zh_r_weixin.jsonl" <<'PY'
import sys, os, json
src, out = sys.argv[1], sys.argv[2]
CAP = 250e6
n=0; tot=0
with open(src, encoding='utf-8', errors='ignore') as fh, open(out,'w',encoding='utf-8') as w:
    for line in fh:
        try: o=json.loads(line)
        except: continue
        content=(o.get('content') or '')
        title=(o.get('title') or '')
        name=(o.get('name') or '?')
        if len(content)<80: continue
        rec={"text":(title+"\n"+content).strip(),"src":"weixin:"+str(name),"cat":"modern"}
        s=json.dumps(rec,ensure_ascii=False)+"\n"
        w.write(s); tot+=len(s.encode('utf-8')); n+=1
        if tot>=CAP: break
print("weixin lines:",n,"bytes:",tot)
PY
ls -la "$D/zh_r_weixin.jsonl"
echo "DONE"
'''
CMD = (
    "bash -c " + repr(SCRIPT) + " > /tmp/weixin.log 2>&1; "
    "cat /tmp/weixin.log | curl -s -X POST '" + ECS + "/upload?name=status_weixin.txt&key=k0u4mail' --data-binary @-"
)
body = json.dumps({"cmd": CMD, "id": "fetch-weixin-" + str(int(time.time())), "ts": int(time.time())}).encode()
req = urllib.request.Request(ECS + "/task", data=body, headers={"Content-Type": "application/json"}, method="POST")
try:
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status, r.read().decode()[:150])
except Exception as e:
    print("POST ERR:", e)
