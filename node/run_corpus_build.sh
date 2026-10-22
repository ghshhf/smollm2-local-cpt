#!/bin/sh
# node/run_corpus_build.sh —— 在 node1 上原地完成「下载 → 筛选 → 抽句 → 合成」
# 结果全部落在 NAS（/mnt/workspace/node1），不往本机搬。
# 用法: sh run_corpus_build.sh          （可后台: setsid nohup sh run_corpus_build.sh >logs/corpus_build.log 2>&1 &）
set -u
W=${W:-/mnt/workspace/node1}
cd "$W" || exit 1
mkdir -p tools logs data/code_raw corpus_04b data/audio_assets

export POLARIS_HF=${POLARIS_HF:-https://hf-mirror.com}
PY=${PY:-/usr/local/bin/python3}
PIPM="-i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=== [0/5] 环境 $(date -u +%FT%TZ) ==="
$PY -V
echo "HF 源: $POLARIS_HF"
$PY -c "import pyarrow" 2>/dev/null || $PY -m pip install -q pyarrow $PIPM
$PY -c "import edge_tts" 2>/dev/null || $PY -m pip install -q edge-tts $PIPM
$PY -c "import pyarrow,sys;print('pyarrow ok')" 2>&1 | tail -1
$PY -c "import edge_tts,sys;print('edge_tts ok')" 2>&1 | tail -1

echo "=== [1/5] 拉原始代码 (上限 ${RAW_GB:-4}GB) ==="
$PY -u tools/fetch_code_raw.py --raw-cap-gb ${RAW_GB:-4} || echo "[warn] fetch 有失败，继续"

echo "=== [2/5] 筛选 ==="
$PY -u tools/filter_code.py || echo "[warn] filter 失败"

echo "=== [3/5] 抽句 ==="
$PY -u tools/build_audio_corpus.py --stage text --n ${N_SENT:-40000}

echo "=== [4/5] 合成语音 ==="
$PY -u tools/build_audio_corpus.py --stage tts --workers ${WORKERS:-8}

echo "=== [5/5] 盘点 $(date -u +%FT%TZ) ==="
du -sh data/code_raw corpus_04b data/audio_assets 2>/dev/null
wc -l corpus_04b/code_instruct.jsonl corpus_04b/code_raw.jsonl data/audio_assets/manifest.jsonl 2>/dev/null
echo "=== done $(date -u +%FT%TZ) ==="
