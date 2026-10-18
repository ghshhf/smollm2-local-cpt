W=/mnt/workspace/node1/chain
OUT=/tmp/q_r7r10_out.txt
{
echo "===== $(date '+%m-%d %H:%M') ====="
echo "--- corpus_reserve/ ---"
ls -la $W/corpus_reserve/ 2>/dev/null || echo "(no corpus_reserve)"
echo "--- corpus_r6/ ---"
ls -la $W/corpus_r6/ 2>/dev/null || echo "(no corpus_r6)"
echo "--- 顶层 /mnt/workspace/node1/ ---"
ls -la /mnt/workspace/node1/ 2>/dev/null
echo "--- 找 pretrain_corpus / gutenberg / 已有语料 ---"
ls -d /mnt/workspace/node1/*corpus* /mnt/workspace/*corpus* /mnt/workspace/*gutenberg* 2>/dev/null
echo "--- 已有语料 jsonl (全盘, >5MB) ---"
find /mnt/workspace -maxdepth 4 -name "*.jsonl" -size +5M 2>/dev/null | head -30
echo "--- 行数与首条采样 ---"
for f in $W/corpus_reserve/*.jsonl; do
  [ -e "$f" ] || continue
  n=$(wc -l < "$f" 2>/dev/null)
  echo "== $f  lines=$n  size=$(stat -c %s "$f")"
  head -c 400 "$f"; echo
done
echo "--- 磁盘 ---"
df -h /mnt/workspace 2>/dev/null | tail -2
echo "--- 训练状态 ---"
cat $W/done_steps.txt 2>/dev/null; cat $W/heartbeat 2>/dev/null
} > $OUT 2>&1
cp $OUT /tmp/q_r7r10_report.txt
curl -s -m 60 -F "file=@$OUT" "http://47.99.131.12:8081/upload?name=q_r7r10_report.txt&key=k0u4mail" >/dev/null 2>&1
echo "done"
