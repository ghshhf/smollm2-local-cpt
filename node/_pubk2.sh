cd /mnt/workspace/node1/chain
if [ ! -f corpus_reserve/zh_r_knowledge.jsonl.pre_wiki ]; then mv corpus_reserve/zh_r_knowledge.jsonl corpus_reserve/zh_r_knowledge.jsonl.pre_wiki; fi
cp corpus_reserve_refined/zh_r_knowledge.jsonl corpus_reserve/zh_r_knowledge.jsonl
echo "=== published ==="
ls -la corpus_reserve/
echo "=== 链状态 ==="
cat done_steps.txt; cat heartbeat 2>/dev/null
