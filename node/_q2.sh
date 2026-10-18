cd /mnt/workspace/node1/chain
{ echo "#### TIME"; date '+%m-%d %H:%M:%S'
  echo "#### HEARTBEAT"; cat heartbeat 2>/dev/null
  echo "#### DONE"; cat done_steps.txt 2>/dev/null
  echo "#### CHAIN_ALIVE"; P=$(cat chain.pid 2>/dev/null); { [ -n "$P" ] && kill -0 "$P" 2>/dev/null && echo "chain ALIVE pid=$P"; } || echo "chain DEAD"
  echo "#### TRAINER"; (ps -eo pid,etime,args | grep -v grep | grep -q "train_remote_v4" && echo "trainer ALIVE") || echo "trainer DEAD"
  echo "#### STEPS(last10)"; grep -E "^step |\[train\] step" chain.log 2>/dev/null | tail -10
  echo "#### LOG_TAIL"; tail -6 chain.log 2>/dev/null
  echo "#### CKPT"; ls -la ckpt/ 2>/dev/null | tail -6
  echo "#### CORPUS"; ls corpus_r5/ 2>/dev/null | head -3; echo "files=$(ls corpus_r5/ 2>/dev/null | wc -l)"
} 2>&1 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=now.txt&key=k0u4mail' --data-binary @-
