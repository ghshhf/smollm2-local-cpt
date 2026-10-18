$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== TIME ==='; date '+%m-%d %H:%M:%S'; echo '=== PROCS ==='; ps aux | grep -E 'train_remote_v4|chain_v1' | grep -v grep || echo 'NO_TRAIN_PROC'; echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; echo '=== corpus_r5 ==='; ls -la /mnt/workspace/node1/chain/corpus_r5/ 2>/dev/null || echo 'corpus_r5 MISSING'; echo '=== corpus_R5 link ==='; ls -la /mnt/workspace/node1/chain/ | grep -i corpus; echo '=== ckpt ==='; ls -la /mnt/workspace/node1/chain/ckpt/ 2>/dev/null | head -8; echo '=== cache on NAS ==='; ls -la /mnt/workspace/node1/chain/*.npy 2>/dev/null || echo 'no npy on NAS'; echo '=== /root ==='; ls /root/r4work/ 2>/dev/null || echo '/root/r4work empty'; echo '=== chain.log tail 1200 ==='; tail -c 1200 /mnt/workspace/node1/chain/chain.log; echo '=== train script ver ==='; wc -c /mnt/workspace/node1/chain/train_remote_v4.py; } | curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=status2_r5.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'status-r5boot-1'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
