$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== T ==='; date '+%H:%M:%S'; echo '=== loss tail ==='; grep -E 'step [0-9]+/' /mnt/workspace/node1/chain/chain.log | tail -8; echo '=== procs ==='; ps -eo pid,etime,cmd | grep -E 'train_remote_v4|chain_v1' | grep -v grep | head -4; echo '=== ckpt files ==='; ls -la /mnt/workspace/node1/chain/run_tmp/checkpoint-latest/ 2>/dev/null | head -8; echo '=== done ==='; cat /mnt/workspace/node1/chain/done_steps.txt; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=chainlog8.txt&key=k0u4mail' --data-binary @-"
$body = @{cmd=$cmd; id='dump6'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }
