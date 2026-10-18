$ts=[DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd="{ echo '=== now ==='; date '+%m-%d %H:%M:%S'; echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt 2>/dev/null; echo '=== chain.log tail ==='; tail -c 1000 /mnt/workspace/node1/chain/chain.log; echo '=== procs ==='; ps aux | grep -E 'train_remote|chain_v1' | grep -v grep; echo '=== ckpt ==='; ls -la /mnt/workspace/node1/chain/ckpt/ 2>/dev/null | head -8; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status3.txt&key=k0u4mail' --data-binary @-"
$body=@{cmd=$cmd; id='status-3'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING status-3"
Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10 | ForEach-Object { $_ | ConvertTo-Json -Compress }
