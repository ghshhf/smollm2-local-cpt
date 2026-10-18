$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== procs ==='; ps -eo pid,etime,cmd | grep -E 'train_remote_v4|chain_v1' | grep -v grep; echo '=== done ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo '=== log tail ==='; tail -c 1200 /mnt/workspace/node1/chain/chain.log; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=ps2.txt&key=k0u4mail' --data-binary @-"
$body = @{cmd=$cmd; id='ps2'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }
