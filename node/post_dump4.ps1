$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== T ==='; date '+%H:%M:%S'; echo '=== log tail ==='; tail -c 1800 /mnt/workspace/node1/chain/chain.log; echo; echo '=== done ==='; cat /mnt/workspace/node1/chain/done_steps.txt; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=chainlog6.txt&key=k0u4mail' --data-binary @-"
$body = @{cmd=$cmd; id='dump4'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }
