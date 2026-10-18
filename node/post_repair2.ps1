$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "cd /mnt/workspace/node1/chain && curl -s -o repair_chain.sh 'http://47.99.131.12:8081/file/repair_chain.sh' && sh repair_chain.sh"
$body = @{cmd=$cmd; id='repair-2'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }