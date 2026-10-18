$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== PUSH FROM ECS ==='; curl -s -m120 -o /mnt/workspace/node1/chain/base_check.py 'http://47.99.131.12:8081/file/base_check.py'; wc -c /mnt/workspace/node1/chain/base_check.py; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=bcheck3.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'bcheck-r4done-1'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "NOTE: same id, will be skipped if duplicate"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
