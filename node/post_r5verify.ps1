$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== R5 staged? ==='; ls -la /mnt/workspace/node1/chain/corpus_r5/ 2>/dev/null || echo 'corpus_r5 MISSING'; echo '=== refresh train script ==='; curl -s -m120 -o /mnt/workspace/node1/chain/train_remote_v4.py 'http://47.99.131.12:8081/file/train_remote_v4.py'; wc -c /mnt/workspace/node1/chain/train_remote_v4.py; grep -c 'retries' /mnt/workspace/node1/chain/train_remote_v4.py; echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo '=== chain.log tail ==='; tail -c 900 /mnt/workspace/node1/chain/chain.log; } | curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=r5verify.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'r5verify-1'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
