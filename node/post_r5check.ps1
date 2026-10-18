$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== R5 corpus check ==='; ls -la /mnt/workspace/node1/chain/ | grep -i corpus; echo '--- corpus_R5 direct ---'; ls -la /mnt/workspace/node1/chain/corpus_R5/ 2>/dev/null || echo 'corpus_R5 MISSING'; echo '--- corpus_r5 direct ---'; ls -la /mnt/workspace/node1/chain/corpus_r5/ 2>/dev/null || echo 'corpus_r5 MISSING'; echo '--- find any r5 jsonl ---'; find /mnt/workspace/node1/chain -maxdepth 2 -iname '*r5*' 2>/dev/null; echo '--- root r4work ---'; ls -la /root/r4work/corpus/ 2>/dev/null || echo 'no /root/r4work/corpus'; echo '=== upload probe ==='; echo hello-from-node1 | curl -s -m30 -X POST 'http://47.99.131.12:8081/upload?name=probe1.txt&key=k0u4mail' --data-binary @- -w 'HTTP=%{http_code}'; echo; echo '=== done_steps now ==='; cat /mnt/workspace/node1/chain/done_steps.txt; } | curl -s -m120 -X POST 'http://47.99.131.12:8081/upload?name=r5check.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'r5check-1'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
