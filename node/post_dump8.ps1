$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "{ echo '=== TIME ==='; date '+%m-%d %H:%M:%S'; echo '=== PROCS ==='; ps aux | grep -E 'train_remote_v4|chain_v1' | grep -v grep; echo '=== done_steps ==='; cat /mnt/workspace/node1/chain/done_steps.txt; echo '=== ckpt files ==='; ls -la /mnt/workspace/node1/chain/ckpt/ 2>/dev/null | head; echo '=== run_tmp ==='; ls -la /mnt/workspace/node1/chain/run_tmp/ 2>/dev/null | head; echo '=== chain.log tail 3000 ==='; tail -c 3000 /mnt/workspace/node1/chain/chain.log; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=status_r5.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'dump-r5-1'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
