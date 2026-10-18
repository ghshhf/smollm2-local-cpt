$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "cd /mnt/workspace/node1/chain && curl -s -m120 -o base_check.py 'http://47.99.131.12:8081/file/base_check.py' && echo 'bc_size='`$(wc -c < base_check.py) && { echo '=== BCK START ==='; python3 base_check.py /mnt/workspace/node1/chain/ckpt /mnt/workspace/node1/chain/corpus_r4/zh_text_r4.jsonl 300; } 2>&1 | tail -20"
$body = @{ cmd = $cmd; id = 'bcheck-r4done-2'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
