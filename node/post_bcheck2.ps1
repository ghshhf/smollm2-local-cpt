$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "curl -s -o /tmp/base_check.py 'http://47.99.131.12:8081/file/base_check.py' && { echo '=== corpus dir ==='; ls -la /mnt/workspace/node1/chain/corpus_r4/; echo '=== sample lines ==='; head -c 600 /mnt/workspace/node1/chain/corpus_r4/zh_text_r4.jsonl; echo; echo '=== base_next ==='; ls -la /root/r4work/base_next/; echo '=== CE on node1 corpus ==='; python3 /tmp/base_check.py /mnt/workspace/node1/chain/r4base_std /mnt/workspace/node1/chain/corpus_r4/zh_text_r4.jsonl 200; } 2>&1 | tail -c 4000 | curl -s -X POST 'http://47.99.131.12:8081/upload?name=bcheck2.txt&key=k0u4mail' --data-binary @-"
$body = @{cmd=$cmd; id='bcheck2'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }
