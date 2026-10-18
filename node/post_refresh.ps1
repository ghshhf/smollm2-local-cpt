$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
# 只刷新 W 里的训练脚本副本(不 kill 正在跑的进程), 下一个 chunk 生效
$cmd = "curl -s -o /mnt/workspace/node1/chain/train_remote_v4.py 'http://47.99.131.12:8081/file/train_remote_v4.py' && python3 -m py_compile /mnt/workspace/node1/chain/train_remote_v4.py && echo REFRESH_OK && wc -c /mnt/workspace/node1/chain/train_remote_v4.py | curl -s -X POST 'http://47.99.131.12:8081/upload?name=refresh.txt&key=k0u4mail' --data-binary @-"
$body = @{cmd=$cmd; id='refresh1'; ts=$ts} | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host "RESP: $($r | ConvertTo-Json -Compress)"
} catch { Write-Host "ERR: $_" }
