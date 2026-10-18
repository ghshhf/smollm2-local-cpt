$ts = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$cmd = "W=/mnt/workspace/node1/chain; ECS=http://47.99.131.12:8081; mkdir -p `$W/r5dl && cd `$W/r5dl && for p in part00 part01 part02 part03; do curl -s -m300 -o zh_text_r5.jsonl.`$p `$ECS/file/zh_text_r5.jsonl.`$p; done; SZ=`$(cat zh_text_r5.jsonl.part* | wc -c); EXPECT=230917741; if [ `$SZ -eq `$EXPECT ]; then mkdir -p `$W/corpus_r5 && cat zh_text_r5.jsonl.part* > `$W/corpus_r5/zh_text_r5.jsonl && echo R5_STAGE_OK size=`$(wc -c < `$W/corpus_r5/zh_text_r5.jsonl) lines=`$(wc -l < `$W/corpus_r5/zh_text_r5.jsonl); else echo R5_STAGE_FAIL got=`$SZ expect=`$EXPECT; ls -l; fi; rm -rf `$W/r5dl; { echo '=== refresh train script ==='; curl -s -m120 -o `$W/train_remote_v4.py `$ECS/file/train_remote_v4.py; wc -c `$W/train_remote_v4.py; } | curl -s -X POST 'http://47.99.131.12:8081/upload?name=r5stage2.txt&key=k0u4mail' --data-binary @-"
$body = @{ cmd = $cmd; id = 'r5stage-2'; ts = $ts } | ConvertTo-Json -Compress
Write-Host "POSTING: $body"
try {
  $r = Invoke-RestMethod -Uri http://localhost:8081/task -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("RESULT: " + ($r | ConvertTo-Json -Compress))
} catch {
  Write-Host ("ERR: " + $_.Exception.Message)
}
