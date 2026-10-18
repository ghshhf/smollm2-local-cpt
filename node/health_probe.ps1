# health_probe.ps1 — 本地 SYSTEM 级定时探活 node1 训练实例。
# 机制: 经 ECS /task 派活 -> node1 跑 probe_health.sh 回显 -> 本地读回传判定存活。
# 判活: 回传含 "TOK=<本次ts>" 且 "alive"。
# 容错: 单次无回可能是 node1 队列积压(重任务期), 故连续 2 次无回才判 DEAD, 避免误报。
# 输出: 最新状态写回 ECS node_health_latest.txt (供对话侧随时读); DEAD 时额外写 alert。
$ECS   = "http://47.99.131.12:8081"
$NODE  = "/mnt/workspace/node1"
$STATE = "D:\work\_local_ai\train_15b\node\health_state.json"
$ALERTDIR = "D:\work\_local_ai\train_15b\node\alerts"

$ts   = [DateTimeOffset]::Now.ToUnixTimeSeconds()
$now  = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
$cmd  = "bash $NODE/probe_health.sh $ts"

function Post-Task($bodyObj){
  try {
    $j = $bodyObj | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "$ECS/task" -Method Post -Body $j -ContentType "application/json" -TimeoutSec 15
    return $true
  } catch { return $false }
}
function Read-Back($name){
  try {
    $r = Invoke-WebRequest -Uri "$ECS/file/$name" -TimeoutSec 15 -UseBasicParsing
    return $r.Content
  } catch { return $null }
}
function Upload-Txt($name, $text){
  try {
    Invoke-RestMethod -Uri "$ECS/upload?name=$name&key=k0u4mail" -Method Post -Body $text -ContentType "text/plain; charset=utf-8" -TimeoutSec 20 | Out-Null
  } catch {}
}

# 1) 派活
$posted = Post-Task @{cmd=$cmd; id="probe_$ts"; ts=[int]$ts}

# 2) 等待 node1 执行 + 回传 (队列积压期最长需 60~75s)
Start-Sleep -Seconds 75
$resp = Read-Back "probe_$ts.txt"

# 3) 判定
$alive = ($null -ne $resp) -and ($resp -match "alive") -and ($resp -match "TOK=$ts")

# 4) 状态机
$state = @{last_ok=$now; fail_streak=0; status="GREEN"; done_steps="?"; detail=""}
if (Test-Path $STATE) {
  try { $state = Get-Content $STATE -Raw | ConvertFrom-Json -AsHashtable } catch {}
}
$done = "?"
if ($resp -match "done_steps=(\d+)") { $done = $Matches[1] }
if ($alive) {
  $state.last_ok    = $now
  $state.fail_streak = 0
  $state.status     = "GREEN"
  $state.done_steps = $done
} else {
  $state.fail_streak = [int]$state.fail_streak + 1
  if ([int]$state.fail_streak -ge 2) {
    $state.status = "DEAD"
    # 写报警
    $msg = "NODE1_DEAD at $now (fail_streak=$($state.fail_streak))`n请到 ModelScope 控制台粘贴 bootstrap 三行重启:`nimport urllib.request`nurllib.request.urlretrieve('http://47.99.131.12:8081/file/bootstrap_all.py?key=k0u4mail','/mnt/workspace/node1/bootstrap_all.py')`nexec(open('/mnt/workspace/node1/bootstrap_all.py').read())"
    $af = "$ALERTDIR\NODE_DEAD_$ts.txt"
    New-Item -ItemType Directory -Force -Path $ALERTDIR | Out-Null
    Set-Content -Path $af -Value $msg -Encoding ASCII
    Upload-Txt "node_dead_$ts.txt" $msg
  } else {
    $state.status = "UNSURE"
  }
}
$state.detail = if ($alive) { $resp.Trim() } else { "no-response" }
Set-Content -Path $STATE -Value (ConvertTo-Json $state -Compress) -Encoding ASCII

# 5) 写回最新状态 (对话侧随时可查)
$latest = "STATUS=$($state.status)`nTS=$now`nDONE_STEPS=$done`nFAIL_STREAK=$($state.fail_streak)`nLAST_OK=$($state.last_ok)"
Upload-Txt "node_health_latest.txt" $latest
Write-Output "probe done: status=$($state.status) done=$done"
