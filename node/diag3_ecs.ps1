# diag3: ECS mailserver health + node1 heartbeat + new file sizes
try {
  $h = Invoke-RestMethod -Uri http://localhost:8081/health -TimeoutSec 6
  Write-Host "HEALTH: $($h | ConvertTo-Json -Compress)"
} catch { Write-Host "HEALTH_ERR: $_" }

$msgs = @(Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue)
Write-Host "MSG_COUNT: $($msgs.Count)"
Write-Host "LAST_MSGS:"; $msgs | Select-Object -Last 5 | ForEach-Object { Write-Host $_ }

$t = @(Get-Content C:\llama\mail\tasks.jsonl -ErrorAction SilentlyContinue)
$c = @(Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue)
Write-Host "TASKS_PENDING: $($t.Count)  CONSUMED: $($c.Count)"

Write-Host "FILES:"
Get-ChildItem C:\llama\files\node1_chain.sh, C:\llama\files\repair_chain.sh -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "$($_.Name) $($_.Length)" }
