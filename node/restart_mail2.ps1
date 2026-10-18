$ErrorActionPreference = 'Continue'
Write-Host "=== BEFORE ==="
Get-Date -Format 'HH:mm:ss'
try { $h = Invoke-RestMethod -Uri http://localhost:8081/health -TimeoutSec 5; Write-Host ("health: " + ($h | ConvertTo-Json -Compress)) } catch { Write-Host "health ERR: $($_.Exception.Message)" }
Write-Host "=== RESTART bridge-mailbox ==="
schtasks /End /TN "bridge-mailbox" 2>&1 | Out-String | Write-Host
Start-Sleep -Seconds 3
schtasks /Run /TN "bridge-mailbox" 2>&1 | Out-String | Write-Host
Start-Sleep -Seconds 8
Write-Host "=== AFTER ==="
try { $h2 = Invoke-RestMethod -Uri http://localhost:8081/health -TimeoutSec 8; Write-Host ("health: " + ($h2 | ConvertTo-Json -Compress)) } catch { Write-Host "health ERR: $($_.Exception.Message)" }
try {
  $t = Invoke-RestMethod -Uri http://localhost:8081/task -TimeoutSec 8
  Write-Host ("task: " + ($t | ConvertTo-Json -Compress -Depth 3))
} catch { Write-Host "task ERR: $($_.Exception.Message)" }
Write-Host "=== MAIL tail 2 ==="
Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 2
Write-Host "=== CONSUMED tail 4 ==="
Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 4
$now = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Write-Host "NOW_TS=$now"
