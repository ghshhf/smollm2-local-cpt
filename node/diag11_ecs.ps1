$ErrorActionPreference = 'Continue'
Write-Host "=== HEALTH ==="
try { $h = Invoke-RestMethod -Uri http://localhost:8081/health -TimeoutSec 8; Write-Host ($h | ConvertTo-Json -Compress) } catch { Write-Host "HEALTH ERR: $($_.Exception.Message)" }
Write-Host "=== TASKS tail ==="
Get-Content C:\llama\mail\tasks.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 6
Write-Host "=== CONSUMED tail ==="
Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 8
Write-Host "=== MAIL tail (heartbeats) ==="
Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 4
Write-Host "=== FILES recent ==="
Get-ChildItem C:\llama\files -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String
$now = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Write-Host "NOW_TS=$now"
