$ErrorActionPreference = 'Continue'
Write-Host "=== HEALTH ==="
try { $h = Invoke-RestMethod -Uri http://localhost:8081/health -TimeoutSec 8; Write-Host ($h | ConvertTo-Json -Compress) } catch { Write-Host "HEALTH ERR: $($_.Exception.Message)" }
Write-Host "=== MAIL tail 6 (heartbeat?) ==="
Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 6 | ForEach-Object { $o = $_ | ConvertFrom-Json; $t = $o.text; if ($t.Length -gt 200) { $t = $t.Substring(0,200) + "..." }; Write-Host ("[{0}] {1}: {2}" -f [DateTimeOffset]::FromUnixTimeSeconds($o.ts).ToLocalTime().ToString('HH:mm:ss'), $o.from, $t) }
Write-Host "=== CONSUMED tail 6 ==="
Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 6
$now = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Write-Host "NOW_TS=$now ($([DateTimeOffset]::FromUnixTimeSeconds($now).ToLocalTime().ToString('HH:mm:ss')))"
Write-Host "=== TASKS pending (last 3) ==="
Get-Content C:\llama\mail\tasks.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 3 | ForEach-Object { $o = $_ | ConvertFrom-Json; Write-Host ("id={0} ts={1}" -f $o.id, [DateTimeOffset]::FromUnixTimeSeconds($o.ts).ToLocalTime().ToString('HH:mm:ss')) }
