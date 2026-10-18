$ErrorActionPreference = 'Continue'
Write-Host "=== MAIL: bcheck result ==="
Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue | Select-String -Pattern 'bcheck' | Select-Object -Last 3
Write-Host "=== MAIL: last 3 entries raw ==="
Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 3
Write-Host "=== CONSUMED tail ==="
Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue | Select-Object -Last 5
$now = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Write-Host "NOW_TS=$now"
Write-Host "=== FILES r5 ==="
Get-ChildItem C:\llama\files -ErrorAction SilentlyContinue | Where-Object { $_.Name -like '*r5*' -or $_.Name -like '*verify*' } | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String
