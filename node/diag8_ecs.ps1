Start-Sleep -Seconds 90
Write-Host "=== consumed tail ==="
@(Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue) | Select-Object -Last 8
Write-Host "=== last messages ==="
@(Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue) | Select-Object -Last 8