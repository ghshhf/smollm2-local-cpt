Start-Sleep -Seconds 45
Write-Host "=== consumed tail ==="
if (Test-Path C:\llama\mail\consumed.jsonl) {
  Get-Content C:\llama\mail\consumed.jsonl -Tail 3
} else { Write-Host "no consumed.jsonl" }
Write-Host "=== queue (tasks.jsonl tail) ==="
if (Test-Path C:\llama\mail\tasks.jsonl) {
  Get-Content C:\llama\mail\tasks.jsonl -Tail 5
} else { Write-Host "no tasks.jsonl" }
Write-Host "=== node1 heartbeat ==="
if (Test-Path C:\llama\mail\messages.jsonl) {
  Get-Content C:\llama\mail\messages.jsonl -Tail 2
}
Write-Host "=== files on /file ==="
Get-ChildItem C:\llama\files | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String
