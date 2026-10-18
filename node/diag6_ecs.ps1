Start-Sleep -Seconds 105
Write-Host "=== consumed tail ==="
@(Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue) | Select-Object -Last 4
Write-Host "=== status2_node1.txt ==="
$tries = 0
while ($tries -lt 12) {
  try {
    $c = Invoke-RestMethod -Uri http://localhost:8081/file/status2_node1.txt -TimeoutSec 5
    if ($c -and $c.Length -gt 50) { Write-Host $c; break }
  } catch { }
  Start-Sleep -Seconds 5; $tries++
}
if ($tries -ge 12) { Write-Host "STATUS2_NOT_READY" }