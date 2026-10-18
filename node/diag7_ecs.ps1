Start-Sleep -Seconds 150
Write-Host "=== consumed tail ==="
@(Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue) | Select-Object -Last 6
Write-Host "=== last messages ==="
@(Get-Content C:\llama\mail\messages.jsonl -ErrorAction SilentlyContinue) | Select-Object -Last 6
foreach ($n in @('peek.txt','chainlog2.txt','chainlog3.txt')) {
  Write-Host "=== $n ==="
  try { $c = Invoke-RestMethod -Uri "http://localhost:8081/file/$n" -TimeoutSec 5; if ($c) { Write-Host $c } else { Write-Host "(empty)" } }
  catch { Write-Host "(not ready)" }
}