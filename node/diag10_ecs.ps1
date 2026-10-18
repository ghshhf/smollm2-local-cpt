Start-Sleep -Seconds 170
Write-Host "=== consumed tail 6 ==="
if (Test-Path C:\llama\mail\consumed.jsonl) { Get-Content C:\llama\mail\consumed.jsonl -Tail 6 }
Write-Host "=== last node1 result (messages) ==="
if (Test-Path C:\llama\mail\messages.jsonl) {
  Get-Content C:\llama\mail\messages.jsonl -Tail 3 | ForEach-Object {
    $j = $_ | ConvertFrom-Json
    $t = $j.text | ConvertFrom-Json
    if ($t.type -eq 'result') { Write-Host ("RESULT " + $t.task_id + " code=" + $t.code); Write-Host $t.out }
    else { Write-Host ("MSG from=" + $j.from + " " + $j.text) }
  }
}
Write-Host "=== peek2.txt ==="
try { (Invoke-WebRequest -Uri 'http://localhost:8081/file/peek2.txt' -TimeoutSec 15).Content } catch { Write-Host "no peek2: $_" }
