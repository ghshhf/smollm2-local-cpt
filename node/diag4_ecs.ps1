$t = @(Get-Content C:\llama\mail\tasks.jsonl -ErrorAction SilentlyContinue)
$c = @(Get-Content C:\llama\mail\consumed.jsonl -ErrorAction SilentlyContinue)
Write-Host "TASKS_TOTAL: $($t.Count)  CONSUMED_TOTAL: $($c.Count)"
Write-Host "LAST_TASK:"; if ($t.Count) { $t[-1] }
Write-Host "LAST_CONSUMED:"; if ($c.Count) { $c[-1] }