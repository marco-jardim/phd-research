# Sequential modes: no more than four concurrent Anthropic requests overall.
# Run this script through an unlimited-time background executor.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$started = [DateTimeOffset]::UtcNow
$out = 'data/sprint3b/anthropic_sem_mo'
$firstRunStarted = $started
if (Test-Path "$out/run_vigilancia.log") {
    $firstLine = Get-Content "$out/run_vigilancia.log" -TotalCount 1
    $parsedStart = [DateTimeOffset]::MinValue
    if ([DateTimeOffset]::TryParse(($firstLine -split ' ')[-1], [ref]$parsedStart)) {
        $firstRunStarted = $parsedStart
    }
}
foreach ($mode in @('vigilancia', 'confirmacao')) {
    $log = "$out/run_$mode.log"
    "START_OR_RESUME $mode $([DateTimeOffset]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
    & python -u scripts/run_exp2_sem_model_outputs.py --modo $mode --modelo claude-haiku-5-5 --pares "data/sprint3b/anthropic/exp2_llm_sync_$mode.json" --saida "$out/exp2_llm_sync_$mode.json" --workers 4 >> $log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Runner failed for $mode (see sanitized log)" }
    "END $mode $([DateTimeOffset]::UtcNow.ToString('o'))" | Out-File $log -Append -Encoding utf8
}
$finished = [DateTimeOffset]::UtcNow
@{ started_at = $firstRunStarted.ToString('o'); resumed_at = $started.ToString('o'); finished_at = $finished.ToString('o'); wall_seconds = ($finished - $firstRunStarted).TotalSeconds; resume_wall_seconds = ($finished - $started).TotalSeconds; wall_includes_interruption = $true; workers = 4; modes_sequential = $true; smoke_successes_reused = $true } | ConvertTo-Json | Out-File "$out/execucao.json" -Encoding utf8NoBOM
& python -u scripts/run_exp2_sem_model_outputs.py --summarize-only
if ($LASTEXITCODE -ne 0) { throw 'Summary/audit failed' }
