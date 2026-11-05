param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Local = Join-Path $Root '.local/subamostragem'
New-Item -ItemType Directory -Force -Path $Local | Out-Null
$Head = git -C $Root rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve HEAD' }
git -C $Root ls-files --error-unmatch tests/test_subamostragem.py | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Commit the new tests before recording HEAD verification' }
$Paths = @('gzcmd/subamostragem.py', 'scripts/subamostragem_treino.py', 'scripts/subamostragem_treino.ps1', 'scripts/verificar_subamostragem.ps1', 'tests/test_subamostragem.py')
git -C $Root diff --exit-code HEAD -- @Paths
if ($LASTEXITCODE -ne 0) { throw 'Own paths differ from HEAD; verification would be misattributed' }

function Run-Pytest([string]$Directory, [string]$Label) {
    $InitialStatus = @(git -C $Directory status --short)
    $Log = Join-Path $Local "pytest-$Label.log"
    @("HEAD: $Head", 'COMMAND: python -m pytest tests -q -rs', 'INITIAL git status --short:', $InitialStatus, 'PYTEST:') |
        Set-Content $Log
    $Process = Start-Process -FilePath $Python -ArgumentList @('-m', 'pytest', 'tests', '-q', '-rs') `
        -WorkingDirectory $Directory -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $Local "pytest-$Label.stdout.log") `
        -RedirectStandardError (Join-Path $Local "pytest-$Label.err.log")
    $Process.WaitForExit()
    $Process.Refresh()
    $Code = $Process.ExitCode
    $CompletedHead = git -C $Directory rev-parse HEAD
    $Dirty = @(git -C $Directory status --porcelain)
    Get-Content (Join-Path $Local "pytest-$Label.stdout.log") | Add-Content $Log
    Get-Content (Join-Path $Local "pytest-$Label.err.log") | Add-Content $Log
    @{ HEAD = $Head; completed_HEAD = $CompletedHead; initial_status = $InitialStatus; completed_status = $Dirty; clean_commit_evidence = ($Label -eq 'clean-clone' -and -not $InitialStatus.Count -and -not $Dirty.Count -and $CompletedHead -eq $Head); command = 'python -m pytest tests -q -rs'; directory = $Directory; exit_code = $Code; completed_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content (Join-Path $Local "pytest-$Label-HEAD.json")
    if ($Code -ne 0) { throw "pytest failed ($Label), exit $Code; see $Local/pytest-$Label.log" }
    if ($Label -eq 'clean-clone' -and ($CompletedHead -ne $Head -or $Dirty.Count)) { throw 'Clean clone changed during pytest; cannot attribute run to the commit' }
}

Run-Pytest $Root 'windows'
$Clone = Join-Path $Local ("clone-" + $Head.Substring(0, 12))
if (Test-Path $Clone) { throw "Clone destination already exists: $Clone" }
git clone --no-hardlinks --single-branch --branch pq/arranjo-corte $Root $Clone
if ($LASTEXITCODE -ne 0) { throw 'Clean clone failed' }
$CloneHead = git -C $Clone rev-parse HEAD
if ($CloneHead -ne $Head) {
    git -C $Clone checkout --detach $Head
    if ($LASTEXITCODE -ne 0) { throw 'Cannot checkout verification HEAD in clone' }
}
if (git -C $Clone status --porcelain) { throw 'Clone is not clean' }
$OldDataRoot = $env:PHD_RESEARCH_DATA_ROOT
try {
    # Explicitly exercise the portable suite without borrowing private inputs.
    $env:PHD_RESEARCH_DATA_ROOT = $Clone
    Run-Pytest $Clone 'clean-clone'
} finally {
    $env:PHD_RESEARCH_DATA_ROOT = $OldDataRoot
}
