param(
    [string]$Source = 'D:\git\phd-research\data\COMPARADORSEMIDENT.csv',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$Local = Join-Path $Root '.local/subamostragem'
New-Item -ItemType Directory -Force -Path $Local | Out-Null
$Script = Join-Path $PSScriptRoot 'subamostragem_treino.py'
$Clock = [Diagnostics.Stopwatch]::StartNew()

function Run-Child([string]$Action, [string]$Ratio, [int]$Seed) {
    $Name = "$Action-$Ratio-$Seed"
    $Arguments = @('"' + $Script + '"', $Action, '--source', '"' + $Source + '"', '--ratio', $Ratio, '--seed', "$Seed")
    $Child = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $Local "$Name.log") `
        -RedirectStandardError (Join-Path $Local "$Name.err.log") -PassThru -NoNewWindow
    # Explicit wait, with a hard three-hour wall-clock limit including all children.
    $Remaining = [Math]::Max(1, [int](10800 - $Clock.Elapsed.TotalSeconds))
    if (-not $Child.WaitForExit($Remaining * 1000)) {
        Stop-Process -Id $Child.Id
        throw 'STOP: three-hour wall-clock limit reached'
    }
    $Child.Refresh()
    if ($Child.ExitCode -ne 0) { throw "Child failed: $Name; see $Local/$Name.err.log" }
}

Run-Child 'prepare' 'original' 1000
Run-Child 'worker' 'original' 1000
$Reference = Get-Content (Join-Path $Local 'replica_original_1000.json') -Raw | ConvertFrom-Json
# A measured conservative projection: every replica costs as much as full training.
$Projection = $Reference.seconds * 41
@{ reference_seconds = $Reference.seconds; projected_seconds = $Projection; method = '41 times measured unsampled reference (conservative)' } |
    ConvertTo-Json | Set-Content (Join-Path $Local 'projection.json')
if ($Projection -gt 10800) { throw "STOP: projection exceeds 3 hours ($Projection seconds)" }
foreach ($Ratio in @('1', '5', '20', '50')) {
    foreach ($Seed in 1000..1009) {
        if ($Clock.Elapsed.TotalSeconds -gt 10800) { throw 'STOP: three-hour limit' }
        Run-Child 'worker' $Ratio $Seed
    }
}
Run-Child 'aggregate' 'original' 1000
