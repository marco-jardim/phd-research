#Requires -Version 7.2
# Compila a tese com a mesma sequência do CI (.github/workflows/latex.yml).
# Substitui o uso interativo do build_pdf.bat, com falha em erros de compilação.
# Uso: .\scripts\build_pdf.ps1 [-Quick | -Clean]
# -Quick: apenas a primeira passada; -Clean: remove os arquivos auxiliares.
param([switch]$Quick, [switch]$Clean)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false # Os códigos de saída são tratados explicitamente.
$SrcDir = Join-Path $PSScriptRoot '..\text\latex'
$OutDir = Join-Path $PSScriptRoot '..\text\pdf'

function Invoke-BuildStep {
    param(
        [string]$Step,
        [string]$Command,
        [string[]]$Arguments,
        [switch]$AllowBibtexWarnings
    )

    $OutputFile = Join-Path $env:TEMP "tese-build-$Step.txt"
    Write-Host "[BUILD] $Step..."
    & $Command @Arguments *> $OutputFile
    if ($AllowBibtexWarnings) {
        Get-Content -LiteralPath $OutputFile | ForEach-Object { Write-Host $_ }
        if (Select-String -LiteralPath $OutputFile -Pattern "^error|I couldn't open|I found no" -Quiet) {
            throw '[ERRO] bibtex'
        }
        return
    }
    if ($LASTEXITCODE -ne 0) {
        $ExitCode = $LASTEXITCODE
        $Tail = Get-Content -LiteralPath $OutputFile -Tail 20
        throw "Passo '$Step' falhou (código $ExitCode). Últimas 20 linhas de ${OutputFile}:`n$($Tail -join "`n")"
    }
}

try {
    Push-Location $SrcDir
    try {
        if ($Clean) {
            Write-Host '[CLEAN] Removendo arquivos auxiliares...'
            $Extensions = @(
                'aux', 'log', 'toc', 'lof', 'lot', 'bbl', 'blg', 'brf', 'out',
                'nlo', 'nls', 'ilg', 'idx', 'ind', 'synctex.gz'
            )
            foreach ($Extension in $Extensions) {
                $AuxFile = "tese.$Extension"
                if (Test-Path -LiteralPath $AuxFile) {
                    Remove-Item -LiteralPath $AuxFile -Force
                }
            }
            foreach ($Pattern in @('capitulo*.aux', 'apendice*.aux', 'glossario.aux')) {
                Get-ChildItem -Path . -Filter $Pattern -File | Remove-Item -Force
            }
            Write-Host '[CLEAN] Concluído.'
        } else {
            if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
                throw 'pdflatex não encontrado no PATH. Instale TeX Live ou MiKTeX e tente novamente.'
            }

            $env:max_print_line = '10000'
            $PdfArguments = @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', 'tese.tex')
            Invoke-BuildStep -Step '1-pdflatex' -Command 'pdflatex' -Arguments $PdfArguments
            if ($Quick) {
                Write-Host '[QUICK] Pulando bibtex/makeindex. Apenas 1 passada.'
            } else {
                Invoke-BuildStep -Step '2-bibtex' -Command 'bibtex' -Arguments @('tese') -AllowBibtexWarnings
                if (Test-Path -LiteralPath 'tese.nlo' -PathType Leaf) {
                    Invoke-BuildStep -Step '3-makeindex' -Command 'makeindex' -Arguments @('tese.nlo', '-s', 'nomencl.ist', '-o', 'tese.nls')
                } else {
                    Write-Host '[INFO] Nenhum arquivo .nlo encontrado, pulando nomenclatura.'
                }
                Invoke-BuildStep -Step '4-pdflatex' -Command 'pdflatex' -Arguments $PdfArguments
                Invoke-BuildStep -Step '5-pdflatex' -Command 'pdflatex' -Arguments $PdfArguments
            }

            $Errors = @(Select-String -LiteralPath 'tese.log' -Pattern 'LaTeX Error' -CaseSensitive)
            if ($Errors.Count -gt 0) {
                throw "O log contém erros de LaTeX:`n$($Errors.Line -join "`n")"
            }
            $LogPatterns = @('multiply defined', 'There were multiply-defined labels')
            if (-not $Quick) {
                $LogPatterns += 'Citation .* undefined', 'Reference .* undefined', 'There were undefined references'
            }
            $Undefined = @(Select-String -LiteralPath 'tese.log' -Pattern $LogPatterns -CaseSensitive)
            if ($Undefined.Count -gt 0) {
                throw "Há citações/referências indefinidas ou rótulos duplicados:`n$($Undefined.Line -join "`n")"
            }
            $Summary = @(Select-String -LiteralPath 'tese.log' -Pattern 'Output written on tese\.pdf \([0-9]+ pages?' -CaseSensitive)
            if ($Summary.Count -eq 0) {
                throw 'Resumo de geração do PDF não encontrado no log.'
            }
            $Summary | ForEach-Object { Write-Host $_.Line }
            $Boxes = @(Select-String -LiteralPath 'tese.log' -Pattern 'Overfull|Underfull' -CaseSensitive).Count
            Write-Host "[INFO] Overfull/Underfull: $Boxes"

            New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
            $Destination = Join-Path $OutDir 'tese.pdf'
            Copy-Item -LiteralPath 'tese.pdf' -Destination $Destination -Force
            $Pdf = Get-Item -LiteralPath $Destination
            Write-Host "[OK] PDF gerado com sucesso: $($Pdf.FullName)"
            Write-Host ('Tamanho: {0:N1} KB' -f ($Pdf.Length / 1KB))
        }
    } finally {
        Pop-Location
    }
} catch {
    Write-Host "[FALHA] $($_.Exception.Message)"
    exit 1
}

exit 0
