@echo off
REM ==========================================================================
REM  build_pdf.bat — Compila a tese LaTeX e gera PDF em text\pdf
REM
REM  Uso:
REM    scripts\build_pdf.bat            (compilação completa)
REM    scripts\build_pdf.bat --quick    (apenas 1x pdflatex, sem bib/index)
REM    scripts\build_pdf.bat --clean    (remove arquivos auxiliares)
REM ==========================================================================

setlocal enabledelayedexpansion

REM --- Paths -----------------------------------------------------------------
set "REPO_ROOT=%~dp0.."
set "SRC_DIR=%REPO_ROOT%\text\latex"
set "OUT_DIR=%REPO_ROOT%\text\pdf"
set "MAIN=tese"

REM --- Ensure output dir exists ----------------------------------------------
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM --- Parse args ------------------------------------------------------------
set "MODE=full"
if /I "%~1"=="--quick" set "MODE=quick"
if /I "%~1"=="--clean" set "MODE=clean"

REM --- Clean mode ------------------------------------------------------------
if "%MODE%"=="clean" (
    echo [CLEAN] Removendo arquivos auxiliares...
    pushd "%SRC_DIR%"
    for %%x in (aux log toc lof lot bbl blg out nlo nls ilg idx ind synctex.gz) do (
        if exist "%MAIN%.%%x" del /q "%MAIN%.%%x"
    )
    REM Limpar aux de capitulos
    for %%f in (capitulo*.aux apendicea.aux apendiceb.aux glossario.aux) do (
        if exist "%%f" del /q "%%f"
    )
    popd
    echo [CLEAN] Concluido.
    goto :end
)

REM --- Check toolchain -------------------------------------------------------
where pdflatex >nul 2>&1 || (
    echo [ERRO] pdflatex nao encontrado no PATH.
    echo        Instale TeX Live ou MiKTeX e tente novamente.
    exit /b 1
)

REM --- Build -----------------------------------------------------------------
pushd "%SRC_DIR%"

echo.
echo ============================================================
echo  Compilando: %MAIN%.tex
echo  Modo: %MODE%
echo  Saida: %OUT_DIR%\%MAIN%.pdf
echo ============================================================
echo.

REM --- Passo 1: pdflatex (primeira passada) ----------------------------------
echo [1/5] pdflatex (1a passada)...
pdflatex -interaction=nonstopmode -halt-on-error "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] pdflatex falhou na 1a passada. Rodando novamente com log visivel:
    echo.
    pdflatex -interaction=nonstopmode "%MAIN%.tex"
    goto :fail
)

if "%MODE%"=="quick" (
    echo [QUICK] Pulando bibtex/makeindex. Apenas 1 passada.
    goto :copy
)

REM --- Passo 2: bibtex -------------------------------------------------------
echo [2/5] bibtex...
bibtex "%MAIN%" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] bibtex retornou warnings (pode ser normal na primeira compilacao).
)

REM --- Passo 3: makeindex (nomenclatura) -------------------------------------
echo [3/5] makeindex (nomenclatura)...
if exist "%MAIN%.nlo" (
    makeindex "%MAIN%.nlo" -s nomencl.ist -o "%MAIN%.nls" >nul 2>&1
    if errorlevel 1 (
        echo [AVISO] makeindex retornou warnings.
    )
) else (
    echo [INFO] Nenhum arquivo .nlo encontrado, pulando nomenclatura.
)

REM --- Passo 4: pdflatex (segunda passada — resolve refs) --------------------
echo [4/5] pdflatex (2a passada)...
pdflatex -interaction=nonstopmode -halt-on-error "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] pdflatex falhou na 2a passada.
    goto :fail
)

REM --- Passo 5: pdflatex (terceira passada — refs finais + backrefs) ---------
echo [5/5] pdflatex (3a passada)...
pdflatex -interaction=nonstopmode -halt-on-error "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] pdflatex falhou na 3a passada.
    goto :fail
)

REM --- Copy PDF to output dir ------------------------------------------------
:copy
echo.
if exist "%MAIN%.pdf" (
    copy /Y "%MAIN%.pdf" "%OUT_DIR%\%MAIN%.pdf" >nul
    echo [OK] PDF gerado com sucesso:
    echo      %OUT_DIR%\%MAIN%.pdf
    echo.
    REM Mostrar tamanho do arquivo
    for %%A in ("%OUT_DIR%\%MAIN%.pdf") do (
        set "SIZE=%%~zA"
        set /a "SIZE_KB=!SIZE! / 1024"
        echo      Tamanho: !SIZE_KB! KB
    )
) else (
    echo [ERRO] PDF nao foi gerado.
    goto :fail
)

popd
goto :end

:fail
echo.
echo [FALHA] Compilacao falhou. Verifique o log em:
echo         %SRC_DIR%\%MAIN%.log
popd
exit /b 1

:end
endlocal
exit /b 0
