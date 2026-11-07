# Suíte e build de teste após W10a/W10b — pq/banca-4

## Execução

- [x] Conferir histórico, status inicial e processos da W10a.
- [x] Executar a suíte completa e classificar as falhas.
- [x] Corrigir apenas contratos textuais obsoletos, EOL e proveniência.
- [x] Compilar PDF de teste e auditar o log e os PDFs oficiais.
- [x] Registrar evidências e commits locais, sem push.

HEAD inicial: `2ef1861` (`tests: confere texto W10b contra agregados publicados`).
`git status --short` inicial vazio. A consulta `Get-CimInstance Win32_Process`,
filtrada por `pq-banca-4` e `pytest` na linha de comando, não encontrou processo
anterior da W10a; portanto não houve PID anterior pelo qual aguardar.

Procedimento de build consultado em
`D:\git\phd-research\docs\plans\RELATORIO_BUILD_OFICIAL_2026-11-07.md`.
Foi usado `scripts/build_pdf.py` com `--jobname` e `--outname` de teste, sem
alterar os PDFs oficiais nem a prosa LaTeX.

## Ambiente e histórico inicial

Python `C:\Users\Marquinho\miniconda3\python.exe`, versão `3.13.11`;
TeX Live 2024 (`C:\texlive\2024\bin\windows`); `core.autocrlf=input`.

```text
2ef1861 tests: confere texto W10b contra agregados publicados
b7a7e3c slides: qualifica recomendacao da fila e respostas da defesa
927a30e text: integra robustez da fusao e limites da referencia
328aa18 tests: conferir numeros e contratos editoriais da W10a
f0da70b text: delimitar fusao e erro da referencia no Estudo 4
8c7486c tables: documentar robustez da fusao e revisores no Estudo 4
ed7a8aa analysis: agregado do probe com quatro modelos da Anthropic
ec50280 analysis: sensibilidade das conclusoes ao erro da referencia
a1fb410 chore: normalizar eol dos artefatos da fusao
951a48b analysis: robustez da fusao de filas (bootstrap, faixa, topo)
e229723 tests: verifica rastreabilidade numerica e bibliografica W9a
b33a513 bib: acrescenta nove referencias verificadas para W9
```

## Build de teste

Comando executado em segundo plano por `Start-Process -WindowStyle Hidden -Wait`:

```text
python scripts/build_pdf.py --jobname tese_banca4_2ef1861 --outname tese_banca4_2ef1861
```

Exit code `0`; estabilização em quatro passadas. Logs do processo:
`$env:TEMP\build_banca4_w10.log` e `$env:TEMP\build_banca4_w10.err.log`.
Log final LaTeX: `text/latex/tese_banca4_2ef1861.log`.

| Artefato de teste | Páginas físicas | Bytes | SHA-256 |
| :-- | --: | --: | :-- |
| `text/latex/tese_banca4_2ef1861.pdf` | 268 | 1283456 | `6eacc5e5ae8e470db5a4ef3c128c3811690456c4f981fea800c3101e75583e7a` |
| `text/pdf/tese_banca4_2ef1861.pdf` | 268 | 1283456 | `6eacc5e5ae8e470db5a4ef3c128c3811690456c4f981fea800c3101e75583e7a` |

Contagem com `pypdf`; auditoria por expressões regulares e hashes com `hashlib`.
Zero `Citation .* undefined`, zero `Reference .* undefined`, zero `LaTeX Error`,
zero linhas iniciadas por `!` e zero avisos de nova passada no log final.
**Overfull > 10 pt: lista vazia**; não há página ou arquivo:linha a listar nessa
faixa. Busca direta no log confirmou que o maior Overfull é `8.88258pt`.
Permanecem 10 Overfull (todos abaixo de 10 pt) e 55 Underfull; não houve
aprovação visual.

### PDFs oficiais preservados

SHA-256 antes e depois, iguais em todos os três caminhos:

| PDF oficial | Antes | Depois |
| :-- | :-- | :-- |
| `text/latex/tese.pdf` | `9b18921dc5e0056f2af9d1bd0e3839ffd769e13bfb1a3e94fdfa2b9f074dda16` | `9b18921dc5e0056f2af9d1bd0e3839ffd769e13bfb1a3e94fdfa2b9f074dda16` |
| `text/pdf/tese.pdf` | `9b18921dc5e0056f2af9d1bd0e3839ffd769e13bfb1a3e94fdfa2b9f074dda16` | `9b18921dc5e0056f2af9d1bd0e3839ffd769e13bfb1a3e94fdfa2b9f074dda16` |
| `text/slides/defesa.pdf` | `0acbc57ee6fdc4efc3add8f0c0ebf1d295f5ceb33387f65a5cf6e6f3f08eb93b` | `0acbc57ee6fdc4efc3add8f0c0ebf1d295f5ceb33387f65a5cf6e6f3f08eb93b` |

Evidências locais: `$env:TEMP\banca4_w10_official_before.json` e
`$env:TEMP\banca4_w10_audit.json` (comparação programática com asserção de
igualdade). Os PDFs de teste são ignorados e não serão versionados.

## Suíte completa

Comando solicitado: `python -m pytest tests -q`, por
`Start-Process -WindowStyle Hidden -Wait`, em segundo plano. Saída em
`$env:TEMP\pytest_banca4_w10.log`; stderr em
`$env:TEMP\pytest_banca4_w10.err.log`. A primeira execução terminou com código
`1`, linha final literal:

```text
2 failed, 942 passed, 14 skipped, 2 warnings in 871.16s (0:14:31)
```

O log original foi preservado como `$env:TEMP\pytest_banca4_w10_initial.log`
e `.initial.err.log`. Depois dos ajustes abaixo, uma segunda execução completa
foi concluída, usando os nomes de log solicitados, com exit code `0` e linha
final literal:

```text
972 passed, 15 skipped, 2 warnings in 432.32s (0:07:12)
```

O total mudou de 958 para 987 testes entre as execuções devido às alterações
concorrentes descritas abaixo. Esta tarefa não acrescentou skips; o skip
adicional pertence ao estado concorrente coletado pela segunda execução. Os
dois warnings do statsmodels permaneceram. Não houve falha na segunda suíte.

Critérios de intervenção: contratos textuais podem acompanhar a redação atual
desde que preservem a regra; EOL exige atributo explícito e renormalização;
proveniência exige o modo `--refresh-provenance-only` do script responsável.
Falha numérica ou lógica será registrada com arquivo, asserção e trecho, sem
alteração dos valores ou do algoritmo. Nenhum teste será removido ou pulado.

### Classificação e correções

As duas falhas são textuais, sem divergência de números ou lógica analítica:

1. `tests/test_review2_w6.py:170`,
   `test_queue_codes_have_ordering_definitions_in_each_conclusion_paragraph`:
   a asserção `any(term in paragraph for term in definitions[code])` não
   reconhecia a forma curta da W10a, `ordenação pela probabilidade calibrada
   (O3)`, no parágrafo de sensibilidade ao erro da referência. O contrato passa
   a aceitar essa forma local exata, mas exige a definição completa de cada
   código utilizado no documento, inclusive a qualificação fora da amostra de
   O3. As demais definições locais permanecem obrigatórias.
2. `tests/test_s8a.py:79`,
   `test_new_framing_contributions_and_adoption_caveats_in_each_summary`:
   `assert term in text` falhava para `('conclusao.tex', 'ordem decrescente')`.
   A conclusão agora recomenda `a fusão como ordenação da fila de revisão`,
   não uma ordem genérica de escore. A introdução continua obrigada a conter
   `ordem decrescente`; a conclusão passa a exigir explicitamente fusão no
   lugar do escore, validação em coorte nova com referência cega, ausência de
   vantagem demonstrada sobre a probabilidade e parada por curva de esforço
   conforme orçamento humano. As demais ressalvas e os contratos dos resumos
   foram preservados.

Nenhuma falha de EOL ou hash de proveniência: não foi necessária alteração de
`.gitattributes`, renormalização ou execução de refresh. Nenhuma falha numérica
ou lógica a encaminhar. Nenhum valor agregado, algoritmo ou fonte `.tex` mudou.
Não foram acrescentados skips ou alterados marcadores de teste.

Verificação direcionada por `python -m pytest tests/test_review2_w6.py
tests/test_s8a.py -q`, exit code `0`, linha final literal:

```text
26 passed in 3.49s
```

Logs: `$env:TEMP\pytest_banca4_w10_targeted.log` e
`$env:TEMP\pytest_banca4_w10_targeted.err.log`. `git diff --check` sem erros.
Os dois warnings da primeira suíte vieram do teste
`tests/test_m8_blocagem.py::test_loglinear_known_missing_cell` no statsmodels
(`divide by zero encountered in scalar divide` e `PerfectSeparationWarning`);
não foram suprimidos.

## Commits e concorrência no worktree

Commit dos contratos: `8b599d2d0b6ef1affd4a46d9b589d6f951a5d828`,
`tests: atualizar contratos textuais apos W10a e W10b`, somente
`tests/test_review2_w6.py` e `tests/test_s8a.py`, sem trailer.

Durante a execução surgiram arquivos de outras tarefas neste mesmo worktree.
Cinco arquivos de propagação por pessoa apareceram staged enquanto os testes
rodavam, e alterações de contexto por óbito apareceram fora do índice. Um
primeiro commit local (`c852977`) incorporou inadvertidamente os cinco arquivos
staged concorrentes. Ele foi imediatamente substituído: confirmou-se que HEAD
ainda era esse próprio commit, executou-se `git reset --soft HEAD^` e o novo
commit usou `git commit --only -- tests/test_review2_w6.py tests/test_s8a.py`.
A saída confirmou apenas os dois testes no commit `8b599d2` e os cinco arquivos
concorrentes novamente staged, com os demais arquivos preservados. Não se
removeu conteúdo, não se reescreveu commit de outra tarefa e não houve push.

A segunda suíte foi iniciada antes dessa detecção; seu resultado representa o
estado de trabalho efetivamente lido por pytest, e não uma certificação de
snapshot imutável do HEAD inicial. O build de teste continua identificado pelo
HEAD inicial `2ef1861`; nenhuma edição de fonte `.tex` foi feita nesta tarefa.
O relatório, ignorado pela regra `.gitignore:224:docs/*`, requer `git add -f`
explícito; seu commit também é restrito ao próprio caminho.

Ao término da suíte, `git status --short` retornou vazio, e os cinco commits
mais recentes eram:

```text
86fd3d1 analysis: propagacao por identificador de pessoa do Sinan
8a48347 analysis: atributos de contexto por obito na ordenacao
8b599d2 tests: atualizar contratos textuais apos W10a e W10b
8bbfa94 analysis: resposta do LLM como evidencia bayesiana
2f7ac64 analysis: atributos de contexto por obito na ordenacao
```

Isso confirma avanço concorrente da branch, não execução sobre um snapshot
isolado. O relatório local ainda estava ignorado e, portanto, não aparecia
nesse status. O commit do relatório usa prefixo `chore:`, sem trailer e sem
push. Não houve ocorrência de `index.lock` nas verificações anteriores aos
commits; o procedimento de espera de 30 segundos, até cinco vezes, estava
previsto nos comandos, mas não precisou ser acionado.
