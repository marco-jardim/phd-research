# Reprocessamento M5/M5b/M7 por REFREC

Os scripts de âncoras e o carregador M7 validam `split_by` no fingerprint do
cache M1 e honram `evaluation.split_by`. M5 reconstrói a margem bruta que o
cache M1 não persiste, usando os mesmos cinco folds externos e internos,
StratifiedGroupKFold e seeds 42 a 47. A isotônica reconstruída reproduziu o
cache REFREC com diferença máxima de 1,11e-16. E3 também conserva a separação
de grupos em todas as etapas de ajuste. Não há alteração no pacote externo.

Beta reutiliza essa margem, verificando fonte, hash, partição e folds. As
alternativas globais por momentos continuam distintas da calibração totalmente
aninhada: não se apresenta o agrupamento do classificador como isolamento de
toda a calibração por âncoras. O veredito conjunto de M5 continua negativo,
com Brier dentro da tolerância e ECE fora dela; ambas as variantes Beta
continuam sem atender simultaneamente aos dois critérios.

M7 recebe `--offline`, carregando exclusivamente a função pura já salva em
`gzcmd/escore_programado.py`. Esse caminho não chama geração ou refinamento,
não muda a função e não seleciona novo candidato. Calibração e diagnóstico
linear usam os folds externos REFREC e, no ajuste linear interno, grupos
disjuntos. A reavaliação solicitada não reexecuta transporte por PASSO; os
resultados anteriores foram preservados e o novo resumo identifica esse
escopo explicitamente. AUC cinza do escore fixo: 0,97515839; do XGBoost
isotônico OOF por REFREC: 0,99514223.

```powershell
python scripts/preservar_m5_m7_row.py
python scripts/validar_ancoras_ml.py --reuse-margins
python scripts/validar_ancoras_beta.py
python scripts/m7_escore_programado.py --offline
python scripts/finalizar_pdf_refrec.py --clean-old
python scripts/build_pdf.py --jobname tese_refrec2 --outname tese_refrec2
python scripts/finalizar_pdf_refrec.py --publish
python -m pytest tests -q
```

O manifesto de preservação armazena hashes dos 23 artefatos originais por
linha. As duas tabelas históricas são preservadas como `_row.tex.txt` em
`data/`, com os mesmos bytes, para não manter marcas editoriais antigas entre
os `.tex` ativos. Dados M7 `_row` também ficam ignorados pelo git.

A limpeza seleciona somente extensões de build e os padrões autorizados,
nunca fontes `.tex` nem os dois `tese.pdf`. A publicação exige ausência de
erros e referências/citações indefinidas no log final, verifica identidade
SHA256 dos dois PDFs copiados e só então remove as marcas `tese_refrec2`.
O JSON de auditoria conserva as contagens, hashes e arquivos removidos.
