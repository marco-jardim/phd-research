# S8c.1: comparação com revisão humana do MATCH do LLM

`scripts/comparador_p_match_humano.py` utiliza exclusivamente quatro CSVs versionados: os pontos de mesma carga, a curva de esforço, o VPP das aceitações do LLM e a partição dos 36 PAR=2. Não ajusta modelos nem lê dados por potencial par. O script verifica as concordâncias entre aceitação automática, VP do LLM e recuperação total de P antes de somar a carga adicional.

| Modo | Encaminhamentos humanos originais de P | MATCH do LLM ao humano | Revisões humanas totais de P | PAR=2 de P | O3 na mesma carga | k mínimo de O3 para igualar P |
|---|---:|---:|---:|---:|---:|---:|
| Vigilância | 96 | 25 | 121 | 23 | 22 em 121 | 125 |
| Confirmação | 17 | 12 | 29 | 9 | 9 em 29 | 29 |

O revisor perfeito é hipótese dos dois lados, incluindo os encaminhamentos originais de P e os MATCH adicionais do LLM. Sob essa hipótese, os FP são zero em ambos, porque os não vínculos enviados ao humano são rejeitados. Esse zero não é resultado observado de revisão humana. Na vigilância, há diferença de um PAR=2 na mesma carga, e O3 exige quatro revisões adicionais para igualar a recuperação de P; na confirmação há igualdade na mesma carga. O artefato não apresenta uma ordenação ou política como dominante em todas as cargas.

`data/comparador_p_match_humano_metadata.json` identifica a análise como exploratória posterior à avaliação, sem registro confirmatório, e documenta as fontes e as hipóteses. O script também materializa `data/comparador_mesma_carga_erro05.csv`, que passa a fornecer a coluna editorial "Com erro 5%". A projeção reduz em 5% somente os vínculos recuperados pela revisão humana; não modela falsos positivos humanos. A ausência desse componente não autoriza pressupor zero FP quando a revisão é imperfeita.

## Evidência de C1 a C9

- C1: documentada como parcial a restauração anterior de `tab_cap8_cascata_v31.tex`. `c958b85` é um objeto de conteúdo, cuja legenda e `tabcolsep` de 4 pt diferem da versão atual de 3 pt. A tabela histórica não foi alterada nesta fatia, cuja autorização restringe alterações de tabelas às editoriais. Não se atribuiu essa restrição a um hash inexistente da tabela.
- C2: o documento de desvios agora cita a emenda 1 da robustez e discrimina as reversões do teste da S3 e do suplemento e a alteração do teste da robustez, posteriores à geração dos resultados, com preservação da identidade anterior de execução.
- C3: os dois documentos foram atualizados para o contrato atual de `test_s6a.py`, sem afirmar que ainda confere a antiga frase completa do cenário.
- C4: os valores de referência e duração são lidos dos CSVs de carga e horas. Um teste altera cópias desses insumos para demonstrar que o cálculo os acompanha; inconsistências nas contagens são recusadas.
- C5: as limitações do cenário de 5% estão declaradas nos documentos, nos metadados novos e nas notas das tabelas editoriais. Não se modelam FP humanos.
- C6: o gerador editorial lê a projeção de um CSV materializado, sem recalculá-la na apresentação; alvos ausentes ou duplicados são recusados.
- C7: esforço, FP por região e os dois PGF têm fim de linha LF explícito no `.gitattributes`, coerente com os geradores e comparações de bytes. Os dois geradores de tabelas não congelados passam a escrever com `newline` explícito. Não se alteraram os arquivos das figuras em `text/slides/`.
- C8: a seção 3 do plano fornecido não contém item C8; não foi criada uma exigência adicional.
- C9: PDFs com nome `text/slides/defesa_*.pdf` passam a ser ignorados, sem ignorar `defesa.pdf`.
