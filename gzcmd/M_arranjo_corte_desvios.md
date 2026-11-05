# Arranjo sequencial: desvios e limites do registro

Este documento complementa o registro histórico, sem alterar seus arquivos ou hashes. P permanece o braço principal. P-Platt é análise de sensibilidade. O comparador a posteriori não é registrado.

As tabelas históricas e seus testes byte a byte permanecem intactos; a apresentação corrigida é gerada separadamente em `text/latex/tables/editorial/`, verificada por testes novos e incluída nos capítulos sem substituir os artefatos de proveniência.

## Cronologia e desvios

- O registro inicial, commit `8c16a41`, fixou hashes de código que só entrou no git em `83c7842`. A versão originalmente registrada não existe no histórico do git. A primeira S1 utilizou essa versão. O histórico disponível não permite reconstruí-la apenas por checkout.
- O suplemento, emenda 3, autorizou 69 chamadas adicionais, apesar da regra inicial `no sensitivity API calls`. A declaração `parameter_change: false` não elimina o desvio de protocolo. O suplemento foi registrado antes da S3.
- O comparador a posteriori aplica o corte seguido da cascata publicada, sem recalibração. Sua análise foi feita antes do registro inicial, de 5 de novembro de 2026, às 01:22:17 UTC. Os metadados originais da S3 registram 02:11:52 UTC; seu commit `16d5876` consta no git log às 02:17:13 UTC do mesmo dia. A referência anterior a 04:13 UTC foi corrigida pelo histórico. Portanto, os resultados de PAR=2 da cascata já eram conhecidos quando se registraram recalibração, braços e avaliação. O enquadramento sequencial foi escolhido com conhecimento desses resultados.
- A especificidade constava dos desfechos registrados. Sua ausência na apresentação inicial é uma omissão de relato. O texto revisto distingue esse desfecho da taxa esperada de falsos vínculos.

## Reprodutibilidade e interpretação

A reprodução numérica deve ser descrita como concordância dentro de `1e-14`, com partições idênticas, não como identidade bit a bit. A análise de robustez, com registro prévio próprio, examina threads (`n_jobs`), sementes e partições. É não confirmatória e mede estabilidade nos mesmos dados, não generalização.

Os cortes do OpenRecLink são o menor SCORE entre os PAR=1 de cada passo. Coincidem com Oliveira (2026) até a quarta casa decimal. São derivados do rótulo, não estimativas independentes dele.

As taxas do revisor têm denominadores pequenos. Em grey_low, e_fn foi 1,0 em quatro folds na vigilância (0, 1, 3 e 4), conforme `data/arranjo_corte_taxas_crossfit.csv`, linhas 2, 8, 20 e 26. Essa contagem corrige os dois folds mencionados na instrução editorial. P-PAR2-grey tem apenas dois ou três PAR=2 em algumas estimativas. Sua transferência para os pares selecionados abaixo do corte não foi validada em dados novos.

## Fontes preservadas

Consultar `prereg_arranjo_corte.yaml`, `M_arranjo_corte.md`, `prereg_robustez_arranjo.yaml`, `M_robustez_arranjo*.md` e os agregados `data/arranjo_corte_*` e `data/robustez_arranjo_*`. Este complemento não corrige retroativamente os registros, não altera fingerprints e não transforma análises de sensibilidade em resultado principal.
