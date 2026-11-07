# Fusão exploratória post hoc de filas abaixo do corte

CPU, sem LLM. Mesmos cortes e mesmas filas do comparador. Revisor perfeito simulado por PAR=2; 61.485 pares únicos, 36 PAR=2. Não é validação independente nem altera a hipótese registrada O3 versus O1, margem de 25%.

Os 36 PAR=2 estão nos 2.522 candidatos SCORE ≥ −5,648: provável viés de verificação a favor de O1. Nenhuma política teve acesso a resultados futuros. A1/A2 usam apenas as respostas já observadas pelo braço escolhido.

## Definições fixadas antes das curvas fundidas

F1: alternar O3/O1; F2: alternar O3/O1/MACD; F3: RRF(60) O3/O1/MACD; F4: RRF(60) O3/O1. A1: Thompson Beta(1,1), janela de 200 revisões do braço; A2: sem janela. Deduplicação global. Não houve seleção de parâmetros. Fixação a priori para esta execução post hoc, não pré-registro.

200 repetições por política, seeds 0–199. Métricas adaptativas são média [P10; P90] das métricas por repetição; não são inversões da curva média. Quantis lineares, não IC inferenciais.

## Tabela (todas as métricas)

| Fila | revisoes_9 | revisoes_23 | revisoes_30 | revisoes_36 | par2_top_17 | par2_top_58 | par2_top_96 | par2_top_125 | par2_top_440 | par2_top_2522 | par2_top_3040 | auc_recuperacao_3040 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O1 | 51.00 | 501.00 | 1060.00 | 2521.00 | 3.00 | 12.00 | 19.00 | 19.00 | 21.00 | 36.00 | 36.00 | 0.794673 |
| O3 | 29.00 | 125.00 | 612.00 | 23443.00 | 6.00 | 20.00 | 22.00 | 23.00 | 26.00 | 33.00 | 34.00 | 0.856908 |
| O3_MACD | 46.00 | 118.00 | 469.00 | 3040.00 | 6.00 | 12.00 | 21.00 | 23.00 | 29.00 | 34.00 | 36.00 | 0.874342 |
| F1 | 49.00 | 114.00 | 689.00 | 1787.00 | 7.00 | 12.00 | 17.00 | 25.00 | 28.00 | 36.00 | 36.00 | 0.900630 |
| F2 | 21.00 | 112.00 | 513.00 | 1999.00 | 7.00 | 13.00 | 17.00 | 24.00 | 29.00 | 36.00 | 36.00 | 0.907237 |
| F3 | 12.00 | 64.00 | 255.00 | 1777.00 | 10.00 | 20.00 | 25.00 | 26.00 | 31.00 | 36.00 | 36.00 | 0.934859 |
| F4 | 10.00 | 112.00 | 437.00 | 1387.00 | 16.00 | 18.00 | 18.00 | 26.00 | 30.00 | 36.00 | 36.00 | 0.925320 |
| A1 | 26.61 [20.00; 33.00] | 96.38 [85.00; 114.00] | 447.69 [366.90; 482.10] | 1829.47 [1787.80; 1872.20] | 7.30 [7.00; 8.00] | 12.05 [11.00; 14.00] | 21.52 [18.00; 23.00] | 23.46 [23.00; 24.00] | 29.52 [29.00; 31.00] | 36.00 [36.00; 36.00] | 36.00 [36.00; 36.00] | 0.908205 [0.906608; 0.910903] |
| A2 | 26.61 [20.00; 33.00] | 96.38 [85.00; 114.00] | 459.64 [365.80; 515.10] | 1705.85 [1548.00; 2022.60] | 7.30 [7.00; 8.00] | 12.05 [11.00; 14.00] | 21.52 [18.00; 23.00] | 23.46 [23.00; 24.00] | 29.50 [29.00; 31.00] | 36.00 [36.00; 36.00] | 36.00 [36.00; 36.00] | 0.914351 [0.912778; 0.915628] |

AUC: integral da recuperação em degraus, C(0)=0 e C(k)/36 em [k,k+1), até 3.040; normalização por 3.040. Empates individuais e RRF: hash lexical crescente, exatamente k sem expansão; intercalação segue a ordem dos braços; empate nas amostras Thompson: O1, O3, MACD.

## Curva resumida e melhores recuperações por orçamento

| k | O1 | O3 | O3_MACD | F1 | F2 | F3 | F4 | A1 | A2 | Melhor(es), pela média |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 3.00 | 6.00 | 6.00 | 7.00 | 7.00 | 10.00 | 16.00 | 7.30 | 7.30 | F4 |
| 58 | 12.00 | 20.00 | 12.00 | 12.00 | 13.00 | 20.00 | 18.00 | 12.05 | 12.05 | O3, F3 |
| 96 | 19.00 | 22.00 | 21.00 | 17.00 | 17.00 | 25.00 | 18.00 | 21.52 | 21.52 | F3 |
| 125 | 19.00 | 23.00 | 23.00 | 25.00 | 24.00 | 26.00 | 26.00 | 23.46 | 23.46 | F3, F4 |
| 440 | 21.00 | 26.00 | 29.00 | 28.00 | 29.00 | 31.00 | 30.00 | 29.52 | 29.50 | F3 |
| 2522 | 36.00 | 33.00 | 34.00 | 36.00 | 36.00 | 36.00 | 36.00 | 36.00 | 36.00 | O1, F1, F2, F3, F4, A1, A2 |
| 3040 | 36.00 | 34.00 | 36.00 | 36.00 | 36.00 | 36.00 | 36.00 | 36.00 | 36.00 | O1, O3_MACD, F1, F2, F3, F4, A1, A2 |

## Leitura curta

- Recuperar 9 PAR=2: menor esforço médio em F4 (10.00 revisões).
- Recuperar 23 PAR=2: menor esforço médio em F3 (64.00 revisões).
- Recuperar 30 PAR=2: menor esforço médio em F3 (255.00 revisões).
- Recuperar 36 PAR=2: menor esforço médio em F4 (1387.00 revisões).
- Dominação simultânea de O1 e O3 em 9/23/36 (não pior em todos): F2, F3, F4, A1, A2.
- Menor esforço e maior recuperação pontual não implicam dominância em toda a curva. Comparações adaptativas acima usam médias; a dispersão consta da tabela.

## Reprodução e proveniência

As curvas completas individuais reproduziram sensibilidade_o3_macd_curva.csv; O1/O3 também reproduziram comparador_mesma_carga_curva.csv e os resumos, antes de qualquer fusão. O3 usa p_cal_ml OOF original validado por align_original; MACD é recalculada com prepare/nested_oof_probabilities importados, mesmos folds, versões e atributos. Nenhum código existente foi copiado ou alterado.

Estimativas, tempos, versões, SHA-256 dos códigos/entradas/saídas e definições constam de data/fusao_filas_metadata.json. CSVs contêm somente agregados, sem identificadores ou valores individuais.
