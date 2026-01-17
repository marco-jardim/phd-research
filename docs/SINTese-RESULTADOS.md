# Sintese e interpretacao dos resultados (execucao)

Este documento consolida os principais achados obtidos apos a execucao dos notebooks em `notebooks/`.
O problema e altamente desbalanceado: no dataset lido (`Shape do dataset: (61696, 60)`), ha `PAR=1` (211) e `PAR=2` (36), totalizando 247 pares verdadeiros (~0,40%).
Isso torna `accuracy` pouco informativa; por isso o estudo usa principalmente `Precision`, `Recall`, `F1`, `AUC-ROC` e `AUC-PR`.

## 1) Resultado consolidado (Notebook 01 - comparacao de tecnicas)

Fonte: `comparacao_modelos.csv`.

Ranking observado (melhores valores por metrica, entre os modelos avaliados):

- Melhor equilibrio (F1): `6. Random Forest + SMOTE`
  - Precision: 0.8902
  - Recall: 0.9865
  - F1: 0.9359
  - AUC-ROC: 0.9999
  - AUC-PR: 0.9727

- Melhor precisao entre os avaliados no baseline: `3. Gradient Boosting`
  - Precision: 0.9444
  - Recall: 0.9189
  - F1: 0.9315
  - AUC-PR: 0.9307

- Melhor recall (empatado) no baseline: `6. Random Forest + SMOTE` e `1. Logistic Regression`
  - Ambos atingem Recall: 0.9865
  - Interpretacao: a regressao logistica atinge recall alto a custo de precisao muito baixa (0.3333), enquanto RF+SMOTE mantem precisao alta (0.8902).

- Observacao importante sobre AUC-ROC:
  - Os valores de AUC-ROC ficaram muito altos (proximos de 1) mesmo quando a precisao foi baixa (ex: Logistic Regression e Stacking).
  - Isso e coerente com bases desbalanceadas: ROC pode parecer excelente sem necessariamente gerar um ponto operacional bom.
  - Por isso AUC-PR e mais util para comparar modelos quando a classe positiva e rara.

Interpretacao do Notebook 01:

- O ganho mais consistente vem de duas ideias simples:
  - tratar desbalanceamento (SMOTE e/ou pesos)
  - ajustar o threshold de decisao conforme o objetivo
- Entre os modelos avaliados, `Random Forest + SMOTE` se destacou por entregar simultaneamente recall muito alto e precisao alta.

## 2) Resultado e interpretacao (Notebook 02 - estrategia de maximo recall)

O notebook implementa uma estrategia agressiva para reduzir falsos negativos:

- Reamostragem (SMOTE e variantes) para aumentar presenca da classe positiva no treino.
- `class_weight` extremo em RandomForest (ex: {0:1, 1:500}).
- Ensemble (soft voting) com threshold baixo (ex: 0.15), explicitamente escolhido para elevar recall.

Interpretacao:

- O efeito esperado dessa estrategia e:
  - aumentar significativamente o numero de candidatos retornados
  - aumentar recall (nao perder pares)
  - aceitar queda de precisao (mais revisao manual)

Saida operacional:

- O notebook gera rankings/probabilidades (ex. coluna `PROBA_PAR_RECALL` e `RANK_RECALL`) para priorizacao.
- Quando o objetivo e varrer e revisitar casos possiveis, o ranking e tao importante quanto uma classe binaria final.

## 3) Resultado e interpretacao (Notebook 03 - estrategia de maxima precisao)

O notebook busca minimizar falsos positivos usando uma combinacao de filtros conservadores:

- Selecao de features (RandomForest importance; TOP_N_FEATURES=25).
- SMOTE moderado (`sampling_strategy=0.3`) para nao degradar qualidade.
- Modelos com regularizacao e thresholds altos (XGBoost/LightGBM/RF calibrado).
- Stacking + thresholds altos (ex: 0.8).
- Consensus voting (unanimidade) para reduzir falsos positivos.
- Regras de negocio (`regras_alta_confianca`) e, por fim, modelo hibrido (ML + regras).

Interpretacao:

- As tecnicas se alinham ao objetivo: reduzir ao maximo a taxa de falsos positivos.
- O custo dessa estrategia e inevitavel: recall cai (mais pares verdadeiros ficam de fora), mas os pares retornados tendem a ser mais confiaveis e exigem menos revisao.

Saida operacional:

- O resultado mais util e uma lista curta de candidatos de alta confianca.
- Exemplo de artefato gerado: `candidatos_revisao_ml.csv` (colunas como `ML_PROBA` e `nota final`) para revisao dirigida.

## 4) Qual metrica "manda" neste estudo

- Se o objetivo e "nao perder match" (triagem ampla / revisao manual): maximize `Recall` e use `AUC-PR` como metrica de suporte.
- Se o objetivo e "alta confianca" (automacao / baixo custo de erro): maximize `Precision` e acompanhe `Recall` para nao colapsar.
- Para comparar modelos em base desbalanceada:
  - prefira `AUC-PR` a `AUC-ROC`.
  - use `F1` quando quiser um compromisso entre precision e recall, mas sempre confirme se esse compromisso faz sentido operacionalmente.

## 5) Recomendacoes praticas

- Padronizar a exportacao de metricas finais de cada notebook em um `results.json` (por notebook), para rastreabilidade e comparacao historica.
- Manter dois pipelines:
  - pipeline "recall" (gera ranking amplo)
  - pipeline "precisao" (gera lista curta de alta confianca)
