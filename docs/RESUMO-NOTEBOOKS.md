# Resumo dos notebooks (SIM x SINAN-TB)

## Dados e desbalanceamento

- Base utilizada: `data/COMPARADORSEMIDENT.csv` (no notebook aparece como `COMPARADORSEMIDENT.csv`, assumindo execucao no mesmo diretorio do CSV).
- Alvo: `TARGET = 1` quando `PAR in {1,2}`, caso contrario `0`.
- Os notebooks deixam explicito que ha forte desbalanceamento (classe positiva rara) e aplicam tecnicas para mitigacao:
  - `class_weight='balanced'` (e pesos manuais extremos em alguns modelos).
  - SMOTE/variantes (SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek) e, em alguns pontos, undersampling.

## Acuracia dos algoritmos

- A metrica "acuracia" (accuracy) nao e reportada como metrica principal.
- Em linkage, com desbalanceamento forte, os notebooks priorizam `precision`, `recall`, `F1`, `AUC-ROC` e `AUC-PR`.
- Portanto, nao ha um valor unico de "acuracia" consolidado no codigo do notebook; os resultados sao impressos em execucao (nao estao salvos no arquivo `.ipynb` pois `execution_count` e `outputs` estao vazios).

## Como os parametros foram escolhidos

### Notebook 01: `01_analise_comparativa_tecnicas.ipynb`

- Split estratificado (70/30) com `random_state=42`.
- Modelos base e parametros fixados "na mao" (sem busca sistematica):
  - LogisticRegression: `class_weight='balanced'`, `max_iter=1000`.
  - RandomForest: `n_estimators=200`, `max_depth=10`, `class_weight='balanced'`.
  - GradientBoosting: `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`.
  - SVM RBF: `class_weight='balanced'`, `probability=True`.
  - MLP: camadas `(128,64,32)`, `early_stopping=True`.
- Balanceamento adicional: SMOTE (`k_neighbors=3`) + RandomForest.
- Threshold: funcao `optimize_threshold()` varre `thresholds = 0.01..0.99` e seleciona:
  - max F1;
  - ou max recall com restricao `precision > 0.01`;
  - ou max precision com restricao `recall > 0.3`.
  O proprio resumo do notebook recomenda thresholds ~0.15-0.20 (recall) e ~0.70-0.80 (precision).

### Notebook 02: `02_estrategia_maximo_recall.ipynb`

- Objetivo declarado: maximizar recall (aceitando mais falsos positivos).
- Feature engineering agressivo e "manual" (novas variaveis, pesos e flags relaxadas), incluindo um score ponderado `score_recall` com pesos escolhidos heuristcamente.
- Balanceamento: testa varias tecnicas (SMOTE/BorderlineSMOTE/ADASYN/SMOTETomek e SMOTE extremo 1:1).
- Parametros/tuning sao principalmente heuristicas para recall:
  - RandomForest com `class_weight={0:1,1:500}` e threshold 0.3.
  - Ensemble soft voting com pesos `[2,1,1,1]` e avaliacao de thresholds de 0.15 a 0.40; escolhe `best_th_recall=0.15`.
  - Cascade com estagio 1 usando `class_weight` muito alto e threshold 0.1 para "passar candidatos".

### Notebook 03: `03_estrategia_maxima_precisao.ipynb`

- Objetivo declarado: maximizar precisao (pares de alta confianca).
- Feature engineering conservador: flags rigorosas (`nome_perfeito >= 0.95`, `dtnasc_exato`, etc.) e `score_precisao` baseado em evidencias fortes.
- Feature selection: RandomForest para importance e seleciona `TOP_N_FEATURES = 25`.
- SMOTE moderado: `sampling_strategy=0.3`.
- Modelos com regularizacao e thresholds altos:
  - XGBoost com `max_depth=4`, `learning_rate=0.05`, `reg_alpha=1.0`, `reg_lambda=2.0`, `scale_pos_weight=10`; testa thresholds 0.6-0.9.
  - LightGBM conservador (`num_leaves=15`, `min_child_samples=50`) com thresholds 0.6-0.9.
  - RandomForest calibrado (isotonic) com thresholds 0.7-0.9.
  - Stacking com meta-learner `LogisticRegression(C=0.1)` e avaliacao de thresholds; guarda th=0.8.
- Regras de negocio: funcao `regras_alta_confianca()` atribui score (nome, dt nasc, mae, municipio, endereco, nota final) e testa limiares; usa `best_limiar=8`.
- Modelo final "hibrido": classifica como par se `ML >= 0.7` e `score_regras >= 7`.

## Nota pratica (para reproduzir metricas)

Como os notebooks estao sem outputs salvos, para responder com numeros (precision/recall/F1/AUC) e necessario executa-los.
Os arquivos escritos em execucao incluem, por exemplo:
- `comparacao_modelos.csv`
- `candidatos_recall_todos.csv`, `candidatos_recall_obito_prioritario.csv`
- `candidatos_precisao_alta_confianca.csv`, `candidatos_precisao_media_confianca.csv`
