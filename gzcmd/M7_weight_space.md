# M7 — programação no espaço de pesos: auditoria e escore sem gradiente

## Veredito

**Não incorporar como substituto do XGBoost nem como guarda automática.** O papel sustentado por este ensaio é auditoria offline/geração de hipóteses interpretáveis. Uma feature extra ou escore de transporte exigiria experimento incremental aninhado e validação externa, não executados aqui.
Área cinza: AUC programado=0.97515839; nota=0.87315686; XGBoost OOF=0.99292018; logística L2=0.99159006. Brier ancorado do programado=0.00274294, mediana calibrada dos vínculos=0.97788079.
Não há evidência de que legibilidade de regras torne o escore calibrado ou uma guarda segura. Não confundir zero FP numa amostra pequena com garantia de segurança.

## Reprodução, dados e privacidade

```powershell
python scripts/m7_auditoria_xgb.py
python scripts/m7_escore_programado.py
python scripts/m7_relatorio.py
python -m pytest tests/test_m7_escore_programado.py -q
```

Python Miniconda; seed 42; extrato lido com dtype=str e conversão explícita por to_numeric, decimal vírgula convertido. Gabarito offline PAR∈{1,2}; identificadores/linhas de pares nunca são enviados à API. Allowlist M1 de 30 comparações/nota, sem identificadores, sexo ou datas brutas. Cache M1 validado por hashes da fonte/código/CSV e versões; folds persistidos (0 a 4), alinhados por pair_id e conferidos contra StratifiedKFold. Nenhum cache/configuração/módulo existente foi alterado.
O dicionário é derivado do cabeçalho e de faixas observadas, não de um features.py inexistente neste pacote. A semântica exata de algumas proximidades/inversões não é identificável pelo cabeçalho: as descrições são hipóteses nominais, não documentação independente do comparador. Zero após imputação não distingue ausência de discordância.
Todas as saídas M7 são parâmetros, código, hashes, métricas e agregados. Não há PII, pair_ids ou linhas de pares nos prompts ou artefatos M7. A chave é carregada do .env pelo transporte já existente; nunca impressa/persistida. Chamadas sequenciais com uma tentativa e ledger gravado antes do envio; checkpoints impedem repetição automática.

## E1 — leitura como auditoria

Fábrica _classifier de gzcmd/ml_oof.py usada pelo runner ml_xgb: 300 árvores, profundidade 6, learning_rate=0,05, subsample/colsample=0,8, scale_pos_weight=#neg/#pos, seed=42. Ajuste em fold !=1 e teste fold=1 (numeração zero-based). Dump completo com estatísticas em e1_tree_dump.json; seleção das 40 árvores com maior soma de split gains, respeitando limite de 155.000 caracteres sem cortar árvore. Somar ganhos não identifica efeito causal nem importância global independente.
Árvores efetivamente enviadas: 40. SHAP exato via booster.predict(pred_contribs=True), em margem log-odds; médias assinadas e absolutas por feature. SHAP dos 151 vínculos cinzentos e dos erros v3.1 de ambos os modos salvo em e1_shap.json; somente subconjuntos de treino enviados à API. Erros do motor: {"vigilancia_fn": 5, "vigilancia_fp": 8, "confirmacao_fn": 15, "confirmacao_fp": 1}.
Prompt resumido: dicionário + dump priorizado + importâncias gain/total_gain/weight/cover + SHAP agregado de treino; pedir especificação, modos de falha ranqueados e regras booleanas (aceitação MATCH) para ≥9 e cinza. Sem linhas de pares, rótulos individuais ou exemplos identificáveis.
### Especificação inferida pelo LLM

Observed from the truncated dump: a gradient-boosted regression-tree ensemble (300 trees in total, dump covers trees 0-39 only) over 22 pairwise comparator features plus the composite 'nota final'. Each tree routes a row through axis-aligned threshold splits (yes/no branches, with a missing branch) to a leaf value; the output is the additive sum of leaf values, a raw margin that is not a calibrated probability. Positive margin is read as MATCH-leaning. Feature dictionary says missing values were zero-filled by allowlist M1, so zero conflates absence with disagreement. Fold 1 is held out; SHAP aggregates come from fold != 1 training rows only. The dump cannot confirm the full ensemble specification, the objective function, or the tree-count cutoff used at scoring. Gain-based importance is split-weighted, not causal. SHAP group sizes for the 'vigilancia' and 'confirmacao' groups are very small (n = 1 to 11), so those means are not stable estimates.

### Hipóteses de falha (ordem do LLM, não contagens observadas)

```json
[
  {
    "mode": "Name spelling or partial-name errors (hypothesis, supported by split structure)",
    "responsible_features": [
      "NOME qtd frag muito parec",
      "NOME qtd frag iguais",
      "NOME qtd frag comuns",
      "NOMEMAE qtd frag muito parec"
    ],
    "evidence": "NOME qtd frag muito parec has the largest total gain (about 362,863) and is the root split in trees 0, 2-5, 7, 8, and others. Spelling variants push this count below the 0.8 threshold, which lowers the margin."
  },
  {
    "mode": "Mother's name missing or heavily variant, treated as zero (zero-fill ambiguity)",
    "responsible_features": [
      "NOMEMAE qtd frag iguais",
      "NOMEMAE qtd frag muito parec",
      "NOMEMAE ult frag igual",
      "NOMEMAE prim frag igual"
    ],
    "evidence": "Zero-fill means a missing mother's name looks like a full disagreement. NOMEMAE ult frag igual has high gain share relative to its count (total gain about 10,473). Missing and disagreeing mother's names cannot be told apart in the features."
  },
  {
    "mode": "Birth date error: one-digit typo or day/month/year inversion (observed for one-digit; inversion features unused)",
    "responsible_features": [
      "DTNASC dt iguais",
      "DTNASC dt ap 1digi",
      "DTNASC dt inv mes",
      "DTNASC dt inv dia",
      "DTNASC dt inv ano"
    ],
    "evidence": "DTNASC dt iguais and DTNASC dt ap 1digi appear often in splits (gain about 25,710 and 15,508). DTNASC dt inv dia and dt inv ano never appear in the included trees and have zero importance. Inversion handling is not visible in the dump, so it may be weak or absent."
  },
  {
    "mode": "Migration or municipality change, so CODMUNRES no longer matches (hypothesis)",
    "responsible_features": [
      "CODMUNRES local igual",
      "CODMUNRES local prox",
      "CODMUNRES uf igual"
    ],
    "evidence": "CODMUNRES local igual has notable gain (about 1,320) but a low weight count, and CODMUNRES uf features never split in included trees. Migration cases may lose matches through the municipality features, but the dump does not show it directly."
  },
  {
    "mode": "Address changes, abbreviations, or missing address (observed low influence)",
    "responsible_features": [
      "ENDERECO via prox",
      "ENDERECO texto prox",
      "ENDERECO tokens jacc",
      "ENDERECO via igual"
    ],
    "evidence": "ENDERECO via prox (gain about 18,981) and ENDERECO texto prox (about 13,716) split often in later-depth nodes. Their group SHAP means are negative for the gray positive group, so address variation pulls the margin down, but the effect is smaller than name and birth-date features."
  },
  {
    "mode": "Composite-score dominance and double counting (observed)",
    "responsible_features": [
      "nota final"
    ],
    "evidence": "nota final has the largest total gain (about 396,744) and the highest weight count (1,361 splits). It likely aggregates the same name, date, and address signals, so the model may depend heavily on it and inherit its errors."
  }
]
```

### Regras propostas

Alta: `f['nota final'] >= 7.86 and f['DTNASC dt iguais'] >= 1 and f['NOME qtd frag iguais'] >= 0.75 and f['CODMUNRES local igual'] >= 1`

Cinza: `f['nota final'] >= 5.47 and f['nota final'] < 7.86 and (f['DTNASC dt iguais'] >= 1 or f['DTNASC dt ap 1digi'] >= 1) and (f['NOMEMAE qtd frag comuns'] >= -0.25 or f['NOMEMAE ult frag igual'] >= 1) and f['NOME qtd frag muito parec'] >= 0.8`

Síntese da avaliação independente agrupada: regra alta seleciona 16/95, TP=16, FP=0, cobertura=0.16842105, contra 78/95 e zero FP da guarda atual. Regra cinza seleciona 8, TP=3, FP=5, precisão=0.37500000, recall=0.01986755. As expressões por grupo estão no JSON de auditoria; a expressão inicial acima não é reutilizada nessa avaliação.

Cada regra é interpretada por AST restrita sobre features; não há eval de código arbitrário. As regras são avaliadas em cada escopo indicado (não são aplicadas fora dele). Precisão=TP/(TP+FP); cobertura=selecionados/N da banda; recall=TP/vínculos da banda. Colunas de erros contam erros existentes que a regra aceitaria; aceitar um FN pode resgatar vínculo, aceitar um FP piora segurança.
```text
         rule scope              split  scope_n  selected  tp  fp  precision   coverage     recall  vigilancia_fn_selected  vigilancia_fp_selected  confirmacao_fn_selected  confirmacao_fp_selected
     llm_high  high pooled_exploratory       95        20  20   0 1.00000000 0.21052632 0.21505376                       0                       0                        0                        0
     llm_high  high      heldout_fold1       18         3   3   0 1.00000000 0.16666667 0.16666667                       0                       0                        0                        0
     llm_high  gray pooled_exploratory    21620        12  12   0 1.00000000 0.00055504 0.07947020                       0                       0                        0                        0
     llm_high  gray      heldout_fold1     4240         2   2   0 1.00000000 0.00047170 0.06250000                       0                       0                        0                        0
     llm_gray  high pooled_exploratory       95         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
     llm_gray  high      heldout_fold1       18         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
     llm_gray  gray pooled_exploratory    21620        79  74   5 0.93670886 0.00365402 0.49006623                       0                       0                        2                        0
     llm_gray  gray      heldout_fold1     4240        12  12   0 1.00000000 0.00283019 0.37500000                       0                       0                        0                        0
current_guard  high pooled_exploratory       95        78  78   0 1.00000000 0.82105263 0.83870968                       0                       0                        0                        0
current_guard  high      heldout_fold1       18        16  16   0 1.00000000 0.88888889 0.88888889                       0                       0                        0                        0
current_guard  gray pooled_exploratory    21620         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
current_guard  gray      heldout_fold1     4240         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_high  high    grouped_oof_all       95        16  16   0 1.00000000 0.16842105 0.17204301                       0                       0                        0                        0
crossfit_high  high             fold_0       18         3   3   0 1.00000000 0.16666667 0.17647059                       0                       0                        0                        0
crossfit_high  high             fold_1       18         3   3   0 1.00000000 0.16666667 0.16666667                       0                       0                        0                        0
crossfit_high  high             fold_2       15         5   5   0 1.00000000 0.33333333 0.35714286                       0                       0                        0                        0
crossfit_high  high             fold_3       18         2   2   0 1.00000000 0.11111111 0.11111111                       0                       0                        0                        0
crossfit_high  high             fold_4       26         3   3   0 1.00000000 0.11538462 0.11538462                       0                       0                        0                        0
crossfit_high  gray    grouped_oof_all    21620        64  64   0 1.00000000 0.00296022 0.42384106                       0                       0                        1                        0
crossfit_high  gray             fold_0     4363        19  19   0 1.00000000 0.00435480 0.57575758                       0                       0                        0                        0
crossfit_high  gray             fold_1     4240        10  10   0 1.00000000 0.00235849 0.31250000                       0                       0                        0                        0
crossfit_high  gray             fold_2     4343        14  14   0 1.00000000 0.00322358 0.41176471                       0                       0                        0                        0
crossfit_high  gray             fold_3     4294        12  12   0 1.00000000 0.00279460 0.38709677                       0                       0                        0                        0
crossfit_high  gray             fold_4     4380         9   9   0 1.00000000 0.00205479 0.42857143                       0                       0                        1                        0
crossfit_gray  high    grouped_oof_all       95        12  12   0 1.00000000 0.12631579 0.12903226                       0                       0                        0                        0
crossfit_gray  high             fold_0       18         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_gray  high             fold_1       18         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_gray  high             fold_2       15         5   5   0 1.00000000 0.33333333 0.35714286                       0                       0                        0                        0
crossfit_gray  high             fold_3       18         7   7   0 1.00000000 0.38888889 0.38888889                       0                       0                        0                        0
crossfit_gray  high             fold_4       26         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_gray  gray    grouped_oof_all    21620         8   3   5 0.37500000 0.00037003 0.01986755                       0                       0                        0                        0
crossfit_gray  gray             fold_0     4363         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_gray  gray             fold_1     4240         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
crossfit_gray  gray             fold_2     4343         4   0   4 0.00000000 0.00092102 0.00000000                       0                       0                        0                        0
crossfit_gray  gray             fold_3     4294         4   3   1 0.75000000 0.00093153 0.09677419                       0                       0                        0                        0
crossfit_gray  gray             fold_4     4380         0   0   0        NaN 0.00000000 0.00000000                       0                       0                        0                        0
```

**OOF E1 e limites:** pooled_exploratory aplica uma regra pós-auditoria ao próprio treino e não é validação externa. A auditoria inicial não é reutilizada nas regras crossfit: suas máscaras de erros históricos usam cache M1 OOF congelado e podem ter dependência indireta dos rótulos externos. Para eliminar essa dependência e cobrir todos os cinco folds dentro de três chamadas E1, duas chamadas novas/sem histórico treinam excluindo [0,1,4] e [2,3]. Só recebem parâmetros do modelo local, SHAP de vínculos cinzentos do treino e proxies de erro calculados no treino (não máscaras do cache global). grouped_oof_all compõe essas regras independentes; fold_0…fold_4 discriminam os resultados. Nenhum rótulo do grupo avaliado entra em parâmetros, escolha de subconjuntos SHAP ou prompt. Há OOF para todos os pares sobre as mesmas partições persistidas, mas treino agrupado 40%/60%, NÃO cinco auditorias com 80% de treino. Regras/partições em e1_crossfit_audit.json. Essa diferença de tamanho de treino torna a comparação conservadora e não exatamente pareada ao treinamento 5-fold original. A guarda comparadora usa p_cal_ml OOF original em todos os pares. Referência histórica: 78/95, zero FP. Nenhuma regra nova é incorporada.

## E2 — escrita a partir da especificação

Duas chamadas independentes dos rótulos: (1) gerar função pura a partir do dicionário e da descrição SIM × Sinan-TB; (2) revisão semântica do primeiro código, pedindo controle de dupla contagem, raridade, ausência e migração. Nenhum parâmetro XGBoost, SHAP, métrica/rótulo de teste ou par individual é fornecido ao gerador. Refinamento é semântico, não orientado por resultados. Código limitado por AST a uma função sem I/O/imports/loops/globals, com operações escalares e features.get; todas as predições verificadas como finitas em [0,1]. Zero rótulos embutidos no código.
Em cada fold externo, seleciona-se a melhor das duas funções pela AUC cinza apenas do treino. Como funções fixas não são ajustadas aos pares, essa validação de seleção não precisa refazer treino do escore. A calibração usa somente as taxas das âncoras nota≥9 e nota<5 do treino externo e a implementação fit_masked_anchor_platt/rates/predict de validar_ancoras_ml.py (mesmo kernel do pacote, Jeffreys apenas em taxas de fronteira). Logística L2 C=1, padronização ajustada só no treino, 5 margens OOF internas para ajustar o Platt por âncoras. Nota usa o mesmo Platt ancorado no treino. XGBoost usa o cache original 5×5 isotônico, não é recalibrado com informação do teste.
AUC/AP são sobre o escore bruto para comparar ordenação; Brier e mediana calibrada são após Platt ancorado (XGBoost mantém isotônica). Para o programado, a entrada do Platt é logit(clip(escore,10^-6,1−10^-6)), pois o escore é limitado a [0,1] enquanto o adaptador M5 foi projetado para margens. programado_platt_direto mostra a sensibilidade a usar o escore diretamente: o limite α≤10 pode ficar ativo e tornar a comparação dependente da unidade do escore. Ambos são ajustados só no treino, sem escolher a transformação por resultados de teste. Transformações/calibradores por fold podem mudar a ordenação pooled; as duas medianas são explicitadas. e2_training_bands.csv contém AUC/AP agregadas por banda somente no treino de cada fold; essas métricas não foram enviadas ao LLM.
```text
                  model scope     n  positive        auc         ap      brier  median_positive_raw  median_positive_calibrated
             programado total 61696       247 0.98974460 0.87368955 0.00113941           0.58235000                  0.98473283
             programado  gray 21620       151 0.97515839 0.80535849 0.00274294           0.57290000                  0.97788079
programado_platt_direto total 61696       247 0.98974460 0.87368955 0.00292339           0.58235000                  0.18661234
programado_platt_direto  gray 21620       151 0.97515839 0.80535849 0.00540572           0.57290000                  0.17767577
                   nota total 61696       247 0.96775760 0.56247462 0.00262302           8.06000000                  0.53451979
                   nota  gray 21620       151 0.87315686 0.21632791 0.00724830           7.03000000                  0.06780407
           logistica_l2 total 61696       247 0.99662282 0.93158969 0.00085534           5.13581242                  0.99991918
           logistica_l2  gray 21620       151 0.99159006 0.88611169 0.00215903           4.64647878                  0.99976135
      xgb_oof_isotonica total 61696       247 0.99403231 0.93391840 0.00063745           1.00000000                  1.00000000
      xgb_oof_isotonica  gray 21620       151 0.99292018 0.90696445 0.00151761           0.98181818                  0.98181818
```

Módulo gzcmd/escore_programado.py: versão 0 escolhida exclusivamente no treino do fold 1; AUCs de seleção=[0.9790522083348141, 0.9004755949074492]. O módulo exporta escore bruto, não calibrador/probabilidade operacional. A avaliação principal é do procedimento de seleção por fold; não é uma alegação de que uma versão global foi escolhida por desempenho no teste.

## Transporte entre estratos PASSO

Corte fixado pela ordem dos valores únicos PASSO: ≤6.0 versus >6.0. Sem rótulos para escolher corte. Cada direção reserva 25% do estrato de origem como validação (seed 42, estratificado); ajusta XGBoost e escolhe/calibra o programado apenas nos 75% restantes. Compara AUC no holdout de origem versus todo estrato destino, que nunca participa do ajuste. Queda positiva indica piora; negativa indica aumento. Métricas total/cinza, N e positivos explicitam diferenças de composição.
```text
    direction      model scope  source_validation_n  source_validation_positive  target_n  target_positive  within_auc  transport_auc    auc_drop
early_to_late programado total                  663                          20     59047              167  0.99821151     0.98799980  0.01021171
early_to_late programado  gray                  586                           5     19259              133  0.99277108     0.98110689  0.01166419
early_to_late        xgb total                  663                          20     59047              167  0.99961120     0.98392234  0.01568886
early_to_late        xgb  gray                  586                           5     19259              133  0.99862306     0.97297168  0.02565138
late_to_early programado total                14762                          42      2649               80  0.98651980     0.98396750  0.00255230
late_to_early programado  gray                 4835                          34      2361               18  0.99238823     0.92253521  0.06985302
late_to_early        xgb total                14762                          42      2649               80  0.99453368     0.99646263 -0.00192896
late_to_early        xgb  gray                 4835                          34      2361               18  0.99898612     0.99162991  0.00735620
```

PASSO representa estrato de blocagem/processo, não ano/local independente: não demonstra transporte temporal/geográfico. AUC também pode mudar por case-mix e raridade; estratos sem duas classes têm AUC ausente, não zero. O programado não tem pesos supervisionados para transportar; seleção é ajustada na origem. Platt só é ajustado se há ≥30 pares em cada âncora da origem, conforme M5; insuficiência é registrada, sem substituir por taxas do destino. AUC usa escores brutos e independe desse ajuste. O XGBoost de transporte é o classificador bruto com mesma fábrica; não há ajuste no destino.

## Chamadas e custo

5 tentativas; 5 concluídas. 264244 tokens de entrada + 32619 de saída. Custo estimado USD 0.09202190 (não fatura). Todas as chamadas Claude Haiku 5.5 via AnthropicReviewer de run_exp2_anthropic.py, max_tokens=12000 e sem parâmetro temperature no Messages API. Nenhum Fireworks/OpenCode Go.
Preço público consultado em 02/11/2026: entrada USD 0,10/MTok e saída USD 0,50/MTok até 100.000 tokens; acima disso USD 0,50/2,50. Sem cache, ferramentas ou prêmio regional. Fonte: https://platform.claude.com/docs/en/about-claude/pricing. Ledger e cálculo por chamada em calls.json/cost.json.

## Limites e decisão operacional

Separação por par, não por pessoa; registros compartilhados podem produzir dependência. Sem inferência causal de importância/SHAP, sem prova de estabilidade fora do extrato. Comparar score programado com logística separa interpretabilidade de eventual sinal linear; transportar entre PASSO é teste de sensibilidade interno. Regra falha não deve virar veto a nascimento ausente ou migração. Melhor uso imediato: especificação/auditoria, documentando hipóteses para estudo prospectivo; guarda/feature extra/escala de transporte ficam fora do motor até validação incremental e externa.

## Artefatos e verificação

scripts/m7_{common,auditoria_xgb,escore_programado,relatorio}.py; gzcmd/escore_programado.py; tests/test_m7_escore_programado.py; data/m7/e1_* (dump/SHAP/prompt/resposta/regras), e2_* (prompts/respostas/validação/bandas/ajustes/transporte), calls.json e cost.json. Este relatório copia números diretamente desses arquivos; não inventa métricas/uso. Verificação de testes deve ser reportada pelo executor com a saída do comando, não presumida por este renderizador.

### Verificação executada

```text
python -m pytest tests -q
........................................................................ [ 66%]
....................................                                     [100%]
108 passed in 8.29s
```
Exit code: 0. Verificação de privacidade dos artefatos: passed: key absent from all M7 artifacts; no pair/label/identifier columns in output CSVs. Nenhum commit realizado.
