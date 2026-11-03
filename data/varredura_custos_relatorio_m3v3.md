# Varredura de custos — revisões humanas equivalentes

Taxas M3: gzcmd\error_rates_by_band_estimadas.yaml; SHA256=6dd0e6145219ed6307fd0b175f46ca68ec02f0d930704a1c72fc48a8b3d91e1d.
60 pontos por modo; p_cal_ml OOF fixo, seed 42. Expectativas pooled, sem respostas LLM simuladas ou chamadas reais.
C_llm=0,005 deriva do arredondamento de 5s/1200s=1/240; humano base=1.
Erro humano é cenário, sem concordância inter-humana medida. A perda humana inclui o erro esperado.
Orçamento limita humano direto; UNSURE LLM adiciona carga humana esperada, discriminada separadamente.
O ponto ∞ é categórico na figura (posição gráfica 400), não um limite finito.
A mudança simultânea de custo LLM e erro humano não garante perda menor: barato não significa infalível.
## Antes/depois: C_llm 0,25/e_h=0 → 0,005/e_h=0,05
```
       modo cenario  C_llm  e_fn_h  lote_llm  humano_direto  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  vinculos_perdidos_auto_nonmatch  FP_automaticos  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total
 vigilancia   antes  0.250    0.00       134            174             178.427691         105          21224                                5               2                  0.799117                  0.956074           1013.431434
 vigilancia  depois  0.005    0.05       486             88             104.185072         127          20936                                5               8                  0.871020                  0.950899           1087.693598
confirmacao   antes  0.250    0.00         1            187             187.136418          73          21376                               12               1                  0.607287                  0.939271            647.084109
confirmacao  depois  0.005    0.05        15             99             101.585928          79          21444                               15               1                  0.639177                  0.907698            954.829574
```

Dados completos: varredura_custos_vigilancia.csv e varredura_custos_confirmacao.csv.
Figura: text/latex/figures/fig_cap8_varredura_custos.pdf (+ PNG). Nenhum .tex modificado.
