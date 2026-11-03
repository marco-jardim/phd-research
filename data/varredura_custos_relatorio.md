# Varredura de custos — revisões humanas equivalentes

Taxas M3: gzcmd\error_rates_by_band_v4.yaml; SHA256=53d02f80bcdab84b90e0e0cf9c792782f564b56faedf1215c4b3267420edc357.
60 pontos por modo; p_cal_ml OOF fixo, seed 42. Expectativas do bloco claude-haiku-5-5, sem respostas LLM simuladas ou chamadas reais.
C_llm=0,005 deriva do arredondamento de 5s/1200s=1/240; humano base=1.
Erro humano é cenário, sem concordância inter-humana medida. A perda humana inclui o erro esperado.
Orçamento limita humano direto; UNSURE LLM adiciona carga humana esperada, discriminada separadamente.
O ponto ∞ é categórico na figura (posição gráfica 400), não um limite finito.
A mudança simultânea de custo LLM e erro humano não garante perda menor: barato não significa infalível.
## Antes/depois: C_llm 0,25/e_h=0 → 0,005/e_h=0,05
```
       modo cenario  C_llm  e_fn_h  lote_llm  humano_direto  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  vinculos_perdidos_auto_nonmatch  FP_automaticos  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total
 vigilancia   antes  0.250    0.00       198            122             128.504756         101          21216                                4               2                  0.797203                  0.969716            965.295179
 vigilancia  depois  0.005    0.05       301             52              60.933128         111          21173                                4               5                  0.875178                  0.963350            989.648691
confirmacao   antes  0.250    0.00         0            188             188.000000          73          21376                               12               1                  0.607287                  0.939271            647.139552
confirmacao  depois  0.005    0.05        25             83              85.341903          79          21450                               15               1                  0.680180                  0.902510            944.813697
```

Dados completos: varredura_custos_vigilancia.csv e varredura_custos_confirmacao.csv.
Figura: text/latex/figures/fig_cap8_varredura_custos.pdf (+ PNG). Nenhum .tex modificado.
