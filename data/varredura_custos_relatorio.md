# Varredura de custos — revisões humanas equivalentes

Taxas M3: D:\git\phd-research\gzcmd\error_rates_by_band_v4.yaml; SHA256=9fc27bfe8a3f0cd92648a98586e915594295537a1aebd17c273c11a5771913d2.
60 pontos por modo; p_cal_ml OOF fixo, seed 42. Expectativas do bloco claude-haiku-5-5, sem respostas LLM simuladas ou chamadas reais.
C_llm=0,005 deriva do arredondamento de 5s/1200s=1/240; humano base=1.
Erro humano é cenário, sem concordância inter-humana medida. A perda humana inclui o erro esperado.
Orçamento limita humano direto; UNSURE LLM adiciona carga humana esperada, discriminada separadamente.
O ponto ∞ é categórico na figura (posição gráfica 400), não um limite finito.
A mudança simultânea de custo LLM e erro humano não garante perda menor: barato não significa infalível.
## Antes/depois: C_llm 0,25/e_h=0 → 0,005/e_h=0,05
```
       modo cenario  C_llm  e_fn_h  lote_llm  humano_direto  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  vinculos_perdidos_auto_nonmatch  FP_automaticos  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total
 vigilancia   antes  0.250    0.00       212            207             218.545010          75          21158                                5               2                  0.591428                  0.962594           1208.199643
 vigilancia  depois  0.005    0.05       347             80              98.819165         132          21093                                5              16                  0.821610                  0.950656           1291.415361
confirmacao   antes  0.250    0.00         0            229             229.000000          47          21376                               16               0                  0.445344                  0.923077            731.866755
confirmacao  depois  0.005    0.05        44            121             124.317491          55          21432                               19               0                  0.556636                  0.876790           1121.492177
```

Dados completos: varredura_custos_vigilancia.csv e varredura_custos_confirmacao.csv.
Figura: text/latex/figures/fig_cap8_varredura_custos.pdf (+ PNG). Nenhum .tex modificado.
