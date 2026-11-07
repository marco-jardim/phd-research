# Fechamento observado v3.1 por REFREC

## Verificação da entrega

```text
python -m pytest tests -q
198 passed, 2 warnings in 15.03s
```

Os dois avisos são do statsmodels no teste M8 de célula ausente (divisão por
zero e separação perfeita), sem supressão. O comando combinado anterior
atingiu 120 segundos após a regeneração e antes de emitir o resumo do pytest;
não foi contabilizado como aprovação. A execução acima foi separada e sem
timeout. Os quatro testes novos cobrem o padrão REFREC com perfil histórico
preservado, a identidade dos suplementos, sua identificação na tabela e a
regeneração da cascata diretamente das células do CSV.

Os processos dos dois modos foram iniciados por Start-Process em paralelo,
aguardados com Wait-Process e encerrados com código zero. Ambos os JSONs estão
com `complete: true`, sem erros HTTP ou de parser. Não foram necessárias
chamadas adicionais após a reestimação: os dois arquivos de fechamento têm
zero linhas. O rerun totalizou **172 pares, 371 chamadas, 1.177.774 tokens de
entrada e 671.494 de saída**, com **US$ 0,4535244** de custo tarifário estimado.
As chaves não foram impressas nem persistidas nos artefatos.

Foram validados os hashes dos **49 artefatos por linha** e dos **26 snapshots
M3v3**, a identidade do cache padrão com o cache REFREC e a cobertura integral
das decisões LLM. As faixas e as decisões do motor particionam os 61.696 pares;
os filhos MATCH/UNSURE/NONMATCH conservam contagens e vínculos do ramo LLM.
As sensibilidades mantêm os 247 vínculos no denominador.

## Resultado e limites de interpretação

Na vigilância, a v3.1 REFREC aceita 194 vínculos e 34 falsos vínculos, perde
15 vínculos como NONMATCH e encaminha 111 pares ao humano, dos quais 38 são
vínculos. Seu lote LLM contém 347 pares: 33 MATCH, 31 UNSURE e 283 NONMATCH.
A sensibilidade automática é **78,5% (IC 95%: 73,0%; 83,2%)** e a projeção
com humano sob erro de 5% é 93,16%.

Na confirmação, aceita 132 vínculos e nenhum falso vínculo, perde 28 vínculos
como NONMATCH e encaminha 127 pares ao humano, dos quais 87 são vínculos.
Seu lote LLM contém 44 pares: 14 MATCH, 6 UNSURE e 24 NONMATCH. A sensibilidade
automática é **53,4% (IC 95%: 47,2%; 59,6%)** e a projeção com humano é 86,90%.

Frente ao perfil por linha preservado, as sensibilidades automáticas caem de
82,19% para 78,54% e de 65,18% para 53,44%. A comparação combina split e taxas
reestimadas. Não se interpreta essa diferença como efeito isolado do agrupamento.

Em `grey_low` na vigilância, as taxas medidas usam 95 pares: 7 FP/93 não
vínculos (**7,53%**, IC Wilson 3,69% a 14,73%), 2 FN/2 vínculos (**100,00%**,
IC 34,24% a 100,00%) e 7 UNSURE/95 pares (**7,37%**, IC 3,61% a 14,44%).
**O denominador de apenas dois vínculos impede tratar a taxa FN como precisa.**
Na confirmação não há observação nessa banda; permanecem os fallbacks fixos,
não taxas medidas. Não houve suavização adicional das estimativas.

O leave-out recalculado também dispõe de respostas em todos os pares que
selecionou: 388 na vigilância e 41 na confirmação, sensibilidades automáticas
observadas de 80,16% e 52,63%. É uma análise de perturbação das taxas, não
validação independente de toda a seleção adaptativa.

A NP mantém inviabilidade em 0,001 e 0,002; o mínimo global esperado com as
guardas do novo OOF é aproximadamente 0,006113. Os valores completos estão
na comparação NP regenerada, sem reutilizar o mínimo do OOF por linha.

## Arquivos de implementação

- `gzcmd/gzcmd_v3_config.yaml`, `.gitignore`, `gzcmd/final_funnel.py`
- `gzcmd/error_rates_by_band_v4.yaml` e `gzcmd/error_rates_by_band_v4_lote_antigo.yaml`
- `gzcmd/M_refrec_fechamento.md`
- `scripts/preparar_refrec_final.py`, `scripts/fechar_refrec_final.py`
- `scripts/funil_gzcmd_v2.py`, `scripts/avaliar_reruns_exp2.py`
- `scripts/auditar_funil_adversarial.py`, `scripts/regenerar_gzcmd_v4.py`
- `scripts/avaliar_oof_grupos.py`
- `tests/test_refrec_final.py`, `tests/test_oof_grupos_pr7.py`, `tests/test_m1_m2_m6.py`

Dados, tabelas e figuras foram regenerados pelos scripts acima; a lista de
originais e cópias por linha está no manifesto de preservação. Os capítulos
e os PDFs de build da tese não foram editados. Sem commit.

Padrão REFREC adotado por decisão do usuário. OOF aninhado, seed 42; perfil histórico v3/Platt preservado.
Taxas Haiku v4 reestimadas uma vez incluindo os suplementos por linha e REFREC inicial; congeladas antes de quaisquer chamadas adicionais de fechamento.
Classificador e calibração são OOF por grupo. As taxas LLM são estimadas em coortes selecionadas adaptativamente; o replay observado não é validação independente das taxas.
Todas as decisões LLM da cascata final são observadas. UNSURE é encaminhamento humano. Sensibilidade após revisão humana é projeção com erro hipotético de 5%, não resultado humano observado.
FN_resolvidos são vínculos classificados NONMATCH. FN_cadeia inclui ainda vínculos encaminhados ao humano, não aceitos automaticamente.
Custo em US$ aplica a tarifa histórica aos tokens registrados por par; não é fatura. O custo alocado à cascata não precisa coincidir com o gasto total dos lotes executados.
A comparação linha/REFREC combina mudança do split e reestimação das taxas; não isola causalmente apenas o split. A comparação com taxas fixas permanece no relatório OOF por grupo.

## Comparação por linha e REFREC
```
split_by        modo versao  VP_cadeia  FP_cadeia  FN_resolvidos  pares_ao_humano  pares_ao_llm  sens_cadeia_observada  sens_ci95_low  sens_ci95_high  sens_com_humano_erro_05  custo_llm_usd
     row  vigilancia     v3        144         35             91               61          1410               0.582996       0.520695        0.642755                 0.629150       3.785245
     row  vigilancia   v3.1        203         19             14               83           301               0.821862       0.769327        0.864540                 0.937247       0.826020
     row confirmacao     v3        100          6            130               25           435               0.404858       0.345553        0.467078                 0.470243       1.163579
     row confirmacao   v3.1        161          1             23               90            25               0.651822       0.590497        0.708497                 0.894130       0.070229
  refrec  vigilancia     v3        144         35             91               61          1410               0.582996       0.520695        0.642755                 0.629150       3.785245
  refrec  vigilancia   v3.1        194         34             15              111           347               0.785425       0.730063        0.832045                 0.931579       0.919832
  refrec confirmacao     v3        100          6            130               25           435               0.404858       0.345553        0.467078                 0.470243       1.163579
  refrec confirmacao   v3.1        132          0             28              127            44               0.534413       0.472155        0.595617                 0.869028       0.119193
```

## Taxas grey_low
```
       modo     band  n  n_vinculos  n_nao_vinculos  fp  fn  n_unsure  n_lt_10  denom_fp_lt_10  denom_fn_lt_10     e_fp  e_fp_ci95_low  e_fp_ci95_high  e_fp_fixed  e_fp_abs_diff  e_fn  e_fn_ci95_low  e_fn_ci95_high  e_fn_fixed  e_fn_abs_diff        u  u_ci95_low  u_ci95_high
 vigilancia grey_low 95           2              93   7   2         7    False           False            True 0.075269       0.036937        0.147297        0.08       0.004731   1.0        0.34238             1.0        0.12           0.88 0.073684    0.036149     0.144357
confirmacao grey_low  0           0               0   0   0         0     True            True            True      NaN            NaN             NaN        0.08            NaN   NaN            NaN             NaN        0.12            NaN      NaN         NaN          NaN
```

## Consumo completo dos reruns
```
                             lote        modo   n  elapsed_s  cost_estimate_usd  calls  http_errors  response_parse_errors  input_tokens  output_tokens  cache_read_input_tokens  cache_creation_input_tokens  max_request_input_tokens
anthropic_sem_mo_v4_lote31_refrec  vigilancia 155     728.65           0.412064    336            0                      0       1067792         610570                        0                            0                      4292
anthropic_sem_mo_v4_lote31_refrec confirmacao  17      78.97           0.041460     35            0                      0        109982          60924                        0                            0                      3151
```

Artefatos por linha preservados com sufixo _row e manifesto SHA256 em data/arquivos_preservados_row.json.
Nenhum capítulo foi editado. Os CSVs de pendentes não contêm rótulos, grupos ou campos pessoais.
Reprodução offline após os runs: python scripts/fechar_refrec_final.py --finalize. Não reestimar novamente após coletar o fechamento sem avaliar a nova seleção.
