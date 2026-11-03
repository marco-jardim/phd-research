# Reavaliação com Claude Haiku 5.5, prompt v4 e dossiê sem model_outputs

A referência é somente by_mode.<modo>.models.claude-haiku-5-5 do YAML v4. O pooled misto com DeepSeek V4.1 Flash não foi usado.
Custos aprovados preservados: C_llm=0,005; C_humano=1; e_h=0,05; humano direto ilimitado.
OOF, guardas e custos permanecem fixos. O perfil v3 mantém Platt/taxas fixas/custo LLM=0,25 históricos; v3.1 usa as taxas Haiku v4.
Kimi K2.5 e Claude Haiku 5.5 v3 com MO são comparadores históricos. Suas pendências usam taxas históricas próprias, identificadas no CSV; nunca alimentam o motor novo.
A referência não é substituída pelo comparador de menor perda. Scores do motor no dossiê antigo tornam essa comparação potencialmente contaminada.
A perda mistura erros conhecidos e tempo de encaminhamento observados com perdas esperadas dos pendentes e do revisor humano imperfeito.
A política v3.1 tem cobertura Haiku v4 integral após incorporar o lote complementar. Não há inferência de respostas LLM pendentes na referência.
A projeção anterior foi preservada em colunas pre_lote31. O cenário com revisor humano continua sendo hipótese de erro de 5%, separado do desfecho automático observado.
Os deltas abaixo usam expectativas da varredura, sem replay, no mesmo ponto C_llm=0,005/e_h=0,05/orçamento infinito. Assim isolam a troca das taxas/revisor de referência.
Arquivos anteriores preservados com sufixo _m3v3 e hashes em arquivos_preservados_m3v3.json; nenhuma resposta LLM foi gerada.

## Gabarito do Haiku v4 no lote antigo, sem decisões automáticas do motor
```
       modo    n  VP  FP  FN  UNSURE
 vigilancia 1410  73  34  19      61
confirmacao  435  64   6  20      25
```

## Funil v3 e v3.1, referência Haiku v4
```
       modo versao                     modelo  LLM  AUTO_MATCH  AUTO_MATCH_FP  AUTO_NONMATCH_vinc_cinza  DEFERRED  UNSURE  sem_resposta_LLM  sens_observada  sens_esperada  sens_esperada_com_humano       perda
 vigilancia     v3 Claude Haiku 5.5 v4 sem MO 1410          68              1                        69         0      61                 0        0.582996       0.582996                  0.631579 5313.500000
 vigilancia   v3.1 Claude Haiku 5.5 v4 sem MO  301         111              5                         4        52      31                 0        0.821862       0.821862                  0.937247 1072.485421
confirmacao     v3 Claude Haiku 5.5 v4 sem MO  435          32              0                       107         0      25                 0        0.404858       0.404858                  0.473684 3333.750000
confirmacao   v3.1 Claude Haiku 5.5 v4 sem MO   25          79              1                        15        83       7                 0        0.651822       0.651822                  0.894130  828.183288
```

## Perda por revisor
```
       modo versao                     modelo  FP  FN  UNSURE  DEFERRED  sem_resposta_LLM  perda_observada  perda_esperada       perda  revisor_referencia
 vigilancia     v3                  Kimi K2.5   7  91      41         0                 0         5013.500        0.000000 5013.500000               False
 vigilancia   v3.1                  Kimi K2.5   7  14      15        52               209          837.460      705.455454 1542.915454               False
 vigilancia     v3 Claude Haiku 5.5 v3 com MO   1  82     262         0                 0         4724.500        0.000000 4724.500000               False
 vigilancia   v3.1 Claude Haiku 5.5 v3 com MO   5  12      34        52               209          736.460      721.505468 1457.965468               False
 vigilancia     v3 Claude Haiku 5.5 v4 sem MO  35  91      61         0                 0         5313.500        0.000000 5313.500000                True
 vigilancia   v3.1 Claude Haiku 5.5 v4 sem MO  19  14      31        52                 0          974.505       97.980421 1072.485421                True
confirmacao     v3                  Kimi K2.5   3 120       0         0                 0         2808.750        0.000000 2808.750000               False
confirmacao   v3.1                  Kimi K2.5   2  21       0        83                 7          703.090      251.090527  954.180527               False
confirmacao     v3 Claude Haiku 5.5 v3 com MO   1 115     171         0                 0         2679.750        0.000000 2679.750000               False
confirmacao   v3.1 Claude Haiku 5.5 v3 com MO   2  20       3        83                 7          686.090      266.170851  952.260851               False
confirmacao     v3 Claude Haiku 5.5 v4 sem MO   6 130      25         0                 0         3333.750        0.000000 3333.750000                True
confirmacao   v3.1 Claude Haiku 5.5 v4 sem MO   1  23       7        83                 0          650.125      178.058288  828.183288                True
```

## Lotes para revisão
```
       modo  lote_llm_v31  respondidos_haiku_v4  pendentes_haiku_v4
 vigilancia           301                   301                   0
confirmacao            25                    25                   0
```

## Diferenças frente às taxas anteriores
```
       modo                         metrica         m3v3     haiku_v4  diferenca_v4_m3v3
 vigilancia                        lote_llm   486.000000   301.000000        -185.000000
 vigilancia                   humano_direto    88.000000    52.000000         -36.000000
 vigilancia                      AUTO_MATCH   127.000000   111.000000         -16.000000
 vigilancia                   AUTO_NONMATCH 20936.000000 21173.000000         237.000000
 vigilancia vinculos_perdidos_auto_nonmatch     5.000000     4.000000          -1.000000
 vigilancia                  FP_automaticos     8.000000     5.000000          -3.000000
 vigilancia        sens_esperada_automatica     0.871020     0.875178           0.004158
 vigilancia        sens_esperada_com_humano     0.950899     0.963350           0.012452
 vigilancia            perda_esperada_total  1087.693598   989.648691         -98.044906
confirmacao                        lote_llm    15.000000    25.000000          10.000000
confirmacao                   humano_direto    99.000000    83.000000         -16.000000
confirmacao                      AUTO_MATCH    79.000000    79.000000           0.000000
confirmacao                   AUTO_NONMATCH 21444.000000 21450.000000           6.000000
confirmacao vinculos_perdidos_auto_nonmatch    15.000000    15.000000           0.000000
confirmacao                  FP_automaticos     1.000000     1.000000           0.000000
confirmacao        sens_esperada_automatica     0.639177     0.680180           0.041003
confirmacao        sens_esperada_com_humano     0.907698     0.902510          -0.005188
confirmacao            perda_esperada_total   954.829574   944.813697         -10.015877
```

## Confirmação NP
```
       modo policy_kind  max_fp_rate_auto  max_fp_expected_total  max_fn_rate_auto  C_llm  e_fp_h  e_fn_h  orcamento_humano  viavel              status  lote_llm  humano_direto  humano_via_llm_esperado  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  AUTO_MATCH_vinculos  AUTO_MATCH_FP  AUTO_NONMATCH_vinculos  ALWAYS_MATCH  ALWAYS_NONMATCH  vinculos_perdidos_auto_nonmatch  vinculos_perdidos_cinza  vinculos_perdidos_total_nonmatch  FP_automaticos  FP_auto_match  fp_esperado_auto_match_incl_guardas  taxa_fp_esperada_auto_match  fp_esperado_total  vinculos_esperados_auto_match_incl_guardas  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total     motivo  fp_esperado_guardas  n_guardas_match  taxa_minima_atingivel  n_match_na_taxa_minima  np_fp_esperado_auto  np_n_match_auto  np_taxa_fp_auto  np_fp_limite
confirmacao        cost               NaN                    NaN              0.02  0.005    0.05    0.05               inf    True custo_sem_restricao      25.0           83.0                 2.341903              85.341903        79.0        21450.0                 78.0            1.0                    15.0          78.0          39981.0                             15.0                     15.0                              18.0             1.0            1.0                             1.227239                     0.007817           2.380354                                  155.772761                  0.680180                  0.902510            944.813697        NaN                  NaN              NaN                    NaN                     NaN                  NaN              NaN              NaN           NaN
confirmacao          np             0.001                    NaN              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             1.118148             78.0               0.007405                   151.0                  NaN              NaN              NaN           NaN
confirmacao          np             0.002                    NaN              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             1.118148             78.0               0.007405                   151.0                  NaN              NaN              NaN           NaN
confirmacao          np             0.010                    NaN              0.02  0.005    0.05    0.05               inf    True              viavel      73.0          156.0                 5.189768             161.189768        97.0        21311.0                 95.0            2.0                     9.0          78.0          39981.0                              9.0                      9.0                              12.0             2.0            2.0                             1.747609                     0.009986           8.306351                                  173.252391                  0.749006                  0.929028           1417.014941        NaN             1.118148             78.0               0.007405                   151.0             1.747609            175.0         0.009986          1.75
```

## Desfecho final da aceitação automática
```
 VP_cadeia  FP_cadeia  FN_cadeia  VN_cadeia  VP_resolvidos  FP_resolvidos  FN_resolvidos  VN_resolvidos  FN_nonmatch_observado  VN_nonmatch_observado  sens_cadeia_observada  espec_cadeia_observada  pares_ao_humano  vinculos_ao_humano  nao_vinculos_ao_humano  humano_direto  humano_unsure  pares_ao_llm  llm_sem_resposta                   tipo_metrica        modo versao  sens_esperada_pre_lote31  espec_esperada_aceitacao_pre_lote31  sens_esperada_com_humano  espec_esperada_com_humano  hipotese_erro_humano  perda_unidades_humanas  perda_erro_humano_esperada  custo_llm_usd  custo_llm_usd_por_par                         custo_usd_tipo  custo_llm_usd_lote31
       144         35        103      61414            144             35             91          61365                     91                  61365               0.582996                0.999430               61                  12                      49              0             61          1410                 0 observada_aceitacao_automatica  vigilancia     v3                  0.582996                             0.999430                  0.631579                   0.999430                  0.00             5313.500000                    0.000000       3.785245               0.002685 estimativa_tarifaria_tokens_observados              0.000000
       203         19         44      61430            203             19             14          61377                     14                  61377               0.821862                0.999691               83                  30                      53             52             31           301                 0 observada_aceitacao_automatica  vigilancia   v3.1                  0.850964                             0.999714                  0.937247                   0.999648                  0.05             1072.485421                   97.980421       0.826020               0.002744 estimativa_tarifaria_tokens_observados              0.589057
       100          6        147      61443            100              6            130          61435                    130                  61435               0.404858                0.999902               25                  17                       8              0             25           435                 0 observada_aceitacao_automatica confirmacao     v3                  0.404858                             0.999902                  0.473684                   0.999902                  0.00             3333.750000                    0.000000       1.163579               0.002675 estimativa_tarifaria_tokens_observados              0.000000
       161          1         86      61448            161              1             23          61421                     23                  61421               0.651822                0.999984               90                  63                      27             83              7            25                 0 observada_aceitacao_automatica confirmacao   v3.1                  0.655806                             0.999983                  0.894130                   0.999962                  0.05              828.183288                  178.058288       0.070229               0.002809 estimativa_tarifaria_tokens_observados              0.018017
```

## Fontes v4
```json
[
  {
    "mode": "vigilancia",
    "path": "data/sprint3b/anthropic_sem_mo_v4/exp2_llm_sync_vigilancia.json",
    "sha256": "d30a30b710334874abad9ff24b9124c02b1fc33899393290d6d713f00fd776e1"
  },
  {
    "mode": "vigilancia",
    "path": "data/sprint3b/anthropic_sem_mo_v4_lote31/exp2_llm_sync_vigilancia.json",
    "sha256": "e21baf47edfdd28c3937e9c9401c3b61dd806edcc20d0b1454bdb25a4b7f870c"
  },
  {
    "mode": "confirmacao",
    "path": "data/sprint3b/anthropic_sem_mo_v4/exp2_llm_sync_confirmacao.json",
    "sha256": "a952409dd45d8b755592d16a8919ac098f709d56a6d161a6def76d70af1c4980"
  },
  {
    "mode": "confirmacao",
    "path": "data/sprint3b/anthropic_sem_mo_v4_lote31/exp2_llm_sync_confirmacao.json",
    "sha256": "b3aac9bf8d09765c95a135d907e063ecce8a47b906fbb7507e2c79fe7dd35747"
  }
]
```
Relatórios detalhados: funil_gzcmd_relatorio.md, varredura_custos_relatorio.md e np_confirmacao_relatorio.md.
A NP continua inviável em 0,001/0,002 pelas guardas obrigatórias; alterar taxas do revisor não modifica esse risco automático OOF.
