# Reavaliação com Claude Haiku 5.5, prompt v4 e dossiê sem model_outputs

A referência é somente by_mode.<modo>.models.claude-haiku-5-5 do YAML v4. O pooled misto com DeepSeek V4.1 Flash não foi usado.
Custos aprovados preservados: C_llm=0,005; C_humano=1; e_h=0,05; humano direto ilimitado.
OOF, guardas e custos permanecem fixos. O perfil v3 mantém Platt/taxas fixas/custo LLM=0,25 históricos; v3.1 usa as taxas Haiku v4.
Kimi K2.5 e Claude Haiku 5.5 v3 com MO são comparadores históricos. Suas pendências usam taxas históricas próprias, identificadas no CSV; nunca alimentam o motor novo.
A referência não é substituída pelo comparador de menor perda. Scores do motor no dossiê antigo tornam essa comparação potencialmente contaminada.
A perda mistura erros conhecidos e tempo de encaminhamento observados com perdas esperadas dos pendentes e do revisor humano imperfeito.
A política v3.1 tem cobertura Haiku v4 integral após incorporar o lote complementar. Não há inferência de respostas LLM pendentes na referência.
A projeção anterior foi preservada em colunas pre_lote31. O cenário com revisor humano continua sendo hipótese de erro de 5%, separado do desfecho automático observado.
Os deltas abaixo usam expectativas da varredura, sem replay, no mesmo ponto C_llm=0,005/e_h=0,05/orçamento infinito. Frente ao arquivo M3v3, podem combinar alterações de taxas e de perfil OOF; não isolam um efeito causal.
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
 vigilancia   v3.1 Claude Haiku 5.5 v4 sem MO  347         132             16                         5        80      31                 0        0.785425       0.785425                  0.931579 1326.768152
confirmacao     v3 Claude Haiku 5.5 v4 sem MO  435          32              0                       107         0      25                 0        0.404858       0.404858                  0.473684 3333.750000
confirmacao   v3.1 Claude Haiku 5.5 v4 sem MO   44          55              0                        19       121       6                 0        0.534413       0.534413                  0.869028  955.930365
```

## Perda por revisor
```
       modo versao                     modelo  FP  FN  UNSURE  DEFERRED  sem_resposta_LLM  perda_observada  perda_esperada       perda  revisor_referencia
 vigilancia     v3                  Kimi K2.5   7  91      41         0                 0         5013.500        0.000000 5013.500000               False
 vigilancia   v3.1                  Kimi K2.5  18  20      16        80               190         1276.785      359.773205 1636.558205               False
 vigilancia     v3 Claude Haiku 5.5 v3 com MO   1  82     262         0                 0         4724.500        0.000000 4724.500000               False
 vigilancia   v3.1 Claude Haiku 5.5 v3 com MO  16  16      53        80               190         1093.785      383.838776 1477.623776               False
 vigilancia     v3 Claude Haiku 5.5 v4 sem MO  35  91      61         0                 0         5313.500        0.000000 5313.500000                True
 vigilancia   v3.1 Claude Haiku 5.5 v4 sem MO  34  15      31        80                 0         1202.735      124.033152 1326.768152                True
confirmacao     v3                  Kimi K2.5   3 120       0         0                 0         2808.750        0.000000 2808.750000               False
confirmacao   v3.1                  Kimi K2.5   1  25       0       121                20          721.120      525.634514 1246.754514               False
confirmacao     v3 Claude Haiku 5.5 v3 com MO   1 115     171         0                 0         2679.750        0.000000 2679.750000               False
confirmacao   v3.1 Claude Haiku 5.5 v3 com MO   1  23      11       121                20          692.120      576.330955 1268.450955               False
confirmacao     v3 Claude Haiku 5.5 v4 sem MO   6 130      25         0                 0         3333.750        0.000000 3333.750000                True
confirmacao   v3.1 Claude Haiku 5.5 v4 sem MO   0  28       6       121                 0          687.220      268.710365  955.930365                True
```

## Lotes para revisão
```
       modo  lote_llm_v31  respondidos_haiku_v4  pendentes_haiku_v4
 vigilancia           347                   347                   0
confirmacao            44                    44                   0
```

## Diferenças frente às taxas anteriores
```
       modo                         metrica         m3v3     haiku_v4  diferenca_v4_m3v3
 vigilancia                        lote_llm   486.000000   347.000000        -139.000000
 vigilancia                   humano_direto    88.000000    80.000000          -8.000000
 vigilancia                      AUTO_MATCH   127.000000   132.000000           5.000000
 vigilancia                   AUTO_NONMATCH 20936.000000 21093.000000         157.000000
 vigilancia vinculos_perdidos_auto_nonmatch     5.000000     5.000000           0.000000
 vigilancia                  FP_automaticos     8.000000    16.000000           8.000000
 vigilancia        sens_esperada_automatica     0.871020     0.821610          -0.049410
 vigilancia        sens_esperada_com_humano     0.950899     0.950656          -0.000242
 vigilancia            perda_esperada_total  1087.693598  1291.415361         203.721763
confirmacao                        lote_llm    15.000000    44.000000          29.000000
confirmacao                   humano_direto    99.000000   121.000000          22.000000
confirmacao                      AUTO_MATCH    79.000000    55.000000         -24.000000
confirmacao                   AUTO_NONMATCH 21444.000000 21432.000000         -12.000000
confirmacao vinculos_perdidos_auto_nonmatch    15.000000    19.000000           4.000000
confirmacao                  FP_automaticos     1.000000     0.000000          -1.000000
confirmacao        sens_esperada_automatica     0.639177     0.556636          -0.082541
confirmacao        sens_esperada_com_humano     0.907698     0.876790          -0.030908
confirmacao            perda_esperada_total   954.829574  1121.492177         166.662603
```

## Confirmação NP
```
       modo policy_kind  max_fp_rate_auto  max_fp_expected_total  max_fn_rate_auto  C_llm  e_fp_h  e_fn_h  orcamento_humano  viavel              status  lote_llm  humano_direto  humano_via_llm_esperado  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  AUTO_MATCH_vinculos  AUTO_MATCH_FP  AUTO_NONMATCH_vinculos  ALWAYS_MATCH  ALWAYS_NONMATCH  vinculos_perdidos_auto_nonmatch  vinculos_perdidos_cinza  vinculos_perdidos_total_nonmatch  FP_automaticos  FP_auto_match  fp_esperado_auto_match_incl_guardas  taxa_fp_esperada_auto_match  fp_esperado_total  vinculos_esperados_auto_match_incl_guardas  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total     motivo  fp_esperado_guardas  n_guardas_match  taxa_minima_atingivel  n_match_na_taxa_minima  np_fp_esperado_auto  np_n_match_auto  np_taxa_fp_auto  np_fp_limite      rates_model
confirmacao        cost               NaN                    NaN              0.02  0.005    0.05    0.05               inf    True custo_sem_restricao      44.0          121.0                 3.317491             124.317491        55.0        21432.0                 55.0            0.0                    19.0          63.0          39981.0                             19.0                     19.0                              22.0             0.0            0.0                             0.826295                     0.007003           2.882196                                  117.173705                  0.556636                  0.876790           1121.492177        NaN                  NaN              NaN                    NaN                     NaN                  NaN              NaN              NaN           NaN claude-haiku-5-5
confirmacao          np             0.001                    NaN              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             0.672449             63.0               0.006113                   110.0                  NaN              NaN              NaN           NaN claude-haiku-5-5
confirmacao          np             0.002                    NaN              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             0.672449             63.0               0.006113                   110.0                  NaN              NaN              NaN           NaN claude-haiku-5-5
confirmacao          np             0.010                    NaN              0.02  0.005    0.05    0.05               inf    True              viavel     108.0          241.0                 5.970415             246.970415        70.0        21233.0                 70.0            0.0                    13.0          63.0          39981.0                             13.0                     13.0                              16.0             0.0            0.0                             1.294276                     0.009731          10.956194                                  131.705724                  0.626879                  0.901425           1859.278733        NaN             0.672449             63.0               0.006113                   110.0             1.294276            133.0         0.009731          1.33 claude-haiku-5-5
```

## Desfecho final da aceitação automática
```
 VP_cadeia  FP_cadeia  FN_cadeia  VN_cadeia  VP_resolvidos  FP_resolvidos  FN_resolvidos  VN_resolvidos  FN_nonmatch_observado  VN_nonmatch_observado  sens_cadeia_observada  espec_cadeia_observada  pares_ao_humano  vinculos_ao_humano  nao_vinculos_ao_humano  humano_direto  humano_unsure  pares_ao_llm  llm_sem_resposta                   tipo_metrica  sens_ci95_low  sens_ci95_high        modo versao  sens_com_humano_erro_05  sens_esperada_pre_lote31  espec_esperada_aceitacao_pre_lote31  sens_esperada_com_humano  espec_esperada_com_humano  hipotese_erro_humano  perda_unidades_humanas  perda_erro_humano_esperada  custo_llm_usd  custo_llm_usd_por_par                         custo_usd_tipo  custo_llm_usd_lote31
       144         35        103      61414            144             35             91          61365                     91                  61365               0.582996                0.999430               61                  12                      49              0             61          1410                 0 observada_aceitacao_automatica       0.520695        0.642755  vigilancia     v3                 0.629150                  0.582996                             0.999430                  0.631579                   0.999430                  0.00             5313.500000                    0.000000       3.785245               0.002685 estimativa_tarifaria_tokens_observados              0.000000
       194         34         53      61415            194             34             15          61342                     15                  61342               0.785425                0.999447              111                  38                      73             80             31           347                 0 observada_aceitacao_automatica       0.730063        0.832045  vigilancia   v3.1                 0.931579                  0.800981                             0.999502                  0.931579                   0.999387                  0.05             1326.768152                  124.033152       0.919832               0.002651 estimativa_tarifaria_tokens_observados              0.507729
       100          6        147      61443            100              6            130          61435                    130                  61435               0.404858                0.999902               25                  17                       8              0             25           435                 0 observada_aceitacao_automatica       0.345553        0.467078 confirmacao     v3                 0.470243                  0.404858                             0.999902                  0.473684                   0.999902                  0.00             3333.750000                    0.000000       1.163579               0.002675 estimativa_tarifaria_tokens_observados              0.000000
       132          0        115      61449            132              0             28          61409                     28                  61409               0.534413                1.000000              127                  87                      40            121              6            44                 0 observada_aceitacao_automatica       0.472155        0.595617 confirmacao   v3.1                 0.869028                  0.527159                             0.999998                  0.869028                   0.999967                  0.05              955.930365                  268.710365       0.119193               0.002709 estimativa_tarifaria_tokens_observados              0.048665
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
    "mode": "vigilancia",
    "path": "data/sprint3b/anthropic_sem_mo_v4_lote31_refrec/exp2_llm_sync_vigilancia.json",
    "sha256": "cc68b8658866627aefc25d008bb7c978df53f554cd92425b0502eb695ffd468c"
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
  },
  {
    "mode": "confirmacao",
    "path": "data/sprint3b/anthropic_sem_mo_v4_lote31_refrec/exp2_llm_sync_confirmacao.json",
    "sha256": "abc76bda740c23707ff95f674f7c32c1d87199fcec5f9a3c028e30bc804308f0"
  }
]
```
Relatórios detalhados: funil_gzcmd_relatorio.md, varredura_custos_relatorio.md e np_confirmacao_relatorio.md.
A viabilidade NP deve ser lida na tabela recalculada. A alteração do perfil OOF pode mudar o risco das guardas, embora trocar apenas taxas LLM não o altere.
