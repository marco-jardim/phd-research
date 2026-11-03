# Reavaliação offline GZ-CMD — M1/M2/M6 bayesiano

M1 preservado: PAR∈{1,2}; 5 folds externos × 5 internos, seed 42; isotônica somente no treino OOF interno.
Guarda A-: nota<5. Guarda A+ relaxada: nota≥9 e p_cal_ml≥0,9, sem exigir nascimento/município exatos.
Fora das guardas: argmin de (1-p)·C_fp, p·C_fn, C_llm+(1-p)·e_fp·C_fp+p·e_fn·C_fn+u·L_h, L_h.
L_h=C_humano+p·e_fn_h·C_fn+(1-p)·e_fp_h·C_fp. Default e_fp_h=e_fn_h=0,05: hipótese de cenário, sem concordância inter-humana medida na tese.
Unidade: revisões humanas equivalentes. C_humano=1; C_llm=0,005 (5 s / 20 min = 1/240 ≈0,00417, arredondado); C_fp/C_fn=10/50 e 100/20.
LLM somente com ganho positivo sobre a melhor alternativa não LLM; orçamento alocado por ganho decrescente. Excesso executa a melhor alternativa não LLM.
Orçamento humano limita encaminhamentos diretos, default ilimitado; UNSURE LLM é carga adicional. Ganho humano usa a melhor alternativa disponível; excesso vai a LLM com capacidade/ganho, senão AUTO_*.
AUTO_MATCH/AUTO_NONMATCH são regras explícitas de argmin, não guardas. DEFERRED ocorre quando humano minimiza a perda e há orçamento.
M3 pooled por modo alimenta a triagem. Para n<10 usa taxas fixas (u fixo default 0); denominador zero conserva fallback fixo do mapping M3.
Perda por modelo: classificações conhecidas e tempo humano têm perda observada; erro humano e LLM pendente têm perda esperada (não há gabarito de erro humano).
C_llm é somado para todas as chamadas, inclusive observadas; perda_sem_custo_llm permite comparar a definição residual M6 anterior.
Pendente não é DEFERRED: colunas tipo_perda/esperada e perda_observada/perda_esperada discriminam a estimativa.
Sensibilidade/especificidade observadas contam só decisões conhecidas; esperadas completam LLM pendentes pelas taxas condicionais de erro.
M3 só fornece u marginal: aproximado por classe e limitado a 1-e_fn/1-e_fp. Colunas *_com_humano incluem o erro humano configurado, não um humano perfeito.
As taxas M3 foram estimadas no lote antigo, não OOF, e não garantem generalização ao lote novo. Expectativas NÃO são resultados observados.
Seleção por perda mista é condicional às taxas M3 e à hipótese de erro humano, não evidência observada de superioridade.
O extrato inteiro (61.696 pares) é o universo; 1.410/435 eram lotes LLM. Validação OOF por par, não por pessoa.
K2.5/confirmação: usa explicitamente data/qw2/qw2_confirmacao_kimi.json se o original estiver ausente (protocolo pode diferir; sem inferência causal).
Nenhuma chamada LLM, PII ou alteração LaTeX. O perfil v3 + Platt preserva o resultado histórico.

## vigilancia
ANTES {'NONMATCH': 60214, 'LLM_REVIEW': 1410, 'MATCH': 72} -> DEPOIS {'NONMATCH': 60917, 'LLM_REVIEW': 486, 'MATCH': 205, 'DEFERRED': 88}
Vínculos cinzentos antes descartados=69; agora={'AUTO_MATCH': 50, 'DEFERRED': 10, 'LLM': 5, 'AUTO_NONMATCH': 4}.
Guarda ≥9: 78 pares, FP=0.
Novos pair_ids no lote LLM=387, vínculos=17.
SANIDADE: perdidos cinza=5 (≤15), LLM=486 (≤1500), FP AUTO_MATCH cinza=8 (≤10): True.
Modelo escolhido=Haiku, perda mista=905.109284; somente cobertura histórica completa elegível; K3 parcial reportado separadamente.
Decisões ≥9:
```
                n  vinc
decisao_motor          
ALWAYS_MATCH   78    78
LLM            17    15
```

## confirmacao
ANTES {'NONMATCH': 61225, 'LLM_REVIEW': 435, 'MATCH': 36} -> DEPOIS {'NONMATCH': 61425, 'MATCH': 157, 'DEFERRED': 99, 'LLM_REVIEW': 15}
Vínculos cinzentos antes descartados=107; agora={'AUTO_MATCH': 57, 'DEFERRED': 36, 'AUTO_NONMATCH': 13, 'LLM': 1}.
Guarda ≥9: 78 pares, FP=0.
Novos pair_ids no lote LLM=6, vínculos=1.
SANIDADE: perdidos cinza=15 (≤15), LLM=15 (≤1500), FP AUTO_MATCH cinza=1 (≤10): True.
Modelo escolhido=DeepSeek, perda mista=802.037670; somente cobertura histórica completa elegível; K3 parcial reportado separadamente.
Decisões ≥9:
```
                n  vinc
decisao_motor          
ALWAYS_MATCH   78    78
DEFERRED       17    15
```

## Custo versus NP — confirmação
```
       modo policy_kind  max_fp_rate_auto max_fp_expected_total  max_fn_rate_auto  C_llm  e_fp_h  e_fn_h  orcamento_humano  viavel              status  lote_llm  humano_direto  humano_via_llm_esperado  humano_total_esperado  AUTO_MATCH  AUTO_NONMATCH  AUTO_MATCH_vinculos  AUTO_MATCH_FP  AUTO_NONMATCH_vinculos  ALWAYS_MATCH  ALWAYS_NONMATCH  vinculos_perdidos_auto_nonmatch  vinculos_perdidos_cinza  vinculos_perdidos_total_nonmatch  FP_automaticos  FP_auto_match  fp_esperado_auto_match_incl_guardas  taxa_fp_esperada_auto_match  fp_esperado_total  vinculos_esperados_auto_match_incl_guardas  sens_esperada_automatica  sens_esperada_com_humano  perda_esperada_total     motivo  fp_esperado_guardas  n_guardas_match  taxa_minima_atingivel  n_match_na_taxa_minima  np_fp_esperado_auto  np_n_match_auto  np_taxa_fp_auto  np_fp_limite
confirmacao        cost               NaN                  None              0.02  0.005    0.05    0.05               inf    True custo_sem_restricao      15.0           99.0                 2.585928             101.585928        79.0        21444.0                 78.0            1.0                    15.0          78.0          39981.0                             15.0                     15.0                              18.0             1.0            1.0                             1.227239                     0.007817           2.619835                                  155.772761                  0.639177                  0.907698            954.829574        NaN                  NaN              NaN                    NaN                     NaN                  NaN              NaN              NaN           NaN
confirmacao          np             0.001                  None              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             1.118148             78.0               0.007405                   151.0                  NaN              NaN              NaN           NaN
confirmacao          np             0.002                  None              0.02  0.005    0.05    0.05               inf   False            inviavel       NaN            NaN                      NaN                    NaN         NaN            NaN                  NaN            NaN                     NaN           NaN              NaN                              NaN                      NaN                               NaN             NaN            NaN                                  NaN                          NaN                NaN                                         NaN                       NaN                       NaN                   NaN guardas_fp             1.118148             78.0               0.007405                   151.0                  NaN              NaN              NaN           NaN
confirmacao          np             0.010                  None              0.02  0.005    0.05    0.05               inf    True              viavel      57.0          172.0                15.600942             187.600942        97.0        21311.0                 95.0            2.0                     9.0          78.0          39981.0                              9.0                      9.0                              12.0             2.0            2.0                             1.747609                     0.009986           8.320382                                  173.252391                  0.708003                  0.934216           1420.073280        NaN             1.118148             78.0               0.007405                   151.0             1.747609            175.0         0.009986          1.75
```
Guardas integram o limite. NA indica inviabilidade, não desempenho perfeito ou zero FP.
## Variante de nascimento
```
 pares_ge9  fora_guarda_atual  guarda_atual_n  guarda_atual_FP  variante_n  variante_vinculos  variante_FP  recuperados_dos_17  FP_recuperados  variante_fp_esperado  criterio_zero_fp_atendido  adotada
        95                 17              78                0          22                 22            0                   0               0              0.092256                       True    False
```
diff_ano>0 com ambos os anos conhecidos é conflito forte. p≥0,95 é subconjunto da guarda p≥0,9; recupera zero dos 17, logo não foi adotada.
## FUNIL
```
       modo versao faixa           etapa   decisao_motor     n  vinc
 vigilancia     v3 TODAS pares_revisados           TODAS 61696   247
 vigilancia     v3 TODAS   decisao_motor    ALWAYS_MATCH     4     4
 vigilancia     v3 TODAS   decisao_motor ALWAYS_NONMATCH 13233     0
 vigilancia     v3 TODAS   decisao_motor      AUTO_MATCH    68    67
 vigilancia     v3 TODAS   decisao_motor   AUTO_NONMATCH 46981    72
 vigilancia     v3 TODAS   decisao_motor             LLM  1410   104
 vigilancia     v3    <5 pares_revisados           TODAS 39981     3
 vigilancia     v3    <5   decisao_motor ALWAYS_NONMATCH 10411     0
 vigilancia     v3    <5   decisao_motor   AUTO_NONMATCH 29570     3
 vigilancia     v3 [5,9) pares_revisados           TODAS 21620   151
 vigilancia     v3 [5,9)   decisao_motor ALWAYS_NONMATCH  2821     0
 vigilancia     v3 [5,9)   decisao_motor   AUTO_NONMATCH 17411    69
 vigilancia     v3 [5,9)   decisao_motor             LLM  1388    82
 vigilancia     v3   >=9 pares_revisados           TODAS    95    93
 vigilancia     v3   >=9   decisao_motor    ALWAYS_MATCH     4     4
 vigilancia     v3   >=9   decisao_motor ALWAYS_NONMATCH     1     0
 vigilancia     v3   >=9   decisao_motor      AUTO_MATCH    68    67
 vigilancia     v3   >=9   decisao_motor             LLM    22    22
 vigilancia   v3.1 TODAS pares_revisados           TODAS 61696   247
 vigilancia   v3.1 TODAS   decisao_motor    ALWAYS_MATCH    78    78
 vigilancia   v3.1 TODAS   decisao_motor ALWAYS_NONMATCH 39981     3
 vigilancia   v3.1 TODAS   decisao_motor      AUTO_MATCH   127   119
 vigilancia   v3.1 TODAS   decisao_motor   AUTO_NONMATCH 20936     5
 vigilancia   v3.1 TODAS   decisao_motor        DEFERRED    88    20
 vigilancia   v3.1 TODAS   decisao_motor             LLM   486    22
 vigilancia   v3.1    <5 pares_revisados           TODAS 39981     3
 vigilancia   v3.1    <5   decisao_motor ALWAYS_NONMATCH 39981     3
 vigilancia   v3.1 [5,9) pares_revisados           TODAS 21620   151
 vigilancia   v3.1 [5,9)   decisao_motor      AUTO_MATCH   127   119
 vigilancia   v3.1 [5,9)   decisao_motor   AUTO_NONMATCH 20936     5
 vigilancia   v3.1 [5,9)   decisao_motor        DEFERRED    88    20
 vigilancia   v3.1 [5,9)   decisao_motor             LLM   469     7
 vigilancia   v3.1   >=9 pares_revisados           TODAS    95    93
 vigilancia   v3.1   >=9   decisao_motor    ALWAYS_MATCH    78    78
 vigilancia   v3.1   >=9   decisao_motor             LLM    17    15
confirmacao     v3 TODAS pares_revisados           TODAS 61696   247
confirmacao     v3 TODAS   decisao_motor    ALWAYS_MATCH     4     4
confirmacao     v3 TODAS   decisao_motor ALWAYS_NONMATCH 13233     0
confirmacao     v3 TODAS   decisao_motor      AUTO_MATCH    32    32
confirmacao     v3 TODAS   decisao_motor   AUTO_NONMATCH 47992   110
confirmacao     v3 TODAS   decisao_motor             LLM   435   101
confirmacao     v3    <5 pares_revisados           TODAS 39981     3
confirmacao     v3    <5   decisao_motor ALWAYS_NONMATCH 10411     0
confirmacao     v3    <5   decisao_motor   AUTO_NONMATCH 29570     3
confirmacao     v3 [5,9) pares_revisados           TODAS 21620   151
confirmacao     v3 [5,9)   decisao_motor ALWAYS_NONMATCH  2821     0
confirmacao     v3 [5,9)   decisao_motor   AUTO_NONMATCH 18422   107
confirmacao     v3 [5,9)   decisao_motor             LLM   377    44
confirmacao     v3   >=9 pares_revisados           TODAS    95    93
confirmacao     v3   >=9   decisao_motor    ALWAYS_MATCH     4     4
confirmacao     v3   >=9   decisao_motor ALWAYS_NONMATCH     1     0
confirmacao     v3   >=9   decisao_motor      AUTO_MATCH    32    32
confirmacao     v3   >=9   decisao_motor             LLM    58    57
confirmacao   v3.1 TODAS pares_revisados           TODAS 61696   247
confirmacao   v3.1 TODAS   decisao_motor    ALWAYS_MATCH    78    78
confirmacao   v3.1 TODAS   decisao_motor ALWAYS_NONMATCH 39981     3
confirmacao   v3.1 TODAS   decisao_motor      AUTO_MATCH    79    78
confirmacao   v3.1 TODAS   decisao_motor   AUTO_NONMATCH 21444    15
confirmacao   v3.1 TODAS   decisao_motor        DEFERRED    99    69
confirmacao   v3.1 TODAS   decisao_motor             LLM    15     4
confirmacao   v3.1    <5 pares_revisados           TODAS 39981     3
confirmacao   v3.1    <5   decisao_motor ALWAYS_NONMATCH 39981     3
confirmacao   v3.1 [5,9) pares_revisados           TODAS 21620   151
confirmacao   v3.1 [5,9)   decisao_motor      AUTO_MATCH    79    78
confirmacao   v3.1 [5,9)   decisao_motor   AUTO_NONMATCH 21444    15
confirmacao   v3.1 [5,9)   decisao_motor        DEFERRED    82    54
confirmacao   v3.1 [5,9)   decisao_motor             LLM    15     4
confirmacao   v3.1   >=9 pares_revisados           TODAS    95    93
confirmacao   v3.1   >=9   decisao_motor    ALWAYS_MATCH    78    78
confirmacao   v3.1   >=9   decisao_motor        DEFERRED    17    15
```

## METRICAS
```
       modo versao   modelo  pares  vinc  LLM  AUTO_MATCH  AUTO_MATCH_FP  AUTO_NONMATCH_vinc_cinza  LLM_excecao_ge9  cinza_sens_observada  cinza_sens_esperada  FP  FN  TP    TN  UNSURE  DEFERRED  DEFERRED_motor  sem_resposta_LLM  esperada tipo_perda  perda_observada  perda_esperada       perda  custo_llm  perda_sem_custo_llm  perda_erro_humano_esperada  humanos_encaminhados_observados  FP_esperados_pendentes  FN_esperados_pendentes  UNSURE_esperados_pendentes  sens_observada  sens_esperada  sens_esperada_com_humano  espec_observada  espec_esperada  espec_esperada_com_humano  sens_inferior  sens_superior  espec_inferior  espec_superior
 vigilancia     v3    Haiku  61696   247 1410          68              1                        69               22              0.105960             0.105960   1  82 106 61245     262         0               0                 0     False  observada         4724.500        0.000000 4724.500000    352.500          4372.000000                    0.000000                              262                0.000000                0.000000                     0.00000        0.429150       0.429150                  0.668016         0.996680        0.996680                   0.999984       0.429150       0.668016        0.996680        0.999984
 vigilancia   v3.1    Haiku  61696   247  486         127              8                         5               17              0.788079             0.811381   8  10 200 60976      27        88              88               387      True      mista          695.495      209.614284  905.109284      2.430           902.679284                  118.581602                              115                3.325139                0.927936                     9.44950        0.809717       0.865920                  0.949296         0.992303        0.998144                   0.999732       0.809717       0.959514        0.992303        0.999870
confirmacao     v3 DeepSeek  61696   247  435          32              0                       107               58              0.079470             0.079470   2 118 105 61402      69         0               0                 0     False  observada         2737.750        0.000000 2737.750000    108.750          2629.000000                    0.000000                               69                0.000000                0.000000                     0.00000        0.425101       0.425101                  0.522267         0.999235        0.999235                   0.999967       0.425101       0.522267        0.999235        0.999967
confirmacao   v3.1 DeepSeek  61696   247   15          79              1                        15                0              0.523179             0.526184   1  19 157 61411       3        99              99                 6      True      mista          582.045      219.992670  802.037670      0.075           801.962670                  213.864608                              102                0.000000                0.291024                     0.27758        0.635628       0.637465                  0.906873         0.999382        0.999459                   0.999958       0.635628       0.923077        0.999382        0.999984
```

## PERDAS
```
       modo          modelo versao  C_fp  C_fn  C_llm  C_humano  e_fp_h  e_fn_h orcamento_humano  disponivel  respostas_lote_antigo  lote_antigo  fonte_alternativa  cobertura_completa_antigo  respostas_lote_atual   FP    FN    TP      TN  UNSURE  DEFERRED  DEFERRED_motor  sem_resposta_LLM esperada tipo_perda  perda_observada  perda_esperada       perda  custo_llm  perda_sem_custo_llm  perda_erro_humano_esperada  humanos_encaminhados_observados  FP_esperados_pendentes  FN_esperados_pendentes  UNSURE_esperados_pendentes  sens_observada  sens_esperada  sens_esperada_com_humano  espec_observada  espec_esperada  espec_esperada_com_humano  sens_inferior  sens_superior  espec_inferior  espec_superior  n_comum  perda_coorte_comum  selecionado
 vigilancia       Kimi K2.5     v3  10.0  50.0  0.250       1.0    0.00    0.00             None        True                   1410         1410              False                       True                1410.0  7.0  91.0 150.0 61407.0    41.0       0.0             0.0               0.0    False  observada         5013.500        0.000000 5013.500000    352.500          4661.000000                    0.000000                             41.0                0.000000                0.000000                    0.000000        0.607287       0.607287                  0.631579         0.999317        0.999317                   0.999886       0.607287       0.631579        0.999317        0.999886   1410.0         1403.500000        False
 vigilancia       Kimi K2.5   v3.1  10.0  50.0  0.005       1.0    0.05    0.05             None        True                   1410         1410              False                       True                  99.0 13.0  10.0 200.0 60989.0     9.0      88.0            88.0             387.0     True      mista          727.495      186.258465  913.753465      2.430           911.323465                  100.749103                             97.0                3.325139                0.989167                    0.864641        0.809717       0.872679                  0.949634         0.992514        0.998468                   0.999671       0.809717       0.959514        0.992514        0.999788     99.0          165.901033        False
 vigilancia        DeepSeek     v3  10.0  50.0  0.250       1.0    0.00    0.00             None        True                   1410         1410              False                       True                1410.0  4.0 114.0 120.0 61398.0    60.0       0.0             0.0               0.0    False  observada         6152.500        0.000000 6152.500000    352.500          5800.000000                    0.000000                             60.0                0.000000                0.000000                    0.000000        0.485830       0.485830                  0.538462         0.999170        0.999170                   0.999935       0.485830       0.538462        0.999170        0.999935   1410.0         2542.500000        False
 vigilancia        DeepSeek   v3.1  10.0  50.0  0.005       1.0    0.05    0.05             None        True                   1410         1410              False                       True                  99.0  9.0  10.0 200.0 60988.0    14.0      88.0            88.0             387.0     True      mista          692.495      242.586249  935.081249      2.430           932.651249                  103.876913                            102.0                3.325139                2.068997                    0.073118        0.809717       0.865425                  0.942348         0.992498        0.998464                   0.999733       0.809717       0.959514        0.992498        0.999854     99.0          134.430448        False
 vigilancia           Haiku     v3  10.0  50.0  0.250       1.0    0.00    0.00             None        True                   1410         1410              False                       True                1410.0  1.0  82.0 106.0 61245.0   262.0       0.0             0.0               0.0    False  observada         4724.500        0.000000 4724.500000    352.500          4372.000000                    0.000000                            262.0                0.000000                0.000000                    0.000000        0.429150       0.429150                  0.668016         0.996680        0.996680                   0.999984       0.429150       0.668016        0.996680        0.999984   1410.0         1114.500000        False
 vigilancia           Haiku   v3.1  10.0  50.0  0.005       1.0    0.05    0.05             None        True                   1410         1410              False                       True                  99.0  8.0  10.0 200.0 60976.0    27.0      88.0            88.0             387.0     True      mista          695.495      209.614284  905.109284      2.430           902.679284                  118.581602                            115.0                3.325139                0.927936                    9.449500        0.809717       0.865920                  0.949296         0.992303        0.998144                   0.999732       0.809717       0.959514        0.992303        0.999870     99.0          144.670446         True
 vigilancia Kimi K3 parcial     v3  10.0  50.0  0.250       1.0    0.00    0.00             None       False                      0         1410              False                      False                   NaN  NaN   NaN   NaN     NaN     NaN       NaN             NaN               NaN      NaN        NaN              NaN             NaN         NaN        NaN                  NaN                         NaN                              NaN                     NaN                     NaN                         NaN             NaN            NaN                       NaN              NaN             NaN                        NaN            NaN            NaN             NaN             NaN      NaN                 NaN        False
 vigilancia Kimi K3 parcial   v3.1  10.0  50.0  0.005       1.0    0.05    0.05             None       False                      0         1410              False                      False                   NaN  NaN   NaN   NaN     NaN     NaN       NaN             NaN               NaN      NaN        NaN              NaN             NaN         NaN        NaN                  NaN                         NaN                              NaN                     NaN                     NaN                         NaN             NaN            NaN                       NaN              NaN             NaN                        NaN            NaN            NaN             NaN             NaN      NaN                 NaN        False
confirmacao       Kimi K2.5     v3 100.0  20.0  0.250       1.0    0.00    0.00             None        True                    435          435               True                       True                 435.0  3.0 120.0 127.0 61446.0     0.0       0.0             0.0               0.0    False  observada         2808.750        0.000000 2808.750000    108.750          2700.000000                    0.000000                              0.0                0.000000                0.000000                    0.000000        0.514170       0.514170                  0.514170         0.999951        0.999951                   0.999951       0.514170       0.514170        0.999951        0.999951    343.0          565.750000        False
confirmacao       Kimi K2.5   v3.1 100.0  20.0  0.005       1.0    0.05    0.05             None        True                    435          435               True                       True                   9.0  1.0  20.0 157.0 61413.0     0.0      99.0            99.0               6.0     True      mista          599.045      204.751359  803.796359      0.075           803.721359                  200.064973                             99.0                0.000000                0.232819                    0.000000        0.635628       0.638057                  0.903441         0.999414        0.999496                   0.999959       0.635628       0.919028        0.999414        0.999984      6.0           40.030000        False
confirmacao        DeepSeek     v3 100.0  20.0  0.250       1.0    0.00    0.00             None        True                    435          435              False                       True                 435.0  2.0 118.0 105.0 61402.0    69.0       0.0             0.0               0.0    False  observada         2737.750        0.000000 2737.750000    108.750          2629.000000                    0.000000                             69.0                0.000000                0.000000                    0.000000        0.425101       0.425101                  0.522267         0.999235        0.999235                   0.999967       0.425101       0.522267        0.999235        0.999967    343.0          483.750000        False
confirmacao        DeepSeek   v3.1 100.0  20.0  0.005       1.0    0.05    0.05             None        True                    435          435              False                       True                   9.0  1.0  19.0 157.0 61411.0     3.0      99.0            99.0               6.0     True      mista          582.045      219.992670  802.037670      0.075           801.962670                  213.864608                            102.0                0.000000                0.291024                    0.277580        0.635628       0.637465                  0.906873         0.999382        0.999459                   0.999958       0.635628       0.923077        0.999382        0.999984      6.0           29.905000         True
confirmacao           Haiku     v3 100.0  20.0  0.250       1.0    0.00    0.00             None        True                    435          435              False                       True                 435.0  1.0 115.0  91.0 61318.0   171.0       0.0             0.0               0.0    False  observada         2679.750        0.000000 2679.750000    108.750          2571.000000                    0.000000                            171.0                0.000000                0.000000                    0.000000        0.368421       0.368421                  0.534413         0.997868        0.997868                   0.999984       0.368421       0.534413        0.997868        0.999984    343.0          403.750000        False
confirmacao           Haiku   v3.1 100.0  20.0  0.005       1.0    0.05    0.05             None        True                    435          435              False                       True                   9.0  1.0  20.0 156.0 61411.0     3.0      99.0            99.0               6.0     True      mista          602.045      227.954213  829.999213      0.075           829.924213                  222.253989                            102.0                0.000000                0.174614                    2.177936        0.631579       0.632943                  0.903570         0.999382        0.999433                   0.999956       0.631579       0.919028        0.999382        0.999984      6.0           44.530000        False
confirmacao Kimi K3 parcial     v3 100.0  20.0  0.250       1.0    0.00    0.00             None        True                    343          435              False                      False                 343.0  3.0 126.0  93.0 61363.0    19.0       0.0             0.0              92.0     True      mista         2924.750      260.359382 3185.109382    108.750          3076.359382                    0.000000                             19.0                1.506089                4.129394                    4.162597        0.376518       0.420816                  0.483941         0.998600        0.999817                   0.999941       0.376518       0.489879        0.998600        0.999951    343.0          724.750000        False
confirmacao Kimi K3 parcial   v3.1 100.0  20.0  0.005       1.0    0.05    0.05             None        True                    343          435              False                      False                   6.0  1.0  19.0 156.0 61411.0     1.0      99.0            99.0               9.0     True      mista          580.030      222.295009  802.325009      0.075           802.250009                  205.505704                            100.0                0.070636                0.471580                    0.249128        0.631579       0.635554                  0.905367         0.999382        0.999493                   0.999958       0.631579       0.923077        0.999382        0.999984      6.0           25.405000        False
```

## REFERENCIA ORIGINAL CINZA
```
       modo    modelo  cinza_vinc  FP  FN  TP    TN  UNSURE  DEFERRED  DEFERRED_motor  sem_resposta_LLM  esperada tipo_perda  perda_observada  perda_esperada   perda  custo_llm  perda_sem_custo_llm  perda_erro_humano_esperada  humanos_encaminhados_observados  FP_esperados_pendentes  FN_esperados_pendentes  UNSURE_esperados_pendentes  sens_observada  sens_esperada  sens_esperada_com_humano  espec_observada  espec_esperada  espec_esperada_com_humano  sens_inferior  sens_superior  espec_inferior  espec_superior
 vigilancia Kimi K2.5         151   6  88  57 21428      41         0               0                 0     False  observada          4848.00             0.0 4848.00     347.00               4501.0                         0.0                               41                     0.0                     0.0                         0.0        0.377483       0.377483                  0.417219         0.998090        0.998090                   0.999721       0.377483       0.417219        0.998090        0.999721
confirmacao  DeepSeek         151   1 115  12 21423      69         0               0                 0     False  observada          2563.25             0.0 2563.25      94.25               2469.0                         0.0                               69                     0.0                     0.0                         0.0        0.079470       0.079470                  0.238411         0.997857        0.997857                   0.999953       0.079470       0.238411        0.997857        0.999953
```
