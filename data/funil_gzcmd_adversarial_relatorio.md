# Auditoria A1/A3/A5/A6

## Verificação da entrega

`python -m pytest tests -q`: **175 passed, 2 warnings in 9.95s**.
Os dois avisos são do statsmodels no teste `test_loglinear_known_missing_cell`
(divisão por zero e separação perfeita), sem supressão. Seis testes novos
cobrem Wilson, denominadores com UNSURE, reestimação e fallback, formatação,
reconciliação aditiva e especificidade com respostas pendentes.

Auditorias adicionais confirmaram os 26 artefatos históricos preservados,
as contagens de treino leave-out (1.318 e 417), tabelas com ICs, formatos
numéricos e PDFs válidos das três figuras. `git diff --check` não identificou
erros de espaço nos arquivos versionados desta alteração; o aviso de finais
de linha no Windows é esperado. Nenhuma compilação LaTeX foi executada.

O leave-out ainda seleciona 18/2 pares da coorte remanescente de estimação.
Não é validação independente integral. Na confirmação, três novos selecionados
continuam sem resposta LLM existente e foram mantidos como pendentes.

Sem novas chamadas LLM. Calibração OOF preservada. Taxas vigentes não foram alteradas.
Leave-out exclui os 92/18 identificadores reutilizados da coorte antiga do respectivo modo, pois as taxas são específicas por modo. Não usa suplementos para estimar taxas nem mistura os modos.
Sensibilidade/especificidade do estágio LLM excluem UNSURE; sens_todos_vinculos mantém todos os vínculos no denominador.
UNSURE permanece encaminhamento humano. As projeções com humano pressupõem erro de 5%, não medido.
A substituição dos reutilizados por expectativas é análise de sensibilidade, não IC nem avaliação independente completa.
Mesmo o lote novo foi selecionado pela política informada por taxas antigas; leave-out testa a influência direta da reutilização, não elimina toda seleção adaptativa.
O leave-out é uma perturbação única, não uma validação cruzada: novos selecionados podem pertencer à coorte restante usada na reestimação. A coluna llm_ainda_nas_taxas quantifica essa sobreposição residual; não se alega independência integral da variante.
Se o leave-out selecionar pares sem resposta existente, a projeção e os limites são reportados, não acertos inventados.
A5: 93,7%/89,4% são projeções humanas pós-lote; pré-lote eram 94,55%/89,52%. Não são o mesmo momento.
O efeito de arredondar o custo é um contrafactual separado: ambas as saídas publicadas usam 0,005, não se soma esse efeito à reconciliação.
A1: scripts/generate_missing_figures.py, função fig_score_bands, lê df['nota final'] e pd.cut nessa coluna. Contagens confirmadas no extrato: 39.981/21.620/95.
Figura cascata: escolhida a apresentação apenas dos ramos e contagens. A legenda do capítulo deve deixar de citar custo e sensibilidade com humano; esses indicadores permanecem na tabela. Capítulos não foram editados.
ICs Wilson são descritivos, sem correção por dependência entre pares ou seleção da política.

## Arquivos alterados ou gerados nesta auditoria
Código: scripts/auditar_funil_adversarial.py; scripts/avaliar_reruns_exp2.py; scripts/fig_varredura_custos.py; scripts/fig_cascata_v31.py; scripts/generate_missing_figures.py; gzcmd/intervals.py; gzcmd/final_funnel.py; tests/test_auditoria_adversarial.py; gzcmd/M_auditoria_adversarial.md.
Dados: data/funil_gzcmd_leaveout.csv; data/funil_gzcmd_leaveout_taxas.csv; data/funil_gzcmd_novos_metricas.csv; data/varredura_vs_cascata.csv; data/tab_score_bands_proveniencia.csv; data/funil_gzcmd_final_metricas.csv; data/funil_gzcmd_adversarial_metadata.json; este relatório.
Tabelas: text/latex/tables/tab_cap8_reruns.tex; tab_cap8_cascata_v31.tex; tab_score_bands.tex; tab_qw2_confusion.tex.
Figuras PDF/PNG em text/latex/figures/: fig_cap8_varredura_custos; fig_cap8_cascata_v31; fig_cap8_np_confirmacao.
Frase para o capítulo (vigilancia): A varredura prospectiva estima sensibilidade automática de 87,5% e, sob erro humano hipotético de 5,0%, 96,3%; a cascata pré-lote estimava 85,1% e 94,5%, respectivamente, pois incorporava respostas observadas de 92 pares reutilizados em vez de suas expectativas por banda, diferença de 2,4214 e 1,7879 pontos percentuais, sem contribuição de orçamento ou arredondamento do custo na comparação do ponto aprovado.
Frase para o capítulo (confirmacao): A varredura prospectiva estima sensibilidade automática de 68,0% e, sob erro humano hipotético de 5,0%, 90,3%; a cascata pré-lote estimava 65,6% e 89,5%, respectivamente, pois incorporava respostas observadas de 18 pares reutilizados em vez de suas expectativas por banda, diferença de 2,4374 e 0,7318 pontos percentuais, sem contribuição de orçamento ou arredondamento do custo na comparação do ponto aprovado.

## funil_gzcmd_leaveout.csv
```
       modo                               cenario       n  llm  reutilizados_original  excluidos_taxas  llm_ainda_nas_taxas  llm_adicionados  llm_removidos  acoes_alteradas  inteiramente_observado  FP  FN  TP    TN  UNSURE  DEFERRED  DEFERRED_motor  sem_resposta_LLM  esperada tipo_perda  perda_observada  perda_esperada       perda  custo_llm  perda_sem_custo_llm  perda_erro_humano_esperada  humanos_encaminhados_observados  FP_esperados_pendentes  FN_esperados_pendentes  UNSURE_esperados_pendentes  sens_observada  sens_esperada  sens_esperada_com_humano  espec_observada  espec_esperada  espec_esperada_com_humano  sens_inferior  sens_superior  espec_inferior  espec_superior  FP_aceitacao_esperados  espec_aceitacao_esperada  espec_aceitacao_inferior  espec_aceitacao_superior  VP_cadeia  FP_cadeia  FN_cadeia  VN_cadeia  VP_resolvidos  FP_resolvidos  FN_resolvidos  VN_resolvidos  FN_nonmatch_observado  VN_nonmatch_observado  sens_cadeia_observada  espec_cadeia_observada  pares_ao_humano  vinculos_ao_humano  nao_vinculos_ao_humano  humano_direto  humano_unsure  pares_ao_llm  llm_sem_resposta                   tipo_metrica
 vigilancia                    original_observado 61696.0  301                     92              0.0                 92.0              0.0            0.0              0.0                    True  19  14 203 61377      31        52              52                 0      True      mista          974.505       97.980421 1072.485421      1.505          1070.980421                   97.980421                               83                0.000000                0.000000                    0.000000        0.821862       0.821862                  0.937247         0.998828        0.998828                   0.999648       0.821862       0.943320        0.998828        0.999691               19.000000                  0.999691                  0.999691                  0.999691      203.0       19.0       44.0    61430.0          203.0           19.0           14.0        61377.0                   14.0                61377.0               0.821862                0.999691             83.0                30.0                    53.0           52.0           31.0         301.0               0.0 observada_aceitacao_automatica
 vigilancia                              leaveout 61696.0  314                     92             92.0                 18.0             18.0            5.0             23.0                    True  17  14 205 61380      32        48              48                 0      True      mista          951.570       92.861373 1044.431373      1.570          1042.861373                   92.861373                               80                0.000000                0.000000                    0.000000        0.829960       0.829960                  0.937652         0.998877        0.998877                   0.999681       0.829960       0.943320        0.998877        0.999723               17.000000                  0.999723                  0.999723                  0.999723      205.0       17.0       42.0    61432.0          205.0           17.0           14.0        61380.0                   14.0                61380.0               0.829960                0.999723             80.0                28.0                    52.0           48.0           32.0         314.0               0.0 observada_aceitacao_automatica
 vigilancia reutilizados_substituidos_expectativa     NaN  301                     92              NaN                  NaN              NaN            NaN              NaN                   False  13   9 201 61303      26        52              52                92      True      mista          659.045      142.716350  801.761350      1.505           800.256350                   97.325184                               78                2.381330                0.335466                    4.344557        0.813765       0.846076                  0.955126         0.997624        0.998873                   0.999707       0.813765       0.963563        0.997624        0.999788               15.279504                  0.999751                  0.998438                  0.999788        NaN        NaN        NaN        NaN            NaN            NaN            NaN            NaN                    NaN                    NaN                    NaN                     NaN              NaN                 NaN                     NaN            NaN            NaN           NaN               NaN                            NaN
confirmacao                    original_observado 61696.0   25                     18              0.0                 18.0              0.0            0.0              0.0                    True   1  23 161 61421       7        83              83                 0      True      mista          650.125      178.058288  828.183288      0.125           828.058288                  178.058288                               90                0.000000                0.000000                    0.000000        0.651822       0.651822                  0.894130         0.999544        0.999544                   0.999962       0.651822       0.906883        0.999544        0.999984                1.000000                  0.999984                  0.999984                  0.999984      161.0        1.0       86.0    61448.0          161.0            1.0           23.0        61421.0                   23.0                61421.0               0.651822                0.999984             90.0                63.0                    27.0           83.0            7.0          25.0               0.0 observada_aceitacao_automatica
confirmacao                              leaveout 61696.0   30                     18             18.0                  2.0              5.0            0.0              5.0                   False   2  23 161 61417       7        83              83                 3      True      mista          750.135      183.284250  933.419250      0.150           933.269250                  178.362288                               90                0.040796                0.038095                    0.065455        0.651822       0.651822                  0.894130         0.999479        0.999526                   0.999945       0.651822       0.906883        0.999479        0.999967                2.044776                  0.999967                  0.999919                  0.999967        NaN        NaN        NaN        NaN            NaN            NaN            NaN            NaN                    NaN                    NaN                    NaN                     NaN              NaN                 NaN                     NaN            NaN            NaN           NaN               NaN                            NaN
confirmacao reutilizados_substituidos_expectativa     NaN   25                     18              NaN                  NaN              NaN            NaN              NaN                   False   1  19 158 61416       1        83              83                18      True      mista          564.035      208.784103  772.819103      0.125           772.694103                  166.442986                               84                0.114098                1.447608                    1.889209        0.639676       0.676196                  0.901447         0.999463        0.999538                   0.999960       0.639676       0.923077        0.999463        0.999984                1.076539                  0.999982                  0.999902                  0.999984        NaN        NaN        NaN        NaN            NaN            NaN            NaN            NaN                    NaN                    NaN                    NaN                     NaN              NaN                 NaN                     NaN            NaN            NaN           NaN               NaN                            NaN
```

## funil_gzcmd_leaveout_taxas.csv
```
       modo  cenario      band   n  positivos  negativos  FP  FN  UNSURE     e_fp     e_fn        u  fallback_n_lt10  fallback_fp  fallback_fn
 vigilancia original  grey_low   0          0          0   0   0       0 0.080000 0.120000 0.000000             True         True         True
 vigilancia original  grey_mid 362          5        357   8   0       9 0.022409 0.000000 0.024862            False        False        False
 vigilancia original grey_high 930         43        887  24   4      42 0.027057 0.093023 0.045161            False        False        False
 vigilancia original near_high  96         34         62   2  10       7 0.032258 0.294118 0.072917            False        False        False
 vigilancia original       low   0          0          0   0   0       0 0.100000 0.150000 0.000000             True         True         True
 vigilancia original      high  22         22          0   0   5       3 0.010000 0.227273 0.136364            False         True        False
 vigilancia leaveout  grey_low   0          0          0   0   0       0 0.080000 0.120000 0.000000             True         True         True
 vigilancia leaveout  grey_mid 344          1        343   8   0       6 0.023324 0.000000 0.017442            False        False        False
 vigilancia leaveout grey_high 876         39        837  18   0      40 0.021505 0.000000 0.045662            False        False        False
 vigilancia leaveout near_high  76         33         43   2   9       7 0.046512 0.272727 0.092105            False        False        False
 vigilancia leaveout       low   0          0          0   0   0       0 0.100000 0.150000 0.000000             True         True         True
 vigilancia leaveout      high  22         22          0   0   5       3 0.010000 0.227273 0.136364            False         True        False
confirmacao original  grey_low   0          0          0   0   0       0 0.080000 0.120000 0.000000             True         True         True
confirmacao original  grey_mid   0          0          0   0   0       0 0.060000 0.100000 0.000000             True         True         True
confirmacao original grey_high 281         10        271   4   2       8 0.014760 0.200000 0.028470            False        False        False
confirmacao original near_high  96         34         62   2   8       8 0.032258 0.235294 0.083333            False        False        False
confirmacao original       low   0          0          0   0   0       0 0.100000 0.150000 0.000000             True         True         True
confirmacao original      high  58         57          1   0  10       9 0.000000 0.175439 0.155172            False        False        False
confirmacao leaveout  grey_low   0          0          0   0   0       0 0.080000 0.120000 0.000000             True         True         True
confirmacao leaveout  grey_mid   0          0          0   0   0       0 0.060000 0.100000 0.000000             True         True         True
confirmacao leaveout grey_high 275          7        268   4   1       6 0.014925 0.142857 0.021818            False        False        False
confirmacao leaveout near_high  94         33         61   2   7       8 0.032787 0.212121 0.085106            False        False        False
confirmacao leaveout       low   0          0          0   0   0       0 0.100000 0.150000 0.000000             True         True         True
confirmacao leaveout      high  48         48          0   0   8       5 0.010000 0.166667 0.104167            False         True        False
```

## funil_gzcmd_novos_metricas.csv
```
       modo        coorte   n  VP  FP  FN  VN  UNSURE  UNSURE_pos  UNSURE_neg     sens   espec  sens_denominador  espec_denominador  sens_todos_vinculos
 vigilancia somente_novos 209  17   8   2 156      26           6          20 0.894737 0.95122                19                164                 0.68
confirmacao somente_novos   7   2   0   1   3       1           1           0 0.666667 1.00000                 3                  3                 0.50
```

## varredura_vs_cascata.csv
```
       modo                  metrica  cascata_pre_lote  cascata_pos_lote  varredura  efeito_reutilizacao_pp  efeito_orcamento_pp  diferenca_pre_lote_pp  efeito_respostas_novas_pp  efeito_custo_exato_vs_arredondado_pp  erro_exibicao_1_casa_varredura_pp  llm_orcamento_diferencas  llm_custo_exato_diferencas
 vigilancia sens_esperada_automatica          0.850964          0.821862   0.875178                2.421368                  0.0               2.421368                  -2.910193                                   0.0                          -0.017796                         0                           0
 vigilancia sens_esperada_com_humano          0.945471          0.937247   0.963350                1.787910                  0.0               1.787910                  -0.822433                                   0.0                          -0.035040                         0                           0
confirmacao sens_esperada_automatica          0.655806          0.651822   0.680180                2.437435                  0.0               2.437435                  -0.398372                                   0.0                          -0.017993                         0                           0
confirmacao sens_esperada_com_humano          0.895192          0.894130   0.902510                0.731779                  0.0               0.731779                  -0.106288                                   0.0                           0.048977                         0                           0
```

## ICs finais
```
       modo versao  sens_cadeia_observada  sens_ci95_low  sens_ci95_high
 vigilancia     v3               0.582996       0.520695        0.642755
 vigilancia   v3.1               0.821862       0.769327        0.864540
confirmacao     v3               0.404858       0.345553        0.467078
confirmacao   v3.1               0.651822       0.590497        0.708497
```
