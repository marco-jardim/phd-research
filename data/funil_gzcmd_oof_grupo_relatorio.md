# OOF aninhado por grupo

## Síntese da revisão PR7

O padrão continua por linha. O experimento utilizou 61.696 pares, 1.515 grupos
REFREC e 19.388 grupos COMPREC, com scikit-learn 1.8.0. A configuração
`evaluation.split_by` passou a ser honrada pelo runner, sem alterar seu valor
atual `row`. Os caches distinguem os três perfis. As probabilidades por grupo
estão em `data/pcal_ml_oof_refrec.csv` e `data/pcal_ml_oof_comprec.csv`, ignoradas
pelo git por `data/pcal_ml_oof*.csv`.

Os valores de referência 0,993/0,907/0,982 correspondem, na área cinza, à AUC,
AP e mediana da probabilidade nos vínculos, não ao Brier. Os valores calculados
foram 0,992920/0,906964/0,981818; o Brier foi 0,001518. Por REFREC, a AUC foi
0,995142, AP 0,862262, Brier 0,002206 e mediana 0,894737. Por COMPREC, os valores
foram 0,993554, 0,911375, 0,001443 e 0,981132, respectivamente. As métricas do
universo completo constam abaixo e em `data/pcal_ml_oof_grupos_metricas.csv`.

**A diferença na cascata por REFREC é material, embora nenhuma AUC fique
abaixo de 0,98.** Em vigilância, há 180 VP conhecidos e 15 vínculos LLM
pendentes: a sensibilidade automática fica entre 72,87% e 78,95%, contra
82,19% por linha. Em confirmação, são 120 VP conhecidos e 12 vínculos LLM
pendentes, intervalo de 48,58% a 53,44%, contra 65,18%. Esses limites completam
somente respostas LLM pendentes, sem transformar encaminhamentos humanos em
vínculos aceitos. Mesmo o limite superior fica 3,2389 e 11,7409 pontos
percentuais abaixo do perfil por linha. As expectativas de banda são 78,80%
e 52,21%, não resultados inteiramente observados. A decisão de trocar o padrão
continua com o usuário.

## M1: bandas sem respostas

O resultado refuta a hipótese de invariância da seleção proposta no despacho.
No lote por linha, vigilância contém **43/301 pares em grey_low**; confirmação
contém **0/25 em grey_mid**. Essas bandas não têm respostas Haiku v4 na coorte
de estimação. As substituições são análises contrafactuais, não novas taxas
validadas para as bandas vazias.

| Modo | Substituição | Lote original | Lote novo | Adicionados | Removidos | Sem resposta existente |
|---|---|---:|---:|---:|---:|---:|
| Vigilância | e_fp/e_fn de grey_mid | 301 | 384 | 83 | 0 | 83 |
| Vigilância | Todas as taxas de grey_mid | 301 | 384 | 83 | 0 | 83 |
| Confirmação | e_fp/e_fn de grey_high | 25 | 72 | 47 | 0 | 47 |
| Confirmação | Todas as taxas de grey_high | 25 | 70 | 45 | 0 | 45 |

Incluindo a taxa de UNSURE da vizinha, as sensibilidades automáticas esperadas
ficam em 84,50% e 66,74%. Não se substituiu o perfil vigente. Os resultados
completos estão em `data/funil_gzcmd_bandas_sem_dado.csv`; a reprodução usa
`python scripts/avaliar_bandas_sem_dado.py`.

## A2: rótulos por par

O saneamento inspecionou 15 CSVs e removeu rótulos de 11 deles, preservando
escores, decisões e identificadores pseudônimos. Os geradores correspondentes
também foram corrigidos; contagens agregadas continuam nos relatórios. Foram
removidos `y` dos quatro CSVs Haiku/DeepSeek sem MO e dos dois lotes novos,
e `gabarito` dos três CSVs de decisões Sage, da ancoragem e da comparação
pareada. Inventário: `data/privacidade_pr7_auditoria.json`.

Os oito legados abaixo **não foram alterados**, com hashes registrados para
decisão do usuário:

- `data/exp2_comparacao_revisores_confirmacao.csv`
- `data/exp2_comparacao_revisores_vigilancia.csv`
- `data/exp2_tres_revisores_confirmacao.csv`
- `data/exp2_tres_revisores_vigilancia.csv`
- `data/exp2_gabarito_por_par.csv`
- `data/sprint3b/anthropic/gabarito_confirmacao_haiku.csv`
- `data/sprint3b/opencode_go/gabarito_confirmacao_deepseek.csv`
- `data/sprint3b/opencode_go_kimi/gabarito_confirmacao_kimi_k3.csv`

O inventário também lista `data/exp2_gabarito_reconstruido.csv` para revisão,
sem modificá-lo nem inferir sua proveniência.

## M7/M8/M9 e arquivos históricos

O journal exige identidade persistida antes de recuperar resultados sem
checkpoint. Os testes cobrem divergência de prompt/modelo/coorte, ausência de
identidade e migração somente a partir de checkpoint previamente validado.
Não foram adicionadas identidades retroativamente aos runs de dados usados
nesta análise.

As quatro chamadas históricas indicadas e a reconstrução do dossiê da ablação
fixam `fit_platt`/`v3`. A regressão executa a política real e verifica lotes
1.410/435, quatro ALWAYS_MATCH e igualdade dos identificadores com as coortes
Haiku históricas. O teste de taxas foi movido para `tests/`; os temporários de
persistência usam `tmp_path`; a deriva utiliza seed 42 e restaura o estado RNG.

Não há citação de `_m3v3` em arquivos `.tex`. **23 arquivos foram retirados
somente do índice git**, com os mesmos bytes locais, comprovados por SHA256.
As remoções estão preparadas no índice, sem commit. O inventário está em
`data/arquivos_m3v3_desversionados.json`; a regra de ignore impede reintrodução
acidental. O manifesto anterior de preservação continua válido.

## Arquivos de implementação desta entrega

- `gzcmd/ml_oof.py`, `gzcmd/runner.py`, `gzcmd/config.py`
- `.gitignore`, `gzcmd/M_oof_grupos_pr7.md`
- `scripts/avaliar_oof_grupos.py`, `scripts/avaliar_bandas_sem_dado.py`
- `scripts/auditar_privacidade_pr7.py`, `scripts/retirar_arquivos_m3v3.py`
- `scripts/fig_varredura_custos.py`, `scripts/funil_gzcmd_v2.py`
- `scripts/run_exp2_sem_model_outputs.py`, `scripts/run_exp2_sage.py`
- `scripts/run_exp2_llm_sync.py`, `scripts/run_exp2_llm_batch.py`
- `scripts/run_qw2_kimi.py`, `scripts/pilot_c2_517.py`
- `tests/test_oof_grupos_pr7.py`, `tests/test_exp2_sem_model_outputs.py`
- `tests/test_drift_monitor.py`, `tests/test_estimar_taxas_erro_revisor.py`
- Remoção do caminho antigo `scripts/test_estimar_taxas_erro_revisor.py`

Nenhum capítulo `.tex` foi editado, nenhuma chamada LLM foi feita e nenhum
identificador REFREC/COMPREC ou rótulo foi persistido nos novos artefatos OOF.

## Auditorias de conservação e privacidade

Foram verificadas as partições de 61.696 pares em todas as cascatas, incluindo
a igualdade de contagens e vínculos entre o ramo LLM e seus filhos MATCH,
NONMATCH, UNSURE e SEM_RESPOSTA. Os dois CSVs de grupo têm 61.696 pair_id
distintos, probabilidades finitas em [0,1] e folds 0 a 4, sem grupos ou rótulos
nas colunas. Ambos estão efetivamente ignorados pelo git.

Os 15 CSVs do escopo de saneamento estão sem `y`, `gabarito` e `PAR`. Os hashes
dos oito legados são idênticos aos inventariados antes de qualquer intervenção
nesses arquivos. Os 23 snapshots retirados do índice permanecem locais e com
hashes idênticos; os 26 arquivos do manifesto histórico também foram validados.
`git diff --check` e `git diff --cached --check` não apontaram erros de espaço.
Avisos de finais de linha no Windows não foram tratados como falha.

Seed 42; cinco folds externos e cinco internos; XGBoost e calibração isotônica.
StratifiedGroupKFold mantém grupos disjuntos em ambos os níveis; nenhum identificador de grupo foi persistido.
O padrão permanece row. evaluation.split_by é honrado pelo runner; este script escolhe cada perfil explicitamente.
REFREC e COMPREC são análises separadas: nenhum deles garante independência simultânea dos dois lados do pareamento.
As métricas são calculadas de novo, sem interpretar 0,993/0,907/0,982 como uma sequência inequívoca de AUC/AP/Brier.
AUC/AP/Brier são OOF. p_cal_platt permanece apenas comparador histórico ajustado globalmente, não é a predição usada nesta avaliação.
A cascata usa taxas Haiku v4 e custos aprovados. Respostas existentes são reutilizadas, pendentes explicitados; sens_observada é a fração de todos os vínculos já aceita automaticamente, portanto limite inferior quando há pendências.
sens_esperada e espec_aceitacao_esperada completam pendências por expectativas, não por classificações observadas. UNSURE é encaminhamento humano, cujo erro hipotético de 5% permanece separado.
sens_automatica_pendentes_superior aceita todos os vínculos LLM pendentes, sem converter encaminhamentos humanos em aceitação automática. Já sens_superior herdada inclui também encaminhamentos humanos, não sendo o limite específico das pendências LLM.
## Métricas
```
split_by    recorte     n  vinculos      auc       ap    brier  mediana_p_vinculos
     row      total 61696       247 0.994032 0.933918 0.000637            1.000000
     row area_cinza 21620       151 0.992920 0.906964 0.001518            0.981818
  refrec      total 61696       247 0.995534 0.910727 0.000886            0.952381
  refrec area_cinza 21620       151 0.995142 0.862262 0.002206            0.894737
 comprec      total 61696       247 0.994045 0.935469 0.000634            1.000000
 comprec area_cinza 21620       151 0.993554 0.911375 0.001443            0.981132
```
## Cascatas lado a lado
```
tipo_linha split_by        modo     n  vinculos   LLM inteiramente_observado  AUTO_MATCH  AUTO_MATCH_FP  ALWAYS_MATCH  vinculos_auto_nonmatch  vinculos_auto_nonmatch_cinza  pendentes_vinculos   FP   FN    TP      TN  UNSURE  DEFERRED  DEFERRED_motor  sem_resposta_LLM esperada tipo_perda  perda_observada  perda_esperada       perda  custo_llm  perda_sem_custo_llm  perda_erro_humano_esperada  humanos_encaminhados_observados  FP_esperados_pendentes  FN_esperados_pendentes  UNSURE_esperados_pendentes  sens_observada  sens_esperada  sens_esperada_com_humano  espec_observada  espec_esperada  espec_esperada_com_humano  sens_inferior  sens_superior  espec_inferior  espec_superior  FP_aceitacao_esperados  espec_aceitacao_esperada  espec_aceitacao_inferior  espec_aceitacao_superior  VP_cadeia  FP_cadeia  FN_cadeia  VN_cadeia  VP_resolvidos  FP_resolvidos  FN_resolvidos  VN_resolvidos  FN_nonmatch_observado  VN_nonmatch_observado  sens_cadeia_observada  espec_cadeia_observada  pares_ao_humano  vinculos_ao_humano  nao_vinculos_ao_humano  humano_direto  humano_unsure  pares_ao_llm  llm_sem_resposta                   tipo_metrica  sens_automatica_pendentes_superior versao  nivel faixa etapa
  metricas      row  vigilancia 61696       247 301.0                   True       111.0            5.0          78.0                     7.0                           4.0                 0.0 19.0 14.0 203.0 61377.0    31.0      52.0            52.0               0.0     True      mista          974.505       97.980421 1072.485421      1.505          1070.980421                   97.980421                             83.0                0.000000                0.000000                    0.000000        0.821862       0.821862                  0.937247         0.998828        0.998828                   0.999648       0.821862       0.943320        0.998828        0.999691               19.000000                  0.999691                  0.999691                  0.999691      203.0       19.0       44.0    61430.0          203.0           19.0           14.0        61377.0                   14.0                61377.0               0.821862                0.999691             83.0                30.0                    53.0           52.0           31.0         301.0               0.0 observada_aceitacao_automatica                            0.821862    NaN    NaN   NaN   NaN
  metricas      row confirmacao 61696       247  25.0                   True        79.0            1.0          78.0                    18.0                          15.0                 0.0  1.0 23.0 161.0 61421.0     7.0      83.0            83.0               0.0     True      mista          650.125      178.058288  828.183288      0.125           828.058288                  178.058288                             90.0                0.000000                0.000000                    0.000000        0.651822       0.651822                  0.894130         0.999544        0.999544                   0.999962       0.651822       0.906883        0.999544        0.999984                1.000000                  0.999984                  0.999984                  0.999984      161.0        1.0       86.0    61448.0          161.0            1.0           23.0        61421.0                   23.0                61421.0               0.651822                0.999984             90.0                63.0                    27.0           83.0            7.0          25.0               0.0 observada_aceitacao_automatica                            0.651822    NaN    NaN   NaN   NaN
  metricas   refrec  vigilancia 61696       247 453.0                  False       109.0           16.0          63.0                     7.0                           4.0                15.0 34.0 13.0 180.0 61220.0    28.0      66.0            66.0             155.0     True      mista         1085.490      190.866343 1276.356343      2.265          1274.091343                  117.460688                             94.0                6.119047                0.171498                    2.865265        0.728745       0.787964                  0.939398         0.996273        0.998410                   0.999299       0.728745       0.947368        0.996273        0.999447               40.201717                  0.999346                  0.997168                  0.999447        NaN        NaN        NaN        NaN            NaN            NaN            NaN            NaN                    NaN                    NaN                    NaN                     NaN              NaN                 NaN                     NaN            NaN            NaN           NaN               NaN                            NaN                            0.789474    NaN    NaN   NaN   NaN
  metricas   refrec confirmacao 61696       247  43.0                  False        55.0            0.0          63.0                    22.0                          19.0                12.0  0.0 27.0 120.0 61404.0     6.0     122.0           122.0              17.0     True      mista          668.130      316.250844  984.380844      0.215           984.165844                  273.718479                            128.0                0.153279                1.307283                    0.973821        0.485830       0.522073                  0.863310         0.999268        0.999343                   0.999965       0.485830       0.890688        0.999268        1.000000                0.108797                  0.999998                  0.999919                  1.000000        NaN        NaN        NaN        NaN            NaN            NaN            NaN            NaN                    NaN                    NaN                    NaN                     NaN              NaN                 NaN                     NaN            NaN            NaN           NaN               NaN                            NaN                            0.534413    NaN    NaN   NaN   NaN
```
## Materialidade
Diferença material de sensibilidade automática conhecida em vigilancia: -9.3117 p.p.; pendências não são respostas observadas.
Mesmo aceitando corretamente todos os vínculos LLM pendentes, a sensibilidade automática REFREC em vigilancia fica -3.2389 p.p. abaixo da por linha, mantendo encaminhamentos humanos separados.
Diferença material de sensibilidade automática conhecida em confirmacao: -16.5992 p.p.; pendências não são respostas observadas.
Mesmo aceitando corretamente todos os vínculos LLM pendentes, a sensibilidade automática REFREC em confirmacao fica -11.7409 p.p. abaixo da por linha, mantendo encaminhamentos humanos separados.
A decisão de trocar o padrão cabe ao usuário; nenhuma troca foi feita.
