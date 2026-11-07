# Comparação dos quatro revisores Anthropic — 2026-11-07

## Reprodução e limites

`python scripts/probe_frontier_agregado.py --source-root D:/git/phd-research`

Status: **exploratório post hoc** para a documentação; o probe possui pré-registro local anterior à inferência.
Sem novas chamadas de API ou inferência. Apenas agregados são publicados. Os hashes de origem e os parâmetros constam de `data/probe_frontier_metadata.json`.
Protocolo A+B independentes, árbitro no desacordo; prompt v4 e dossiê sem saídas do motor. max_tokens=2500, temperature ausente, thinking não configurado (raciocínio adaptativo padrão da API). Opus 5.5 e Fable 5.1 não permitem desligar o raciocínio.

Vigilância: população P=344; S1=25, S2=13, S3=24, S4=88 (N4=282; peso=282/88). Confirmação: censo P=41. Réplicas Opus/Fable: S1–S3 (62) e confirmação (41). Seed 20261107.
Wilson 95% só em contagens não ponderadas. TOTAL_PONDERADO é Horvitz–Thompson; não somar TOTAL_AMOSTRA, estratos e TOTAL_PONDERADO entre si. κ indefinido (decisões constantes) é NA, não zero.

PAR=0 não prova ausência de vínculo. VP/FP são relativos ao PAR, sem adjudicação independente. Ausências não são UNSURE; concordância/κ usam casos completos em comum. Sem teste de superioridade, inferência causal ou substituição dos resultados principais.

## Probe: contagens e VPP (valores sem arredondamento editorial)

| model | modo | passada | estrato | n | MATCH | NONMATCH | UNSURE | VP | FP | PAR2_recuperados | PAR2_descartados | PAR2_unsure | VPP | VPP_IC95_inf | VPP_IC95_sup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | vigilancia | historico | S1 | 25.0 | 25.0 | 0.0 | 0.0 | 4.0 | 21.0 | 4.0 | 0.0 | 0.0 | 0.16 | 0.0640345145667644 | 0.3465362160717886 |
| claude-haiku-5-5 | vigilancia | historico | S2 | 13.0 | 0.0 | 6.0 | 7.0 | 0.0 | 0.0 | 0.0 | 6.0 | 7.0 | NA | NA | NA |
| claude-haiku-5-5 | vigilancia | historico | S3 | 24.0 | 0.0 | 20.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-haiku-5-5 | vigilancia | historico | S4 | 88.0 | 0.0 | 83.0 | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-haiku-5-5 | vigilancia | historico | TOTAL_AMOSTRA | 150.0 | 25.0 | 109.0 | 16.0 | 4.0 | 21.0 | 4.0 | 6.0 | 7.0 | 0.16 | 0.0640345145667644 | 0.3465362160717886 |
| claude-haiku-5-5 | vigilancia | historico | TOTAL_PONDERADO | 344.00000000000006 | 25.0 | 291.9772727272726 | 27.022727272727273 | 4.0 | 21.0 | 4.0 | 6.0 | 7.0 | 0.16 | NA | NA |
| claude-sonnet-5-5 | vigilancia | historico | S1 | 21.0 | 21.0 | 0.0 | 0.0 | 4.0 | 17.0 | 4.0 | 0.0 | 0.0 | 0.19047619047619047 | 0.07667565052761793 | 0.40000564690204926 |
| claude-sonnet-5-5 | vigilancia | historico | S2 | 7.0 | 1.0 | 2.0 | 4.0 | 1.0 | 0.0 | 1.0 | 2.0 | 4.0 | 1.0 | 0.20654931437723745 | 1.0 |
| claude-sonnet-5-5 | vigilancia | historico | S3 | 24.0 | 2.0 | 3.0 | 19.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6576197724933469 |
| claude-sonnet-5-5 | vigilancia | historico | S4 | 87.0 | 0.0 | 82.0 | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-sonnet-5-5 | vigilancia | historico | TOTAL_AMOSTRA | 139.0 | 24.0 | 87.0 | 28.0 | 5.0 | 19.0 | 5.0 | 2.0 | 4.0 | 0.20833333333333334 | 0.09244825352851857 | 0.40470453253188665 |
| claude-haiku-5-5 | confirmacao | historico | C | 41.0 | 12.0 | 28.0 | 1.0 | 8.0 | 4.0 | 8.0 | 4.0 | 0.0 | 0.6666666666666666 | 0.39062208887279953 | 0.8618799089087867 |
| claude-haiku-5-5 | confirmacao | historico | TOTAL_AMOSTRA | 41.0 | 12.0 | 28.0 | 1.0 | 8.0 | 4.0 | 8.0 | 4.0 | 0.0 | 0.6666666666666666 | 0.39062208887279953 | 0.8618799089087867 |
| claude-haiku-5-5 | confirmacao | historico | TOTAL_PONDERADO | 41.0 | 12.0 | 28.0 | 1.0 | 8.0 | 4.0 | 8.0 | 4.0 | 0.0 | 0.6666666666666666 | NA | NA |
| claude-sonnet-5-5 | confirmacao | historico | C | 33.0 | 9.0 | 18.0 | 6.0 | 6.0 | 3.0 | 6.0 | 4.0 | 0.0 | 0.6666666666666666 | 0.3542021355803963 | 0.879416181613089 |
| claude-sonnet-5-5 | confirmacao | historico | TOTAL_AMOSTRA | 33.0 | 9.0 | 18.0 | 6.0 | 6.0 | 3.0 | 6.0 | 4.0 | 0.0 | 0.6666666666666666 | 0.3542021355803963 | 0.879416181613089 |
| claude-fable-5-1 | confirmacao | principal | C | 41.0 | 11.0 | 27.0 | 3.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-fable-5-1 | confirmacao | principal | TOTAL_AMOSTRA | 41.0 | 11.0 | 27.0 | 3.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-fable-5-1 | confirmacao | principal | TOTAL_PONDERADO | 41.0 | 11.0 | 27.0 | 3.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | NA | NA |
| claude-fable-5-1 | vigilancia | principal | S1 | 25.0 | 22.0 | 1.0 | 2.0 | 4.0 | 18.0 | 4.0 | 0.0 | 0.0 | 0.18181818181818182 | 0.07306884583422052 | 0.38516607437126543 |
| claude-fable-5-1 | vigilancia | principal | S2 | 13.0 | 1.0 | 9.0 | 3.0 | 1.0 | 0.0 | 1.0 | 9.0 | 3.0 | 1.0 | 0.20654931437723745 | 1.0 |
| claude-fable-5-1 | vigilancia | principal | S3 | 24.0 | 0.0 | 18.0 | 6.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-fable-5-1 | vigilancia | principal | S4 | 88.0 | 0.0 | 83.0 | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-fable-5-1 | vigilancia | principal | TOTAL_AMOSTRA | 150.0 | 23.0 | 111.0 | 16.0 | 5.0 | 18.0 | 5.0 | 9.0 | 3.0 | 0.21739130434782608 | 0.09663978026586204 | 0.4190348301626401 |
| claude-fable-5-1 | vigilancia | principal | TOTAL_PONDERADO | 344.00000000000006 | 23.0 | 293.9772727272726 | 27.022727272727273 | 5.0 | 18.0 | 5.0 | 9.0 | 3.0 | 0.21739130434782608 | NA | NA |
| claude-fable-5-1 | confirmacao | replica | C | 41.0 | 11.0 | 27.0 | 3.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-fable-5-1 | confirmacao | replica | TOTAL_AMOSTRA | 41.0 | 11.0 | 27.0 | 3.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-fable-5-1 | vigilancia | replica | S1 | 25.0 | 21.0 | 2.0 | 2.0 | 4.0 | 17.0 | 4.0 | 0.0 | 0.0 | 0.19047619047619047 | 0.07667565052761793 | 0.40000564690204926 |
| claude-fable-5-1 | vigilancia | replica | S2 | 13.0 | 0.0 | 9.0 | 4.0 | 0.0 | 0.0 | 0.0 | 9.0 | 4.0 | NA | NA | NA |
| claude-fable-5-1 | vigilancia | replica | S3 | 24.0 | 1.0 | 19.0 | 4.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7934506856227626 |
| claude-fable-5-1 | vigilancia | replica | TOTAL_AMOSTRA | 62.0 | 22.0 | 30.0 | 10.0 | 4.0 | 18.0 | 4.0 | 9.0 | 4.0 | 0.18181818181818182 | 0.07306884583422052 | 0.38516607437126543 |
| claude-haiku-5-5 | confirmacao | replica_historico | C | 38.0 | 11.0 | 26.0 | 1.0 | 8.0 | 3.0 | 8.0 | 4.0 | 0.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-haiku-5-5 | confirmacao | replica_historico | TOTAL_AMOSTRA | 38.0 | 11.0 | 26.0 | 1.0 | 8.0 | 3.0 | 8.0 | 4.0 | 0.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-haiku-5-5 | vigilancia | replica_historico | S1 | 21.0 | 19.0 | 0.0 | 2.0 | 4.0 | 15.0 | 4.0 | 0.0 | 0.0 | 0.21052631578947367 | 0.08507676632916253 | 0.43334278443814167 |
| claude-haiku-5-5 | vigilancia | replica_historico | S2 | 11.0 | 1.0 | 6.0 | 4.0 | 1.0 | 0.0 | 1.0 | 6.0 | 4.0 | 1.0 | 0.20654931437723745 | 1.0 |
| claude-haiku-5-5 | vigilancia | replica_historico | S3 | 21.0 | 0.0 | 20.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-haiku-5-5 | vigilancia | replica_historico | S4 | 84.0 | 0.0 | 80.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-haiku-5-5 | vigilancia | replica_historico | TOTAL_AMOSTRA | 137.0 | 20.0 | 106.0 | 11.0 | 5.0 | 15.0 | 5.0 | 6.0 | 4.0 | 0.25 | 0.11186170140766569 | 0.468700877618744 |
| claude-opus-5-5 | confirmacao | principal | C | 41.0 | 11.0 | 28.0 | 2.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-opus-5-5 | confirmacao | principal | TOTAL_AMOSTRA | 41.0 | 11.0 | 28.0 | 2.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-opus-5-5 | confirmacao | principal | TOTAL_PONDERADO | 41.0 | 11.0 | 28.0 | 2.0 | 8.0 | 3.0 | 8.0 | 2.0 | 2.0 | 0.7272727272727273 | NA | NA |
| claude-opus-5-5 | vigilancia | principal | S1 | 25.0 | 21.0 | 1.0 | 3.0 | 4.0 | 17.0 | 4.0 | 0.0 | 0.0 | 0.19047619047619047 | 0.07667565052761793 | 0.40000564690204926 |
| claude-opus-5-5 | vigilancia | principal | S2 | 13.0 | 0.0 | 8.0 | 5.0 | 0.0 | 0.0 | 0.0 | 8.0 | 5.0 | NA | NA | NA |
| claude-opus-5-5 | vigilancia | principal | S3 | 24.0 | 0.0 | 22.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-opus-5-5 | vigilancia | principal | S4 | 88.0 | 0.0 | 86.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-opus-5-5 | vigilancia | principal | TOTAL_AMOSTRA | 150.0 | 21.0 | 117.0 | 12.0 | 4.0 | 17.0 | 4.0 | 8.0 | 5.0 | 0.19047619047619047 | 0.07667565052761793 | 0.40000564690204926 |
| claude-opus-5-5 | vigilancia | principal | TOTAL_PONDERADO | 344.00000000000006 | 21.0 | 306.59090909090895 | 16.40909090909091 | 4.0 | 17.0 | 4.0 | 8.0 | 5.0 | 0.19047619047619047 | NA | NA |
| claude-opus-5-5 | confirmacao | replica | C | 41.0 | 11.0 | 29.0 | 1.0 | 8.0 | 3.0 | 8.0 | 3.0 | 1.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-opus-5-5 | confirmacao | replica | TOTAL_AMOSTRA | 41.0 | 11.0 | 29.0 | 1.0 | 8.0 | 3.0 | 8.0 | 3.0 | 1.0 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 |
| claude-opus-5-5 | vigilancia | replica | S1 | 25.0 | 20.0 | 2.0 | 3.0 | 4.0 | 16.0 | 4.0 | 0.0 | 0.0 | 0.2 | 0.08065766257979806 | 0.41601743225189364 |
| claude-opus-5-5 | vigilancia | replica | S2 | 13.0 | 0.0 | 7.0 | 6.0 | 0.0 | 0.0 | 0.0 | 7.0 | 6.0 | NA | NA | NA |
| claude-opus-5-5 | vigilancia | replica | S3 | 24.0 | 0.0 | 20.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | NA | NA | NA |
| claude-opus-5-5 | vigilancia | replica | TOTAL_AMOSTRA | 62.0 | 20.0 | 29.0 | 13.0 | 4.0 | 16.0 | 4.0 | 7.0 | 6.0 | 0.2 | 0.08065766257979806 | 0.41601743225189364 |

S1 da vigilância: Haiku histórico 4/25=16%; Opus principal 4/21=19,047619…%; Fable principal 4/22=18,181818…%. Sonnet cobre só 139/150 e 33/41 neste probe, não confundir com a coorte histórica completa abaixo.

## Concordância e réplicas

| model_a | passada_a | model_b | passada_b | modo | estrato | n | agreement | kappa |
|---|---|---|---|---|---|---|---|---|
| claude-fable-5-1 | principal | claude-fable-5-1 | replica | confirmacao | TOTAL_COMUM | 41 | 0.9512195121951219 | 0.900243309002433 |
| claude-fable-5-1 | principal | claude-haiku-5-5 | historico | confirmacao | TOTAL_COMUM | 41 | 0.8780487804878049 | 0.7405063291139241 |
| claude-fable-5-1 | principal | claude-haiku-5-5 | replica_historico | confirmacao | TOTAL_COMUM | 38 | 0.8947368421052632 | 0.7816091954022989 |
| claude-fable-5-1 | principal | claude-opus-5-5 | principal | confirmacao | TOTAL_COMUM | 41 | 0.975609756097561 | 0.9486215538847117 |
| claude-fable-5-1 | principal | claude-opus-5-5 | replica | confirmacao | TOTAL_COMUM | 41 | 0.9512195121951219 | 0.8940568475452195 |
| claude-fable-5-1 | principal | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 33 | 0.7272727272727273 | 0.5147058823529412 |
| claude-fable-5-1 | replica | claude-haiku-5-5 | historico | confirmacao | TOTAL_COMUM | 41 | 0.8780487804878049 | 0.7405063291139241 |
| claude-fable-5-1 | replica | claude-haiku-5-5 | replica_historico | confirmacao | TOTAL_COMUM | 38 | 0.8947368421052632 | 0.7816091954022989 |
| claude-fable-5-1 | replica | claude-opus-5-5 | principal | confirmacao | TOTAL_COMUM | 41 | 0.975609756097561 | 0.9486215538847117 |
| claude-fable-5-1 | replica | claude-opus-5-5 | replica | confirmacao | TOTAL_COMUM | 41 | 0.9512195121951219 | 0.8940568475452195 |
| claude-fable-5-1 | replica | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 33 | 0.7878787878787878 | 0.622549019607843 |
| claude-fable-5-1 | principal | claude-fable-5-1 | replica | vigilancia | TOTAL_COMUM | 62 | 0.8870967741935484 | 0.8182579564489111 |
| claude-fable-5-1 | principal | claude-haiku-5-5 | historico | vigilancia | TOTAL_COMUM | 150 | 0.8733333333333333 | 0.702194357366771 |
| claude-fable-5-1 | principal | claude-haiku-5-5 | replica_historico | vigilancia | TOTAL_COMUM | 137 | 0.9197080291970803 | 0.7934767712758668 |
| claude-fable-5-1 | principal | claude-opus-5-5 | principal | vigilancia | TOTAL_COMUM | 150 | 0.92 | 0.7963340122199594 |
| claude-fable-5-1 | principal | claude-opus-5-5 | replica | vigilancia | TOTAL_COMUM | 62 | 0.8387096774193549 | 0.7447509263071224 |
| claude-fable-5-1 | principal | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 139 | 0.8129496402877698 | 0.6118152524167563 |
| claude-fable-5-1 | replica | claude-haiku-5-5 | historico | vigilancia | TOTAL_COMUM | 62 | 0.7580645161290323 | 0.6131447587354409 |
| claude-fable-5-1 | replica | claude-haiku-5-5 | replica_historico | vigilancia | TOTAL_COMUM | 53 | 0.8867924528301887 | 0.8146853146853148 |
| claude-fable-5-1 | replica | claude-opus-5-5 | principal | vigilancia | TOTAL_COMUM | 62 | 0.8709677419354839 | 0.7891156462585035 |
| claude-fable-5-1 | replica | claude-opus-5-5 | replica | vigilancia | TOTAL_COMUM | 62 | 0.8870967741935484 | 0.819467554076539 |
| claude-fable-5-1 | replica | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 52 | 0.5384615384615384 | 0.35670103092783506 |
| claude-haiku-5-5 | historico | claude-haiku-5-5 | replica_historico | confirmacao | TOTAL_COMUM | 38 | 1.0 | 1.0 |
| claude-haiku-5-5 | historico | claude-opus-5-5 | principal | confirmacao | TOTAL_COMUM | 41 | 0.9024390243902439 | 0.7850589777195283 |
| claude-haiku-5-5 | historico | claude-opus-5-5 | replica | confirmacao | TOTAL_COMUM | 41 | 0.926829268292683 | 0.8328804347826088 |
| claude-haiku-5-5 | historico | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 33 | 0.8484848484848485 | 0.7236180904522613 |
| claude-haiku-5-5 | replica_historico | claude-opus-5-5 | principal | confirmacao | TOTAL_COMUM | 38 | 0.9210526315789473 | 0.8301043219076005 |
| claude-haiku-5-5 | replica_historico | claude-opus-5-5 | replica | confirmacao | TOTAL_COMUM | 38 | 0.9473684210526315 | 0.8823529411764705 |
| claude-haiku-5-5 | replica_historico | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 30 | 0.9 | 0.8105263157894738 |
| claude-haiku-5-5 | historico | claude-haiku-5-5 | replica_historico | vigilancia | TOTAL_COMUM | 137 | 0.9343065693430657 | 0.8285833449186708 |
| claude-haiku-5-5 | historico | claude-opus-5-5 | principal | vigilancia | TOTAL_COMUM | 150 | 0.9133333333333333 | 0.7840531561461794 |
| claude-haiku-5-5 | historico | claude-opus-5-5 | replica | vigilancia | TOTAL_COMUM | 62 | 0.8064516129032258 | 0.6959542296689823 |
| claude-haiku-5-5 | historico | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 139 | 0.8129496402877698 | 0.6121485297274094 |
| claude-haiku-5-5 | replica_historico | claude-opus-5-5 | principal | vigilancia | TOTAL_COMUM | 137 | 0.9635036496350365 | 0.8981715474951688 |
| claude-haiku-5-5 | replica_historico | claude-opus-5-5 | replica | vigilancia | TOTAL_COMUM | 53 | 0.9245283018867925 | 0.8764568764568764 |
| claude-haiku-5-5 | replica_historico | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 127 | 0.7952755905511811 | 0.5435443737904342 |
| claude-opus-5-5 | principal | claude-opus-5-5 | replica | confirmacao | TOTAL_COMUM | 41 | 0.975609756097561 | 0.9450402144772118 |
| claude-opus-5-5 | principal | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 33 | 0.7575757575757576 | 0.56 |
| claude-opus-5-5 | replica | claude-sonnet-5-5 | historico | confirmacao | TOTAL_COMUM | 33 | 0.7878787878787878 | 0.6071428571428571 |
| claude-opus-5-5 | principal | claude-opus-5-5 | replica | vigilancia | TOTAL_COMUM | 62 | 0.9193548387096774 | 0.8705636743215031 |
| claude-opus-5-5 | principal | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 139 | 0.7841726618705036 | 0.5339740724184175 |
| claude-opus-5-5 | replica | claude-sonnet-5-5 | historico | vigilancia | TOTAL_COMUM | 52 | 0.5 | 0.3038105046343975 |

## Completude do probe

| model | modo | passada | n_expected | n_analisavel | n_sem_resultado |
|---|---|---|---|---|---|
| claude-haiku-5-5 | vigilancia | historico | 150 | 150 | 0 |
| claude-sonnet-5-5 | vigilancia | historico | 150 | 139 | 11 |
| claude-haiku-5-5 | confirmacao | historico | 41 | 41 | 0 |
| claude-sonnet-5-5 | confirmacao | historico | 41 | 33 | 8 |
| claude-fable-5-1 | confirmacao | principal | 41 | 41 | 0 |
| claude-fable-5-1 | vigilancia | principal | 150 | 150 | 0 |
| claude-fable-5-1 | confirmacao | replica | 41 | 41 | 0 |
| claude-fable-5-1 | vigilancia | replica | 62 | 62 | 0 |
| claude-haiku-5-5 | confirmacao | replica_historico | 41 | 38 | 3 |
| claude-haiku-5-5 | vigilancia | replica_historico | 150 | 137 | 13 |
| claude-opus-5-5 | confirmacao | principal | 41 | 41 | 0 |
| claude-opus-5-5 | vigilancia | principal | 150 | 150 | 0 |
| claude-opus-5-5 | confirmacao | replica | 41 | 41 | 0 |
| claude-opus-5-5 | vigilancia | replica | 62 | 62 | 0 |

Haiku réplica: 17 respostas truncadas afetaram 16 pares, sem retry pago; 137/150 e 38/41 decisões válidas. Opus/Fable completos, zero truncadas. Comparações da réplica Haiku podem sofrer seleção por ausência.

## Custos

Probe: US$ 44.4403321 (US$ 44,44), teto US$ 90; inclui pilotos diretos, principais e réplicas Batch. São estimativas por usage/preço, não fatura. Sonnet histórico não gerou gasto novo no probe.
Custo por par revisado = custo / decisões finais válidas por passada (inclui piloto). Réplicas contam novamente; não é custo por par único. Custos de chamadas falhas estão no numerador. Raciocínio já integra tokens de saída, não somar outra vez. NA = não registrado, não zero.

| model | escopo | chamadas | tokens_entrada | tokens_saida | tokens_raciocinio | usd_total | pares_revisados | usd_por_par_revisado | respostas_truncadas | chamadas_batch | chamadas_direto | transporte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | probe | 400 | 1275407 | 730075 | 500665 | 0.2462891 | 175 | 0.001407366285714285714285714286 | 17 | 400 | 0 | batch |
| claude-sonnet-5-5 | probe | 0 | 0 | 0 | NA | 0 | 0 | NA | 0 | 0 | 0 | historico_reutilizado |
| claude-opus-5-5 | probe | 633 | 2021045 | 692577 | 310568 | 11.293188 | 304 | 0.03714864473684210526315789474 | 0 | 613 | 20 | batch+direto |
| claude-fable-5-1 | probe | 653 | 2105712 | 859031 | 491984 | 32.900855 | 304 | 0.1082264967105263157894736842 | 0 | 633 | 20 | batch+direto |
| claude-haiku-5-5 | coorte_historica | 4879 | 15660017 | 8886843 | NA | 6.0094232 | 2233 | 0.002691188177339901477832512315 | NA | 0 | 4879 | direto |
| claude-sonnet-5-5 | coorte_historica | 4608 | 14888191 | 4654334 | NA | 76.319722 | 2120 | 0.03599986886792452830188679245 | NA | 0 | 4608 | direto |

## Coorte histórica completa Haiku × Sonnet (sem custo novo)

Original: runs `anthropic_sem_mo_v4` e `anthropic_sonnet_sem_mo_v4`. União histórica: acrescenta os suplementos lote31; decisões repetidas são deduplicadas e conflitos rejeitados. Arranjo, robustez e filtro são intervenções posteriores e ficam fora desta comparação/custo histórico. Não se misturam com a coorte original.
Exigidos por código: mesmo prompt SHA-256, auditoria sem campos proibidos e fingerprint do dossiê idêntico entre modelos em cada par comum. Faixas usam `nota final` do comparador.

| coorte | modo | n_haiku | n_sonnet | n_comum |
|---|---|---|---|---|
| original | vigilancia | 1410 | 1410 | 1410 |
| original | confirmacao | 435 | 435 | 435 |
| uniao_historica | vigilancia | 1774 | 1665 | 1665 |
| uniao_historica | confirmacao | 459 | 455 | 455 |

| coorte | model | modo | faixa | n | MATCH | NONMATCH | UNSURE | VP | FP | PAR2_total | PAR2_recuperados | PAR2_descartados | PAR2_unsure | VPP | VPP_IC95_inf | VPP_IC95_sup | n_concordantes | agreement | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | claude-haiku-5-5 | vigilancia | [5,7) | 362 | 11 | 342 | 9 | 3 | 8 | 1 | 0 | 0 | 1 | 0.2727272727272727 | 0.09746059290024889 | 0.5656453011761291 | 329 | 0.9088397790055248 | 0.46977363515312914 |
| original | claude-sonnet-5-5 | vigilancia | [5,7) | 362 | 14 | 316 | 32 | 5 | 9 | 1 | 1 | 0 | 0 | 0.35714285714285715 | 0.1634473175355414 | 0.612355768960926 | 329 | 0.9088397790055248 | 0.46977363515312914 |
| original | claude-haiku-5-5 | vigilancia | [7,8) | 930 | 62 | 826 | 42 | 38 | 24 | 5 | 4 | 1 | 0 | 0.6129032258064516 | 0.48847843539018354 | 0.7241535483288614 | 796 | 0.8559139784946237 | 0.4991640677748127 |
| original | claude-sonnet-5-5 | vigilancia | [7,8) | 930 | 78 | 734 | 118 | 38 | 40 | 5 | 4 | 0 | 1 | 0.48717948717948717 | 0.37948952024893845 | 0.5960729876923999 | 796 | 0.8559139784946237 | 0.4991640677748127 |
| original | claude-haiku-5-5 | vigilancia | [8,9) | 96 | 20 | 69 | 7 | 18 | 2 | 5 | 2 | 3 | 0 | 0.9 | 0.6989663547715128 | 0.9721335187862319 | 82 | 0.8541666666666666 | 0.709279688513952 |
| original | claude-sonnet-5-5 | vigilancia | [8,9) | 96 | 25 | 58 | 13 | 22 | 3 | 5 | 2 | 3 | 0 | 0.88 | 0.7004420607907267 | 0.9583318284955964 | 82 | 0.8541666666666666 | 0.709279688513952 |
| original | claude-haiku-5-5 | vigilancia | >=9 | 22 | 14 | 5 | 3 | 14 | 0 | 2 | 0 | 2 | 0 | 1.0 | 0.7846891972623643 | 1.0 | 19 | 0.8636363636363636 | 0.7040358744394619 |
| original | claude-sonnet-5-5 | vigilancia | >=9 | 22 | 17 | 4 | 1 | 17 | 0 | 2 | 0 | 2 | 0 | 1.0 | 0.8156818649911483 | 1.0 | 19 | 0.8636363636363636 | 0.7040358744394619 |
| original | claude-haiku-5-5 | vigilancia | TOTAL | 1410 | 107 | 1242 | 61 | 73 | 34 | 13 | 6 | 6 | 1 | 0.6822429906542056 | 0.5890181027627126 | 0.7628358011369958 | 1226 | 0.8695035460992908 | 0.5547271622609645 |
| original | claude-sonnet-5-5 | vigilancia | TOTAL | 1410 | 134 | 1112 | 164 | 82 | 52 | 13 | 7 | 5 | 1 | 0.6119402985074627 | 0.5274100524412393 | 0.6902312889713205 | 1226 | 0.8695035460992908 | 0.5547271622609645 |
| original | claude-haiku-5-5 | confirmacao | [7,8) | 281 | 10 | 263 | 8 | 6 | 4 | 1 | 0 | 1 | 0 | 0.6 | 0.31267376973365824 | 0.8318196702937638 | 239 | 0.8505338078291815 | 0.30662123259502966 |
| original | claude-sonnet-5-5 | confirmacao | [7,8) | 281 | 11 | 234 | 36 | 8 | 3 | 1 | 0 | 0 | 1 | 0.7272727272727273 | 0.43435469882387084 | 0.902539407099751 | 239 | 0.8505338078291815 | 0.30662123259502966 |
| original | claude-haiku-5-5 | confirmacao | [8,9) | 96 | 22 | 66 | 8 | 20 | 2 | 5 | 2 | 3 | 0 | 0.9090909090909091 | 0.7218505315839417 | 0.9747045710090046 | 84 | 0.875 | 0.7595993322203672 |
| original | claude-sonnet-5-5 | confirmacao | [8,9) | 96 | 25 | 57 | 14 | 22 | 3 | 5 | 2 | 3 | 0 | 0.88 | 0.7004420607907267 | 0.9583318284955964 | 84 | 0.875 | 0.7595993322203672 |
| original | claude-haiku-5-5 | confirmacao | >=9 | 58 | 38 | 11 | 9 | 38 | 0 | 5 | 2 | 3 | 0 | 1.0 | 0.908190131774416 | 0.9999999999999999 | 49 | 0.8448275862068966 | 0.6689917564996829 |
| original | claude-sonnet-5-5 | confirmacao | >=9 | 58 | 43 | 9 | 6 | 43 | 0 | 5 | 2 | 3 | 0 | 1.0 | 0.9179901967742088 | 1.0 | 49 | 0.8448275862068966 | 0.6689917564996829 |
| original | claude-haiku-5-5 | confirmacao | TOTAL | 435 | 70 | 340 | 25 | 64 | 6 | 11 | 4 | 7 | 0 | 0.9142857142857143 | 0.8253432750766051 | 0.9601233303548878 | 372 | 0.8551724137931035 | 0.6586960582850738 |
| original | claude-sonnet-5-5 | confirmacao | TOTAL | 435 | 79 | 300 | 56 | 73 | 6 | 11 | 4 | 6 | 1 | 0.9240506329113924 | 0.8440458309717235 | 0.9647279515496333 | 372 | 0.8551724137931035 | 0.6586960582850738 |
| uniao_historica | claude-haiku-5-5 | vigilancia | [5,7) | 586 | 39 | 514 | 33 | 22 | 17 | 5 | 0 | 1 | 4 | 0.5641025641025641 | 0.40975691648669477 | 0.7069524612431037 | 548 | 0.9351535836177475 | 0.7540018338286144 |
| uniao_historica | claude-sonnet-5-5 | vigilancia | [5,7) | 586 | 42 | 485 | 59 | 24 | 18 | 5 | 1 | 1 | 3 | 0.5714285714285714 | 0.4220623987810458 | 0.7088234882381393 | 548 | 0.9351535836177475 | 0.7540018338286144 |
| uniao_historica | claude-haiku-5-5 | vigilancia | [7,8) | 961 | 64 | 853 | 44 | 38 | 26 | 5 | 4 | 1 | 0 | 0.59375 | 0.4714520812860451 | 0.7054309075028039 | 822 | 0.8553590010405827 | 0.49860555899630277 |
| uniao_historica | claude-sonnet-5-5 | vigilancia | [7,8) | 961 | 80 | 758 | 123 | 38 | 42 | 5 | 4 | 0 | 1 | 0.475 | 0.36924733297407564 | 0.5830435735027005 | 822 | 0.8553590010405827 | 0.49860555899630277 |
| uniao_historica | claude-haiku-5-5 | vigilancia | [8,9) | 96 | 20 | 69 | 7 | 18 | 2 | 5 | 2 | 3 | 0 | 0.9 | 0.6989663547715128 | 0.9721335187862319 | 82 | 0.8541666666666666 | 0.709279688513952 |
| uniao_historica | claude-sonnet-5-5 | vigilancia | [8,9) | 96 | 25 | 58 | 13 | 22 | 3 | 5 | 2 | 3 | 0 | 0.88 | 0.7004420607907267 | 0.9583318284955964 | 82 | 0.8541666666666666 | 0.709279688513952 |
| uniao_historica | claude-haiku-5-5 | vigilancia | >=9 | 22 | 14 | 5 | 3 | 14 | 0 | 2 | 0 | 2 | 0 | 1.0 | 0.7846891972623643 | 1.0 | 19 | 0.8636363636363636 | 0.7040358744394619 |
| uniao_historica | claude-sonnet-5-5 | vigilancia | >=9 | 22 | 17 | 4 | 1 | 17 | 0 | 2 | 0 | 2 | 0 | 1.0 | 0.8156818649911483 | 1.0 | 19 | 0.8636363636363636 | 0.7040358744394619 |
| uniao_historica | claude-haiku-5-5 | vigilancia | TOTAL | 1665 | 137 | 1441 | 87 | 92 | 45 | 17 | 6 | 7 | 4 | 0.6715328467153284 | 0.5891489046040097 | 0.744559652475308 | 1471 | 0.8834834834834835 | 0.6209692560431823 |
| uniao_historica | claude-sonnet-5-5 | vigilancia | TOTAL | 1665 | 164 | 1305 | 196 | 101 | 63 | 17 | 7 | 6 | 4 | 0.6158536585365854 | 0.5395699364340087 | 0.6868341967700262 | 1471 | 0.8834834834834835 | 0.6209692560431823 |
| uniao_historica | claude-haiku-5-5 | confirmacao | [7,8) | 294 | 19 | 267 | 8 | 15 | 4 | 3 | 2 | 1 | 0 | 0.7894736842105263 | 0.5666572155618583 | 0.9149232336708375 | 250 | 0.8503401360544217 | 0.43113456464379946 |
| uniao_historica | claude-sonnet-5-5 | confirmacao | [7,8) | 294 | 20 | 236 | 38 | 17 | 3 | 3 | 2 | 0 | 1 | 0.85 | 0.6395811352592431 | 0.9476312541037835 | 250 | 0.8503401360544217 | 0.43113456464379946 |
| uniao_historica | claude-haiku-5-5 | confirmacao | [8,9) | 98 | 22 | 68 | 8 | 20 | 2 | 5 | 2 | 3 | 0 | 0.9090909090909091 | 0.7218505315839417 | 0.9747045710090046 | 86 | 0.8775510204081632 | 0.7614604462474645 |
| uniao_historica | claude-sonnet-5-5 | confirmacao | [8,9) | 98 | 25 | 59 | 14 | 22 | 3 | 5 | 2 | 3 | 0 | 0.88 | 0.7004420607907267 | 0.9583318284955964 | 86 | 0.8775510204081632 | 0.7614604462474645 |
| uniao_historica | claude-haiku-5-5 | confirmacao | >=9 | 63 | 41 | 13 | 9 | 41 | 0 | 7 | 4 | 3 | 0 | 1.0 | 0.9143324298155681 | 0.9999999999999999 | 53 | 0.8412698412698413 | 0.6666666666666666 |
| uniao_historica | claude-sonnet-5-5 | confirmacao | >=9 | 63 | 46 | 10 | 7 | 46 | 0 | 7 | 4 | 3 | 0 | 1.0 | 0.9229264369144198 | 1.0000000000000002 | 53 | 0.8412698412698413 | 0.6666666666666666 |
| uniao_historica | claude-haiku-5-5 | confirmacao | TOTAL | 455 | 82 | 348 | 25 | 76 | 6 | 15 | 8 | 7 | 0 | 0.926829268292683 | 0.8494215619114518 | 0.9660352227433409 | 389 | 0.8549450549450549 | 0.6734023578544395 |
| uniao_historica | claude-sonnet-5-5 | confirmacao | TOTAL | 455 | 91 | 305 | 59 | 85 | 6 | 15 | 8 | 6 | 1 | 0.9340659340659341 | 0.8635353380058843 | 0.9694337135283846 | 389 | 0.8549450549450549 | 0.6734023578544395 |

Os custos históricos somam os transport/cost_estimate dos runs originais e lote31 uma única vez, sem somar journals e execucao.json de novo. Não são parte dos US$ 44,44 do probe.
