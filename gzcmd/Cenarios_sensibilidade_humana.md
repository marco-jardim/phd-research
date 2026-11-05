# Cenários de sensibilidade humana e concordância dos modos do filtro

`scripts/cenarios_sensibilidade_humana.py` produz os cenários de sensibilidade humana a partir dos agregados versionados, sem treino, chamadas de LLM ou nova classificação de potenciais pares. Os três valores de sensibilidade são hipóteses: 0,95, 0,99 e 1,00. A sensibilidade humana real não foi medida. A coluna `fonte_s` preserva a referência ao cenário de erro de 0,05 da configuração, não a uma estimativa empírica.

Para cada estratégia, calcula-se a sensibilidade total como a soma dos vínculos aceitos sem revisão humana e dos vínculos encaminhados ao humano multiplicados por sua sensibilidade hipotética, dividida pelo total de vínculos do padrão-ouro. Os 211 aceitos pelo corte e o denominador de 247 são lidos do CSV de métricas, não fixados no gerador. A revisão integral encaminha os 36 PAR=2; P mantém suas aceitações automáticas e seus encaminhamentos originais; P com MATCH ao humano submete à revisão todas as aceitações do LLM abaixo do corte. O3 é avaliado nos quatro orçamentos solicitados. Para o filtro, usam-se somente os seis pontos existentes entre k=17 e k=125, sem interpolação.

As saídas são `data/cenarios_sensibilidade_humana.csv`, com 78 linhas, `data/cenarios_sensibilidade_humana_amplitude.csv`, com 26 configurações, e `data/cenarios_sensibilidade_humana_ordem.csv`, que distingue inversão de ordem estrita de mudança nos empates. O CSV principal separa k, o número de revisões humanas, os vínculos enviados ao humano e os vínculos perdidos por cobertura. No filtro, k é o tamanho da fila enviada ao LLM, não a carga humana H.

## Amplitude e ordem

Entre sensibilidade humana de 0,95 e 1,00, a maior amplitude é **0,728745 ponto percentual**, na revisão integral abaixo do corte. Excluindo a revisão integral, o máximo é **0,526316 ponto percentual**, em O3 com 440 revisões. Portanto, a aproximação de "no máximo cerca de 0,6 ponto percentual" aplica-se às estratégias de carga reduzida, não à referência de revisão integral.

| Estratégia | Vigilância: amplitude em pontos percentuais | Confirmação: amplitude em pontos percentuais |
|---|---:|---:|
| Revisão integral | 0,728745 | 0,728745 |
| P | 0,384615 | 0,020243 |
| P com MATCH ao humano | 0,465587 | 0,182186 |
| O3, k=96 ou k=121 | 0,445344 | 0,445344 |
| O3, k=125 | 0,465587 | 0,465587 |
| O3, k=440 | 0,526316 | 0,526316 |
| Filtro, k=17 | 0,080972 | 0,080972 |
| Filtro, k=29 | 0,121457 | 0,121457 |
| Filtro, k=58 | 0,222672 | 0,222672 |
| Filtro, k=96, k=121 ou k=125 | 0,263158 | 0,263158 |

Não ocorre inversão de ordem estrita entre as configurações solicitadas nos três cenários. Há, porém, três comparações em que surge empate somente quando a sensibilidade humana chega a 1,00: P contra P com MATCH ao humano nos dois modos e P contra O3 com k=125 na vigilância. Essas comparações têm cargas humanas diferentes. Nas comparações com a mesma carga humana não há inversão nem mudança dos empates. Assim, "a ordem não muda" precisa distinguir manutenção da desigualdade estrita de aparecimento de igualdade; os resultados não constituem teste de dominância ajustado por recursos.

A especificidade humana é assumida igual a 1,00. Não se modelam falsos positivos humanos. Essa hipótese não remove os FP automáticos observados de P: variar a sensibilidade do humano altera apenas os vínculos encaminhados a ele. As perdas de cobertura permanecem, por exemplo, 13 e 27 para P e para P com MATCH ao humano, e 23 para o filtro a partir de k=96, independentemente da sensibilidade humana.

## Por que os modos do filtro têm recuperações iguais

A conferência offline utilizou o coletor histórico congelado do filtro, que constrói uma única fila O3 antes do laço por modo, em ordem decrescente de `p_cal_ml`, com desempate pela chave interna. A probabilidade e a fila não dependem dos custos de vigilância ou confirmação. As respostas foram lidas de arquivos distintos por modo, com os campos de modo e os hashes conferidos contra as fontes dos metadados históricos. Não foi encontrado uso acidental do mesmo arquivo nos dois modos.

No topo 125, **119 decisões coincidem e seis diferem**. Todas as decisões dos **23 PAR=2 coincidem**, e as seis discordâncias estão entre os 102 não vínculos. A matriz completa está em `data/filtro_llm_concordancia_modos.csv`, somente com contagens. Não há identificadores nem decisões por potencial par nessa saída.

| Topo k | Decisões coincidentes | Decisões diferentes | PAR=2 com decisão coincidente | PAR=2 com decisão diferente |
|---:|---:|---:|---:|---:|
| 17 | 15 | 2 | 6 | 0 |
| 29 | 27 | 2 | 9 | 0 |
| 58 | 54 | 4 | 20 | 0 |
| 96 | 90 | 6 | 22 | 0 |
| 121 | 115 | 6 | 22 | 0 |
| 125 | 119 | 6 | 23 | 0 |

Nos topos 17 e 29, as duas diferenças se compensam nas contagens: um não vínculo recebe NONMATCH na vigilância e UNSURE na confirmação, enquanto outro apresenta o movimento oposto. Ambos os modos, assim, têm a mesma carga humana e recuperação nesses dois pontos, embora não sejam decisões idênticas por potencial par. No topo 125, há um movimento NONMATCH para UNSURE e cinco no sentido inverso, todos em não vínculos; a confirmação descarta quatro não vínculos adicionais e envia 22 potenciais pares ao humano, contra 26 na vigilância.

A igualdade dos 13 PAR=2 recuperados nos dois modos a partir de k=96 decorre da concordância de todas as decisões sobre os PAR=2. Entre os topos 96 e 125, o único PAR=2 adicional é descartado como NONMATCH nos dois modos: o total descartado passa de nove para dez, sem aumento na recuperação. As diferenças entre modos afetam a carga humana e o VPN, não o número de vínculos recuperados nesses pontos.

`data/filtro_llm_metadata.json` tem hash fixado e não foi editado. A conferência e suas fontes foram registradas separadamente em `data/filtro_llm_concordancia_modos_metadata.json`. O teste de regeneração dessa conferência recebe `skipif` quando faltam os dados privados; a regeneração dos cenários e os testes das contagens versionadas não dependem deles.
