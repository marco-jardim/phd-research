# Roteiro da defesa

A apresentação tem 24 quadros principais e 14 de reserva, estes fora dos 40 minutos. Os tempos e as metas de palavras orientam o ensaio e a distribuição da fala; não são duração medida.

## Cronometragem

A meta de palavras usa 130 palavras por minuto. A soma dos quadros principais é de 2.280 segundos, ou 38,0 minutos, com 2 minutos de folga.

| Bloco | Quadros | Segundos | Minutos |
| :- | :- | -: | -: |
| Abertura | S01 a S06 | 440 | 7,3 |
| Material e medida | S07 a S09 | 330 | 5,5 |
| Resultados | S10 a S18 | 960 | 16,0 |
| Fechamento | S19 a S24 | 550 | 9,2 |
| Total | S01 a S24 | 2.280 | 38,0 |

| Quadro | Tema | Segundos | Meta de palavras |
| :- | :- | -: | -: |
| S01 | Capa | 20 | 43 |
| S02 | Problema de saúde pública | 80 | 173 |
| S03 | Corte e *grey area* | 120 | 260 |
| S04 | Objetivo geral | 60 | 130 |
| S05 | Cinco objetivos específicos | 60 | 130 |
| S06 | Resposta antecipada | 100 | 217 |
| S07 | Dados e referência operacional | 100 | 217 |
| S08 | Esquema do pós-processamento | 110 | 238 |
| S09 | A medida | 120 | 260 |
| S10 | Objetivo 1: CdR | 80 | 173 |
| S11 | Objetivo 2: classificadores | 100 | 217 |
| S12 | Objetivo 2: probabilidade calibrada | 90 | 195 |
| S13 | Objetivo 3: curva de esforço | 140 | 303 |
| S14 | Objetivo 3: incerteza | 120 | 260 |
| S15 | Objetivo 3: tabela das filas | 100 | 217 |
| S16 | Objetivo 4: LLM como aceitador | 120 | 260 |
| S17 | Objetivo 4: mesma carga | 110 | 238 |
| S18 | Objetivo 5: equidade | 100 | 217 |
| S19 | O que a tese sustenta | 130 | 282 |
| S20 | Limitações | 100 | 217 |
| S21 | Fluxo recomendado | 140 | 303 |
| S22 | Critério de refutação | 90 | 195 |
| S23 | Contribuições | 70 | 152 |
| S24 | Agradecimentos | 20 | 43 |
| Total | | 2.280 | 4.940 |

## Fala-guia por quadro

### S01. Capa

A tese, orientada por Rejane Sobrino Pinheiro no Instituto de Estudos em Saúde Coletiva da UFRJ, trata do pós-processamento do *linkage* (relacionamento de registros) probabilístico de bases de saúde. O objeto é a revisão manual dos potenciais pares que o OpenRecLink deixa sem decisão automática.

Transição: para situar o problema, convém começar pela vigilância do óbito por tuberculose.

Se perguntarem: não se aplica.

Fonte: `tese.tex:275` (título), `tese.tex:541` (orientadora).

### S02. A vigilância do óbito por tuberculose depende do *linkage* entre óbitos e notificações

A tuberculose é condição marcadora do desempenho dos serviços de saúde: o que acontece com o caso informa sobre a rede que o atendeu. O Ministério da Saúde trata o óbito por tuberculose como evento sentinela, por ser desfecho evitável. No estado do Rio de Janeiro, o coeficiente de mortalidade foi de 4,8 por 100 mil habitantes em 2024, o terceiro maior entre as unidades da federação, em dados preliminares. A contagem correta desses óbitos depende de ligar o Sistema de Informações sobre Mortalidade, o SIM, ao Sistema de Informação de Agravos de Notificação, o Sinan. Como os dois sistemas não têm identificador unívoco comum, a ligação se faz por *linkage* probabilístico: compara-se nome, nome da mãe, data de nascimento e outros campos, e calcula-se um escore de semelhança. Os erros têm consequência. Um falso positivo, isto é, um vínculo aceito indevidamente, pode levar ao encerramento como óbito do caso de uma pessoa viva. Um falso negativo faz o óbito de uma pessoa notificada parecer não notificado, com risco de investigação desnecessária. A tese não mediu o custo dessas consequências.

Transição: esse é o problema de saúde pública; a tese o encara pelo ponto em que o OpenRecLink deixa de decidir sozinho.

Se perguntarem: R04 (subnotificação e perda de vínculo).

Fonte: `capitulo3.tex:5` (4,8 por 100 mil, 2024, terceiro maior, evento sentinela), `capitulo3.tex:7` (consequências do falso positivo e do falso negativo; custo não medido), `tese.tex:548` (identificador unívoco), `capitulo_dados.tex:59-66` (campos comparados).

### S03. O ponto de corte aceita 211 dos 247 pares verdadeiros; os outros 36 só a revisão manual de origem recuperou

O ponto de partida é o processo de *linkage* tal como foi feito. O OpenRecLink forma, por blocagem, os potenciais pares, isto é, as combinações de um registro do SIM com um registro do Sinan que compartilham chaves comuns. Cada potencial par recebe um escore de semelhança. Esse escore é comparado a um ponto de corte, definido para cada passo de blocagem: acima dele, o par é aceito como vínculo; abaixo, fica para a revisão manual. Esse corte é único por passo e não há limiar inferior de rejeição. Nesta tese, o corte é parte do processo e é preservado: os vínculos que ele aceita não são alterados. Dos 247 pares verdadeiros do extrato, o corte aceitou 211, ou 85,4%. Os outros 36, 14,6%, só foram recuperados pela revisão manual de origem. Fala-se em par verdadeiro segundo a referência operacional, formada pelo corte e por essa revisão, e não em verdade absoluta. A região abaixo do corte recebe aqui o nome de *grey area* (área cinza) e reúne 61.485 dos 61.696 potenciais pares. É nela que a revisão manual atua, e é nela que a tese procura reduzir o esforço necessário para recuperar os 36 pares verdadeiros. Essa revisão é dispendiosa, e o custo dessa carga não foi medido. O corte decide sozinho a maior parte dos pares verdadeiros; a pergunta da tese diz respeito ao restante. Esse é o desenho original do processo, descrito aqui sem alteração.

Transição: com a região delimitada, resta formular a pergunta que a tese procura responder.

Se perguntarem: R03 (um corte por passo e os dois limiares de Fellegi e Sunter).

Fonte: `capitulo3.tex:9` (211, 247, 36, 85,4%, 14,6%, custo não medido, corte preservado), `capitulo3.tex:11` (61.696), `capitulo_dados.tex:23` (corte único por passo, sem limiar inferior), `capitulo_dados.tex:73` (referência operacional).

### S04. Objetivo geral

O objetivo geral tem duas partes. A primeira é desenvolver e avaliar um pós-processamento: um conjunto de etapas aplicadas depois da decisão do OpenRecLink, que reduza a revisão manual necessária na *grey area*, sem alterar os vínculos que o corte aceitou. A segunda é atribuir essa redução a um componente. O pós-processamento foi separado em cinco: a preservação do corte; a probabilidade de par verdadeiro, calibrada fora da amostra; a ordem da fila de revisão; a escolha do ponto de parada pela curva de esforço; e a decisão por *expected loss*, que é o custo esperado de cada ação possível, com revisão assistida por um modelo de linguagem. Separar os componentes permite perguntar qual deles, e não o conjunto, responde pela redução. Essa pergunta organiza o restante da apresentação.

Transição: a pergunta se desdobra em cinco objetivos específicos, um por estudo.

Se perguntarem: não se aplica.

Fonte: `capitulo3.tex:17` (objetivo), `capitulo3.tex:39` (cinco componentes), `conclusao.tex:3` (componentes), `glossario.tex:52` (*expected loss*).

### S05. Cinco objetivos específicos, cinco passos da resposta

Os cinco objetivos encadeiam-se do instrumento à equidade. O primeiro reimplementa em Python, em código aberto, o Comparador de Registros, o CdR, que calcula a semelhança campo a campo e produz os subescores usados depois; compara-se o resultado ao protótipo em Java. O segundo estima a probabilidade de que um potencial par seja verdadeiro e a calibra fora da amostra; ele se divide entre os Estudos 2 e 3, sobre classificadores, e o Estudo 4. O terceiro compara ordenações da fila pelo número de revisões necessário, com a incerteza estimada por *bootstrap*, que reamostra os óbitos. O quarto verifica se a automação reduz a revisão além da ordenação. O quinto avalia a equidade entre estratos de processo e subgrupos demográficos. Cada resultado adiante indica, no rodapé, a que objetivo responde.

Transição: antes dos detalhes, convém antecipar a resposta.

Se perguntarem: não se aplica.

Fonte: `capitulo3.tex:25-33` (objetivos), `estudo3_estrategias.tex:3` (Estudo 3 responde parte do objetivo 2), `estudo4_gzcmd.tex:3` (Estudo 4), `estudo4_gzcmd.tex:123` (*bootstrap* agrupado por óbito).

### S06. A redução veio da ordem da fila de revisão, não da automação

A resposta vem antes dos métodos. Nesta coorte, a redução do esforço de revisão veio da ordem em que os potenciais pares são revisados, e não da decisão automática nem do modelo de linguagem. Ordenar a fila pela probabilidade calibrada, isto é, a probabilidade estimada por um modelo que não viu os potenciais pares do mesmo óbito, exigiu menos revisões que ordenar pelo escore do OpenRecLink, com reduções pontuais de 75,0% e 43,1% nas duas metas menores. Essa diferença é estimativa pontual: o intervalo de confiança não exclui o zero. Quando se combinam as duas ordens, em fusões, o intervalo de confiança de 95% fica inteiramente abaixo de zero frente ao escore, nas metas de 9 e 36 pares verdadeiros. Nenhuma fusão superou a probabilidade isolada. Já a decisão automática, com o modelo de linguagem como aceitador, trouxe 4 pares verdadeiros e 21 falsos positivos na vigilância: o modelo só apoia a revisão. Tudo isso é *post hoc*, de coorte única e sob revisor perfeito simulado, e depende de validação em coorte nova. Revisor perfeito simulado quer dizer que toda revisão recupera o par verdadeiro, sem falso positivo humano; não é acerto observado. Os quadros seguintes mostram como se chegou a cada uma dessas afirmações, na ordem dos objetivos.

Transição: para ler esses resultados, é preciso conhecer os dados e a referência contra a qual tudo é medido.

Se perguntarem: R08 (tabela do *bootstrap*) e R07 (multiplicidade).

Fonte: `conclusao.tex:5`, `conclusao.tex:17` (75,0% e 43,1%), `conclusao.tex:27` (4 e 21), `estudo4_gzcmd.tex:119` (revisor perfeito), `estudo4_gzcmd.tex:307-309`.

### S07. Dados: três passos de blocagem e uma referência operacional de revisora única

O extrato reúne óbitos do SIM de 2019 a 2022 e notificações do Sinan-TB de 2013 a 2022, em três passos de blocagem exportados do OpenRecLink: 6, 9 e 11. São 61.696 potenciais pares, dos quais 247 são pares verdadeiros, uma razão de 1 para 249. O passo 11, com chave de blocagem menos específica, gerou 93,4% dos potenciais pares e apenas 40 pares verdadeiros, 25 deles abaixo do corte. Cada par tem uma classificação de referência, o campo PAR: 0 para não par, 1 para par verdadeiro aceito pelo corte, 2 para par verdadeiro recuperado pela revisão manual de origem. Essa referência operacional resulta do corte e dessa revisão, feita por uma revisora única. Não houve dupla revisão, e a ordem de exame, as fontes consultadas e o critério de parada não estão documentados. A tese adota como hipótese de trabalho que o CdR precedeu a revisão. Sem suposições sobre esse mecanismo, a direção e a magnitude do viés de verificação não são identificáveis. Essa limitação é de documentação e de delineamento, e não do trabalho de revisão de origem. Quanto à ética, o projeto foi aprovado pelos comitês do IESC-UFRJ e da Secretaria Municipal de Saúde, e o extrato é pseudonimizado, mas não anônimo.

Transição: com os dados e a referência definidos, falta mostrar como o pós-processamento se organiza.

Se perguntarem: R01 (viés de verificação) e R02 (ética, LGPD e APIs externas).

Fonte: `capitulo_dados.tex:19` (2019 a 2022, 2013 a 2022), `capitulo_dados.tex:27` e `:29` (comitês, pseudonimização), `capitulo_dados.tex:57` (93,4%), `capitulo_dados.tex:73` (campo PAR), `capitulo_dados.tex:77` (revisora única, hipótese de trabalho), `tables/editorial/tab_blocagem_rendimento.tex:13` (passo 11, 40 pares verdadeiros), `estudo5_equidade.tex:52` (25 dos 36 no passo 11).

### S08. O corte é preservado; o pós-processamento atua abaixo dele

O esquema mostra onde cada componente atua. À esquerda, o OpenRecLink gera os potenciais pares e calcula o escore. O ponto de corte, preservado, separa os vínculos aceitos, que o pós-processamento não altera, da *grey area*. Abaixo do corte, o primeiro componente é o CdR, que compara os campos de identificação e devolve subescores de semelhança, mais uma nota final agregada; é o objetivo 1. O segundo é o XGBoost, um algoritmo de aprendizado de máquina que combina muitas árvores de decisão construídas em sequência. Ele converte os subescores e a nota final em uma probabilidade de par verdadeiro, calibrada fora da amostra; é o objetivo 2. Com essa probabilidade ordena-se a fila de revisão, e a revisão manual segue a fila até o orçamento, isto é, o número de revisões que o serviço pode fazer; é o objetivo 3. À parte, avaliou-se a decisão automática, com o modelo de linguagem, como alternativa à ordenação; é o objetivo 4. A equidade, objetivo 5, examina o desempenho por estrato de processo. O corte permanece em todos os cenários: o que se compara é o que se faz com os potenciais pares abaixo dele. Nenhum componente corrige o OpenRecLink; cada um mede quanto de esforço a revisão manual exige. Os botões numerados no rodapé dos quadros seguintes retomam esses objetivos.

Transição: para comparar as filas é preciso uma medida comum de esforço, que é o tema do próximo quadro.

Se perguntarem: R09 (as quatro ações) e R06 (por que XGBoost).

Fonte: `capitulo3.tex:39`, `conclusao.tex:83` (fluxo), `estudo4_gzcmd.tex:36` (XGBoost sobre os 29 subescores e a nota final), `glossario.tex:34` (CdR).

### S09. A medida: revisões necessárias para recuperar 9, 23 ou 36 pares verdadeiros

A medida é a curva de esforço: para cada fila, quantas revisões são necessárias para recuperar k pares verdadeiros. Supõe-se um revisor perfeito simulado, que recupera todo par verdadeiro revisado e não produz falso positivo; isso não é acerto humano observado. As metas vêm do arranjo sequencial, o ASR, registrado antes da avaliação: ele preserva o corte e, abaixo dele, decide pela política de quatro ações, com o modelo de linguagem como revisor assistente. O ASR opera em dois modos. A vigilância prioriza a sensibilidade, a proporção de pares verdadeiros recuperados, e recupera 23 pares verdadeiros. A confirmação prioriza o valor preditivo positivo (VPP), a proporção de vínculos aceitos que são pares verdadeiros, e recupera 9. A terceira meta, 36, é o total de pares verdadeiros abaixo do corte. Quatro filas são comparadas. O1 ordena pelo escore do OpenRecLink. O3, pela probabilidade calibrada, que vem de um XGBoost validado em partições que mantêm no mesmo grupo todos os potenciais pares de um mesmo óbito. F3 e F4 são fusões por posição recíproca: somam-se, para cada potencial par, os inversos de uma constante mais sua posição em cada fila. A F3 reúne ainda uma fila com medidas contínuas de diferença entre datas de nascimento, as MACD. A mesma curva permite ao serviço escolher a fila e o ponto de parada conforme o orçamento de revisões disponível. Os quadros seguintes usam sempre essas três metas e essas quatro filas.

Transição: com a medida definida, os resultados seguem a ordem dos objetivos, a começar pelo instrumento.

Se perguntarem: R05 (horas como cenário) e R07 (as variantes de fusão).

Fonte: `estudo4_gzcmd.tex:23` (ASR), `estudo4_gzcmd.tex:117` (metas), `estudo4_gzcmd.tex:119` (revisor perfeito), `estudo4_gzcmd.tex:202` (fusões), `glossario.tex:60` (fusão por posição recíproca), `glossario.tex:74` (MACD), `glossario.tex:126` (VPP), `conclusao.tex:17` (vigilância prioriza a sensibilidade, confirmação o VPP), `conclusao.tex:27` (23 recuperados pelo ASR).

### S10. O CdR em Python é aberto e não equivale ao protótipo em Java

Este quadro responde ao primeiro objetivo. O CdR, proposto por Lucena, foi reimplementado em Python e está disponível em código aberto. Para descrever as diferenças em relação ao protótipo em Java, usou-se um lote sintético de 500 pares escolhidos para exercitar casos de fronteira, sem registros reais. Compararam-se os subescores de nome e de data e a nota final. A nota final foi idêntica em 11,2% dos pares, com diferença de até 4 pontos. Esse valor reúne diferenças de critério e de arredondamento e não estima a concordância no extrato. As diferenças são de três tipos. Há erros de implementação do protótipo, corrigidos no Python, como o carregador que descarta a primeira linha das tabelas de frequência. Há divergências de critério em relação à dissertação de Lucena, como o peso das inversões de data. E há melhorias do Python, entre elas os subescores de endereço e de município. A conclusão é que as implementações não são equivalentes e que o efeito sobre dados reais não foi medido. Os estudos seguintes tomam a nota final do Python como dada.

Transição: com o instrumento descrito, passa-se à pergunta sobre qual classificador estima melhor a probabilidade.

Se perguntarem: R13 (diferenças entre as implementações).

Fonte: `estudo1_comparador.tex:3`, `:31` (500 pares), `:71` (11,2%, até 4 pontos, arredondamento), `:78-93` (três tipos), `:128` (conclusão do capítulo).

### S11. A ordem dos classificadores depende da métrica, e as primeiras estimativas eram otimistas

Este quadro responde ao segundo objetivo, na parte dos classificadores. Comparou-se sete classificadores sobre os potenciais pares. Com limiar de 0,5, o maior F1, que é a média harmônica entre o VPP e a sensibilidade, foi do Random Forest (floresta aleatória), 0,910. Pela área sob a curva precisão-sensibilidade, a ordem se inverte pontualmente: a regressão logística supera o Random Forest, com intervalo de confiança da diferença que inclui zero. A regressão logística foi treinada com pesos de classe balanceados, o que deixa o limiar de 0,5 mal posicionado para ela. O balanceamento não trouxe ganho uniforme: o F1 médio de nove estratégias variou de 0,880 a 0,918. Essas estimativas eram otimistas por três razões: indicadores de encerramento por óbito, que constituem vazamento de alvo; partição por linha, e não por óbito; e limiares escolhidos no próprio teste. O vazamento ocorre porque esses indicadores derivam da situação de encerramento do caso no Sinan. No controle com partição por óbito, o F1 do Random Forest com SMOTE, sobreamostragem sintética da classe minoritária, caiu de 0,931 para 0,846. O Estudo 3, que derivou estratégias de vigilância e de confirmação, mostrou que o Random Forest com SMOTE detectou 67 vínculos contra 43 da decisão por nota final de 8 ou mais, mas 25 dos 27 recuperados já eram aceitos pelo corte. Os dois estudos não fundamentam a adoção desses *pipelines*.

Transição: a pergunta seguinte é como transformar essas variáveis em uma probabilidade em que se possa confiar.

Se perguntarem: R14 (Estudo 3 e protocolo).

Fonte: `estudo2_classificadores.tex:11` (sete classificadores), `:62` (F1, AUC-PR, pesos de classe), `:122` (três fontes de otimismo, vazamento), `conclusao.tex:13` (0,880 a 0,918, 0,931 e 0,846), `estudo3_estrategias.tex:23` (67 e 43), `:25` (27 e 25).

### S12. A probabilidade calibrada fora da amostra separa os pares na faixa intermediária da nota final

Ainda no segundo objetivo, a probabilidade de par verdadeiro foi estimada por um XGBoost com 300 árvores, sobre 30 atributos: os 29 subescores do CdR e a nota final. A expressão fora da amostra significa que cada probabilidade vem de um modelo treinado sem os potenciais pares do mesmo óbito. A calibração ajusta os valores do modelo para que correspondam à frequência observada de pares verdadeiros e usou regressão isotônica, uma curva monotônica não paramétrica, ajustada em partições internas, sem acesso aos *labels* do teste. A avaliação deste quadro cobre a faixa intermediária da nota final do CdR, entre 5 e 9, que tem 21.620 potenciais pares e 151 pares verdadeiros. Essa faixa não é a *grey area*, que é definida pelo corte do OpenRecLink. Nela, a área sob a curva ROC, que mede a capacidade de separar pares verdadeiros de não pares, foi de 0,995, contra 0,873 da nota final. A precisão média, que resume a qualidade da ordem no topo da lista, foi de 0,862, contra 0,216. A mediana da probabilidade nos pares verdadeiros foi de 0,895. A avaliação é fora da amostra, mas na mesma coorte; não equivale a validação externa.

Transição: com a probabilidade em mãos, o terceiro objetivo compara as filas.

Se perguntarem: R06 (por que XGBoost).

Fonte: `estudo4_gzcmd.tex:36` (300 árvores, 30 atributos), `:38` (calibração em partições internas), `:40` (21.620, 151, 0,995, 0,873, 0,862, 0,216, 0,895), `capitulo3.tex:11` (faixa de 5 a menos de 9).

### S13. A ordem da fila muda o número de revisões: 125 contra 501 para 23 pares verdadeiros, estimativa pontual

Este quadro e os dois seguintes respondem ao terceiro objetivo. A figura mostra, para cada fila abaixo do corte, quantos pares verdadeiros se recuperam conforme aumenta o número de revisões; o eixo horizontal está em escala logarítmica. Para recuperar os 23 pares verdadeiros da meta da vigilância, a fila pela probabilidade calibrada, O3, exigiu 125 revisões, contra 501 da fila pelo escore do OpenRecLink, O1, uma redução pontual de 75,0%. Na meta de 9, foram 29 contra 51, redução de 43,1%. A fila pela probabilidade recalibrada do ASR, O4, exigiu 106 e 27. A fila pela nota final do CdR, O2, recuperou 9 pares com 11 revisões, mas precisou de 4.163 para os 23. Os pontos do ASR incluem o custo adicional do modelo de linguagem. A vantagem de O3 está no topo da fila: abaixo do corte, a precisão média foi de 0,283 em O3 e de 0,130 em O1, enquanto a AUC foi ligeiramente maior em O1, 0,990 contra 0,985. Há uma inversão na cauda. Para recuperar todos os 36, O3 exigiu 23.443 revisões e O1, 2.521, e a vantagem de O3 sobre O1 vai até cerca de 2.500 revisões. O *bootstrap* sorteia, com reposição, os óbitos e reordena as filas em cada réplica, sem reajustar o modelo. Com 2.000 réplicas, a diferença de O3 menos O1 teve mediana de 21 revisões a menos na meta de 9, com intervalo de 52 a menos a 13 a mais; de 175 a menos na de 23, de 1.006 a menos a 327 a mais; e de 579,5 a mais na de 36, de 1.962 a menos a 23.566 a mais. Em todos, o intervalo inclui o zero. Tudo isso é estimativa pontual, em análise *post hoc*, sob revisor perfeito simulado.

Transição: resta saber se a diferença resiste à incerteza, o que o quadro seguinte examina.

Se perguntarem: R08 (tabela completa) e R06 (Random Forest no lugar do XGBoost).

Fonte: `estudo4_gzcmd.tex:186` (custo adicional do modelo de linguagem), `:194` (29, 51, 75,0%, 43,1%, 106, 27, 0,283, 0,130, 0,990, 0,985, 23.443, 2.521, 2.000 réplicas, medianas e intervalos), `:196` (O2: 11 revisões; 4.163 em `:331`), `:198` (cerca de 2.500 revisões), `:123` (procedimento do *bootstrap*).

### S14. Com IC, só as fusões com o escore reduzem as revisões frente a O1, e nenhuma supera O3

A figura responde à pergunta sobre a incerteza. Cada painel corresponde a uma meta, 9, 23 e 36 pares verdadeiros. Em cada um, mostra-se a mediana e o intervalo de confiança de 95% da diferença de revisões entre uma fila e O1, nas três filas O3, F3 e F4. Valores negativos significam menos revisões que O1. O marcador cheio indica intervalo inteiramente abaixo de zero. As fusões F3 e F4 têm marcador cheio nas metas de 9 e de 36: F4 exigiu, em mediana, 38 revisões a menos na meta de 9 e 1.203 a menos na de 36, e F3, 35 e 894 a menos. Na meta de 23 nenhum intervalo ficou abaixo de zero, e O3 isolada não tem marcador cheio em nenhuma meta. Frente a O3, nenhuma fusão teve intervalo inteiramente abaixo de zero. Há duas cautelas. F4 foi escolhida entre seis variantes avaliadas nos mesmos 36 pares verdadeiros, sem correção para multiplicidade. E, na meta de 36, parte do ganho de F4 vem de priorizar a faixa de 2.522 potenciais pares de maior escore, que contém os 36 pares verdadeiros. A reamostragem sorteia óbitos, não registros, porque os potenciais pares de um mesmo óbito não são independentes, e é condicional às probabilidades e à referência fixas. A regra das metas muda quando a réplica tem menos de 36 pares verdadeiros, e então valem as mesmas frações do total da réplica. Esse achado é *post hoc*, sob revisor perfeito simulado.

Transição: os valores pontuais de cada fila, lado a lado, mostram por que nenhuma ordenação domina.

Se perguntarem: R07 (multiplicidade) e R08 (tabela completa).

Fonte: `estudo4_gzcmd.tex:123` (reamostragem e regra das metas), `:225` (38, 1.203, 35, 894, frente a O3, seis variantes), `:229` (faixa de 2.522), `conclusao.tex:21-23`.

### S15. Nenhuma ordenação domina em todas as cargas

Lido por colunas, o quadro mostra que nenhuma fila é a melhor em todas as cargas. Com as 96 revisões que o ASR encaminha ao revisor humano na vigilância, F3 recupera 25 pares verdadeiros, O3, 22, O1, 19 e F4, 18. Nas metas de revisões necessárias, F4 é a menor em 9 e em 36, com 10 e 1.387, e F3 é a menor em 23, com 64. O3 é a terceira em 23, com 125, e a pior na recuperação integral, com 23.443 revisões. O1 é a mais lenta nas metas menores, mas supera O3 na integral, com 2.521. O negrito marca o melhor valor pontual de cada coluna, e F3 e F4 se alternam. Na confirmação, com o orçamento de 17 revisões, F4 recupera 16 pares verdadeiros, a nota final do CdR, 9, e O3, 6. As diferenças entre as fusões e O3 não foram demonstradas: nenhum intervalo frente a O3 ficou abaixo de zero. Os valores são pontuais, *post hoc* e sob revisor perfeito simulado, e as fusões foram escolhidas entre variantes. Daí a recomendação de escolher a fila pela curva de esforço, conforme o orçamento disponível, em vez de fixar uma só. Também por isso a F3, que depende das MACD, só entra na recomendação quando essas medidas estiverem disponíveis.

Transição: até aqui a automação ficou de fora; o quarto objetivo a examina.

Se perguntarem: R08 e R05 (confirmação com 17 revisões).

Fonte: `estudo4_gzcmd.tex:147` (96 revisões do ASR), `:214-217` e `:241-243` (valores da tabela), `:225` (25, 22, 19, 18), `conclusao.tex:23`, `conclusao.tex:87` (17 revisões: F4 16, O2 9, O3 6).

### S16. Na vigilância, abaixo do corte, a automação aceitou 4 pares verdadeiros e 21 falsos positivos

O quarto objetivo verifica se a automação reduz a revisão além da ordenação, e a que custo em falsos positivos. O ASR preserva o corte e, abaixo dele, aplica a política de quatro ações: aceitar como vínculo, rejeitar, enviar à revisão por um modelo de linguagem ou enviar ao revisor humano, escolhendo a ação de menor *expected loss*. O modelo de linguagem é o Claude Haiku 5.5, que revisa o dossiê de cada potencial par, um resumo estruturado e pseudonimizado dos subescores. Na vigilância, o ASR enviou 344 potenciais pares ao modelo e 96 ao revisor humano. Todas as 25 aceitações abaixo do corte vieram do modelo: 4 eram pares verdadeiros e 21, falsos positivos. O VPP dessas aceitações foi de 16,0%, com intervalo de confiança de 95% de 6,4 a 34,7. As proporções de erro do revisor previam 9,06 falsos positivos, contra 21 observados. Dez dos 21 estão em região sem revisão documentada; se algum for par verdadeiro, o VPP está subestimado. Dos 36 pares verdadeiros abaixo do corte, 13 não foram recuperados na vigilância: um pela guarda de nota final, seis pela comparação de custos e seis por decisão de não par do modelo, sem chegar ao revisor. Na confirmação, foram 27 pares perdidos e 4 falsos positivos. Por isso, nas condições avaliadas, o modelo não deve aceitar vínculos sozinho. O ASR teve registro interno anterior à avaliação, sem critério de sucesso; a leitura como resultado negativo é posterior. Os números supõem revisor perfeito simulado nos encaminhamentos.

Transição: a comparação com a ordenação, na mesma carga de revisão humana, vem a seguir.

Se perguntarem: R10 (modelos, dossiê e ancoragem) e R09 (as quatro ações).

Fonte: `estudo4_gzcmd.tex:89` (dossiê), `:147` (344, 96, 25, 4, 21), `:149` (confirmação: 27 perdidos, 4 falsos positivos), `:311` (9,06 previstos, 6,4 a 34,7, registro interno), `:313` (dez dos 21), `conclusao.tex:27` (13 perdidos: 1, 6 e 6).

### S17. Na vigilância, a automação recupera no máximo um par verdadeiro a mais que a fila ordenada

Na mesma carga humana, a comparação é direta. Com as 96 revisões que o ASR encaminha ao revisor humano, o arranjo recupera 23 pares verdadeiros, quatro deles aceitos pelo modelo de linguagem, e O3 recupera 22. A diferença máxima é de um par verdadeiro em 36, e o ASR acrescenta 21 falsos positivos e a avaliação de 344 potenciais pares pelo modelo. Se os vínculos propostos pelo modelo também forem encaminhados ao humano, o ASR usa 121 revisões para os 23, e O3, 125. Essa diferença não sustenta superioridade, e a ordem dos custos depende do custo atribuído ao modelo. Na confirmação, a ordenação pela nota final do CdR recupera os mesmos 9 pares do ASR com 11 revisões, contra 17. O modelo de linguagem também foi testado como filtro de NONMATCH, que descarta antes do humano os pares que classifica como não pares. Com a carga humana fixada em 22 revisões, recuperou 13 pares verdadeiros contra 6 de O3, mas descartou 9 dos 22 do topo da fila. A vantagem sobre O3 existe só com essa carga fixa, que pertence a uma réplica; na outra foram 18 revisões, e na unidade de custo da tese o filtro perde para O3. Quatro modelos foram comparados como revisores: Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 e Claude Fable 5.1, com VPP de 16,0% a 19,0% nas aceitações da vigilância, e nenhum ganho demonstrável pelo tamanho. Esses usos permanecem hipóteses exploratórias, sob revisor perfeito simulado.

Transição: resta o quinto objetivo, sobre se o desempenho se distribui de modo equânime.

Se perguntarem: R05 (confirmação), R10 (modelos) e R11 (filtro).

Fonte: `estudo4_gzcmd.tex:196` (96, 23, 22, 121, 125, 344, 11 e 17), `:271-273` (13 contra 6, 9 dos 22, 22 e 18, 1.172,5 contra 796), `:285` (ordem dos custos), `:297` (16,0, 19,0, 19,0 e 18,2), `conclusao.tex:29`.

### S18. A perda de sensibilidade concentra-se no passo 11 e é herdada do corte

O quinto objetivo avalia se o desempenho se distribui igualmente entre estratos. A diferença mais clara está no passo 11 da blocagem. Na vigilância, a sensibilidade automática do ASR foi de 42,5% nesse passo, contra 95,0% no passo 6 e 96,1% no passo 9. A razão está no corte: isolado, ele aceita 15 dos 40 pares verdadeiros do passo 11, 37,5%, contra 95,0% e 94,5% nos outros dois passos. A disparidade é herdada do corte, e o ASR, que o preserva, pouco a altera: acrescenta dois dos 25 pares verdadeiros abaixo do corte nesse passo. Na confirmação, a sensibilidade automática no passo 11 foi de 47,5%. Com revisor perfeito nos 14 pares encaminhados, o limite seria de 77,5%, valor que não é acerto observado. O passo é um estrato do processo, não um atributo demográfico. Entre sexo e faixa etária, nenhuma rejeição após correção de Benjamini-Hochberg foi encontrada, o que não demonstra paridade: com 71 pares verdadeiros no sexo feminino e 176 no masculino, os denominadores limitam a detecção, e o poder não foi calculado. A correção de Benjamini-Hochberg controla a proporção esperada de falsas descobertas entre as comparações. A raça/cor não consta do extrato. A equidade da fila recomendada não foi examinada.

Transição: fechados os resultados, cabe dizer com precisão o que a tese sustenta e o que não sustenta.

Se perguntarem: R12 (equidade completa).

Fonte: `estudo5_equidade.tex:52` (15 de 40, 37,5%, 95,0%, 94,5%, dois dos 25, 47,5%), `:54` (14 encaminhados, 77,5%), `:42` (71 e 176), `conclusao.tex:31-33`.

### S19. O que a tese sustenta e o que não sustenta

O quadro separa o que os dados sustentam do que não sustentam. A tese sustenta que as fusões F3 e F4 reduziram as revisões frente ao escore do OpenRecLink nas metas de 9 e de 36 pares verdadeiros, com intervalo de confiança inteiramente abaixo de zero, em análise *post hoc* e sob revisor simulado sem erros. Sustenta que a ordem dos potenciais pares na fila altera o número de revisões. Sustenta que a automação recuperou, na vigilância, no máximo um par verdadeiro a mais que a ordenação pela probabilidade calibrada e produziu 21 falsos positivos. Sustenta que a disparidade de sensibilidade se concentra no passo 11 e acompanha o corte. E sustenta que o CdR em Python não é equivalente ao protótipo em Java. A tese não sustenta que a probabilidade calibrada isolada reduza as revisões, porque os intervalos incluem o zero nas três metas. Não sustenta que as fusões superem a probabilidade isolada, nem que alguma ordenação domine em todas as cargas. Não sustenta que o modelo de linguagem aceite vínculos sozinho, que as estratégias dos Estudos 2 e 3 devam ser adotadas, que a redução valha para outra coorte ou para revisores reais, nem que diminua a subnotificação. Os Estudos 2 e 3 não fundamentam adoção porque suas estimativas são otimistas, pelo vazamento de alvo, pela partição por linha e pelos limiares escolhidos no teste. A calibração por âncoras, que ajusta uma sigmoide a pares de nota final extrema, tampouco é absoluta na faixa intermediária da nota final. E a tese não sustenta que haja paridade entre subgrupos demográficos. Todo o conjunto vale para esta coorte e depende da validação descrita adiante.

Transição: as limitações explicam por que esses limites existem.

Se perguntarem: R07 (multiplicidade) e R14 (Estudos 2 e 3).

Fonte: `conclusao.tex:37-57` (listas), `conclusao.tex:13` (três fontes de otimismo), `conclusao.tex:56` (âncoras).

### S20. Três limitações condicionam a leitura

A primeira limitação é a referência operacional. Ela resulta do corte e da revisão manual de origem, feita por revisora única, sem classificação independente e sem registro da ordem de exame, das fontes e da parada. Os 36 pares verdadeiros abaixo do corte estão entre os 2.522 potenciais pares de maior escore, padrão compatível com revisão restrita a essa faixa, o que favoreceria O1. Se a revisora usou as evidências do CdR, o favorecimento seria de O2, O3 e O4. Sem suposições sobre esse mecanismo, a direção e a magnitude do viés não são identificáveis. A segunda é a coorte única, com 247 pares verdadeiros e 36 abaixo do corte. O intervalo de confiança da sensibilidade automática do ASR na vigilância, 87,0%, vai de 82,3% a 90,7%. Em 971 das 2.000 réplicas do *bootstrap* restaram menos de 36 pares verdadeiros. O ganho pontual de O3 na vigilância não se manteve quando o Random Forest substituiu o XGBoost. A terceira é que o revisor humano não foi observado: todas as contagens supõem revisor perfeito simulado, e a análise é *post hoc*. Há ainda uma limitação de blocagem: pares da mesma pessoa que os passos 6, 9 e 11 não geraram ficam fora do extrato, e as sensibilidades são condicionais aos potenciais pares gerados. Essas limitações são do delineamento e da documentação disponível, e não do trabalho de revisão de origem.

Transição: com os limites claros, a recomendação para o serviço pode ser formulada.

Se perguntarem: R01 (viés de verificação) e R08 (intervalos).

Fonte: `conclusao.tex:71` (2.522, O1, O2 a O4), `conclusao.tex:73` (blocagem), `conclusao.tex:75` (87,0%, 82,3 a 90,7, 971 de 2.000, Random Forest), `conclusao.tex:77` (revisor não observado).

### S21. Fluxo recomendado, a validar em coorte nova

A recomendação, a validar em coorte nova com referência cega, é preservar o corte do OpenRecLink, que gera os potenciais pares e aceita os de escore no ponto de corte ou acima. Abaixo dele, o CdR calcula os subescores, e o XGBoost os converte, com a nota final, em probabilidade calibrada fora da amostra. A fila de revisão manual segue essa probabilidade, isolada, em O3, ou fundida ao escore do OpenRecLink, em F4, ou em F3 quando houver as MACD. Nenhuma ordenação dominou em todas as cargas; por isso a escolha entre elas se faz pela curva de esforço. O serviço traçaria a curva de cada fila sobre uma amostra revisada e adotaria a que recuperasse mais pares verdadeiros no orçamento disponível. O ponto de parada é fixado pelo orçamento humano, lido nessa curva. Os potenciais pares abaixo da parada não foram examinados e devem ser registrados como não examinados, não como não pares: a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado. O modelo de linguagem pode apoiar a revisão, mas não deve aceitar vínculos sozinho; a decisão de vínculo que ele propõe segue ao revisor humano. O envio de dossiês a provedores externos, como nesta avaliação, não deve se estender a dados do serviço, que exigiriam modelo executado em ambiente controlado. Se a probabilidade for aprendida com subamostragem dos negativos, convém evitar as razões 1:1 e 1:5, que reduziram a mediana de pares verdadeiros entre os 96 primeiros de 22, na execução sem subamostragem, para 13 e 18,5. O que a tese avaliou foi uma hipótese de trabalho: nenhuma dessas etapas foi validada em outra coorte. A tese não mediu o ganho de horas em um serviço real.

Transição: falta dizer como essa recomendação poderia ser refutada.

Se perguntarem: R04 (não examinados), R05 (horas), R06 (estimador) e R11 (filtro).

Fonte: `conclusao.tex:81-93`, `estudo4_gzcmd.tex:265` (subamostragem), `estudo4_gzcmd.tex:327-329` (recomendação e pares não examinados).

### S22. A hipótese pode ser refutada: o critério está declarado

A hipótese central exige outra coorte ou período, não usados para fixar o corte nem as regras. Ordenação e orçamento são definidos antes dos desfechos, e a referência é cega e independente. A hipótese seria refutada se a ordenação O3 não atingisse a recuperação do ASR com pelo menos 25% menos revisões que O1, pelo limite inferior do intervalo de 95% entre partições predefinidas, em cada meta, 23 na vigilância e 9 na confirmação. A margem foi escolhida com os dados de desenvolvimento à vista, nos quais as reduções pontuais foram de 75,0% e 43,1%, e não foi testada. Uma segunda interpretação, de que as quatro ações não acrescentam ganho material, seria refutada se o ASR recuperasse, na mesma carga humana, mais pares verdadeiros que O3, por diferença superior à amplitude entre réplicas. Como o intervalo da diferença entre O3 e O1 incluiu zero nas três metas, o tamanho da coorte nova deve vir de um cálculo de poder. O registro deve incluir F4 e CM como comparações secundárias. CM é a fila de O3 acrescida de atributos de contexto por óbito e das MACD. Nenhuma dessas avaliações foi realizada; a refutação é um critério proposto, não um resultado.

Transição: com o critério declarado, fecham-se as contribuições.

Se perguntarem: R09 (quatro ações) e R07 (comparações secundárias).

Fonte: `conclusao.tex:97` e `:99`, `estudo4_gzcmd.tex:331`, `capitulo1.tex:43`, `glossario.tex:20` (CM).

### S23. Contribuições: a medida, a decomposição entre ordenação e automação, e o código aberto

A primeira contribuição é a medida: a tese separa os 211 pares verdadeiros aceitos pelo corte dos 36 recuperados na revisão manual abaixo dele e usa a curva de esforço para escolher a fila e o ponto de parada. A segunda é a decomposição, na mesma carga humana, entre ordenar a fila e automatizar a decisão. A ordenação por escore com revisão manual já havia sido descrita por Sariyar e Borg, em 2010, e o aprendizado ativo, em que o modelo indica quais pares rotular, por Sariyar e Borg, em 2012; a revisão assistida por tecnologia usa um modelo que ordena os itens e uma regra de parada. A tese não reivindica essas técnicas; acrescenta a comparação posterior ao registro entre ordenar e automatizar. A terceira é o código aberto do CdR em Python e do pacote GZ-CMD, que registra cada decisão, sem auditoria independente. A quarta são os agregados e o *benchmark* de acesso restrito.

Transição: encerra-se com os agradecimentos.

Se perguntarem: não se aplica.

Fonte: `conclusao.tex:61-67`.

### S24. Agradecimentos

Registram-se os agradecimentos à orientadora, Rejane Sobrino Pinheiro; ao LinkDataPop, pelo extrato e pela revisão manual de origem; aos criadores do OpenRecLink; à banca; e à CAPES, pelo Programa CAPES-Epidemias. A palavra está com a banca para a discussão.

Transição: abre-se a arguição.

Se perguntarem: ver a seção de perguntas prováveis.

Fonte: `tese.tex:480-490`.

## Reserva

Falas curtas para apresentar cada quadro de reserva quando for chamado. Cada uma tem até 80 palavras.

### R01. Viés de verificação da referência

A referência é o produto do corte e da revisão manual de origem, feita por revisora única, sem dupla revisão. Ordem de exame, fontes e parada não estão documentadas, e o acesso da revisora à nota final do CdR é hipótese de trabalho. Os 36 pares verdadeiros estão entre os 2.522 de maior escore, o que favoreceria O1; o uso das evidências do CdR favoreceria O2, O3 e O4. A direção do viés, portanto, não é identificável sem suposições.

Fonte: `capitulo_dados.tex:77`, `conclusao.tex:71`, `estudo4_gzcmd.tex:335`.

### R02. Ética, LGPD e APIs externas

O projeto tem parecer do comitê de ética do IESC-UFRJ e do da Secretaria Municipal de Saúde. O extrato não tem nomes, nome da mãe nem endereço nominal, mas contém quase-identificadores; é pseudonimizado, não anônimo. Os dossiês enviados às APIs não traziam datas completas nem os identificadores REFREC e COMPREC, mas incluíam o `pair_id`, um hash desses identificadores que não garante anonimização. A aplicabilidade da LGPD a essa transferência não foi analisada nos pareceres e é limitação declarada.

Fonte: `capitulo_dados.tex:27`, `:29`, `:35`, `:39`, `:51`.

### R03. Corte único por passo e os limiares de Fellegi e Sunter

O modelo de Fellegi e Sunter prevê dois limiares: acima do superior, aceita; abaixo do inferior, descarta; entre eles, há região sem decisão automática. O OpenRecLink aplicou um único corte por passo, sem limiar inferior, de modo que a *grey area* reúne todos os potenciais pares abaixo do corte, e não só os de escore intermediário. O corte é preservado porque integra o processo que produziu a referência, sem presumir superioridade sobre uma estratégia determinística.

Fonte: `capitulo2.tex:16`, `capitulo_dados.tex:23`, `capitulo3.tex:9`.

### R04. Subnotificação e perda de vínculo

Subnotificação é a ausência, no Sinan, do caso de uma pessoa cujo óbito consta no SIM; perda de vínculo é erro do próprio *linkage*, que faz caso notificado parecer subnotificado. Os potenciais pares abaixo do ponto de parada ficam registrados como não examinados: a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado. Os 247 pares verdadeiros correspondem a 204 óbitos, e o extrato não estima subnotificação.

Fonte: `capitulo1.tex:20`, `conclusao.tex:91`, `capitulo_dados.tex:57`.

### R05. Revisões em horas: a unidade é a revisão

A unidade de esforço é a revisão, porque os 20 minutos por par são hipótese de planejamento, e não tempo medido. Nesse cenário, as 96 revisões da vigilância equivalem a 32 horas, e as 17 da confirmação, a 6. Não há equivalência medida entre revisão por modelo de linguagem e humana. Na confirmação, com 17 revisões, a nota final do CdR recupera 9 pares, F4, 16 e O3, 6.

Fonte: `estudo4_gzcmd.tex:119`, `:257`, `tables/editorial/tab_cap8_esforco.tex:10`, `:14`, `conclusao.tex:87`.

### R06. Por que XGBoost

O XGBoost foi escolhido entre as configurações do comparativo QW1, na mesma base, em que teve o maior F1 médio na validação cruzada, 0,924, contra 0,913 do Random Forest; isso não estabelece superioridade da família. Com Random Forest no lugar do XGBoost, o ganho pontual de O3 na vigilância não se manteve, com 498 revisões contra 501 de O1, mas o da confirmação sim, com 32 contra 51.

Fonte: `estudo2_classificadores.tex:88`, `:92`, `estudo4_gzcmd.tex:36`, `:194`.

### R07. Multiplicidade na escolha de F4

A F4 foi escolhida entre seis variantes: F1 a F4 e duas filas adaptativas por amostragem de Thompson, todas avaliadas nos mesmos 36 pares verdadeiros, sem correção para multiplicidade. Fixar as variantes antes do cálculo não constitui pré-registro, e a verificação da referência pode favorecer O1 na cauda. Na validação, F4 e CM, que é O3 com contexto por óbito e MACD, entram como comparações secundárias; a primária é O3 contra O1.

Fonte: `estudo4_gzcmd.tex:202`, `:331`, `conclusao.tex:99`.

### R08. Tabela completa do *bootstrap*

A tabela traz a mediana e o intervalo de 95% da diferença de revisões, em 2.000 réplicas agrupadas por óbito. O3 menos O1 inclui zero nas três metas; F3 e F4 menos O1 ficam abaixo de zero nas metas de 9 e de 36; nenhuma fusão fica inteiramente abaixo de zero frente a O3. Em 971 réplicas restaram menos de 36 pares verdadeiros, e as metas passam a 25%, 64% e 100% do total, o que altera o estimando.

Fonte: `estudo4_gzcmd.tex:194`, `:225`, `tables/editorial/tab_cap8_fusao_robustez.tex:10-15`, `conclusao.tex:75`.

### R09. As quatro ações da política

A política compara, para cada potencial par, a *expected loss*, o custo esperado, de quatro ações: aceitar como vínculo, rejeitar, revisão humana e revisão por modelo de linguagem, e escolhe a de menor valor. Guardas fixas rejeitam nota final abaixo de 5 e aceitam nota final de 9 ou mais com probabilidade de pelo menos 0,9. Com revisor perfeito, a ordenação com parada por orçamento coincide com a regra de duas ações.

Fonte: `estudo4_gzcmd.tex:53`, `:55-60`, `:317`.

### R10. Modelos de linguagem, dossiê e ancoragem

Quatro modelos foram comparados como revisores: Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 e Claude Fable 5.1. O VPP nas aceitações da vigilância foi de 16,0%, 19,0%, 19,0% e 18,2%, sem ganho demonstrável pelo tamanho. O dossiê traz 14 dos 29 subescores. Sem as saídas do motor no dossiê, 18 pares verdadeiros rejeitados antes não foram rejeitados de novo, achado compatível com ancoragem ou abstenção generalizada, sem identificar o mecanismo.

Fonte: `estudo4_gzcmd.tex:89`, `:137`, `:297`, `tables/editorial/tab_cap8_tamanho_modelo.tex:11-14`.

### R11. Filtro de NONMATCH

No filtro, o modelo de linguagem descarta o que classifica como não par, e o revisor humano vê o restante. Com 96 potenciais pares enviados ao modelo, a carga humana foi de 22 revisões, e o filtro recuperou 13 pares verdadeiros contra 6 de O3 na mesma carga. Descartou, porém, 9 dos 22 pares verdadeiros do topo, e a carga de 22 pertence a uma réplica. Na unidade de custo da tese, o filtro custa 1.172,5 e O3, 796.

Fonte: `estudo4_gzcmd.tex:271-273`, `tables/editorial/tab_cap8_filtro_llm.tex:12`.

### R12. Equidade completa

O ASR teve sensibilidade de 87,0% na vigilância e de 88,7% na confirmação, com VPP de 91,1% e de 98,2%. Por passo, a sensibilidade no passo 11 foi de 42,5% e 47,5%; o corte isolado aceita 37,5% dos pares verdadeiros desse passo. Por sexo, não houve rejeição após correção, o que não demonstra paridade, e o poder não foi calculado. Raça/cor não consta do extrato, e a equidade da fila recomendada não foi examinada.

Fonte: `estudo5_equidade.tex:34`, `:42`, `:52`, `:99`, `:101`.

### R13. Diferenças entre o CdR em Python e o protótipo em Java

Os erros do protótipo em Java, corrigidos no Python, incluem o carregador que descarta a primeira linha e inverte a ordem das tabelas de frequência. As divergências de critério em relação a Lucena incluem o peso de concordância 0,8 das inversões de data, que o Java segue e o Python substitui por 1,0. As melhorias incluem os subescores de endereço e de município e o código aberto. A observação é sobre as implementações, e a equivalência não foi demonstrada.

Fonte: `estudo1_comparador.tex:78-93`, `:128`.

### R14. Estudo 3 e protocolo

O Estudo 3 mostrou que o Random Forest com SMOTE detectou 67 vínculos no teste, contra 43 da decisão por nota final de 8 ou mais, mas 25 dos 27 recuperados já eram aceitos pelo corte. As estimativas são otimistas por vazamento de alvo, partição por linha e limiares escolhidos no teste, e por isso não sustentam adoção. No Estudo 4, a partição é por óbito, com cinco partições externas, cinco internas e semente 42.

Fonte: `estudo3_estrategias.tex:23`, `:25`, `estudo2_classificadores.tex:122`, `estudo4_gzcmd.tex:36-38`.

## Perguntas prováveis

Cada resposta tem até 70 palavras. "Apoio" indica o quadro ou a reserva que sustenta a resposta na tela.

### Referência e viés de verificação

**P1. O que autoriza chamar um par de "verdadeiro"?**

"Verdadeiro" remete à referência operacional, que combina o corte do OpenRecLink, campo PAR igual a 1, e a revisão manual de origem abaixo dele, PAR igual a 2. Não é verdade absoluta. Sem classificação independente, não se estima a concordância entre avaliadores, e a avaliação pode ser otimista, em magnitude que só uma revisão independente e cega quantificaria.

Apoio: S03, S07, R01. Fonte: `capitulo_dados.tex:73`, `:77`.

**P2. Em que direção o viés da referência opera?**

Depende de dois mecanismos não documentados. Se a revisão seguiu o escore do OpenRecLink, os pares verdadeiros se concentram no topo dessa ordem, o que favorece O1. Se usou a nota final e os subescores do CdR, favorece O2, O3 e O4. Sem suposições, direção e magnitude não são identificáveis. Nos cenários de erro da referência, a margem de 25% de O3 na meta de 23 persiste pela mediana na vigilância.

Apoio: R01. Fonte: `capitulo_dados.tex:77`, `estudo4_gzcmd.tex:339`, `conclusao.tex:71`.

**P3. Os 21 falsos positivos são mesmo falsos?**

Dez dos 21 estão abaixo do limiar por passo, em região sem revisão documentada. Se algum for par verdadeiro não verificado, o VPP de 16,0% está subestimado. O teste de identificadores fortes foi inconclusivo: nenhum grupo amostrado apresentou concordância máxima simultânea nos sete subescores examinados. A restrição à aceitação autônoma é cautela diante do VPP e da incerteza da referência.

Apoio: S16, R10. Fonte: `estudo4_gzcmd.tex:313`, `:339`, `:341`.

**P4. O CdR foi usado na revisão dos passos exportados?**

A documentação de origem registra o uso do CdR na revisão do passo aberto. Para os passos 6, 9 e 11, a ordem entre o CdR e a revisão, a data e o que a revisora consultou não estão documentados. Por isso a tese adota como hipótese de trabalho que o CdR precedeu a revisão, com nota final e subescores disponíveis, e declara o efeito possível sobre O2 a O4.

Apoio: S07, R01. Fonte: `capitulo_dados.tex:77`, `estudo4_gzcmd.tex:361`.

### Posição frente ao OpenRecLink e por que preservar o corte

**P5. Por que preservar o corte?**

O corte integra o processo que produziu a referência, e preservá-lo evita presumir que outra regra seja melhor, inclusive uma estratégia determinística. Também mantém os 211 pares verdadeiros que ele aceita. O pós-processamento atua só abaixo do corte, na *grey area*; a medida é o esforço de revisão necessário ali.

Apoio: S03, S08, R03. Fonte: `capitulo3.tex:9`, `:17`, `conclusao.tex:81`.

**P6. Como a *grey area* se relaciona com o modelo de Fellegi e Sunter?**

No modelo, dois limiares dividem o espaço de escores: acima do superior aceita-se, abaixo do inferior descarta-se, e entre eles há região sem decisão automática. O OpenRecLink aplicou um único corte por passo, sem limiar inferior. A *grey area* é, portanto, tudo que está abaixo do corte. O corte da nota final em 9, que recuperaria 37,7%, é cenário hipotético do CdR, e não o do processo.

Apoio: R03. Fonte: `capitulo2.tex:16`, `capitulo_dados.tex:23`, `capitulo3.tex:11`.

**P7. A tese corrige o OpenRecLink?**

Não. O trabalho mede e procura reduzir o custo da etapa de revisão manual dos potenciais pares abaixo do corte, etapa que o processo já prevê. Os vínculos aceitos pelo corte permanecem, e nenhum resultado avalia o OpenRecLink como ferramenta. O que se propõe, a validar em coorte nova, é uma ordem de revisão e um ponto de parada, com a escolha da fila feita pela curva de esforço.

Apoio: S04, S08, S21. Fonte: `capitulo3.tex:17`, `:39`, `conclusao.tex:81`.

### Subnotificação e perda de vínculo

**P8. A redução de revisões diminui a subnotificação?**

Não. A tese não estimou efeito sobre a subnotificação. A redução de revisões não é redução de óbitos não notificados. Os potenciais pares abaixo do ponto de parada devem ser registrados como não examinados, não como não pares: a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado, o que separa a subnotificação da perda de vínculo.

Apoio: S21, R04. Fonte: `conclusao.tex:54`, `:91`, `capitulo1.tex:20`.

**P9. Por que 247 pares verdadeiros e 1.515 declarações de óbito?**

Os 247 pares verdadeiros correspondem a 204 óbitos distintos do SIM, porque uma mesma pessoa pode ter sido notificada mais de uma vez no Sinan. O extrato tem 1.515 declarações de óbito com ao menos um potencial par nos três passos. A razão entre 204 e 1.515 não estima subnotificação, pois o extrato reúne três passos probabilísticos, após a retirada dos vínculos determinísticos.

Apoio: R04. Fonte: `capitulo_dados.tex:57`.

### Custo da revisão e horas

**P10. Quantas horas a ordenação economiza?**

A tese não mede horas economizadas. Como cenário de planejamento, adotou-se 20 minutos por par: 96 revisões equivalem a 32 horas na vigilância, e 17 revisões, a 6 horas na confirmação. Os cenários são hipóteses, sem equivalência medida entre revisão por modelo de linguagem e humana, e não foram comparados ao volume de trabalho da vigilância do óbito no município.

Apoio: R05. Fonte: `estudo4_gzcmd.tex:119`, `:257`, `tables/editorial/tab_cap8_esforco.tex:10`, `:14`.

**P11. E se o revisor humano real errar?**

A sensibilidade humana real não foi medida; todas as contagens supõem revisor perfeito simulado. Como sensibilidade, repetiu-se o cálculo com omissão humana de 5% e com sensibilidades de 0,95, 0,99 e 1,00, sem inversão de ordem entre estratégias com a mesma carga humana. A revisão humana observada, com dupla revisão e concordância medida, é próximo passo.

Apoio: S09, S20. Fonte: `estudo4_gzcmd.tex:119`, `:279`, `conclusao.tex:77`, `:101`.

**P12. Como obter a curva de esforço no serviço, sem referência?**

A curva depende de amostra revisada de forma independente. O serviço traçaria a curva de cada fila sobre uma amostra revisada e adotaria a que recuperasse mais pares verdadeiros no orçamento disponível. A referência da coorte nova pode vir de amostragem estratificada, em revisão cega, com dois revisores e arbitragem.

Apoio: S21, R01. Fonte: `conclusao.tex:89`, `:101`, `estudo4_gzcmd.tex:341`, `glossario.tex:42`.

### Ética, LGPD e APIs externas

**P13. Dados de saúde foram enviados a APIs externas?**

Parte do processamento ocorreu fora da instituição. O extrato pseudonimizado foi à plataforma Modal nos experimentos QW1, e os dossiês de revisão foram enviados a APIs externas: Anthropic, OpenCode Go, Levanto, Fireworks AI e OpenAI. A tese registra isso como limitação e recomenda, para dados do serviço, execução local em ambiente controlado, com modelo executado ali, não avaliado.

Apoio: R02. Fonte: `capitulo_dados.tex:35`, `:51`, `conclusao.tex:93`.

**P14. Os dossiês não tinham identificadores internos, mas incluíam o `pair_id`?**

Os dossiês não continham datas completas nem os identificadores internos REFREC e COMPREC, mas continham o `pair_id`. Ele é um hash SHA-256, sem sal, da composição de COMPREC, REFREC e PASSO, truncado em 16 caracteres, recalculável por quem detém os identificadores e por enumeração: o produto das cardinalidades é de 88.118.460 combinações. A tese declara que ele não garante anonimização.

Apoio: R02. Fonte: `capitulo_dados.tex:35`, `:39`, `:51`, `:123`.

**P15. Qual a base legal do tratamento?**

O tratamento no laboratório se fundamenta nos arts. 7º, IV, e 11, II, c, da Lei n. 13.709/2018, que admitem o tratamento em estudos por órgão de pesquisa, e no art. 13, que o restringe ao órgão de pesquisa, em ambiente controlado, com pseudonimização sempre que possível. A aplicabilidade à transferência a operadores externos não foi objeto de análise específica nos pareceres e é limitação declarada.

Apoio: R02. Fonte: `capitulo_dados.tex:33`, `:35`.

### Escolha do estimador, calibração e partição

**P16. Por que XGBoost?**

O XGBoost veio do comparativo QW1, na mesma base: teve o maior F1 médio na validação cruzada, 0,924, contra 0,913 do Random Forest. Isso informou a escolha entre as configurações avaliadas, sem estabelecer superioridade da família, e as métricas fora da amostra não incorporam essa seleção. Com Random Forest, o ganho pontual de O3 na vigilância não se manteve, com 498 revisões contra 501 de O1; na confirmação, 32 contra 51.

Apoio: S12, R06. Fonte: `estudo2_classificadores.tex:88`, `:92`, `estudo4_gzcmd.tex:36`, `:194`.

**P17. Por que não uma rede neural?**

No QW1, o TabNet ficou em último na validação cruzada, com F1 de 0,821, e as duas MLP tiveram F1 de 0,859 e 0,844. O resultado do TabNet não caracteriza falha geral das redes neurais: a base tem 247 pares verdadeiros, e o experimento não isola esse efeito. O desempenho histórico informou a escolha do XGBoost entre as configurações avaliadas, sem estabelecer superioridade da família.

Apoio: R06. Fonte: `estudo2_classificadores.tex:88`, `:92`.

**P18. Por que isotônica, e não Platt scaling ou âncoras?**

O *Platt scaling* aplicado à nota final só muda a escala e não acrescenta informação discriminativa. A calibração por âncoras preservou a ordem da margem do XGBoost, mas não forneceu probabilidades absolutas comparáveis às da isotônica supervisionada. A política usa, por isso, a isotônica. O ASR é o braço da isotônica pela regra registrada, e o Platt estava previsto só para partições com menos de dez pares PAR igual a 2.

Apoio: S12, R06. Fonte: `estudo4_gzcmd.tex:36`, `:46`, `:313`.

**P19. Qual o papel da partição por óbito?**

Na validação aninhada, as partições são agrupadas por REFREC, de modo que um mesmo óbito não contribui para o treino e o teste da mesma partição. COMPREC não é agrupado simultaneamente, o que foi examinado como sensibilidade. Foram cinco partições externas e cinco internas, com semente 42. A avaliação não equivale a validação externa, e o agrupamento reduziu a precisão média em relação à partição por potencial par.

Apoio: S12, R14. Fonte: `estudo4_gzcmd.tex:36-38`, `:40`.

**P20. Qual o custo computacional do *bootstrap* e da calibração aninhada?**

A tese informa o ambiente, não tempos de execução do *bootstrap* nem da calibração aninhada. Os modelos arbóreos, a regressão logística, a SVM e a MLP foram treinados em estação local com 32 GB de memória; o TabNet, em nuvem, com GPU NVIDIA L4 de 24 GB. O ambiente foi fixado em Python 3.12, com Papermill e Git, e a fragilidade numérica entre *threads*, sementes e partições é risco de implantação.

Apoio: R06. Fonte: `capitulo_dados.tex:115`, `:117`, `estudo4_gzcmd.tex:355`.

**P21. Como se reproduz a avaliação?**

A reprodução integral da avaliação do ASR exige o checkout original, com os bytes do pré-registro, e insumos privados. O depósito do Zenodo de acesso restrito reúne os resultados históricos sobre a base inteira, não o arranjo atual. Os agregados e os arquivos por `pair_id` estão no repositório público da tese, sem *label*; os *labels* estão no depósito restrito.

Apoio: R14. Fonte: `estudo4_gzcmd.tex:359`, `conclusao.tex:67`.

**P22. Subamostrar os negativos ajuda?**

No treinamento, a subamostragem reduz só os negativos e mantém a prevalência na avaliação. As razões 1:1 e 1:5 degradaram o topo da fila: a mediana de pares verdadeiros entre os 96 primeiros caiu de 22 para 13 e 18,5. As razões 1:20 e 1:50 não se distinguem de modo demonstrável do treino sem subamostragem. A tese sustenta evitar 1:1 e 1:5.

Apoio: S21. Fonte: `estudo4_gzcmd.tex:265`.

**P23. O que as MACD acrescentam?**

As MACD são atributos derivados das datas de nascimento. Na ablação da política de três ações, as revisões esperadas diminuíram 55,0% na vigilância e 34,7% na confirmação. Na fila, O3 com MACD exigiu 46, 118 e 3.040 revisões nas metas de 9, 23 e 36, contra 29, 125 e 23.443 de O3: encurtou a cauda, mas piorou o topo. A F3, que as usa, só entra na recomendação quando elas estiverem disponíveis.

Apoio: S09, S21. Fonte: `estudo4_gzcmd.tex:133`, `:200`, `conclusao.tex:83`.

**P24. Os Estudos 2 e 3 valem como evidência?**

São otimistas por três fontes: indicadores de encerramento por óbito, que constituem vazamento de alvo, partição por linha e limiares escolhidos no teste. Por isso não fundamentam a adoção dos *pipelines* avaliados, e o GZ-CMD não usa os indicadores. No controle, a partição por óbito reduziu o F1 do Random Forest com SMOTE de 0,931 para 0,846, e a retirada dos indicadores pouco o alterou.

Apoio: S11, R14. Fonte: `capitulo3.tex:43`, `conclusao.tex:13`, `estudo2_classificadores.tex:122`.

### Ordenações, fusões e multiplicidade

**P25. Por que recomendar O3 ou as fusões?**

Porque a recomendação é uma hipótese de trabalho condicionada à curva de esforço, e nenhuma ordenação dominou em todas as cargas. O3 reduziu só pontualmente e é a pior na recuperação integral; as fusões reduziram frente ao escore nas metas de 9 e 36, sem superar O3 com intervalo. Por isso se recomenda a probabilidade calibrada, isolada ou fundida, e a escolha pela curva do serviço.

Apoio: S15, S21. Fonte: `conclusao.tex:81-89`, `estudo4_gzcmd.tex:327`.

**P26. As fusões são melhores que O3?**

Não foi demonstrado: nenhum intervalo de F3 ou de F4 frente a O3 ficou inteiramente abaixo de zero. F4 menos O3 teve mediana de 17 revisões a menos na meta de 9, com intervalo de 44 a menos até zero, e de 25 a menos na de 23, com intervalo de 402 a menos a 59 a mais. A vantagem pontual existe, mas não foi demonstrada.

Apoio: S14, R08. Fonte: `tables/editorial/tab_cap8_fusao_robustez.tex:13-14`, `estudo4_gzcmd.tex:225`.

**P27. Quanto do ganho de F4 vem da faixa de 2.522 potenciais pares?**

Dentro da faixa de 2.522 potenciais pares que contém os 36 pares verdadeiros, recalcular as filas reduz a cauda de O3 de 23.443 para 1.330 revisões e a de F4 de 1.387 para 840, enquanto O1 permanece em 2.521. Parte do ganho de F4 vem de priorizar a faixa em que a revisão de origem se concentrou, e por isso a vantagem não se atribui só à ordenação interna.

Apoio: S14, R08. Fonte: `estudo4_gzcmd.tex:229`.

**P28. E a multiplicidade das seis variantes?**

F4 foi escolhida entre seis variantes avaliadas nos mesmos 36 pares verdadeiros, sem correção para multiplicidade e depois do registro. Fixar as variantes antes do cálculo não constitui pré-registro, e o intervalo do *bootstrap* não incorpora a seleção. Na validação, F4 entra como comparação secundária; a primária é O3 contra O1, e a O4, probabilidade recalibrada do ASR, também é secundária.

Apoio: R07. Fonte: `estudo4_gzcmd.tex:123`, `:202`, `:331`, `conclusao.tex:77`, `:99`.

**P29. O que é a fila adaptativa?**

A fila adaptativa aprende com as próprias revisões por amostragem de Thompson, parente do aprendizado ativo, com precedente na construção de coleções de avaliação. Nesta coorte exigiu, em média, 96 revisões para recuperar 23 pares verdadeiros, contra 125 de O3, em exploração posterior ao registro e sem validação. É próximo passo, não recomendação.

Apoio: R07. Fonte: `conclusao.tex:101`, `estudo4_gzcmd.tex:202`.

**P30. O que é a v4 candidata?**

É o registro prospectivo para uma coorte nova: a ordenação pela probabilidade calibrada frente ao escore do OpenRecLink como comparação primária; F4 e a fila com contexto e MACD como secundárias, com F3 quando houver as MACD. Não é política validada.

Apoio: S22. Fonte: `glossario.tex:120`, `capitulo1.tex:41`, `:43`.

### Automação e modelo de linguagem

**P31. Por que o modelo de linguagem não deve aceitar vínculos sozinho?**

Das 25 aceitações abaixo do corte, 21 eram falsos positivos, e o VPP foi de 16,0%, possivelmente subestimado. Os usos subordinados ao revisor humano, como filtro de NONMATCH ou fonte de evidência, são hipóteses exploratórias. O ASR não é o resultado principal: teve registro interno sem critério de sucesso, e a leitura como resultado negativo é posterior.

Apoio: S16, R10. Fonte: `estudo4_gzcmd.tex:157`, `:311-313`.

**P32. O filtro de NONMATCH ajuda?**

O filtro encaminha ao humano os MATCH e UNSURE e descarta os NONMATCH. Com 96 pares ao modelo e 22 revisões, recuperou 13 pares verdadeiros contra 6 de O3, com VPN de 87,8%, mas descartou 9 dos 22 do topo. A vantagem existe só com a carga fixada em 22, de uma réplica, e na unidade de custo da tese o filtro perde para O3.

Apoio: S17, R11. Fonte: `estudo4_gzcmd.tex:271-273`.

**P33. Um modelo maior resolve?**

Em V1, os VPP foram de 16,0%, 19,0%, 19,0% e 18,2% para Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 e Claude Fable 5.1, com cobertura incompleta do Sonnet. Nenhum modelo maior mostrou ganho demonstrável, e o dossiê ampliado não atingiu a utilidade pré-registrada. Os pares foram escolhidos depois dos resultados, e a variação entre réplicas tem amplitude comparável à das diferenças.

Apoio: S17, R10. Fonte: `estudo4_gzcmd.tex:297`, `:301-303`.

**P34. Por que a margem de 25%?**

A margem é escolha prospectiva de relevância, fixada com os dados de desenvolvimento à vista, nos quais as reduções pontuais foram de 75,0% e 43,1%. Não foi testada. Como o intervalo da diferença incluiu zero nas três metas, o tamanho da coorte nova deve vir de cálculo de poder, com a variabilidade observada como ponto de partida.

Apoio: S22. Fonte: `estudo4_gzcmd.tex:331`, `conclusao.tex:97`, `:99`.

### Equidade

**P35. O que dizer do passo 11 e da raça/cor?**

A disparidade no passo 11 é herdada do corte: isolado, ele aceita 15 dos 40 pares verdadeiros do passo. O passo é estrato de processo, não atributo demográfico. A raça/cor não consta do extrato, e a geolocalização é mais grosseira que o bairro, o que impede avaliar eixos de desigualdade em saúde. A análise descreve disparidades técnicas, sem avaliar se são evitáveis e injustas.

Apoio: S18, R12. Fonte: `estudo5_equidade.tex:52`, `:99`, `:103`, `conclusao.tex:31`.

**P36. A equidade da fila recomendada foi examinada?**

Não. O Estudo 5 analisou as aceitações do ASR, da cascata, do XGBoost e do corte da nota final. Como o corte é preservado, a disparidade por passo que ele traz permanece o ponto de partida. O Estudo 4 descreve só a recuperação por passo e por sexo entre os 96 primeiros da fila, sem testes nem intervalos por subgrupo.

Apoio: S18, R12. Fonte: `estudo5_equidade.tex:101`, `conclusao.tex:33`.

**P37. Há paridade entre sexos?**

Não foi demonstrada. Não houve rejeição após Benjamini-Hochberg em sexo ou faixa etária, mas são 71 pares verdadeiros no sexo feminino e 176 no masculino, e o poder não foi calculado. A paridade simultânea de sensibilidade e VPP é factível; a incompatibilidade sob prevalências distintas refere-se ao VPP e ao *equalized odds*.

Apoio: S18, R12. Fonte: `conclusao.tex:33`, `estudo5_equidade.tex:42`.

### Reimplementação do CdR

**P38. O CdR em Python equivale ao protótipo em Java?**

Não. Os subescores e a nota final usados nos estudos são os do Python. A versão corrige erros do protótipo, diverge de Lucena em pontos documentados e acrescenta subescores. A equivalência não foi demonstrada, e o efeito em dados reais não foi medido. As divergências não alteram os estudos seguintes, que tomam a nota final como dada.

Apoio: S10, R13. Fonte: `estudo1_comparador.tex:128`.

**P39. Bastam 500 pares sintéticos?**

O lote não pretende representar os dados: foi construído para exercitar casos de fronteira, em dois sublotes de 250 pares, um ASCII e outro com acentos, com semente fixa e sem registros reais. Por isso o valor de 11,2% não estima a concordância no extrato. A comparação com registros reais não foi feita, e essa é a limitação declarada.

Apoio: S10, R13. Fonte: `estudo1_comparador.tex:31`, `:71`, `:134`.

**P40. Por que a fala não nomeia trabalhos que usaram o protótipo em Java?**

A apresentação trata das diferenças entre implementações. A tese registra que os erros do primeiro tipo podem alterar a nota final e a seleção dos potenciais pares nos estudos que usaram o protótipo, e que o efeito não foi medido em dados reais. Não se atribui a nenhum estudo um efeito específico, e a comparação com registros reais não foi feita.

Apoio: R13. Fonte: `estudo1_comparador.tex:95`, `:134`.

### Validação, poder e próximos passos

**P41. Como a hipótese seria validada em outra coorte?**

Em outra coorte ou período, com ordenação e orçamento fixados antes dos desfechos e referência cega independente. O registro declara, antes do acesso aos dados, a ordenação, o orçamento e o ponto de parada. A generalização para outras localidades, períodos e pares de bases depende de revalidação com referências locais.

Apoio: S22. Fonte: `conclusao.tex:75`, `:97`, `:99`.

**P42. O tamanho da coorte nova está definido?**

Não. O intervalo da diferença entre O3 e O1 incluiu zero nas três metas, então o tamanho deve vir de cálculo de poder, que tome a variabilidade observada como ponto de partida e o número de pares verdadeiros abaixo do corte da nova coorte. O poder dos testes de equidade também não foi calculado.

Apoio: S22. Fonte: `conclusao.tex:99`, `estudo5_equidade.tex:42`.

**P43. O que a tese acrescenta à literatura, e como se compara a Lages e ao CIDACS-RL?**

A tese não reivindica a ordenação por escore com revisão manual, de Sariyar e Borg, 2010, nem o aprendizado ativo ou a revisão assistida por tecnologia. Acrescenta a comparação, posterior ao registro, entre ordenar e automatizar. No levantamento, nenhum estudo consultado teve como foco a região abaixo de um corte único. Lages e colaboradores trataram a faixa intermediária, com outros dados; a tese não comparou resultados com o CIDACS-RL.

Apoio: S23. Fonte: `conclusao.tex:63`, `capitulo2.tex:47`, `:61`.

## Rastreabilidade

Números ditos na fala que não aparecem escritos no quadro correspondente, com a fonte na tese.

| Quadro | Número falado | Fonte |
| :- | :- | :- |
| S02 | 4,8 por 100 mil habitantes, em 2024, terceiro maior entre as unidades da federação | `capitulo3.tex:5` |
| S03 | 85,4%; 14,6%; 61.696 potenciais pares | `capitulo3.tex:9`, `:11` |
| S06 | 75,0% e 43,1%; intervalo de confiança de 95% | `conclusao.tex:17`, `:5` |
| S07 | 2019 a 2022; 2013 a 2022; 93,4%; 40 pares verdadeiros no passo 11; campo PAR com 0, 1 e 2 | `capitulo_dados.tex:19`, `:57`, `:73`; `tables/editorial/tab_blocagem_rendimento.tex:13` |
| S10 | diferença de até 4 pontos | `estudo1_comparador.tex:71` |
| S11 | sete classificadores; F1 médio de 0,880 a 0,918; 67 contra 43; 25 dos 27; limiar de nota final 8 | `estudo2_classificadores.tex:11`; `conclusao.tex:13`; `estudo3_estrategias.tex:23`, `:25` |
| S12 | 300 árvores; 30 atributos; 29 subescores; nota final de 5 a 9; 21.620 potenciais pares; 151 pares verdadeiros; AUC de 0,873 da nota final; mediana de 0,895 | `estudo4_gzcmd.tex:36`, `:40`; `capitulo3.tex:11` |
| S13 | 29 e 51; 75,0%; 43,1%; O4 com 106 e 27; O2 com 11 e 4.163; precisão média 0,283 e 0,130; AUC 0,990 e 0,985; 23.443 e 2.521; cerca de 2.500; 2.000 réplicas; medianas 21, 175 e 579,5 e seus intervalos | `estudo4_gzcmd.tex:194`, `:196`, `:198` |
| S14 | intervalo de 95%; F4 com 38 e 1.203; F3 com 35 e 894; faixa de 2.522 potenciais pares | `estudo4_gzcmd.tex:225`, `:229` |
| S15 | orçamento de 17 revisões: F4 com 16, nota final do CdR com 9, O3 com 6 | `conclusao.tex:87` |
| S16 | 344 ao modelo e 96 ao revisor humano; 25 aceitações; intervalo de 6,4 a 34,7; 9,06 previstos; dez dos 21; 13 não recuperados (1, 6 e 6); 27 perdidos e 4 falsos positivos na confirmação | `estudo4_gzcmd.tex:147`, `:149`, `:311`, `:313`; `conclusao.tex:27` |
| S17 | 121 e 125 revisões; 11 e 17 na confirmação; 13 contra 6 e 9 dos 22 com carga de 22; carga de 18 na outra réplica; VPP de 16,0%, 19,0%, 19,0% e 18,2% | `estudo4_gzcmd.tex:196`, `:271-273`, `:297` |
| S18 | 15 de 40 (37,5%), 95,0% e 94,5%; dois dos 25; 47,5%; 14 encaminhados e limite de 77,5%; 71 e 176 pares verdadeiros | `estudo5_equidade.tex:42`, `:52`, `:54` |
| S19 | 21 falsos positivos | `conclusao.tex:7`, `:27` |
| S20 | 2.522 potenciais pares; 87,0% com intervalo de 82,3% a 90,7%; 971 das 2.000 réplicas | `conclusao.tex:71`, `:75` |
| S21 | razões 1:1 e 1:5; 96 primeiros; mediana de 22, 13 e 18,5 | `conclusao.tex:89` |
| S22 | reduções pontuais de 75,0% e 43,1% | `conclusao.tex:97` |
| S23 | anos 2010 e 2012 das referências | `conclusao.tex:63` |

## Siglas e termos

| Sigla ou termo | Forma usada na tese |
| :- | :- |
| SIM | Sistema de Informações sobre Mortalidade |
| Sinan | Sistema de Informação de Agravos de Notificação |
| Sinan-TB | Sinan para a tuberculose |
| CdR | Comparador de Registros |
| ASR | arranjo sequencial, que preserva o corte e decide abaixo dele pela política de quatro ações |
| LLM | modelo de linguagem de grande porte |
| IC | intervalo de confiança |
| VPP | valor preditivo positivo |
| VPN | valor preditivo negativo |
| AUC | área sob a curva ROC |
| AUC-PR | área sob a curva precisão-sensibilidade |
| MACD | Medidas Contínuas de Diferença de Datas de nascimento |
| O1 | fila pelo escore do OpenRecLink |
| O2 | fila pela nota final do CdR |
| O3 | fila pela probabilidade calibrada fora da amostra do XGBoost |
| O4 | fila pela probabilidade recalibrada de ASR |
| F3 | fusão por posição recíproca de O3, O1 e O3 com MACD |
| F4 | fusão por posição recíproca de O3 e O1 |
| CM | O3 acrescida de atributos de contexto por óbito e das MACD |
| PAR | campo da referência operacional: 0, não par; 1, par verdadeiro pelo corte; 2, par verdadeiro pela revisão manual de origem |

