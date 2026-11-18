# Roteiro da defesa, deck em HTML

O deck em HTML (`text/slides/defesa.html`) tem 28 quadros principais, S01 a S28, e 26 de reserva, R01 a R26, estes fora dos 40 minutos. Este roteiro acompanha esse deck. O roteiro do deck Beamer (`roteiro_defesa.md`) permanece como estava, com a numeração antiga (S01 a S25 e R01 a R14), porque a substituição do Beamer pelo HTML é premissa ainda a confirmar pela autora. Os tempos e as metas de palavras orientam o ensaio e a distribuição da fala; não são duração medida.

Correspondência com o roteiro antigo: S01 a S18 iguais; S19 novo (modelo de linguagem com a nota final oculta); S20 a S23 correspondem aos antigos S19 a S22; S24 novo (explorações posteriores); S25 corresponde ao antigo S23, com item novo; S26 ao antigo S24; S27 novo (código aberto); S28 ao antigo S25. R01 a R14 iguais; R15 a R26 novas.

## Uso do deck

- Avançar: seta para a direita, seta para baixo, `PageDown` ou o botão Próximo. Voltar: seta para a esquerda, seta para cima, `PageUp` ou o botão Anterior. `Home` leva ao S01 e `End` ao último quadro da sequência, a reserva R26, e não ao S28; para os agradecimentos, usar `defesa.html#S28` ou o índice.
- A barra de espaço aciona o botão que estiver em foco: depois de clicar em Anterior ou de fechar o índice ou o glossário abertos pelo botão, ela volta um quadro ou reabre o painel. Na defesa, usar as setas ou `PageDown` e `PageUp`, que funcionam com qualquer foco.
- A sequência das setas percorre os 54 quadros: depois do S28 vem o R01. O contador no rodapé mostra o identificador e a posição, por exemplo S14 e 14 de 54.
- Índice: a tecla `i` (ou o botão Índice) abre a lista dos 54 quadros, principais e de reserva; `Esc` ou `i` fecha.
- Link direto pelo hash da URL: `defesa.html#S14` abre o S14, e `defesa.html#R08` abre a reserva R08. A URL acompanha o quadro mostrado, de modo que o link de qualquer quadro pode ser copiado.
- Reservas: clicar em "reserva R08" no quadro abre a reserva e a registra no histórico do navegador; o link "voltar", no alto da reserva, ou o botão Voltar do navegador retorna ao quadro de origem, sem perder o lugar na sequência.
- Impressão em PDF: `Ctrl+P` e salvar como PDF; sai um quadro por página, em 16:9, com os 54 quadros em ordem (reservas ao final). Botões, índice e links "voltar" não são impressos.
- Lembrete de siglas e termos: a maioria dos quadros traz no rodapé, acima da ressalva, o bloco "Lembrete:" com as siglas e os termos não consagrados usados nele e não definidos no próprio quadro, cada um com uma definição curta; quando o espaço não comporta todos, o último item é "demais termos: tecla g", e alguns quadros, como o S01 e o S28, não trazem o bloco. Os termos que ficam fora dele se consultam pela tecla `g`. O bloco serve à banca mista, de saúde coletiva e de computação, e a fala retoma a sigla com um aposto na primeira menção; o lembrete sai na impressão em PDF.
- Glossário: a tecla `g` (ou o botão Glossário) abre o painel, que começa pela seção "Neste quadro", com todos os termos do quadro atual e suas definições, e segue com a lista alfabética de todos os termos; cada entrada traz links para os quadros em que o termo aparece, e clicar num deles fecha o painel e abre o quadro. `Esc` ou `g` fecha o painel e devolve o foco ao ponto anterior. Com o painel aberto, as setas não mudam de quadro. Na impressão em PDF, o glossário sai como páginas finais, depois da R26, só com a lista alfabética.
- Uso na defesa: ao receber uma pergunta sobre um termo ou uma sigla, `g` mostra a definição sem sair do quadro, e `g` ou `Esc` devolve a tela ao quadro.
- Modo de conferência de estouro: acrescentar `?verificar=1` ao endereço; o resultado fica no atributo `data-estouro` do elemento `html`. Serve para a preparação, não para a defesa.

## Cronometragem

A meta de palavras usa 130 palavras por minuto. A soma dos quadros principais é de 2.400 segundos, ou 40,0 minutos, sem folga: a exposição de 40 minutos é o limite fixado pela autora.

| Bloco | Quadros | Segundos | Minutos |
| :- | :- | -: | -: |
| Abertura | S01 a S06 | 430 | 7,2 |
| Material e medida | S07 a S10 | 320 | 5,3 |
| Resultados | S11 a S20 | 1.000 | 16,7 |
| Fechamento | S21 a S28 | 650 | 10,8 |
| Total | S01 a S28 | 2.400 | 40,0 |

| Quadro | Tema | Segundos | Meta de palavras |
| :- | :- | -: | -: |
| S01 | Capa | 20 | 43 |
| S02 | Problema de saúde pública | 70 | 152 |
| S03 | Corte e *grey area* | 120 | 260 |
| S04 | Objetivo geral | 60 | 130 |
| S05 | Cinco objetivos específicos | 60 | 130 |
| S06 | Resposta antecipada | 100 | 217 |
| S07 | Dados e referência operacional | 100 | 217 |
| S08 | Esquema do pós-processamento | 100 | 217 |
| S09 | A medida | 60 | 130 |
| S10 | As quatro filas | 60 | 130 |
| S11 | Objetivo 1: CdR | 80 | 173 |
| S12 | Objetivo 2: classificadores | 90 | 195 |
| S13 | Objetivo 2: probabilidade calibrada | 90 | 195 |
| S14 | Objetivo 3: curva de esforço | 130 | 282 |
| S15 | Objetivo 3: incerteza | 120 | 260 |
| S16 | Objetivo 3: tabela das filas | 100 | 217 |
| S17 | Objetivo 4: automação como aceitadora | 120 | 260 |
| S18 | Objetivo 4: mesma carga | 110 | 238 |
| S19 | Objetivo 4: LLM sem a nota final | 60 | 130 |
| S20 | Objetivo 5: equidade | 100 | 217 |
| S21 | O que a tese sustenta | 130 | 282 |
| S22 | Limitações | 100 | 217 |
| S23 | Fluxo recomendado | 130 | 282 |
| S24 | Explorações posteriores | 60 | 130 |
| S25 | Critério de refutação e próximos passos | 90 | 195 |
| S26 | Contribuições | 60 | 130 |
| S27 | Código aberto | 60 | 130 |
| S28 | Agradecimentos | 20 | 43 |
| Total | | 2.400 | 5.202 |

## Vocabulário mínimo por bloco

Para o ensaio: as siglas e os termos não consagrados que a banca precisa ter em mente ao entrar em cada bloco, com o quadro em que cada um é apresentado. A lista é gerada de `defesa.html` por `scripts/lembretes_deck.py`, na opção de vocabulário, e não se edita à mão. "Apresentados" são os termos cujo primeiro quadro principal, na ordem S01 a S28, os traz em `data-termos` ou em marcação de definição; "trazidos" são os usados em quadros do bloco e apresentados em bloco anterior. Os termos que o glossário grafa em itálico saem em itálico. As definições estão no glossário (tecla `g`).

[//]: # (VOCABULARIO)

- **Abertura (S01 a S06).** Apresentados no bloco: *linkage* [S01]; vigilância do óbito [S02]; SIM [S02]; Sinan [S02]; Sinan-TB [S02]; evento sentinela [S02]; ponto de corte [S03]; par verdadeiro e não par [S03]; revisão manual [S03]; OpenRecLink [S03]; escore [S03]; potenciais pares [S03]; passo [S03]; blocagem [S03]; vínculo [S03]; *grey area* [S03]; referência operacional [S03]; CdR [S05]; fora da amostra [S05]; *bootstrap* [S05]; partição por óbito [S05]; probabilidade calibrada [S06]; IC [S06]; meta [S06]; vigilância (modo) [S06]; LLM [S06]; *post hoc* [S06]; revisor perfeito [S06]. Trazidos de blocos anteriores: nenhum.
- **Material e medida (S07 a S10).** Apresentados no bloco: nota final [S08]; XGBoost [S08]; orçamento [S08]; curva de esforço [S09]; ASR [S09]; confirmação (modo) [S09]; sensibilidade [S09]; VPP [S09]; O1 [S10]; O3 [S10]; F3 [S10]; fusão por posição recíproca [S10]; MACD [S10]; F4 [S10]; O2 [S10]; O4 [S10]. Trazidos de blocos anteriores: ponto de corte [S03]; par verdadeiro e não par [S03]; revisão manual [S03]; OpenRecLink [S03]; escore [S03]; potenciais pares [S03]; passo [S03]; blocagem [S03]; *grey area* [S03]; referência operacional [S03]; CdR [S05]; fora da amostra [S05]; probabilidade calibrada [S06]; vigilância (modo) [S06]; LLM [S06]; revisor perfeito [S06].
- **Resultados (S11 a S20).** Apresentados no bloco: subescores [S11]; F1 (métrica) [S12]; *Random Forest* [S12]; AUC-PR [S12]; SMOTE [S12]; *pipeline* [S12]; faixa intermediária da nota final [S13]; ROC [S13]; AUC [S13]; precisão média [S13]; filtro [S18]; NONMATCH [S18]; MATCH [S19]; sensibilidade automática [S20]. Trazidos de blocos anteriores: ponto de corte [S03]; par verdadeiro e não par [S03]; OpenRecLink [S03]; escore [S03]; potenciais pares [S03]; passo [S03]; vínculo [S03]; *grey area* [S03]; CdR [S05]; fora da amostra [S05]; partição por óbito [S05]; probabilidade calibrada [S06]; IC [S06]; meta [S06]; vigilância (modo) [S06]; LLM [S06]; *post hoc* [S06]; revisor perfeito [S06]; nota final [S08]; XGBoost [S08]; orçamento [S08]; curva de esforço [S09]; ASR [S09]; confirmação (modo) [S09]; sensibilidade [S09]; VPP [S09]; O1 [S10]; O3 [S10]; F3 [S10]; F4 [S10]; O2 [S10]; O4 [S10].
- **Fechamento (S21 a S28).** Apresentados no bloco: subnotificação [S21]; *labels* [S24]; GZ-CMD [S24]; PyPI [S24]; desempate pelo LLM [S24]; carga humana [S26]; *expected loss* [S27]; política v3 e v3.1 [S27]. Trazidos de blocos anteriores: ponto de corte [S03]; par verdadeiro e não par [S03]; revisão manual [S03]; OpenRecLink [S03]; escore [S03]; potenciais pares [S03]; passo [S03]; vínculo [S03]; referência operacional [S03]; CdR [S05]; fora da amostra [S05]; *bootstrap* [S05]; probabilidade calibrada [S06]; IC [S06]; meta [S06]; vigilância (modo) [S06]; LLM [S06]; *post hoc* [S06]; revisor perfeito [S06]; nota final [S08]; XGBoost [S08]; orçamento [S08]; curva de esforço [S09]; ASR [S09]; sensibilidade [S09]; O1 [S10]; O3 [S10]; F3 [S10]; MACD [S10]; F4 [S10]; O2 [S10]; O4 [S10]; subescores [S11]; MATCH [S19].

[//]: # (/VOCABULARIO)

## Fala-guia por quadro

### S01. Aprendizado de Máquina Aplicado ao Pós-Processamento do *Linkage* Probabilístico de Bases de Dados de Saúde

Tempo: 20 s (meta de 43 palavras).

Defesa de doutorado de Marco Elisio Oliveira Jardim, do IESC-UFRJ, em 2026, sob orientação de Rejane Sobrino Pinheiro. A tese trata do pós-processamento do *linkage* probabilístico, e o objeto é a revisão manual abaixo do ponto de corte do OpenRecLink.

Transição: começa-se pelo problema de saúde pública, a vigilância do óbito por tuberculose.

Perguntas prováveis: nenhuma específica.

### S02. A vigilância do óbito por tuberculose depende do *linkage* (relacionamento de registros) entre óbitos e notificações

Tempo: 70 s (meta de 152 palavras).

O Ministério da Saúde trata o óbito por tuberculose como evento sentinela, por ser desfecho evitável. No estado do Rio de Janeiro, o coeficiente de mortalidade foi de 4,8 por 100 mil habitantes em 2024, o terceiro maior entre as unidades da federação, em dados preliminares. Contar esses óbitos exige ligar registros do Sistema de Informações sobre Mortalidade, o SIM, aos casos do Sistema de Informação de Agravos de Notificação para tuberculose, o Sinan-TB, isto é, descobrir quais registros das duas bases se referem à mesma pessoa. Sem identificador unívoco comum, o *linkage* probabilístico compara nome, nome da mãe, data de nascimento e outros campos e calcula um escore de similaridade. Um falso positivo, o vínculo aceito indevidamente, pode encerrar no Sinan, como óbito, o caso de uma pessoa viva. Um falso negativo faz o óbito de uma pessoa notificada parecer não notificado. O custo dessas consequências não foi medido.

Transição: cabe descrever primeiro como o OpenRecLink decide.

Perguntas prováveis: Pinheiro, coautora dos estudos de subnotificação e de acurácia do *linkage* em tuberculose: o plano não registra arguição específica, e a distinção entre subnotificação e perda de vínculo está em R04, pelo índice. Coeli: a relação dos pares verdadeiros abaixo do corte com a subnotificação (R04, pelo índice).

### S03. O ponto de corte aceita 211 dos 247 pares verdadeiros; os outros 36 só a revisão manual de origem recuperou

Tempo: 120 s (meta de 260 palavras).

O OpenRecLink forma os potenciais pares por blocagem, a etapa que gera candidatos a partir de chaves comuns. No processo de origem, foi aplicado com um ponto de corte por passo, sem limiar inferior de rejeição (reserva R03). Acima do corte, o par é aceito como vínculo; abaixo, segue para a revisão manual. Nesta tese o corte é parte do processo e é preservado: os vínculos que ele aceita não são alterados. Dos 247 pares verdadeiros do extrato, o corte aceitou 211; os outros 36 só foram recuperados pela revisão manual de origem. Em resumo, o corte aceita a grande maioria dos pares verdadeiros, e a revisão manual abaixo dele recupera os demais. Fala-se em par verdadeiro segundo a referência operacional, formada pelo corte e por essa revisão, e não em verdade absoluta. A região abaixo do corte recebe o nome de *grey area* (área cinza) e reúne 61.485 potenciais pares. É nela que a revisão manual atua, e é nela que a tese procura reduzir o esforço para recuperar os 36 pares verdadeiros. O custo da revisão manual não foi medido, e a medida usada é o número de revisões.

Transição: com a região delimitada, resta formular a pergunta que a tese procura responder.

Perguntas prováveis: Camargo Jr.: o que autoriza chamar um par de verdadeiro, por que preservar o corte e como a *grey area* se relaciona com os dois limiares de Fellegi e Sunter (R03; R01, pelo índice).

### S04. Objetivo geral

Tempo: 60 s (meta de 130 palavras).

O objetivo geral tem duas partes. A primeira é desenvolver e avaliar um pós-processamento do *linkage* probabilístico entre o SIM e o Sinan-TB, isto é, etapas aplicadas depois da decisão do OpenRecLink, que reduza a revisão manual necessária para recuperar os pares verdadeiros situados na *grey area*, região abaixo do ponto de corte, preservando os vínculos que o corte aceitou. A segunda é identificar qual componente responde por essa redução. O pós-processamento foi separado em cinco componentes: a preservação do corte; a probabilidade de par verdadeiro, calibrada fora da amostra; a ordem da fila de revisão; a escolha do ponto de parada pela curva de esforço; e a decisão por *expected loss*, o custo esperado de cada ação possível, com revisão assistida por um modelo de linguagem. Separar os componentes permite perguntar qual deles, e não o conjunto, responde pela redução.

Transição: a pergunta se desdobra em cinco objetivos específicos.

Perguntas prováveis: nenhuma específica.

### S05. Cinco objetivos específicos, cinco passos da resposta

Tempo: 60 s (meta de 130 palavras).

Os cinco objetivos específicos encadeiam-se do instrumento à equidade. O primeiro reimplementa em Python, em código aberto, o Comparador de Registros, o CdR, que calcula a similaridade campo a campo e produz os subescores, e classifica suas diferenças frente ao protótipo em Java. O segundo estima a probabilidade de que um potencial par seja verdadeiro e a calibra fora da amostra, nos Estudos 2 e 3 e no Estudo 4. O terceiro compara ordenações da fila pelo número de revisões, com a incerteza estimada por *bootstrap* agrupado por óbito, isto é, reamostrando os óbitos e não os pares. O quarto verifica se a automação reduz a revisão além da ordenação, e a que custo em falsos positivos. O quinto avalia a equidade por estrato de processo e por subgrupo demográfico, em especial no passo 11.

Transição: a resposta vem antes dos detalhes.

Perguntas prováveis: nenhuma específica.

### S06. A redução veio da ordem da fila de revisão, não da automação

Tempo: 100 s (meta de 217 palavras).

Nesta coorte, a redução do esforço de revisão veio da ordem em que os potenciais pares são revisados, e não da automação. Ordenar a fila pela probabilidade calibrada, isto é, estimada por um modelo que não viu os potenciais pares do mesmo óbito, exigiu menos revisões que ordenar pelo escore do OpenRecLink nas duas metas menores, isto é, nos números de pares verdadeiros a recuperar abaixo do corte, em estimativa pontual com intervalo de confiança que não exclui o zero. Só as fusões da probabilidade calibrada com o escore reduziram as revisões frente ao escore, com intervalo de confiança da diferença inteiramente abaixo de zero, nas metas de 9 e de 36 pares verdadeiros; frente à probabilidade isolada, nenhuma fusão teve intervalo da diferença inteiramente abaixo de zero. Essas fusões foram escolhidas entre seis variantes, sem correção. Já o modelo de linguagem de grande porte, na vigilância, aceitou abaixo do corte 4 pares verdadeiros e 21 falsos positivos: só apoia a revisão. Tudo isso é *post hoc*, de coorte única e sob revisor perfeito simulado, cuja revisão sempre acerta e não é acerto observado, e a validar em coorte nova.

Transição: os dados e a referência vêm primeiro.

Perguntas prováveis: Costa: multiplicidade na escolha das fusões (R07, pelo índice). Qualquer membro: o intervalo de cada diferença (R08, pelo índice).

### S07. Dados: três passos de blocagem e uma referência operacional de revisora única

Tempo: 100 s (meta de 217 palavras).

O extrato reúne os potenciais pares do município do Rio de Janeiro, gerados em três passos de blocagem: 6, 9 e 11. São 61.696 potenciais pares, com cerca de um par verdadeiro para cada 249 não pares, e o passo 11 reúne 25 dos 36 pares verdadeiros abaixo do corte. A referência operacional resulta do corte e da revisão manual de origem, feita por revisora única, sem dupla revisão e sem registro da ordem de exame, das fontes e da parada. Que o CdR, o comparador que gera os subescores e a nota final, tenha precedido a revisão é hipótese de trabalho, e a direção e a magnitude do viés de verificação não são identificáveis (reserva R01). A limitação é de delineamento e de documentação disponível, e não do trabalho de revisão de origem. O projeto teve aprovação nos dois comitês de ética, e os dados são pseudonimizados, o que não os torna anônimos (reserva R02).

Transição: com os dados e a referência descritos, falta o esquema que mostra onde cada componente atua.

Perguntas prováveis: Coeli: o erro da referência com revisora única (R01) e o envio de dossiês a interfaces externas de programação (R02). Camargo Jr.: o que autoriza chamar um par de verdadeiro (R01).

### S08. O corte é preservado; o pós-processamento atua abaixo dele

Tempo: 100 s (meta de 217 palavras).

O corte é preservado e o pós-processamento atua abaixo dele; o esquema mostra onde cada componente atua. O OpenRecLink gera os potenciais pares por blocagem e calcula o escore; o ponto de corte por passo, preservado, separa os vínculos aceitos e mantidos da *grey area*. Abaixo do corte, o primeiro componente é o CdR, que devolve subescores e uma nota final, objetivo 1. O segundo é um algoritmo que combina muitas árvores de decisão construídas em sequência, o XGBoost, e converte esses valores em probabilidade calibrada fora da amostra, objetivo 2. Com essa probabilidade ordena-se a fila, e a revisão manual segue a fila até o orçamento, o número de revisões que o serviço pode fazer, objetivo 3. A decisão automática e o modelo de linguagem, objetivo 4, atuam abaixo do corte, e a equidade por estrato de processo, objetivo 5, é avaliada sobre as decisões resultantes, incluídas as do corte. O corte permanece no arranjo sequencial e no fluxo recomendado, e nenhum componente corrige o OpenRecLink: o que se mede é a carga de revisão abaixo dele. A decisão automática está na reserva R09.

Transição: a medida comum de esforço é o tema do quadro seguinte.

Perguntas prováveis: Camargo Jr.: se a tese corrige o OpenRecLink (não: o corte é preservado; R03, pelo índice). Qualquer membro: o que faz a política de decisão (R09).

### S09. A medida: revisões necessárias para recuperar 9, 23 ou 36 pares verdadeiros

Tempo: 60 s (meta de 130 palavras).

A medida é a curva de esforço: para cada fila, quantas revisões são necessárias para recuperar k pares verdadeiros. O revisor é perfeito simulado: toda revisão recupera o par verdadeiro, sem falso positivo humano, o que não é acerto humano observado. As metas são o que o arranjo sequencial, o ASR, recuperou abaixo do corte; ele preserva o corte e, abaixo dele, decide pela política de quatro ações com o modelo de linguagem. Na vigilância, que prioriza a sensibilidade, a proporção de pares verdadeiros recuperados, o ASR recuperou 23. Na confirmação, que prioriza o valor preditivo positivo (VPP), a proporção de vínculos aceitos que são pares verdadeiros, recuperou 9. A terceira meta, 36, reúne todos os pares verdadeiros abaixo do corte.

Transição: as quatro filas que serão medidas vêm a seguir.

Perguntas prováveis: Coeli: revisões em horas (R05, pelo índice). Bastos: a escolha da métrica (R05, pelo índice).

### S10. Quatro filas ordenam os mesmos potenciais pares abaixo do corte

Tempo: 60 s (meta de 130 palavras).

Quatro filas ordenam os mesmos potenciais pares abaixo do corte. A O1 ordena pelo escore do OpenRecLink. A O3 ordena pela probabilidade calibrada fora da amostra, estimada por XGBoost. As outras duas são fusões por posição recíproca: em cada fila, o potencial par recebe o inverso da soma de uma constante com a sua posição, e a fusão soma esses inversos. A F4, a fusão das filas pela probabilidade e pelo escore, soma os inversos de O3 e O1, e a F3, a mesma fusão acrescida da fila pela probabilidade com as Medidas Contínuas de Diferença de Datas, as MACD, soma ainda os inversos dessa terceira fila. Na curva de esforço aparecem também a O2, a fila pela nota final do CdR, o comparador que gera os subescores, e a O4, a fila pela probabilidade recalibrada do ASR, como comparações.

Transição: os resultados seguem a ordem dos objetivos, a começar pelo instrumento.

Perguntas prováveis: Costa: as variantes de fusão e a multiplicidade (R07, pelo índice).

### S11. O CdR em Python é aberto e não equivale ao protótipo em Java

Tempo: 80 s (meta de 173 palavras).

Primeiro objetivo: o CdR, o comparador que gera os subescores e a nota final, de Lucena (2013), foi reimplementado em Python, em código aberto. Em um lote sintético de 500 casos de fronteira, sem registros reais, a nota final foi idêntica em 11,2% dos pares. Esse valor soma diferenças de critério e de arredondamento e não estima a concordância no extrato. As diferenças são de três tipos: erros do protótipo em Java, corrigidos no Python; divergências de critério em relação a Lucena; e melhorias do Python, como os subescores de município e de endereço, estes derivados de critérios já acrescentados à versão em Java. A equivalência não foi demonstrada, e o efeito em dados reais não foi medido (reservas R13 e R24).

Transição: passa-se ao segundo objetivo, começando pela comparação de classificadores.

Perguntas prováveis: Bastos: a suficiência dos 500 casos sintéticos e a comparação com o protótipo em Java, incluindo trabalhos que o usaram (R13). Qualquer membro: quais atributos pesam no modelo (R24).

### S12. A ordem dos classificadores depende da métrica, e as primeiras estimativas eram otimistas

Tempo: 90 s (meta de 195 palavras).

Segundo objetivo, na parte dos classificadores. Na comparação inicial de classificadores, no limiar de 0,5, o maior F1 foi do Random Forest (floresta aleatória), 0,910; o F1 é a média harmônica entre a sensibilidade e o VPP, a proporção de vínculos que são pares verdadeiros. Pela área sob a curva precisão-sensibilidade, a AUC-PR, a regressão logística o supera só pontualmente, com intervalo de confiança da diferença que inclui zero: a ordem depende da métrica. As primeiras estimativas eram otimistas. Com partição por óbito, o F1 do Random Forest com SMOTE, sobreamostragem sintética da classe minoritária, cai de 0,931 para 0,846. No Estudo 3, a vantagem sobre a nota final reproduzia, quase inteiramente, aceitações do corte. A queda foi medida no Random Forest com SMOTE, e os dois estudos não fundamentam a adoção dos *pipelines* (fluxos de processamento) avaliados (reservas R14 e R25).

Transição: a pergunta seguinte é como obter uma probabilidade confiável.

Perguntas prováveis: Bastos: vazamento, partição por óbito e escolha da métrica (R14 e R25).

### S13. A probabilidade calibrada fora da amostra separa os pares na faixa intermediária da nota final

Tempo: 90 s (meta de 195 palavras).

Ainda no segundo objetivo, a probabilidade foi estimada por um XGBoost com 300 árvores sobre 30 atributos: a nota final e os 29 subescores do CdR, o comparador que os calcula. Fora da amostra significa que cada probabilidade vem de um modelo treinado sem os potenciais pares do mesmo óbito; a calibração ajusta os valores do modelo à frequência observada de pares verdadeiros. A avaliação deste quadro cobre a faixa intermediária da nota final do CdR, entre 5 e 9, distinta da *grey area*, que é definida pelo corte do OpenRecLink. Nela, a probabilidade calibrada teve área sob a curva ROC, a AUC, que mede a capacidade de separar pares verdadeiros de não pares, de 0,995, e precisão média de 0,862, contra 0,216 da nota final. A avaliação é fora da amostra, mas na mesma coorte, e não equivale a validação externa (reservas R06 e R22).

Transição: com a probabilidade em mãos, o terceiro objetivo compara as filas.

Perguntas prováveis: Costa: por que XGBoost e não outro estimador (R06). Bastos: a partição por óbito (R14, pelo índice). Qualquer membro: a subamostragem dos negativos no treinamento (R22).

### S14. A ordem da fila muda o número de revisões: 125 contra 501 para 23 pares verdadeiros, estimativa pontual

Tempo: 130 s (meta de 282 palavras).

Este quadro e os dois seguintes respondem ao terceiro objetivo. A figura mostra, para cada fila abaixo do corte, quantos pares verdadeiros se recuperam conforme aumentam as revisões humanas; quanto mais à esquerda a curva atinge uma meta, menos revisões a fila exige. O eixo está em escala logarítmica e termina em 2.522 revisões. Para os 23 pares verdadeiros da meta da vigilância, a probabilidade calibrada, O3, exigiu 125 revisões, contra 501 do escore do OpenRecLink, O1. Os dois pontos do ASR, o arranjo que preserva o corte e decide abaixo dele, em 96 e 17 revisões, trazem anotados os falsos positivos e os pares avaliados pelo modelo de linguagem. A vantagem de O3 está no topo da fila, e não na AUC: a precisão média, que valoriza os pares verdadeiros no topo, é maior em O3 do que em O1 abaixo do corte. A legenda traz ainda a O2, a fila pela nota final do CdR, e a O4, a fila pela probabilidade recalibrada do ASR, como comparações. Na recuperação integral a ordem se inverte. Tudo isso é estimativa pontual, *post hoc*, sem intervalo de confiança que exclua o zero, sob revisor perfeito simulado, aquele cuja revisão sempre acerta (reserva R08).

Transição: a incerteza dessas diferenças vem a seguir.

Perguntas prováveis: Costa: o ganho de O3 na vigilância depende do estimador (R06, pelo índice). Qualquer membro: a tabela completa do *bootstrap* (R08).

### S15. Com IC, só as fusões com o escore reduzem as revisões frente a O1, e nenhuma supera O3

Tempo: 120 s (meta de 260 palavras).

A figura mostra a incerteza das diferenças, estimada por *bootstrap* agrupado por óbito, isto é, sorteando os óbitos com reposição. Cada painel é uma meta, 9, 23 e 36 pares verdadeiros, e mostra a mediana e o intervalo de confiança de 95% da diferença de revisões de O3, da F4, a fusão das filas pela probabilidade e pelo escore, e da F3, a mesma fusão acrescida da fila pela probabilidade com as MACD, as medidas contínuas de diferença de datas, frente a O1; valor negativo significa menos revisões que O1, e o marcador cheio indica intervalo inteiramente abaixo de zero. F3 e F4 têm marcador cheio nas metas de 9 e de 36: F4 exigiu, em mediana, 38 revisões a menos na meta de 9 e 1.203 a menos na de 36, e F3, 35 e 894 a menos. Na meta de 23 nenhum intervalo ficou abaixo de zero, e O3 isolada não tem marcador cheio em nenhuma meta. Frente a O3, nenhuma fusão teve intervalo inteiramente abaixo de zero. F4 foi escolhida entre seis variantes, sem correção para multiplicidade, e o achado é *post hoc* (reservas R07 e R08).

Transição: os valores pontuais de cada fila, lado a lado, mostram por que nenhuma ordenação domina.

Perguntas prováveis: Costa: multiplicidade na escolha de F4 (R07). Qualquer membro: a tabela completa do *bootstrap* (R08).

### S16. Nenhuma ordenação domina em todas as cargas

Tempo: 100 s (meta de 217 palavras).

Lido por colunas, o quadro mostra que nenhuma fila é a melhor em todas as cargas. Com 96 revisões, a O1, a fila pelo escore do OpenRecLink, recupera 19 pares verdadeiros; a O3, a fila pela probabilidade calibrada, 22; a F4, a fusão da probabilidade com o escore, 18; e a F3, a mesma fusão acrescida da fila pela probabilidade com as MACD, 25. Nas revisões necessárias, F4 é a menor em 9 e em 36, com 10 e 1.387, e F3 é a menor em 23, com 64. O3 vem em terceiro na meta de 23, com 125, e é a pior na recuperação integral, com 23.443; O1 é a mais lenta nas metas menores, mas supera O3 na integral, com 2.521. As diferenças são valores pontuais, *post hoc*, e as fusões foram escolhidas entre variantes: na meta de 23 nenhuma fusão teve intervalo abaixo de zero frente a O1, e frente a O3, em nenhuma meta. Por isso a fila se escolhe pela curva de esforço, conforme o orçamento humano, em vez de fixar uma só (reservas R08, R20 e R23).

Transição: até aqui a automação ficou de fora, e o quarto objetivo a examina.

Perguntas prováveis: Qualquer membro: as curvas de esforço completas (R20), as filas com contexto por óbito (R23) e os intervalos (R08).

### S17. Na vigilância, abaixo do corte, a automação aceitou 4 pares verdadeiros e 21 falsos positivos

Tempo: 120 s (meta de 260 palavras).

O quarto objetivo verifica se a automação reduz a revisão além da ordenação, e a que custo em falsos positivos. No ASR, o arranjo que preserva o corte e decide abaixo dele, a política escolhe entre aceitar, rejeitar, enviar à revisão por modelo de linguagem ou ao revisor humano a ação de menor *expected loss*. O modelo de linguagem é o Claude Haiku 5.5, que revisa o dossiê de cada potencial par, um resumo estruturado e pseudonimizado. No modo de vigilância, que prioriza a sensibilidade, o ASR enviou 344 potenciais pares ao modelo e 96 ao revisor humano. Todas as 25 aceitações abaixo do corte vieram do modelo: 4 eram pares verdadeiros e 21, falsos positivos, um VPP, a proporção das aceitações que são pares verdadeiros, de 16,0%, com intervalo de confiança de 95% de 6,4 a 34,7. Dez dos 21 estão em região sem revisão documentada, e, se algum for par verdadeiro, o VPP está subestimado. Nas condições avaliadas, o modelo não deve aceitar vínculos sozinho. O ASR teve registro interno anterior à avaliação, sem critério de sucesso, e a leitura como resultado negativo é posterior; os números supõem revisor perfeito simulado (reservas R10 e R21).

Transição: a comparação com a ordenação, na mesma carga de revisão humana, vem a seguir.

Perguntas prováveis: Qualquer membro: modelos maiores e dossiê (R10); por que o arranjo sequencial e não as cascatas (R21); a política de quatro ações (R09, pelo índice). Costa: custo computacional (sem reserva; na vigilância, o ASR enviou 344 potenciais pares ao modelo, com 755 chamadas e custo estimado de US$ 0,94, e, na confirmação, 41, a US$ 0,11, pela tarifa aplicada ao consumo registrado, sem custo de pessoal nem processamento local; o QW1 rodou na plataforma Modal, em GPU NVIDIA L4).

### S18. Na vigilância, a automação recupera no máximo um par verdadeiro a mais que a fila ordenada

Tempo: 110 s (meta de 238 palavras).

Na mesma carga, com 96 revisões humanas, o ASR, o arranjo que preserva o corte e decide abaixo dele, recupera 23 pares verdadeiros, e a O3, a fila pela probabilidade calibrada, recupera 22. A diferença é de no máximo um par verdadeiro, e o ASR acrescenta 21 falsos positivos; um par de diferença não sustenta superioridade. No modo de confirmação, que prioriza o VPP, a ordenação pela nota final do CdR, o comparador que gera os subescores, recupera os mesmos 9 pares verdadeiros do ASR com 11 revisões, contra 17 (reserva R05). Quatro modelos de linguagem foram comparados como revisores, sem ganho demonstrável pelo tamanho, o que está na reserva R10. O filtro de NONMATCH, a decisão de não par que descarta antes do revisor humano os pares assim classificados, recupera mais pares que a O3 só com a carga humana fixada e descarta pares verdadeiros do topo, o que está na reserva R11. O filtro e os usos subordinados ao revisor humano são hipóteses exploratórias, *post hoc* e sob revisor perfeito simulado.

Transição: resta a medição do modelo de linguagem sem a nota final, e depois o quinto objetivo, a equidade.

Perguntas prováveis: Coeli: revisões em horas e a confirmação (R05). Qualquer membro: modelos maiores (R10) e o filtro de NONMATCH (R11).

### S19. Com a nota final oculta, o LLM deu NONMATCH a 16 dos 36 pares verdadeiros abaixo do corte, o que limita seu uso como filtro

Tempo: 60 s (meta de 130 palavras).

Exploração posterior, de uma realização: com a nota final oculta, o Claude Haiku 5.5 revisou uma amostra estratificada de 717 potenciais pares (reserva R15). Abaixo do corte, deu NONMATCH a 16 dos 36 pares verdadeiros, 44,4%, com intervalo de confiança de 95% de 29,5% a 60,4%, o que limita seu uso como filtro (reserva R11). Deu MATCH a 2 de 470 não pares, mas a dezenas de não pares no topo da ordem pela nota final. Se revisasse todos os potenciais pares abaixo do corte, o VPP, a proporção de aceitos que seriam pares verdadeiros, seria projetado em 2,9%. Esse não é o VPP das aceitações da política, de 16,0%, com nota final visível e 25 aceitos: a comparação é só de ordem de grandeza e não altera os 16,0%.

Transição: resta o quinto objetivo, a equidade.

Perguntas prováveis: Qualquer membro: a medição completa, com a amostra e os intervalos (R15); o filtro de NONMATCH (R11).

### S20. A perda de sensibilidade concentra-se no passo 11 e é herdada do corte

Tempo: 100 s (meta de 217 palavras).

O quinto objetivo avalia se o desempenho se distribui igualmente entre estratos. Na vigilância, o ASR, o arranjo que preserva o corte e decide abaixo dele, teve sensibilidade automática, isto é, sem contar a revisão humana, de 42,5% no passo 11, contra 95,0% no passo 6 e 96,1% no passo 9. A disparidade vem do corte: isolado, ele aceita 15 dos 40 pares verdadeiros do passo 11. O ASR preserva o corte e pouco altera essa disparidade, de modo que a perda de sensibilidade já existe no corte e é herdada pelo arranjo. O passo é um estrato do processo, não um atributo demográfico. Entre sexo e faixa etária, a correção de Benjamini-Hochberg, que controla a proporção esperada de falsas descobertas, não deixou nenhuma rejeição, o que não demonstra paridade, porque os denominadores limitam a detecção e o poder não foi calculado. A raça/cor não consta do extrato. A equidade da fila recomendada não foi examinada (reserva R12).

Transição: fechados os resultados, cabe separar o que a tese sustenta do que não sustenta.

Perguntas prováveis: Qualquer membro: o passo 11, a raça/cor e a paridade por sexo (R12).

### S21. O que a tese sustenta e o que não sustenta

Tempo: 130 s (meta de 282 palavras).

A tese sustenta que a F4, a fusão das filas pela probabilidade e pelo escore, e a F3, a mesma fusão acrescida da fila pela probabilidade com as MACD, as medidas contínuas de diferença de datas, reduziram as revisões frente a O1, a fila pelo escore do OpenRecLink, nas metas de 9 e de 36 pares verdadeiros, com intervalo de confiança inteiramente abaixo de zero; ambas foram escolhidas entre seis variantes, sem correção para multiplicidade. Sustenta que a ordem da fila altera o número de revisões e que a disparidade de sensibilidade se concentra no passo 11. Sustenta ainda que, na vigilância, a automação recuperou no máximo um par verdadeiro a mais que O3, a fila pela probabilidade calibrada, com 21 falsos positivos. A tese não sustenta que a probabilidade calibrada isolada reduza as revisões, porque os intervalos incluem o zero. Não sustenta que F3 ou F4 superem O3, nem que alguma ordenação domine em todas as cargas. Não sustenta que o modelo de linguagem aceite vínculos sozinho ou reduza a carga. Não sustenta que a redução valha para outra coorte ou diminua a subnotificação, nem que haja paridade entre subgrupos demográficos. As estimativas dos Estudos 2 e 3 são otimistas. Todo o conjunto é *post hoc*, de coorte única e sob revisor perfeito simulado (reservas R07 e R14).

Transição: as limitações explicam por que esses limites existem.

Perguntas prováveis: Costa: multiplicidade (R07). Bastos: os Estudos 2 e 3 como evidência (R14).

### S22. Três limitações condicionam a leitura

Tempo: 100 s (meta de 217 palavras).

A primeira limitação é a referência operacional. Ela resulta do corte e da revisão manual de origem, feita por revisora única, sem classificação independente e sem registro da ordem de exame, das fontes e da parada. Os 36 pares verdadeiros abaixo do corte estão entre os 2.522 potenciais pares de maior escore, padrão compatível com revisão restrita a essa faixa, o que favoreceria O1; se a revisão incorporou as evidências do CdR, o comparador que gera os subescores, o favorecimento seria da O2, a fila pela nota final, da O3, a fila pela probabilidade calibrada, e da O4, a fila pela probabilidade recalibrada do ASR, o arranjo que preserva o corte. A direção e a magnitude do viés não são identificáveis (reserva R01). A segunda é a coorte única, com 247 pares verdadeiros e 36 abaixo do corte, o que deixa os intervalos largos (reserva R08). A terceira é que o revisor humano não foi observado: todas as contagens supõem revisor perfeito simulado, e a análise é *post hoc*. Essas limitações são de delineamento e de documentação disponível, e não do trabalho de revisão de origem.

Transição: a recomendação para o serviço vem a seguir.

Perguntas prováveis: Coeli: o erro da referência com revisora única (R01). Qualquer membro: os intervalos (R08).

### S23. Fluxo recomendado, a validar em coorte nova

Tempo: 130 s (meta de 282 palavras).

A recomendação, a validar em coorte nova com referência cega, é preservar o corte do OpenRecLink, que aceita os potenciais pares de escore no corte ou acima. Abaixo dele, o Comparador de Registros, o CdR, calcula subescores e nota final, e o XGBoost os converte em probabilidade calibrada fora da amostra. A fila de revisão manual segue essa probabilidade, isolada, em O3, a fila pela probabilidade calibrada, ou fundida ao escore, em F4, a fusão das filas pela probabilidade e pelo escore, ou, quando houver as MACD, as medidas contínuas de diferença de datas, em F3, a mesma fusão acrescida da fila pela probabilidade com essas medidas. Nenhuma ordenação dominou em todas as cargas, e por isso a escolha se faz pela curva de esforço: o serviço traçaria a curva de cada fila sobre uma amostra revisada e adotaria a que recuperasse mais pares verdadeiros no orçamento disponível. A revisão manual vai até o orçamento humano, e a curva desta coorte não fixa o ponto de parada em outra. Os potenciais pares abaixo da parada não foram examinados e devem ser registrados como não examinados, não como não pares, e a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado (reserva R04). O modelo de linguagem só apoia a revisão: o MATCH segue ao revisor humano. A recomendação é hipótese de trabalho, e nenhuma etapa foi validada em outra coorte.

Transição: antes do critério de refutação, cabe mostrar que as explorações feitas depois dos resultados principais não alteram essa recomendação.

Perguntas prováveis: Coeli: a relação com a subnotificação (R04), as revisões em horas (R05, pelo índice) e os dados do serviço enviados a interfaces externas (R02, pelo índice). Camargo Jr.: se o fluxo substitui o OpenRecLink (não: o corte é preservado). Pinheiro: o plano não registra arguição específica; o fluxo é o tema do quadro.

### S24. Explorações posteriores não alteram a recomendação: contra O1, ordenação sem *labels*, nenhuma estratégia testada a superou de forma consistente

Tempo: 60 s (meta de 130 palavras).

As explorações posteriores usaram um só extrato, revisor perfeito simulado e a versão 0.7.0 do pacote, experimental e que não foi publicada no PyPI (reserva R19). As referências foram O1, a fila pelo escore do OpenRecLink, e O2, a fila pela nota final, ambas sem *labels* (a classificação de referência de cada potencial par), e O1 é a mais forte delas. Contra O1, nem a fila que reaprende com *labels* verificados nem o desempate pelo modelo de linguagem superaram a ordenação de forma consistente; o desempate entra como hipótese. O3 e F4 dependem de *labels* e não são comparáveis a essas explorações, que orientam o desenho da coorte nova, não alimentam os resultados do capítulo e têm diferenças de um a três pares em 36, dentro do ruído. Nada disso altera a recomendação.

Transição: resta a refutação.

Perguntas prováveis: Qualquer membro: se a evidência dessas explorações é forte (R18); a versão do pacote (R19); o desempate pelo modelo de linguagem (R16) e a comparação com O1 (R17).

### S25. A hipótese pode ser refutada: o critério e os próximos passos estão declarados

Tempo: 90 s (meta de 195 palavras).

A hipótese central exige outra coorte, com referência cega, e ordenação e orçamento fixados antes dos desfechos. Seria refutada se a O3, a fila pela probabilidade calibrada, não atingisse a recuperação do ASR, o arranjo que preserva o corte, com pelo menos 25% menos revisões que a O1, a fila pelo escore do OpenRecLink, pelo limite inferior do intervalo de 95% entre partições predefinidas, em cada meta, 23 na vigilância e 9 na confirmação. A margem foi escolhida com os dados à vista e não foi testada. O critério para a interpretação de que as quatro ações não acrescentam ganho material está na reserva R09. Como o intervalo da diferença entre O3 e O1 incluiu zero nas três metas, o tamanho da coorte nova deve vir de um cálculo de poder. Os próximos passos são a revisão humana observada, com dupla revisão, e a medida do tempo por revisão com e sem o modelo de linguagem, cujo efeito não foi estimado. Nenhuma dessas avaliações foi realizada.

Transição: seguem as contribuições.

Perguntas prováveis: Qualquer membro: a margem de 25% e as comparações secundárias (R07, pelo índice); o critério para as quatro ações (R09).

### S26. Contribuições: a medida, a decomposição entre ordenação e automação, e o código aberto

Tempo: 60 s (meta de 130 palavras).

A primeira contribuição é a medida: a tese separa os 211 pares verdadeiros aceitos pelo corte dos 36 recuperados na revisão manual abaixo dele, e usa a curva de esforço. A segunda é a decomposição, na mesma carga humana, entre ordenar a fila e automatizar a decisão. Ordenar por escore e revisar manualmente já estava descrito por Sariyar e Borg (2010), e a tese não reivindica isso: a comparação é posterior ao registro (reserva R26). A terceira é o código aberto, o CdR, o comparador que gera os subescores e a nota final, em Python, e o pacote GZ-CMD, que implementa a política de decisão, sem exercício independente de auditoria das justificativas.

Transição: o código aberto merece um quadro próprio.

Perguntas prováveis: Qualquer membro: o que a tese acrescenta à literatura (R26).

### S27. O código está aberto: o CdR em Python e o pacote gzcmd-record-*linkage*

Tempo: 60 s (meta de 130 palavras).

O CdR em Python é de código aberto, sob licença GPL-3.0. O pacote gzcmd-record-linkage, ou GZ-CMD, está na versão 0.6.1 no PyPI, publicada em 14 de novembro de 2026, sob licença MIT. Reúne a política de decisão por *expected loss* (reserva R09), a probabilidade calibrada fora da amostra, as fusões de filas com *bootstrap* e o arranjo sequencial, registra cada decisão e documenta o que deixa de recuperar, e mantém a política v3 como padrão, para reproduzir o perfil histórico. A instalação não fornece os modelos ajustados nem reproduz, por si só, os resultados da tese, e não houve exercício independente de auditoria das justificativas. A versão 0.7.0 é experimental e não foi publicada no PyPI. Versões e módulos estão na reserva R19.

Transição: encerra-se com os agradecimentos.

Perguntas prováveis: Qualquer membro: se a instalação reproduz a tese e quais versões existem (R19); a política de quatro ações (R09).

### S28. Agradecimentos

Tempo: 20 s (meta de 43 palavras).

Os agradecimentos vão à orientadora, Rejane Sobrino Pinheiro; ao LinkDataPop, pelo extrato e pela revisão manual de origem; aos criadores do OpenRecLink; à banca; e à CAPES. A palavra está com a banca para a discussão.

Transição: abre-se a arguição.

Perguntas prováveis: nenhuma específica; segue a discussão.

## Reserva

Respostas curtas para quando um quadro de reserva for chamado, de duas a quatro frases. "Origem" indica os quadros principais que remetem a ela por link; as demais se chegam pelo índice (tecla `i`) ou pelo hash.

### R01. Reserva: sem suposições sobre a revisão de origem, o viés de verificação da referência não é identificável

Pergunta: O que garante que a referência não favorece uma das filas?

Origem: S07 e S22.

A referência resulta do corte e da revisão manual de origem, feita por revisora única, sem classificação independente nem registro da ordem de exame, das fontes e da parada. Os 36 pares verdadeiros abaixo do corte estão entre os 2.522 potenciais pares de maior escore, padrão compatível com revisão restrita a essa faixa, o que favoreceria a O1, a fila pelo escore do OpenRecLink; se a revisão usou as evidências do CdR, o comparador que gera os subescores, o favorecimento seria da O2, a fila pela nota final, da O3, a fila pela probabilidade calibrada, e da O4, a fila pela probabilidade recalibrada do ASR, o arranjo que preserva o corte. Sem suposições, a direção do viés não é identificável, e a margem de 25% de O3 frente a O1 na meta de 23 permanece, pela mediana, em todos os cenários de erro da referência na vigilância, embora um cenário cruze zero no percentil 97,5. A limitação é do delineamento e da documentação disponível, e não do trabalho de revisão.

### R02. Reserva: o extrato é pseudonimizado, não anônimo, e parte do processamento ocorreu fora da instituição

Pergunta: Dados de saúde foram enviados a interfaces externas de programação (APIs)?

Origem: S07.

Sim. O extrato é pseudonimizado, não anônimo: não tem nomes, nome da mãe nem endereço nominal, mas tem quase-identificadores, e os pareceres são do Comitê de Ética em Pesquisa do IESC-UFRJ e da Secretaria Municipal de Saúde do Rio de Janeiro. O extrato, com datas completas e identificadores internos, foi à plataforma Modal nos experimentos QW1, a comparação de classificadores tabulares, e os dossiês, sem datas completas nem identificadores internos, mas com o pair_id, foram a APIs externas, inclusive nas explorações posteriores, em desacordo com a recomendação de execução local. O pair_id é um SHA-256 sem sal, recalculável por quem detém os identificadores, e não garante anonimização; a aplicabilidade da Lei Geral de Proteção de Dados a essa transferência não foi analisada nos pareceres, e para dados do serviço a recomendação é execução local.

### R03. Reserva: o OpenRecLink usou um corte por passo, não os dois limiares de Fellegi e Sunter, e o corte é preservado

Pergunta: Como a *grey area* se relaciona com os dois limiares de Fellegi e Sunter, e por que preservar o corte?

Origem: S03.

No modelo de Fellegi e Sunter, dois limiares dividem os escores: acima do superior aceita-se, abaixo do inferior descarta-se, e entre eles há uma região sem decisão automática. No processo de origem, o OpenRecLink foi aplicado com um corte por passo, sem limiar inferior, de modo que a *grey area* reúne todos os 61.485 potenciais pares abaixo do corte. O corte é preservado por integrar o processo que produziu a referência, sem presumir superioridade sobre estratégia determinística. A faixa intermediária da nota final, calculada pelo CdR, o comparador que gera os subescores, vai de 5 a menos de 9 e é outra região, com 21.620 potenciais pares e 151 pares verdadeiros.

### R04. Reserva: pares não examinados não equivalem a óbitos não notificados, e o extrato não estima subnotificação

Pergunta: A redução de revisões diminui a subnotificação?

Origem: S23.

Não, e a tese não estimou efeito sobre a subnotificação. Subnotificação é a ausência, no Sinan, do caso de pessoa cujo óbito consta no SIM, e perda de vínculo é erro do próprio *linkage*, que faz caso notificado parecer subnotificado. Abaixo do ponto de parada os potenciais pares ficam registrados como não examinados, não como não pares, e a ausência de vínculo por esgotamento do orçamento não equivale a óbito não notificado. Os 247 pares verdadeiros correspondem a 204 óbitos distintos, e o extrato, que reúne só três passos probabilísticos, não permite estimar subnotificação pela razão entre óbitos e declarações.

### R05. Reserva: a unidade de esforço é a revisão; as horas são cenário, e na confirmação a ordenação pela nota final iguala ASR

Pergunta: Quantas horas a ordenação economiza?

Origem: S18.

A tese não mede horas economizadas: a unidade de esforço é a revisão, e as horas são cenário, de 20 minutos por par, hipótese de planejamento sem equivalência medida entre revisão por modelo de linguagem e humana. Nesse cenário, as 96 revisões da vigilância equivalem a 32 horas, e as 17 da confirmação, a 6 horas. Na confirmação, com 17 revisões, a O2, a fila pela nota final do CdR, o comparador que gera os subescores, recupera 9 pares verdadeiros, a F4, a fusão da probabilidade com o escore, 16, e a O3, a fila pela probabilidade calibrada, 6, e a O2 chega aos 9 com 11 revisões, contra 17 do ASR, o arranjo que preserva o corte e decide abaixo dele. A tese não comparou os cenários ao volume de trabalho da vigilância do óbito no município.

### R06. Reserva: o XGBoost veio do comparativo QW1, e o ganho de O3 na vigilância depende do estimador

Pergunta: Por que XGBoost?

Origem: S13.

O XGBoost veio do comparativo QW1, a comparação de classificadores tabulares na mesma base, em que teve F1 médio de 0,924 ± 0,030, contra 0,913 ± 0,029 do Random Forest, em validação cruzada 5 × 5. Isso informou a escolha, sem estabelecer superioridade da família, e as métricas fora da amostra não incorporam essa seleção. Com o Random Forest no lugar, com os hiperparâmetros do NB01, sem ajuste, o ganho pontual da O3, a fila pela probabilidade calibrada, na vigilância não se manteve, com 498 revisões contra 501 da O1, a fila pelo escore do OpenRecLink, enquanto na confirmação se manteve, com 32 contra 51.

### R07. Reserva: F4 foi escolhida entre seis variantes, sem correção para multiplicidade, em análise *post hoc*

Pergunta: F4 não é fruto de uma escolha entre muitas variantes?

Origem: S15 e S21.

É: a F4, a fusão da probabilidade com o escore, foi escolhida entre seis variantes, F1 a F4 e duas filas por amostragem de Thompson, todas avaliadas nos mesmos 36 pares verdadeiros, sem correção para multiplicidade e em análise *post hoc*. Fixar as variantes antes do cálculo não constitui pré-registro, o intervalo do *bootstrap* não incorpora a seleção, e a verificação da referência pode favorecer a O1, a fila pelo escore do OpenRecLink, na cauda. Na validação, F4 e CM, a fila que acrescenta à probabilidade atributos de contexto por óbito e as MACD, as medidas de diferença entre datas de nascimento, entram como comparações secundárias, com ordenação, orçamento e parada declarados antes do acesso aos dados.

### R08. Reserva: *bootstrap* agrupado por óbito, com 2.000 réplicas, das diferenças de revisões entre filas

Pergunta: Como foi calculada a incerteza das diferenças entre filas?

Origem: S14, S15, S16 e S22.

O *bootstrap* reamostra os óbitos, com 2.000 réplicas, e reordena as filas sem reajustar o modelo, de modo que a reamostragem é condicional às probabilidades fora da amostra e à referência fixas. A diferença entre a O3, a fila pela probabilidade calibrada, e a O1, a fila pelo escore do OpenRecLink, inclui zero nas três metas; a das fusões, a F4, a fusão da probabilidade com o escore, e a F3, a mesma fusão acrescida da fila pela probabilidade com as MACD, as medidas contínuas de diferença de datas, frente à O1 fica inteiramente abaixo de zero nas metas de 9 e de 36, e nenhuma fusão fica inteiramente abaixo de zero frente à O3. Em 971 das 2.000 réplicas restaram menos de 36 pares verdadeiros, e as metas passam a 25%, 64% e 100% do total, o que muda o estimando.

### R09. Reserva: a política de expected loss escolhe, para cada potencial par, entre quatro ações

Pergunta: O que faz a política das quatro ações?

Origem: S08, S25 e S27.

Para cada potencial par, a política calcula a *expected loss*, o custo esperado, de quatro ações: aceitar como vínculo, rejeitar, enviar à revisão humana e enviar à revisão por modelo de linguagem, e escolhe a de menor valor. Guardas fixas rejeitam nota final abaixo de 5 e aceitam nota final de 9 ou mais com probabilidade de pelo menos 0,9. Com revisor perfeito, a ordenação com parada por orçamento coincide com a regra de duas ações, o que não é evidência a favor das quatro. A interpretação de que as quatro ações não acrescentam ganho seria refutada se o ASR, o arranjo que preserva o corte e decide abaixo dele, recuperasse mais pares verdadeiros que a O3, a fila pela probabilidade calibrada, na mesma carga humana, além da amplitude entre réplicas.

### R10. Reserva: quatro modelos de linguagem tiveram VPP próximo na vigilância, sem ganho demonstrável pelo tamanho

Pergunta: Um modelo de linguagem maior resolve?

Origem: S17 e S18.

Quatro modelos foram comparados, Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 e Claude Fable 5.1, com VPP, a proporção de aceitações que são pares verdadeiros, de 16,0%, 19,0%, 19,0% e 18,2% na vigilância, sem ganho demonstrável pelo tamanho; a comparação é *post hoc*, e a cobertura do Claude Sonnet 5.5 é incompleta. Dez dos 21 falsos positivos do Claude Haiku 5.5 estão em região sem revisão documentada, e, se algum for par verdadeiro, o VPP de 16,0% está subestimado. O dossiê traz 14 dos 29 subescores, e 18 pares verdadeiros rejeitados por ao menos dois de três revisores originais não foram rejeitados pelo DeepSeek V4.1 Flash nem pelo Claude Haiku 5.5 sem as saídas do motor, achado compatível com ancoragem ou abstenção generalizada, sem identificar o mecanismo.

### R11. Reserva: o filtro de NONMATCH recupera mais que O3 só com a carga humana fixada e descarta pares verdadeiros

Pergunta: O filtro de NONMATCH ajuda?

Origem: S18 e S19.

No filtro, o modelo de linguagem descarta o que classifica como NONMATCH, e o revisor humano vê o restante, MATCH e UNSURE. Com *k* = 96 potenciais pares enviados ao modelo, a carga humana foi de 22 revisões, e o filtro recuperou 13 pares verdadeiros contra 6 da O3, a fila pela probabilidade calibrada, mas descartou 74 pares, nove deles verdadeiros (12,2%). A vantagem existe só com a carga fixada em 22, que é de uma réplica, e na unidade de custo da tese o filtro custa 1.172,5, contra 796 de O3; é uso exploratório, sem validação prospectiva.

### R12. Reserva: a disparidade por passo vem do corte; sexo e faixa etária não tiveram rejeição, o que não demonstra paridade

Pergunta: A disparidade por passo vem do modelo ou do corte?

Origem: S20.

Vem do corte: isolado, ele aceita 15 dos 40 pares verdadeiros do passo 11 (37,5%), contra 76 de 80 (95,0%) e 120 de 127 (94,5%) nos passos 6 e 9. O ASR, que preserva o corte, teve sensibilidade global de 87,0% na vigilância e de 88,7% na confirmação, e no passo 11, de 42,5% e 47,5%. Por sexo e faixa etária não houve rejeição, o que não demonstra paridade: são 71 pares verdadeiros no sexo feminino e 176 no masculino, com poder não calculado. Raça/cor não consta do extrato, e a equidade da fila recomendada não foi examinada.

### R13. Reserva: as diferenças entre o CdR em Python e o protótipo em Java são de três tipos

Pergunta: O CdR em Python equivale ao protótipo em Java?

Origem: S11.

Não, e a equivalência não foi demonstrada. As diferenças são de três tipos: erros do protótipo corrigidos no Python, como o carregador que descarta a primeira linha das tabelas de frequência; divergências de critério em relação a Lucena (2013), como o peso de concordância 0,8 das inversões de data, que o Java segue e o Python atribui 1,0; e melhorias do Python, como os subescores de município e de endereço, estes derivados de critérios já acrescentados à versão em Java. No lote sintético de 500 pares de fronteira a nota final foi idêntica em 11,2%, com diferença de até 4 pontos, e o efeito em dados reais não foi medido. O Estudo 1 cita Paiva (2025) entre os trabalhos que usaram a versão em Java, sem atribuir a ele nenhum efeito específico.

### R14. Reserva: o Estudo 3 não sustenta adoção, e a partição por óbito está no Estudo 4

Pergunta: Os Estudos 2 e 3 valem como evidência?

Origem: S12 e S21.

Não fundamentam a adoção dos *pipelines* avaliados. No Estudo 3, com teste de 18.509 potenciais pares e 74 pares verdadeiros, o Random Forest com SMOTE detectou 67 vínculos, contra 43 da nota final de 8 ou mais, mas dos 27 vínculos recuperados por ele e não detectados pela nota final de 8 ou mais, 25 já eram aceitos pelo corte. O otimismo vem de três fontes: indicadores de encerramento, que constituem vazamento de alvo, partição por linha e limiares escolhidos no teste. No Estudo 4 a partição é por óbito, com cinco partições externas e cinco internas e semente 42, sem os indicadores de encerramento.

### R15. Reserva: com a nota final oculta, o Claude Haiku 5.5 deu MATCH a 2 de 470 não pares e NONMATCH a 16 dos 36 pares verdadeiros abaixo do corte

Pergunta: O que mostra a medição do Claude Haiku 5.5 com a nota final oculta?

Origem: S19.

Em amostra estratificada de 717 potenciais pares, com os 247 pares verdadeiros e até 100 não pares por faixa, o Claude Haiku 5.5 deu MATCH a 2 de 470 não pares e NONMATCH a 16 dos 36 pares verdadeiros abaixo do corte (44,4%; IC 95% de 29,5% a 60,4%). O VPP, a proporção de aceitos que são pares verdadeiros, projetado ao extrato abaixo do corte, se o modelo revisasse todos os potenciais pares, é de 2,9% (IC 95% de 0,8% a 12,1%), e não o das aceitações da política, de 16,0%, medido com a nota final visível; a comparação é só de ordem de grandeza. O número pequeno de falsos positivos vale só para essa amostra, pois no topo da ordem pela nota final o mesmo modelo deu MATCH a dezenas de não pares (reserva R16). É exploração posterior, com uma única realização do modelo, de evidência moderada para as proporções medidas diretamente e fraca para o VPP projetado.

### R16. Reserva: frente a O2, o desempate pelo LLM encontrou de 2 a 6 pares verdadeiros a mais com 200 a 400 revisões, em uma realização, com IC que excluiu zero em 4 de 48 combinações

Pergunta: O desempate pelo modelo de linguagem sobre a nota final do CdR ajuda?

Origem: S24.

Frente à O2, a fila pela nota final, o desempate encontrou de 2 a 6 pares verdadeiros a mais com 200 a 400 revisões, em uma realização, e o IC 95% do *bootstrap* pareado excluiu zero em 4 das 48 combinações de variante, janela e orçamento, todas com 400 revisões e janelas de 2.000 ou 3.000, sem correção para multiplicidade. O modelo não decide nada: dá veredito aos primeiros pares abaixo do corte na ordem pela nota final e reordena essa janela, e o humano decide todo par que revisa. Com 25 a 100 revisões, esta e outras duas variantes deram resultados mistos, de −3 a +2. A evidência é fraca, com humano simulado perfeito, e entra como hipótese para a coorte nova.

### R17. Reserva: contra O1, nenhuma exploração superou a ordenação pelo escore do OpenRecLink de forma consistente

Pergunta: Alguma exploração superou a ordenação pelo escore do OpenRecLink?

Origem: S24.

Nenhuma, de forma consistente. A fila com aprendizado perde para a O1, a fila pelo escore do OpenRecLink, entre 96 e 400 revisões, de −7,1 a −1,6 pares com a escala de O1, e com 50 e 800 revisões fica acima, em +1,2 e +2,8 pares, dentro do ruído de um a três pares. O desempate pelo modelo de linguagem sobre O1 variou de −3,5 a +5,7 pares em média, com decisões reais para só 8% a 18% de cada janela, e o *bandit*, que escolhe a cada passo qual faixa de posição revisar, sobre faixas de O1 ficou de −10,1 a +0,4 par. O1 tem grandes blocos de empate, por isso a tabela usa o desempate aleatório, e a O3, a fila pela probabilidade calibrada, e a F4, a fusão da probabilidade com o escore, que dependem de *labels*, não foram comparadas a essas explorações.

### R18. Reserva: a evidência das explorações posteriores é moderada para o FP e o FN do LLM e fraca ou muito fraca para as filas com aprendizado

Pergunta: Quão forte é a evidência das explorações posteriores?

Origem: S24.

É moderada para os falsos positivos e falsos negativos do Claude Haiku 5.5 sem a nota final e fraca ou muito fraca para as filas com aprendizado. O falso negativo de 44,4% abaixo do corte estende o exame do dossiê, e o VPP, a proporção de aceitos que são pares verdadeiros, projetado, de força fraca, tem a mesma direção e não altera os 16,0%. A fila autocalibrável não superou a O2, a fila pela nota final, nem a O1, a fila pelo escore do OpenRecLink, a partir de 100 revisões, e o ganho da fila com aprendizado sobre O2 não se confirmou na replicação. Nada nessas explorações altera os resultados dos estudos originais, em que a redução de revisões veio da ordenação; comparar com a O3, a fila pela probabilidade calibrada, e com a F4, a fusão da probabilidade com o escore, e replicar em outro extrato ficam pendentes.

### R19. Reserva: o pacote gzcmd-record-*linkage* está na versão 0.6.1 no PyPI, e a instalação não reproduz, por si só, os resultados da tese

Pergunta: Qual versão do pacote foi usada, e a instalação reproduz a tese?

Origem: S24 e S27.

O pacote gzcmd-record-linkage está na versão 0.6.1 no PyPI, publicada em 14 de novembro de 2026, que difere da 0.6.0 apenas na documentação. A tese documenta essa versão e registra que, desde a 0.6.0, o pacote inclui os componentes experimentais da política v3.1 que ela emprega. A versão 0.7.0 é experimental, não foi publicada no PyPI e foi usada nas explorações posteriores, distinta da avaliada nos capítulos. A instalação não fornece os modelos ajustados nem reproduz, por si só, os resultados da tese; o pacote não gera potenciais pares nem executa blocagem, o perfil histórico se obtém com a função run_v3, a calibração por *Platt scaling* e a política v3, e não houve exercício independente de auditoria das justificativas.

### R20. Reserva: curvas de esforço de O1, O3, F3 e F4, sem ordenação que domine em todas as cargas

Pergunta: Como se lê a curva de esforço das fusões?

Origem: S16.

Cada curva dá, para a fila, as revisões necessárias para recuperar k pares verdadeiros, e menos revisões para o mesmo k indicam fila mais eficiente nessa meta. Com 96 revisões, a O1, a fila pelo escore do OpenRecLink, recupera 19, a O3, a fila pela probabilidade calibrada, 22, a F4, a fusão da probabilidade com o escore, 18, e a F3, a mesma fusão acrescida da fila pela probabilidade com as MACD, as medidas contínuas de diferença de datas, 25; os 36 exigem 2.521 revisões na O1, 23.443 na O3, 1.387 na F4 e 1.777 na F3. São valores pontuais, *post hoc*, e as fusões foram escolhidas entre variantes; nenhuma ordenação domina em todas as cargas, e nenhum intervalo de F3 ou de F4 ficou inteiramente abaixo de zero frente a O3 (reserva R08).

### R21. Reserva: as cascatas v3.1 e v3, aplicadas à base inteira, tiveram sensibilidade automática menor que a de ASR, que preserva o corte

Pergunta: Por que o arranjo sequencial e não as cascatas?

Origem: S17.

As cascatas v3.1 e v3, etapa de desenvolvimento anterior ao ASR, o arranjo que preserva o corte e decide abaixo dele, aplicam a política aos 61.696 potenciais pares sem preservar o corte e tiveram sensibilidade automática menor: na vigilância, 78,5% na v3.1 e 58,3% na v3, contra 87,0% do ASR. A vantagem do ASR vem inteiramente dos 211 pares verdadeiros que o corte já aceita, preservados por construção; a cascata v3.1 não aceitou automaticamente 27 deles na vigilância. Nos 36 pares verdadeiros abaixo do corte, na vigilância, o ASR aceitou 4 contra 10 da cascata e perdeu 13 contra 9. Os valores descrevem decisões da cadeia, não o desempenho isolado do modelo, e a sensibilidade observada não é projeção prospectiva.

### R22. Reserva: a subamostragem de 1:1 e de 1:5 dos negativos degradou o topo da fila; 1:20 e 1:50 não se distinguiram de forma demonstrável do treino sem subamostragem

Pergunta: Subamostrar os negativos no treinamento ajuda?

Origem: S13.

As razões 1:1 e 1:5 degradaram o topo da fila: a mediana de pares verdadeiros entre os 96 primeiros foi de 13,0 e de 18,5, contra 22 sem subamostragem. As razões 1:20 e 1:50 não se distinguiram de forma demonstrável do treino sem subamostragem, com 21,0 e 22,5. Os resultados sustentam evitar 1:1 e 1:5, mas não uma razão que melhore de forma estável o topo e a calibração; é análise exploratória, as medianas são entre sementes e o treino sem subamostragem tem uma só execução.

### R23. Reserva: as filas C e CM, com contexto por óbito, exigiram menos revisões que O3 pontualmente, sem vantagem demonstrada sobre F4

Pergunta: As filas com contexto por óbito, C e CM, superam F4?

Origem: S16.

Não há vantagem demonstrada sobre a F4, a fusão da probabilidade com o escore, embora as filas C, que acrescenta à probabilidade atributos de contexto por óbito e por registro candidato, e CM, que acrescenta ainda as MACD, as medidas de diferença entre datas de nascimento, tenham exigido, pontualmente, menos revisões que a O3, a fila pela probabilidade calibrada. Os atributos de contexto são calculados sem *labels*; CM menos a O1, a fila pelo escore do OpenRecLink, teve intervalo inteiramente abaixo de zero somente na meta de 36, com mediana de −1.296 revisões (−2.414 a −409). Contra a O3 e a F4 nenhum intervalo ficou inteiramente abaixo de zero, e a F4 foi melhor na meta de 9, com 10 revisões contra 14 de CM. É análise exploratória, *post hoc*, sem correção para multiplicidade; o mesmo registro do Sinan-TB pode cair em partições diferentes, e a generalização a bases novas não está demonstrada.

### R24. Reserva: na importância por valores de Shapley (SHAP), a família do nome responde por 38,5%, valor que descreve o modelo e não efeitos causais

Pergunta: Quais atributos pesam no modelo?

Origem: S11.

Na importância por valores de Shapley (SHAP), calculada para o Random Forest com SMOTE ajustado na base inteira, que não é o XGBoost que gera a O3, a fila pela probabilidade calibrada, a família do nome responde por 38,5% e a do nascimento por 14,2%, valores que descrevem o modelo e não efeitos causais. A soma não elimina a dependência entre famílias e não ordena atributos redundantes, e os indicadores de óbito constituem vazamento de alvo. Os valores foram calculados dentro da amostra de treinamento, em modelo que não é o avaliado no teste da ablação; além disso, o gerador histórico admite gravar importância de Gini no mesmo arquivo, de modo que o CSV, sozinho, não certifica o método.

### R25. Reserva: a ordem dos classificadores depende da métrica, e as nove estratégias de balanceamento não puderam ser ordenadas

Pergunta: Qual classificador é melhor, e o balanceamento ajuda?

Origem: S12.

Depende da métrica. No limiar de 0,5, o Random Forest teve F1 de 0,910, o Random Forest com SMOTE, de 0,908, e o Gradient Boosting, de 0,870, sem diferença distinguível pelo IC 95%; pela AUC-PR, a área sob a curva precisão-sensibilidade, o Random Forest teve 0,9564, o Random Forest com SMOTE, 0,9726 e a regressão logística, 0,9667. A regressão logística, com pesos de classe balanceados, tem o limiar de 0,5 mal posicionado por construção, e o VPP, a proporção de vínculos que são pares verdadeiros, baixo nesse ponto não indica menor discriminação. As nove estratégias de balanceamento tiveram F1 médio de 0,880 a 0,918, sem ordenar os métodos, e as estimativas internas são otimistas e não fundamentam a adoção dos *pipelines* avaliados.

### R26. Reserva: o que a tese acrescenta à literatura é a comparação, na mesma carga humana, entre ordenar a fila e automatizar a decisão

Pergunta: O que a tese acrescenta à literatura?

Origem: S26.

Acrescenta a comparação, posterior ao registro e na mesma carga humana, entre ordenar a fila e automatizar a decisão, incluindo o modelo de linguagem como aceitador e como filtro, com leitura negativa para a automação feita depois da avaliação. Não reivindica a ordenação por escore com revisão manual, já descrita por Sariyar e Borg (2010), o aprendizado ativo no *linkage*, avaliado por Sariyar e Borg (2012), nem a revisão assistida por tecnologia. Nenhum estudo consultado teve como foco a região abaixo de um ponto de corte único, e a aproximação mais próxima são as buscas manuais abaixo do limiar de Jiao et al. (2021), que acharam 11 pares verdadeiros adicionais. O levantamento não pretende ser exaustivo, e a ausência de uma informação na fonte consultada não estabelece ausência do procedimento na literatura.

## Rastreabilidade

A fala de cada quadro não traz número que o quadro correspondente (corpo, ressalva, título ou figura) não mostre. Os números abaixo aparecem só nas perguntas prováveis, que são material de preparação e não fala, com a linha da tese que os traz. O contrato de testes confere que cada número citado está na linha indicada.

| Quadro | Número falado | Fonte |
| :- | :- | :- |
| S17 | 755 | `conclusao.tex:27` |
| S17 | 0,94 | `estudo4_gzcmd.tex:147` |
| S17 | 0,11; 41 | `estudo4_gzcmd.tex:149` |

