# Roteiro da defesa

A apresentação tem 30 quadros principais e 19 de reserva, estes fora dos 40 minutos. Os tempos orientam a distribuição da fala, não representam duração medida.

## Cronometragem

| Bloco | Slides | Minutos | Acumulado |
| :-- | :-- | :-- | :-- |
| Capa | 1 | 0,5 | 0,5 |
| Problema e tese central | 2 a 4 | 4,0 | 4,5 |
| Dados e referência operacional | 5 a 7 | 4,0 | 8,5 |
| Referencial e decisão de limiar | 8 | 1,5 | 10,0 |
| Estudo 1: comparador | 9 e 10 | 2,0 | 12,0 |
| Estudo 2: estimador | 11 a 13 | 4,0 | 16,0 |
| Estudo 3: estratégias e detecção | 14 a 16 | 4,0 | 20,0 |
| Estudo 4: GZ-CMD | 17 a 22 | 10,0 | 30,0 |
| Estudo 5: equidade | 23 e 24 | 3,0 | 33,0 |
| Posicionamento | 25 | 1,5 | 34,5 |
| Limitações | 26 e 27 | 2,5 | 37,0 |
| Conclusões e contribuições | 28 e 29 | 2,5 | 39,5 |
| Agradecimentos | 30 | 0,5 | 40,0 |
| Total principal | 1 a 30 | 40,0 | 40,0 |

## Fala-guia por slide

### 1. Capa
A pesquisa foi desenvolvida no IESC-UFRJ, sob orientação de Rejane Sobrino Pinheiro, com foco no pós-processamento dos potenciais pares de bases de saúde. A apresentação acompanha cinco estudos. O resultado principal do GZ-CMD é o arranjo sequencial depois do corte do OpenRecLink; a cascata sobre a base inteira é comparação.

### 2. Informação sobre o cuidado
A tuberculose é uma condição marcadora, e a vinculação entre notificações e óbitos permite examinar desfechos do cuidado. O problema estudado não é apenas computacional: uma decisão incorreta de vínculo pode modificar a informação usada pela vigilância.

### 3. Da decisão ao orçamento
A pergunta operacional é como concentrar a revisão nos potenciais pares com maior probabilidade de vínculo, sem tratar menor esforço como melhora de acurácia. Primeiro se define o corte do OpenRecLink; depois, comparam-se ordenações sob o mesmo orçamento humano. A curva de esforço permite ao serviço explicitar a recuperação e os erros associados a cada orçamento.

### 4. Hipótese gerada nesta avaliação
A hipótese gerada nesta avaliação é que, abaixo do corte, ordenar pela probabilidade calibrada fora da amostra do classificador (O3) pode atingir a recuperação de P com menos revisões que a ordenação pelo escore do OpenRecLink (O1), pela revisão de menos pares na ordem escolhida, não pela política de quatro ações. A comparação com a probabilidade recalibrada de P (O4) é secundária, com o mesmo XGBoost. A meta é a recuperação de P no orçamento fixado antes dos desfechos, nos dois modos. A ordenação pela nota final do CdR (O2) não integra a hipótese: na confirmação, recupera nove vínculos com 11 revisões, sem FP nem LLM. LLM designa modelo de linguagem de grande porte; CdR, Comparador de Registros; OOF, fora da amostra; FP, falso positivo; VPP, valor preditivo positivo. P usa isotônica. Exige validação registrada em dados novos.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36. PAR é o *label*: 1 indica aceitação pelo corte; 2, recuperação pela revisão. O volume e a ordem históricos não foram observados. As duas fronteiras da nota do CdR aceitam 93 vínculos e 2 FP acima de 9, encaminham 21.620 ao revisor humano e projetam 95,7% no cenário de sensibilidade humana de 95%, não medida. Os cenários consideram 95%, 99% e 100% de sensibilidade humana, com especificidade perfeita assumida. Se a revisão tivesse seguido escore do OpenRecLink decrescente, o cenário seria de pelo menos 2.156 pares por passo ou 2.522 no limiar global, equivalentes a 718 h 40 min e 840 h 40 min sob a hipótese de 20 min por par, não tempo observado. Isso não reconstrói o trabalho histórico.

### 7. Faixas e referência operacional
A nota do CdR não é o escore probabilístico do OpenRecLink. Nas faixas da nota: abaixo de 5 há 39.981 pares e 3 vínculos; em [5,9), 21.620 e 151; a partir de 9, 95 e 93. Houve revisora única, sem classificação independente. Oliveira (2026, p. 34 e 36 a 38, Quadros 5 e 6) documenta CdR apenas no passo aberto, com 5 pares, não nos passos 6, 9 e 11. Os 36 PAR=2 se distribuem em 4, 7 e 25 nesses passos. A precedência do CdR à revisão e a disponibilidade de nota e subescores são hipóteses de trabalho, não fatos documentados para esses passos. Sob essa hipótese, haveria viés de verificação e incorporação em direção a O2 a O4; a magnitude não foi estimada.

### 8. Mapa conceitual
O modelo de Fellegi e Sunter é o referencial do escore probabilístico; a classificação por corte é decisão de limiar. O processo de origem utilizou OpenRecLink original, em C, CdR e revisão clerical por revisora única, sem atribuição pessoal das etapas. O autor não participou desse processo e recebeu somente o extrato pós-CdR. A ordem entre CdR e revisão não está documentada. A tese reorganiza a decisão posterior, sem reestimar os parâmetros de comparação nem substituir a blocagem.

### 9. Comparador em Python
O comparador torna explícita a evidência campo a campo, com subescores e uma nota ponderada por pesos fixos. A equivalência parcial com o protótipo Java de Peçanha (2015) foi executada em pares sintéticos. A captura apresenta a interface do Comparador de Registros, versão 0.1, *commit* 3b43598, com dados sintéticos, não uma interface do GZ-CMD.

### 10. Equivalência Java e Python
A comparação em 500 pares sintéticos encontrou concordância de 100,0% nas datas por igualdade e por um dígito, de 99,2% nas abreviações e nota final idêntica em 11,2%. Nas inversões de data, Lucena e o Java atribuem peso 0,8, enquanto o Python usa 1,0. Essa diferença foi mantida e declarada como limitação. O lote de fronteira não estima concordância nos dados reais.

### 11. Comparação de protocolos
RF+SMOTE atingiu F1 de 0,931 na ablação original. No fluxo Modal, XGBoost isolado atingiu 0,943 e híbrido OR 0,958, com 68 VP, 0 FP e 6 FN, contra 0,610 da nota 8. O XGBoost do QW1 atingiu 0,920 no teste; os protocolos não são intercambiáveis, e a escolha dos limiares nos testes torna os resultados selecionados exploratórios.

### 12. Balanceamento
As nove estratégias de balanceamento foram avaliadas no Random Forest, com F1 médio entre 0,880 e 0,918. Essa variação descreve o experimento realizado, não demonstra que uma estratégia de reamostragem seja necessária ou superior em qualquer modelo.

### 13. Estimador do GZ-CMD
O GZ-CMD usa outro XGBoost, com 30 atributos e calibração isotônica aninhada, agrupando por REFREC nos níveis externo e interno. Na área cinza da nota, a AUC é 0,995 e a AP é 0,862; o corte ilustrativo de 0,5 produz 119 verdadeiros positivos e 28 falsos positivos. Esses números descrevem o estimador original, não o arranjo. Nele, o classificador é ajustado na parcela de treinamento, e a isotônica é reajustada apenas com as probabilidades OOF internas do treino abaixo do corte, sem os *labels* da partição externa.

### 14. Finalidades operacionais
A vigilância atribui maior consequência à perda do vínculo, enquanto a confirmação atribui maior consequência ao falso vínculo. A fronteira permite explicitar esse compromisso, sem transformar os pontos explorados no teste em limiares validados para uso externo.

### 15. Detecção de vínculos
No teste da ablação, o RF+SMOTE detectou 67 vínculos e a nota 8, comparador hipotético, detectou 43: são 24 a mais frente à nota 8, comparador hipotético. Dos 27 recuperados exclusivamente pelo modelo, 25 são PAR=1, já aceitos pelo corte do OpenRecLink de origem, e dois são PAR=2. Os 27 correspondem a 24 registros distintos do SIM; os 247 vínculos do extrato, a 204. Não se estima redução da subnotificação. Fonte: `data/banca_b_recuperados_por_par.json`.

### 16. Limites da comparação
Os indicadores derivados do encerramento por óbito constituem vazamento de alvo. A sensibilidade 2×2 foi executada: o F1 do RF+SMOTE passou de 0,931 para 0,938 sem óbito, mas caiu para 0,846 com REFREC. No QW1 local na GPU, o XGBoost passou de 0,937 para 0,878 com REFREC, sem substituir o resultado histórico de 0,920. O otimismo aparece sobretudo na partição por linha, embora mudem os vínculos e denominadores. O GZ-CMD já utiliza agrupamento por REFREC, sem indicadores de óbito; permanece possível dependência por COMPREC.

### 17. Arquitetura depois do corte
O arranjo preserva a aceitação acima do corte e aplica o GZ-CMD abaixo dele. As guardas usam nota e probabilidade recalibrada; as faixas estratificam o erro do revisor. A isotônica é reajustada apenas no treino abaixo do corte, sem *labels* da partição externa. As proporções de erro do revisor foram congeladas antes das chamadas novas, sem realimentação por essas respostas, mas P reutiliza sobretudo lotes históricos também usados para estimar as proporções do revisor. Não é avaliação prospectiva. *Grey-Zone* no nome remete ao conceito de Fellegi e Sunter, a região que a regra automática não decide; nos resultados, área cinza da nota designa somente [5,9).

### 18. Duas ações e orçamento
Com duas ações, revisor humano e não vínculo, revisor perfeito e perdas aditivas, a perda da revisão é C_H e a do não vínculo é p × C_FN. Com p verdadeiro e calibrado, compensa revisar quando p ≥ C_H/C_FN; sob orçamento, revisam-se os maiores p até a parada. O1 e O2 não são probabilidades; com p estimado, não se valida esse limiar. Atribuir a falha às entradas, diante de 9,06 FP previstos e 21 observados, é hipótese de trabalho, sem isolamento causal nem dispensa da avaliação empírica do critério. O fluxo continua hipótese a validar.

### 19. Arranjo sequencial com registro interno
P conserva os 211 vínculos do corte e acrescenta 4 na vigilância e 8 na confirmação. Não reproduz 13 e 27 dos 36 vínculos recuperados pela revisão clerical de origem. Aceita 25 e 12 potenciais pares abaixo do corte: 4 e 8 são PAR=2; dos demais, 21 e 4 são FP contra o *label*, rejeitados ou não examinados nessa revisão. Portanto, não se atribui essa condição aos 25 e 12 como um todo. As revisões humanas são 96 na vigilância e 17 na confirmação; os pares avaliados pelo LLM são 344 e 41. A comparação de carga, o VPP e os mecanismos são análises exploratórias *post hoc*, não registradas. S3 designa a avaliação com os *labels*; os fechamentos e a cronologia estão na reserva.

### 20. Curva de esforço e ponto de parada
A curva de esforço relaciona revisões humanas e vínculos recuperados. O3 é a probabilidade calibrada OOF, e O4 a probabilidade recalibrada de P. Na vigilância, nenhuma ordenação domina; na confirmação, O2 domina fracamente P: recupera 9 PAR=2 com 11 revisões, sem FP nem LLM; O3 precisa de 29. Para todos os 36, O1 requer 2.521 e O3, 23.443, fora do limite gráfico de 2.522. Os pontos de P incluem 344 pares adicionais avaliados pelo LLM na vigilância e 41 na confirmação. Revisor perfeito é hipótese dos dois lados. Se a revisão histórica seguiu o escore do OpenRecLink, favoreceria O1; O3 e O4 foram treinados nos mesmos *labels* verificados. O4 iguala P com 106 revisões na vigilância e 27 na confirmação, contra 125 e 29 de O3. No orçamento humano original de P (96 e 17), P recupera 23 e 9; O3 recupera 22 e 6, deixando 14 e 30 dos 36 fora. Fontes: `comparador_mesma_carga.csv:19,22,27,32,37`.

### 21. Aceitador e filtro: papéis distintos
Como aceitador, o LLM teve VPP de 16,0% na vigilância, com 21 FP contra 9,06 previstos, análise *post hoc*. Como filtro de NONMATCH, em k=96 na vigilância são 74 pares avaliados pelo LLM como NONMATCH e 22 revisões humanas: 13 PAR=2 contra 6 sem filtro na mesma carga. O VPN é 87,8% em k=96 (IC 95%: 78,5 a 93,5); 78,0% em k=58. A sensibilidade do filtro é 59,1% (IC 95%: 38,7 a 76,7): descarta 9 dos 22 vínculos do topo. A vantagem é condicionada a H=22; na unidade de custo da tese, perde para O3 com 96 revisões. Na confirmação k=125, descarta 10 dos 23 vínculos. MATCH e inconclusivos vão ao humano, nunca à aceitação autônoma. É hipótese exploratória a validar, com protocolo fixado antes dos desfechos e respostas de datas diferentes. Fontes: `filtro_llm.csv:4,5,13`; `estudo4_gzcmd.tex`, filtro.

### 22. O LLM como aceitador não se paga abaixo do corte
P permanece principal porque o registro fixou a isotônica como calibração padrão, com Platt previsto apenas para partições com menos de 10 PAR=2; P-Platt, também registrado como braço, passou a sensibilidade na emenda 3. Com MATCH ao humano, P usa 121 revisões para 23 vínculos na vigilância; O3 iguala em 125. Na confirmação, ambos recuperam nove em 29, mas O2 requer 11. Revisor perfeito é hipótese dos dois lados. Uma diferença de um vínculo em 36 não sustenta superioridade. Perdas totais, incluindo FN: vigilância, P 957,72, O3 775, P-MATCH 772,72, O2 1.296 (12 recuperados); confirmação, 957,2, 569, 569,2 e 551. Custos C_FP/C_FN: 10/50 e 100/20; C_H=1 e C_LLM=0,005. FP humano zero é hipótese assimétrica à política, não medida. P-Platt tem perda menor nos quatro cenários de estabilidade, sem justificar escolha retroativa. Fontes: `custos_equivalentes_exploratorios.csv:2-5,8-10,12`; `robustez_perda_ponderada.csv:2-9`.

### 23. Equidade por sexo
No arranjo P de vigilância, o VPP é de 86,3% no sexo feminino e 93,3% no masculino, sem rejeição após Benjamini-Hochberg na família den30, p ajustado de 0,360 (`equidade_arranjo_den30_pairwise_tests.csv:3`). Na cascata sobre a base inteira (comparação), é de 87,9% e 84,0%. A aceitação do arranjo combina corte e respostas de vínculo do LLM; o encaminhamento humano conta como não aceito. Os 211 vínculos acima do corte são aceitos por construção. As diferenças de sensibilidade entre subgrupos dependem da proporção de vínculos PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36 vínculos. As diferenças de VPP dependem dos 21 e quatro falsos positivos abaixo do corte. Ausência de rejeição não comprova equidade. Raça/cor não está disponível.

### 24. Desigualdade operacional
A disparidade por passo é herdada do corte e do *label*, não um efeito isolado do GZ-CMD. O corte aceita 15 dos 40 vínculos do passo 11 (37,5%), contra 76 de 80 no passo 6 (95,0%) e 120 de 127 no passo 9 (94,5%). O arranjo aceita dois dos 25 PAR=2 do passo 11 na vigilância e quatro na confirmação: 42,5% e 47,5%. Na vigilância, fica abaixo dos 45,0% da cascata sobre a base inteira. Com mínimo de 30 nos dois denominadores antes de Benjamini-Hochberg, P tem 3/26 e 2/26 rejeições; a cascata, 6/28 e 1/21, todas operacionais. Cada decisão muda a família elegível; as contagens não ordenam equidade. Fonte: `data/equidade_arranjo_den30_recomputacao.json`; `data/par_por_passo.csv`.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. Na linha desta tese, a tabela de posicionamento registra SIM e Sinan-TB e a política com regras de guarda, *expected loss* e revisão assistida por modelo de linguagem; a tese propõe essa composição operacional, com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
A revisora foi única, sem classificação independente. O corte integra o *label*. Se a revisão histórica seguiu o escore do OpenRecLink, favoreceria O1; O3 e O4 também usaram os mesmos *labels* verificados. A ordem histórica é desconhecida, e 2.156 revisões constituem cenário. REFREC não separa COMPREC. Respostas históricas reutilizadas, proporções com denominadores mínimos e transferência de coortes mais fáceis impedem interpretação prospectiva. A sensibilidade humana não foi medida, e a especificidade humana é tratada como perfeita: os cenários não modelam FP humanos. O comparador *a posteriori* era conhecido antes do registro interno; o código registrado está ausente do git, e há desvios de protocolo. O registro cobre recalibração, braços e avaliação; a análise de estabilidade mede estabilidade nos mesmos dados, não generalização.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. A transferência internacional à Modal e a APIs externas não foi objeto de análise específica nos pareceres, limitação declarada em `capitulo_dados.tex:31`. O Zenodo é público com arquivos restritos. O repositório contém agregados e escores/decisões por pseudônimo, sem *label*; MIT vale apenas para o código. A exportação por HMAC foi implementada, mas a matriz não foi depositada nem saneou retroativamente as execuções. A equivalência Java é parcial; a comparação com Lages não foi realizada.

### 28. Perdas de cobertura e cenários humanos
As perdas são de cobertura, não de erro humano. Os cenários de sensibilidade humana de 95%, 99% e 100% assumem especificidade perfeita. As amplitudes de até 0,53 p.p. nas cargas reduzidas e 0,73 p.p. na revisão integral são limites por construção, não achados empíricos. Sob revisor perfeito, P recupera 23 e 9 dos 36, não reproduzindo 13 e 27, mas aceita 21 e 4 FP. Em 96 e 17 revisões, O3 recupera 22 e 6, deixando 14 e 30 fora. Para igualar P, O3 requer 125 e 29; O4, 106 e 27. Para os 36, O1 requer 2.521 e O3, 23.443. As filas são comuns aos modos; os orçamentos, distintos. O2 recupera nove com 11 revisões na confirmação, sem FP nem LLM. Fonte: `comparador_mesma_carga.csv:2,19,22,27,32,37`; `cenarios_sensibilidade_humana.csv:3,21`.

### 29. Contribuições e recomendação final
As contribuições são a decomposição por PAR, a curva de esforço, a política com registro de decisões, o CdR em Python, o código público e o *benchmark* restrito. Não houve auditoria independente nem identificação causal de ancoragem. Preservar o corte e revisar em ordem decrescente de uma probabilidade ou escore é proposta a validar. A curva de esforço exige amostra independente verificada para escolher fila e parada conforme o orçamento humano. Na vigilância, nenhuma ordenação domina; na confirmação, O2 domina fracamente P. Restantes são não examinados, não óbitos sem notificação: permanecem na investigação; duplicações não estimadas. O filtro exige orçamento humano fixo (H=22 em uma réplica). Nas condições avaliadas, não deve aceitar sozinho. Só foi avaliado por API externa; modelo em ambiente controlado não avaliado. Adoção exige validação registrada em dados novos.

A recomendação exploratória é a fusão RRF do escore com a probabilidade calibrada no lugar do escore isolado, a validar com referência cega; vantagem sobre a probabilidade não demonstrada. Revisão cega independente não foi possível antes da defesa e integra o protocolo da coorte nova.

### 30. Agradecimentos
Agradece-se à orientação, à banca e à equipe responsável pelo processamento e pela revisão. Abre-se a discussão com a banca.

### 31. Backup: base inteira (comparação), cascata gráfica
A figura sintetiza a cascata sobre a base inteira por política e modo, não o arranjo. Os subtotais humanos incluem encaminhamento direto e inconclusivos do LLM, não constituindo novos ramos a somar novamente.

### 32. Backup: base inteira (comparação), contagens
As contagens da cascata sobre a base inteira separam aceitação automática, perda explícita e vínculo pendente ao revisor humano. A sensibilidade automática inclui o LLM, mas não a decisão humana ainda não realizada.

### 33. Backup: faixas OOF
A tabela detalha a discriminação do estimador por faixa da nota, com corte 0,5 fixado antes da análise. O desempenho do estimador isolado não deve ser atribuído à política v3.1, que pode rejeitar pela guarda ou escolher revisão por custo.

### 34. Backup: QW1
As redes neurais foram comparadas no QW1, com validação interna e teste da mesma base. Os resultados não demonstram uma hierarquia universal entre famílias, e o F1 de 0,943 não pertence a essa tabela.

### 35. Backup: ablação
As origens experimentais diferem, mas a função de regras é a mesma após a correção e recomputação local, sem novo treinamento. No fluxo Modal, híbrido OR tem F1 de 0,958 (68/0/6), XGBoost isolado 0,943; a cascata ML seguida de regras tem 0,825 (52/0/22), e AND, regras seguidas de ML e consenso com regras têm 0,896. Os limiares foram escolhidos no teste. Fonte: `tables/tab_ablation_best_category.tex:14-20`.

### 36. Backup: balanceamento
A figura permite consultar todas as estratégias avaliadas no Random Forest. A dispersão é interna ao protocolo e não mede transferência para outra população ou período.

### 37. Backup: detecção
A figura reaproveita a comparação de métodos no teste da ablação, com 74 vínculos. A contagem é de vínculos detectados de pessoas notificadas por TB, não uma nova contagem de mortes cuja causa básica seja TB.

### 38. Backup: ablação do dossiê
Os 18 vínculos foram selecionados por erro compartilhado, e a retirada conjunta das saídas modificou as decisões dos revisores. Passar a inconclusivo não equivale a reconhecer corretamente o vínculo, e não há estimativa de falsos positivos da coorte inteira nesse subconjunto.

### 39. Backup: instruções
A distinção entre arquivo escrito e mensagem efetivamente enviada é necessária, pois o carregador da v3 interrompia o sistema no primeiro separador. Na v4, a política de evidência antecede o separador e permite a abstenção diante de informação insuficiente.

### 40. Backup: formalização
As perdas incorporam probabilidade, consequências do erro e custo da revisão. A ação LLM inclui a possibilidade de inconclusivo e seu encaminhamento ao humano, sob capacidade disponível e após aplicação das guardas.

### 41. Backup: base inteira (comparação), sensibilidade
Os exercícios de exclusão e de expectativas por faixa pertencem à cascata sobre a base inteira. Avaliam dependências das escolhas e não são intervalos de confiança. A sobreposição residual e a seleção adaptativa permanecem.

### 42. Backup: privacidade
O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). A ausência de nomes no dossiê não transforma o extrato em dado anônimo nem elimina risco de reidentificação. O hash determinístico das execuções históricas pode ser recalculado por quem detém os identificadores de origem. A exportação restrita com `pair_id` por HMAC foi implementada, sem depósito da matriz nem saneamento retroativo dos artefatos históricos.

### 43. Backup: base inteira (comparação), vigilância REFREC
A reserva mantém a comparação histórica da cascata sobre a base inteira: 194 aceitos, 15 perdidos e 38 vínculos pendentes ao revisor humano, totalizando 247. O fechamento de P é (211+4)+13+19=247, com custo LLM estimado de US$ 0,94. A proporção aninhada de P no *label* não recebe IC. Os cenários de sensibilidade humana são 95%, 99% e 100%, não medidas do serviço.

### 44. Backup: base inteira (comparação), confirmação REFREC
Na base inteira, a cascata aceita 132, perde 28 e encaminha 87 vínculos, totalizando 247. O fechamento de P é (211+8)+27+1=247, com custo LLM estimado de US$ 0,11. Encaminhar não é perder o vínculo. A sensibilidade humana real não foi medida; os cenários de 95%, 99% e 100% tratam a especificidade como perfeita, sem modelar FP humanos.

### 45. Backup: análises de sensibilidade do arranjo e comparadores
A tabela preserva os braços e os mecanismos: perdas de P por guarda/custo/LLM de 1/6/6 na vigilância e 1/22/4 na confirmação. O registro antecede a S3 em 49 minutos, com comparador conhecido e código registrado fora do git. P-Platt constava como braço no registro, foi reclassificado como sensibilidade na emenda 3, e o critério 18/20 foi registrado depois da S3. A posição dos FP frente aos limiares permanece rastreável. Fonte: `estudo4_gzcmd.tex:316,322,328`.

### 46. Backup: análise de estabilidade com registro interno, não confirmatória
O bloco numérico (N) varia *threads* e semente; o bloco de partição (R), as partições. A dominância de Pareto ocorreu em 7/20 e 4/20 na vigilância e 7/20 e 5/20 na confirmação, sem inversa. Aplicando custos às medianas marginais, P-Platt tem perda menor nos quatro cenários: diferenças de 144,1/94,4 e 101,6/143,0. Isso não é mediana da perda. No bloco numérico, a variação máxima mediana de p é 0,46 em P e 0,17 em P-Platt. Respostas do LLM fixas: não se mede sua variabilidade. Não há superioridade de P nem validação externa. Fonte: `data/robustez_perda_ponderada.csv:2-9`.

### 47. Backup: subamostragem
A subamostragem é exploratória. Na isotônica, a mediana do topo 96 cai de 22 para 13 com 1:1 e 18,5 com 1:5; 1:20 e 1:50 são indistinguíveis da referência. Não se sustenta preferência pela prevalência original frente a essas duas razões: a referência tem uma execução e cada razão, dez sementes. O Brier mediano piora em todas as proporções; isso não basta para escolher a ordenação. Não se infere melhora de equidade. A queda de 25 p.p. no passo 6 tem denominador de 4 PAR=2. Fonte: `subamostragem_treino_resumo.csv:24,25,786`.

### 48. Backup: filtro de NONMATCH
Na vigilância, em k=96 são 74 pares avaliados pelo LLM como NONMATCH e 22 revisões humanas: 13 PAR=2 contra seis sem filtro, apenas sob H=22. VPN: 87,8% em k=96; 78,0% em k=58. Sensibilidade 59,1% (IC 38,7 a 76,7), com nove dos 22 vínculos descartados. Na unidade de custo, filtro 1.172,5 contra O3 796 com 96 revisões. Na confirmação, k=125 descarta 10 dos 23 vínculos. As decisões dos PAR=2 coincidem entre modos nesse topo; há seis diferenças nos não vínculos. Protocolo fixado antes dos desfechos, respostas de datas diferentes; exploração, não validação. Fontes: `filtro_llm.csv:4,5,13`; `estudo4_gzcmd.tex:578`.

### 49. Backup: mecanismos, cenários e registro interno
Os mecanismos da política original são distintos do erro do revisor humano: guarda, decisão por custo e resposta do LLM. Os cenários humanos não medem seu erro nem modelam FP humanos. O registro interno antecede a S3 em 49 minutos. O critério prospectivo, não testado, exige outra coorte, referência cega e ordenação e orçamento prévios. A ordenação primária, pela probabilidade calibrada do classificador (O3), deve atingir a recuperação de P com pelo menos 25% menos revisões que a ordenação pelo escore do OpenRecLink (O1), pelo limite inferior do IC 95% entre partições predefinidas, nas duas metas. A margem foi fixada com os dados à vista: reduções de 75,0% e 43,1%. A nota final do CdR (O2) fica fora da hipótese. Ganho das quatro ações sobre duas ações (O3) exige diferença acima da amplitude entre réplicas (de 9 a 21 e de 21 a 29 perdas). Fontes: `data/comparador_mesma_carga.csv`; `data/robustez_arranjo_resumo.csv`.

## Perguntas prováveis

São 86 respostas, com no máximo 70 palavras cada, sem contar pergunta e fonte.

### 1. O probabilístico é inútil? Onde está o módulo Python do OpenRecLink?

O corte do OpenRecLink aceitou 211/247 vínculos, ou 85,4%. A entrega Python é o CdR, não a blocagem. A equivalência com o Java de Peçanha (2015) foi avaliada: 100,0% nas datas por igualdade/um dígito, 99,2% nas abreviações e nota idêntica em 11,2% dos pares sintéticos. Inversões pesam 0,8 em Lucena/Java e 1,0 no Python, limitação declarada. Não há equivalência integral.

Fonte: `capitulo_dados.tex`, seção Referência operacional (`sec:padrao-ouro`); `estudo1_comparador.tex`, Resultados (`sec:e1-resultados`). Slides 6, 9 e 10.

### 2. Aprendizado de máquina é só exploração de possibilidades?

Há exploração, com limiares escolhidos no teste: F1 de 0,931 na ablação; no fluxo Modal, 0,943 para XGBoost isolado e 0,958 para híbrido OR. O GZ-CMD usa outro estimador, com 30 atributos, calibração aninhada e agrupamento por REFREC: AUC de 0,995 e AP de 0,862 na área cinza da nota. Isso sustenta discriminação interna, não validação externa da política.

Fonte: `estudo3_estrategias.tex`, Limitações (`sec:e3-limitacoes`); `estudo4_gzcmd.tex`, Classificador supervisionado e calibração fora da amostra (`sec:e4-classificador-oof`). Slides 11 e 13.

### 3. Isso já foi feito por Lages ou pelo CIDACS?

Há precedentes de classificação supervisionada e redução da revisão. O CIDACS constitui referência brasileira de infraestrutura e métodos de *linkage*. Aqui, avaliam-se ordenação, esforço e política com registro de decisões, sem exercício independente de auditoria das justificativas, numa base de razão 1:249. Lages trabalha com razão 1:5 e outro delineamento. Não foi realizada comparação empírica entre as implementações; a tabela de posicionamento não é uma prova de superioridade.

Fonte: `capitulo2.tex`, Classificação supervisionada, experiência brasileira e `tab:posicionamento`; `estudo1_comparador.tex`, Discussão (`sec:e1-discussao`). Slide 25.

### 4. Por que F1 próximo de 0,93 aqui e 0,996 em Lages?

O valor 0,943 pertence ao XGBoost isolado do fluxo Modal; híbrido OR tem 0,958, com limiares escolhidos no teste. QW1 tem 0,920. Lages reporta 0,996 em outro conjunto e protocolo, com razão de classes distinta. A comparação não identifica a causa da diferença nem ordena estimadores. O resultado por faixa OOF é separado, sem atribuir o desempenho global apenas aos casos difíceis.

Fonte: `capitulo2.tex`, `tab:posicionamento`; `tab:qw1-deep-learning`; `tab:ablation-best-category`; `tab:e4-desempenho-faixas-oof`. Slides 11, 25 e 33.

### 5. Como foi construída a referência operacional?

O processo de origem utilizou OpenRecLink original, em C, CdR e revisão clerical por revisora única, sem classificação independente. Dos 247 vínculos, 211 vieram do corte e 36 da revisão. A ordem CdR/revisão e a disponibilidade da nota não estão documentadas nos passos exportados. A monografia não sustenta atribuir pessoalmente essas etapas; o autor não participou delas e recebeu o extrato pós-CdR.

Fonte: `capitulo_dados.tex`, Referência operacional (`sec:padrao-ouro`); `apendice_linkage_origem.tex` (`ap:linkage-origem`). Slides 6 e 7.

### 6. O arranjo é determinístico?

O corte e as guardas são determinísticos, mas a cadeia inclui probabilidade estimada, custos e Claude Haiku 5.5, cuja geração é estocástica. Na vigilância, são 211 vínculos pelo corte e quatro pela política, com 13 perdidos e 19 vínculos entre 96 encaminhamentos ao revisor humano. Abaixo do corte, as aceitações vieram do LLM, com 21 falsos positivos. Não há garantia de repetibilidade integral.

Fonte: `data/arranjo_corte_metricas.csv:2,12`; `data/arranjo_corte_s1_resumo.csv:2-15`; `sec:e4-politica-v31`. Slides 17, 19 a 21.

### 7. Como o arranjo se compara à revisão integral?

A revisão integral tem amplitude de 0,73 p.p. entre sensibilidade humana de 95% e 100%; nas estratégias de carga reduzida, o máximo é 0,53 p.p. Há cenário de 99%. As perdas são de cobertura: vínculos não chegam ao revisor humano. A especificidade humana é assumida perfeita. Os cenários não medem erro real nem FP humanos; a curva explicita o orçamento.

Fonte: `data/cenarios_sensibilidade_humana.csv`; `data/arranjo_corte_metricas.csv:8,12,26,30`. Slides 6 e 28.

### 8. Dados identificados foram enviados à nuvem?

O extrato é pseudonimizado, não anônimo. Foi enviado à Modal com datas completas, sexo e bairro; os dossiês enviados a APIs externas continham atributos derivados e pseudônimo, sem nomes nem datas completas. A transferência internacional não foi objeto de análise específica nos pareceres. O autor não recebeu registros nominais. A fundamentação na LGPD não equivale à demonstração de autorização para todos os envios.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`), ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`). Slides 27 e 42.

### 9. Como discutir equidade sem raça/cor?

A falta de raça/cor impede examinar disparidades raciais. Avaliaram-se sensibilidade e VPP, não toda a igualdade de oportunidades de erro. No passo 11, P tem 42,5% e 47,5% nos dois modos; na vigilância, fica abaixo dos 45,0% da base inteira (comparação). A diferença reflete sobretudo o corte e o *label*. Ausência de rejeição demográfica não demonstra equidade.

Fonte: `data/equidade_arranjo_group_metrics.csv`, passo; `data/equidade_arranjo_pairwise_tests.csv`, rejeições de P. Slides 23 e 24.

### 10. Por que o arranjo, e não a cascata publicada?

O arranjo preserva 211 vínculos por construção. Nos 36 PAR=2, aceita 4 e 8 contra 10 e 8 da cascata, nos dois modos; perde 13 e 27. A vantagem total vem do corte, não de maior recuperação abaixo dele. O comparador *a posteriori* era conhecido antes do registro de recalibração, braços e avaliação. O registro não protegeu o enquadramento nem demonstrou superioridade.

Fonte: `data/arranjo_corte_metricas.csv:2,4,12,22,30`; `data/arranjo_corte_par2.csv:3,7,12,16`; `gzcmd/prereg_arranjo_corte.yaml`. Slides 17, 19, 20, 26 e 28.

### 11. Por que não adotar P-Platt?

P usa isotônica por decisão de registro, com 32 positivos no primeiro ajuste. P-Platt era braço e virou sensibilidade na emenda 3; o critério 18/20 veio depois da S3. Dominância de Pareto: vigilância 7/20 e 4/20; confirmação 7/20 e 5/20. Pela perda aplicada às medianas marginais, P-Platt é menor nos quatro cenários. Isso não sustenta superioridade de P nem validação externa.

Fonte: `data/arranjo_corte_metricas.csv:12,14`; `data/robustez_arranjo_resumo.csv:2-5`; `data/robustez_perda_ponderada.csv:2-9`; `estudo4_gzcmd.tex:322,328,539`. Slides 22 e 46.

### 12. Qual é a carga comparável à da revisão clerical de origem?

A ordem histórica não foi documentada. Se tivesse sido decrescente por escore do OpenRecLink, o cenário seria de 2.156 pares por passo ou 2.522 no limiar global, equivalentes a 718 h 40 min e 840 h 40 min sob 20 min por par. Não são horas observadas. A curva de esforço evita depender desse cenário; o serviço define a aceitabilidade das perdas.

Fonte: `data/carga_abaixo_corte.csv:5,9`; `data/carga_abaixo_corte_horas.csv:3,2`. Slides 6 e 20.

### 13. O que significa VPP de 16% do LLM?

Na vigilância de P, quatro das 25 respostas de vínculo abaixo do corte eram vínculos, VPP de 16,0%, com IC de 6,4 a 34,7%. As demais 21 eram falsos positivos. Na confirmação, foram oito de 12, ou 66,7%. Esses resultados pertencem à região selecionada pela política, não estimam uma capacidade geral de Claude Haiku 5.5.

Fonte: `data/arranjo_corte_vpp_llm.csv:2,8`. Slide 21.

### 14. De onde vêm os vínculos perdidos?

Em P, as perdas distribuem-se por guarda/custo/LLM, na vigilância: 1/6/6, totalizando 13; na confirmação: 1/22/4, totalizando 27. Os encaminhamentos são desfechos do arranjo; não se atribuem exclusivamente ao LLM. As aceitações e os falsos positivos abaixo do corte vieram do LLM. Essa decomposição identifica os ramos da decisão, sem isolar causalmente o efeito da calibração.

Fonte: `data/arranjo_corte_perdas_mecanismo.csv:2-4,50-52`. Slide 20.

### 15. Como justificar perder 75% dos vínculos da revisão na confirmação?

São 27 dos 36 PAR=2, ou 75,0%, não uma manutenção da recuperação da referência de origem. A confirmação encaminha somente 17 potenciais pares ao revisor humano, mas a redução de encaminhamentos vem com essas perdas. O estudo documenta a troca, não demonstra sua aceitabilidade clínica ou epidemiológica. A adoção depende de decisão do serviço e validação em dados novos.

Fonte: `data/arranjo_corte_par2.csv:16`; `data/arranjo_corte_metricas.csv:30`. Slides 20 e 28.

### 16. Por que as proporções congeladas não anteciparam os falsos positivos?

As proporções de erro foram estimadas em coortes com potenciais pares mais fáceis e denominadores mínimos. A seleção abaixo do corte muda a composição. Em P na vigilância, previam cerca de 9,06 falsos positivos, contra 21 observados. A validação cruzada não garante transferência dessas proporções. É necessário estimar e validar o erro na região de uso, sem realimentação oportunista pelos desfechos.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `data/arranjo_corte_taxas_crossfit.csv`. Slides 21 e 26.

### 17. A AUC de 0,995 elimina o viés de incorporação?

Não. A AUC descreve discriminação interna na referência operacional; não mede independência do *label* nem calibração absoluta. O corte define PAR=1. A disponibilidade da nota à revisora é hipótese de trabalho nos passos exportados. REFREC reduz uma dependência, mas não separa simultaneamente COMPREC nem cria referência externa. O GZ-CMD não usa os atributos de óbito dos experimentos exploratórios.

Fonte: `tab:e4-desempenho-faixas-oof`; `data/arranjo_corte_metadata.json`, `verification_bias` e `incorporation_bias`. Slides 13, 16 e 26.

### 18. O arranjo pode ser reproduzido somente com o Zenodo?

Não. O Zenodo é público com arquivos restritos e cobre a cascata v3.1 sobre a base inteira, não o arranjo. Seus agregados estão no repositório; as saídas por par ficam locais. A reprodução depende de acesso autorizado, campos ausentes do depósito e versões registradas. O código inicialmente registrado está fora do histórico. Nem o depósito nem a instalação do pacote demonstram validação externa.

Fonte: repositório, `gzcmd/prereg_arranjo_corte.yaml` e `gzcmd/prereg_robustez_arranjo.yaml`; disponibilidade dos dados na tese. Slides 29 e 42.

### 19. O registro interno elimina os desvios de protocolo?

Não. O código inicialmente registrado não está no git, e o suplemento fez chamadas de sensibilidade apesar da vedação original. O comparador *a posteriori* já era conhecido antes do registro; o enquadramento veio depois. Esses fatos limitam a proteção do registro e precisam ser declarados. P continua principal; não se reescreve o protocolo para tornar outra política vencedora.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`, emendas e suplemento; `data/arranjo_corte_metadata.json`. Slide 26.

### 20. Como escolher a ordenação e o ponto de parada?

Primária: probabilidade calibrada do classificador (O3); secundária: probabilidade recalibrada de P (O4); referência: escore do OpenRecLink (O1). Meta: recuperação de P, orçamento fixado antes dos desfechos. Para 23 e 9 vínculos, O4 requer 106 e 27 revisões; O3, 125 e 29. Em 96 e 17 revisões, O3 deixa 14 e 30 dos 36 fora. Para os 36, O1 requer 2.521 e O3, 23.443.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `data/comparador_mesma_carga.csv`. Slides 20 e 28.

### 21. A análise de estabilidade confirma generalização e repetibilidade?

Não. É análise interna não confirmatória, com respostas do LLM fixas. Em P na vigilância, no bloco de partição, Jaccard mediano de 0,43 em relação à partição original, FP de nove a 31 e perdas de nove a 21. P-Platt tem menor perda aplicada às medianas nos quatro cenários. A estabilidade descreve apenas aspectos internos; não valida generalização nem mede estocasticidade do revisor.

Fonte: `data/robustez_arranjo_resumo.csv:270,278,654`; `data/robustez_arranjo_metadata.json`. Slide 46.

### 22. Qual é a vantagem demonstrada frente ao processo de *linkage* de origem?

P acrescenta quatro aceitações automáticas e 21 FP na vigilância; 96 encaminhamentos contêm 19 vínculos. Não reproduz 13 e 27 vínculos nos dois modos. Não há melhora de acurácia. Frente a O3, MATCH ao humano acrescenta no máximo um vínculo na mesma carga; na confirmação, O2 recupera nove com 11 revisões, sem FP nem LLM. A carga histórica da revisão clerical de origem não foi observada.

Fonte: `data/arranjo_corte_metricas.csv:12,30`; `data/arranjo_corte_vpp_llm.csv:2,8`. Slides 19 e 28.

### 23. Houve varredura de custos no arranjo?

Não houve varredura no arranjo. Os custos C_FP/C_FN são 10/50 na vigilância e 100/20 na confirmação; C_H=1 e C_LLM=0,005. Incluindo FN, O2 custa 1.296 na vigilância para 12 vínculos e 551 na confirmação para nove. FP humano zero é hipótese, não medida. A varredura da base inteira não valida esses custos nem sua aceitabilidade para o serviço.

Fonte: `data/custos_equivalentes_exploratorios.csv:2,5,8,12`; `estudo4_gzcmd.tex:509,624`, custos e varredura na base inteira. Slides 18 e 22.

### 24. O que muda na mesma carga com MATCH do LLM ao revisor humano?

Na vigilância, P usa 121 revisões para 23 PAR=2; O3 recupera 22 nessa carga e iguala P em 125. Na confirmação, ambos recuperam 9 em 29. Revisor perfeito é hipótese dos dois lados. Há 344 pares adicionais avaliados pelo LLM na vigilância. Uma diferença de um vínculo em 36 não sustenta superioridade.

Fonte: `data/comparador_p_match_humano.csv:2,3`. Slide 22.

### 25. Há fonte para a ordem histórica de revisão?

Não. Oliveira (2026, p. 34) informa revisão manual sem ordem, limite ou parada. Ordem decrescente do escore do OpenRecLink é hipótese: 2.156 e 2.522 são cenários, não reconstrução histórica. O CdR está documentado no passo aberto, com cinco pares, não nos passos 6, 9 e 11; sua precedência à revisão nesses passos é hipótese de trabalho, sem declaração datada que a comprove.

Fonte: `capitulo_dados.tex`, discussão de Oliveira, Quadros 5 e 6; `data/carga_abaixo_corte.csv:5,9`. Slides 6 e 7.

### 26. Onde estão os falsos positivos em relação ao escore do OpenRecLink?

Na vigilância, 17 dos 21 FP de P estão em escore do OpenRecLink ≥ −5,648 e 4 abaixo; na confirmação, os 4 estão acima desse limiar. Frente ao limiar por passo, são 11 acima e 10 abaixo na vigilância; 4 e 0 na confirmação. A contagem é exploratória *post hoc*; não prova que a revisão clerical de origem examinou ou rejeitou individualmente esses pares.

Fonte: `data/fp_arranjo_por_regiao.csv:8,9,16,17,24,25,32,33`. Slides 19 e 45.

### 27. O que protege um registro feito 49 minutos antes?

Registra escolhas declaradas antes da S3, não sucesso confirmatório. Foram 49 minutos, de 01:22:17 a 02:11:52 UTC. O comparador era conhecido; faltam código original no git, critério de sucesso e carimbo independente. P-Platt virou sensibilidade na emenda 3; o critério 18/20 veio depois da avaliação. Emendas e desvios de protocolo limitam a proteção do registro, sem autorizar escolher retrospectivamente o vencedor.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`; `data/arranjo_corte_metadata.json`; `gzcmd/M_arranjo_corte_desvios.md`. Slides 45 e 49.

### 28. De onde vêm os 20 minutos e os cenários humanos?

São parâmetros de cenário, não medidas do serviço. Vinte minutos por par convertem revisões em ordem de grandeza de horas. A configuração motivou sensibilidade humana de 95%; avaliam-se também 99% e 100%. A especificidade é assumida perfeita: não se modelam FP humanos. A diferença principal é cobertura, não erro de quem revisa. A adoção requer medir tempo, erro e concordância.

Fonte: `gzcmd/gzcmd_v3_config.yaml`; `data/cenarios_sensibilidade_humana.csv`; `data/esforco_arranjo.csv`. Slides 18 e 28.

### 29. Por que Grey-Zone não é a área cinza da nota?

Grey-Zone designa a região que requer exame, não apenas [5,9) da nota do CdR. O corte do OpenRecLink usa outro escore. A proposta é preservar o corte e escolher ordenação e parada pela curva de esforço, com referência independente. Na confirmação, O2 domina fracamente P; isso não é universal. Restantes: não examinados. Filtro LLM apenas com orçamento humano fixo, sem aceitação autônoma, a validar.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `data/comparador_p_match_humano.csv`; `data/filtro_llm.csv`. Slides 4, 18 e 29.

### 30. O filtro LLM ajuda sem perder vínculos?

O protocolo fixado antes dos desfechos registrou “ajuda” na mesma carga humana, não ausência de perdas. Na vigilância k=96, H=22: sensibilidade 59,1%, VPN 87,8%; nove dos 22 vínculos descartados. A perda ponderada posterior favorece O3 com 96 revisões, outro orçamento. Na confirmação k=125, descarta dez dos 23. Respostas de datas diferentes e réplicas estocásticas não constituem validação prospectiva.

Fonte: `data/filtro_llm.csv:4,5,13`; `data/filtro_llm_metadata.json`. Slides 21 e 48.

### 31. A subamostragem sustenta preferência pela prevalência original?

Não frente a 1:20 e 1:50, indistinguíveis da referência. No topo 96, 1:1 e 1:5 degradam a recuperação: medianas de 13 e 18,5 contra 22 na prevalência original. A referência tem uma execução; cada razão, dez sementes. O Brier mediano piora em todas, mas não demonstra superioridade de ordenação da referência. A comparação é exploratória, sem evidência de melhora de equidade.

Fonte: `data/subamostragem_treino_resumo.csv:18,210,402,594,786`, variante isotonic. Slides 29 e 47.

### 32. E se a sensibilidade humana for 99%?

99% é parâmetro de cenário, não medida. O limite de 0,73 p.p. é aritmético: 5% sobre 36 dos 247 vínculos; nas cargas reduzidas, até 0,53 p.p. A ordem não se inverte na mesma carga humana. Especificidade perfeita e FP humanos zero são hipóteses, embora a política adote erro humano de 0,05. Vínculos fora do orçamento não chegam ao revisor: perda de cobertura.

Fonte: `data/cenarios_sensibilidade_humana.csv`, cenários 0,95, 0,99 e 1,00. Slide 28.

### 33. Por que não trocar o revisor humano pelo LLM?

Como aceitador, o LLM teve VPP de 16,0% na vigilância. Com MATCH encaminhado ao humano, a política recupera no máximo 1 vínculo a mais que O3 na mesma carga, com 344 pares adicionais avaliados pelo LLM. Uma diferença de um vínculo em 36 não sustenta superioridade. O uso possível é filtro experimental de NONMATCH, nunca aceitação autônoma.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `data/comparador_p_match_humano.csv:2,3`; `data/filtro_llm.csv`. Slides 21, 22 e 29.

### 34. O que aconteceu com os números da qualificação?

O E1 saneado usa Random Forest calibrado por sigmoide interna, partições por COMPREC e atributos permitidos: 39 com MACD, 30 sem. Com MACD, F1 esperado, não observado: 0,941 na vigilância e 0,916 na confirmação; reduções de revisões: 55,0% e 34,7%. A seleção histórica incluía SCORE e identificadores, vazamento corrigido: 0,954 e reduções de 14,5%/17,6% não sustentam a interpretação atual. Não houve chamada a LLM.

Fonte: `data/exp1_vazamento_comparacao.csv`; `data/ablation_comprec_allowlist_metadata.json`; `gzcmd/classifier.py`; `gzcmd/eval.py`.

### 35. O limiar do OpenRecLink é inútil hoje?

Não. O corte isolado aceita 211 dos 247 vínculos (85,4%) sem revisão, e a tese o preserva por representar o processo de *linkage* de origem e definir PAR=1 na referência, não por superioridade demonstrada sobre regras determinísticas. O pós-processamento atua abaixo dele, onde estão os 36 vínculos recuperados pela revisão; a comparação com estratégias determinísticas é discutida no Estudo 3.

Fonte: `data/arranjo_corte_metricas.csv:2`; `capitulo1.tex` e `capitulo3.tex`, justificativa do corte; `estudo3_estrategias.tex`, Discussão. Slides 6 e 19.

### 36. Há penalização de nome comum? O aprendizado de máquina é só exploração?

Sim. No CdR, a proporção de fragmentos comuns do primeiro registro entra com sinal negativo, penalizando a composição do nome, sem medir concordância. Separadamente, a guarda de homonímia da v3, ausente da v3.1, encaminha à revisão, sem excluir, pares com nota ≥ 7, subescores de nascimento e endereço nulos e diferença de anos maior que 5. Resultados com limiares escolhidos no teste são exploratórios.

Fonte: `estudo1_comparador.tex:23,39,95`, penalidade por fragmentos comuns; `gzcmd/guardrails.py:108-116,134-170`; `gzcmd/gzcmd_v3_config.yaml:131-132`; `estudo4_gzcmd.tex`, regras de guarda (`sec:e4-regras-guarda`); `capitulo2.tex`, critérios de Lucena; `estudo3_estrategias.tex`, Limitações (`sec:e3-limitacoes`). Slides 9, 11 e 13.

### 37. Existe manual de uso?

Sim, para o pacote. O Apêndice do pacote descreve requisitos e instalação; o repositório gzcmd-record-linkage traz README e o caderno `gzcmd_passo_a_passo.ipynb`, que percorre o fluxo da v3 sobre dados 100% sintéticos, com guia em `notebooks/README.md`. Não há manual operacional do serviço; a adoção exige validação em dados novos.

Fonte: `apendice_pacote.tex` (`ap:pacote`); repositório `gzcmd-record-linkage`, `README.md` e `notebooks/README.md`. Slide 29.

### 38. Quais provedores externos receberam dados?

Os dossiês LLM foram enviados a APIs externas da Anthropic, OpenCode Go, Levanto, Fireworks AI e OpenAI; no piloto, Fireworks AI serviu Kimi K2.5, DeepSeek V3.2, Qwen3-VL-30B e GLM-5, e OpenAI, GPT-4o. O QW1 transferiu o extrato pseudonimizado à Modal para o TabNet. Nenhum envio continha nomes. Para dados identificados, recomenda-se ambiente local e controlado, sem APIs externas.

Fonte: `capitulo_dados.tex`, ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`); `scripts/pilot_c2_517.py:63-84`; `scripts/qw1_modal_runner.py`. Slides 27 e 42.

### 39. O que aconteceu com o cuidado terapêutico?

Foi retirado. A tese não avalia episódios de cuidado, pensamento sistêmico nem painéis de *business intelligence*; mantém a tuberculose como condição marcadora, usada para qualificar os dados vinculados entre SIM e Sinan-TB. Os resultados descrevem decisões de vínculo, não desfechos terapêuticos.

Fonte: `estudo4_gzcmd.tex`, introdução (`cap:estudo4`); `capitulo2.tex`, A tuberculose como condição marcadora. Slide 2.

### 40. Há interface para o serviço?

O Estudo 1 mostra a captura da interface do Comparador de Registros, versão 0.1, com dados sintéticos. O CdR não é o objeto central da tese. O GZ-CMD é biblioteca Python sem interface gráfica: lê o extrato pós-CdR e produz decisões, encaminhamentos e registro de decisões, sem exercício independente de auditoria das justificativas. Não se produziu interface para o serviço.

Fonte: `estudo1_comparador.tex`, captura da interface; `apendice_pacote.tex` (`ap:pacote`); `capitulo_dados.tex`, ética dos modelos (`sec:dados-etica`). Slide 9.

### 41. Existe módulo Python do OpenRecLink? A tese usou a saída do OpenRecLink?

Não foi localizada fonte citável desse módulo; por isso, a tese não o menciona. A tese não consumiu a saída do OpenRecLink diretamente. O processo de origem utilizou OpenRecLink original, em C, CdR e revisão clerical por revisora única, sem atribuição pessoal das etapas. O autor não participou delas e recebeu apenas o extrato pós-CdR, que o GZ-CMD lê.

Fonte: `apendice_linkage_origem.tex` (`ap:linkage-origem`); `data/proveniencia_oliveira_meta.json`. Slides 8 e 9.

### 42. Os dados são públicos? Qual é a posição do autor?

O autor integra o LinkDataPop no projeto dos CAAE 43856720.1.0000.5286 e 43856720.1.3001.5279. O Zenodo é público com arquivos de acesso restrito, mediante solicitação e acordo de uso. O repositório contém agregados e escores/decisões por pseudônimo, sem *label* nem valores de campo; o pseudônimo não garante anonimização. A licença MIT vale somente para o código. A transferência internacional tem limitação ética declarada.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`); `apendice_pacote.tex`. Slides 27 e 42.

### 43. O que falsificaria a hipótese central?

Prospectivo, não testado: outra coorte, referência cega, ordenação e orçamento prévios. Probabilidade calibrada (O3) deve atingir recuperação de P com pelo menos 25% menos revisões que escore do OpenRecLink (O1), pelo limite inferior do IC 95% entre partições predefinidas, nas duas metas. Não atingir refuta a hipótese. Margem fixada com dados à vista. Quatro ações: ganho acima da amplitude entre réplicas, 9 a 21 e 21 a 29 perdas.

Fonte: `conclusao.tex`, abertura; `estudo4_gzcmd.tex`, Discussão (`sec:e4-discussao`); `data/comparador_mesma_carga.csv` como referência de desenvolvimento, não teste do critério.

### 44. O que o registro interno prova sem critério de sucesso?

Documenta escolhas declaradas em determinada versão, não comprova hipótese nem prioridade científica com carimbo independente. Não havia critério de sucesso; o comparador era conhecido, o código original falta no git e houve emendas registradas no arquivo de registro. A leitura como resultado negativo é posterior. Seriam necessários protocolo público anterior aos resultados, código recuperável e autorizações verificáveis.

Fonte: `estudo4_gzcmd.tex:316,322,326,328,672`; `gzcmd/prereg_arranjo_corte.yaml:256-259,277`; `gzcmd/M_arranjo_corte_desvios.md`.

### 45. Há validação fora desses dados?

Não. OOF e partições agrupadas são avaliações internas da mesma coorte. P reutiliza respostas das coortes que estimaram o erro do revisor; o filtro combina respostas de datas diferentes, com erros anteriormente conhecidos. A estabilidade mantém as respostas do LLM fixas. Validação externa exigiria outra coorte, referência independente, protocolo registrado e revisão humana observada, sem seleção retrospectiva da fila ou das instruções.

Fonte: `estudo4_gzcmd.tex:320,328,334,668,670`; `conclusao.tex:59,76`.

### 46. Em que direção atua o viés de verificação?

Se a revisão seguiu o escore do OpenRecLink, favoreceria O1: os 36 PAR=2 concentram-se acima de −5,6485; O1 alcança todos em 2.521 revisões. A ordem não foi documentada. Se nota e subescores do CdR orientaram a revisão, hipótese de trabalho, o favorecimento alcança O2 a O4. Não há direção única demonstrada nem magnitude estimada; faltam revisão cega e casos não verificados.

Fonte: `capitulo_dados.tex:75,77`; `conclusao.tex:43`; `data/comparador_mesma_carga.csv:5`.

### 47. Quanto do F1 próximo de 0,93 reproduz o corte do OpenRecLink?

Não foi estimada uma fração causal do F1. Há decomposição: dos 67 verdadeiros positivos do RF+SMOTE, 62 são PAR=1 e cinco PAR=2; dos 27 recuperados contra a nota hipotética 8, 25 já eram aceitos pelo corte e dois vieram da revisão. O ganho contra a nota reproduz sobretudo aceitações do corte, sob vazamento de alvo e seleção de limiar no teste.

Fonte: `data/banca_b_recuperados_por_par.json`; `estudo3_estrategias.tex:99-101`.

### 48. Os CAAE cobrem nuvem, APIs, depósito e dispensa de TCLE?

Os pareceres identificados no texto cobrem o projeto; o depósito restrito depende de solicitação e acordo de uso. A transferência internacional não foi objeto de análise específica nos pareceres. Não há cobertura documentada de cada envio à nuvem ou API, nem comprovação de dispensa de TCLE aplicável a essas operações.

Fonte: `capitulo_dados.tex:27-33`; `sec:dados-acesso-etica`; `sec:dados-etica`.

### 49. O que os intervalos do LLM e a seleção de modelos permitem concluir?

Na vigilância, VPP de 16,0% tem IC de 6,4 a 34,7%; sensibilidade do filtro de 59,1%, IC de 38,7 a 76,7%. São estimativas internas e imprecisas. Dez modelos nas mesmas coortes e instruções v4 desenhadas após v3 introduzem maldição do vencedor. Não há correção desse otimismo. US$ 1,13 das novas execuções exclui cerca de US$ 76 do Claude Sonnet 5.5.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `estudo4_gzcmd.tex:461,541,578,668`.

### 50. As 687 horas poupadas comparam a mesma meta?

Não. Comparam 96 revisões de P com cenário de 2.156 revisões, que busca todos os vínculos, não os 23 recuperáveis de P na vigilância. Para 23, O1 requer 501 revisões, O3 125; P usa 96 mais 344 pares LLM. A hipótese de 20 minutos por par não foi medida. A comparação pertinente é de mesma recuperação e orçamento, não economia histórica.

Fonte: `data/esforco_arranjo.csv:2`; `data/comparador_mesma_carga.csv:4,24`; `conclusao.tex:21`.

### 51. Quantos registros de óbito distintos há entre os vínculos?

Os 247 vínculos correspondem a 204 registros distintos do SIM; os 67 verdadeiros positivos do teste, a 60, e os 27 recuperados contra a nota 8, a 24. Vínculo é relação entre registros, não novo óbito. Os 13 e 27 PAR=2 perdidos são contagens por par. Quantos óbitos distintos representam e quantos mantêm outro vínculo aceito: não estimado nesta tese. Não se estima redução da subnotificação.

Fonte: `data/banca_b_recuperados_por_par.json:2-3,26,37`; `data/arranjo_corte_par2.csv`; `estudo3_estrategias.tex:24,99-101`.

### 52. Haneef (2022) trata de classificação de potenciais pares?

Não. Haneef et al. (2022) é citado como diretriz para uso de aprendizado de máquina com dados vinculados na estimação de indicadores (capitulo3:14), não como estudo de classificação de potenciais pares.

Fonte: `capitulo3.tex:14`; chave bibliográfica `Haneef2022guidelines`.

### 53. Quais campos o LLM recebe? Há inversão de data?

Recebe 14 dos 29 subescores: três de nome, três de nome da mãe, um de nascimento, seis de endereço e um municipal, além de compostos, nota e indicadores disponíveis. Faltam penalidade por nome comum da mãe, inversões de data e UF. A instrução v4 solicita evidências ausentes e usa nomes que não são chaves literais do dossiê. Não se mediu o efeito dessa incompatibilidade.

Fonte: `data/dossie_campos_enviados.csv`; `tab:dossie-campos`; `apendice_prompts.tex`.

### 54. A ablação prova ancoragem sem controle negativo?

Não. É compatível com ancoragem ou abstenção generalizada. DeepSeek V4.1 Flash produziu zero NONMATCH em 1.845 revisões completas com v4 sem saídas do motor. Nos 18 selecionados, sem_pcal é nome histórico: retira somente model_outputs, preservando demais campos e prompt original, com versão não identificada nos originais nem nos agregados. Sem controle negativo com não vínculos, as transições não identificam mecanismo nem efeito sobre falsos positivos.

Fonte: `data/ablacao_dossie_taxa_base.csv`; `sec:e4-ablacao-dossie`.

### 55. Com o custo dos 17,5 segundos, P-MATCH vence O3?

Não na vigilância e com decisões mantidas. O custo relativo é 17,5/1.200, aproximadamente 0,0146. P-MATCH usa 121 revisões mais 344c; O3, 125. A ordem inverte quando c supera 4/344, aproximadamente 0,0116. Com 0,0146, P-MATCH chega a 126,0, acima de 125, com a mesma recuperação. É cálculo condicional, não reexecução da política.

Fonte: `data/custos_equivalentes_exploratorios.csv`; `data/exp2_latencias_revisores.csv`; `sec:e4-discussao`.

### 56. O orçamento restringiu alguma decisão de P?

Não no cenário principal. O limite humano direto era ilimitado; os lotes LLM de 344 na vigilância e 41 na confirmação ficaram abaixo dos limites de 2.000 e 1.000. A alocação gulosa sob escassez não foi exercitada por P. A varredura na base inteira é outro exercício e não demonstra desempenho do arranjo com orçamento restritivo.

Fonte: `data/arranjo_corte_metricas.csv`; `gzcmd/gzcmd_v31_p_config.yaml`; `sec:e4-politica-v31`.

### 57. Sem referência conhecida, como escolher fila e parada no serviço?

As curvas retrospectivas não resolvem essa escolha sem supervisão. Seriam necessários orçamento e custos acordados com o profissional do serviço, amostra independente revisada às cegas e avaliação de filas e pontos de parada fora dos dados usados para escolhê-los. Uma probabilidade validada permitiria examinar o limiar de custo. Esse procedimento é proposta prospectiva, não implantação realizada nem escolha validada pela tese.

Fonte: `sec:e4-mesma-carga`; `sec:concl-recomendacoes`; Proposição de duas ações.

### 58. Quantos FP repetem COMPREC ou REFREC já aceito?

Dos 21 FP de P na vigilância, um compartilha REFREC com um PAR=1 aceito acima do corte, nenhum compartilha COMPREC e 20 não compartilham nenhum dos dois. Na confirmação, nenhum dos quatro FP compartilha esses identificadores com essa referência. Cardinalidade N:1 e margem de conflito não foram implementadas; a contagem não examina conflitos entre novas aceitações nem estima quantos erros essa regra evitaria.

Fonte: `data/fp_cardinalidade.json`; `sec:e4-regras-guarda`.

### 59. Se o LLM não conhece o modo, o que diferencia os lotes do filtro?

São réplicas estocásticas: mesma fila e instruções, respostas de fontes e datas diferentes, sem modo informado. No topo 125, 119 decisões coincidem e seis diferem, todas entre não vínculos. Em k=96, H é 22 e 18, respectivamente. A diferença de carga não demonstra efeito de um perfil decisório; H=22 pertence a uma réplica, não é propriedade estável do filtro.

Fonte: `data/filtro_llm_concordancia_modos.csv`; `data/filtro_llm.csv`; `gzcmd/M4_prompt_v4.md`.

### 60. Probabilidade que varia com threads é implantável?

Não se demonstrou prontidão para implantação. A mudança máxima de P foi 0,65 no bloco numérico e 0,80 entre partições; semente e threads foram variados conjuntamente no primeiro. Fixar ambiente, versões e paralelismo é necessário para reprodução, mas não resolve a instabilidade entre amostras. Seria preciso validar decisões e limiares em dados novos antes de qualquer implantação.

Fonte: `data/robustez_arranjo_resumo.csv`; `sec:e4-robustez`; `gzcmd/M_arranjo_corte_desvios.md`.

### 61. Em 2027, qual versão do modelo reproduziria o resultado?

Não há garantia de reprodução por alias. Foram registrados nomes e parâmetros, não snapshots imutáveis dos provedores. Quotas e descontinuação restringem o acesso; Kimi K3 não completou a coorte por esgotamento de cota. Retentativas após truncamento ou JSON inválido também reamostram a geração. As latências históricas não medem Haiku v4 sem saídas do motor. Uma versão futura exigiria nova avaliação.

Fonte: `sec:e4-selecao-llm`; `sec:e4-limitacoes`; `scripts/run_exp2_anthropic.py`.

### 62. Quem fixou C_FN=50? E se fosse 25?

O autor fixou custos como preferências de decisão, não valores econômicos medidos. Na vigilância, C_FP/C_FN é 10/50; na confirmação, 100/20. Não houve varredura dessa razão no arranjo. Reduzir C_FN para 25 alteraria a comparação de perdas e poderia mudar ações; não se dispõe dessa reexecução. A aceitabilidade dos custos precisa ser discutida com o profissional do serviço.

Fonte: `gzcmd/gzcmd_v31_p_config.yaml`; `sec:e4-politica-v31`; `sec:e4-limitacoes`.

### 63. grey_low cobre 3 a 5 ou 5 a 6?

No motor, cobre [5,6). As demais faixas são low abaixo de 5, grey_mid [6,7), grey_high [7,8), near_high [8,9) e high a partir de 9. Os intervalos diferentes do arquivo v3 não foram enviados pelo carregador e não redefinem o motor. Área cinza da nota designa somente [5,9); região abaixo do corte usa o escore do OpenRecLink.

Fonte: `sec:e4-politica-v31`; `gzcmd/gzcmd_v3_config.yaml`; `apendice_prompts.tex`.

### 64. Quais as consequências de FN e FP no fluxo da vigilância do óbito?

FN faz um óbito de pessoa notificada parecer não notificado: pode gerar investigação desnecessária, notificação pós-óbito duplicada e manutenção do encerramento anterior. FP pode encerrar como óbito a notificação de outra pessoa e fazer parecer notificado um óbito sem notificação. A investigação não depende do vínculo. Em P, 19 dos 21 FP da vigilância atingem notificações não encerradas por óbito.

Fonte: `capitulo1.tex`, vigilância do óbito; `data/equidade_arranjo_group_metrics.csv`; Oliveira (2026), p. 48.

### 65. Qual seria o custo anual em comparação à rotina da SMS-Rio?

Não estimado nesta tese. Os cenários de carga referem-se aos óbitos de 2019 a 2022, não a um ciclo anual. Os 20 minutos por par são hipótese, não medida. As 718 h 40 min e 840 h 40 min não foram comparadas ao volume de investigação já realizado no município; não constituem economia anual observada nem custo orçamentário do serviço.

Fonte: `data/carga_abaixo_corte_horas.csv`; `data/esforco_arranjo.csv`; `capitulo_dados.tex`, período do SIM.

### 66. Os 13 e 27 PAR=2 perdidos representam quantos óbitos sem outro vínculo?

Não estimado nesta tese. São potenciais pares perdidos, não contagens de óbitos distintos. Um óbito pode manter outro vínculo aceito. Não foi produzida a contagem de REFREC distintos desses perdidos nem sua interseção com os vínculos mantidos; não se pode converter essas perdas em óbitos sem notificação associada.

Fonte: `data/arranjo_corte_par2.csv`; `data/banca_b_recuperados_por_par.json`, cardinalidade do extrato.

### 67. Como conciliar 21,4% de subnotificação com 204 entre 1.515?

Oliveira encontrou 205 dos 959 óbitos por TB pulmonar sem notificação, 21,4%, no Rio de Janeiro entre 2019 e 2022. Os 204 são óbitos vinculados entre os 1.515 com candidato nos três passos exportados, após retirada de vínculos determinísticos, de um processo com 29 passos probabilísticos. Essa razão não estima subnotificação: população, seleção e etapa do processo são diferentes.

Fonte: Oliveira (2026), p. 44; `data/extrato_resumo.csv`; `data/proveniencia_oliveira_meta.json`; `data/banca_b_recuperados_por_par.json`.

### 68. A avaliação abrange causa associada, TB-HIV e toda a base do SIM?

Não. O extrato inclui TB como causa básica, não menções como causa associada, inclusive TB-HIV. O desempenho nesse grupo é desconhecido. Os 1.515 registros do SIM têm ao menos um potencial par nos passos exportados; não são toda a base do SIM. A vigilância do óbito com menção de TB tem escopo mais amplo que o avaliado.

Fonte: `data/extrato_resumo.csv`; `data/proveniencia_oliveira_meta.json`; `capitulo_dados.tex`, fontes e escopo; Oliveira (2026), protocolo de vigilância.

### 69. A exclusão progressiva mudaria os candidatos se P decidisse desde o passo 6?

Pode mudar. Os registros do Sinan já pareados são retirados dos passos seguintes; os candidatos dos passos 9 e 11 dependem das decisões anteriores na origem. P foi avaliado nesse universo fixo. Não se reconstruíram os candidatos que surgiriam ou desapareceriam sob suas decisões anteriores. O efeito de substituir o processo sequencial completo não foi estimado nesta tese.

Fonte: `apendice_linkage_origem.tex`, exclusão progressiva; `data/par_por_passo.csv`; `capitulo_dados.tex`, limites do extrato.

### 70. Por que foram escolhidos os passos 6, 9 e 11? Há transferência demonstrada?

O critério de escolha não foi informado. O extrato disponível contém esses passos e 247 vínculos, de 2.549 pares no processo de origem. Não se demonstrou representatividade dos passos nem transferência para os demais, para a etapa determinística ou para outros municípios. A avaliação é condicionada aos candidatos recebidos, não mede a sensibilidade de toda a blocagem.

Fonte: `data/proveniencia_oliveira_meta.json`, `nao_informados` e `total_pares_processo`; `data/extrato_resumo.csv`.

### 71. A análise por sexo detecta erros da blocagem por sobrenome e sexo?

Não. Os três passos exigem Soundex do último fragmento do nome; 9 e 11 também exigem sexo concordante. Sobrenome alterado ou sexo discordante pode impedir a geração de candidato. A análise restrita a vínculos conhecidos não detecta essa perda diferencial. Não foi relatada comparação de óbitos vinculados e não vinculados; o efeito sobre mulheres não foi estimado nesta tese.

Fonte: `apendice_linkage_origem.tex`, chaves de blocagem; `estudo5_equidade.tex`, limitações.

### 72. A vantagem observada da ordenação vale para qualquer meta de recuperação?

Não. Na vigilância, para os 23 PAR=2 recuperáveis de P, O3 requer 125 revisões, O4 106 e O1 501. Para todos os 36, O1 requer 2.521 e O3 23.443. O2 requer 4.163 para 23. Esses resultados internos não demonstram vantagem universal da probabilidade nem da nota do CdR; a hipótese é gerada nos dados, não confirmada externamente.

Fonte: `data/comparador_mesma_carga.csv`, metas de recuperação na vigilância.

### 73. Os 21 FP são erros de identidade ou podem ser pares não examinados?

São discordâncias com a referência operacional, não erros de identidade confirmados independentemente. PAR=0 inclui rejeitados ou não examinados. Dez dos 21 estão abaixo do limiar por passo, onde a revisão não está documentada. Se algum for vínculo não verificado, o VPP de 16,0% estará subestimado. A direção e a magnitude desse efeito não foram estimadas nesta tese.

Fonte: `data/fp_arranjo_por_regiao.csv`; `data/arranjo_corte_vpp_llm.csv`; `capitulo_dados.tex`, referência operacional.

### 74. Quem responde por um óbito não investigado após decisão automatizada?

A responsabilidade não é transferida ao modelo. O profissional do serviço e a gestão precisam definir supervisão e encaminhamento; a tese não estabeleceu protocolo de responsabilização. Não examinado por orçamento não significa óbito não notificado nem autoriza excluir a investigação. O número de óbitos que ficariam sem investigação e a responsabilidade jurídica por casos concretos não foram estimados nesta tese.

Fonte: `capitulo3.tex`, uso no serviço; `conclusao.tex`, recomendações e limitações.

### 75. Qual estimador e quais atributos produziram o E1 histórico e o saneado?

Random Forest com 200 árvores, profundidade 10, pesos balanceados e calibração sigmoide em validação cruzada interna no treino, não Platt da nota. A seleção histórica usou 100 atributos com MACD, 91 sem, incluindo SCORE e identificadores. O saneamento restringiu a 39 e 30. Retiraram-se conjuntamente campos brutos, identificadores e aliases; não se isolou causalmente o efeito do SCORE nem se provou ausência universal de vazamento.

Fonte: `gzcmd/classifier.py`; `gzcmd/eval.py`; `data/ablation_comprec_allowlist_metadata.json`; `data/exp1_vazamento_comparacao.csv`.

### 76. Quanto das 27 perdas decorre da hipótese de 5% de FP humano?

A fração não foi estimada no arranjo. Com C_FP=100 e e_FP,h=0,05, L_H=6−4p: revisão supera não vínculo para p>0,25, contra p>0,05 sem erro humano. Na base inteira, essa hipótese reduz encaminhamentos diretos de 229 para 121 e sensibilidade esperada de 92,3% para 87,7%, com C_LLM=0,005. A avaliação assume especificidade humana perfeita, assimetria não medida.

Fonte: `data/varredura_custos_confirmacao.csv`; `gzcmd/gzcmd_v31_p_config.yaml`; `data/cenarios_sensibilidade_humana.csv`.

### 77. O teste t pareado é adequado com partições sobrepostas?

As cinco partições reutilizam a mesma amostra e seus testes se sobrepõem. Parear MACD com e sem na mesma partição não torna as rodadas independentes. Os valores de p podem ser otimistas e são exploratórios, sem correção por múltiplas comparações. O saneamento removeu atributos indevidos, mas não resolveu essa dependência. Inferência confirmatória exigiria desenho e referência independentes.

Fonte: `data/exp1_testes_t_allowlist.csv`; `data/ablation_comprec_allowlist_metadata.json`; `scripts/exp1_testes_t.py`.

### 78. As proposições demonstram a vantagem da política de quatro ações?

Não. As proposições exigem probabilidade verdadeira e calibrada, perdas aditivas e hipóteses específicas sobre revisão. A regra sob orçamento trata de duas ações, revisor humano e não vínculo. Sua coincidência com ordenação por probabilidade não comprova a política de quatro ações, com guardas e proporções de erro por faixa. A avaliação empírica dessas quatro ações permanece necessária e não demonstrou superioridade.

Fonte: `apendice_formalizacao.tex`, hipóteses e regra sob orçamento; `data/comparador_p_match_humano.csv`.

### 79. Que conceito de equidade foi avaliado? Disparidade é iniquidade?

Equidade em saúde envolve evitar diferenças injustas e evitáveis. A tese mediu disparidades de sensibilidade e VPP em estratos demográficos e operacionais da referência disponível; não demonstrou que cada diferença fosse evitável ou injusta. Raça/cor não estava disponível. Ausência de rejeição estatística não comprova equidade, e diferença por passo não equivale a iniquidade em saúde. Essa classificação normativa não foi estimada nesta tese.

Fonte: `estudo5_equidade.tex`, objetivo e limitações; `data/equidade_arranjo_den30_recomputacao.json`.

### 80. Por que v3.1 e P não usam MACD no calibrador se o E1 saneado reduz 55% das revisões?

Na sensibilidade *post hoc*, MACD, medidas contínuas de diferença de datas, exigiram 46/118/3.040 revisões para nove/23/36 vínculos, contra 29/125/23.443 sem elas; a precisão média (AP) caiu de 0,283 para 0,261. Não há melhora uniforme nem substituição retrospectiva da fila. A redução de 55,0% do E1 não se transfere: não foi avaliado nesta tese no arranjo recalibrado. Validar essa extensão continua como trabalho futuro.

Fonte: `data/exp1_vazamento_comparacao.csv`; `data/pcal_ml_oof.json`; `data/arranjo_corte_metadata.json`.

### 81. Combinar as ordenações não seria melhor?

Na análise *post hoc*, F3 recuperou 36 com 1.777 revisões e 23 com 64; F4, 36 com 1.387, contra 2.521 pelo escore. O *bootstrap* agrupado favorece F4 frente ao escore nas metas nove e 36, não 23. Frente à probabilidade calibrada, não demonstrou vantagem. Recomenda-se F4 no lugar do escore, a validar com referência cega; parte do ganho depende da faixa verificada.

Fonte: `data/fusao_filas.csv`, F3/F4/O1; `data/fusao_robustez_bootstrap.csv`; `data/fusao_robustez_faixa.csv`.

### 82. O algoritmo consegue escolher a ordenação sozinho?

A fila adaptativa aprende com as próprias revisões, sem consultar respostas futuras. Para recuperar 23 vínculos, a média foi de 96,38 revisões, com percentis P10 a P90 de 85 a 114, contra 125 pela probabilidade calibrada. Esses percentis descrevem repetições, não intervalos de confiança. Explorar ordenações menos produtivas também consome revisões; a vantagem depende da meta e requer validação, pois a análise é *post hoc* e presume revisor perfeito.

Fonte: `data/fusao_filas.csv`, A1/A2, `revisoes_23`; `data/fusao_filas_metadata.json`.

### 83. Dá para dispensar o revisor humano?

Não como conclusão desta tese. Uma operação sem revisor humano no *linkage* seria possível apenas sem encerramento automático: vínculos prováveis seguiriam à investigação do óbito pela vigilância, com validação humana pontual por amostra. A responsabilidade do serviço permanece, e essa operação não foi avaliada. A coorte nova deve avaliar a referência amostral, a correção dos *labels* de LLM e a regra de parada antes de qualquer adoção.

Fonte: `text/latex/conclusao.tex`, seção Trabalhos futuros; proposta não avaliada.

### 84. Um modelo maior resolveria?

Não demonstrado. Claude Haiku 5.5, Claude Sonnet 5.5, Claude Opus 5.5 e Claude Fable 5.1 tiveram VPP de 16,0%, 20,8%, 19,0% e 21,7% na vigilância do probe. Sonnet cobre 139 dossiês; os demais, 150. Houve mudanças de abstenção, sem resolver a aceitação autônoma. Dossiê e referência limitam a interpretação, não apenas tamanho; o custo por par dos maiores foi muito superior, sem comparação de preços controlada.

Fonte: `data/probe_frontier_agregado.csv`, TOTAL_AMOSTRA; `data/probe_frontier_custos.csv`; `data/probe_haiku_sonnet_coorte.csv`; tabela de apoio abaixo.

### 85. E se a referência estiver errada?

C1: VPP de 50,0% requer reclassificar 40,48% dos FP uniformemente, sem retirar PAR=2; na grade, 50,0%. C2: 13 e 27 perdas mudam com 5,0% de remoção; a 20,0%, restam 10,4 e 21,6. C3, O3 com margem de 25%, e C4, F4 frente a O1, mantêm medianas nos cenários; C3 cruza zero no P97,5 de um cenário. Revisão cega independente, impossível antes da defesa, integra o protocolo da nova coorte.

Fonte: `data/vies_referencia.csv`; `data/vies_referencia_viradas.csv`; soluções algébricas e convenções em `docs/plans/RELATORIO_VIES_REFERENCIA_2026-11-07.md`.

### 86. A fusão é melhor que a probabilidade calibrada?

Não demonstrado. Nenhum intervalo de 95,0% das diferenças F4 menos O3 ficou inteiramente abaixo de zero no *bootstrap* agrupado por óbito. Frente ao escore, as medianas foram 38 e 1.203 revisões a menos nas metas nove e 36, com intervalos negativos, mas não na meta 23. Parte do ganho na cauda depende da faixa verificada. São probabilidades e referência fixas, com metas híbridas; não validação independente.

Fonte: `data/fusao_robustez_bootstrap.csv`; `data/fusao_robustez_faixa.csv`; `data/fusao_robustez_metadata.json`.

## Figuras e critérios

### Apoio às respostas 84 a 86

Os custos são estimativas por uso e preço, não fatura nem comparação controlada: Haiku usa Batch no probe, Opus e Fable combinam Batch e chamadas diretas, e Sonnet reutiliza a coorte histórica direta. O denominador conta decisões válidas por passada, incluindo réplicas, não pares únicos. Não se atribui causalmente o desempenho ao tamanho do modelo.

| Modelo | Vigilância do probe: VP/FP | VPP (%) | Custo por par revisado (US$) | Escopo do custo |
| :-- | :-- | --: | --: | :-- |
| Claude Haiku 5.5 | 4/21 | 16,0 | 0,0014 | probe |
| Claude Sonnet 5.5 | 5/19 | 20,8 | 0,0360 | coorte histórica |
| Claude Opus 5.5 | 4/17 | 19,0 | 0,0371 | probe |
| Claude Fable 5.1 | 5/18 | 21,7 | 0,1082 | probe |

Fonte: `data/probe_frontier_agregado.csv`, TOTAL_AMOSTRA; `data/probe_frontier_custos.csv`. Na coorte original comum Haiku/Sonnet, a concordância foi de 86,9% na vigilância (1.410 dossiês) e de 85,5% na confirmação (435), sem demonstração de superioridade do maior (`data/probe_haiku_sonnet_coorte.csv`, original/TOTAL).

Figuras reaproveitadas, sem alterar os originais:

| Slide | Arquivo | Uso e ressalva |
| :-- | :-- | :-- |
| 20 | `text/slides/figures/curva_esforco.pgf` | Curva exploratória por script, com pontos de P |
| 9 | `figures/fig_cdr_gui_mapeamento.png` | Interface do Comparador de Registros, versão 0.1 (*commit* 3b43598), dados sintéticos |
| 31 | `text/latex/figures/fig_cap8_cascata_v31.png` | Cascata por REFREC sobre a base inteira (comparação) |
| 36 | `figures/imbalance_sensitivity.pgf` | Estratégias de balanceamento no Random Forest |
| 37 | `text/latex/fig_doc/epi_deteccao.pgf` | Detecção no teste da ablação, não arranjo nem cascata |

A captura do slide 9 está em `../../figures/fig_cdr_gui_mapeamento.png`, caminho relativo ao diretório de compilação `text/slides`, sem `graphicspath` adicional. A tabela do slide 25 reduz a tabela R2 e incorpora Paiva (2025) do texto adjacente, indicando que utilizou base distinta da desta tese, sem lhe atribuir a autoria do protótipo Java.

Critérios editoriais: menção ao referencial clássico restrita ao slide 8; probabilidade recalibrada abaixo do corte; arranjo P principal e cascata REFREC sobre a base inteira como comparação; corte operacional 211/247 por construção do *label*; curva de esforço exploratória *post hoc*, não registrada; F1 0,943 do XGBoost isolado e 0,958 do híbrido OR no fluxo Modal; perdas de 13 e 27 vínculos junto do acréscimo automático; limitações antes das conclusões; equivalência Java parcial executada e comparação com Lages não realizada; sensibilidade 2×2 e QW1 local distinguidos dos resultados históricos; exportador com HMAC implementado, matriz não depositada; área cinza da nota apenas para [5,9). As tabelas foram reduzidas no arquivo de slides.

Os quadros de reserva apoiam a discussão de métodos e limites. As remissões usam seções e agregados, não a paginação do PDF.

### Rastreabilidade dos números do arranjo

Linhas incluem o cabeçalho. A coluna `publicada` dos CSVs designa a cascata sobre a base inteira (comparação). Percentuais e custos foram arredondados apenas para apresentação.

| Número ou conjunto de números | CSV:linha em `data/` |
| :-- | :-- |
| 61.696, 247, corte 211/247, 85,4% por construção, 36 abaixo | `arranjo_corte_metricas.csv:2` |
| Até 61.485 potenciais pares na revisão integral, diretamente do campo `pares_humano` | `arranjo_corte_metricas.csv:8` |
| Vigilância P: 211 pelo corte e 4 pela política, total 215; 87,0%, 21 FP, 13 perdidos, 19 vínculos em 96 encaminhamentos, 344 LLM, US$ 0,94, sensibilidade humana de 95%, 99% e 100%: 94,35%, 94,66% e 94,74% | `arranjo_corte_metricas.csv:2,12`; `arranjo_corte_par2.csv:7`; `cenarios_sensibilidade_humana.csv:5-7` |
| Confirmação P: 211 pelo corte e 8 pela política, total 219; 88,7%, 4 FP, 27 perdidos, 1 vínculo em 17 encaminhamentos, 41 LLM, US$ 0,11, sensibilidade humana de 95%, 99% e 100%: 89,05%, 89,06% e 89,07% | `arranjo_corte_metricas.csv:20,30`; `arranjo_corte_par2.csv:16`; `cenarios_sensibilidade_humana.csv:44-46` |
| Fechamentos (211+4)+13+19=247 e (211+8)+27+1=247 | `arranjo_corte_metricas.csv:2,12,20,30` |
| PAR=2, aceitos/humano/perdidos: P 4/19/13 contra cascata 10/17/9; P 8/1/27 contra cascata 8/11/17 | `arranjo_corte_par2.csv:3,7,12,16` |
| Revisão integral de até 61.485: 99,27%, 99,85% e 100,00%, nos cenários humanos de 95%, 99% e 100% | `cenarios_sensibilidade_humana.csv:2-4` |
| Nenhum MATCH de P pelo motor abaixo do corte | `arranjo_corte_s1_resumo.csv:2-15,78-88` |
| Aceitações do LLM: 25 (4+21) e 12 (8+4) | `arranjo_corte_s1_resumo.csv:2-15,78-88` e `arranjo_corte_metricas.csv:2,12,30` |
| Área cinza da nota [5,9): 21.620, 151, corte 123; P 127 nos dois modos, 4 além do corte; base inteira 113/65 | `arranjo_corte_metricas.csv:3,5,13,23,31` |
| Base inteira v3.1: 194/34/15/38/111/347, 78,5%, IC 73,0 a 83,2%, US$ 0,92 | `arranjo_corte_metricas.csv:4` |
| Base inteira v3.1: 132/0/28/87/127/44, 53,4%, IC 47,2 a 59,6%, US$ 0,12 | `arranjo_corte_metricas.csv:22` |
| P-Platt: 217/3/77/110, US$ 0,30 e 13 perdidos; 219/3/16/25 | `arranjo_corte_metricas.csv:14,32` |
| A3: 219/19/100/367 e 219/5/22/32 | `arranjo_corte_metricas.csv:16,34` |
| P-PAR2-grey: 211/21/116/265 e 217/0/22/18 | `arranjo_corte_metricas.csv:18,36` |
| B1: 194/40/115/374 e 131/4/131/70 | `arranjo_corte_metricas.csv:6,24` |
| A posteriori, sem registro interno: 221/34/90/330 e 219/0/51/25; 9 perdidos na vigilância contra 13 de P; 17 na confirmação | `arranjo_corte_aposteriori.csv:4,12`; `arranjo_corte_metricas.csv:10,12` |
| VPP feminino/masculino em P: 86,3/93,3%; base inteira (comparação): 87,9/84,0% | `equidade_arranjo_group_metrics.csv:2,3,26,27` |
| Sensibilidade no passo 11 em P: 42,5/47,5%; passos 6/9: 95,0/96,1% e 96,3/96,9% | `equidade_arranjo_group_metrics.csv:17-19,65-67` |
| BH após mínimo de 30 nos dois denominadores: P vigilância 3/26 | `equidade_arranjo_den30_recomputacao.json:10-13` |
| BH após o mesmo filtro: P confirmação 2/26 | `equidade_arranjo_den30_recomputacao.json:18-21` |
| BH após o mesmo filtro, base inteira: vigilância 6/28 e confirmação 1/21 | `equidade_arranjo_den30_recomputacao.json:26-29,34-37` |
| 211 aceitos por construção; diferenças de sensibilidade dependem da proporção de PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36; diferenças de VPP dependem dos 21 e 4 FP abaixo do corte | `arranjo_corte_metricas.csv:2,12,30`; `equidade_arranjo_group_metrics.csv`; `equidade_arranjo_global.csv:2,4` |
| Na vigilância, não aceitos por passo 11/6/9: 23/4/5, todos PAR=2; no passo 11, 42,5% em P contra 45,0% na base inteira (comparação) | `equidade_arranjo_group_metrics.csv:17-19,41` |
| Passo 11: 25 dos 36 vínculos com PAR=2; P aceita 2/25 na vigilância e 4/25 na confirmação | `par_por_passo.csv:8-10,17-19,26-28,53-55` |
| Corte isolado nos passos 11/6/9: 15/40 (37,5%), 76/80 (95,0%) e 120/127 (94,5%) | `par_por_passo.csv:5-10,14-19,23-28` |
| PAR=2 aceitos nos passos 6/9/11: 0/2/2 na vigilância e 1/3/4 na confirmação | `par_por_passo.csv:8,17,26,35,44,53` |
| Pendentes de P∪B1 com chamada nova: 10 na vigilância e 14 na confirmação, não de P isolado; reutilização histórica e ausência de avaliação prospectiva | `arranjo_corte_s1_metadata.json:477-481`, campo `pending_P_union_B1_by_mode`; `arranjo_corte_metadata.json:355,357` |
| Cenários de carga sob ordem hipotética: 2.156 por passo e 2.522 global; escore do OpenRecLink global ≥ −5,6485, sob hipótese não documentada de revisão decrescente de escore do OpenRecLink | `carga_abaixo_corte.csv:5,9`; precisão em `fusao_robustez_metadata.json` |
| Hipótese de 20 min por par: por passo, pelo menos 718 h 40 min; global, pelo menos 840 h 40 min; não são tempos observados | `carga_abaixo_corte_horas.csv:3,2` |
| P, vigilância: VPP do LLM 4/25 = 16,0%, IC 6,4 a 34,7%; 9,06 FP previstos contra 21 observados | `arranjo_corte_vpp_llm.csv:2` |
| P, confirmação: VPP 8/12 = 66,7%, IC 39,1 a 86,2%; P-Platt vigilância: 4/7 = 57,1%, IC 25,0 a 84,2% | `arranjo_corte_vpp_llm.csv:8,3` |
| Perdas de P por guarda/custo/LLM: 1/6/6 na vigilância e 1/22/4 na confirmação | `arranjo_corte_perdas_mecanismo.csv:2-4,50-52` |
| Perda de 13/36 = 36,1% e 27/36 = 75,0% dos PAR=2 | `arranjo_corte_par2.csv:7,16` |
| Dominância de P-Platt: vigilância nos blocos numérico (N)/partição (R), 7/20 e 4/20; confirmação 7/20 e 5/20; inversa 0/20 em todos os casos; FP menores em 20/20 nos dois blocos da vigilância | `robustez_arranjo_resumo.csv:2-5` |
| Vigilância: P-Platt perde em mediana 1/2 vínculos a mais, blocos numérico (N)/partição (R) | `robustez_arranjo_resumo.csv:404,488`, diferenças P menos P-Platt −1/−2 |
| P perde mais PAR=2 que o *a posteriori* em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação | `robustez_arranjo_resumo.csv:6,8,10,12`, proporções multiplicadas por 20 |
| P na vigilância, bloco de partição (R): FP de 9 a 31, perdas de 9 a 21 e Jaccard mediano 0,43 em relação à partição original | `robustez_arranjo_resumo.csv:270,278,654` |
| VPP por sexo de P na vigilância sem rejeição na família den30: p ajustado BH de 0,360 | `equidade_arranjo_den30_pairwise_tests.csv:3` |
| Mudança máxima da probabilidade de P: 0,65 no bloco numérico (N) e 0,80 no bloco de partição (R) | `robustez_arranjo_resumo.csv:632,652` |
| Estabilidade: 242 potenciais pares novos, US$ 0,6255, arredondados para US$ 0,63 | `robustez_arranjo_metadata.json:206,208` |
| Revisões humanas equivalentes na vigilância, sob revisor perfeito: P 307,72; O3 125; P com MATCH ao humano 122,72 | `custos_equivalentes_exploratorios.csv:2-4` |
| Perdas totais, com FN: vigilância P 957,72, O3 775, P-MATCH 772,72, O2 na mesma carga 1.296 (12 vínculos); confirmação P 957,2, O3 569, P-MATCH 569,2 e O2 551 (nove vínculos) | `custos_equivalentes_exploratorios.csv:2-5,8-10,12` |
| C_FP/C_FN por modo: vigilância 10/50, confirmação 100/20; C_H=1 e C_LLM=0,005 | `custos_equivalentes_exploratorios.csv:2,8` |
| Diferenças P menos P-Platt de perda aplicada às medianas marginais: 144,1/94,4 na vigilância e 101,6/143,0 na confirmação; variação máxima mediana de p no bloco numérico: 0,46/0,17 | `robustez_perda_ponderada.csv:2-9` |
| Filtro em k=96, H=22: sensibilidade 13/22 = 59,1%, IC de Wilson 38,7 a 76,7%; 9 de 22 descartados; VPN 87,8%, IC 78,5 a 93,5%; perda 1.172,5 contra O3 796 | `filtro_llm.csv:5`; `estudo4_gzcmd.tex:578`, cálculo da tese; `custos_equivalentes_exploratorios.csv:7` |
| Recuperados contra nota 8: 25 PAR=1 e dois PAR=2; 24 registros distintos; 247 vínculos correspondem a 204 registros distintos; VP do RF+SMOTE: 62 PAR=1, cinco PAR=2, 60 registros distintos | `banca_b_recuperados_por_par.json:2-3,18-37` |
| Fluxo Modal: XGBoost isolado 0,943; OR 0,958 (68/0/6); AND, regras seguidas de ML e consenso com regras 0,896; ML seguido de regras 0,825 (52/0/22) | `ablation_results.csv:74,87,91,100,103`; `text/latex/tables/tab_ablation_best_category.tex:14-20` |
| Execuções novas: US$ 1,13 (0,2448 do arranjo inicial + 0,6255 da análise de estabilidade + 0,2633 do filtro LLM = 1,1336), sem representar fatura e sem pessoal e infraestrutura | `arranjo_corte_metadata.json:285,293,330,338`, soma dos custos novos de S2 e suplemento; `robustez_arranjo_metadata.json:208`; `filtro_llm_metadata.json:171`, `new_execution.total_cost_estimate_usd`; `estudo4_gzcmd.tex:539` |

Os resultados históricos de v3, dos estudos 1 a 3 e da restrição na base inteira têm escopos distintos dos resultados do arranjo, conforme as fontes indicadas nos respectivos quadros.

### Comparação exploratória por orçamento

Análises exploratórias *post hoc*, não registradas. A curva não valida uma política. O3 e O4 usam o mesmo XGBoost OOF; revisor perfeito é hipótese; os cenários humanos consideram 95%, 99% e 100% de sensibilidade, com especificidade perfeita assumida. Se a revisão histórica seguiu o escore, o viés de verificação favoreceria O1.

| Modo | Ordenação | Revisões K1 | PAR=2 em K1 | k mínimo para igualar P | PAR=2 de P com revisor perfeito | FP de P | CSV:linha |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| vigilancia | O1_SCORE | 96 | 19 | 501 | 23 | 21 | `comparador_mesma_carga.csv:2` |
| confirmacao | O1_SCORE | 17 | 3 | 51 | 9 | 4 | `comparador_mesma_carga.csv:7` |
| vigilancia | O2_CdR | 96 | 12 | 4163 | 23 | 21 | `comparador_mesma_carga.csv:12` |
| confirmacao | O2_CdR | 17 | 9 | 11 | 9 | 4 | `comparador_mesma_carga.csv:17` |
| vigilancia | O3_p_cal_ml | 96 | 22 | 125 | 23 | 21 | `comparador_mesma_carga.csv:22` |
| confirmacao | O3_p_cal_ml | 17 | 6 | 29 | 9 | 4 | `comparador_mesma_carga.csv:27` |
| vigilancia | O4_P_recalibrada | 96 | 22 | 106 | 23 | 21 | `comparador_mesma_carga.csv:32` |
| confirmacao | O4_P_recalibrada | 17 | 5 | 27 | 9 | 4 | `comparador_mesma_carga.csv:37` |

### Esforço, somente como cenário

A referência de 2.156 revisões pressupõe ordem decrescente de escore do OpenRecLink, que não foi documentada, e meta diferente da recuperação de P. Para os mesmos 23 vínculos na vigilância, O1 requer 501 revisões e O3 125, contra 96 de P mais 344 pares LLM (`comparador_mesma_carga.csv:4,24`). Horas são ordem de grandeza a 20 min por par, não medida. As 687 horas poupadas abaixo descrevem somente aquele cenário, não economia histórica nem comparação de mesma recuperação. Para O3, FP é zero sob revisor perfeito; não se divide por zero. K2 equipara avaliações totais de P, não orçamento humano.

| Modo | Braço | Carga | Revisões | Horas | Horas poupadas | Horas poupadas por vínculo não recuperado | Horas poupadas por FP | CSV:linha |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- | :-- |
| vigilancia | P | K1_humana | 96 | 32 | 687 | 52,82 | 32,70 | `esforco_arranjo.csv:2` |
| vigilancia | O3 | K1_humana | 96 | 32 | 687 | 49,05 | não se aplica | `esforco_arranjo.csv:3` |
| vigilancia | P | K2_total | 96 | 32 | 687 | 52,82 | 32,70 | `esforco_arranjo.csv:4` |
| vigilancia | O3 | K2_total | 440 | 147 | 572 | 57,20 | não se aplica | `esforco_arranjo.csv:5` |
| confirmacao | P | K1_humana | 17 | 6 | 713 | 26,41 | 178,25 | `esforco_arranjo.csv:6` |
| confirmacao | O3 | K1_humana | 17 | 6 | 713 | 23,77 | não se aplica | `esforco_arranjo.csv:7` |
| confirmacao | P | K2_total | 17 | 6 | 713 | 26,41 | 178,25 | `esforco_arranjo.csv:8` |
| confirmacao | O3 | K2_total | 58 | 19 | 699 | 43,71 | não se aplica | `esforco_arranjo.csv:9` |

### FP por região do escore do OpenRecLink, análise exploratória

Global: limiar operacional −5,6485 para os 36 PAR=2, incluindo o bloco com escore aproximadamente −5,64846; o literal −5,648 seleciona 2.495 candidatos e 31 PAR=2. As contagens de FP da tabela preservam o limiar literal da análise original. Por passo: limite compatível com PAR=2 naquele passo. Acima ou igual e abaixo são posições no escore, não prova de exame histórico pela revisão clerical de origem. Fonte da precisão: `data/fusao_robustez_metadata.json`.

| Modo | Limiar | Região SCORE | FP | CSV:linha |
| :-- | :-- | :-- | --: | :-- |
| vigilancia | global | maior_ou_igual | 17 | `fp_arranjo_por_regiao.csv:8` |
| vigilancia | global | menor | 4 | `fp_arranjo_por_regiao.csv:9` |
| vigilancia | por_passo | maior_ou_igual | 11 | `fp_arranjo_por_regiao.csv:16` |
| vigilancia | por_passo | menor | 10 | `fp_arranjo_por_regiao.csv:17` |
| confirmacao | global | maior_ou_igual | 4 | `fp_arranjo_por_regiao.csv:24` |
| confirmacao | global | menor | 0 | `fp_arranjo_por_regiao.csv:25` |
| confirmacao | por_passo | maior_ou_igual | 4 | `fp_arranjo_por_regiao.csv:32` |
| confirmacao | por_passo | menor | 0 | `fp_arranjo_por_regiao.csv:33` |

### P com MATCH ao revisor humano: mesma carga humana

Exploratória, hipótese de revisor perfeito dos dois lados. Os FP abaixo do corte tornam-se zero nesse cenário, não por acerto autônomo do LLM. São pares avaliados pelo LLM, não chamadas de API.

| Modo | P: revisões originais | MATCH ao humano | P: revisões totais | PAR=2 de P | O3: revisões iguais | PAR=2 de O3 | O3: revisões para igualar P | Ganho P na mesma carga | CSV:linha |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| vigilancia | 96 | 25 | 121 | 23 | 121 | 22 | 125 | 1 | `comparador_p_match_humano.csv:2` |
| confirmacao | 17 | 12 | 29 | 9 | 29 | 9 | 29 | 0 | `comparador_p_match_humano.csv:3` |

### Sensibilidade humana: parâmetro de cenário

Sensibilidade humana real não medida. Especificidade humana tratada como perfeita; FP humanos não modelados. As perdas de cobertura são vínculos que não chegam ao revisor humano. As amplitudes entre os cenários de 95% e 100% são lidas do artefato, não confundidas com a parcela não examinada.

| Modo | Estratégia | k | Revisões humanas | Cobertura perdida | s | Sensibilidade total (%) | Amplitude (p.p.) | CSV:linha |
| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |
| vigilancia | humano_integral | 61485 | 61485 | 0 | 95 | 99,27 | 0,73 | `cenarios_sensibilidade_humana.csv:2` |
| vigilancia | humano_integral | 61485 | 61485 | 0 | 99 | 99,85 | 0,73 | `cenarios_sensibilidade_humana.csv:3` |
| vigilancia | humano_integral | 61485 | 61485 | 0 | 100 | 100,00 | 0,73 | `cenarios_sensibilidade_humana.csv:4` |
| vigilancia | P | 96 | 96 | 13 | 95 | 94,35 | 0,38 | `cenarios_sensibilidade_humana.csv:5` |
| vigilancia | P | 96 | 96 | 13 | 99 | 94,66 | 0,38 | `cenarios_sensibilidade_humana.csv:6` |
| vigilancia | P | 96 | 96 | 13 | 100 | 94,74 | 0,38 | `cenarios_sensibilidade_humana.csv:7` |
| vigilancia | P_MATCH_ao_humano | 121 | 121 | 13 | 95 | 94,27 | 0,47 | `cenarios_sensibilidade_humana.csv:8` |
| vigilancia | P_MATCH_ao_humano | 121 | 121 | 13 | 99 | 94,64 | 0,47 | `cenarios_sensibilidade_humana.csv:9` |
| vigilancia | P_MATCH_ao_humano | 121 | 121 | 13 | 100 | 94,74 | 0,47 | `cenarios_sensibilidade_humana.csv:10` |
| vigilancia | O3 | 96 | 96 | 14 | 95 | 93,89 | 0,45 | `cenarios_sensibilidade_humana.csv:11` |
| vigilancia | O3 | 96 | 96 | 14 | 99 | 94,24 | 0,45 | `cenarios_sensibilidade_humana.csv:12` |
| vigilancia | O3 | 96 | 96 | 14 | 100 | 94,33 | 0,45 | `cenarios_sensibilidade_humana.csv:13` |
| vigilancia | O3 | 121 | 121 | 14 | 95 | 93,89 | 0,45 | `cenarios_sensibilidade_humana.csv:14` |
| vigilancia | O3 | 121 | 121 | 14 | 99 | 94,24 | 0,45 | `cenarios_sensibilidade_humana.csv:15` |
| vigilancia | O3 | 121 | 121 | 14 | 100 | 94,33 | 0,45 | `cenarios_sensibilidade_humana.csv:16` |
| vigilancia | O3 | 125 | 125 | 13 | 95 | 94,27 | 0,47 | `cenarios_sensibilidade_humana.csv:17` |
| vigilancia | O3 | 125 | 125 | 13 | 99 | 94,64 | 0,47 | `cenarios_sensibilidade_humana.csv:18` |
| vigilancia | O3 | 125 | 125 | 13 | 100 | 94,74 | 0,47 | `cenarios_sensibilidade_humana.csv:19` |
| vigilancia | O3 | 440 | 440 | 10 | 95 | 95,43 | 0,53 | `cenarios_sensibilidade_humana.csv:20` |
| vigilancia | O3 | 440 | 440 | 10 | 99 | 95,85 | 0,53 | `cenarios_sensibilidade_humana.csv:21` |
| vigilancia | O3 | 440 | 440 | 10 | 100 | 95,95 | 0,53 | `cenarios_sensibilidade_humana.csv:22` |
| vigilancia | filtro_LLM | 17 | 5 | 32 | 95 | 86,96 | 0,08 | `cenarios_sensibilidade_humana.csv:23` |
| vigilancia | filtro_LLM | 17 | 5 | 32 | 99 | 87,03 | 0,08 | `cenarios_sensibilidade_humana.csv:24` |
| vigilancia | filtro_LLM | 17 | 5 | 32 | 100 | 87,04 | 0,08 | `cenarios_sensibilidade_humana.csv:25` |
| vigilancia | filtro_LLM | 29 | 7 | 30 | 95 | 87,73 | 0,12 | `cenarios_sensibilidade_humana.csv:26` |
| vigilancia | filtro_LLM | 29 | 7 | 30 | 99 | 87,83 | 0,12 | `cenarios_sensibilidade_humana.csv:27` |
| vigilancia | filtro_LLM | 29 | 7 | 30 | 100 | 87,85 | 0,12 | `cenarios_sensibilidade_humana.csv:28` |
| vigilancia | filtro_LLM | 58 | 17 | 25 | 95 | 89,66 | 0,22 | `cenarios_sensibilidade_humana.csv:29` |
| vigilancia | filtro_LLM | 58 | 17 | 25 | 99 | 89,83 | 0,22 | `cenarios_sensibilidade_humana.csv:30` |
| vigilancia | filtro_LLM | 58 | 17 | 25 | 100 | 89,88 | 0,22 | `cenarios_sensibilidade_humana.csv:31` |
| vigilancia | filtro_LLM | 96 | 22 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:32` |
| vigilancia | filtro_LLM | 96 | 22 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:33` |
| vigilancia | filtro_LLM | 96 | 22 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:34` |
| vigilancia | filtro_LLM | 121 | 26 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:35` |
| vigilancia | filtro_LLM | 121 | 26 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:36` |
| vigilancia | filtro_LLM | 121 | 26 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:37` |
| vigilancia | filtro_LLM | 125 | 26 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:38` |
| vigilancia | filtro_LLM | 125 | 26 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:39` |
| vigilancia | filtro_LLM | 125 | 26 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:40` |
| confirmacao | humano_integral | 61485 | 61485 | 0 | 95 | 99,27 | 0,73 | `cenarios_sensibilidade_humana.csv:41` |
| confirmacao | humano_integral | 61485 | 61485 | 0 | 99 | 99,85 | 0,73 | `cenarios_sensibilidade_humana.csv:42` |
| confirmacao | humano_integral | 61485 | 61485 | 0 | 100 | 100,00 | 0,73 | `cenarios_sensibilidade_humana.csv:43` |
| confirmacao | P | 17 | 17 | 27 | 95 | 89,05 | 0,02 | `cenarios_sensibilidade_humana.csv:44` |
| confirmacao | P | 17 | 17 | 27 | 99 | 89,06 | 0,02 | `cenarios_sensibilidade_humana.csv:45` |
| confirmacao | P | 17 | 17 | 27 | 100 | 89,07 | 0,02 | `cenarios_sensibilidade_humana.csv:46` |
| confirmacao | P_MATCH_ao_humano | 29 | 29 | 27 | 95 | 88,89 | 0,18 | `cenarios_sensibilidade_humana.csv:47` |
| confirmacao | P_MATCH_ao_humano | 29 | 29 | 27 | 99 | 89,03 | 0,18 | `cenarios_sensibilidade_humana.csv:48` |
| confirmacao | P_MATCH_ao_humano | 29 | 29 | 27 | 100 | 89,07 | 0,18 | `cenarios_sensibilidade_humana.csv:49` |
| confirmacao | O3 | 96 | 96 | 14 | 95 | 93,89 | 0,45 | `cenarios_sensibilidade_humana.csv:50` |
| confirmacao | O3 | 96 | 96 | 14 | 99 | 94,24 | 0,45 | `cenarios_sensibilidade_humana.csv:51` |
| confirmacao | O3 | 96 | 96 | 14 | 100 | 94,33 | 0,45 | `cenarios_sensibilidade_humana.csv:52` |
| confirmacao | O3 | 121 | 121 | 14 | 95 | 93,89 | 0,45 | `cenarios_sensibilidade_humana.csv:53` |
| confirmacao | O3 | 121 | 121 | 14 | 99 | 94,24 | 0,45 | `cenarios_sensibilidade_humana.csv:54` |
| confirmacao | O3 | 121 | 121 | 14 | 100 | 94,33 | 0,45 | `cenarios_sensibilidade_humana.csv:55` |
| confirmacao | O3 | 125 | 125 | 13 | 95 | 94,27 | 0,47 | `cenarios_sensibilidade_humana.csv:56` |
| confirmacao | O3 | 125 | 125 | 13 | 99 | 94,64 | 0,47 | `cenarios_sensibilidade_humana.csv:57` |
| confirmacao | O3 | 125 | 125 | 13 | 100 | 94,74 | 0,47 | `cenarios_sensibilidade_humana.csv:58` |
| confirmacao | O3 | 440 | 440 | 10 | 95 | 95,43 | 0,53 | `cenarios_sensibilidade_humana.csv:59` |
| confirmacao | O3 | 440 | 440 | 10 | 99 | 95,85 | 0,53 | `cenarios_sensibilidade_humana.csv:60` |
| confirmacao | O3 | 440 | 440 | 10 | 100 | 95,95 | 0,53 | `cenarios_sensibilidade_humana.csv:61` |
| confirmacao | filtro_LLM | 17 | 5 | 32 | 95 | 86,96 | 0,08 | `cenarios_sensibilidade_humana.csv:62` |
| confirmacao | filtro_LLM | 17 | 5 | 32 | 99 | 87,03 | 0,08 | `cenarios_sensibilidade_humana.csv:63` |
| confirmacao | filtro_LLM | 17 | 5 | 32 | 100 | 87,04 | 0,08 | `cenarios_sensibilidade_humana.csv:64` |
| confirmacao | filtro_LLM | 29 | 7 | 30 | 95 | 87,73 | 0,12 | `cenarios_sensibilidade_humana.csv:65` |
| confirmacao | filtro_LLM | 29 | 7 | 30 | 99 | 87,83 | 0,12 | `cenarios_sensibilidade_humana.csv:66` |
| confirmacao | filtro_LLM | 29 | 7 | 30 | 100 | 87,85 | 0,12 | `cenarios_sensibilidade_humana.csv:67` |
| confirmacao | filtro_LLM | 58 | 15 | 25 | 95 | 89,66 | 0,22 | `cenarios_sensibilidade_humana.csv:68` |
| confirmacao | filtro_LLM | 58 | 15 | 25 | 99 | 89,83 | 0,22 | `cenarios_sensibilidade_humana.csv:69` |
| confirmacao | filtro_LLM | 58 | 15 | 25 | 100 | 89,88 | 0,22 | `cenarios_sensibilidade_humana.csv:70` |
| confirmacao | filtro_LLM | 96 | 18 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:71` |
| confirmacao | filtro_LLM | 96 | 18 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:72` |
| confirmacao | filtro_LLM | 96 | 18 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:73` |
| confirmacao | filtro_LLM | 121 | 22 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:74` |
| confirmacao | filtro_LLM | 121 | 22 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:75` |
| confirmacao | filtro_LLM | 121 | 22 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:76` |
| confirmacao | filtro_LLM | 125 | 22 | 23 | 95 | 90,43 | 0,26 | `cenarios_sensibilidade_humana.csv:77` |
| confirmacao | filtro_LLM | 125 | 22 | 23 | 99 | 90,64 | 0,26 | `cenarios_sensibilidade_humana.csv:78` |
| confirmacao | filtro_LLM | 125 | 22 | 23 | 100 | 90,69 | 0,26 | `cenarios_sensibilidade_humana.csv:79` |

### Filtro de NONMATCH: cobertura e risco de descarte

Exploração com protocolo fixado antes dos desfechos, respostas de datas diferentes. NONMATCH descarta; MATCH e inconclusivos vão ao revisor humano, sem aceitação autônoma. Para k=17 e k=29, as contagens iguais decorrem da coincidência das decisões sobre PAR=2; nos não vínculos há decisões diferentes entre modos. A coincidência sobre PAR=2 continua nos topos maiores apresentados. Fonte: `filtro_llm_concordancia_modos.csv:2-19`.

| Modo | Topo k | Humano | PAR=2 recuperados | Sem filtro, mesma carga | VPN (%) | PAR=2 descartados | CSV:linha |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| vigilancia | 17 | 5 | 4 | 3 | 83,3 | 2 | `filtro_llm.csv:2` |
| vigilancia | 29 | 7 | 6 | 4 | 86,4 | 3 | `filtro_llm.csv:3` |
| vigilancia | 58 | 17 | 11 | 6 | 78,0 | 9 | `filtro_llm.csv:4` |
| vigilancia | 96 | 22 | 13 | 6 | 87,8 | 9 | `filtro_llm.csv:5` |
| vigilancia | 121 | 26 | 13 | 7 | 90,5 | 9 | `filtro_llm.csv:6` |
| vigilancia | 125 | 26 | 13 | 7 | 89,9 | 10 | `filtro_llm.csv:7` |
| confirmacao | 17 | 5 | 4 | 3 | 83,3 | 2 | `filtro_llm.csv:8` |
| confirmacao | 29 | 7 | 6 | 4 | 86,4 | 3 | `filtro_llm.csv:9` |
| confirmacao | 58 | 15 | 11 | 6 | 79,1 | 9 | `filtro_llm.csv:10` |
| confirmacao | 96 | 18 | 13 | 6 | 88,5 | 9 | `filtro_llm.csv:11` |
| confirmacao | 121 | 22 | 13 | 6 | 90,9 | 9 | `filtro_llm.csv:12` |
| confirmacao | 125 | 22 | 13 | 6 | 90,3 | 10 | `filtro_llm.csv:13` |

### Subamostragem: degradação em 1:1 e 1:5; 1:20 e 1:50 indistinguíveis

Variante isotonic, população abaixo do corte. Medianas e intervalos entre sementes nas proporções subamostradas; a referência original tem uma única execução. Sem alegar que todos os subgrupos perdem a mesma quantidade. O Brier mediano é maior que a referência original em todas as proporções subamostradas.

| Razão | Métrica | Mediana | Mínimo | Máximo | Referência | CSV:linha |
| :-- | :-- | --: | --: | --: | --: | :-- |
| 1:1 | brier | 0,000999 | 0,000750 | 0,001234 | 0,000629 | `subamostragem_treino_resumo.csv:4` |
| 1:1 | par2_top_96 | 13,0 | 8,0 | 17,0 | 22,0 | `subamostragem_treino_resumo.csv:18` |
| 1:1 | sens96_passo_6 | 0,25 | 0,00 | 0,25 | 0,50 | `subamostragem_treino_resumo.csv:24` |
| 1:1 | sens96_sexo_M | 0,25 | 0,08 | 0,38 | 0,58 | `subamostragem_treino_resumo.csv:36` |
| 1:20 | brier | 0,000698 | 0,000608 | 0,000845 | 0,000629 | `subamostragem_treino_resumo.csv:196` |
| 1:20 | par2_top_96 | 21,0 | 18,0 | 22,0 | 22,0 | `subamostragem_treino_resumo.csv:210` |
| 1:20 | sens96_passo_6 | 0,50 | 0,50 | 0,75 | 0,50 | `subamostragem_treino_resumo.csv:216` |
| 1:20 | sens96_sexo_M | 0,54 | 0,50 | 0,58 | 0,58 | `subamostragem_treino_resumo.csv:228` |
| 1:5 | brier | 0,000806 | 0,000623 | 0,001016 | 0,000629 | `subamostragem_treino_resumo.csv:388` |
| 1:5 | par2_top_96 | 18,5 | 16,0 | 21,0 | 22,0 | `subamostragem_treino_resumo.csv:402` |
| 1:5 | sens96_passo_6 | 0,25 | 0,25 | 0,50 | 0,50 | `subamostragem_treino_resumo.csv:408` |
| 1:5 | sens96_sexo_M | 0,46 | 0,35 | 0,54 | 0,58 | `subamostragem_treino_resumo.csv:420` |
| 1:50 | brier | 0,000648 | 0,000527 | 0,000806 | 0,000629 | `subamostragem_treino_resumo.csv:580` |
| 1:50 | par2_top_96 | 22,5 | 20,0 | 24,0 | 22,0 | `subamostragem_treino_resumo.csv:594` |
| 1:50 | sens96_passo_6 | 0,50 | 0,50 | 0,75 | 0,50 | `subamostragem_treino_resumo.csv:600` |
| 1:50 | sens96_sexo_M | 0,60 | 0,50 | 0,65 | 0,58 | `subamostragem_treino_resumo.csv:612` |
| original | brier | 0,000629 | 0,000629 | 0,000629 | 0,000629 | `subamostragem_treino_resumo.csv:772` |
| original | par2_top_96 | 22,0 | 22,0 | 22,0 | 22,0 | `subamostragem_treino_resumo.csv:786` |
| original | sens96_passo_6 | 0,50 | 0,50 | 0,50 | 0,50 | `subamostragem_treino_resumo.csv:792` |
| original | sens96_sexo_M | 0,58 | 0,58 | 0,58 | 0,58 | `subamostragem_treino_resumo.csv:804` |

### Cobertura e recuperação integral por ordenação

| Modo | Ordenação | P: recuperados | Não recuperados no topo K1 | k para igualar P | k para os 36 | CSV:linha |
| :-- | :-- | --: | --: | --: | --: | :-- |
| vigilancia | O1_SCORE | 23 | 17 | 501 | 2521 | `comparador_mesma_carga.csv:2` |
| confirmacao | O1_SCORE | 9 | 33 | 51 | 2521 | `comparador_mesma_carga.csv:7` |
| vigilancia | O2_CdR | 23 | 24 | 4163 | 24248 | `comparador_mesma_carga.csv:12` |
| confirmacao | O2_CdR | 9 | 27 | 11 | 24248 | `comparador_mesma_carga.csv:17` |
| vigilancia | O3_p_cal_ml | 23 | 14 | 125 | 23443 | `comparador_mesma_carga.csv:22` |
| confirmacao | O3_p_cal_ml | 9 | 30 | 29 | 23443 | `comparador_mesma_carga.csv:27` |
| vigilancia | O4_P_recalibrada | 23 | 14 | 106 | 53591 | `comparador_mesma_carga.csv:32` |
| confirmacao | O4_P_recalibrada | 9 | 31 | 27 | 53591 | `comparador_mesma_carga.csv:37` |

### Pontos específicos da conferência final

| Número | CSV:linha em `data/` |
| :-- | :-- |
| Com 17 revisões, O2 recupera 9 e O3 recupera 6 | `comparador_mesma_carga.csv:17,27` |
| Vigilância k=96: 74 NONMATCH avaliados pelo LLM e 22 revisões humanas; VPN 87,8%; k=58: VPN 78,0% | `filtro_llm.csv:5,4` |
| Confirmação k=125: 22 revisões, 13 vínculos recuperados e 10 dos 23 descartados, ponto distinto de k=96 na vigilância | `filtro_llm.csv:13` |
| Original: 22 no topo 96, uma única execução; passo 6 com 4 PAR=2 | `subamostragem_treino_resumo.csv:786,25` |
