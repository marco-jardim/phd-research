# Roteiro da defesa

Commit final dos fontes LaTeX e dos PDFs desta repassada: `e6b626dcf1e4f5f46f29c1e66c5e19d564d8dca0`. O commit documental subsequente apenas registra esta referência no cabeçalho, sem alterar as entradas compiladas ou os PDFs. A numeração corresponde aos quadros do PDF, sem sobreposições: 30 principais e 19 de reserva, estes fora dos 40 minutos. O tempo é estimativa, não duração medida. O ensaio foi descartado por decisão do autor em 04/11/2026.

## Cronometragem

| Bloco | Slides | Minutos | Acumulado |
| :-- | :-- | :-- | :-- |
| Capa | 1 | 0,5 | 0,5 |
| Problema e tese central | 2 a 4 | 4,0 | 4,5 |
| Dados e referência operacional | 5 a 7 | 4,0 | 8,5 |
| Referencial e decisão de limiar | 8 | 1,5 | 10,0 |
| Estudo 1: comparador | 9 e 10 | 2,0 | 12,0 |
| Estudo 2: estimador | 11 a 13 | 4,0 | 16,0 |
| Estudo 3: estratégias e impacto | 14 a 16 | 4,0 | 20,0 |
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
A hipótese gerada nesta avaliação é que a ordenação por probabilidade ou escore concentra vínculos abaixo do corte. O ganho em esforço vem de revisar menos pares na ordem certa, não de uma superioridade geral da política. A hipótese exige validação registrada em dados novos. LLM designa modelo de linguagem de grande porte; CdR, Comparador de Registros; OOF, fora da amostra; FP, falso positivo; VPP, valor preditivo positivo. P é a isotônica registrada internamente como principal; O3 é a probabilidade calibrada OOF na prevalência original e O4 a recalibrada de P, com o mesmo XGBoost. Nenhuma ordenação domina em todas as cargas. Preservar o corte e revisar abaixo dele em ordem decrescente de uma probabilidade ou escore, escolhendo ordenação e ponto de parada pela curva de esforço conforme o orçamento humano. Se usar probabilidade aprendida, treinar na prevalência original.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36. PAR é o *label*: 1 indica aceitação pelo corte; 2, recuperação pela revisão. O volume e a ordem históricos não foram observados. As duas fronteiras da nota do CdR aceitam 93 vínculos e 2 FP acima de 9, encaminham 21.620 ao revisor humano e projetam 95,7% no cenário de sensibilidade humana de 95%, não medida. Os cenários consideram 95%, 99% e 100% de sensibilidade humana, com especificidade perfeita assumida. Se a revisão tivesse seguido escore do OpenRecLink decrescente, o cenário seria de pelo menos 2.156 pares por passo ou 2.522 no limiar global. Isso não reconstrói o trabalho histórico.

### 7. Faixas e padrão-ouro
A nota do CdR não é o escore probabilístico do OpenRecLink. Nas faixas da nota: abaixo de 5 há 39.981 pares e 3 vínculos; em [5,9), 21.620 e 151; a partir de 9, 95 e 93. Houve revisora única, sem classificação independente. Oliveira (2026, p. 34 e 36 a 38, Quadros 5 e 6) documenta CdR apenas no passo aberto, com 5 pares, não nos passos 6, 9 e 11. Os 36 PAR=2 se distribuem em 4, 7 e 25 nesses passos. A referência tem viés de verificação e incorporação, mas não se atribui ao CdR uma exposição não documentada nos três passos.

### 8. Mapa conceitual
O referencial clássico organiza a decisão entre vínculo, não vínculo e revisão. A tese trabalha depois da geração dos potenciais pares e do cálculo dos subescores, reorganizando a decisão sobre a saída do CdR, sem reestimar os parâmetros de comparação nem substituir a blocagem do OpenRecLink.

### 9. Comparador em Python
O comparador torna explícita a evidência campo a campo, com subescores e uma nota ponderada por pesos fixos. A equivalência parcial com o protótipo Java de Peçanha (2015) foi executada em pares sintéticos. A captura apresenta a interface do Comparador de Registros, versão 0.1, *commit* 3b43598, com dados sintéticos, não uma interface do GZ-CMD.

### 10. Equivalência Java e Python
A comparação em 500 pares sintéticos encontrou concordância de 100,0% nas datas por igualdade e por um dígito, de 99,2% nas abreviações e nota final idêntica em 11,2%. Nas inversões de data, Lucena e o Java atribuem peso 0,8, enquanto o Python usa 1,0. Essa diferença foi mantida e declarada como limitação. O lote de fronteira não estima concordância nos dados reais.

### 11. Comparação de protocolos
RF+SMOTE atingiu F1 de 0,931 na ablação original, e o XGBoost do fluxo Modal atingiu 0,943, contra 0,610 do limiar 8. O XGBoost do QW1 atingiu 0,920 no teste; os protocolos não são intercambiáveis, e a escolha dos limiares nos testes torna os resultados selecionados exploratórios.

### 12. Balanceamento
As nove estratégias de balanceamento foram avaliadas no Random Forest, com F1 médio entre 0,880 e 0,918. Essa variação descreve o experimento realizado, não demonstra que uma estratégia de reamostragem seja necessária ou superior em qualquer modelo.

### 13. Estimador do GZ-CMD
O GZ-CMD usa outro XGBoost, com 30 atributos e calibração isotônica aninhada, agrupando por REFREC nos níveis externo e interno. Na área cinza da nota, a AUC é 0,995 e a AP é 0,862; o corte ilustrativo de 0,5 produz 119 verdadeiros positivos e 28 falsos positivos. Esses números descrevem o estimador original, não o arranjo. Nele, o classificador continua ajustado na base inteira, mas a isotônica é reajustada apenas com as probabilidades OOF internas do treino abaixo do corte, sem os *labels* da partição externa.

### 14. Finalidades operacionais
A vigilância atribui maior consequência à perda do vínculo, enquanto a confirmação atribui maior consequência ao falso vínculo. A fronteira permite explicitar esse compromisso, sem transformar os pontos explorados no teste em limiares validados para uso externo.

### 15. Impacto epidemiológico
No teste da ablação, o RF+SMOTE detectou 67 vínculos e o limiar 8 detectou 43, uma diferença líquida de 24, ou 55,8%. São 27 recuperados exclusivamente pelo modelo, menos três exclusivos do limiar; esse resultado tem como denominador 74 vínculos do teste e não mede o impacto do arranjo nem da cascata sobre a base inteira.

### 16. Limites do impacto
Os indicadores derivados do encerramento por óbito constituem vazamento de alvo. A sensibilidade 2×2 foi executada: o F1 do RF+SMOTE passou de 0,931 para 0,938 sem óbito, mas caiu para 0,846 com REFREC. No QW1 local na GPU, o XGBoost passou de 0,937 para 0,878 com REFREC, sem substituir o resultado histórico de 0,920. O otimismo aparece sobretudo na partição por linha, embora mudem os vínculos e denominadores. O GZ-CMD já utiliza agrupamento por REFREC, sem indicadores de óbito; permanece possível dependência por COMPREC.

### 17. Arquitetura depois do corte
O arranjo preserva a aceitação acima do corte e aplica o GZ-CMD abaixo dele. As guardas usam nota e probabilidade recalibrada; as faixas estratificam o erro do revisor. A isotônica é reajustada apenas no treino abaixo do corte, sem *labels* da partição externa. As taxas foram congeladas antes das chamadas novas, sem realimentação por essas respostas, mas P reutiliza sobretudo lotes históricos também usados para estimar as proporções do revisor. Não é avaliação prospectiva. *Grey-Zone* no nome remete ao conceito de Fellegi e Sunter, a região que a regra automática não decide; nos resultados, área cinza da nota designa somente [5,9).

### 18. Duas ações e orçamento
Com duas ações, revisor humano e não vínculo, e sob revisor perfeito, a perda da revisão é C_H e a do não vínculo é p × C_FN. Portanto, compensa revisar quando p ≥ C_H/C_FN. Sob orçamento, escolhem-se os pares de maior probabilidade até o ponto de parada. Os demais recebem não vínculo. A ordenação é o GZ-CMD restrito a essas duas ações, não um método decisório incompatível com *expected loss*. A falha da política de quatro ações abaixo do corte está nas entradas: 9,06 FP previstos contra 21 observados e calibração na prevalência de 36 vínculos entre 61.485 pares. O fluxo proposto continua hipótese a validar.

### 19. Arranjo sequencial com registro interno
P conserva os 211 vínculos do corte e acrescenta 4 na vigilância e 8 na confirmação. Não reproduz 13 e 27 dos 36 vínculos do serviço. Aceita 25 e 12 potenciais pares abaixo do corte: 4 e 8 são PAR=2; dos demais, 21 e 4 são FP contra o *label*, rejeitados ou não examinados pelo serviço. Portanto, não se atribui a condição de rejeitado ou não examinado aos 25 e 12 como um todo. As revisões humanas são 96 na vigilância e 17 na confirmação; os pares avaliados pelo LLM são 344 e 41. O comparador do serviço, a carga, o VPP e os mecanismos são análises exploratórias *post hoc*, não registradas. S3 designa a avaliação com os *labels*. IC aninhados no *label* foram retirados; os fechamentos e a cronologia permanecem na reserva.

### 20. Curva de esforço e ponto de parada
A curva de esforço relaciona revisões humanas e vínculos recuperados. O3 é a probabilidade calibrada OOF, e O4 a probabilidade recalibrada de P. O2 recupera 9 PAR=2 com 11 revisões; O3 precisa de 29. Para todos os 36, O1 requer 2.521 e O3, 23.443, fora do limite gráfico de 2.522. Nenhuma ordenação domina em todas as cargas. Os pontos de P incluem 344 pares adicionais avaliados pelo LLM na vigilância e 41 na confirmação. Revisor perfeito é hipótese dos dois lados. Se a revisão histórica seguiu o escore do OpenRecLink, favoreceria O1; O3 e O4 foram treinados nos mesmos *labels* verificados. A curva gera uma hipótese de escolha do ponto de parada, não validação externa. O4 iguala P com 106 revisões na vigilância e 27 na confirmação, contra 125 e 29 de O3. No orçamento humano original de P (96 e 17), P recupera 23 e 9; O3 recupera 22 e 6, deixando 14 e 30 dos 36 fora. Fontes: `comparador_mesma_carga.csv:22,27,32,37`.

### 21. Aceitador e filtro: papéis distintos
Como aceitador, o LLM teve VPP de 16,0% na vigilância, com 21 FP contra 9,06 previstos, análise *post hoc*. Como filtro de NONMATCH, em k=96 na vigilância são 74 pares avaliados pelo LLM como NONMATCH e 22 revisões humanas: 13 PAR=2 contra 6 sem filtro. O VPN é 87,8% em k=96; 78,0% em k=58, na vigilância. Em outro ponto, confirmação com k=125, são 22 revisões e 13 vínculos, mas 10 dos 23 vínculos do topo são descartados. MATCH e inconclusivos vão ao humano, nunca à aceitação autônoma. É hipótese exploratória a validar, com protocolo fixado antes dos desfechos e respostas de datas diferentes. Fontes: `filtro_llm.csv:4,5,13`; `arranjo_corte_vpp_llm.csv:2`.

### 22. O LLM como aceitador não se paga abaixo do corte
P permanece principal porque foi registrado antes. Se MATCH do LLM vai ao revisor humano, P passa de 96 para 121 revisões na vigilância, recuperando 23 PAR=2; O3 recupera 22 na mesma carga e iguala P em 125. Na confirmação, ambos recuperam 9 em 29. Revisor perfeito é hipótese dos dois lados. A política com LLM acrescenta no máximo 1 vínculo na mesma carga humana, ao custo de 344 pares avaliados pelo LLM na vigilância e 41 na confirmação. Uma diferença de um vínculo em 36 não sustenta superioridade; não justifica substituir o revisor humano pelo LLM. P-Platt continua análise de sensibilidade, não escolha retroativa de vencedor.

### 23. Equidade por sexo
No arranjo P de vigilância, o VPP é de 86,3% no sexo feminino e 93,3% no masculino, sem rejeição após Benjamini-Hochberg, p ajustado de 0,378. Na cascata sobre a base inteira (comparação), é de 87,9% e 84,0%. A aceitação do arranjo combina corte e respostas de vínculo do LLM; o encaminhamento humano conta como não aceito. Os 211 vínculos acima do corte são aceitos por construção. As diferenças de sensibilidade entre subgrupos dependem da proporção de vínculos PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36 vínculos. As diferenças de VPP dependem dos 21 e quatro falsos positivos abaixo do corte. O poder é limitado por esses números e pela incorporação do corte ao *label*; ausência de rejeição não comprova equidade. Raça/cor não está disponível.

### 24. Desigualdade operacional
A disparidade por passo é herdada do corte e do *label*, não um efeito isolado do GZ-CMD. O corte sozinho aceita 15 dos 40 vínculos do passo 11, ou 37,5%, contra 76 de 80 no passo 6 (95,0%) e 120 de 127 no passo 9 (94,5%). No passo 11 estão 25 dos 36 vínculos com PAR=2. O arranjo aceita dois desses 25 na vigilância e quatro na confirmação, chegando a 17/40 (42,5%) e 19/40 (47,5%). Nos passos 6/9/11, as aceitações PAR=2 são 0/2/2 na vigilância e 1/3/4 na confirmação, conforme `data/par_por_passo.csv`. Nos passos 6 e 9, a sensibilidade do arranjo é de pelo menos 95,0% nos dois modos. Na vigilância, fica abaixo da cascata sobre a base inteira (comparação) no passo 11: 42,5% contra 45,0%. Após Benjamini-Hochberg, são sete e três rejeições em P, todas operacionais; na base inteira (comparação), nove e uma. A contagem não ordena equidade. A equidade do processo completo depende da revisão humana ainda não observada.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. A tese propõe sua composição operacional com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
A revisora foi única, sem classificação independente. O corte integra o *label*. Se a revisão histórica seguiu o escore do OpenRecLink, favoreceria O1; O3 e O4 também usaram os mesmos *labels* verificados. A ordem histórica é desconhecida, e 2.156 revisões constituem cenário. REFREC não separa COMPREC. Respostas históricas reutilizadas, taxas com denominadores mínimos e transferência de coortes mais fáceis impedem interpretação prospectiva. A sensibilidade humana não foi medida, e a especificidade humana é tratada como perfeita: os cenários não modelam FP humanos. O comparador *a posteriori* era conhecido antes do registro interno; o código registrado está ausente do git, e há desvios de protocolo. O registro cobre recalibração, braços e avaliação; a análise de estabilidade mede estabilidade nos mesmos dados, não generalização.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. O processamento LLM utilizou APIs externas. A exportação restrita com identificador de par por HMAC foi implementada em `scripts/export_matriz_desidentificada.py`, mas a matriz não foi depositada e não se afirma saneamento retroativo das execuções. A equivalência Java parcial foi executada; a comparação empírica com Lages não foi realizada.

### 28. Perdas de cobertura e cenários humanos
As perdas são de cobertura, não de erro humano. Os cenários de sensibilidade humana de 95%, 99% e 100%, com especificidade perfeita assumida, têm amplitudes máximas de 0,53 p.p. nas estratégias de carga reduzida e 0,73 p.p. na revisão integral. Sob revisor perfeito, P recupera 23 e 9 dos 36, não reproduzindo 13 e 27, mas aceita 21 e 4 FP. Em 96 e 17 revisões, O3 recupera 22 e 6, deixando 14 e 30 fora. Para igualar P, O3 requer 125 e 29; O4, 106 e 27. Para os 36, O1 requer 2.521 e O3, 23.443. Preservar o corte; escolher probabilidade ou escore e ponto de parada pela curva de esforço conforme o orçamento humano. Nenhuma ordenação domina em todas as cargas. Horas e cenário de carga ficam na reserva 49. Fonte: `comparador_mesma_carga.csv:2,22,27,32,37`; `cenarios_sensibilidade_humana.csv:3,21`.

### 29. Contribuições e recomendação final
As contribuições são a decomposição por PAR; a curva de esforço como instrumento de escolha da ordenação e do ponto de parada; a investigação do mesmo LLM como aceitador ruim e filtro potencial; a política auditável de quatro ações, o CdR em Python, pacote e *benchmark* abertos; e o resultado negativo com registro interno. A hipótese final é preservar o corte. Abaixo dele, revisar em ordem decrescente de uma probabilidade ou escore, escolhendo a ordenação e o ponto de parada pela curva de esforço conforme o orçamento humano; nenhuma ordenação domina em todas as cargas. Se a ordenação for por probabilidade aprendida, treinar na prevalência original: a subamostragem não melhorou a fila. O restante recebe não vínculo. O LLM pode entrar apenas como filtro de NONMATCH antes da revisão, nunca como aceitador autônomo. Subamostragem não ajudou. A adoção exige validação registrada em dados novos.

### 30. Agradecimentos
Agradece-se à orientação, à banca e à equipe responsável pelo processamento e pela revisão. As perdas são sobretudo de cobertura. P não reproduz 13 vínculos na vigilância e 27 na confirmação, com 21 e 4 FP. Na mesma carga humana, o LLM acrescenta no máximo 1 vínculo, sem justificar seu uso como substituto do revisor humano. A hipótese gerada é preservar o corte, escolher probabilidade ou escore e ponto de parada pela curva de esforço conforme o orçamento humano; nenhuma ordenação domina em todas as cargas. Se usar probabilidade aprendida, treinar na prevalência original. O restante recebe não vínculo. O filtro de NONMATCH permanece experimental, nunca aceitando vínculos sozinho.

### 31. Backup: base inteira (comparação), cascata gráfica
A figura sintetiza a cascata sobre a base inteira por política e modo, não o arranjo. Os subtotais humanos incluem encaminhamento direto e inconclusivos do LLM, não constituindo novos ramos a somar novamente.

### 32. Backup: base inteira (comparação), contagens
As contagens da cascata sobre a base inteira separam aceitação automática, perda explícita e vínculo pendente ao revisor humano. A sensibilidade automática inclui o LLM, mas não a decisão humana ainda não realizada.

### 33. Backup: faixas OOF
A tabela detalha a discriminação do estimador por faixa da nota, com corte 0,5 fixado antes da análise. O desempenho do estimador isolado não deve ser atribuído à política v3.1, que pode rejeitar pela guarda ou escolher revisão por custo.

### 34. Backup: QW1
As redes neurais foram comparadas no QW1, com validação interna e teste da mesma base. Os resultados não demonstram uma hierarquia universal entre famílias, e o F1 de 0,943 não pertence a essa tabela.

### 35. Backup: ablação
As categorias reúnem implementações e origens experimentais distintas, com escolha exploratória de limiares nos testes. O resultado nulo de uma cascata do fluxo Modal decorre do corte de regras que nenhum vínculo do teste alcançou, não de impossibilidade geral da combinação.

### 36. Backup: balanceamento
A figura permite consultar todas as estratégias avaliadas no Random Forest. A dispersão é interna ao protocolo e não mede transferência para outra população ou período.

### 37. Backup: detecção
A figura reaproveita a comparação de métodos no teste da ablação, com 74 vínculos. A contagem é de vínculos verdadeiros detectados de pessoas notificadas por TB, não uma nova contagem de mortes cuja causa básica seja TB.

### 38. Backup: ablação do dossiê
Os 18 vínculos foram selecionados por erro compartilhado, e a retirada conjunta das saídas modificou as decisões dos revisores. Passar a inconclusivo não equivale a reconhecer corretamente o vínculo, e não há estimativa de falsos positivos da coorte inteira nesse subconjunto.

### 39. Backup: instruções
A distinção entre arquivo escrito e mensagem efetivamente enviada é necessária, pois o carregador da v3 interrompia o sistema no primeiro separador. Na v4, a política de evidência antecede o separador e permite a abstenção diante de informação insuficiente.

### 40. Backup: formalização
As perdas incorporam probabilidade, consequências do erro e custo da revisão. A ação LLM inclui a possibilidade de inconclusivo e seu encaminhamento ao humano, sob capacidade disponível e após aplicação das guardas.

### 41. Backup: base inteira (comparação), sensibilidade
Os exercícios de exclusão e de expectativas por banda pertencem à cascata sobre a base inteira. Avaliam dependências das escolhas e não são intervalos de confiança. A sobreposição residual e a seleção adaptativa permanecem.

### 42. Backup: privacidade
O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). A ausência de nomes no dossiê não transforma o extrato em dado anônimo nem elimina risco de reidentificação. O hash determinístico das execuções históricas pode ser recalculado por quem detém os identificadores de origem. A exportação restrita com `pair_id` por HMAC foi implementada, sem depósito da matriz nem saneamento retroativo dos artefatos históricos.

### 43. Backup: base inteira (comparação), vigilância REFREC
A reserva mantém a comparação histórica da cascata sobre a base inteira: 194 aceitos, 15 perdidos e 38 vínculos pendentes ao revisor humano, totalizando 247. O fechamento de P é (211+4)+13+19=247, com custo LLM estimado de US$ 0,94. A proporção aninhada de P no *label* não recebe IC. Os cenários de sensibilidade humana são 95%, 99% e 100%, não medidas do serviço.

### 44. Backup: base inteira (comparação), confirmação REFREC
Na base inteira, a cascata aceita 132, perde 28 e encaminha 87 vínculos, totalizando 247. O fechamento de P é (211+8)+27+1=247, com custo LLM estimado de US$ 0,11. Encaminhar não é perder o vínculo. A sensibilidade humana real não foi medida; os cenários de 95%, 99% e 100% tratam a especificidade como perfeita, sem modelar FP humanos.

### 45. Backup: análises de sensibilidade do arranjo e comparadores
A tabela de reserva preserva os braços de sensibilidade e os mecanismos de perda, sem substituir P por outro resultado depois da avaliação. As perdas de P por guarda/custo/LLM são 1/6/6 na vigilância e 1/22/4 na confirmação. A posição dos FP frente aos limiares permanece rastreável. O registro interno precede a S3 (avaliação com os *labels*) em 49 minutos; o resultado *a posteriori* já era conhecido e o código registrado está ausente do git.

### 46. Backup: análise de estabilidade com registro interno, não confirmatória
O bloco numérico (N) varia *threads* e semente; o bloco de partição (R) varia as partições. A análise de estabilidade não é confirmatória nem valida generalização. A dominância de P-Platt foi 7/20 e 4/20 na vigilância, 7/20 e 5/20 na confirmação, sem dominância inversa. No bloco de partição (R), o Jaccard mediano de P na vigilância é 0,43 em relação à partição original, FP vão de 9 a 31 e perdas de 9 a 21. Essa amplitude marginal não mede a incerteza da diferença pareada. Uma diferença de um vínculo em 36 não sustenta superioridade.

### 47. Backup: subamostragem
A subamostragem é exploratória e não melhorou a fila nem a calibração de forma que justifique substituir o treino na prevalência original. Na isotônica, a mediana do topo 96 cai de 22 para 13 com 1:1 e 18,5 com 1:5; 1:20 e 1:50 ficam dentro da variação entre sementes. O Brier mediano piora em todas as proporções. Não se infere melhora de equidade. A referência original tem uma única execução, não intervalo entre sementes. A queda de 25 p.p. no passo 6 tem denominador de 4 PAR=2. Fonte: `subamostragem_treino_resumo.csv:24,25,786`, população abaixo do corte, variante isotonic.

### 48. Backup: filtro de NONMATCH
Na vigilância, em k=96 são 74 pares avaliados pelo LLM como NONMATCH e 22 revisões humanas, com 13 PAR=2 contra 6 sem filtro. VPN: 87,8% em k=96; 78,0% em k=58. Na confirmação, no topo 125, também são 22 revisões e 13 vínculos; VPN de 90,3%, mas 10 dos 23 vínculos do topo são descartados. As decisões dos PAR=2 coincidem entre modos no topo 125, enquanto há 6 diferenças entre não vínculos. Isso explica as recuperações iguais sem afirmar que todas as respostas são compartilhadas. Protocolo fixado antes dos desfechos, respostas de datas diferentes; exploração, não validação. Fonte: `filtro_llm.csv:4,5,13`; `filtro_llm_concordancia_modos.csv:18,19`.

### 49. Backup: mecanismos, cenários e registro interno
Os mecanismos da política original são distintos do erro do revisor humano: guarda, decisão por custo e resposta do LLM. Os cenários humanos de 95%, 99% e 100% não são medidas e não modelam FP humanos. O registro interno antecede a S3 em 49 minutos; os horários e a versão dos metadados permanecem documentados na reserva.

## Perguntas prováveis

São 33 respostas, com 60, 58, 58, 61, 66, 63, 60, 68, 57, 70, 63, 62, 55, 55, 57, 59, 56, 62, 58, 67, 65, 60, 58, 52, 52, 61, 59, 60, 63, 55, 57, 65 e 57 palavras, respectivamente, sem contar pergunta e fonte. A primeira reúne a pertinência do probabilístico e o escopo do módulo Python, incluindo a equivalência Java. Mantém-se o limite de 70 palavras, para duração estimada de até 30 segundos. As fontes usam identificadores da tese e CSVs, sem depender da paginação.

### 1. O probabilístico é inútil? Onde está o módulo Python do OpenRecLink?

O corte do OpenRecLink aceitou 211/247 vínculos, ou 85,4%. A entrega Python é o CdR, não a blocagem. A equivalência com o Java de Peçanha (2015) foi avaliada: 100,0% nas datas por igualdade/um dígito, 99,2% nas abreviações e nota idêntica em 11,2% dos pares sintéticos. Inversões pesam 0,8 em Lucena/Java e 1,0 no Python, limitação declarada. Não há equivalência integral.

Fonte: `capitulo_dados.tex`, seção Padrão-ouro (`sec:padrao-ouro`); `estudo1_comparador.tex`, Resultados (`sec:e1-resultados`). Slides 6, 9 e 10.

### 2. Aprendizado de máquina é só exploração de possibilidades?

Há exploração, e seus limites estão declarados: F1 de 0,931 na ablação e 0,943 no fluxo Modal usaram limiares escolhidos no teste. O GZ-CMD usa outro estimador, com 30 atributos, calibração aninhada e agrupamento por REFREC, com AUC de 0,995 e AP de 0,862 na área cinza da nota. Isso sustenta discriminação interna, não validação externa da política.

Fonte: `estudo3_estrategias.tex`, Limitações (`sec:e3-limitacoes`); `estudo4_gzcmd.tex`, Classificador supervisionado e calibração fora da amostra (`sec:e4-classificador-oof`). Slides 11 e 13.

### 3. Isso já foi feito por Lages ou pelo CIDACS?

Há precedentes de classificação supervisionada e redução da revisão. O CIDACS constitui referência brasileira de infraestrutura e métodos de *linkage*. Aqui, avaliam-se ordenação, esforço e política auditável numa base de razão 1:249. Lages trabalha com razão 1:5 e outro delineamento. Não foi realizada comparação empírica entre as implementações; a tabela de posicionamento não é uma prova de superioridade.

Fonte: `capitulo2.tex`, Classificação supervisionada, experiência brasileira e `tab:posicionamento`; `estudo1_comparador.tex`, Discussão (`sec:e1-discussao`). Slide 25.

### 4. Por que F1 próximo de 0,93 aqui e 0,996 em Lages?

O valor 0,943 pertence ao XGBoost do fluxo Modal, com seleção de limiar no teste; o QW1 tem 0,920. Lages reporta 0,996 em outro conjunto e protocolo, com razão de classes distinta. Essa comparação não identifica a causa da diferença nem ordena estimadores. Aqui, o resultado por faixa OOF é apresentado separadamente, evitando atribuir o desempenho global apenas aos casos difíceis.

Fonte: `capitulo2.tex`, `tab:posicionamento`; `tab:qw1-deep-learning`; `tab:ablation-best-category`; `tab:e4-desempenho-faixas-oof`. Slides 11, 25 e 33.

### 5. Como foi construído o padrão-ouro?

Luís Guilherme Buteri Alves realizou o pré-processamento, e Daniela Medina Coeli Bastos configurou a vinculação e revisou os potenciais pares. Dos 247 vínculos, 211 vieram do corte e 36 da revisão abaixo dele. O corte integra o *label*; o CdR só é documentado no passo aberto. Não houve segunda classificação independente nem kappa. A referência incorpora o corte e limita a sensibilidade aos potenciais pares exportados.

Fonte: `capitulo_dados.tex`, Padrão-ouro (`sec:padrao-ouro`). Slides 6 e 7.

### 6. O arranjo é determinístico?

O corte e as guardas são determinísticos, mas a cadeia inclui probabilidade estimada, custos e Claude Haiku 5.5, cuja geração é estocástica. Na vigilância, são 211 vínculos pelo corte e quatro pela política, com 13 perdidos e 19 vínculos entre 96 encaminhamentos ao revisor humano. Abaixo do corte, as aceitações vieram do LLM, com 21 falsos positivos. Não há garantia de repetibilidade integral.

Fonte: `data/arranjo_corte_metricas.csv:2,12`; `data/arranjo_corte_s1_resumo.csv:2-15`; `sec:e4-politica-v31`. Slides 17, 19 a 21.

### 7. Como o arranjo se compara à revisão integral?

A revisão integral tem amplitude de 0,73 p.p. entre sensibilidade humana de 95% e 100%; nas estratégias de carga reduzida, o máximo é 0,53 p.p. Há cenário de 99%. As perdas são de cobertura: vínculos não chegam ao revisor humano. A especificidade humana é assumida perfeita. Os cenários não medem erro real nem FP humanos; a curva explicita o orçamento.

Fonte: `data/cenarios_sensibilidade_humana.csv`; `data/arranjo_corte_metricas.csv:8,12,26,30`. Slides 6 e 28.

### 8. Dados identificados foram enviados à nuvem?

O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). O autor recebeu apenas extrato pseudonimizado, sem registros nominais nem correspondência com pessoas. A base legal é a LGPD (arts. 7º, IV, 11, II, c, e 13). Dossiês enviados a APIs externas continham nota, atributos derivados e pseudônimo; nas execuções auditadas, retiraram-se as saídas do motor. Persiste risco de reidentificação.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`), ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`). Slides 27 e 42.

### 9. Como discutir equidade sem raça/cor?

A falta de raça/cor impede examinar disparidades raciais. Avaliaram-se sensibilidade e VPP, não toda a igualdade de oportunidades de erro. No passo 11, P tem 42,5% e 47,5% nos dois modos; na vigilância, fica abaixo dos 45,0% da base inteira (comparação). A diferença reflete sobretudo o corte e o *label*. Ausência de rejeição demográfica não demonstra equidade.

Fonte: `data/equidade_arranjo_group_metrics.csv`, passo; `data/equidade_arranjo_pairwise_tests.csv`, rejeições de P. Slides 23 e 24.

### 10. Por que o arranjo, e não a cascata publicada?

O arranjo preserva 211 vínculos e acrescenta quatro ou oito pela política, com 13 ou 27 perdidos. Nos 36 PAR=2, aceita 4 e 8 contra 10 e 8 da cascata sobre a base inteira, na vigilância e confirmação. A vantagem vem do corte, não de maior recuperação abaixo dele. O comparador *a posteriori* já era conhecido antes do registro; este cobre recalibração, braços e avaliação, não o enquadramento escolhido depois.

Fonte: `data/arranjo_corte_metricas.csv:2,4,12,22,30`; `data/arranjo_corte_par2.csv:3,7,12,16`; `gzcmd/prereg_arranjo_corte.yaml`. Slides 17, 19, 20, 26 e 28.

### 11. Por que não adotar P-Platt?

P é principal porque foi registrado antes. P-Platt, análise de sensibilidade, dominou P na avaliação original da vigilância. A análise de estabilidade com registro interno, não confirmatória, não sustentou a dominância: vigilância 7/20 e 4/20, confirmação 7/20 e 5/20, nos blocos numérico (N)/partição (R). Na vigilância, menos FP coexistem com perdas medianas de um/dois vínculos a mais. Adotar *Platt scaling* exige dados novos.

Fonte: `data/arranjo_corte_metricas.csv:12,14`; `data/robustez_arranjo_resumo.csv:2-5,404,488`. Slides 22 e 46.

### 12. Qual é a carga comparável à do serviço?

A ordem histórica não foi documentada. Se tivesse sido decrescente por escore do OpenRecLink, o cenário seria de 2.156 pares por passo ou 2.522 no limiar global, equivalentes a 718 h 40 min e 840 h 40 min sob 20 min por par. Não são horas observadas. A curva de esforço evita depender desse cenário; o serviço define a aceitabilidade das perdas.

Fonte: `data/carga_abaixo_corte.csv:5,9`; `data/carga_abaixo_corte_horas.csv:3,2`. Slides 6 e 20.

### 13. O que significa VPP de 16% do LLM?

Na vigilância de P, quatro das 25 respostas de vínculo abaixo do corte eram vínculos, VPP de 16,0%, com IC de 6,4 a 34,7%. As demais 21 eram falsos positivos. Na confirmação, foram oito de 12, ou 66,7%. Esses resultados pertencem à região selecionada pela política, não estimam uma capacidade geral de Claude Haiku 5.5.

Fonte: `data/arranjo_corte_vpp_llm.csv:2,8`. Slide 21.

### 14. De onde vêm os vínculos perdidos?

Em P, as perdas distribuem-se por guarda/custo/LLM, na vigilância: 1/6/6, totalizando 13; na confirmação: 1/22/4, totalizando 27. Os encaminhamentos são desfechos do arranjo; não se atribuem exclusivamente ao LLM. As aceitações e os falsos positivos abaixo do corte vieram do LLM. Essa decomposição identifica os ramos da decisão, sem isolar causalmente o efeito da calibração.

Fonte: `data/arranjo_corte_perdas_mecanismo.csv:2-4,50-52`. Slide 20.

### 15. Como justificar perder 75% dos vínculos da revisão na confirmação?

São 27 dos 36 PAR=2, ou 75,0%, não uma manutenção da recuperação do serviço. A confirmação encaminha somente 17 potenciais pares ao revisor humano, mas a redução de encaminhamentos vem com essas perdas. O estudo documenta a troca, não demonstra sua aceitabilidade clínica ou epidemiológica. A adoção depende de decisão do serviço e validação em dados novos.

Fonte: `data/arranjo_corte_par2.csv:16`; `data/arranjo_corte_metricas.csv:30`. Slides 20 e 28.

### 16. Por que as taxas congeladas não anteciparam os falsos positivos?

As taxas foram estimadas em coortes com potenciais pares mais fáceis e denominadores mínimos. A seleção abaixo do corte muda a composição. Em P na vigilância, previam cerca de 9,06 falsos positivos, contra 21 observados. A validação cruzada não garante transferência dessas proporções. É necessário estimar e validar o erro na região de uso, sem realimentação oportunista pelos desfechos.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `data/arranjo_corte_taxas_crossfit.csv`. Slides 21 e 26.

### 17. A AUC de 0,995 elimina o viés de incorporação?

Não. A AUC descreve discriminação interna no padrão-ouro disponível; não mede independência do *label* nem calibração absoluta. O corte define PAR=1; o CdR só é documentado no passo aberto. O agrupamento por REFREC reduz uma dependência, mas não separa simultaneamente COMPREC nem cria referência externa. O GZ-CMD não usa os atributos de óbito dos experimentos exploratórios.

Fonte: `tab:e4-desempenho-faixas-oof`; `data/arranjo_corte_metadata.json`, `verification_bias` e `incorporation_bias`. Slides 13, 16 e 26.

### 18. O arranjo pode ser reproduzido somente com o Zenodo?

Não. O depósito e o pacote cobrem a cascata v3.1 sobre a base inteira, não o arranjo. O repositório contém código, registros, configurações e agregados do arranjo e da análise de estabilidade. As saídas por par ficam locais, e o extrato pseudonimizado não é público. Reexecutar as etapas depende de acesso autorizado e das versões registradas, não apenas de instalar o pacote.

Fonte: repositório, `gzcmd/prereg_arranjo_corte.yaml` e `gzcmd/prereg_robustez_arranjo.yaml`; disponibilidade dos dados na tese. Slides 29 e 42.

### 19. O registro interno elimina os desvios de protocolo?

Não. O código inicialmente registrado não está no git, e o suplemento fez chamadas de sensibilidade apesar da vedação original. O comparador *a posteriori* já era conhecido antes do registro; o enquadramento veio depois. Esses fatos limitam a proteção do registro e precisam ser declarados. P continua principal; não se reescreve o protocolo para tornar outra política vencedora.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`, emendas e suplemento; `data/arranjo_corte_metadata.json`. Slide 26.

### 20. Como escolher a ordenação e o ponto de parada?

Nenhuma ordenação domina em todas as cargas. Para igualar os 23 e 9 vínculos de P, O4 requer 106 e 27 revisões; O3, 125 e 29. Em 96 e 17 revisões, O3 deixa 14 e 30 dos 36 fora. Para os 36, O1 requer 2.521 e O3, 23.443. Preservar o corte; escolher probabilidade ou escore e parada pela curva conforme o orçamento humano. É hipótese a validar.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `data/comparador_mesma_carga.csv`. Slides 20 e 28.

### 21. A análise de estabilidade confirma generalização e repetibilidade?

Não. É análise com registro interno, não confirmatória, nos mesmos dados. Em P na vigilância, bloco de partição (R), o Jaccard mediano em relação à partição original é 0,43 no bloco de partição (R). Nesse bloco, FP variam de nove a 31 e perdas de nove a 21; a réplica original é um ponto dessa distribuição. Essa instabilidade não valida externamente nem garante repetibilidade integral.

Fonte: `data/robustez_arranjo_resumo.csv:270,278,654`; `data/robustez_arranjo_metadata.json`. Slide 46.

### 22. Qual é a vantagem demonstrada frente à prática do serviço?

P não reproduz 13 vínculos na vigilância e 27 na confirmação. Aceita 25 e 12 pares abaixo do corte: 4 e 8 são PAR=2; dos demais, 21 e 4 são FP contra o *label*, rejeitados ou não examinados pelo serviço. Não há melhora de acurácia. A curva explicita a cobertura e a troca sob orçamento; sua adoção depende de validação.

Fonte: `data/arranjo_corte_metricas.csv:12,30`; `data/arranjo_corte_vpp_llm.csv:2,8`. Slides 19 e 28.

### 23. Houve varredura de custos no arranjo?

Não houve varredura no arranjo. Na confirmação, C_FP é cinco vezes C_FN, e 22 perdas vieram de AUTO_NONMATCH por custo. Isso descreve o perfil avaliado, não sua aceitabilidade nem um efeito causal isolado da calibração. A varredura apresentada no Estudo 4 foi feita sobre a base inteira e não pode ser transferida ao arranjo como análise já executada.

Fonte: `gzcmd/gzcmd_v3_config.yaml`; `data/arranjo_corte_perdas_mecanismo.csv:51`; `estudo4_gzcmd.tex`, varredura na base inteira. Slides 18 e 20.

### 24. O que muda na mesma carga com MATCH do LLM ao revisor humano?

Na vigilância, P usa 121 revisões para 23 PAR=2; O3 recupera 22 nessa carga e iguala P em 125. Na confirmação, ambos recuperam 9 em 29. Revisor perfeito é hipótese dos dois lados. Há 344 pares adicionais avaliados pelo LLM na vigilância. Uma diferença de um vínculo em 36 não sustenta superioridade.

Fonte: `data/comparador_p_match_humano.csv:2,3`. Slide 22.

### 25. Há fonte para a ordem histórica de revisão?

Não. Oliveira (2026, p. 34) informa revisão manual da área cinza, sem descrever ordem, limite ou parada. Ordem decrescente de escore do OpenRecLink é hipótese. Portanto, 2.156 e 2.522 são cenários, não reconstrução histórica. O CdR só é documentado no passo aberto, com 5 pares, não nos passos 6, 9 e 11.

Fonte: `capitulo_dados.tex`, discussão de Oliveira, Quadros 5 e 6; `data/carga_abaixo_corte.csv:5,9`. Slides 6 e 7.

### 26. Onde estão os falsos positivos em relação ao escore do OpenRecLink?

Na vigilância, 17 dos 21 FP de P estão em escore do OpenRecLink ≥ −5,648 e 4 abaixo; na confirmação, os 4 estão acima desse limiar. Frente ao limiar por passo, são 11 acima e 10 abaixo na vigilância; 4 e 0 na confirmação. A contagem é exploratória *post hoc*; não prova que o serviço examinou ou rejeitou individualmente esses pares.

Fonte: `data/fp_arranjo_por_regiao.csv:8,9,16,17,24,25,32,33`. Slides 19 e 45.

### 27. O que protege um registro feito 49 minutos antes?

É registro interno versionado antes da S3, não registro público. O intervalo foi de 49 minutos, de 01:22:17 a 02:11:52 UTC, conforme os metadados na versão `16d5876`. O comparador *a posteriori* já era conhecido, e o código registrado está ausente do git. O registro cobre recalibração, braços e avaliação, não as análises exploratórias posteriores nem o enquadramento escolhido depois.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`; `data/arranjo_corte_metadata.json`; `gzcmd/M_arranjo_corte_desvios.md`. Slides 45 e 49.

### 28. De onde vêm os 20 minutos e os cenários humanos?

São parâmetros de cenário, não medidas do serviço. Vinte minutos por par convertem revisões em ordem de grandeza de horas. A configuração motivou sensibilidade humana de 95%; avaliam-se também 99% e 100%. A especificidade é assumida perfeita: não se modelam FP humanos. A diferença principal é cobertura, não erro de quem revisa. A adoção requer medir tempo, erro e concordância.

Fonte: `gzcmd/gzcmd_v3_config.yaml`; `data/cenarios_sensibilidade_humana.csv`; `data/esforco_arranjo.csv`. Slides 18 e 28.

### 29. Por que Grey-Zone não é a área cinza da nota?

Grey-Zone é a região em que a regra automática requer exame, não apenas [5,9) da nota do CdR. O corte do OpenRecLink usa outro escore. Preservar o corte; escolher probabilidade ou escore e parada pela curva conforme o orçamento humano; nenhuma ordenação domina sempre. Se usar probabilidade aprendida, treinar na prevalência original. Restante: não vínculo. Filtro LLM experimental, sem aceitação autônoma, a validar.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `data/comparador_p_match_humano.csv`; `data/filtro_llm.csv`. Slides 4, 18 e 29.

### 30. O filtro LLM ajuda sem perder vínculos?

Não. Na vigilância, k=96: 74 pares avaliados pelo LLM como NONMATCH e 22 revisões humanas recuperam 13 PAR=2 contra 6 sem filtro. VPN: 87,8% em k=96; 78,0% em k=58. Em outro ponto, confirmação k=125, descarta 10 dos 23 vínculos. É hipótese exploratória a validar, com protocolo fixado antes dos desfechos e respostas de datas diferentes.

Fonte: `data/filtro_llm.csv:4,5,13`; `data/filtro_llm_metadata.json`. Slides 21 e 48.

### 31. Por que manter o treino na prevalência original?

A subamostragem não ajudou a fila nem a calibração. No topo 96, a isotônica recupera mediana de 13 com 1:1 e 18,5 com 1:5, contra 22 na prevalência original; 1:20 e 1:50 ficam dentro da variação. O Brier mediano piora em todas as proporções. É análise exploratória, sem justificar mudança de treino ou afirmar melhora de equidade.

Fonte: `data/subamostragem_treino_resumo.csv:18,210,402,594,786`, variante isotonic. Slides 29 e 47.

### 32. E se a sensibilidade humana for 99%?

É um dos três cenários, não medida. De 95% a 100%, a sensibilidade total muda no máximo 0,53 p.p. nas estratégias de carga reduzida e 0,73 p.p. na revisão integral. A ordem entre as estratégias não se inverte. A especificidade humana é assumida perfeita. Os vínculos fora do orçamento não chegam ao revisor humano: a perda principal é de cobertura, não erro de quem revisou.

Fonte: `data/cenarios_sensibilidade_humana.csv`, cenários 0,95, 0,99 e 1,00. Slide 28.

### 33. Por que não trocar o revisor humano pelo LLM?

Como aceitador, o LLM teve VPP de 16,0% na vigilância. Com MATCH encaminhado ao humano, a política recupera no máximo 1 vínculo a mais que O3 na mesma carga, com 344 pares adicionais avaliados pelo LLM. Uma diferença de um vínculo em 36 não sustenta superioridade. O uso possível é filtro experimental de NONMATCH, nunca aceitação autônoma.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `data/comparador_p_match_humano.csv:2,3`; `data/filtro_llm.csv`. Slides 21, 22 e 29.

## Figuras e critérios de entrega

Figuras reaproveitadas, sem alterar os originais:

| Slide | Arquivo | Uso e ressalva |
| :-- | :-- | :-- |
| 20 | `text/slides/figures/curva_esforco.pgf` | Curva exploratória por script, com pontos de P |
| 9 | `figures/fig_cdr_gui_mapeamento.png` | Interface do Comparador de Registros, versão 0.1 (*commit* 3b43598), dados sintéticos |
| 31 | `text/latex/figures/fig_cap8_cascata_v31.png` | Cascata por REFREC sobre a base inteira (comparação) |
| 36 | `figures/imbalance_sensitivity.pgf` | Estratégias de balanceamento no Random Forest |
| 37 | `text/latex/fig_doc/epi_deteccao.pgf` | Detecção no teste da ablação, não arranjo nem cascata |

A captura do slide 9 está em `../../figures/fig_cdr_gui_mapeamento.png`, caminho relativo ao diretório de compilação `text/slides`, sem `graphicspath` adicional. A tabela do slide 25 reduz a tabela R2 e incorpora Paiva (2025) do texto adjacente, indicando que utilizou base distinta da desta tese, sem lhe atribuir a autoria do protótipo Java.

Critérios editoriais: menção ao referencial clássico restrita ao slide 8; probabilidade recalibrada abaixo do corte; arranjo P principal e cascata REFREC sobre a base inteira como comparação; corte operacional 211/247 por construção do *label*; curva de esforço exploratória *post hoc*, não registrada; F1 0,943 atribuído ao fluxo Modal; perdas de 13 e 27 vínculos junto do acréscimo automático; limitações antes das conclusões; equivalência Java parcial executada e comparação com Lages não realizada; sensibilidade 2×2 e QW1 local distinguidos dos resultados históricos; exportador com HMAC implementado, matriz não depositada; área cinza da nota apenas para [5,9). As tabelas foram reduzidas no arquivo de slides.

O ensaio foi descartado pelo autor em 04/11/2026. A revisão pela orientadora foi dispensada, conforme a auditoria privada. Os testes automáticos verificam estrutura e compilação, não constituem evidência de teste presencial de projeção.

Compilação oficial da repassada final do PR #52, autorizada pelo autor em 05/11/2026. A tese foi regenerada por `python scripts/build_pdf.py --jobname tese --outname tese`, com BibTeX e três passagens de pdflatex até estabilizar, gerando `text/latex/tese.pdf` e a cópia idêntica `text/pdf/tese.pdf`. Não há `.nlo`, portanto o script oficial não executou makeindex. Os slides foram regenerados por `python text/slides/build_slides.py`, em duas passagens de pdflatex, com `TEXINPUTS=C:/Users/Marquinho/.local/share/opencode/worktree/d66565/pq-f-slides/.texdeps//;`. Foram usadas somente as dependências já existentes nesse worktree.

### Verificação técnica do build oficial após a repassada final

A tese oficial manteve **224 páginas**, e a ficha permanece em **`224 f.: il.`**. Não foi necessária nova edição da ficha nesta etapa. Segue-se a convenção dos builds anteriores: as folhas da ficha correspondem às páginas físicas do PDF.

Os slides têm **49 páginas**, com 30 quadros principais e 19 de reserva. As entradas dos slides não foram modificadas nesta repassada; permanecem as três trocas terminológicas de robustez para estabilidade já existentes no PR #51. Não houve mudança de resultados, números ou fontes dos slides. As 33 respostas têm no máximo 70 palavras, conforme contagem por espaços.

Os logs finais da tese e dos slides não contêm erros LaTeX, referências ou citações indefinidas, pedidos de nova passagem ou marcadores `??`, também ausentes no texto extraído dos PDFs. Permanecem somente os cinco Overfull herdados na tese; nos slides não há Overfull. Não foi realizado novo ensaio presencial de projeção.

Logs e evidências locais, não versionados: `.local/negacoes/ultima-oficial/tese.log`, `.local/negacoes/ultima-oficial/defesa.log`, `.local/negacoes/ultima-oficial/verificacao.json`, `.local/negacoes/ultima-build-tese.txt`, `.local/negacoes/ultima-build-slides.txt` e `.local/negacoes/ultima-antes-depois.md`.

*Commit* de origem antes da repassada final: `fdbe2bbef7c14133596b721d42aa642dc2301fa6`. A árvore compilada final é a do commit de entrega, que inclui as reversões de prosa, os testes editoriais, este roteiro e os três PDFs oficiais. As dependências externas de TeX não fazem parte dessa árvore. Em outra máquina, é necessário instalar Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los em diretório local de dependências.

SHA-256 do PDF oficial dos slides: `c053b5e3dda19f01ec5142f13c69b82316b4d169c64de1e3f27dc1e3adb9d79b`. Relação dos PDFs oficiais:

| Arquivo | Páginas | SHA-256 |
| :-- | --: | :-- |
| `text/slides/defesa.pdf` | 49 | `c053b5e3dda19f01ec5142f13c69b82316b4d169c64de1e3f27dc1e3adb9d79b` |
| `text/latex/tese.pdf` | 224 | `06355073a822ba4c22dcd1f63f83bd7eda73f892386bfaa3f70de07a37956029` |
| `text/pdf/tese.pdf` | 224 | `06355073a822ba4c22dcd1f63f83bd7eda73f892386bfaa3f70de07a37956029` |

Os identificadores da tese servem como remissões ao texto em revisão paralela. Os números novos ou alterados do arranjo e da equidade têm como fonte os CSVs agregados abaixo.

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
| BH, 7 rejeições em P vigilância, todas operacionais | `equidade_arranjo_pairwise_tests.csv:34,35,37,40,41,43,44` |
| BH, 3 rejeições em P confirmação, todas operacionais | `equidade_arranjo_pairwise_tests.csv:151,156,157` |
| BH na base inteira (comparação): 9 na vigilância e 1 na confirmação rejeições | `equidade_arranjo_pairwise_tests.csv:87,92,93,95,98-102,215` |
| 211 aceitos por construção; diferenças de sensibilidade dependem da proporção de PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36; diferenças de VPP dependem dos 21 e 4 FP abaixo do corte | `arranjo_corte_metricas.csv:2,12,30`; `equidade_arranjo_group_metrics.csv`; `equidade_arranjo_global.csv:2,4` |
| Na vigilância, não aceitos por passo 11/6/9: 23/4/5, todos PAR=2; no passo 11, 42,5% em P contra 45,0% na base inteira (comparação) | `equidade_arranjo_group_metrics.csv:17-19,41` |
| Passo 11: 25 dos 36 vínculos com PAR=2; P aceita 2/25 na vigilância e 4/25 na confirmação | `par_por_passo.csv:8-10,17-19,26-28,53-55` |
| Corte isolado nos passos 11/6/9: 15/40 (37,5%), 76/80 (95,0%) e 120/127 (94,5%) | `par_por_passo.csv:5-10,14-19,23-28` |
| PAR=2 aceitos nos passos 6/9/11: 0/2/2 na vigilância e 1/3/4 na confirmação | `par_por_passo.csv:8,17,26,35,44,53` |
| Pendentes de P∪B1 com chamada nova: 10 na vigilância e 14 na confirmação, não de P isolado; reutilização histórica e ausência de avaliação prospectiva | `arranjo_corte_s1_metadata.json:477-481`, campo `pending_P_union_B1_by_mode`; `arranjo_corte_metadata.json:355,357` |
| Cenários de carga sob ordem hipotética: 2.156 por passo e 2.522 global; escore do OpenRecLink global ≥ −5,648, sob hipótese não documentada de revisão decrescente de escore do OpenRecLink | `carga_abaixo_corte.csv:5,9` |
| Hipótese de 20 min por par: por passo, pelo menos 718 h 40 min; global, pelo menos 840 h 40 min; não são tempos observados | `carga_abaixo_corte_horas.csv:3,2` |
| P, vigilância: VPP do LLM 4/25 = 16,0%, IC 6,4 a 34,7%; 9,06 FP previstos contra 21 observados | `arranjo_corte_vpp_llm.csv:2` |
| P, confirmação: VPP 8/12 = 66,7%, IC 39,1 a 86,2%; P-Platt vigilância: 4/7 = 57,1%, IC 25,0 a 84,2% | `arranjo_corte_vpp_llm.csv:8,3` |
| Perdas de P por guarda/custo/LLM: 1/6/6 na vigilância e 1/22/4 na confirmação | `arranjo_corte_perdas_mecanismo.csv:2-4,50-52` |
| Perda de 13/36 = 36,1% e 27/36 = 75,0% dos PAR=2 | `arranjo_corte_par2.csv:7,16` |
| Dominância de P-Platt: vigilância nos blocos numérico (N)/partição (R), 7/20 e 4/20; confirmação 7/20 e 5/20; inversa 0/20 em todos os casos; FP menores em 20/20 nos dois blocos da vigilância | `robustez_arranjo_resumo.csv:2-5` |
| Vigilância: P-Platt perde em mediana 1/2 vínculos a mais, blocos numérico (N)/partição (R) | `robustez_arranjo_resumo.csv:404,488`, diferenças P menos P-Platt −1/−2 |
| P perde mais PAR=2 que o *a posteriori* em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação | `robustez_arranjo_resumo.csv:6,8,10,12`, proporções multiplicadas por 20 |
| P na vigilância, bloco de partição (R): FP de 9 a 31, perdas de 9 a 21 e Jaccard mediano 0,43 em relação à partição original | `robustez_arranjo_resumo.csv:270,278,654` |
| VPP por sexo de P na vigilância sem rejeição: p ajustado BH de 0,378 | `equidade_arranjo_pairwise_tests.csv:3` |
| Mudança máxima da probabilidade de P: 0,65 no bloco numérico (N) e 0,80 no bloco de partição (R) | `robustez_arranjo_resumo.csv:632,652` |
| Estabilidade: 242 potenciais pares novos, US$ 0,6255, arredondados para US$ 0,63 | `robustez_arranjo_metadata.json:206,208` |
| Consumo novo anterior: US$ 0,2448; somado a US$ 0,6255 da análise de estabilidade, sem representar fatura | `arranjo_corte_metadata.json:285,293,330,338`, soma dos custos novos de S2 e suplemento; `robustez_arranjo_metadata.json:208` |

Os resultados históricos de v3, dos estudos 1 a 3 e da restrição na base inteira foram mantidos com as remissões que já constavam dos slides. Não são resultados novos do arranjo, nem foram reestimados nesta fatia.

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

A referência de 2.156 revisões pressupõe ordem decrescente de escore do OpenRecLink, que não foi documentada. Horas são ordem de grandeza a 20 min por par, não medida. As horas poupadas por vínculo não recuperado ou por FP descrevem a troca contra esse cenário, não economia histórica nem benefício de acurácia. Para O3, FP é zero sob revisor perfeito; não se divide por zero. K2 equipara o número total de avaliações de P (humano mais LLM), não o seu orçamento humano.

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

Global: limiar −5,648; por passo: limite compatível com PAR=2 naquele passo. Acima ou igual e abaixo são posições no escore, não prova de exame histórico pelo serviço.

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

### Subamostragem: evidência para manter a prevalência original

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
