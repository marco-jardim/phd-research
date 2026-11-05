# Roteiro da defesa

Base versionada desta revisão S8b: `0b4dc3d55585071c77e8fc82153bf625d6a6738d`. O hash registrado na verificação técnica identifica a árvore compilada, não o commit final. A numeração corresponde aos quadros do PDF, sem sobreposições: 30 principais e 16 de reserva, estes fora dos 40 minutos. O tempo é estimativa, não duração medida. O ensaio foi descartado por decisão do autor em 04/11/2026.

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

### 4. Tese central
A hipótese central é que, abaixo do corte do OpenRecLink, ordenar por probabilidade calibrada fora da amostra concentra os vínculos e reduz a revisão manual. A política de quatro ações com modelo de linguagem de grande porte (LLM) não superou essa ordenação simples; CdR, Comparador de Registros; P, a calibração isotônica registrada internamente como principal. A comparação de ordenações é exploratória *post hoc*, não registrada, e exige validação em dados novos. O objetivo 4 compara recuperação, FP, encaminhamentos, custo e esforço com corte único, duas fronteiras e revisão por orçamento.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36. PAR é o *label*: 1 indica aceitação pelo corte; 2, recuperação pela revisão. O volume e a ordem históricos não foram observados. As duas fronteiras da nota do CdR aceitam 93 vínculos e 2 FP acima de 9, encaminham 21.620 ao revisor humano e projetam 95,7% sob erro de 5,0%. Se a revisão tivesse seguido SCORE decrescente, o cenário seria de pelo menos 2.156 pares por passo ou 2.522 no limiar global. Isso não reconstrói o trabalho histórico.

### 7. Faixas e padrão-ouro
As faixas são da nota do CdR, não do SCORE do OpenRecLink: abaixo de 5 há 39.981 pares e 3 vínculos; em [5,9), 21.620 e 151; a partir de 9, 95 e 93. Houve revisora única, sem classificação independente. Oliveira (2026, p. 34 e 36 a 38, Quadros 5 e 6) documenta CdR apenas no passo aberto, com 5 pares, não nos passos 6, 9 e 11. Os 36 PAR=2 se distribuem em 4, 7 e 25 nesses passos. A referência tem viés de verificação e incorporação, mas não se atribui ao CdR uma exposição não documentada nos três passos.

### 8. Mapa conceitual
O referencial clássico organiza a decisão entre vínculo, não-vínculo e revisão. A tese trabalha depois da geração dos potenciais pares e do cálculo dos subescores, reorganizando a decisão sobre a saída do CdR, sem reestimar os parâmetros de comparação nem substituir a blocagem do OpenRecLink.

### 9. Comparador em Python
O comparador torna explícita a evidência campo a campo, com subescores e uma nota ponderada por pesos fixos. A equivalência parcial com o protótipo Java de Peçanha (2015) foi executada em pares sintéticos. A captura apresenta a interface do Comparador de Registros, versão 0.1, commit 3b43598, com dados sintéticos, não uma interface do GZ-CMD.

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

### 18. Custos e ações
O custo do falso negativo é maior na vigilância, enquanto o custo do falso positivo é maior na confirmação. As quatro ações são comparadas na mesma escala de utilidade, e essa escala não deve ser confundida com dólares nem com custos econômicos medidos no serviço.

### 19. Arranjo sequencial com registro prévio
P conserva os 211 vínculos do corte e acrescenta 4 na vigilância e 8 na confirmação. Não reproduz 13 e 27 dos 36 vínculos recuperados pelo serviço. Aceita, abaixo do corte, 25 e 12 potenciais pares que o serviço rejeitou ou não examinou, sendo 21 e 4 FP contra o *label*. Encaminha 96 pares na vigilância e 17 na confirmação ao revisor humano. O registro interno antecedeu a S3; o comparador do serviço, a carga, o VPP e os mecanismos de perda são análises exploratórias *post hoc*, não registradas. IC, fechamentos e cronologia ficam nos quadros de reserva 43 a 45.

### 20. Curva de esforço
A curva substitui o quadro principal de decomposição, preservado na reserva 45. O1 ordena por SCORE; O2, pela nota CdR; O3, por p_cal_ml OOF; O4, pela probabilidade recalibrada de P, com o mesmo XGBoost. Em 96 revisões, O3 e O4 recuperam 22 PAR=2, O1 19 e O2 12, contra 23 de P, que acrescenta 21 FP e 344 avaliações pelo LLM. Em 17 revisões, P recupera 9; O3, 6; O4, 5; O1, 3; O2, 9. Para igualar P, O3 requer 125 e 29 revisões; O4, 106 e 27; O1, 501 e 51; O2, 4.163 e 11, respectivamente na vigilância e confirmação. O comparador tem 0 FP sob revisor perfeito, que é hipótese; há cenário de 5,0% de erro. A análise é exploratória *post hoc*, não registrada, com viés de verificação que favorece O1 por construção. O que se sustenta é a ordenação, não a política.

### 21. Revisor de referência
Claude Haiku 5.5 com instruções v4 recebe dossiê sem as saídas decisórias do motor, em protocolo de dois agentes e árbitro. Toda aceitação abaixo do corte em P veio do LLM. Na vigilância, somente quatro das 25 aceitações eram vínculos: VPP de 16,0%, com IC de 6,4 a 34,7%. Na confirmação, são oito de 12, ou 66,7%. P-Platt tem quatro de sete na vigilância, ou 57,1%. As taxas congeladas previam cerca de 9,06 falsos positivos em P na vigilância, contra 21 observados. A transferência de proporções de coortes com potenciais pares mais fáceis subestimou o erro nessa região. A recomendação é encaminhar MATCH do LLM abaixo do corte ao revisor humano, não à aceitação automática, e monitorar essas respostas. Trata-se de proposta exploratória *post hoc*, não registrada, que exige avaliação registrada própria, não de resultado observado ou de política já avaliada.

### 22. Calibração não valida a política
P permanece principal porque foi registrado antes. P-Platt, análise de sensibilidade com *Platt scaling*, dominou P na avaliação original, mas a dominância não se sustentou na robustez não confirmatória: 7/20 e 4/20 na vigilância; 7/20 e 5/20 na confirmação, blocos numérico (N) e de partição (R). Menos FP coexistem com perdas adicionais. Na mesma carga humana, a recuperação pela ordenação simples é próxima da de P sem acrescentar FP sob revisor perfeito, nem avaliações pelo LLM. A política não superou a ordenação; não se escolhe outra política após conhecer os resultados.

### 23. Equidade por sexo
No arranjo P de vigilância, o VPP é de 86,3% no sexo feminino e 93,3% no masculino, sem rejeição após Benjamini-Hochberg, p ajustado de 0,378. Na cascata sobre a base inteira (comparação), é de 87,9% e 84,0%. A aceitação do arranjo combina corte e respostas de vínculo do LLM; o encaminhamento humano conta como não aceito. Os 211 vínculos acima do corte são aceitos por construção. As diferenças de sensibilidade entre subgrupos dependem da proporção de vínculos PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36 vínculos. As diferenças de VPP dependem dos 21 e quatro falsos positivos abaixo do corte. O poder é limitado por esses números e pela incorporação do corte ao *label*; ausência de rejeição não comprova equidade. Raça/cor não está disponível.

### 24. Desigualdade operacional
A disparidade por passo é herdada do corte e do *label*, não um efeito isolado do GZ-CMD. O corte sozinho aceita 15 dos 40 vínculos do passo 11, ou 37,5%, contra 76 de 80 no passo 6 (95,0%) e 120 de 127 no passo 9 (94,5%). No passo 11 estão 25 dos 36 vínculos com PAR=2. O arranjo aceita dois desses 25 na vigilância e quatro na confirmação, chegando a 17/40 (42,5%) e 19/40 (47,5%). Nos passos 6/9/11, as aceitações PAR=2 são 0/2/2 na vigilância e 1/3/4 na confirmação, conforme `data/par_por_passo.csv`. Nos passos 6 e 9, a sensibilidade do arranjo é de pelo menos 95,0% nos dois modos. Na vigilância, fica abaixo da cascata sobre a base inteira (comparação) no passo 11: 42,5% contra 45,0%. Após Benjamini-Hochberg, são sete e três rejeições em P, todas operacionais; na base inteira (comparação), nove e uma. A contagem não ordena equidade. A equidade do processo completo depende da revisão humana ainda não observada.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. A tese propõe sua composição operacional com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
O corte integra o *label*, e a revisora foi única, sem classificação independente. O CdR só é documentado no passo aberto. A ordem histórica de revisão é desconhecida; 2.156 pares são cenário, não carga observada. O viés de verificação favorece O1; REFREC não separa COMPREC, e os 36 PAR=2 limitam o poder. Respostas históricas reutilizadas e taxas com denominadores mínimos impedem interpretação prospectiva. O comparador *a posteriori* já era conhecido antes do registro interno, que cobre recalibração, braços e avaliação. O código registrado está ausente do git, e houve chamadas suplementares não previstas, desvios de protocolo. Ordenação, carga, VPP, perdas e recomendação MATCH são análises ou propostas exploratórias *post hoc*, não registradas. A robustez mede estabilidade, não generalização.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. O processamento LLM utilizou APIs externas. A exportação restrita com identificador de par por HMAC foi implementada em `scripts/export_matriz_desidentificada.py`, mas a matriz não foi depositada e não se afirma saneamento retroativo das execuções. A equivalência Java parcial foi executada; a comparação empírica com Lages não foi realizada.

### 28. Ordenação, política e esforço
A ordenação por probabilidade calibrada concentra os vínculos; a política com LLM não a superou. O3 precisa de 125 revisões para igualar 23 PAR=2 de P na vigilância e de 29 para igualar 9 na confirmação. P não reproduz 13 e 27 vínculos do serviço e aceita 21 e 4 FP contra o *label*. No cenário de 2.156 revisões e 20 min por par, P poupa 2.060 revisões e 686,7 horas na vigilância; 2.139 e 713,0 horas na confirmação. Na mesma carga humana, O3 não recupera 14 e 30 vínculos, com 0 FP sob revisor perfeito. São ordens de grandeza, não horas medidas nem economia histórica observada. A decisão sobre a aceitabilidade cabe ao serviço.

### 29. Contribuições
As contribuições são a decomposição por PAR, que separa o peso dos PAR=1 do pequeno acréscimo nos PAR=2; a curva de esforço como instrumento de decisão; a falha de transferência das taxas do LLM e a ancoragem; a política auditável de quatro ações, o CdR em Python, o pacote e o *benchmark* abertos; e o resultado negativo com registro interno. Recomenda-se preservar o corte, revisar por p_cal_ml decrescente até o orçamento humano e encaminhar MATCH do LLM ao revisor humano, não à aceitação automática, com monitoramento. São recomendações exploratórias *post hoc*, dependentes de validação registrada em dados novos.

### 30. Agradecimentos
O trabalho dependeu da orientação, da equipe responsável pelos dados e da revisão manual. A curva explicita a troca sem demonstrar melhora de acurácia sobre o serviço. P não reproduz 13 vínculos na vigilância e 27 na confirmação, com 21 e 4 FP. A política não superou a ordenação simples. Preservar o corte e revisar por p_cal_ml é recomendação exploratória que exige validação registrada em dados novos.

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
O quadro antes principal fica como comparação histórica entre v3 e v3.1 sobre a base inteira. Na v3.1, são 194 aceitos, 15 perdidos e 38 vínculos ao revisor humano, totalizando 247. Não se devem atribuir esses números ao arranjo.

### 44. Backup: base inteira (comparação), confirmação REFREC
Na cascata v3.1 sobre a base inteira, são 132 aceitos, 28 perdidos e 87 vínculos ao revisor humano, totalizando 247. Os 127 encaminhamentos e a ausência de falso positivo observado pertencem a essa comparação, não a P.

### 45. Backup: análises de sensibilidade do arranjo e comparadores
P continua principal. P-Platt troca a isotônica por *Platt scaling*; A3 também restringe o ajuste do classificador aos potenciais pares abaixo do corte; P-PAR2-grey estima o erro de não-vínculo apenas com PAR=2 nas bandas agrupadas de [5,9). São análises de sensibilidade. B1 mantém a base inteira (comparação) e a probabilidade original, mudando as taxas por validação cruzada: na vigilância, mantém 194 aceitos, com 40 falsos positivos contra 34 da cascata. P-Platt apresenta na vigilância três falsos positivos, 77 encaminhamentos e US$ 0,30, contra 21, 96 e US$ 0,94 de P, com os mesmos 13 perdidos. O comparador *a posteriori* é o corte seguido da cascata publicada, sem recalibração, já conhecido antes do registro. Na vigilância, aceita 221, perde nove e encaminha 90, contra 13 perdidos e 96 encaminhamentos de P, mas tem 34 falsos positivos contra 21. Na confirmação, tem zero falso positivo e 17 perdidos contra quatro e 27 de P, mas encaminha 51 potenciais pares contra 17. Os contrastes não autorizam substituir P depois de conhecidos os resultados.

### 46. Backup: robustez com registro prévio, não confirmatória
A robustez tem dois blocos: numérico (N) varia *threads* e semente do classificador; partição (R) varia as partições. Não avalia generalização, mas estabilidade nos mesmos dados. A dominância original de P-Platt não foi sustentada: 7/20 e 4/20 na vigilância, 7/20 e 5/20 na confirmação, com dominância inversa 0/20 em ambos os modos e blocos. Na vigilância, P-Platt mantém menos FP em 20/20 réplicas de cada bloco, mas perde em mediana um e dois vínculos a mais, com menos encaminhamentos humanos e ao LLM. P perde mais PAR=2 que o comparador *a posteriori* em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação.

Para P na vigilância, o Jaccard mediano do lote LLM em relação à partição original é 0,43 no bloco de partição (R). Nesse bloco, os falsos positivos variam de nove a 31 e as perdas de nove a 21; a réplica original é um ponto dessa distribuição. A mudança máxima da probabilidade é de 0,65 no bloco numérico (N) e 0,80 no bloco de partição (R). As chamadas novas da robustez abrangeram 242 potenciais pares, com custo estimado de US$ 0,6255, cerca de US$ 0,63. O consumo novo do arranjo soma os US$ 0,2448 anteriores e US$ 0,6255 da robustez, sem representar fatura nem reestimar as taxas congeladas.

## Perguntas prováveis

São 29 respostas, com 60, 58, 58, 61, 66, 63, 57, 68, 57, 70, 60, 60, 55, 55, 57, 59, 56, 60, 58, 60, 65, 60, 58, 57, 50, 59, 59, 55 e 53 palavras, respectivamente, sem contar pergunta e fonte. A primeira reúne a pertinência do probabilístico e o escopo do módulo Python, incluindo a equivalência Java. Mantém-se o limite de 70 palavras, para duração estimada de até 30 segundos. As fontes usam identificadores da tese e CSVs, sem depender da paginação.

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

A revisão integral projeta 99,3% sob erro humano de 5,0%. P encaminha 96 pares na vigilância e 17 na confirmação, projetando 94,4% e 89,0%, com 13 e 27 vínculos perdidos. As 2.156 revisões são cenário de ordem decrescente de SCORE, não volume histórico. Projeções não substituem decisões humanas; a curva de esforço compara ordenações sob orçamento explícito.

Fonte: `data/esforco_arranjo.csv:2,6`; `data/arranjo_corte_metricas.csv:8,12,26,30`. Slides 6, 20 e 28.

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

P é principal porque foi registrado antes. P-Platt, análise de sensibilidade, dominou P na avaliação original da vigilância. A robustez com registro prévio, não confirmatória, não sustentou a dominância: vigilância 7/20 e 4/20, confirmação 7/20 e 5/20, nos blocos numérico (N)/partição (R). Na vigilância, menos FP coexistem com perdas medianas de um/dois vínculos a mais. Adotar Platt exige dados novos.

Fonte: `data/arranjo_corte_metricas.csv:12,14`; `data/robustez_arranjo_resumo.csv:2-5,404,488`. Slides 22 e 46.

### 12. Qual é a carga comparável à do serviço?

A ordem histórica não foi documentada. Se tivesse sido decrescente por SCORE, o cenário seria de 2.156 pares por passo ou 2.522 no limiar global, equivalentes a 718 h 40 min e 840 h 40 min sob 20 min por par. Não são horas observadas. A curva de esforço evita depender desse cenário; o serviço define a aceitabilidade das perdas.

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

Não. O depósito e o pacote cobrem a cascata v3.1 sobre a base inteira, não o arranjo. O repositório contém código, registros, configurações e agregados do arranjo e da robustez. As saídas por par ficam locais, e o extrato pseudonimizado não é público. Reexecutar as etapas depende de acesso autorizado e das versões registradas, não apenas de instalar o pacote.

Fonte: repositório, `gzcmd/prereg_arranjo_corte.yaml` e `gzcmd/prereg_robustez_arranjo.yaml`; disponibilidade dos dados na tese. Slides 29 e 42.

### 19. O registro prévio elimina os desvios de protocolo?

Não. O código inicialmente registrado não está no git, e o suplemento fez chamadas de sensibilidade apesar da vedação original. O comparador *a posteriori* já era conhecido antes do registro; o enquadramento veio depois. Esses fatos limitam a proteção do registro e precisam ser declarados. P continua principal; não se reescreve o protocolo para tornar outra política vencedora.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`, emendas e suplemento; `data/arranjo_corte_metadata.json`. Slide 26.

### 20. Por que manter Grey-Zone e mudar o objetivo?

O nome remete à região que a regra automática não decide, no conceito de Fellegi e Sunter, não exclusivamente à faixa [5,9) da nota do CdR. O objetivo agora trata dos potenciais pares não aceitos pelo corte: desenvolver e avaliar a cascata e comparar recuperação, erros, encaminhamentos, custo e carga com a prática do serviço, corte único e duas fronteiras.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `sec:area-cinza`; `data/carga_abaixo_corte.csv`. Slides 4, 8 e 17.

### 21. A robustez confirma generalização e repetibilidade?

Não. É análise com registro prévio, não confirmatória, nos mesmos dados. Em P na vigilância, bloco de partição (R), o Jaccard mediano em relação à partição original é 0,43 no bloco de partição (R). Nesse bloco, FP variam de nove a 31 e perdas de nove a 21; a réplica original é um ponto dessa distribuição. Essa instabilidade não valida externamente nem garante repetibilidade integral.

Fonte: `data/robustez_arranjo_resumo.csv:270,278,654`; `data/robustez_arranjo_metadata.json`. Slide 46.

### 22. Qual é a vantagem demonstrada frente à prática do serviço?

P não reproduz 13 e 27 dos 36 vínculos recuperados pelo serviço, na vigilância e confirmação. Aceita 25 e 12 potenciais pares abaixo do corte que o serviço rejeitou ou não examinou, com 21 e 4 FP contra o *label*. Não há melhora de acurácia. A ordenação concentra vínculos; a curva explicita a troca, que depende da decisão do serviço.

Fonte: `data/arranjo_corte_metricas.csv:12,30`; `data/arranjo_corte_vpp_llm.csv:2,8`; `data/comparador_mesma_carga.csv`. Slides 19, 20 e 28.

### 23. Houve varredura de custos no arranjo?

Não houve varredura no arranjo. Na confirmação, C_FP é cinco vezes C_FN, e 22 perdas vieram de AUTO_NONMATCH por custo. Isso descreve o perfil avaliado, não sua aceitabilidade nem um efeito causal isolado da calibração. A varredura apresentada no Estudo 4 foi feita sobre a base inteira e não pode ser transferida ao arranjo como análise já executada.

Fonte: `gzcmd/gzcmd_v3_config.yaml`; `data/arranjo_corte_perdas_mecanismo.csv:51`; `estudo4_gzcmd.tex`, varredura na base inteira. Slides 18 e 20.

### 24. O que muda com a mesma carga e a ordenação simples?

Em 96 revisões, O3/O4 recuperam 22 PAR=2, O1 19 e O2 12, contra 23 de P, que acrescenta 21 FP e 344 avaliações pelo LLM. Em 17 revisões, P recupera 9; O3, 6; O4, 5; O1, 3; O2, 9. O comparador tem 0 FP sob revisor perfeito, hipótese, não observação. É comparação exploratória *post hoc*, não registrada.

Fonte: `data/comparador_mesma_carga.csv:2,7,12,17,22,27,32,37`. Slides 20 e 22.

### 25. Há fonte para a ordem histórica de revisão?

Não. Oliveira (2026, p. 34) informa revisão manual da área cinza, sem descrever ordem, limite ou parada. Ordem decrescente de SCORE é hipótese. Portanto, 2.156 e 2.522 são cenários, não reconstrução histórica. O CdR só é documentado no passo aberto, com 5 pares, não nos passos 6, 9 e 11.

Fonte: `capitulo_dados.tex`, discussão de Oliveira, Quadros 5 e 6; `data/carga_abaixo_corte.csv:5,9`. Slides 6 e 7.

### 26. Onde estão os falsos positivos em relação ao SCORE?

Na vigilância, 17 dos 21 FP de P estão em SCORE ≥ −5,648 e 4 abaixo; na confirmação, os 4 estão acima desse limiar. Frente ao limiar por passo, são 11 acima e 10 abaixo na vigilância; 4 e 0 na confirmação. A contagem é exploratória *post hoc*; não prova que o serviço examinou ou rejeitou individualmente esses pares.

Fonte: `data/fp_arranjo_por_regiao.csv:8,9,16,17,24,25,32,33`. Slides 19 e 45.

### 27. O que protege um registro feito 49 minutos antes?

É registro interno versionado antes da S3, não registro público. O intervalo foi de 49 minutos, de 01:22:17 a 02:11:52 UTC, conforme os metadados na versão `16d5876`. O comparador *a posteriori* já era conhecido, e o código registrado está ausente do git. O registro cobre recalibração, braços e avaliação, não as análises exploratórias posteriores nem o enquadramento escolhido depois.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`; `data/arranjo_corte_metadata.json`; `gzcmd/M_arranjo_corte_desvios.md`. Slide 45.

### 28. De onde vêm os 20 minutos e os 5% de erro?

São hipóteses de cenário da configuração, não medidas do serviço. Vinte minutos por par convertem revisões em ordem de grandeza de horas; 5,0% é erro humano assumido para projetar recuperação. Sob revisor perfeito, O3 tem 0 FP; isso não estima falsos positivos humanos reais. O estudo requer medir tempo, erro e concordância antes da adoção.

Fonte: `gzcmd/gzcmd_v3_config.yaml`; `data/esforco_arranjo.csv`; `data/comparador_mesma_carga.csv`. Slides 6, 20 e 28.

### 29. Por que Grey-Zone não é a área cinza da nota?

O nome remete à região em que a decisão automática não é suficiente, não exclusivamente a [5,9) da nota do CdR. O corte do OpenRecLink usa outro escore. Na recomendação exploratória, preserva-se esse corte e revisa-se abaixo dele por p_cal_ml decrescente, até o orçamento humano. A adoção exige validação registrada em dados novos.

Fonte: `capitulo3.tex`, `sec:objetivos-especificos`; `data/comparador_mesma_carga.csv`. Slides 4, 17 e 29.

## Figuras e critérios de entrega

Figuras reaproveitadas, sem alterar os originais:

| Slide | Arquivo | Uso e ressalva |
| :-- | :-- | :-- |
| 20 | `text/slides/figures/curva_esforco.pgf` | Curva exploratória por script, com pontos de P |
| 9 | `figures/fig_cdr_gui_mapeamento.png` | Interface do Comparador de Registros, versão 0.1 (commit 3b43598), dados sintéticos |
| 31 | `text/latex/figures/fig_cap8_cascata_v31.png` | Cascata por REFREC sobre a base inteira (comparação) |
| 36 | `figures/imbalance_sensitivity.pgf` | Estratégias de balanceamento no Random Forest |
| 37 | `text/latex/fig_doc/epi_deteccao.pgf` | Detecção no teste da ablação, não arranjo nem cascata |

A captura do slide 9 está em `../../figures/fig_cdr_gui_mapeamento.png`, caminho relativo ao diretório de compilação `text/slides`, sem `graphicspath` adicional. A tabela do slide 25 reduz a tabela R2 e incorpora Paiva (2025) do texto adjacente, indicando que utilizou base distinta da desta tese, sem lhe atribuir a autoria do protótipo Java.

Critérios editoriais: menção ao referencial clássico restrita ao slide 8; probabilidade recalibrada abaixo do corte; arranjo P principal e cascata REFREC sobre a base inteira como comparação; corte operacional 211/247 por construção do *label*; curva de esforço exploratória *post hoc*, não registrada; F1 0,943 atribuído ao fluxo Modal; perdas de 13 e 27 vínculos junto do acréscimo automático; limitações antes das conclusões; equivalência Java parcial executada e comparação com Lages não realizada; sensibilidade 2×2 e QW1 local distinguidos dos resultados históricos; exportador com HMAC implementado, matriz não depositada; área cinza da nota apenas para [5,9). As tabelas foram reduzidas no arquivo de slides.

O ensaio foi descartado pelo autor em 04/11/2026. A revisão pela orientadora foi dispensada, conforme a auditoria privada. Os testes automáticos verificam estrutura e compilação, não constituem evidência de teste presencial de projeção.

Compilação de teste: duas passagens de pdflatex, com jobname `defesa_s8b_teste` e saída em `.local/slides_s8b/`. O `TEXINPUTS` aponta para `.texdeps` do worktree local `pq-f-slides`, que contém Beamer e translator. Esse build não substitui `text/slides/defesa.pdf` nem os PDFs oficiais da tese.

### Verificação técnica em 05/11/2026

O build de teste em duas passagens produziu 46 páginas, com 30 quadros principais e 16 de reserva, sem sobreposições. O log não contém erros LaTeX, referências indefinidas, Overfull ou marcadores de referência não resolvida. Há aviso de nome do pacote externo xcolor, que se identifica como xcolor-2022-06-12. As 29 respostas têm no máximo 70 palavras, conforme contagem por espaços. A inspeção visual abrange os quadros alterados; não constitui ensaio presencial de projeção.

A verificação desta fatia abrange os dois arquivos da apresentação, o contrato editorial dos testes e o build de teste. Resultados históricos do código científico não constituem verificação desta revisão editorial.

Log: `.local/slides_s8b/defesa_s8b_teste.log`. Em outra máquina, é necessário instalar Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los em diretório local de dependências.

SHA-256 da árvore compilada: `942fd9318efdc3f8f291416bc4b0c12322145a9b36d9f7bd0559d9ebbf8d707d`. O manifesto `.local/slides_s8b/arvore_compilada.json` registra os 6 arquivos de entrada do repositório efetivamente lidos pelo pdflatex, identificados pelo recorder e por seus hashes; não inclui dependências externas de TeX nem identifica o commit final. SHA-256 do PDF de teste: `f74324386960a3dc90a7233123023bf1b9f43f30967339f9ca8c936e309c61e8`. Os arquivos da tese e seus PDFs não são alterados nesta fatia.

Os identificadores da tese servem como remissões ao texto em revisão paralela. Os números novos ou alterados do arranjo e da equidade têm como fonte os CSVs agregados abaixo.

### Rastreabilidade dos números do arranjo

Linhas incluem o cabeçalho. A coluna `publicada` dos CSVs designa a cascata sobre a base inteira (comparação). Percentuais e custos foram arredondados apenas para apresentação.

| Número ou conjunto de números | CSV:linha em `data/` |
| :-- | :-- |
| 61.696, 247, corte 211/247, 85,4% por construção, 36 abaixo | `arranjo_corte_metricas.csv:2` |
| Até 61.485 potenciais pares na revisão integral, diretamente do campo `pares_humano` | `arranjo_corte_metricas.csv:8` |
| Vigilância P: 211 pelo corte e 4 pela política, total 215; 87,0%, IC 82,3 a 90,7%, 21 FP, 13 perdidos, 19 vínculos em 96 encaminhamentos, 344 LLM, US$ 0,94, 94,4% com revisor humano | `arranjo_corte_metricas.csv:2,12`; `arranjo_corte_par2.csv:7` |
| Confirmação P: 211 pelo corte e 8 pela política, total 219; 88,7%, IC 84,1 a 92,0%, 4 FP, 27 perdidos, 1 vínculo em 17 encaminhamentos, 41 LLM, US$ 0,11, 89,0% com revisor humano | `arranjo_corte_metricas.csv:20,30`; `arranjo_corte_par2.csv:16` |
| Fechamentos (211+4)+13+19=247 e (211+8)+27+1=247 | `arranjo_corte_metricas.csv:2,12,20,30` |
| PAR=2, aceitos/humano/perdidos: P 4/19/13 contra cascata 10/17/9; P 8/1/27 contra cascata 8/11/17 | `arranjo_corte_par2.csv:3,7,12,16` |
| Revisão integral de até 61.485: 99,3%, hipótese de erro humano de 5,0% | `arranjo_corte_metricas.csv:8,26`, campo `sens_humano_05` |
| Nenhum MATCH de P pelo motor abaixo do corte | `arranjo_corte_s1_resumo.csv:2-15,78-88` |
| Aceitações do LLM: 25 (4+21) e 12 (8+4) | `arranjo_corte_s1_resumo.csv:2-15,78-88` e `arranjo_corte_metricas.csv:2,12,30` |
| Área cinza da nota [5,9): 21.620, 151, corte 123; P 127 nos dois modos, 4 além do corte; base inteira 113/65 | `arranjo_corte_metricas.csv:3,5,13,23,31` |
| Base inteira v3.1: 194/34/15/38/111/347, 78,5%, IC 73,0 a 83,2%, US$ 0,92 | `arranjo_corte_metricas.csv:4` |
| Base inteira v3.1: 132/0/28/87/127/44, 53,4%, IC 47,2 a 59,6%, US$ 0,12 | `arranjo_corte_metricas.csv:22` |
| P-Platt: 217/3/77/110, US$ 0,30 e 13 perdidos; 219/3/16/25 | `arranjo_corte_metricas.csv:14,32` |
| A3: 219/19/100/367 e 219/5/22/32 | `arranjo_corte_metricas.csv:16,34` |
| P-PAR2-grey: 211/21/116/265 e 217/0/22/18 | `arranjo_corte_metricas.csv:18,36` |
| B1: 194/40/115/374 e 131/4/131/70 | `arranjo_corte_metricas.csv:6,24` |
| A posteriori, sem registro prévio: 221/34/90/330 e 219/0/51/25; 9 perdidos na vigilância contra 13 de P; 17 na confirmação | `arranjo_corte_aposteriori.csv:4,12`; `arranjo_corte_metricas.csv:10,12` |
| VPP feminino/masculino em P: 86,3/93,3%; base inteira (comparação): 87,9/84,0% | `equidade_arranjo_group_metrics.csv:2,3,26,27` |
| Sensibilidade no passo 11 em P: 42,5/47,5%; passos 6/9: 95,0/96,1% e 96,3/96,9% | `equidade_arranjo_group_metrics.csv:17-19,65-67` |
| BH, 7 rejeições em P vigilância, todas operacionais | `equidade_arranjo_pairwise_tests.csv:34,35,37,40,41,43,44` |
| BH, 3 rejeições em P confirmação, todas operacionais | `equidade_arranjo_pairwise_tests.csv:151,156,157` |
| BH na base inteira (comparação): 9/1 rejeições | `equidade_arranjo_pairwise_tests.csv:87,92,93,95,98-102,215` |
| 211 aceitos por construção; diferenças de sensibilidade dependem da proporção de PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36; diferenças de VPP dependem dos 21 e 4 FP abaixo do corte | `arranjo_corte_metricas.csv:2,12,30`; `equidade_arranjo_group_metrics.csv`; `equidade_arranjo_global.csv:2,4` |
| Na vigilância, não aceitos por passo 11/6/9: 23/4/5, todos PAR=2; no passo 11, 42,5% em P contra 45,0% na base inteira (comparação) | `equidade_arranjo_group_metrics.csv:17-19,41` |
| Passo 11: 25 dos 36 vínculos com PAR=2; P aceita 2/25 na vigilância e 4/25 na confirmação | `par_por_passo.csv:8-10,17-19,26-28,53-55` |
| Corte isolado nos passos 11/6/9: 15/40 (37,5%), 76/80 (95,0%) e 120/127 (94,5%) | `par_por_passo.csv:5-10,14-19,23-28` |
| PAR=2 aceitos nos passos 6/9/11: 0/2/2 na vigilância e 1/3/4 na confirmação | `par_por_passo.csv:8,17,26,35,44,53` |
| Pendentes de P∪B1 com chamada nova: 10 na vigilância e 14 na confirmação, não de P isolado; reutilização histórica e ausência de avaliação prospectiva | `arranjo_corte_s1_metadata.json:477-481`, campo `pending_P_union_B1_by_mode`; `arranjo_corte_metadata.json:355,357` |
| Cenários de carga sob ordem hipotética: 2.156 por passo e 2.522 global; SCORE global ≥ −5,648, sob hipótese não documentada de revisão decrescente de SCORE | `carga_abaixo_corte.csv:5,9` |
| Hipótese de 20 min por par: por passo, pelo menos 718 h 40 min; global, pelo menos 840 h 40 min; não são tempos observados | `carga_abaixo_corte_horas.csv:3,2` |
| P, vigilância: VPP do LLM 4/25 = 16,0%, IC 6,4 a 34,7%; 9,06 FP previstos contra 21 observados | `arranjo_corte_vpp_llm.csv:2` |
| P, confirmação: VPP 8/12 = 66,7%, IC 39,1 a 86,2%; P-Platt vigilância: 4/7 = 57,1%, IC 25,0 a 84,2% | `arranjo_corte_vpp_llm.csv:8,3` |
| Perdas de P por guarda/custo/LLM: 1/6/6 na vigilância e 1/22/4 na confirmação | `arranjo_corte_perdas_mecanismo.csv:2-4,50-52` |
| Perda de 13/36 = 36,1% e 27/36 = 75,0% dos PAR=2 | `arranjo_corte_par2.csv:7,16` |
| Dominância de P-Platt: vigilância N/P 7/20 e 4/20; confirmação 7/20 e 5/20; inversa 0/20 em todos os casos; FP menores em 20/20 nos dois blocos da vigilância | `robustez_arranjo_resumo.csv:2-5` |
| Vigilância: P-Platt perde em mediana 1/2 vínculos a mais, blocos numérico (N)/partição (R) | `robustez_arranjo_resumo.csv:404,488`, diferenças P menos P-Platt −1/−2 |
| P perde mais PAR=2 que o *a posteriori* em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação | `robustez_arranjo_resumo.csv:6,8,10,12`, proporções multiplicadas por 20 |
| P na vigilância, bloco de partição (R): FP de 9 a 31, perdas de 9 a 21 e Jaccard mediano 0,43 em relação à partição original | `robustez_arranjo_resumo.csv:270,278,654` |
| VPP por sexo de P na vigilância sem rejeição: p ajustado BH de 0,378 | `equidade_arranjo_pairwise_tests.csv:3` |
| Mudança máxima da probabilidade de P: 0,65 no bloco numérico (N) e 0,80 no P | `robustez_arranjo_resumo.csv:632,652` |
| Robustez: 242 potenciais pares novos, US$ 0,6255, arredondados para US$ 0,63 | `robustez_arranjo_metadata.json:206,208` |
| Consumo novo anterior: US$ 0,2448; somado a US$ 0,6255 da robustez, sem representar fatura | `arranjo_corte_metadata.json:285,293,330,338`, soma dos custos novos de S2 e suplemento; `robustez_arranjo_metadata.json:208` |

Os resultados históricos de v3, dos estudos 1 a 3 e da restrição na base inteira foram mantidos com as remissões que já constavam dos slides. Não são resultados novos do arranjo, nem foram reestimados nesta fatia.

### Comparação exploratória por orçamento

Análises exploratórias *post hoc*, não registradas. A curva não valida uma política. O3 e O4 usam o mesmo XGBoost OOF; revisor perfeito é hipótese e o cenário de erro é de 5,0%. O viés de verificação favorece O1 por construção.

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

A referência de 2.156 revisões pressupõe ordem decrescente de SCORE, que não foi documentada. Horas são ordem de grandeza a 20 min por par, não medida. As horas poupadas por vínculo não recuperado ou por FP descrevem a troca contra esse cenário, não economia histórica nem benefício de acurácia. Para O3, FP é zero sob revisor perfeito; não se divide por zero. K2 equipara o número total de avaliações de P (humano mais LLM), não o seu orçamento humano.

| Modo | Braço | Carga | Revisões | Horas | Horas poupadas | Horas poupadas por vínculo não recuperado | Horas poupadas por FP | CSV:linha |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- | :-- |
| vigilancia | P | K1_humana | 96 | 32,0 | 686,7 | 52,8 | 32,7 | `esforco_arranjo.csv:2` |
| vigilancia | O3 | K1_humana | 96 | 32,0 | 686,7 | 49,0 | não se aplica | `esforco_arranjo.csv:3` |
| vigilancia | P | K2_total | 96 | 32,0 | 686,7 | 52,8 | 32,7 | `esforco_arranjo.csv:4` |
| vigilancia | O3 | K2_total | 440 | 146,7 | 572,0 | 57,2 | não se aplica | `esforco_arranjo.csv:5` |
| confirmacao | P | K1_humana | 17 | 5,7 | 713,0 | 26,4 | 178,2 | `esforco_arranjo.csv:6` |
| confirmacao | O3 | K1_humana | 17 | 5,7 | 713,0 | 23,8 | não se aplica | `esforco_arranjo.csv:7` |
| confirmacao | P | K2_total | 17 | 5,7 | 713,0 | 26,4 | 178,2 | `esforco_arranjo.csv:8` |
| confirmacao | O3 | K2_total | 58 | 19,3 | 699,3 | 43,7 | não se aplica | `esforco_arranjo.csv:9` |

### FP por região do SCORE, análise exploratória

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
