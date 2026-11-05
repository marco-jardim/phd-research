# Roteiro da defesa

Base da apresentação e commit final: [ATUALIZAR APÓS BUILD]. A numeração corresponde aos quadros do PDF, sem sobreposições. São 30 quadros principais; os backups ficam reservados às perguntas, fora dos 40 minutos. O tempo é estimativa, não duração medida. O ensaio foi descartado por decisão do autor em 04/11/2026.

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

### 3. Revisão abaixo do corte
A região abaixo do corte do OpenRecLink contém 61.485 potenciais pares, dos quais vieram 36 dos 247 vínculos pela revisão manual. Esse volume é teto de uma revisão integral, não carga histórica observada: o volume examinado e o limite inferior da revisão não estão documentados. A área cinza é outra delimitação, a nota final do CdR em [5,9), com 21.620 potenciais pares e 151 vínculos.

### 4. Tese central
A revisão abaixo do corte do OpenRecLink passa de um problema de volume a um problema de decisão. O corte aceita os potenciais pares acima dele; a política v3.1, recalibrada abaixo do corte, decide os restantes. O percurso articula comparador, estimador, estratégias e equidade, documentando os vínculos que deixam de ser recuperados.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36, numa região de 61.485 potenciais pares com volume examinado não documentado. O corte 9 da nota do CdR é outro objeto, um cenário hipotético. Os 211 aceitos e os zero falsos positivos acima do corte decorrem da construção do rótulo PAR=1; não são avaliação independente da acurácia do corte.

### 7. Faixas e padrão-ouro
As faixas são da nota final do CdR, não do escore probabilístico do OpenRecLink. Há três vínculos abaixo de 5, 151 na faixa intermediária e 93 a partir de 9; uma revisora realizou a classificação, sem segunda avaliação independente para medir concordância humana.

A origem do extrato é o processo de *linkage* descrito por Oliveira (2026, p. 33 a 39), limitado a três passos probabilísticos. Na área cinza do CdR, [5,9), o arranjo P aceita 127 vínculos nos dois modos, sendo 123 do corte e quatro com PAR=2. O acréscimo frente ao corte isolado deve ser lido junto dos 13 e 27 perdidos no conjunto dos 36 vínculos da revisão. A cascata sobre a base inteira (comparação) aceita 113 e 65 nessa faixa.

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
O GZ-CMD usa outro XGBoost, com 30 atributos e calibração isotônica aninhada, agrupando por REFREC nos níveis externo e interno. Na área cinza, a AUC é 0,995 e a AP é 0,862; o corte ilustrativo de 0,5 produz 119 verdadeiros positivos e 28 falsos positivos. Esses números descrevem o estimador original, não o arranjo. Nele, o classificador continua ajustado na base inteira, mas a isotônica é reajustada apenas com as probabilidades OOF internas do treino abaixo do corte, sem os rótulos da partição externa.

### 14. Finalidades operacionais
A vigilância atribui maior consequência à perda do vínculo, enquanto a confirmação atribui maior consequência ao falso vínculo. A fronteira permite explicitar esse compromisso, sem transformar os pontos explorados no teste em limiares validados para uso externo.

### 15. Impacto epidemiológico
No teste da ablação, o RF+SMOTE detectou 67 vínculos e o limiar 8 detectou 43, uma diferença líquida de 24, ou 55,8%. São 27 recuperados exclusivamente pelo modelo, menos três exclusivos do limiar; esse resultado tem como denominador 74 vínculos do teste e não mede o impacto do arranjo nem da cascata sobre a base inteira.

### 16. Limites do impacto
Os indicadores derivados do encerramento por óbito constituem vazamento de alvo. A sensibilidade 2×2 foi executada: o F1 do RF+SMOTE passou de 0,931 para 0,938 sem óbito, mas caiu para 0,846 com REFREC. No QW1 local na GPU, o XGBoost passou de 0,937 para 0,878 com REFREC, sem substituir o resultado histórico de 0,920. O otimismo aparece sobretudo na partição por linha, embora mudem os vínculos e denominadores. O GZ-CMD já utiliza agrupamento por REFREC, sem indicadores de óbito; permanece possível dependência por COMPREC.

### 17. Arquitetura depois do corte
O arranjo preserva a aceitação acima do corte e aplica o GZ-CMD abaixo dele. As guardas usam nota e probabilidade recalibrada; as faixas estratificam o erro do revisor. A isotônica é reajustada apenas no treino abaixo do corte, sem rótulos da partição externa, e a política compara quatro ações por custo esperado. As taxas foram congeladas antes das chamadas novas, sem realimentação por essas respostas. Isso não torna a avaliação prospectiva: as decisões do LLM em P reutilizam sobretudo lotes históricos, também usados para estimar as proporções do revisor. Os 10 e 14 potenciais pares com chamada nova eram pendentes de P∪B1, na vigilância e na confirmação, não de P isolado.

### 18. Custos e ações
O custo do falso negativo é maior na vigilância, enquanto o custo do falso positivo é maior na confirmação. As quatro ações são comparadas na mesma escala de utilidade, e essa escala não deve ser confundida com dólares nem com custos econômicos medidos no serviço.

### 19. Arranjo sequencial com registro prévio
O registro prévio é um YAML versionado no repositório, não um registro público. O braço P é o resultado principal. Na vigilância, são 211 vínculos pelo corte e quatro pela política, totalizando 215/247, ou 87,0%, com IC de 82,3 a 90,7%, 21 falsos positivos e 13 perdidos. Encaminha 96 potenciais pares ao revisor humano, incluindo 19 vínculos, e 344 ao LLM, a US$ 0,94. O fechamento é (211+4)+13+19=247. Na confirmação, são 211 pelo corte e oito pela política, totalizando 219, ou 88,7%, com IC de 84,1 a 92,0%, 4 falsos positivos e 27 perdidos. Encaminha 17 ao revisor humano, incluindo um vínculo, e 41 ao LLM, a US$ 0,11: (211+8)+27+1=247. O acréscimo frente ao corte isolado é de quatro e oito vínculos, com 13 e 27 perdidos; os intervalos se sobrepõem ao do corte.

A cascata v3.1 sobre a base inteira é comparação: 194 e 132 aceitos, ou 78,5% e 53,4%, com 34 e zero falsos positivos, 111 e 127 encaminhamentos ao revisor humano e lotes LLM de 347 e 44. Ela também submete à política os potenciais pares que o corte já aceita. O custo estimado na vigilância é de US$ 0,94 no arranjo e US$ 0,92 na base inteira. Os valores somam consumo estimado, sem pessoal nem infraestrutura.

### 20. O que a troca custa
Nos 36 vínculos com PAR=2, o arranjo aceita menos que a cascata sobre a base inteira na vigilância e empata na confirmação. Perde mais nos dois modos:

| Modo | Aceitos: arranjo × cascata | Ao revisor humano: arranjo × cascata | Perdidos: arranjo × cascata |
| :-- | :-- | :-- | :-- |
| Vigilância | 4 × 10 | 19 × 17 | 13 × 9 |
| Confirmação | 8 × 8 | 1 × 11 | 27 × 17 |

A vantagem global do arranjo sobre a cascata vem inteiramente dos 211 vínculos preservados pelo corte por construção, não de maior recuperação abaixo dele. Os números da comparação são de `data/arranjo_corte_par2.csv:3,7,12,16`.

Todas as aceitações abaixo do corte vieram do LLM: 25 na vigilância, sendo quatro vínculos e 21 falsos positivos, e 12 na confirmação, sendo oito e quatro. Não houve aceitação pelo motor nessa região. A sensibilidade com revisor humano, sob erro hipotético de 5,0%, seria de 94,4% e 89,0%, contra 99,3% com revisão integral de até 61.485 potenciais pares. São projeções, não decisões humanas observadas. A vigilância só apresenta maior sensibilidade quando se incorpora essa revisão projetada; automaticamente, aceita menos que a confirmação.

### 21. Revisor de referência
Claude Haiku 5.5 com instruções v4 revisa a evidência sem receber as saídas decisórias do motor. A ablação é compatível com ancoragem nessas saídas, mas não demonstra causalidade geral; Claude Sonnet 5.5 recuperou dois a quatro vínculos adicionais na cascata sobre a base inteira (comparação), com mais encaminhamentos e tarifa vinte vezes maior por token.

### 22. Limites da automação
Os 96 e 17 encaminhamentos em P são trabalho pendente. Sob erro humano hipotético de 5,0%, as sensibilidades seriam 94,4% e 89,0%. P-Platt, análise de sensibilidade com *Platt scaling*, produz na vigilância três falsos positivos, 77 encaminhamentos e lote de 110 ao LLM, a US$ 0,30, com os mesmos 13 perdidos de P. O resultado depende da calibração, e P-Platt não substitui o braço principal. A restrição de 0,2% de falsos vínculos esperados foi inviável na base inteira (comparação).

### 23. Equidade por sexo
No arranjo P de vigilância, o VPP é de 86,3% no sexo feminino e 93,3% no masculino. Na cascata sobre a base inteira (comparação), é de 87,9% e 84,0%. A aceitação do arranjo combina corte e respostas de vínculo do LLM; o encaminhamento humano conta como não aceito. Como os 211 são aceitos por construção, diferenças de sensibilidade dependem dos 36 com PAR=2 e as de VPP dos 21 ou quatro falsos positivos. O poder é limitado, e ausência de rejeição não comprova equidade. Raça/cor não está disponível.

### 24. Desigualdade operacional
<!-- % CONFERIR par_por_passo.csv -->
A diferença por passo reflete sobretudo o corte e a construção do rótulo, não um efeito isolado do GZ-CMD. Na vigilância, os vínculos não aceitos automaticamente por P são 23, quatro e cinco nos passos 11, 6 e 9. Como P aceita os PAR=1 por construção, pelo menos 23 dos 36 vínculos com PAR=2 estão no passo 11. Nele, a sensibilidade automática do arranjo é de 42,5% na vigilância e 47,5% na confirmação; nos passos 6 e 9, é de pelo menos 95,0%. Na vigilância, o arranjo fica abaixo da cascata sobre a base inteira (comparação) no passo 11: 42,5% contra 45,0%. Após Benjamini-Hochberg, são sete e três rejeições em P, todas operacionais; na base inteira (comparação), nove e uma. A equidade do processo completo depende da revisão humana ainda não observada.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. A tese propõe sua composição operacional com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
O padrão-ouro incorpora decisões de uma revisora e do corte, com zero falsos positivos acima dele por construção. O volume e o limite inferior da revisão não estão documentados, podendo existir vínculos não examinados. A avaliação é condicional à blocagem, e REFREC não separa simultaneamente COMPREC. Os 36 vínculos com PAR=2 produzem intervalos largos. As respostas do LLM em P reutilizam majoritariamente lotes históricos, também usados para estimar as proporções do revisor; permanece seleção adaptativa, e a validação cruzada não equivale a avaliação prospectiva nem externa. Os 10 e 14 potenciais pares com chamada nova eram pendentes de P∪B1, na vigilância e na confirmação. Não se pode atribuir essas contagens a P isolado nem dividi-las pelos lotes de P de 344 e 41. O enquadramento do arranjo como principal foi escolhido depois da análise a posteriori; o registro prévio cobre a recalibração, os braços e a avaliação, não essa escolha.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. O processamento LLM utilizou APIs externas. A exportação restrita com identificador de par por HMAC foi implementada em `scripts/export_matriz_desidentificada.py`, mas a matriz não foi depositada e não se afirma saneamento retroativo das execuções. A equivalência Java parcial foi executada; a comparação empírica com Lages não foi realizada.

### 28. Conclusões
Na vigilância, são 211 vínculos pelo corte e quatro pela política: 87,0% (85,4% pelo corte), com 21 falsos positivos, 96 encaminhamentos e 13 dos 36 vínculos da revisão perdidos. Na confirmação, são 211 pelo corte e oito pela política: 88,7% (85,4% pelo corte), com quatro falsos positivos, 17 encaminhamentos e 27 dos 36 perdidos. A vantagem global sobre a cascata sobre a base inteira vem inteiramente dos 211 preservados por construção. Nos 36 PAR=2, P aceita quatro e oito, contra dez e oito da cascata, e perde 13 e 27, contra nove e 17. A carga de 96 ou 17 é comparada ao teto de até 61.485 potenciais pares, não a uma carga histórica medida. A contribuição é governar a decisão e medir o que ela custa, não prometer automação integral.

### 29. Contribuições
As entregas incluem comparador com equivalência parcial avaliada, estimadores e faixas, pacote decisório 0.4.0 com política v3.1 e artefatos pseudonimizados no Zenodo 3.0.1. O pacote e o depósito cobrem a v3.1 sobre a base inteira, não o arranjo; seus agregados estão no repositório e as saídas por par ficam locais. A vantagem global do arranjo vem da preservação dos 211 pelo corte, não de maior recuperação dos 36 abaixo dele. A avaliação seguinte requer referência independente, observação da revisão humana e transferência externa.

### 30. Agradecimentos
O trabalho dependeu da orientação, da equipe responsável pelos dados e da revisão manual que permitiu a avaliação. Depois do corte, o GZ-CMD organiza a revisão como decisão, com 96 ou 17 encaminhamentos ao revisor humano e 13 ou 27 vínculos perdidos. Esse custo em vínculos deve acompanhar a discussão com o profissional do serviço.

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
P continua principal. P-Platt troca a isotônica por *Platt scaling*; A3 também restringe o ajuste do classificador aos potenciais pares abaixo do corte; P-PAR2-grey estima o erro de não-vínculo apenas com PAR=2 nas bandas agrupadas de [5,9). São análises de sensibilidade. B1 mantém a base inteira (comparação) e a probabilidade original, mudando as taxas por validação cruzada: na vigilância, mantém 194 aceitos, com 40 falsos positivos contra 34 da cascata. P-Platt apresenta na vigilância três falsos positivos, 77 encaminhamentos e US$ 0,30, contra 21, 96 e US$ 0,94 de P, com os mesmos 13 perdidos. Na vigilância, o comparador a posteriori, sem registro prévio, aceita 221 e perde nove vínculos, contra 13 perdidos de P. Na confirmação, tem zero falso positivo e 17 perdidos contra quatro e 27 de P, mas encaminha 51 potenciais pares contra 17. Ambos precisam de nova avaliação, sem substituir P depois de conhecidos os resultados.

## Perguntas prováveis

Número de perguntas e contagens de palavras: [ATUALIZAR APÓS BUILD]. A primeira resposta reúne a pertinência do probabilístico e o escopo do módulo Python, incluindo a equivalência Java. Mantém-se o limite de 70 palavras por resposta, sem contar pergunta e fonte, para duração estimada de até 30 segundos. As fontes usam identificadores da tese e CSVs, sem depender da paginação.

### 1. O probabilístico é inútil? Onde está o módulo Python do OpenRecLink?

O corte do OpenRecLink aceitou 211/247 vínculos, ou 85,4%. A entrega Python é o CdR, não a blocagem. A equivalência com o Java de Peçanha (2015) foi avaliada: 100,0% nas datas por igualdade/um dígito, 99,2% nas abreviações e nota idêntica em 11,2% dos pares sintéticos. Inversões pesam 0,8 em Lucena/Java e 1,0 no Python, limitação declarada. Não há equivalência integral.

Fonte: `capitulo_dados.tex`, seção Padrão-ouro (`sec:padrao-ouro`); `estudo1_comparador.tex`, Resultados (`sec:e1-resultados`). Slides 6, 8 e 9.

### 2. Aprendizado de máquina é só exploração de possibilidades?

Há exploração, e seus limites estão declarados: F1 de 0,931 na ablação e 0,943 no fluxo Modal usaram limiares escolhidos no teste. O GZ-CMD usa outro estimador, com 30 atributos, calibração aninhada e agrupamento por REFREC, com AUC de 0,995 e AP de 0,862 na área cinza. Isso sustenta discriminação interna, não validação externa da política.

Fonte: `estudo3_estrategias.tex`, Limitações (`sec:e3-limitacoes`); `estudo4_gzcmd.tex`, Classificador supervisionado e calibração fora da amostra (`sec:e4-classificador-oof`). Slides 11 e 13.

### 3. Isso já foi feito por Lages ou pelo CIDACS?

Há precedentes de classificação supervisionada e redução da revisão. A contribuição proposta é a composição operacional com guardas, calibração agrupada, custos, revisão seletiva e equidade, numa base de razão 1:249. Lages trabalha com razão 1:5 e outro delineamento. Não foi realizada comparação empírica entre as implementações; a tabela de posicionamento não é uma prova de superioridade.

Fonte: `capitulo2.tex`, Classificação supervisionada, experiência brasileira e `tab:posicionamento`; `estudo1_comparador.tex`, Discussão (`sec:e1-discussao`). Slide 25.

### 4. Por que F1 próximo de 0,93 aqui e 0,996 em Lages?

O valor 0,943 pertence ao XGBoost do fluxo Modal, com seleção de limiar no teste; o QW1 tem 0,920. Lages reporta 0,996 em outro conjunto e protocolo, com razão de classes distinta. Essa comparação não identifica a causa da diferença nem ordena estimadores. Aqui, o resultado por faixa OOF é apresentado separadamente, evitando atribuir o desempenho global apenas aos casos difíceis.

Fonte: `capitulo2.tex`, `tab:posicionamento`; `tab:qw1-deep-learning`; `tab:ablation-best-category`; `tab:e4-desempenho-faixas-oof`. Slides 11, 25 e 33.

### 5. Como foi construído o padrão-ouro?

Luís Guilherme Buteri Alves realizou o pré-processamento, e Daniela Medina Coeli Bastos configurou a vinculação e revisou os potenciais pares. Dos 247 vínculos, 211 vieram do corte e 36 da revisão abaixo dele. O CdR foi executado antes da revisão, com risco de viés de incorporação. Não houve segunda classificação independente nem kappa. A referência incorpora o corte e limita a sensibilidade aos potenciais pares exportados.

Fonte: `capitulo_dados.tex`, Padrão-ouro (`sec:padrao-ouro`). Slides 6 e 7.

### 6. O arranjo é determinístico?

O corte e as guardas são determinísticos, mas a cadeia inclui probabilidade estimada, custos e Claude Haiku 5.5, cuja geração é estocástica. Na vigilância, são 211 vínculos pelo corte e quatro pela política, com 13 perdidos e 19 vínculos entre 96 encaminhamentos ao revisor humano. Abaixo do corte, as aceitações vieram do LLM, com 21 falsos positivos. Não há garantia de repetibilidade integral.

Fonte: `data/arranjo_corte_metricas.csv:12`; `data/arranjo_corte_s1_resumo.csv:2-15`; `sec:e4-politica-v31`. Slides 17, 19 a 21.

### 7. Como o arranjo se compara à revisão integral?

A revisão integral de até 61.485 potenciais pares projeta 99,3% de sensibilidade, sob erro humano de 5,0%. P encaminha 96 na vigilância e 17 na confirmação, com projeções de 94,4% e 89,0%, mas perde 13 e 27 dos 36 vínculos da revisão. Não são decisões humanas observadas; o volume histórico examinado e o limite inferior da revisão não estão documentados.

Fonte: `data/arranjo_corte_metricas.csv:8,12,26,30`; `data/arranjo_corte_par2.csv:7,16`. Slides 6, 19, 20 e 22.

### 8. Dados identificados foram enviados à nuvem?

O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). O autor recebeu apenas extrato pseudonimizado, sem registros nominais nem correspondência com pessoas. A base legal é a LGPD (arts. 7º, IV, 11, II, c, e 13). Dossiês enviados a APIs externas continham nota, atributos derivados e pseudônimo; nas execuções auditadas, retiraram-se as saídas do motor. Persiste risco de reidentificação.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`), ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`). Slides 27 e 42.

### 9. Como discutir equidade sem raça/cor?

A falta de raça/cor impede examinar disparidades raciais. Avaliaram-se sensibilidade e VPP, não toda a igualdade de oportunidades de erro. No passo 11, P tem 42,5% e 47,5% nos dois modos; na vigilância, fica abaixo dos 45,0% da base inteira (comparação). A diferença reflete sobretudo o corte e o rótulo. Ausência de rejeição demográfica não demonstra equidade.

Fonte: `data/equidade_arranjo_group_metrics.csv`, passo; `data/equidade_arranjo_pairwise_tests.csv`, rejeições de P. Slides 23 e 24.

### 10. Por que o arranjo, e não a cascata publicada?

O arranjo preserva 211 vínculos pelo corte e acrescenta quatro ou oito pela política, com 13 ou 27 perdidos. Nos 36 PAR=2, aceita 4/8 contra 10/8 da cascata sobre a base inteira. A vantagem global vem inteiramente da preservação pelo corte, não de maior recuperação abaixo dele. O enquadramento foi escolhido a posteriori; o registro cobre recalibração e avaliação, não essa escolha.

Fonte: `data/arranjo_corte_metricas.csv:2,4,12,22,30`; `data/arranjo_corte_par2.csv:3,7,12,16`; `gzcmd/prereg_arranjo_corte.yaml`. Slides 17, 19, 20, 26 e 28.

### 11. Por que não adotar P-Platt?

P-Platt é análise de sensibilidade, não o braço principal. Na vigilância, tem três falsos positivos, 77 encaminhamentos e US$ 0,30, contra 21, 96 e US$ 0,94 de P, mantendo 13 vínculos perdidos. Esse resultado evidencia dependência da calibração e justifica nova avaliação com registro prévio. Trocar o principal depois de conhecidos os resultados introduziria seleção pelo desempenho observado.

Fonte: `data/arranjo_corte_metricas.csv:12,14`; `data/arranjo_corte_par2.csv:7,8`. Slides 22 e 45.

## Figuras e critérios de entrega

Figuras reaproveitadas, sem alterar os originais:

| Slide | Arquivo | Uso e ressalva |
| :-- | :-- | :-- |
| 9 | `figures/fig_cdr_gui_mapeamento.png` | Interface do Comparador de Registros, versão 0.1 (commit 3b43598), dados sintéticos |
| 31 | `text/latex/figures/fig_cap8_cascata_v31.png` | Cascata por REFREC sobre a base inteira (comparação) |
| 36 | `figures/imbalance_sensitivity.pgf` | Estratégias de balanceamento no Random Forest |
| 37 | `text/latex/fig_doc/epi_deteccao.pgf` | Detecção no teste da ablação, não arranjo nem cascata |

A captura do slide 9 está em `../../figures/fig_cdr_gui_mapeamento.png`, caminho relativo ao diretório de compilação `text/slides`, sem `graphicspath` adicional. A tabela do slide 25 reduz a tabela R2 e incorpora Paiva (2025) do texto adjacente, indicando que utilizou base distinta da desta tese, sem lhe atribuir a autoria do protótipo Java.

Critérios editoriais: menção ao referencial clássico restrita ao slide 8; probabilidade recalibrada abaixo do corte; arranjo P principal e cascata REFREC sobre a base inteira como comparação; corte operacional 211/247 por construção do rótulo; F1 0,943 atribuído ao fluxo Modal; perdas de 13 e 27 vínculos junto do acréscimo automático; limitações antes das conclusões; equivalência Java parcial executada e comparação com Lages não realizada; sensibilidade 2×2 e QW1 local distinguidos dos resultados históricos; exportador com HMAC implementado, matriz não depositada; área cinza apenas para [5,9). As tabelas foram reduzidas no arquivo de slides.

O ensaio foi descartado pelo autor em 04/11/2026. A revisão pela orientadora foi dispensada, conforme a auditoria privada. Os testes automáticos verificam estrutura e compilação, não constituem evidência de teste presencial de projeção.

Compilação de teste: duas passagens de pdflatex, com jobname `defesa_s4b_teste` e saída em `.local/slides_s4b/`. O `TEXINPUTS` aponta para `.texdeps` do worktree local `pq-f-slides`, que contém Beamer e translator. Esse build não substitui `text/slides/defesa.pdf` nem os PDFs oficiais da tese.

### Verificação técnica em 04/11/2026

Contagem final de páginas, quadros de backup, perguntas e palavras, e verificação do log: [ATUALIZAR APÓS BUILD].

A verificação desta fatia se restringe aos dois arquivos da apresentação e ao build de teste. Resultados históricos de testes do código científico não constituem verificação desta revisão editorial.

Log previsto: `.local/slides_s4b/defesa_s4b_teste.log`. Em outra máquina, é necessário instalar Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los em diretório local de dependências.

Commit base e hashes do fechamento final: [ATUALIZAR APÓS BUILD]. Os arquivos da tese e seus PDFs não são alterados nesta fatia.

Os identificadores da tese servem como remissões ao texto em revisão paralela. Os números novos ou alterados do arranjo e da equidade têm como fonte os CSVs agregados abaixo.

### Rastreabilidade dos números do arranjo

Linhas incluem o cabeçalho. A coluna `publicada` dos CSVs designa a cascata sobre a base inteira (comparação). Percentuais e custos foram arredondados apenas para apresentação.

| Número ou conjunto de números | CSV:linha em `data/` |
| :-- | :-- |
| 61.696, 247, corte 211/247, 85,4%, IC 80,5 a 89,3%, 36 abaixo | `arranjo_corte_metricas.csv:2` |
| Até 61.485 potenciais pares na revisão integral, diretamente do campo `pares_humano` | `arranjo_corte_metricas.csv:8` |
| Vigilância P: 211 pelo corte e 4 pela política, total 215; 87,0%, IC 82,3 a 90,7%, 21 FP, 13 perdidos, 19 vínculos em 96 encaminhamentos, 344 LLM, US$ 0,94, 94,4% com revisor humano | `arranjo_corte_metricas.csv:2,12`; `arranjo_corte_par2.csv:7` |
| Confirmação P: 211 pelo corte e 8 pela política, total 219; 88,7%, IC 84,1 a 92,0%, 4 FP, 27 perdidos, 1 vínculo em 17 encaminhamentos, 41 LLM, US$ 0,11, 89,0% com revisor humano | `arranjo_corte_metricas.csv:20,30`; `arranjo_corte_par2.csv:16` |
| Fechamentos (211+4)+13+19=247 e (211+8)+27+1=247 | `arranjo_corte_metricas.csv:2,12,20,30` |
| PAR=2, aceitos/humano/perdidos: P 4/19/13 contra cascata 10/17/9; P 8/1/27 contra cascata 8/11/17 | `arranjo_corte_par2.csv:3,7,12,16` |
| Revisão integral de até 61.485: 99,3%, hipótese de erro humano de 5,0% | `arranjo_corte_metricas.csv:8,26`, campo `sens_humano_05` |
| Nenhum MATCH de P pelo motor abaixo do corte | `arranjo_corte_s1_resumo.csv:2-15,78-88` |
| Aceitações do LLM: 25 (4+21) e 12 (8+4) | `arranjo_corte_s1_resumo.csv:2-15,78-88` e `arranjo_corte_metricas.csv:2,12,30` |
| Área cinza [5,9): 21.620, 151, corte 123; P 127 nos dois modos, 4 além do corte; base inteira 113/65 | `arranjo_corte_metricas.csv:3,5,13,23,31` |
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
| 211 aceitos por construção; diferenças avaliadas nos 36 PAR=2 e 21/4 FP | `arranjo_corte_metricas.csv:2,12,30`; `equidade_arranjo_global.csv:2,4` |
| Na vigilância, não aceitos por passo 11/6/9: 23/4/5, todos PAR=2; no passo 11, 42,5% em P contra 45,0% na base inteira (comparação) | `equidade_arranjo_group_metrics.csv:17-19,41` |
| Pendentes de P∪B1 com chamada nova: 10 na vigilância e 14 na confirmação, não de P isolado; reutilização histórica e ausência de avaliação prospectiva | `arranjo_corte_s1_metadata.json:477-481`, campo `pending_P_union_B1_by_mode`; `arranjo_corte_metadata.json:355,357` |

Os resultados históricos de v3, dos estudos 1 a 3 e da restrição na base inteira foram mantidos com as remissões que já constavam dos slides. Não são resultados novos do arranjo, nem foram reestimados nesta fatia.
