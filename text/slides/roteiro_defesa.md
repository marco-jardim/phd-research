# Roteiro da defesa

Base versionada inicial: `4e8479e2f75f04a39a9b2dc02c8ecdf7d7dbf59c`. O hash registrado na verificação técnica identifica a árvore compilada, não o commit final. A numeração corresponde aos quadros do PDF, sem sobreposições: 30 principais e 16 de reserva, estes fora dos 40 minutos. O tempo é estimativa, não duração medida. O ensaio foi descartado por decisão do autor em 04/11/2026.

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
Sob a hipótese de revisão em ordem decrescente de escore, derivada do rótulo, o limite inferior da carga histórica é de 2.156 potenciais pares com limiar por passo. Com limiar global SCORE ≥ −5,648, são 2.522. Esses números não são uma observação do volume revisado, mas o mínimo compatível com os 36 vínculos recuperados. Os 61.485 são apenas teto teórico, não o contraponto principal. A área cinza da nota do CdR, [5,9), é outra delimitação.

### 4. Tese central
A tese central é uma hipótese condicionada à carga e à aceitação das consequências dos erros pelo serviço. O GZ-CMD pode organizar a revisão dos potenciais pares não decididos pelo corte e reduzir os encaminhamentos, mas a evidência de carga é um limite inferior, e o arranjo recupera menos vínculos que a prática do serviço contra o rótulo. O objetivo 4 é desenvolver e avaliar a cascata por *expected loss*, com custos assimétricos, calibração fora da amostra e revisão assistida por LLM, nos potenciais pares não aceitos pelo corte. Comparam-se vínculos recuperados, falsos positivos, encaminhamentos, custo e carga com o corte seguido de revisão manual e com estratégias de corte único e de duas fronteiras.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36. PAR é o rótulo do padrão-ouro: 1 designa aceitação pelo corte; 2, recuperação pela revisão manual. O corte 9 da nota do CdR é outro objeto, um cenário hipotético. Os 211 aceitos e zero falsos positivos acima do corte decorrem do rótulo, não de avaliação independente. O limite inferior de carga foi apresentado no quadro 3.

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
O arranjo preserva a aceitação acima do corte e aplica o GZ-CMD abaixo dele. As guardas usam nota e probabilidade recalibrada; as faixas estratificam o erro do revisor. A isotônica é reajustada apenas no treino abaixo do corte, sem rótulos da partição externa. As taxas foram congeladas antes das chamadas novas, sem realimentação por essas respostas, mas P reutiliza sobretudo lotes históricos também usados para estimar as proporções do revisor. Não é avaliação prospectiva. *Grey-Zone* no nome remete ao conceito de Fellegi e Sunter, a região que a regra automática não decide; nos resultados, área cinza da nota designa somente [5,9).

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

As perdas correspondem a 36,1% e 75,0% dos 36 PAR=2. Na vigilância, uma perda vem da guarda de nota inferior a 5, seis do não-vínculo automático por custo e seis do não-vínculo do LLM. Na confirmação, são uma, 22 e quatro. Não se pode atribuir todo o resultado apenas à recalibração ou aos custos. A sensibilidade com revisor humano, sob erro hipotético de 5,0%, seria de 94,4% e 89,0%, contra 99,3% na revisão integral teórica. São projeções, não decisões humanas observadas. A vigilância só apresenta maior sensibilidade com essa revisão projetada; automaticamente, aceita menos que a confirmação.

### 21. Revisor de referência
Claude Haiku 5.5 com instruções v4 recebe dossiê sem as saídas decisórias do motor, em protocolo de dois agentes e árbitro. Toda aceitação abaixo do corte em P veio do LLM. Na vigilância, somente quatro das 25 aceitações eram vínculos: VPP de 16,0%, com IC de 6,4 a 34,7%. Na confirmação, são oito de 12, ou 66,7%. P-Platt tem quatro de sete na vigilância, ou 57,1%. As taxas congeladas previam cerca de 9,06 falsos positivos em P na vigilância, contra 21 observados. A transferência de proporções de coortes com potenciais pares mais fáceis subestimou o erro nessa região; o dossiê sem saídas não elimina essa limitação.

### 22. P-Platt e a dependência da calibração
P foi dominado na avaliação original por P-Platt, registrado como análise de sensibilidade com *Platt scaling*. Na vigilância, P-Platt aceita 217, com três falsos positivos, 77 encaminhamentos humanos, 110 ao LLM e US$ 0,30. P aceita 211 pelo corte e quatro pela política, com 21 falsos positivos, 96 ao revisor humano, 344 ao LLM e US$ 0,94. Ambos perdem 13 vínculos. A robustez com registro prévio, não confirmatória, não sustentou a dominância: sete e quatro das 20 réplicas na vigilância, sete e cinco na confirmação, nos blocos N e P. A redução de FP na vigilância ocorreu em todas as réplicas dos dois blocos, mas P-Platt perdeu, em mediana, um e dois vínculos a mais. P permanece principal. A recomendação é preservar o corte; a isotônica abaixo dele não superou as alternativas, e adotar Platt exige validação em dados novos.

### 23. Equidade por sexo
No arranjo P de vigilância, o VPP é de 86,3% no sexo feminino e 93,3% no masculino. Na cascata sobre a base inteira (comparação), é de 87,9% e 84,0%. A aceitação do arranjo combina corte e respostas de vínculo do LLM; o encaminhamento humano conta como não aceito. Os 211 vínculos acima do corte são aceitos por construção. As diferenças de sensibilidade entre subgrupos dependem da proporção de vínculos PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36 vínculos. As diferenças de VPP dependem dos 21 e quatro falsos positivos abaixo do corte. O poder é limitado por esses números e pela incorporação do corte ao rótulo; ausência de rejeição não comprova equidade. Raça/cor não está disponível.

### 24. Desigualdade operacional
A disparidade por passo é herdada do corte e do rótulo, não um efeito isolado do GZ-CMD. O corte sozinho aceita 15 dos 40 vínculos do passo 11, ou 37,5%, contra 76 de 80 no passo 6 (95,0%) e 120 de 127 no passo 9 (94,5%). No passo 11 estão 25 dos 36 vínculos com PAR=2. O arranjo aceita dois desses 25 na vigilância e quatro na confirmação, chegando a 17/40 (42,5%) e 19/40 (47,5%). Nos passos 6/9/11, as aceitações PAR=2 são 0/2/2 na vigilância e 1/3/4 na confirmação, conforme `data/par_por_passo.csv`. Nos passos 6 e 9, a sensibilidade do arranjo é de pelo menos 95,0% nos dois modos. Na vigilância, fica abaixo da cascata sobre a base inteira (comparação) no passo 11: 42,5% contra 45,0%. Após Benjamini-Hochberg, são sete e três rejeições em P, todas operacionais; na base inteira (comparação), nove e uma. A contagem não ordena equidade. A equidade do processo completo depende da revisão humana ainda não observada.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. A tese propõe sua composição operacional com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
O padrão-ouro incorpora o corte e uma revisão com acesso ao CdR, combinando verificação e incorporação. A carga histórica é apenas limitada inferiormente, sob a hipótese do quadro 3. REFREC não separa simultaneamente COMPREC, e os 36 PAR=2 limitam o poder. Respostas históricas reutilizadas, taxas com denominadores pequenos e transferência para potenciais pares mais difíceis impedem interpretar a avaliação como prospectiva. Os 10 e 14 potenciais pares com chamada nova eram pendentes de P∪B1, não de P isolado. O comparador a posteriori, corte seguido da cascata publicada sem recalibração, já era conhecido antes do registro; o enquadramento foi escolhido depois. O registro cobre recalibração, braços e avaliação, não essa escolha. O código inicialmente registrado não está no git, e o suplemento fez chamadas de sensibilidade não previstas. Esses são desvios de protocolo, não alterações retroativas do registro. A robustez mede estabilidade nos mesmos dados, não generalização.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. O processamento LLM utilizou APIs externas. A exportação restrita com identificador de par por HMAC foi implementada em `scripts/export_matriz_desidentificada.py`, mas a matriz não foi depositada e não se afirma saneamento retroativo das execuções. A equivalência Java parcial foi executada; a comparação empírica com Lages não foi realizada.

### 28. Conclusões
Na vigilância, são 211 vínculos pelo corte e quatro pela política, 87,0%, com 21 falsos positivos, 96 encaminhamentos e 13 dos 36 vínculos da revisão perdidos. Na confirmação, são 211 e oito, 88,7%, com quatro falsos positivos, 17 encaminhamentos e 27 perdidos. A hipótese é condicionada à carga: esses encaminhamentos contrastam com um limite inferior de 2.156, não com uma carga histórica medida. Frente à prática do serviço, há pior recuperação e mais falsos positivos contra o rótulo; a única vantagem identificada é a carga inferida. Frente à cascata, a vantagem global vem dos 211 preservados, pois P recupera menos PAR=2 na vigilância e perde mais nos dois modos. Preservar o corte é a recomendação; a isotônica abaixo dele não superou as alternativas.

### 29. Contribuições
As entregas incluem comparador com equivalência parcial avaliada, estimadores e faixas, pacote decisório 0.4.0 com política v3.1 e artefatos pseudonimizados no Zenodo 3.0.1. O pacote e o depósito cobrem a v3.1 sobre a base inteira, não o arranjo; seus agregados estão no repositório e as saídas por par ficam locais. A adoção de Platt requer validação em dados novos, e a revisão humana precisa ser observada no serviço. AUC interna e robustez nos mesmos dados não substituem essa avaliação.

### 30. Agradecimentos
O trabalho dependeu da orientação, da equipe responsável pelos dados e da revisão manual que permitiu a avaliação. Preservar o corte pode reduzir a revisão, se o serviço aceitar o custo em vínculos: 96/17 encaminhamentos frente a pelo menos 2.156, com 13/27 perdidos. A hipótese requer avaliação em dados novos.

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
P continua principal. P-Platt troca a isotônica por *Platt scaling*; A3 também restringe o ajuste do classificador aos potenciais pares abaixo do corte; P-PAR2-grey estima o erro de não-vínculo apenas com PAR=2 nas bandas agrupadas de [5,9). São análises de sensibilidade. B1 mantém a base inteira (comparação) e a probabilidade original, mudando as taxas por validação cruzada: na vigilância, mantém 194 aceitos, com 40 falsos positivos contra 34 da cascata. P-Platt apresenta na vigilância três falsos positivos, 77 encaminhamentos e US$ 0,30, contra 21, 96 e US$ 0,94 de P, com os mesmos 13 perdidos. O comparador a posteriori é o corte seguido da cascata publicada, sem recalibração, já conhecido antes do registro. Na vigilância, aceita 221, perde nove e encaminha 90, contra 13 perdidos e 96 encaminhamentos de P, mas tem 34 falsos positivos contra 21. Na confirmação, tem zero falso positivo e 17 perdidos contra quatro e 27 de P, mas encaminha 51 potenciais pares contra 17. Os contrastes não autorizam substituir P depois de conhecidos os resultados.

### 46. Backup: robustez com registro prévio, não confirmatória
A robustez tem dois blocos: N varia threads e semente do classificador; P varia as partições. Não avalia generalização, mas estabilidade nos mesmos dados. A dominância original de P-Platt não foi sustentada: 7/20 e 4/20 na vigilância, 7/20 e 5/20 na confirmação, com dominância inversa 0/20 em ambos os modos e blocos. Na vigilância, P-Platt mantém menos FP em 20/20 réplicas de cada bloco, mas perde em mediana um e dois vínculos a mais, com menos encaminhamentos humanos e ao LLM. P perde mais PAR=2 que o comparador a posteriori em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação.

Para P na vigilância, o Jaccard mediano do lote LLM entre partições é 0,43, e as perdas variam de nove a 21 no bloco P. A mudança máxima da probabilidade é de 0,65 no bloco N e 0,80 no bloco P. As chamadas novas da robustez abrangeram 242 potenciais pares, com custo estimado de US$ 0,6255, cerca de US$ 0,63. O consumo novo do arranjo soma os US$ 0,2448 anteriores e US$ 0,6255 da robustez, sem representar fatura nem reestimar as taxas congeladas.

## Perguntas prováveis

São 22 respostas, com 60, 56, 56, 61, 66, 63, 50, 68, 57, 62, 59, 58, 55, 53, 57, 59, 56, 60, 58, 60, 59 e 57 palavras, respectivamente, sem contar pergunta e fonte. A primeira reúne a pertinência do probabilístico e o escopo do módulo Python, incluindo a equivalência Java. Mantém-se o limite de 70 palavras, para duração estimada de até 30 segundos. As fontes usam identificadores da tese e CSVs, sem depender da paginação.

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

Fonte: `data/arranjo_corte_metricas.csv:2,12`; `data/arranjo_corte_s1_resumo.csv:2-15`; `sec:e4-politica-v31`. Slides 17, 19 a 21.

### 7. Como o arranjo se compara à revisão integral?

O cenário de revisão integral projeta 99,3% sob erro humano de 5,0%. P encaminha 96/17, projeta 94,4%/89,0% e perde 13/27 vínculos. O contraponto de carga é o limite inferior de 2.156, sob hipótese de revisão decrescente de escore, não uma carga observada. As projeções não substituem decisões humanas no serviço.

Fonte: `data/carga_abaixo_corte.csv:9`; `data/arranjo_corte_metricas.csv:8,12,26,30`; `data/arranjo_corte_par2.csv:7,16`. Slides 3, 19 e 20.

### 8. Dados identificados foram enviados à nuvem?

O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). O autor recebeu apenas extrato pseudonimizado, sem registros nominais nem correspondência com pessoas. A base legal é a LGPD (arts. 7º, IV, 11, II, c, e 13). Dossiês enviados a APIs externas continham nota, atributos derivados e pseudônimo; nas execuções auditadas, retiraram-se as saídas do motor. Persiste risco de reidentificação.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`), ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`). Slides 27 e 42.

### 9. Como discutir equidade sem raça/cor?

A falta de raça/cor impede examinar disparidades raciais. Avaliaram-se sensibilidade e VPP, não toda a igualdade de oportunidades de erro. No passo 11, P tem 42,5% e 47,5% nos dois modos; na vigilância, fica abaixo dos 45,0% da base inteira (comparação). A diferença reflete sobretudo o corte e o rótulo. Ausência de rejeição demográfica não demonstra equidade.

Fonte: `data/equidade_arranjo_group_metrics.csv`, passo; `data/equidade_arranjo_pairwise_tests.csv`, rejeições de P. Slides 23 e 24.

### 10. Por que o arranjo, e não a cascata publicada?

O arranjo preserva 211 vínculos e acrescenta quatro ou oito pela política, com 13 ou 27 perdidos. Nos 36 PAR=2, aceita 4/8 contra 10/8 da cascata sobre a base inteira. A vantagem vem do corte, não de maior recuperação abaixo dele. O comparador a posteriori já era conhecido antes do registro; este cobre recalibração, braços e avaliação, não o enquadramento escolhido depois.

Fonte: `data/arranjo_corte_metricas.csv:2,4,12,22,30`; `data/arranjo_corte_par2.csv:3,7,12,16`; `gzcmd/prereg_arranjo_corte.yaml`. Slides 17, 19, 20, 26 e 28.

### 11. Por que não adotar P-Platt?

P-Platt, registrado como análise de sensibilidade, dominou P na avaliação original: 217 aceitos, três falsos positivos e 13 perdidos na vigilância. A robustez com registro prévio, não confirmatória, não sustentou essa dominância: 7/20 e 4/20 nos blocos N/P. Menos FP coexistem com perdas medianas de um/dois vínculos a mais. P permanece principal; adotar Platt exige validação em dados novos.

Fonte: `data/arranjo_corte_metricas.csv:12,14`; `data/robustez_arranjo_resumo.csv:2,3,404,488`. Slides 22 e 46.

### 12. Qual é a carga comparável à do serviço?

O limite inferior é de 2.156 potenciais pares com limiar por passo e 2.522 com limiar global SCORE ≥ −5,648. Ambos pressupõem revisão em ordem decrescente de escore, derivada do rótulo, não observada. Os 61.485 são teto teórico. O contraste principal é com o limite inferior; não se quantificam horas nem economia histórica sem medir o trabalho realizado.

Fonte: `data/carga_abaixo_corte.csv:5,9`. Slides 3 e 28.

### 13. O que significa VPP de 16% do LLM?

Na vigilância de P, quatro das 25 respostas de vínculo abaixo do corte eram vínculos, VPP de 16,0%, com IC de 6,4 a 34,7%. As demais 21 eram falsos positivos. Na confirmação, foram oito de 12, ou 66,7%. Esses resultados pertencem à região selecionada pela política, não estimam uma capacidade geral de Claude Haiku 5.5.

Fonte: `data/arranjo_corte_vpp_llm.csv:2,8`. Slide 21.

### 14. De onde vêm os vínculos perdidos?

Na vigilância de P, uma perda vem da guarda de nota inferior a 5, seis do não-vínculo automático por custo e seis da resposta de não-vínculo do LLM, totalizando 13. Na confirmação, são uma, 22 e quatro, totalizando 27. Essa decomposição identifica os ramos da decisão; não isola causalmente o efeito da calibração.

Fonte: `data/arranjo_corte_perdas_mecanismo.csv:2-4,50-52`. Slide 20.

### 15. Como justificar perder 75% dos vínculos da revisão na confirmação?

São 27 dos 36 PAR=2, ou 75,0%, não uma manutenção da recuperação do serviço. A confirmação encaminha somente 17 potenciais pares ao revisor humano, mas a redução de encaminhamentos vem com essas perdas. O estudo documenta a troca, não demonstra sua aceitabilidade clínica ou epidemiológica. A adoção depende de decisão do serviço e validação em dados novos.

Fonte: `data/arranjo_corte_par2.csv:16`; `data/arranjo_corte_metricas.csv:30`. Slides 20 e 28.

### 16. Por que as taxas congeladas não anteciparam os falsos positivos?

As taxas foram estimadas em coortes com potenciais pares mais fáceis e denominadores pequenos. A seleção abaixo do corte muda a composição. Em P na vigilância, previam cerca de 9,06 falsos positivos, contra 21 observados. A validação cruzada não garante transferência dessas proporções. É necessário estimar e validar o erro na região de uso, sem realimentação oportunista pelos desfechos.

Fonte: `data/arranjo_corte_vpp_llm.csv:2`; `data/arranjo_corte_taxas_crossfit.csv`. Slides 21 e 26.

### 17. A AUC de 0,995 elimina o viés de incorporação?

Não. A AUC descreve discriminação interna no padrão-ouro disponível; não mede independência do rótulo nem calibração absoluta. O CdR estava disponível antes da revisão, e o corte define PAR=1. O agrupamento por REFREC reduz uma dependência, mas não separa simultaneamente COMPREC nem cria referência externa. O GZ-CMD não usa os atributos de óbito dos experimentos exploratórios.

Fonte: `tab:e4-desempenho-faixas-oof`; `data/arranjo_corte_metadata.json`, `verification_bias` e `incorporation_bias`. Slides 13, 16 e 26.

### 18. O arranjo pode ser reproduzido somente com o Zenodo?

Não. O depósito e o pacote cobrem a cascata v3.1 sobre a base inteira, não o arranjo. O repositório contém código, registros, configurações e agregados do arranjo e da robustez. As saídas por par ficam locais, e o extrato pseudonimizado não é público. Reexecutar as etapas depende de acesso autorizado e das versões registradas, não apenas de instalar o pacote.

Fonte: repositório, `gzcmd/prereg_arranjo_corte.yaml` e `gzcmd/prereg_robustez_arranjo.yaml`; disponibilidade dos dados na tese. Slides 29 e 42.

### 19. O registro prévio elimina os desvios de protocolo?

Não. O código inicialmente registrado não está no git, e o suplemento fez chamadas de sensibilidade apesar da vedação original. O comparador a posteriori já era conhecido antes do registro; o enquadramento veio depois. Esses fatos limitam a proteção do registro e precisam ser declarados. P continua principal; não se reescreve o protocolo para tornar outra política vencedora.

Fonte: `gzcmd/prereg_arranjo_corte.yaml`, emendas e suplemento; `data/arranjo_corte_metadata.json`. Slide 26.

### 20. Por que manter Grey-Zone e mudar o objetivo?

O nome remete à região que a regra automática não decide, no conceito de Fellegi e Sunter, não exclusivamente à faixa [5,9) da nota do CdR. O objetivo agora trata dos potenciais pares não aceitos pelo corte: desenvolver e avaliar a cascata e comparar recuperação, erros, encaminhamentos, custo e carga com a prática do serviço, corte único e duas fronteiras.

Fonte: objetivo 4 autorizado; `sec:area-cinza`; `data/carga_abaixo_corte.csv`. Slides 4, 8 e 17.

### 21. A robustez confirma generalização e repetibilidade?

Não. É análise com registro prévio, não confirmatória, nos mesmos dados. Em P na vigilância, o Jaccard mediano do lote LLM entre partições é 0,43, e as perdas variam de nove a 21. A mudança máxima da probabilidade chega a 0,65 no bloco N e 0,80 no P. Essa instabilidade não é validação externa nem garantia de repetibilidade integral.

Fonte: `data/robustez_arranjo_resumo.csv:278,632,652,654`; `data/robustez_arranjo_metadata.json`. Slide 46.

### 22. Qual é a vantagem demonstrada frente à prática do serviço?

Contra o rótulo, o arranjo recupera menos vínculos e acrescenta falsos positivos. A única vantagem identificada é a carga: 96/17 encaminhamentos frente ao limite inferior de 2.156, sob a hipótese de revisão decrescente de escore. Isso não mede economia histórica nem aceitabilidade das perdas. A tese central permanece hipótese condicionada à carga e à decisão do serviço.

Fonte: `data/carga_abaixo_corte.csv:9`; `data/arranjo_corte_metricas.csv:12,30`; `data/arranjo_corte_par2.csv:7,16`. Slides 3, 4 e 28.

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

Compilação de teste: duas passagens de pdflatex, com jobname `defesa_s5b_teste` e saída em `.local/slides_s5b/`. O `TEXINPUTS` aponta para `.texdeps` do worktree local `pq-f-slides`, que contém Beamer e translator. Esse build não substitui `text/slides/defesa.pdf` nem os PDFs oficiais da tese.

### Verificação técnica em 05/11/2026

O build de teste em duas passagens produziu 46 páginas, com 30 quadros principais e 16 de reserva, sem sobreposições. O log não contém erros LaTeX, referências indefinidas, Overfull ou marcadores de referência não resolvida. As 22 respostas têm no máximo 68 palavras, conforme contagem por espaços. A inspeção visual abrange os quadros alterados; não constitui ensaio presencial de projeção.

A verificação desta fatia abrange os dois arquivos da apresentação, o contrato editorial dos testes e o build de teste. Resultados históricos do código científico não constituem verificação desta revisão editorial.

Log: `.local/slides_s5b/defesa_s5b_teste.log`. Em outra máquina, é necessário instalar Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los em diretório local de dependências.

SHA-256 da árvore compilada: `c5006e1a859ac630b95de50c7fec2ff18a6efb08f9125ffb746b1061bd9ff316`. O manifesto `.local/slides_s5b/arvore_compilada.json` registra os cinco arquivos de entrada do repositório efetivamente lidos pelo pdflatex, identificados pelo recorder e por seus hashes; não inclui dependências externas de TeX nem identifica o commit final. SHA-256 do PDF de teste: `3e8aa4092cb797b23db4389fc81ed4ff78f6581ba4c5a3cb2f0386254655ebd1`. Os arquivos da tese e seus PDFs não são alterados nesta fatia.

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
| 211 aceitos por construção; diferenças de sensibilidade dependem da proporção de PAR=2 em cada subgrupo, definida pelo corte, e da recuperação desses 36; diferenças de VPP dependem dos 21 e 4 FP abaixo do corte | `arranjo_corte_metricas.csv:2,12,30`; `equidade_arranjo_group_metrics.csv`; `equidade_arranjo_global.csv:2,4` |
| Na vigilância, não aceitos por passo 11/6/9: 23/4/5, todos PAR=2; no passo 11, 42,5% em P contra 45,0% na base inteira (comparação) | `equidade_arranjo_group_metrics.csv:17-19,41` |
| Passo 11: 25 dos 36 vínculos com PAR=2; P aceita 2/25 na vigilância e 4/25 na confirmação | `par_por_passo.csv:8-10,17-19,26-28,53-55` |
| Corte isolado nos passos 11/6/9: 15/40 (37,5%), 76/80 (95,0%) e 120/127 (94,5%) | `par_por_passo.csv:5-10,14-19,23-28` |
| PAR=2 aceitos nos passos 6/9/11: 0/2/2 na vigilância e 1/3/4 na confirmação | `par_por_passo.csv:8,17,26,35,44,53` |
| Pendentes de P∪B1 com chamada nova: 10 na vigilância e 14 na confirmação, não de P isolado; reutilização histórica e ausência de avaliação prospectiva | `arranjo_corte_s1_metadata.json:477-481`, campo `pending_P_union_B1_by_mode`; `arranjo_corte_metadata.json:355,357` |
| Limites inferiores de carga: 2.156 por passo e 2.522 global; SCORE global ≥ −5,648, sob hipótese de revisão decrescente de escore derivada do rótulo | `carga_abaixo_corte.csv:5,9` |
| P, vigilância: VPP do LLM 4/25 = 16,0%, IC 6,4 a 34,7%; 9,06 FP previstos contra 21 observados | `arranjo_corte_vpp_llm.csv:2` |
| P, confirmação: VPP 8/12 = 66,7%, IC 39,1 a 86,2%; P-Platt vigilância: 4/7 = 57,1%, IC 25,0 a 84,2% | `arranjo_corte_vpp_llm.csv:8,3` |
| Perdas de P por guarda/custo/LLM: 1/6/6 na vigilância e 1/22/4 na confirmação | `arranjo_corte_perdas_mecanismo.csv:2-4,50-52` |
| Perda de 13/36 = 36,1% e 27/36 = 75,0% dos PAR=2 | `arranjo_corte_par2.csv:7,16` |
| Dominância de P-Platt: vigilância N/P 7/20 e 4/20; confirmação 7/20 e 5/20; inversa 0/20 em todos os casos; FP menores em 20/20 nos dois blocos da vigilância | `robustez_arranjo_resumo.csv:2-5` |
| Vigilância: P-Platt perde em mediana 1/2 vínculos a mais, blocos N/P | `robustez_arranjo_resumo.csv:404,488`, diferenças P menos P-Platt −1/−2 |
| P perde mais PAR=2 que o a posteriori em 20/20 e 19/20 na vigilância e 20/20 nos dois blocos da confirmação | `robustez_arranjo_resumo.csv:6,8,10,12`, proporções multiplicadas por 20 |
| P na vigilância: perdas de 9 a 21 e Jaccard mediano 0,43 entre partições | `robustez_arranjo_resumo.csv:278,654` |
| Mudança máxima da probabilidade de P: 0,65 no bloco N e 0,80 no P | `robustez_arranjo_resumo.csv:632,652` |
| Robustez: 242 potenciais pares novos, US$ 0,6255, arredondados para US$ 0,63 | `robustez_arranjo_metadata.json:206,208` |
| Consumo novo anterior: US$ 0,2448; somado a US$ 0,6255 da robustez, sem representar fatura | `arranjo_corte_metadata.json:285,293,330,338`, soma dos custos novos de S2 e suplemento; `robustez_arranjo_metadata.json:208` |

Os resultados históricos de v3, dos estudos 1 a 3 e da restrição na base inteira foram mantidos com as remissões que já constavam dos slides. Não são resultados novos do arranjo, nem foram reestimados nesta fatia.
