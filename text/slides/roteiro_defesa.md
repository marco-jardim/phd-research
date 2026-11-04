# Roteiro da defesa

Base da apresentação: tese em `dc25212`, após T5. A numeração abaixo corresponde aos quadros do PDF, sem sobreposições. O tempo é uma estimativa para ensaio, não uma duração medida. Os slides 31 a 42 ficam reservados às perguntas, fora dos 40 minutos.

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
A pesquisa foi desenvolvida no IESC-UFRJ, sob orientação de Rejane Sobrino Pinheiro, com foco no pós-processamento dos potenciais pares de bases de saúde. A apresentação acompanha cinco estudos e distingue a evidência exploratória da avaliação final da cascata.

### 2. Informação sobre o cuidado
A tuberculose é uma condição marcadora, e a vinculação entre notificações e óbitos permite examinar desfechos do cuidado. O problema estudado não é apenas computacional: uma decisão incorreta de vínculo pode modificar a informação usada pela vigilância.

### 3. Volume na área cinza
A área cinza da nota final do CdR reúne 21.620 potenciais pares e 151 vínculos. O desafio é preservar a recuperação desses vínculos sem exigir que toda essa faixa passe pelo revisor humano, mantendo explícitas as consequências dos erros.

### 4. Tese central
A área cinza passa de um problema de volume a um problema de decisão. O percurso articula comparador, estimador, estratégias, cascata e equidade, sem reduzir a contribuição a um classificador de melhor desempenho global.

### 5. Fontes e blocagem
Foram analisados 61.696 potenciais pares e 247 vínculos, com razão aproximada de 1:249, gerados em três passos exportados do OpenRecLink. O passo ampliado concentra o volume, mas não a maioria dos vínculos; a avaliação está condicionada ao extrato, e não estima a sensibilidade de toda a blocagem nas bases de origem.

### 6. Referência operacional
O corte do OpenRecLink aceitou 211 dos 247 vínculos, ou 85,4%, e a revisão manual acrescentou 36. O corte 9 da nota do CdR é outro objeto, um cenário hipotético que aceitaria 93 vínculos; o gabarito incorpora a rotina do serviço, de modo que não oferece avaliação independente da acurácia do próprio corte.

### 7. Faixas e padrão-ouro
As faixas são da nota final do CdR, não do escore probabilístico do OpenRecLink. Há três vínculos abaixo de 5, 151 na faixa intermediária e 93 a partir de 9; uma revisora realizou a classificação, sem segunda avaliação independente para medir concordância humana.

### 8. Mapa conceitual
O referencial clássico organiza a decisão entre vínculo, não-vínculo e revisão. A tese trabalha depois da geração dos potenciais pares e do cálculo dos subescores, reorganizando a decisão sobre a saída do CdR, sem reestimar os parâmetros de comparação nem substituir a blocagem do OpenRecLink.

### 9. Comparador em Python
O comparador torna explícita a evidência campo a campo, com subescores e uma nota ponderada por pesos fixos. A implementação em Python foi documentada, mas a equivalência parcial com o protótipo Java não foi executada; não se apresenta uma interface do GZ-CMD nem se simula uma captura de tela ausente.

### 10. Interpretação dos atributos
SHAP informa a contribuição dos atributos à predição, sem interpretação causal, e uma penalidade por nome comum não deve ser chamada de concordância. Os vetores também diferem: a ablação e o QW1 usam 58 atributos, enquanto o GZ-CMD usa os 29 subescores e a nota, sem indicadores de óbito.

### 11. Comparação de protocolos
RF+SMOTE atingiu F1 de 0,931 na ablação original, e o XGBoost do fluxo Modal atingiu 0,943, contra 0,610 do limiar 8. O XGBoost do QW1 atingiu 0,920 no teste; os protocolos não são intercambiáveis, e a escolha dos limiares nos testes torna os resultados selecionados exploratórios.

### 12. Balanceamento
As nove estratégias de balanceamento foram avaliadas no Random Forest, com F1 médio entre 0,880 e 0,918. Essa variação descreve o experimento realizado, não demonstra que uma estratégia de reamostragem seja necessária ou superior em qualquer modelo.

### 13. Estimador final
O GZ-CMD usa outro XGBoost, com 30 atributos e calibração isotônica aninhada, agrupando por REFREC nos níveis externo e interno. Na área cinza, a AUC é 0,995 e a AP é 0,862; o corte ilustrativo de 0,5 produz 119 verdadeiros positivos e 28 falsos positivos, mas não é a política final de decisão.

### 14. Finalidades operacionais
A vigilância atribui maior consequência à perda do vínculo, enquanto a confirmação atribui maior consequência ao falso vínculo. A fronteira permite explicitar esse compromisso, sem transformar os pontos explorados no teste em limiares validados para uso externo.

### 15. Impacto epidemiológico
No teste da ablação, o RF+SMOTE detectou 67 vínculos e o limiar 8 detectou 43, uma diferença líquida de 24, ou 55,8%. São 27 recuperados exclusivamente pelo modelo, menos três exclusivos do limiar; esse resultado tem como denominador 74 vínculos do teste e não mede o impacto da cascata final.

### 16. Limites do impacto
Os indicadores derivados do encerramento por óbito constituem vazamento de alvo, e a partição por linha pode compartilhar registros entre treinamento e teste. Essas limitações atingem o QW1 e a ablação, não os atributos nem a partição por REFREC do GZ-CMD, embora permaneça a dependência possível pelo lado COMPREC.

### 17. Arquitetura atual
As guardas da v3.1 são regras pela nota e pela probabilidade, distintas dos filtros históricos da v3. Depois delas, as faixas informam o perfil de erro do revisor, a probabilidade calibrada alimenta a comparação de custos, e os casos escolhidos seguem ao LLM ou diretamente ao revisor humano.

### 18. Custos e ações
O custo do falso negativo é maior na vigilância, enquanto o custo do falso positivo é maior na confirmação. As quatro ações são comparadas na mesma escala de utilidade, e essa escala não deve ser confundida com dólares nem com custos econômicos medidos no serviço.

### 19. Vigilância
A sensibilidade automática passou de 58,3% para 78,5%, com IC descritivo de 73,0% a 83,2%, e os falsos positivos passaram de 35 para 34. Os vínculos perdidos caíram de 91 para 15, com aumento da revisão humana de 61 para 111 e redução do lote LLM de 1.410 para 347; os 38 vínculos ao humano continuam pendentes.

### 20. Confirmação
A sensibilidade automática passou de 40,5% para 53,4%, com IC descritivo de 47,2% a 59,6%, sem falso positivo automático observado na v3.1. A carga humana passou de 25 para 127 e o lote LLM caiu de 435 para 44; os 87 vínculos encaminhados não podem ser tratados como rejeições nem como acertos humanos observados.

### 21. Revisor de referência
Claude Haiku 5.5 com instruções v4 revisa a evidência sem receber as saídas decisórias do motor. A ablação é compatível com ancoragem nessas saídas, mas não demonstra causalidade geral; Claude Sonnet 5.5 recuperou apenas dois a quatro vínculos adicionais na cascata, com mais encaminhamentos e tarifa vinte vezes maior por token.

### 22. Limites da automação
Sob erro humano hipotético de 5,0%, as sensibilidades após revisão seriam 93,2% e 86,9%, mas essas decisões não foram observadas. A inviabilidade da restrição de 0,2% de falsos vínculos esperados mostra por que a confirmação não pode ser ampliada apenas pela ausência de falsos positivos no extrato.

### 23. Equidade por sexo
No XGBoost com sensibilidade equivalente, o VPP foi de 93,1% no sexo feminino e 70,7% no masculino. Na cascata, essa diferença foi menor, mas a ausência de rejeição não comprova equidade; a análise não cobre raça/cor nem corresponde à avaliação completa de igualdade de oportunidades de erro.

### 24. Desigualdade operacional
Na cascata de vigilância, a sensibilidade automática foi de 45,0% no passo 11 e 90,6% no passo 9. Cinco das nove rejeições de paridade recaem no passo de blocagem, e a revisão humana dos vínculos encaminhados é necessária para avaliar a equidade do processo completo.

### 25. Posicionamento
Os estudos anteriores já apresentam classificação supervisionada, balanceamento e redução de revisão, de modo que a contribuição não é a inexistência desses componentes. A tese propõe sua composição operacional com calibração agrupada, custos, encaminhamento e equidade, sem comparação empírica com Lages e sem ordenar métricas de bases distintas como se fossem do mesmo experimento.

### 26. Limitações de método
O padrão-ouro incorpora decisões de uma revisora e do corte, e a avaliação continua condicional à blocagem. O agrupamento por REFREC não separa simultaneamente COMPREC, e as proporções do revisor usam respostas de lotes selecionados adaptativamente, impedindo tratar a cascata como validação prospectiva.

### 27. Limitações de transferência e governança
O estudo é municipal, sem validação externa, e o extrato é pseudonimizado, não anônimo. O processamento LLM utilizou APIs externas; exportação restrita, HMAC e reconferência documental não podem ser apresentados como salvaguardas já entregues, assim como não foram executadas a equivalência Java e a comparação com Lages.

### 28. Conclusões
A vigilância aceita automaticamente a maior parte dos vínculos, assumindo o custo observado de 34 falsos positivos. A confirmação concentra o esforço humano nos potenciais pares ainda intermediários; reconhecer esse limite faz parte do resultado, não representa falha a ser escondida por uma projeção com revisor perfeito.

### 29. Contribuições
As entregas incluem comparador, avaliação dos estimadores e das faixas, pacote decisório e artefatos pseudonimizados de reprodução, com limites declarados. O passo seguinte requer referência independente, observação da revisão humana e avaliação externa, não apenas ajuste adicional do classificador na mesma base.

### 30. Agradecimentos
O trabalho dependeu da orientação, da equipe responsável pelos dados e da revisão clerical que permitiu a avaliação. A mensagem de encerramento é a transformação da área cinza em decisão, preservando o papel do profissional do serviço onde a evidência não permite avançar automaticamente.

### 31. Backup: cascata gráfica
A figura sintetiza os indicadores da cascata por política e modo. Para consultar os ramos por faixa, deve-se retornar à tabela completa da tese; os subtotais humanos incluem encaminhamento direto e inconclusivos do LLM, não constituindo novos ramos a somar novamente.

### 32. Backup: contagens finais
As contagens separam aceitação automática, perda explícita e vínculo pendente ao humano. A sensibilidade automática inclui o LLM, mas não inclui a decisão humana ainda não realizada.

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

### 41. Backup: sensibilidade
Os exercícios de exclusão e de expectativas por banda avaliam dependências das escolhas e não são intervalos de confiança. A sobreposição residual e a seleção adaptativa permanecem, mesmo quando os valores se aproximam da cascata observada.

### 42. Backup: privacidade
A ausência de nomes no dossiê não transforma o extrato em dado anônimo nem elimina risco de reidentificação. O hash determinístico pode ser recalculado por quem detém os identificadores de origem, e a documentação de acesso e transferência exige reconferência.

## Perguntas prováveis

O plano histórico contém dez perguntas, embora o pedido especifique nove. A primeira resposta reúne a pertinência do probabilístico e o escopo do módulo Python, preservando os dez temas em nove respostas novas. Cada resposta tem menos de 70 palavras, para duração estimada de até 30 segundos, a confirmar em ensaio. As fontes usam as seções e os identificadores estáveis da tese, sem depender da paginação do PDF.

### 1. O probabilístico é inútil? Onde está o módulo Python do OpenRecLink?

O OpenRecLink produziu os potenciais pares e seu corte aceitou 211 dos 247 vínculos, ou 85,4%. A entrega em Python é o CdR e o pós-processamento decisório, não a blocagem nem o OpenRecLink completo. A equivalência parcial com o protótipo Java não foi executada. A proposta complementa a rotina existente, explicitando quando aceitar, rejeitar ou revisar.

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

Luís Guilherme Buteri Alves realizou o pré-processamento, e Daniela Medina Coeli Bastos configurou a vinculação e revisou os potenciais pares. Dos 247 vínculos, 211 vieram do corte e 36 da revisão abaixo dele. Não houve duas classificações independentes nem kappa humano. A referência incorpora o corte, tem viés de verificação e limita a sensibilidade aos potenciais pares exportados.

Fonte: `capitulo_dados.tex`, Padrão-ouro (`sec:padrao-ouro`). Slides 6 e 7.

### 6. A cascata é determinística?

As guardas são determinísticas, mas a cadeia completa inclui probabilidade estimada, comparação de custos e Claude Haiku 5.5, cuja geração é estocástica. Na vigilância, a v3.1 aceitou 194 vínculos, perdeu 15 e encaminhou 38 ao humano. São três saídas diferentes, não uma promessa de repetibilidade integral nem uma demonstração de independência do revisor em relação à seleção dos lotes.

Fonte: `estudo4_gzcmd.tex`, Política v3.1 (`sec:e4-politica-v31`), Cascata final (`sec:e4-cascata`) e Limitações (`sec:e4-limitacoes`). Slides 17, 19 e 21.

### 7. Como a cascata se compara à revisão integral?

Revisar toda a área cinza exigiria 21.620 decisões humanas, como cenário hipotético. A cascata encaminhou 111 potenciais pares na vigilância e 127 na confirmação, com sensibilidades automáticas de 78,5% e 53,4%. As sensibilidades de 93,2% e 86,9% após revisão são projeções sob erro humano de 5,0%, não acertos observados. A carga histórica revisada pelo serviço não está documentada.

Fonte: `tab:cap8-referencia`; `tab:cap8-cascata-v31`; `estudo4_gzcmd.tex`, Cascata final (`sec:e4-cascata`). Slides 6, 19, 20 e 22.

### 8. Dados identificados foram enviados à nuvem?

O extrato de 61.696 linhas é pseudonimizado. O autor trabalhou exclusivamente com ele, sem acesso aos registros nominais nem à correspondência entre identificadores e pessoas. A base legal está nos arts. 7º, IV, 11, II, c, e 13 da LGPD. Dossiês enviados a APIs externas continham também nota, atributos derivados e pseudônimo; as saídas do motor foram retiradas nas execuções auditadas. Persiste risco de reidentificação.

Fonte: `capitulo_dados.tex`, Acesso e aspectos éticos (`sec:dados-acesso-etica`), ética dos modelos (`sec:dados-etica`) e Ambiente computacional (`sec:ambiente-computacional`). Slides 27 e 42.

### 9. Como discutir equidade sem raça/cor?

A ausência de raça/cor limita a avaliação. Foram examinadas sensibilidade e VPP, não toda a igualdade de oportunidades de erro. A cascata apresentou nove rejeições de paridade, concentradas em eixos operacionais; a sensibilidade foi de 45,0% no passo 11 e 90,6% no passo 9. Ausência de rejeição demográfica não demonstra equidade, e a revisão humana pendente pode alterar o resultado completo.

Fonte: `estudo5_equidade.tex`, Resultados (`tab:equidade-v31-rejeicoes-cascata`), Discussão e limitações. Slides 23 e 24.

## Figuras e critérios de entrega

Figuras reaproveitadas, sem alterar os originais:

| Slide | Arquivo | Uso e ressalva |
| :-- | :-- | :-- |
| 31 | `text/latex/figures/fig_cap8_cascata_v31.png` | Síntese visual dos indicadores, cascata por REFREC |
| 36 | `figures/imbalance_sensitivity.pgf` | Estratégias de balanceamento no Random Forest |
| 37 | `text/latex/fig_doc/epi_deteccao.pgf` | Detecção no teste da ablação, não cascata final |

A captura `figures/fig_cdr_gui_*.png` não estava presente no worktree na preparação. O slide 9 mantém um espaço reservado comentado, sem imagem fabricada. A tabela do slide 25 reduz a tabela R2 e incorpora Paiva (2025) do parágrafo imediatamente anterior da mesma seção, pois ele não tem linha própria na tabela atual da tese.

Critérios editoriais contemplados: menção ao referencial clássico restrita ao slide 8; arquitetura identifica a origem da probabilidade; cascata REFREC; corte operacional 211/247; atribuição do F1 0,943 ao fluxo Modal; limitações nos slides 26 e 27, antes das conclusões 28 e 29; equivalência Java e comparação com Lages declaradas não executadas; terminologia área cinza; ausência de travessões na prosa. As tabelas foram reduzidas no arquivo de slides, sem incluir tabelas completas da tese por comando de entrada.

Pendente de validação humana: ensaio cronometrado, conferência visual da legibilidade das figuras de backup e revisão do roteiro com a orientadora. Os testes automáticos verificam estrutura e compilação, mas não substituem essas etapas.

Compilação: `python text/slides/build_slides.py`, com duas passagens de pdflatex. O PDF versionado é `text/slides/defesa.pdf`; os auxiliares já são cobertos pelo `.gitignore` do repositório. A compilação dos slides não escreve nos PDFs oficiais da tese.

### Verificação técnica em 04/11/2026

O build com pdflatex produziu 42 páginas (30 principais e 12 de backup), sem erro LaTeX nem ocorrência de `Overfull` no log final. A leitura por pypdf confirmou 42 páginas, formato 16:9 e nenhuma página sem texto extraível. A busca por quadro encontrou a menção ao referencial clássico apenas no slide 8; a auditoria editorial encontrou zero ocorrências da terminologia vedada e de travessões. As nove respostas têm, respectivamente, 56, 56, 56, 61, 58, 59, 58, 61 e 61 palavras, sem contar pergunta e referência.

O comando `python -m pytest tests -q` apresentou inicialmente 254 aprovações e cinco falhas por dependências locais ausentes no worktree. Após copiar do clone principal, somente para uso local, o extrato, os agregados históricos por linha, a tabela histórica e o auxiliar de preservação, a execução resultou em **259 testes aprovados e dois avisos em 17,04 segundos**. Cinco desses testes são novos, em `tests/test_build_slides.py`; os avisos são de divisão por zero e separação perfeita no teste log-linear de blocagem, não dos slides. Nenhum desses insumos locais integra o commit da apresentação.

Logs locais: `text/slides/build.log`, `text/slides/defesa.log` e `text/slides/pytest.log`, ignorados pelo Git. O TeX Live disponível não continha Beamer nem translator; foram obtidos do CTAN em `.texdeps/`, sem alterar a instalação global. O construtor inclui esse diretório local na busca quando ele existe. Em outra máquina, é necessário instalar os pacotes usuais Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los nesse diretório; não são incorporados ao PDF como uma dependência de execução.
