# Estudo 5: equidade do arranjo principal P

`scripts/equidade_arranjo.py` importa as funções de `equidade_v31.py` sem
alterar esse arquivo ou as rotinas de `qw3_fairness_analysis.py`. Não ajusta
modelos ou limiares, não chama API e não altera o capítulo do Estudo 5.

As decisões finais são recompostas pelas mesmas funções da S3: o corte
aceita os potenciais pares acima dele, enquanto P decide abaixo; MATCH do
motor ou do modelo de linguagem é aceitação automática e UNSURE/DEFERRED é
encaminhamento ao revisor humano. As respostas existentes, inclusive os
suplementos autorizados, são reaproveitadas sem reestimar taxas. A cascata
publicada é analisada nos dois modos como referência. Os totais de P são
conferidos contra os agregados da S3.

O adaptador para `analisar_decisao` usa os nomes de colunas que a função
original exige para contabilizar encaminhamentos humanos. Essa adaptação é
local a uma cópia do quadro em memória; não modifica o módulo importado,
seus parâmetros, seus grupos ou sua família de testes.

Mantêm-se sexo e faixa etária como dimensões demográficas, e quartil do
nome, encerramento da notificação (C_SITUENCE 3 ou 4), município, passo,
ano do óbito e completude como dimensões operacionais. A construção dos
grupos, o mínimo de 30 potenciais pares, Wilson de 95%, qui-quadrado/Fisher,
razões de chances, diferenças e Benjamini-Hochberg são os das funções
originais. Há 58 comparações candidatas por decisão; denominadores zero
produzem omissões, e BH usa os valores de p definidos, separadamente por
decisão. Os testes não corrigem dependência entre potenciais pares que
compartilham REFREC; seus valores de p podem ser otimistas. A comparação
entre decisões não é um teste formal de mudança de disparidade.

Há 403 potenciais pares, com um vínculo, sem faixa etária classificada.
O código original exclui MISSING da inferência. Para preservar a reprodução
das tabelas anteriores, a mesma exclusão é mantida nos quatro agregados
principais. O CSV adicional de cobertura inclui todos os grupos, inclusive
MISSING, e permite verificar 61.696 pares e 247 vínculos em cada dimensão,
sem imputar idade ou acrescentar uma comparação à família histórica.

Os quatro agregados principais conservam o formato `equidade_v31_*`:
global, group_metrics, pairwise_tests e maxdiff. Os resultados da cascata
publicada são comparados numericamente com os arquivos existentes, incluindo
as nove rejeições no modo vigilância e uma no modo confirmação. Os arquivos
de comparação e cobertura e o manifesto são adicionais; nenhum deles contém
linhas por potencial par, identificadores, nomes, endereços ou datas completas.

A referência de 7.452 preditos positivos é conferida pelo corte histórico
6,0 da nota final. Não é o corte SCORE do OpenRecLink e não é a contagem
de aceitos por P. As cifras de VPP de 93,1%/70,7% e de falsos positivos
de 0,04%/0,14% por sexo no texto atual descrevem o XGBoost com sensibilidade
equivalente ao corte, não a cascata publicada. As saídas novas distinguem P
da cascata e não reinterpretam aquelas cifras como resultados desta última.

As tabelas LaTeX novas mantêm as colunas e as convenções das tabelas atuais,
com rótulos próprios. Os geradores originais de grupos e de rejeições são
reaproveitados com adaptações de apresentação para P e para cada modo. Não
há inclusão em capítulo nem alteração das tabelas existentes nesta fatia.
