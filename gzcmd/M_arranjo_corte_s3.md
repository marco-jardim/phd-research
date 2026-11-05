# Arranjo corte: avaliação S3

S3 é determinística, sem novos ajustes de modelos ou chamadas de API. Confere
o pré-registro emendado, a linhagem da S1 e o hash das taxas congeladas, além
das contagens e probabilidades dos arquivos de decisão locais da S1. As
respostas observadas incluem as três coortes históricas, a S2 de P+B1 e o
suplemento de execução autorizado para 69 pares das sensibilidades.

O suplemento foi registrado antes da execução. Não altera braços, custos,
parâmetros, seed, folds ou calibração. Nenhuma resposta nova alimenta a
estimação de confiabilidade. Resultados e journals por par continuam locais;
somente metadados agregados e auditorias são versionados.

O script `scripts/arranjo_corte_orl_gzcmd.py` foi incorporado byte a byte do
clone, sem alterar sua lógica. Sua análise a posteriori é regenerada em
diretório privado e exportada em `data/arranjo_corte_aposteriori.csv`. Esse
comparador aplica o corte à cascata publicada sem recalibrá-la. A avaliação
S3 recompõe o mesmo arranjo a partir das decisões por par e confere contagens
e custo contra a regeneração independente.

São avaliados corte sozinho, cascata publicada, B1 na base inteira, corte
com revisão humana integral abaixo, a posteriori e os quatro braços
restritos P, P-Platt, A3 e P-PAR2-grey. Nos braços restritos, os potenciais
pares acima do corte são aceitos pelo OpenRecLink antes do motor. B1 e a
cascata publicada conservam suas decisões na base inteira.

MATCH do motor ou do LLM é aceitação automática; NONMATCH é vínculo perdido
quando o gabarito é positivo; DEFERRED e UNSURE são encaminhamentos ao humano.
Não há imputação de resposta ausente. A partição aceitos + humano + perdidos
deve fechar para todos os vínculos. UNSURE integra uma vez a carga humana,
embora o mesmo par também tenha consumido LLM.

As métricas usam 247 vínculos na base e 151 na área cinza [5,9), definida pela
nota final, não por SCORE. Wilson de 95% descreve a aceitação automática e,
na tabela primária dos 36 PAR=2, cada categoria da partição. Não corrige
dependência entre potenciais pares nem vieses do gabarito. O corte participa
da definição de PAR=1, e os rótulos também entram nos ajustes supervisionados,
de modo que há ressalvas de verificação e incorporação; OOF por REFREC não
transforma esses resultados em validação externa.

Chamadas e custos são somados somente para os pares selecionados por cada
braço e faixa, a partir dos journals registrados, incluindo o protocolo de
agentes e suas tentativas. Não se atribui o custo do run inteiro a um subconjunto.
As tarifas históricas geram estimativas por tokens, não faturas. O cenário
com humano é (aceitos + 0,95 × vínculos ao humano)/vínculos; não representa
decisões humanas observadas.

As saídas são agregadas. A tabela LaTeX usa booktabs, tamanho footnotesize,
vírgula decimal e ressalvas na legenda. Nenhum capítulo inclui a tabela nesta
fatia; a decisão de texto e enquadramento permanece para S4.
