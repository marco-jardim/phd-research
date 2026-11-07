# v4 candidata: registro anterior à coorte nova

Decisão do autor D20/D21. Esta especificação não altera o perfil histórico
`p_cal="platt"`, `policy_version="v3"`, nem constitui validação da candidata.
As análises de desenvolvimento são exploratórias post hoc. A margem foi fixada
com os resultados de desenvolvimento à vista, não estimada nem testada.

## Comparação e falsificação

A comparação primária é O3 (probabilidade calibrada fora da amostra do
classificador) versus O1 (escore do OpenRecLink). Mantém-se o critério K3:
a hipótese seria refutada se O3 não atingisse a recuperação obtida por P com
pelo menos 25% menos revisões que O1, pelo limite inferior do intervalo de 95%
entre partições predefinidas, em cada uma das duas metas. O4 permanece secundária,
como no critério da tese. Não se atribui o ganho à política de quatro ações.

As metas históricas de 9, 23 e 36 PAR=2 correspondem, na coorte nova, a 25%,
64% e 100% dos vínculos da referência abaixo do corte. As duas primeiras são
as metas primárias; a recuperação integral é secundária. A comparação com a
recuperação de P no orçamento prévio permanece obrigatória, sem substituição
retrospectiva por uma meta mais favorável. Ordenação, orçamento e partições
devem ser fixados antes dos novos desfechos.

## Filas secundárias fixadas

F4 é RRF de O1 e O3, constante k=60, soma de 1/(60+posição), posições em base
1. O desempate é pelo hash lexical crescente, exatamente k candidatos, sem
expansão dos empates. CM é O3 acrescida dos atributos de contexto e das MACD,
com a lista completa no YAML, idêntica a `data/contexto_obito_metadata.json`.
Contexto usa a nota final no universo completo, sem labels; REFREC e COMPREC
são somente chaves, nunca valores preditores. Empates recebem posição mínima;
a segunda nota mantém multiplicidade; grupos unitários têm diferenças zero.
O contexto é transdutivo e COMPREC pode atravessar partições por REFREC.

Preservam-se os hiperparâmetros, seed 42, partições externas e internas 5×5
por REFREC, calibração isotônica aninhada e pesos recalculados no treinamento.
Não haverá ajuste com os desfechos da coorte nova. Os detalhes e dependências
de código são vinculados pelos hashes de `data/v4_candidata_registro.json`.

## Referência, parada e exclusões

A referência cega independente será construída por amostragem estratificada,
com dois revisores humanos e arbitragem, cobrindo discordâncias de P, vínculos
perdidos e não vínculos por faixa de escore. As probabilidades de inclusão e
a análise dos estratos devem ser fixadas antes da coorte, sem tomar PAR=0
como prova de ausência de vínculo.

A regra de parada deverá oferecer garantia estatística de recuperação.
O método, seus parâmetros e a definição operacional da garantia estão
pendentes de escolha ANTES da coorte. Este registro não autoriza iniciar
a avaliação sem resolver essa pendência e fixar orçamento e partições.

Excluem-se da candidata LLM como evidência, dossiê v5 e encaminhamento conformal.
O probe D não atingiu o critério pré-registrado de utilidade; não se infere
inutilidade geral dos campos nem se atribui esse resultado ao encaminhamento
conformal, que não foi testado pelo probe. Não há novas chamadas de LLM.

Reprodução do registro: `python scripts/v4_candidata_registro.py`.
Verificação: `python -m pytest tests/test_w11a.py -q`.
