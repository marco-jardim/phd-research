# Ablação do dossiê: publicação agregada (D8)

`scripts/ablacao_dossie_agregado.py` transforma as duas entradas locais
`data/ablacao_dossie_pares.csv` e `data/ablacao_dossie_resultado.csv` em
`data/ablacao_dossie_agregado.json`. Executar antes de retirar as entradas, ou com
cópias privadas e `--data-root`. O agregado registra SHA-256 dos bytes de entrada,
contagens de revisores que erraram e contagens por modelo/decisão em cada braço.
Não inclui identificadores de par, notas ou linhas individuais.

Os CSVs de entrada permanecem ignorados e não são distribuídos pelo Git. O runner
histórico pode reconstruí-los localmente; ele não é executado por este gerador e
nenhuma chamada de API é necessária. Sem as entradas privadas não se reproduzem
seus hashes a partir do agregado. As contagens publicadas são conferidas contra
`data/sprint3b/ablacao/resumo.json` por testes offline.

A retirada do histórico e da coluna `y` dos quatro CSVs de comparação sem saídas
do motor é feita em mirror isolado. Tags de backup privadas preservam o histórico
antigo localmente; não devem ser enviadas ao GitHub. A reescrita das branches não
constitui purga de caches ou refs internas de PRs do GitHub.
