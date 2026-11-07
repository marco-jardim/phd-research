# S10: limitação dos finais de linha do pré-registro da S1

A conferência do commit `a7d77b2` utilizou dois clones novos, um com
`core.autocrlf=true` e outro com `core.autocrlf=false`. Nos dois, o arquivo
`gzcmd/prereg_arranjo_corte.yaml` foi materializado somente com LF, conforme
a regra explícita do `.gitattributes`, independentemente de `autocrlf`.

## Hashes observados

| Referência | SHA-256 dos bytes |
|---|---|
| Clone novo, `core.autocrlf=true` | `c35d5100aa9af7a97582a36e864e589380b677d6521410e2440b33f1bf611392` |
| Clone novo, `core.autocrlf=false` | `c35d5100aa9af7a97582a36e864e589380b677d6521410e2440b33f1bf611392` |
| `data/arranjo_corte_metadata.json:prereg_sha256` | `3f156cfe3c316cef63a3349f4d46608128cc28893f6691d314c508a3b183c278` |
| Worktree original do Windows, com finais mistos | `3f156cfe3c316cef63a3349f4d46608128cc28893f6691d314c508a3b183c278` |

O `git ls-files --eol` do worktree original registra `i/lf`, `w/mixed` e
`attr/text eol=lf`. Nos clones novos, há 387 finais LF e nenhum CRLF.
O blob Git é a versão LF; o hash histórico corresponde aos bytes mistos do
worktree original, não aos bytes recuperados pelo checkout novo.

## Consequência para a reprodução

**Reproduzir a S1 exige o checkout original do Windows; o blob git é LF e o
worktree original tinha finais mistos.** Mais precisamente, é necessária a
cópia histórica com os bytes registrados, além dos insumos e das versões
correspondentes; um checkout novo do Git, isoladamente, não a recupera.

Em `scripts/arranjo_corte_s1_simular.py`, `verify_prereg` compara primeiro
`sha256(PREREG)` ao hash esperado. A função `sha256`, importada de
`gzcmd/arranjo_corte.py`, lê os bytes sem normalizar finais de linha. Nos dois
clones, o hash calculado por essa mesma função foi o hash LF acima e a chamada
a `verify_prereg` com o hash dos metadados foi interrompida com
`ValueError: Preregistration SHA256 mismatch`, antes de acessar os insumos
privados ou iniciar a S1. A conferência não executou treinamento ou API.

## Renormalização e preservação

Nos dois clones, `git add --renormalize .` seguido de `git status --short`
não apresentou alterações. A estabilidade da renormalização, portanto,
não demonstra correspondência ao hash histórico: ela demonstra estabilidade
do checkout canônico LF sob as regras atuais.

A saída de `git ls-files --eol` foi registrada para 120 caminhos selecionados,
incluindo os pré-registros, o protocolo do filtro, metadados, manifestos,
tabelas históricas e scripts identificados por fingerprints versionados.
Os registros completos desta conferência ficam em `.local/final/`, fora do PR.

O arquivo congelado, os metadados e o código de validação não foram alterados.
Não se recalculou o hash esperado nem se afrouxou o gate. A aprovação dos
testes portáveis, da regeneração agregada e dos builds não deve ser apresentada
como reprodução integral da S1 a partir de um clone novo.
