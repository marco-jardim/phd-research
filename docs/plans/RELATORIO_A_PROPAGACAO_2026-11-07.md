# A — Propagação pelo identificador de pessoa do Sinan

## Leitura em cinco linhas

1. R1 união recupera 2/36 PAR=2 em 2 pares: isso não autoriza dispensar revisão clínica/documental.
2. Precisão contra PAR: 100.00%; PAR=0 propagados: 0. Referência operacional não é verdade adjudicada.
3. Sinal nos FP de P: 0/21 (vigilância), 0/4 (confirmação); concordância por ID não prova vínculo não examinado.
4. R2 sem pessoa propaga 4582 pares, 6 PAR=2 e 4576 PAR=0: compartilhar REFREC não basta.
5. C_IDPESSOA é completo; 13/18 REFREC múltiplos compartilham ID (72,22%) e 16/16 pessoas com múltiplos aceitos compartilham REFREC; C_NOVOIDPESSOA tem só oito valores presentes. Confiabilidade externa não demonstrada.

## Método e limites

Exploratória post hoc, CPU, sem ajuste de modelos, execução de política ou chamadas LLM. Somente vínculos PAR=1 acima dos cortes ORL são sementes; propagação em um salto, para outro COMPREC. União significa OR entre igualdades dentro da mesma coluna, sem cruzar identificadores antigos/novos, sem fusão transitiva e sem preencher ausências. Passo refere-se ao candidato de destino (6/9/11); sementes podem vir de qualquer passo. R2 exige somente REFREC aceito. Precisão indefinida se não há propagação.

FP de P reconstruídos importando `vies_referencia.load_inputs`, com políticas arquivadas/hash e respostas históricas (não novas inferências); conferência adicional importando `fp_cardinalidade.counts`. Um FP propagado seria apenas sinal compatível com vínculo não examinado, não reclassificação automática.

## Cabeçalhos e identificadores (somente agregados)

- `C_IDPESSOA,C,10,0`
- `C_NOVOIDPESSOA,C,10,0`

Registro local de notificação = (PASSO, COMPREC) único no extrato de candidatos. Completude por par sofre repetição de registros; a medida por registro e a cardinalidade deduplicam (PASSO, COMPREC). IDs literais, sem conversão numérica; vazios não concordam. Não se estima validade externa dos identificadores.

| Coluna | Registros locais | Não vazios (%) | Pessoas | Média registros/pessoa | Máximo |
|---|---:|---:|---:|---:|---:|
| C_IDPESSOA | 20004 | 100.00 | 15256 | 1.3112218143681174 | 26 |
| C_NOVOIDPESSOA | 20004 | 0.04 | 8 | 1.0 | 1 |

**Limite de identidade:** 19388 COMPREC literais, 20004 registros (PASSO, COMPREC), 17643 C_IDLINHA distintos; 606 COMPREC apontam para mais de um C_IDLINHA entre passos. Logo, COMPREC sozinho não permite contar notificações únicas no Sinan. A cardinalidade publicada conta registros locais por passo, não notificações físicas únicas. R1 mantém a regra solicitada de COMPREC literal diferente; isso pode excluir números reutilizados entre passos. Completude por registro considera qualquer valor não vazio consistente da coluna naquele registro, sem imputá-lo nos pares durante a propagação.
O metadado também publica completude e cardinalidade por C_IDLINHA (linha de origem); notificações com mais de um ID não vazio são contadas como ambíguas e excluídas somente dessa cardinalidade, sem escolher um valor arbitrário.

### Concordância interna entre PAR=1 acima do corte

| Coluna | REFREC com >1 COMPREC | ID completo | Mesmo ID | Pessoas com >1 COMPREC aceito | Mesmo REFREC |
|---|---:|---:|---:|---:|---:|
| C_IDPESSOA | 18 | 18 | 13 | 16 | 16 |
| C_NOVOIDPESSOA | 18 | 0 | 0 | 0 | 0 |

Proporções, completude por par e histograma de notificações por pessoa estão no metadado. Denominador zero resulta em null, não 100% de confiabilidade. O diagnóstico inverso considera somente COMPREC com vínculo aceito; não trata notificações sem semente como discordantes.

## Propagação por passo

| Regra | Passo | Propagados | PAR=2 | PAR=0 | Precisão PAR | FP vig. | FP conf. |
|---|---|---:|---:|---:|---:|---:|---:|
| R2_REFREC | TOTAL | 4582 | 6 | 4576 | 0.13% | 1 | 0 |
| R2_REFREC | 6 | 207 | 1 | 206 | 0.48% | 0 | 0 |
| R2_REFREC | 9 | 94 | 1 | 93 | 1.06% | 1 | 0 |
| R2_REFREC | 11 | 4281 | 4 | 4277 | 0.09% | 0 | 0 |
| R1_C_IDPESSOA | TOTAL | 2 | 2 | 0 | 100.00% | 0 | 0 |
| R1_C_IDPESSOA | 6 | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_C_IDPESSOA | 9 | 1 | 1 | 0 | 100.00% | 0 | 0 |
| R1_C_IDPESSOA | 11 | 1 | 1 | 0 | 100.00% | 0 | 0 |
| R1_C_NOVOIDPESSOA | TOTAL | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_C_NOVOIDPESSOA | 6 | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_C_NOVOIDPESSOA | 9 | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_C_NOVOIDPESSOA | 11 | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_UNIAO | TOTAL | 2 | 2 | 0 | 100.00% | 0 | 0 |
| R1_UNIAO | 6 | 0 | 0 | 0 | indefinida | 0 | 0 |
| R1_UNIAO | 9 | 1 | 1 | 0 | 100.00% | 0 | 0 |
| R1_UNIAO | 11 | 1 | 1 | 0 | 100.00% | 0 | 0 |

## Reprodução

`python scripts/propagacao_pessoa.py --data-root <dados> --policy-dir <politicas>`

`python -m pytest tests/test_propagacao_pessoa.py tests/test_s9c.py -q`

Entradas privadas opcionais via `tests/local_data.py` e `PHD_RESEARCH_POLICY_DIR`; saídas públicas exclusivamente agregadas. Hashes SHA-256 de código (LF), extrato, políticas e respostas, versões e convenções em `data/propagacao_pessoa_metadata.json`.
