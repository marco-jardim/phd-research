# Protocolo de busca bibliográfica (T6)

Data das buscas: 2026-11-03 (buscas exploratórias de 2026-11-01 não contadas). Período: 2010 a 2026. Triagem por um único revisor, por título e, nos registros mais próximos do critério, por leitura do resumo; sem leitura na íntegra.

Critério de inclusão: classificador supervisionado aplicado à classificação de pares candidatos em bases brasileiras de saúde, treinado em pares rotulados.

## Artefatos e reprodução

| Arquivo | Conteúdo |
|---|---|
| `scripts/protocolo_busca_T6.py` | coleta no PubMed (E-utilities) e no OpenAlex (API), grava os JSONs e regera o CSV; `--offline` regera o CSV sem coletar |
| `scripts/protocolo_busca_T6_tabelas.py` | regras e decisões de triagem; geração do CSV |
| `data/protocolo_busca_T6_pubmed.json` | brutos do PubMed: cadeias exatas, URL, PMIDs e metadados (sem resumos) |
| `data/protocolo_busca_T6_openalex.json` | brutos do OpenAlex: cadeias exatas, URLs de cada página, IDs e metadados (sem resumos; indicadores booleanos do filtro de resumo) |
| `data/protocolo_busca_T6_manual.json` | SciELO, BVS/LILACS (navegador), Consensus, buscas exploratórias e itens conhecidos, com decisão de triagem |
| `data/protocolo_busca_T6.csv` | todos os registros e consultas, com `data_busca`, `string_exata`, `filtros`, `n_recuperado` e decisão; sem resumos e sem dados pessoais |
| `tests/test_protocolo_busca_T6.py` | teste da geração do CSV com fixtures pequenas |

Reprodução: `python scripts/protocolo_busca_T6.py`. Os resultados do PubMed e do OpenAlex podem variar com o tempo; a rodada de referência é a de 2026-11-03. O OpenAlex limita a paginação profunda por cursor com `search` (HTTP 401 na segunda página), de modo que a coleta usa `page=1..N`, com `per-page=200`. SciELO e BVS exigem JavaScript e recusam requisições diretas (HTTP 403): foram consultadas por navegador na primeira rodada; na segunda, o navegador não estava disponível.

## Cadeias

Principal:

```
("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") AND ("machine learning" OR "aprendizado de máquina" OR supervised) AND (Brasil OR Brazil OR SIM OR SINAN OR DATASUS)
```

Ampliada (A1):

```
("record linkage" OR "data linkage" OR "relacionamento de registros" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros" OR "pareamento probabilístico") AND ("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) AND (Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS)
```

No OpenAlex, a cadeia ampliada com `search` excede o limite de operadores booleanos da API (HTTP 429); a consulta O4 usa apenas os termos novos:

```
("data linkage" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros") AND ("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) AND (Brasil OR Brazil OR Brazilian)
```

## Consultas e contagens

| Id | Base | Consulta e filtro | Recuperados |
|---|---|---|---|
| P1 | PubMed | cadeia principal, sem rótulos de campo (mapeamento automático), `datetype=pdat`, 2010 a 2026 | 73 |
| P2 | PubMed | cadeia principal com `[tiab]` | 0 |
| P3 | PubMed | linkage e aprendizado supervisionado em `[tiab]` AND `Brazil[ad]` OR `Brasil[ad]` | 0 |
| P4 | PubMed | mesmo bloco sem restrição de país; nenhum registro com Brazil ou Brasil nas afiliações | 94 |
| P5 | PubMed | cadeia ampliada, sem rótulos de campo | 191 (118 além de P1) |
| P6 | PubMed | cadeia ampliada com `[tiab]` | 0 |
| S1 | SciELO | cadeia principal, sem filtro de ano | 0 |
| S2 | SciELO | `"record linkage"`, sem filtro de ano | 131 (78 títulos únicos; 56 de 2010 em diante) |
| S3 | SciELO | `"relacionamento de registros"` | 2 (1 artigo, 2004) |
| S4 | SciELO | `"pareamento probabilístico"` | 6 (3 títulos) |
| S5 | SciELO | `"record linkage"` AND aprendizado de máquina ou supervisionado | 0 |
| S6 | SciELO | termos em português de linkage AND aprendizado de máquina | 0 |
| SV | SciELO | cadeia ampliada | não executada (navegador indisponível; HTTP 403 por requisição) |
| B1 | BVS/LILACS | cadeia principal | 3 (1 no período) |
| B2 | BVS | `"record linkage"` AND aprendizado de máquina, 2010 a 2026, coleção completa | 71 (1 em LILACS Plus); não triada individualmente |
| B3 | BVS, somente LILACS | linkage em português AND aprendizado de máquina | 1 |
| BV | BVS/LILACS | cadeia ampliada | não executada (idem SV) |
| O1 | OpenAlex | cadeia principal, `title_and_abstract.search` | 7 |
| O2 | OpenAlex | `("record linkage" OR "data linkage") AND ("machine learning" OR supervised OR classifier) AND (Brazil OR Brazilian OR SINAN OR DATASUS OR SUS)`, título e resumo | 13 |
| O3 | OpenAlex | cadeia principal, parâmetro `search` | 2.115 (11 páginas) |
| O4 | OpenAlex | termos novos, parâmetro `search` | 1.153 (6 páginas) |
| O5 | OpenAlex | cadeia ampliada, `title_and_abstract.search` | 11 |
| C1 | Consensus | pedido com o critério de inclusão (thread https://consensus.app/search/revisao-de-escopo-para-tese-de-doutorado-em-saude-/SMYKyRkTRUK3U4ZbqqsQRQ/) | 15 estudos listados |
| - | Scopus | não executada: exige acesso institucional; repetir antes do depósito | n.a. |
| - | Google Scholar | não usado como base (não reprodutível) | n.a. |

A contagem de O1 difere da primeira rodada (10 registros) porque a cadeia foi refeita com o texto exato e coletada por script; O2 manteve 13. Buscas exploratórias de 2026-11-01 (PubMed 6, Scholar 4, Consensus 9, SciELO 2) não têm cadeias preservadas e não entram nas contagens.

## Fluxo

| Etapa | n |
|---|---|
| PubMed, P1 | 73 recuperados; 0 selecionados (12 interoperabilidade ou registro eletrônico; 6 revisão ou editorial; 25 método de linkage sem classificador, incluindo Duarte et al., 2019; 29 aplicação epidemiológica; 1 segmentação de nomes) |
| PubMed, P5 | 191 recuperados; 0 selecionados; 1 relacionado (Pita et al., 2018); 115 sem termo de linkage no título |
| SciELO, S2 | 131 registros, 56 títulos únicos de 2010 em diante; 0 selecionados |
| SciELO, S3 e S4 | 2 e 6 registros; 0 selecionados |
| BVS/LILACS | B1 3 (1 no período); B3 1; 0 selecionados; B2 não triada |
| OpenAlex, 5 consultas | 3.060 registros únicos; 3 selecionados (os mesmos em todas as consultas) e 1 duplicado (preprint de Lages et al.) |
| OpenAlex, O3 (2.115) | 1.894 excluídos por título sem termo de linkage; 196 pelo filtro de palavras-chave sobre o resumo (186 sem Brasil, 10 sem aprendizado supervisionado); 21 por leitura do título (resumo indisponível); 3 incluídos; 1 duplicado |
| Consensus | 15 estudos: 1 incluído estrito (Lages et al., 2026), 5 relacionados brasileiros, 9 relacionados internacionais |
| Item conhecido (TCC Paiva, 2025) | 1, não recuperado por nenhuma cadeia |
| Incluídos após deduplicação | 4 |

Triagem do OpenAlex em três passos: (1) título sem termo de linkage (`linkage`, `record matching`, `entity resolution`, `deduplic*`, `relacionamento`, `pareamento`, `vincula*` e variantes) é excluído; (2) para os demais, o resumo (obtido da API, não gravado) é filtrado por palavras-chave de Brasil e de aprendizado supervisionado, e só os indicadores booleanos ficam no JSON; (3) registros sem resumo disponível são decididos pelo título. Os registros que passaram pelos dois filtros são os três incluídos. O filtro pode excluir estudos cujo resumo não cite o país ou o método.

## Incluídos

| Estudo | Ano | DOI | Recuperação | Método | Base de dados |
|---|---|---|---|---|---|
| Pita et al. | 2017 | 10.1007/978-3-319-64283-3_16 | OpenAlex | árvore de decisão, naïve Bayes, regressão logística, Floresta Aleatória, SVM linear, gradient boosting | Brasil; base não detalhada no resumo (incluído com ressalva) |
| Pereira, Meira Junior e Guerra Junior | 2024 | 10.23889/ijpds.v9i5.2902 | OpenAlex | Floresta Aleatória, LightGBM, XGBoost, redes neurais | SUS |
| Paiva (TCC) | 2025 | sem DOI (http://hdl.handle.net/11422/25950) | item conhecido | árvore de decisão | SIM e Sinan-TB |
| Lages et al. | 2026 | 10.1038/s41598-026-73928-3 | OpenAlex, Consensus | seis algoritmos; LightGBM com melhor desempenho | SIM, SIH-SUS, Sinan, e-SUS VS (Espírito Santo) |

O Consensus classificou apenas Lages et al. (2026) como estudo estrito; Pereira (2024) e Pita (2017) aparecem entre os relacionados. A triagem pelos resumos mantém os dois como incluídos, Pita (2017) com ressalva.

## Relacionados e excluídos próximos

- Brasil: Pita et al. (2018), 10.1109/JBHI.2018.2796941; Duarte et al. (2019), 10.1590/0102-311X00066419; Coeli et al. (2021), 10.1186/s12911-021-01550-6; Morsoleto et al. (2025), 10.5753/kdmile.2025.247777.
- Excluídos com motivo: Brustulin e Marson (2018), 10.1590/0102-311X00088117 (pós-processamento determinístico); Girianelli et al. (2019), 10.1590/1980-549720190045 (concordância entre revisores humanos); Gonçalves et al. (2014), 10.1590/0102-311X00191313 (segmentação de nomes); Lorenzer et al. (2025), 10.3233/SHTI250175 (harmonização de registros eletrônicos).
- Internacionais (Consensus e bib existente): Quezada-Sánchez et al. (2022), 10.1016/j.heliyon.2022.e12311; McNutt et al. (2020), sem DOI; Shahriar (2020), 10.23889/ijpds.v5i5.1571; Powers et al. (2025), 10.1093/jamia/ocaf060; Hand e Christen (2017/2018), 10.1007/s11222-017-9746-6 (chave existente `Hand2018fmeasure`); Christen et al. (2025), 10.23889/ijpds.v10i3.3038; Jiao et al. (2021); Foxcroft et al. (2024); Röchner e Rothlauf (2024).

DOIs conferidos no Crossref em 2026-11-03.

## Limitações

Scopus e Embase não consultados (Scopus a repetir antes do depósito); Google Scholar não usado como base; triagem por um único revisor, sem leitura na íntegra; cadeia ampliada não executada na SciELO e na BVS; B2 (71 registros) não triada individualmente; filtro de palavras-chave sobre resumos no OpenAlex; Consensus é um agente não determinístico, e seus números não foram usados sem conferência; cadeias restritas a título e resumo retornaram zero no PubMed e nenhum estudo incluído foi recuperado por essa base.
