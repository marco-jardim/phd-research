# Protocolo de busca bibliográfica (T6)

Data das buscas: 2026-11-03. Período: 2010 a 2026. Triagem por um único revisor, por título e resumo (resumo lido nos registros mais próximos do critério); sem leitura na íntegra.

Critério de inclusão: classificador supervisionado aplicado à classificação de pares candidatos em bases brasileiras de saúde, treinado em pares rotulados.

Artefatos: `data/protocolo_busca_T6.csv` (metadados bibliográficos, sem resumos e sem dados pessoais), `scripts/protocolo_busca_T6.py` (consultas ao PubMed), `scripts/protocolo_busca_T6_tabelas.py` (geração do CSV). Apêndice: `text/latex/apendice_busca.tex`.

## Cadeia principal (dispatch)

```
("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") AND ("machine learning" OR "aprendizado de máquina" OR supervised) AND (Brasil OR Brazil OR SIM OR SINAN OR DATASUS)
```

## Consultas e contagens

| Id | Base | Consulta e filtro | Recuperados |
|---|---|---|---|
| P1 | PubMed (E-utilities, `esearch`) | cadeia principal, sem rótulos de campo (mapeamento automático de termos), `datetype=pdat`, 2010 a 2026 | 73 |
| P2 | PubMed | cadeia principal com `[tiab]`, mesmo período | 0 |
| P3 | PubMed | `("record linkage"[tiab] OR "data linkage"[tiab]) AND ("machine learning"[tiab] OR supervised[tiab]) AND (Brazil[ad] OR Brasil[ad])` | 0 |
| P4 | PubMed | mesmo bloco sem restrição de país; nenhum dos 94 registros traz Brazil ou Brasil nas afiliações | 94 |
| S1 | SciELO (`search.scielo.org`, interface web) | cadeia principal, sem filtro de ano | 0 |
| S2 | SciELO | `"record linkage"`, sem filtro de ano | 131 (78 títulos únicos; 56 de 2010 em diante) |
| S3 | SciELO | `"relacionamento de registros"` | 2 |
| S4 | SciELO | `"pareamento probabilístico"` | 6 |
| S5 | SciELO | `"record linkage" AND ("machine learning" OR "aprendizado de máquina" OR supervised)` | 0 |
| S6 | SciELO | termos em português de linkage probabilístico AND aprendizado de máquina ou supervisionado | 0 |
| B1 | BVS/LILACS (`pesquisa.bvsalud.org/portal`, interface web) | cadeia principal | 3 (1 no período 2010 a 2026) |
| B2 | BVS | `"record linkage" AND ("machine learning" OR "aprendizado de máquina")`, filtro de ano | 71 na coleção completa (1 em LILACS Plus); não triada individualmente |
| B3 | BVS, somente LILACS | linkage em português AND aprendizado de máquina | 1 |
| O1 | OpenAlex (API, `title_and_abstract.search`) | cadeia principal, 2010 a 2026 | 10 |
| O2 | OpenAlex | `("record linkage" OR "data linkage") AND ("machine learning" OR supervised OR classifier) AND (Brazil OR Brazilian OR SINAN OR DATASUS OR SUS)` | 13 |
| - | Scopus | não executada: exige acesso institucional | n.a. |
| - | Google Scholar | não usado como base (não reprodutível) | n.a. |

Observações: a SciELO e a BVS foram consultadas por navegador, pois as páginas exigem execução de JavaScript e recusam requisições diretas (HTTP 403); as contagens dependem do estado do índice em 2026-11-03. Na SciELO, cada documento pode aparecer em mais de uma versão de idioma, o que explica 131 registros para 78 títulos únicos. O PubMed foi consultado por script. Buscas exploratórias de 2026-11-01 (PubMed 6, Scholar 4, Consensus 9, SciELO 2, segundo o plano da fase 3) não têm cadeias preservadas nesta sessão e não entram nas contagens.

## Fluxo

| Etapa | n |
|---|---|
| PubMed, cadeia principal (P1) | 73 recuperados; 0 selecionados |
| SciELO, "record linkage" (S2) | 131 registros, 56 títulos únicos de 2010 em diante; 0 selecionados |
| SciELO, demais consultas | 0 a 8 registros por consulta; 0 selecionados |
| BVS/LILACS, cadeia principal (B1) | 3 recuperados, 1 no período; 0 selecionados |
| OpenAlex (O1) | 10 recuperados; 3 selecionados; 1 duplicado (preprint de Lages et al.) |
| Item conhecido (dissertação Paiva, 2025) | 1, não recuperada por nenhuma cadeia |
| Incluídos após deduplicação | 4 |

Os 73 registros de P1 foram excluídos pelos motivos: interoperabilidade ou registro eletrônico (12), revisão, editorial ou estrutura conceitual (6), método ou avaliação de linkage sem classificador supervisionado (25, incluindo Duarte et al., 2019), segmentação de nomes (1) e aplicação epidemiológica que usa o linkage como etapa (29).

## Incluídos

| Estudo | Ano | DOI | Recuperação | Método | Base de dados |
|---|---|---|---|---|---|
| Pita et al. | 2017 | 10.1007/978-3-319-64283-3_16 | OpenAlex | árvore de decisão, naïve Bayes, regressão logística, Floresta Aleatória, SVM linear, gradient boosting | Brasil; base não informada no resumo |
| Pereira et al. | 2024 | 10.23889/ijpds.v9i5.2902 | OpenAlex | Floresta Aleatória, LightGBM, XGBoost, redes neurais | SUS |
| Paiva | 2025 | sem DOI (http://hdl.handle.net/11422/25950) | item conhecido | árvore de decisão | SIM e Sinan-TB |
| Lages et al. | 2026 | 10.1038/s41598-026-73928-3 | OpenAlex | seis algoritmos; LightGBM com melhor desempenho | SIM, SIH-SUS, Sinan, e-SUS VS (Espírito Santo) |

Relacionado, fora do critério estrito: Pita et al. (2018), 10.1109/JBHI.2018.2796941, ferramenta híbrida de linkage probabilístico (AtyImo); não recuperado por nenhuma cadeia.

DOIs de Pita 2017 e 2018, Pereira 2024, Lages 2026 e Duarte 2019 conferidos no Crossref em 2026-11-03.

## Excluídos próximos do critério (amostra)

- Duarte et al. (2019), 10.1590/0102-311X00066419: método de escore para identificar pares verdadeiros sem revisão manual (SISMAMA); sem classificador treinado em pares rotulados.
- Brustulin e Marson (2018), 10.1590/0102-311X00088117: pós-processamento determinístico do linkage probabilístico.
- Girianelli et al. (2019), 10.1590/1980-549720190045: concordância entre revisores humanos na classificação de pares.
- Gonçalves et al. (2014), 10.1590/0102-311x00191313: segmentação de nomes com modelo oculto de Markov.
- Lorenzer et al. (2025), 10.3233/SHTI250175: harmonização de registros eletrônicos para aprendizado federado.

A lista completa dos 73 registros de P1, com motivo de exclusão, e as demais amostras estão no CSV.

## Limitações

Scopus e Embase não consultados; Google Scholar não usado como base; triagem por um único revisor, sem leitura na íntegra; B2 (71 registros) não triada individualmente; cadeias restritas a título e resumo retornaram zero e nenhum estudo incluído foi recuperado pelo PubMed, de modo que a cobertura da cadeia principal é limitada e a repetição na Scopus segue pendente.
