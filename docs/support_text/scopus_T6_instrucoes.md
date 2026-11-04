# Busca Scopus do protocolo T6

As consultas abaixo estão preparadas, mas não foram executadas. O acesso institucional permanece pendente. Não se atribuem contagens, datas de execução ou decisões de inclusão a esta etapa antes da consulta pelo autor. Permanecem os quatro estudos incluídos no levantamento de 3 de novembro de 2026: Lages (2026), Pereira (2024), Pita (2017, com ressalva) e Paiva (2025).

## Consultas exatas na busca avançada

Aplicar cada expressão integralmente em **Advanced document search**. Todas delimitam o ano de publicação de 2010 a 2026, inclusive. O protocolo registrado não restringe o idioma; por isso não se acrescenta filtro `LANGUAGE`, nem restrição de tipo documental ou área temática.

### Sc1: cadeia principal, correspondente a P1

```text
TITLE-ABS-KEY(("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") AND ("machine learning" OR "aprendizado de máquina" OR supervised) AND (Brasil OR Brazil OR SIM OR SINAN OR DATASUS)) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

### Sc2: cadeia ampliada, correspondente a P5

```text
TITLE-ABS-KEY(("record linkage" OR "data linkage" OR "relacionamento de registros" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros" OR "pareamento probabilístico") AND ("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) AND (Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS)) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

### Sc3: variante P2, título e resumo

```text
(TITLE("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico") OR ABS("record linkage" OR "relacionamento de registros" OR "pareamento probabilístico")) AND (TITLE("machine learning" OR "aprendizado de máquina" OR supervised) OR ABS("machine learning" OR "aprendizado de máquina" OR supervised)) AND (TITLE(Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS) OR ABS(Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS)) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

### Sc4: variante P3, afiliação no Brasil

```text
(TITLE("record linkage" OR "data linkage") OR ABS("record linkage" OR "data linkage")) AND (TITLE("machine learning" OR supervised) OR ABS("machine learning" OR supervised)) AND AFFILCOUNTRY(Brazil) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

### Sc5: variante P4, referência internacional

```text
(TITLE("record linkage" OR "data linkage") OR ABS("record linkage" OR "data linkage")) AND (TITLE("machine learning" OR supervised) OR ABS("machine learning" OR supervised)) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

### Sc6: variante P6, cadeia ampliada em título e resumo

```text
(TITLE("record linkage" OR "data linkage" OR "relacionamento de registros" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros" OR "pareamento probabilístico") OR ABS("record linkage" OR "data linkage" OR "relacionamento de registros" OR "relacionamento probabilístico" OR "linkage probabilístico" OR "pareamento de registros" OR "pareamento probabilístico")) AND (TITLE("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado) OR ABS("machine learning" OR "aprendizado de máquina" OR supervised OR supervisionado)) AND (TITLE(Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS) OR ABS(Brasil OR Brazil OR Brazilian OR SIM OR SINAN OR DATASUS)) AND PUBYEAR > 2009 AND PUBYEAR < 2027
```

## Equivalência e diferenças entre os índices

As expressões-fonte são as linhas P1 a P6, com `etapa=consulta`, de `data/protocolo_busca_T6.csv`. A adaptação preserva os termos e os operadores booleanos, não o mapeamento automático do PubMed. A equivalência é conceitual e não implica recuperar os mesmos documentos ou contagens.

| Elemento no PubMed | Adaptação para Scopus | Justificativa |
| :-- | :-- | :-- |
| P1: `"record linkage"`, `"relacionamento de registros"`, `"pareamento probabilístico"` | Mesmos três termos, ligados por `OR`, em Sc1 e Sc3 | Preserva integralmente o bloco do domínio |
| P1: `"machine learning"`, `"aprendizado de máquina"`, `supervised` | Mesmos três termos, ligados por `OR`, em Sc1 e Sc3 | Não se substitui `supervised` por um vocabulário controlado |
| P1: `Brasil`, `Brazil`, `SIM`, `SINAN`, `DATASUS` | Mesmos cinco termos em Sc1 | As siglas não são expandidas nem desambiguadas, mantendo a possibilidade de ruído |
| P2: acréscimo de `Brazilian` ao bloco de país | Preservado em Sc3 | A variante P2 não é idêntica a P1 |
| P5 e P6: acréscimos `"data linkage"`, `"relacionamento probabilístico"`, `"linkage probabilístico"`, `"pareamento de registros"` | Preservados, além dos três termos principais, em Sc2 e Sc6 | Reproduz as sete variantes do domínio |
| P5 e P6: acréscimos `supervisionado` e `Brazilian` | Preservados em Sc2 e Sc6 | Reproduz os quatro termos de método e os seis de contexto |
| P3 e P4: `"record linkage" OR "data linkage"`, combinados com `"machine learning" OR supervised` | Mesmos dois blocos em Sc4 e Sc5 | Não se acrescentam os termos das cadeias mais amplas |
| P1 e P5 sem rótulos de campo | `TITLE-ABS-KEY(...)` em Sc1 e Sc2 | Scopus pesquisa título, resumo e palavras-chave; não reproduz o mapeamento automático para MeSH e outros campos do PubMed |
| P2, P3, P4 e P6 com `[tiab]` | `(TITLE(bloco) OR ABS(bloco))` para cada bloco em Sc3 a Sc6 | Preserva a restrição a título e resumo, sem incluir palavras-chave; cada bloco pode ocorrer em um desses campos |
| P3: `Brazil[ad] OR Brasil[ad]` | `AFFILCOUNTRY(Brazil)` em Sc4 | Usa o país normalizado da afiliação; difere da busca textual em endereço e não equivale ao país da base estudada |
| P4 sem bloco de país | Sc5 sem restrição geográfica | Consulta internacional de referência, não inclusão automática no conjunto brasileiro |
| `datetype=pdat; mindate=2010; maxdate=2026` | `PUBYEAR > 2009 AND PUBYEAR < 2027` em todas | Restringe o ano de publicação, não a data de indexação |
| `retmax=1000` | Sem correspondente na expressão | É parâmetro de coleta da API, não critério de elegibilidade; exportar todos os resultados, em lotes se necessário |
| Ausência de filtro de idioma | Sem `LANGUAGE(...)` | Não introduz restrição ausente do protocolo |

As aspas duplas foram mantidas como busca de frases na sintaxe Scopus. Não se empregam chaves de frase exata, truncamento ou operadores de proximidade, pois mudariam a estratégia planejada. As regras de normalização, indexação e busca de frases diferem entre as bases. A busca nas palavras-chave em Sc1 e Sc2 é uma ampliação em relação às variantes restritas, não uma reprodução literal de P1 e P5.

## Acesso institucional e exportação

1. Acessar o Portal de Periódicos CAPES em <https://www.periodicos.capes.gov.br/>. Selecionar o acesso pela Comunidade Acadêmica Federada (CAFe), localizar a Universidade Federal do Rio de Janeiro (UFRJ) e autenticar-se com a credencial institucional. Alternativamente, utilizar a rede institucional ou o acesso remoto autorizado pela UFRJ. Não registrar credenciais nos arquivos do protocolo.
2. Localizar **Scopus** na busca de bases do portal e abrir a plataforma Elsevier. Confirmar que o acesso institucional permite a pesquisa de documentos. A disponibilidade contratual deve ser conferida no momento da execução; a autenticação na CAFe não garante, por si só, acesso à base. Se o acesso não estiver disponível, manter as linhas pendentes, sem data nem contagem, e registrar a ocorrência na nota.
3. Abrir a busca avançada de documentos e executar Sc1 a Sc6 separadamente, sem filtros adicionais. Preservar a expressão efetivamente aceita pelo portal, o histórico da busca, a data local da execução e a contagem exibida antes da deduplicação. Se houver recusa de sintaxe, registrar o erro e qualquer correção antes de considerar a consulta executada.
4. Em cada lista de resultados, selecionar todos os registros e exportar em **CSV**, incluindo **título, autores, ano, DOI, título da fonte e resumo**. Incluir também o identificador Scopus EID, quando disponível, para rastreabilidade. Na interface, esses campos podem estar distribuídos entre informações de citação, informações bibliográficas e resumo/palavras-chave. Conferir as colunas do arquivo exportado, pois exportações padrão podem omitir resumos.
5. Quando houver limite de exportação, dividir em lotes e registrar os intervalos, sem truncar a coleta. Guardar os arquivos brutos separadamente por consulta e lote, por exemplo `data/scopus_T6_Sc1_lote01.csv`, sem substituir o template por um CSV de formato Scopus. Conferir o total exportado com o total exibido, incluindo eventuais registros sem resumo ou DOI.

Referências operacionais: Portal CAPES, acima, e ajuda da Scopus, acessível na própria plataforma em <https://www.scopus.com/>. Os nomes e a localização das opções devem ser conferidos pelo autor no acesso institucional. Não houve validação das consultas no portal nesta preparação.

## Preenchimento do registro e atualização do apêndice

O template `data/protocolo_busca_T6_scopus_template.csv` usa exatamente as 15 colunas do registro T6, inclusive seus nomes: `base`, `consulta`, `etapa`, `data_busca`, `string_exata`, `filtros`, `n_recuperado`, `registro_id`, `primeiro_autor`, `ano`, `titulo`, `periodico`, `doi`, `decisao`, `motivo_ou_nota`. No esquema existente, **`data_busca` é a data de execução e `n_recuperado` é o número de resultados**; não criar colunas `data_execucao` ou `n_resultados`. Ambos os campos permanecem vazios nas seis consultas planejadas. Campo vazio significa consulta não executada, nunca zero resultados.

Após o acesso, trabalhar em uma cópia preenchida, por exemplo `data/protocolo_busca_T6_scopus_executado.csv`, conservando o template sem resultados. Em cada linha `etapa=consulta`, preencher `data_busca` em `AAAA-MM-DD`, `n_recuperado` com a contagem efetiva e a nota com os arquivos exportados, lotes e eventuais diferenças da expressão planejada. Manter vazios os campos bibliográficos dessas linhas. Para cada documento triado, acrescentar linha `etapa=triagem` e a mesma identificação de consulta; preencher o EID em `registro_id`, primeiro autor, ano, título, fonte em `periodico`, DOI, decisão e justificativa. O resumo e a lista completa de autores permanecem no CSV bruto exportado, pois não há essas colunas no esquema T6. Não reescrever o levantamento de 3 de novembro com uma data posterior.

Deduplicar por DOI normalizado e, na ausência dele, por título, ano e autoria, com conferência manual. Registrar as consultas que recuperaram o mesmo documento sem somar duplicatas como estudos distintos. Aplicar os critérios do apêndice, distinguindo classificação supervisionada de pares de predição clínica sobre dados já vinculados; Sc5 fornece apenas referência internacional. Confrontar os registros com os quatro incluídos e com as demais bases. Somente após exportação e triagem, atualizar no apêndice as linhas Scopus, o fluxo deduplicado e as limitações, informando a nova data e a origem de cada alteração. Se nenhum estudo adicional atender ao critério, registrar esse resultado apenas depois da execução; até lá, manter a ressalva de cobertura incompleta.
