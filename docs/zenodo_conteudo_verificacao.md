# Conferência do depósito Zenodo em 4 de novembro de 2026

## Fontes e método

Consulta pública, sem autenticação e somente por GET. Nenhum arquivo ou metadado do Zenodo foi alterado. Script reproduzível: `scripts/verificar_zenodo_conteudo.py`. O extrato local é lido por caminho absoluto, com `dtype=str`, sem cópia; somente cardinalidades e comprimentos de identificadores são exportados. Os ZIPs ausentes da pasta local são processados em memória a partir do registro público. O ZIP GZ-CMD publicado também foi conferido diretamente, pois difere da cópia local.

Artefatos versionáveis: `data/zenodo_conteudo_resumo.csv` e `data/zenodo_conteudo_resumo_meta.json`. O CSV contém nomes, tamanhos descompactados e cabeçalhos de todos os CSVs dos quatro ZIPs, com contagens de linhas e de identificadores distintos apenas nos arquivos solicitados. Campos de contagem vazios significam não medidos, não zero. A expressão regular pedida é uma triagem de nomes de colunas, não prova de presença de dados pessoais: ela também seleciona subescores como `DTNASC dt iguais` e `CODMUNRES local igual`. JSON, JSONL, texto livre e matrizes NPY não foram submetidos a uma auditoria exaustiva de valores nesta conferência.

O endereço de versões do DOI conceito retornou HTTP 404. O endereço `https://zenodo.org/api/records/23948459/versions` retornou as quatro versões publicadas, sem necessidade de alterar o registro. Para versões anteriores, somente o cabeçalho dos CSVs foi lido por requisição parcial; não se afirma identidade de todos os seus bytes.

## Versões e arquivos

| Versão | Registro | Acesso | Arquivos |
| :--- | :--- | :--- | ---: |
| 1.0.0 | 18652289 | open | 7 |
| 2.0.0 | 18674440 | open | 10 |
| 3.0.0 | 23928226 | open | 11 |
| 3.0.1 | 23948459 | open | 11 |

A versão 1.0.0 contém o mesmo cabeçalho de 31 colunas do benchmark atual e o mesmo tamanho declarado, 7.969.179 bytes. Seus outros arquivos são `baselines.json` (8.527 bytes), `manifest.json` (288), `splits.csv` (772.446), `README.md` (3.006), `zenodo_metadata.json` (1.070) e `k_anonymity_report.json` (4.877). A lista completa de nomes e tamanhos de todas as versões consta dos metadados agregados.

Na versão 3.0.1:

| Arquivo | Bytes |
| :--- | ---: |
| benchmark_sim_sinan_tb.csv | 7.969.179 |
| splits.csv | 772.446 |
| baselines.json | 97.309 |
| manifest.json | 357 |
| zenodo_metadata.json | 1.340 |
| README.md | 5.194 |
| k_anonymity_report.json | 4.877 |
| qw2.zip | 791.273 |
| qw3.zip | 257.322 |
| resultados.zip | 10.139.396 |
| v2_gzcmd_v31.zip | 16.956.846 |

O README público não é idêntico ao local, de 5.595 bytes: declara MIT em lugar de CC-BY-NC-4.0 e substitui o bloco final de histórico por uma nota da versão 3.0.1. SHA-256 público: `80d3f1520cf56e3e2ec97ac9addd7817a5da0a81084f739ca6418a94caa04768`; local: `1218f0aef50f2e78c90418d11c68eeedddbfca58104f652a9afe57d1be54f813`. O diff documental consta dos metadados.

O ZIP GZ-CMD publicado tem 150 membros, enquanto a cópia local corresponde ao arquivo da versão 3.0.0, de 16.955.928 bytes. A comparação de SHA-256 por membro encontrou somente a adição de `LICENSE` e mudanças em `README.md`, `manifest.json` e `zenodo_metadata.json`, sem mudança nos membros de dados experimentais. O relatório de privacidade embutido registra 144 arquivos examinados; esse número não significa cobertura dos quatro ZIPs nem de todo o depósito.

## Conteúdo por potencial par

O benchmark contém 61.696 linhas e 31 colunas: PAR, PASSO, 28 subescores e nota final. PAR: 61.449 linhas com valor 0, 211 com valor 1 e 36 com valor 2. Não há campos brutos de datas, sexo ou município nesse CSV; há subescores com nomes referentes a esses atributos.

O arquivo `qw3/qw3_analysis_frame.csv` está efetivamente no ZIP e contém as colunas `index,y_true,y_pred,nome_qtd_quartis,bandeira,faixa_idade,sexo,uf,score_band,score_source,score_band_source`, com 61.696 linhas. Cardinalidades incluindo ausentes: faixa etária 6, sexo 2, indicador de encerramento 2 e UF 2. A análise prévia confundia o código de UF com a quantidade de níveis; não foram encontrados 33 níveis de UF. Os índices e os dados por linha não são reproduzidos neste relatório.

| Agrupamento do frame QW3 | Células observadas | k mínimo | Células com k < 5 | Linhas nessas células |
| :--- | ---: | ---: | ---: | ---: |
| faixa_idade × sexo × bandeira × y_true | 41 | 1 | 6 | 13 |
| faixa_idade × sexo × bandeira | 23 | 6 | 0 | 0 |

Ausentes foram mantidos como categoria, sem criar combinações não observadas. A menor célula tem uma linha; seus atributos não são divulgados. As unidades contadas são potenciais pares, não pessoas distintas. O agrupamento não abrange todas as colunas do frame e não mede risco de reidentificação por informação auxiliar.

O relatório histórico de k-anonimidade mede PASSO e agregados generalizados de similaridade. O esquema selecionado tem três agregados discretizados em quantis, 43 grupos e k mínimo 5. A etapa inicial já discretiza os 29 subescores em quartis, com 9.245 grupos, k mínimo 1 e 7.154 grupos abaixo de 5. Portanto, 7.154 não é a contagem dos vetores numéricos brutos publicados. A conferência direta de PASSO, 28 subescores e nota do benchmark encontrou 24.735 grupos, k mínimo 1 e 23.358 grupos abaixo de 5, reunindo 30.141 linhas. Essa distinção corrige a interpretação preliminar do relatório; similaridades brutas não são nomes ou datas brutas.

## Identificadores e limites da pseudonimização

`funil_gzcmd_vigilancia.csv` e `pcal_ml_oof.csv` possuem, cada um, 61.696 valores distintos de `pair_id`. As cardinalidades dos demais CSVs que contêm essa coluna estão no inventário.

| Campo local | Valores distintos | Comprimento mínimo e máximo em dígitos |
| :--- | ---: | :--- |
| COMPREC | 19.388 | 1 a 5 |
| REFREC | 1.515 | 1 a 4 |
| PASSO | 3 | 1 a 2 |

Não há valores não numéricos nesses campos. O produto das cardinalidades observadas é 88.118.460 combinações, mas pressupõe conhecer os conjuntos de valores observados. O produto dos espaços de inteiros positivos permitidos pelos comprimentos, sem zeros iniciais, é 98.989.110.099; permitindo zero e preenchimento à esquerda até o comprimento máximo, o limite é 100.000.000.000. Não são estimativas de tempo de ataque, nem o número de pares realmente existentes.

O SHA-256 truncado sem segredo pode ser testado por enumeração desse domínio, independentemente do espaço de saída de 64 bits. Esta conferência não executou força bruta, não exportou a correspondência e não identificou pessoas. A recuperação de um identificador interno ainda exige informação adicional para associação nominal. Não há evidência suficiente para inserir na tese a afirmação categórica de ausência de meios razoáveis de reidentificação por terceiros.

## Enquadramento e documentação pendente

Foram conferidas as fontes oficiais da LGPD e das Resoluções CNS 466/2012 e 580/2018. O texto distingue o fundamento para estudos por órgão de pesquisa da comprovação das autorizações de divulgação. O art. 13, § 2º, da LGPD também veda a transferência a terceiros dos dados pessoais abrangidos pelo dispositivo; uma hipótese do art. 33 não afasta automaticamente essa vedação. Não se declara conformidade comprovada antes da anexação do parecer, da decisão de dispensa de TCLE e do instrumento de cessão com escopo de compartilhamento e processamento externo. O depósito foi mantido aberto, conforme decisão do autor.

## Verificação local

- `python scripts/check_bib_usage.py`: 132 entradas, 132 chaves citadas, 0 órfãs e 0 citações sem entrada.
- `python scripts/build_pdf.py --jobname tese_leg --outname tese_leg`: PDF de 1.017 KB, estabilizado em quatro passadas. Os PDFs alternativos não integram o commit.
- Busca no log por citações e referências indefinidas ou erros LaTeX: sem ocorrências.
- `python -m pytest tests -q`: 287 aprovados, 1 ignorado preexistente, 5 falhas e 2 avisos em 22,65 s. Os três testes novos passaram nessa execução.
- As cinco falhas foram reproduzidas individualmente na base limpa `3e9275e`, extraída dentro de `.local/zenodo_base_3e9275e`: 5 falhas em 2,88 s. Uma falha decorre do módulo local ausente `scripts.preservar_m5_m7_row`, três do extrato ignorado `data/COMPARADORSEMIDENT.csv` e uma de `data/funil_gzcmd_final_row.csv`. Não foram copiados esses arquivos nem alterados ou pulados testes para contornar as falhas.
