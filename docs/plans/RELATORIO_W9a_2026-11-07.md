# Relatório W9a

## Lista de execução

- [x] Ler guia de estilo, relatórios e CSVs de origem.
- [x] Integrar sensibilidades exploratórias, comparações prospectivas e glosas.
- [x] Acrescentar as nove referências nas duas bibliografias, sem inventar metadados.
- [x] Criar testes de rastreabilidade e executar testes direcionados.
- [x] Registrar verificações e pendências; commits por grupo com seleção explícita de arquivos.

## Delimitação

Não alterar hipótese, margem ou critério primário. Não executar LLM, análise nova, compilação da tese ou push. Arquivos alheios à W9a permanecem sob responsabilidade da W9b.

## Alterações

- `text/latex/estudo4_gzcmd.tex`: dois parágrafos de sensibilidade exploratória post hoc e tabela com todas as seis políticas de fusão, sem seleção da melhor; metas de 9, 23 e 36 PAR=2 e quantis adaptativos conferidos contra os CSVs. Acrescentadas as comparações secundárias para registro anterior à avaliação da coorte nova, sem alegar registro já realizado. A inferência com predições corrige estimativas baseadas nos labels do LLM, não cada decisão individual. Incluídos Ather, circularidade entre juiz e sistema e compartilhamento dos subescores entre O3 e LLM. Glosas locais em notas na primeira menção por seção e em legendas, incluindo união P/B1 e orçamento misto K2.
- `text/latex/apendice_algoritmo.tex`: glosas de versões, ações, CdR, EVR, MACD e instruções v4, sem modificar ações ou guardas.
- `text/latex/apendice_ablacao.tex`: glosas iniciais e quatro legendas autossuficientes, distinguindo modos e atributos de nascimento; números intactos.
- `text/latex/apendice_formalizacao.tex`: glosas locais de CdR, v3.1, LLM e quantidade selecionada k, sem alterar proposições.
- `text/latex/referencias.bib` e `text/latex/referencias_abnt.bib`: apenas nove entradas novas, com as chaves solicitadas.
- `tests/test_w9a.py`: rastreabilidade dos números por variante e métrica, seis filas e respectivos quantis, unicidade das nove chaves e DOIs, citações, preservação do critério e glosas do censo.
- Exceção mínima ao conjunto de testes novos, necessária à exigência de ajustar asserções obsoletas: `tests/test_review2_w3.py` e `tests/test_review2_w4_texto.py` agora comparam o apêndice ao gerador acrescido somente das glosas exatas autorizadas. A igualdade integral continua obrigatória; não se excluem legendas nem números da comparação. O gerador, fora do escopo, não foi alterado.

## Fontes e decisões numéricas

Leituras anteriores à redação: guia de estilo completo, incluindo anti-LLM, relatórios O3/MACD e fusão e os dois CSVs de origem. Não se executou nova análise nem treinamento.

- `data/sensibilidade_o3_macd.csv`: revisões para as três metas e AP/AUC abaixo do corte. Arredondamento de AP/AUC a três casas.
- `data/fusao_filas.csv`: tabela das seis variantes, com médias e P10/P90 das adaptativas, arredondadas a duas casas pelo critério decimal de metade para cima. A média de A1 para 36 vínculos é 1829.475 no CSV, apresentada como 1.829,48, e não o 1.829,47 do relatório anterior.
- A alegação genérica de superioridade das fusões sobre O1 e O3 nas três metas não vale para F1: requer 49 revisões para nove vínculos, contra 29 de O3. O texto delimita a afirmação a F2, F3 e F4 e explicita essa exceção. A superioridade média das adaptativas refere-se a O1 e O3, não a todas as fusões fixas.
- O parágrafo original do critério prospectivo de falsificação foi comparado ao conteúdo de `7ec7088` e permanece integralmente como prefixo do atual; apenas a frase secundária e uma glosa foram acrescentadas ao fim. Hipótese, margem e critério primário preservados.

## Verificação bibliográfica

Consultas desta fatia aos DOIs por negociação de conteúdo `application/vnd.citationstyles.csl+json`; Lewis e Lam também retornaram metadados pela API Crossref. A primeira tentativa em lote na API Crossref recebeu HTTP 429 para sete entradas; a resolução pelo DOI confirmou posteriormente as nove. Campos adicionais provêm dessas respostas, não de inferência a partir de chaves ou títulos.

| Chave | DOI confirmado | Observação |
|---|---|---|
| Ather2026llmlinkage | 10.1177/18747655261422068 | Hanan Ather, Statistics Canada; volume 42, número 1, páginas 211 a 217 |
| Rahman2019bandits | 10.1145/3308558.3313675 | Três autores completos; páginas 3158 a 3164 |
| Fisch2024stratppi | 10.48550/arxiv.2406.04291 | Seis autores completos; arXiv, nota preprint |
| Alaofi2026llmrelevance | 10.1145/3788872 | Quatro autores completos; volume 44, número 4, páginas 1 a 31 |
| Gienapp2025relevance | 10.1145/3805712.3809713 | Cinco autores completos; publicação em 2026 segundo o DOI; chave 2025 preservada para compatibilidade com W9b |
| Callaghan2020stopping | 10.1186/s13643-020-01521-4 | Max W Callaghan e Finn Müller-Hansen; volume 9, número 1 |
| Lewis2021certifying | 10.1145/3459637.3482415 | David D. Lewis, Eugene Yang e Ophir Frieder; páginas 893 a 902 |
| Lam2026clerical | 10.23889/ijpds.v11i5.3595 | Nove autores completos; volume 11, número 5 |
| DiNunzio2026stopping | 10.1145/3774905.3795598 | Autoria confirmada como Giorgio Maria Di Nunzio, não expansão presumida de G. D.; páginas 333 a 340 |

Pendências de metadados: páginas ou número de artigo de Callaghan e Lam não constaram das respostas consultadas e foram omitidos; não se inventaram valores. Não restaram autores incompletos nem necessidade de `and others`, embora esse formato exista nas bibliografias. A chave de Gienapp conserva 2025, mas o campo `year` usa 2026, confirmado na publicação SIGIR do DOI.

## Verificação executada

- Python: `C:\Users\Marquinho\miniconda3\python.exe`.
- Seleção direcionada: arquivos retornados por `rg -l "estudo4_gzcmd|apendice_algoritmo|apendice_ablacao|apendice_formalizacao|referencias" tests`, acrescidos de `tests/test_w9a.py`, sem executar a suíte inteira.
- Execução final: `python -m pytest -q <arquivos selecionados>`; PID 11056, aguardado por `Wait-Process -Id`, criado com janela oculta. Resultado: **179 passed, 1 skipped in 26.83s**, saída 0. Logs em `C:\Users\Marquinho\AppData\Local\Temp\opencode\w9a_final.out` e `.err`.
- Execução focal anterior dos três módulos alterados: **31 passed in 2.85s**, PID 24020. A igualdade integral do apêndice com as glosas autorizadas foi mantida.
- Na primeira execução ampla houve cinco falhas em asserções sobre arquivos da W9b (equidade, conclusão e Estudo 1); esses arquivos não foram alterados pela W9a. A execução final já passou essas verificações.
- `git diff --check` nos arquivos alterados da fatia: sem erros; somente avisos de normalização CRLF/LF.
- Conferência dos quatro arquivos LaTeX: sem travessões Unicode. Não houve compilação da tese, chamadas a LLM ou push.

## Entrega

Commits separados com prefixos `text:`, `bib:` e `tests:`, sem trailer de IA, sem incluir arquivos da W9b. Este relatório é incluído explicitamente apesar de `docs/plans` ser ignorado no repositório. A inspeção visual do PDF permanece não executada, conforme a proibição de build.
