# Correções de proveniência e viés de incorporação

## Decisão do autor e comparação factual

A análise acionou a regra de parada. Posteriormente, o autor decidiu recalibrar a política abaixo do corte em outra tarefa e autorizou concluir e mesclar estas correções factuais com build verde e estado CLEAN. Resumo e abstract foram preservados; a síntese do Estudo 4 recebeu apenas qualificações dos cenários hipotéticos e do denominador da sensibilidade. A conclusão recebeu a correção factual da ordem entre o CdR e a revisão clerical, sem alteração de seu argumento central.

Na área cinza da nota final do CdR há 151 vínculos, dos quais 123 com PAR igual a 1 e 28 com PAR igual a 2. O corte do OpenRecLink aceitou 123. A cascata aceitou 113 na vigilância e 65 na confirmação.

## Números novos e artefatos de origem

```csv
faixa,PAR,vinculos
<5,1,2
<5,2,1
"[5,9)",1,123
"[5,9)",2,28
>=9,1,86
>=9,2,7
```

Fonte: `data/area_cinza_por_par.csv`, reproduzido por `scripts/area_cinza_por_par.py`. O JSON de mesmo radical com sufixo `_meta` registra o SHA-256 do extrato.

```csv
modo,versao,faixa,PAR,vinculos,aceitos_automaticamente,sensibilidade_automatica,aceitos_corte_orl
vigilancia,v3.1,TODAS,1,211,184,0.8720379146919431,211
vigilancia,v3.1,TODAS,2,36,10,0.2777777777777778,0
vigilancia,v3.1,"[5,9)",1,123,105,0.8536585365853658,123
vigilancia,v3.1,"[5,9)",2,28,8,0.2857142857142857,0
confirmacao,v3.1,TODAS,1,211,124,0.5876777251184834,211
confirmacao,v3.1,TODAS,2,36,8,0.2222222222222222,0
confirmacao,v3.1,"[5,9)",1,123,61,0.4959349593495935,123
confirmacao,v3.1,"[5,9)",2,28,4,0.14285714285714285,0
```

Fonte: `data/sensibilidade_gzcmd_por_par.csv`, reproduzido por `scripts/sensibilidade_gzcmd_por_par.py`; hashes dos insumos e regra de parada em `data/sensibilidade_gzcmd_por_par_meta.json`. As matrizes conferem com `funil_gzcmd_final_metricas.csv`: 194/247 (78,5%) na vigilância e 132/247 (53,4%) na confirmação.

Os cortes 7,0240; 7,8444; 7,9019, as contagens 76/80, 120/127 e 15/40, os 29 passos, os 2.549 pares e os 5 pares do passo aberto constam de `data/proveniencia_oliveira_meta.json`, com páginas impressas e decisões de proveniência do autor separadas da documentação.

## Atualização sobre o PR B

A branch foi atualizada sobre c8a1c0a, merge do PR B. O conflito em capitulo_dados.tex foi resolvido mantendo a descrição de cobertura do SIM corrigida pelo PR B e a restrição do extrato a óbitos por TB. As limitações dos Estudos 2 e 3 agora explicitam o viés de incorporação. O nome foi corrigido para Laboratório de Linkage de Dados Populacionais (LinkDataPop), preservando o itálico onde existente. A busca incluiu tese, bibliografia, slides e roteiro; a grafia antiga estava em capitulo2.tex, capitulo3.tex e capitulo_dados.tex.

## Revisão das alegações do passo 1

Resumo e abstract não afirmam superioridade sobre o corte do OpenRecLink. A conclusão distingue a referência operacional das referências hipotéticas do CdR, sem alteração comparativa. A síntese do Estudo 4 associava a sensibilidade global ao recorte da área cinza; foram corrigidos somente o denominador e a qualificação dos cenários, conforme o registro a seguir.

```json
{
  "decisao_do_autor": "Recalibrar a politica abaixo do corte em outra tarefa; autoriza concluir e mesclar as correcoes factuais com CI verde e CLEAN.",
  "arquivo": "text/latex/estudo4_gzcmd.tex",
  "antes": "O GZ-CMD transforma a área cinza de um problema de volume em um problema de decisão. Na saída do Comparador de Registros posterior ao OpenRecLink, os 21.620 potenciais pares com nota final em $[5,9)$, que concentram 151 dos 247 vínculos, ou seguiriam integralmente ao revisor humano ou seriam descartados por um corte hipotético da nota em 9, que aceitaria 93 vínculos (37,7\\%) com 2 falsos positivos, de modo que a escolha se daria entre uma carga de revisão incompatível com a rotina do serviço e a perda de 62,3\\% dos vínculos (Tabela~\\ref{tab:cap8-referencia}). Ao comparar, para cada potencial par, o \\textit{expected loss} de vínculo, não-vínculo, revisão por LLM e revisão humana, a política v3.1 resolveu a maior parte dessa área sem revisão humana no modo vigilância, aceitando automaticamente 78,5\\% dos vínculos, ao custo de 34 falsos positivos automáticos, e encaminhando 111 potenciais pares ao profissional do serviço. No modo confirmação, em que o falso vínculo custa cinco vezes a perda de um vínculo, a mesma política concentrou o esforço humano em 127 potenciais pares de probabilidade intermediária, sem falso positivo automático, porém com sensibilidade automática de 53,4\\%, e deixou documentado por que não avança além desse ponto: a restrição de falsos vínculos esperados a 0,2\\% mostrou-se inviável com as probabilidades disponíveis, e a calibração absoluta e o erro humano permanecem a validar no serviço.",
  "depois": "O GZ-CMD transforma a área cinza de um problema de volume em um problema de decisão. Nos cenários hipotéticos de referência da saída do Comparador de Registros posterior ao OpenRecLink, os 21.620 potenciais pares com nota final em $[5,9)$, que concentram 151 dos 247 vínculos, ou seguiriam integralmente ao revisor humano ou seriam descartados por um corte hipotético da nota em 9, que aceitaria 93 vínculos (37,7\\%) com 2 falsos positivos, contrastando-se a revisão integral com a perda de 62,3\\% dos vínculos (Tabela~\\ref{tab:cap8-referencia}). Ao comparar, para cada potencial par, o \\textit{expected loss} de vínculo, não-vínculo, revisão por LLM e revisão humana, a política v3.1, aplicada à base inteira, aceitou automaticamente 194 dos 247 vínculos (78,5\\%) no modo vigilância, ao custo de 34 falsos positivos automáticos, e encaminhando 111 potenciais pares ao profissional do serviço. No modo confirmação, em que o falso vínculo custa cinco vezes a perda de um vínculo, a mesma política concentrou o esforço humano em 127 potenciais pares de probabilidade intermediária, sem falso positivo automático, porém com sensibilidade automática de 53,4\\%, e deixou documentado por que não avança além desse ponto: a restrição de falsos vínculos esperados a 0,2\\% mostrou-se inviável com as probabilidades disponíveis, e a calibração absoluta e o erro humano permanecem a validar no serviço.",
  "motivo": "Qualificar os cenarios como hipoteticos e nao atribuir a sensibilidade global de 78,5% ao recorte da area cinza.",
  "resumo_abstract": "Sem alegacao de superioridade sobre o corte do OpenRecLink; preservados.",
  "conclusao_comparacao": "Distingue a referencia operacional das referencias hipoteticas do CdR; sem alteracao comparativa.",
  "nome_laboratorio_arquivos_corrigidos": [
    "text/latex/capitulo2.tex",
    "text/latex/capitulo3.tex",
    "text/latex/capitulo_dados.tex"
  ],
  "bib_slides_roteiro": "Verificados; nenhuma ocorrencia da grafia antiga nesses arquivos."
}
```

## Verificações

### pytest

```text
FAILED tests/test_finalizar_pdf_refrec.py::test_row_archive_preserves_historical_table_bytes_outside_active_tex
FAILED tests/test_oof_grupos_pr7.py::test_legacy_v3_cohort_and_guards[vigilancia-1410]
FAILED tests/test_oof_grupos_pr7.py::test_legacy_v3_cohort_and_guards[confirmacao-435]
FAILED tests/test_oof_grupos_pr7.py::test_runner_honors_configured_split_by
FAILED tests/test_refrec_final.py::test_cascade_table_is_regenerated_from_cells_not_old_counts
5 failed, 313 passed, 3 skipped, 2 warnings in 27.63s
```

### tests_new

```text
...........s..                                                           [100%]
13 passed, 1 skipped in 2.91s
```

### bib

```text
Total .bib entries: 133
Total cited keys:   133

Orphan (in .bib but NOT cited in any .tex): 0

Phantom (cited in .tex but NOT in .bib): 0
```

### build

```text
[1/5] pdflatex (1a passada)...
  [2/5] bibtex...
  [3/5] makeindex: sem .nlo, pulando.
  [4/5] pdflatex (2a passada)...
  [5/5] pdflatex (3a passada)...
    pdflatex (4a passada, estabilizacao)...
  pdflatex estabilizado em 4 passadas.

[OK] PDF gerado com sucesso:
     C:\Users\Marquinho\.local\share\opencode\worktree\d66565\pq-fixA\text\pdf\tese_fa2.pdf
     Tamanho: 1045 KB
```

### slides

```text
/lm/lmsso10.pfb><c:/texlive/2024/texmf-dist/fonts/type1/public/lm/lmsso8.pfb><c
:/texlive/2024/texmf-dist/fonts/type1/public/lm/lmsy10.pfb><c:/texlive/2024/tex
mf-dist/fonts/type1/public/lm/lmsy8.pfb><c:/texlive/2024/texmf-dist/fonts/type1
/public/lm/lmtt10.pfb><c:/texlive/2024/texmf-dist/fonts/type1/public/amsfonts/s
ymbols/msam10.pfb>
Output written on defesa.pdf (42 pages, 583908 bytes).
Transcript written on defesa.log.
PDF: C:\Users\Marquinho\.local\share\opencode\worktree\d66565\pq-fixA\text\slides\defesa.pdf
```

A suíte completa foi executada sem remoção nem supressão de testes. Quatro falhas decorrem de arquivos ignorados ausentes neste worktree; a quinta é o ImportError do módulo `scripts.preservar_m5_m7_row`, que não foi alterado neste PR. Os novos scripts foram também executados diretamente contra os dados do clone principal, conferindo as matrizes e sensibilidades finais.

Auditoria dos logs: nenhuma citação ou referência indefinida; nenhum erro LaTeX e nenhum overfull acima de 20 pt na tese nem nos slides. Busca por travessões nos fontes alterados: vazia. Resumo preservado: 485 palavras; abstract preservado: 486 palavras, por separação em espaços após remoção dos comandos LaTeX. As mudanças do PR B foram mantidas; as únicas alterações adicionais em capitulo2.tex são a grafia do laboratório, e nos Estudos 2 e 3 são as frases sobre incorporação.

## Trechos antes e depois, na íntegra

### `text/latex/capitulo1.tex`

Antes:
```text
Operacionalmente, o protocolo estrutura-se em seis etapas: (i) execução do \textit{linkage} probabilístico mediante OpenRecLink, com múltiplos passos de blocagem \cite{Coeli2002blocking}; (ii) cálculo dos escores de similaridade campo a campo pelo comparador de registros; (iii) engenharia de atributos, incluindo indicadores binários de concordância, escores ponderados e termos de interação; (iv) treinamento de classificadores supervisionados sobre o conjunto rotulado, com avaliação por validação cruzada estratificada; (v) aplicação dos classificadores treinados aos pares da área cinza, com possibilidade de priorização de sensibilidade ou precisão, conforme o objetivo do estudo; e (vi) integração de regras de negócio baseadas no conhecimento do domínio. O \textit{framework} (estrutura metodológica) resultante é configurável e disponibiliza dois \textit{pipelines} (sequências operacionais) pré-definidos: um orientado à vigilância e outro à confirmação. Esses \textit{pipelines} são articulados por uma fronteira de Pareto, que permite ao gestor ou pesquisador calibrar o equilíbrio entre precisão e sensibilidade conforme as necessidades do cenário de aplicação.
```

Depois:
```text
Operacionalmente, o protocolo estrutura-se em seis etapas: (i) execução do \textit{linkage} probabilístico mediante OpenRecLink, com múltiplos passos de blocagem \cite{Coeli2002blocking}; (ii) cálculo dos escores de similaridade campo a campo pelo comparador de registros; (iii) engenharia de atributos, incluindo indicadores binários de concordância, escores ponderados e termos de interação; (iv) treinamento de classificadores supervisionados sobre o conjunto rotulado, com avaliação por validação cruzada estratificada; (v) aplicação dos classificadores treinados aos pares da área cinza, com possibilidade de priorização de sensibilidade ou precisão, conforme o objetivo do estudo; e (vi) integração de regras de negócio baseadas no conhecimento do domínio. O \textit{framework} (estrutura metodológica) resultante é configurável e disponibiliza dois \textit{pipelines} (sequências operacionais) predefinidos: um orientado à vigilância e outro à confirmação. Esses \textit{pipelines} são articulados por uma fronteira de Pareto, que permite ao gestor ou pesquisador calibrar o equilíbrio entre precisão e sensibilidade conforme as necessidades do cenário de aplicação.
```

### `text/latex/capitulo2.tex`

Antes:
```text
\textbf{Experiência brasileira.} \citeonline{Pita2017trainable} propuseram um modelo treinável, com árvores de decisão, \textit{naïve Bayes}, regressão logística, Floresta Aleatória, SVM linear e \textit{gradient boosting}, para avaliar e refinar a acurácia de vínculos probabilísticos, com o objetivo declarado de reduzir ou eliminar a revisão manual; o texto disponível no repositório apresenta acurácia, valor preditivo positivo, sensibilidade e especificidade em \textit{boxplots} de validação cruzada em dez partições, sem tabular os valores das métricas por modelo. \citeonline{Pita2018accuracy} examinaram a acurácia e a escalabilidade do \textit{linkage} probabilístico sobre a coorte de 114 milhões de brasileiros e reportaram, segundo o resumo, acurácia entre 93\% e 97\% de verdadeiros vínculos nos cenários controlados e reais. Em resumo de conferência publicado no \textit{International Journal of Population Data Science}, \citeonline{Pereira2024supervised} geraram mais de 28.000 pares do SUS revisados manualmente, dos quais 5.000 pares estratificados e aleatórios foram usados na avaliação, e compararam Floresta Aleatória, LightGBM, XGBoost e redes neurais, com melhor desempenho do XGBoost (acurácia de 96\%; 93\% com filtros de Bloom para preservação de privacidade); o resumo reporta acurácia, mas não sensibilidade, F$_1$ ou razão de classes. No Laboratório de Linkage e Análise de Dados Populacionais (LinkDataPop), \citeonline{Paiva2025comparador} aplicou árvores de decisão para classificação, que o trabalho denomina árvores de regressão, a potenciais pares do \textit{linkage} probabilístico entre o SIM e o Sinan-TB, com variáveis geradas pela versão em Java do Comparador de Registros, à qual acrescentou seis critérios; os três de endereço não foram usados no classificador. A base continha 1.719 potenciais pares, denominados registros no trabalho, dos quais os 780 da área de pares (753 pares e 27 não pares) foram utilizados no treinamento e no teste, com exclusão dos 939 da área cinza, definida pelo escore do OpenRecLink. O estudo comparou estratégias de balanceamento das classes, incluindo a deduplicação de pares, a replicação aleatória de não pares, que constituíam a classe minoritária, e a ponderação das amostras. Essa base é distinta do extrato desta tese, cujos subescores foram calculados pela implementação em Python de setembro de 2025, posterior à aprovação daquele trabalho, em março de 2025. \citeonline{Oliveira2026obitoTBpulmonar}, monografia de graduação em Saúde Coletiva do IESC-UFRJ sobre o \textit{linkage} para vigilância do óbito por tuberculose pulmonar no município do Rio de Janeiro, descreveu a revisão manual da área cinza, definida pelo escore do OpenRecLink, e o uso do Comparador de Registros para identificar pares não encontrados nas etapas anteriores. \citeonline{Lages2026trauma} desenvolveram, no projeto TRAUMA, método de \textit{linkage} com aprendizado de máquina supervisionado para automatizar a classificação dos potenciais pares que exigiriam revisão manual, validado com registros do SIM, do SIH-SUS, do Sinan e do e-SUS VS do Espírito Santo entre julho de 2018 e junho de 2025, com blocagem em múltiplos passos. Seis algoritmos foram avaliados em 106.363 pares rotulados, o LightGBM apresentou o melhor desempenho (F$_1$ de 0,996 no teste e de 0,993 na validação independente) e a comparação com o CIDACS-RL sobre um padrão-ouro de 11.662 pares mostrou estimativas maiores de sensibilidade, F$_1$ e especificidade para o método TRAUMA, com precisão semelhante. No subconjunto comum de 5.037 pares, a diferença de F$_1$ (0,01236; IC 95\% de 0,00921 a 0,01590) foi estatisticamente significativa por \textit{bootstrap} pareado; as demais diferenças foram interpretadas de forma descritiva. Na versão publicada pela \textit{Scientific Reports}, disponibilizada como \textit{Article in Press}\footnote{Os procedimentos metodológicos e os valores de desempenho citados neste texto foram conferidos na versão da editora, DOI 10.1038/s41598-026-73928-3, e não no \textit{preprint} do Research Square, DOI 10.21203/rs.3.rs-10229230/v1.}, todos os potenciais pares com similaridade igual ou superior a 65\% foram revisados manualmente, inclusive os de similaridade igual ou superior a 90\%, que na rotina são classificados automaticamente e também foram incluídos no treinamento; a razão entre vínculos e não pares é de aproximadamente 1:5 (17.362 e 89.001), o treinamento empregou pesos de classe balanceados, sem descrição de reamostragem para balanceamento, e o ajuste de hiperparâmetros empregou validação cruzada em três partições. Nas seções e tabelas consultadas dessa versão, a análise de desempenho considera a variação do limiar de classificação, sem apresentar estimativas por faixa de similaridade. No Rio de Janeiro, \citeonline{Coeli2021suboptimal} realizaram \textit{linkage} em condições subótimas na atenção primária e afirmaram ter gerado, por revisão clerical extensiva, um conjunto de treinamento que pretendem usar para avaliar classificadores de aprendizado de máquina, precedente do mesmo grupo e da mesma cidade em que se situa esta tese. \citeonline{Duarte2019scoring} e \citeonline{Brustulin2018posprocessamento} constituem precedentes brasileiros de pós-processamento do \textit{linkage} probabilístico sem classificador supervisionado, o primeiro por seleção do escore de corte em amostras revisadas manualmente, seguida da classificação automatizada dos pares, e o segundo por etapa determinística posterior ao \textit{linkage}.
```

Depois:
```text
\textbf{Experiência brasileira.} \citeonline{Pita2017trainable} propuseram um modelo treinável, com árvores de decisão, \textit{naïve Bayes}, regressão logística, Floresta Aleatória, SVM linear e \textit{gradient boosting}, para avaliar e refinar a acurácia de vínculos probabilísticos, com o objetivo declarado de reduzir ou eliminar a revisão manual; o texto disponível no repositório apresenta acurácia, valor preditivo positivo, sensibilidade e especificidade em \textit{boxplots} de validação cruzada em dez partições, sem tabular os valores das métricas por modelo. \citeonline{Pita2018accuracy} examinaram a acurácia e a escalabilidade do \textit{linkage} probabilístico sobre a coorte de 114 milhões de brasileiros e reportaram, segundo o resumo, acurácia entre 93\% e 97\% de verdadeiros vínculos nos cenários controlados e reais. Em resumo de conferência publicado no \textit{International Journal of Population Data Science}, \citeonline{Pereira2024supervised} geraram mais de 28.000 pares do SUS revisados manualmente, dos quais 5.000 pares estratificados e aleatórios foram usados na avaliação, e compararam Floresta Aleatória, LightGBM, XGBoost e redes neurais, com melhor desempenho do XGBoost (acurácia de 96\%; 93\% com filtros de Bloom para preservação de privacidade); o resumo reporta acurácia, mas não sensibilidade, F$_1$ ou razão de classes. No Laboratório de Linkage de Dados Populacionais (LinkDataPop), \citeonline{Paiva2025comparador} aplicou árvores de decisão para classificação, que o trabalho denomina árvores de regressão, a potenciais pares do \textit{linkage} probabilístico entre o SIM e o Sinan-TB, com variáveis geradas pela versão em Java do Comparador de Registros, à qual acrescentou seis critérios; os três de endereço não foram usados no classificador. A base continha 1.719 potenciais pares, denominados registros no trabalho, dos quais os 780 da área de pares (753 pares e 27 não pares) foram utilizados no treinamento e no teste, com exclusão dos 939 da área cinza, definida pelo escore do OpenRecLink. O estudo comparou estratégias de balanceamento das classes, incluindo a deduplicação de pares, a replicação aleatória de não pares, que constituíam a classe minoritária, e a ponderação das amostras. Essa base é distinta do extrato desta tese, cujos subescores foram calculados pela implementação em Python de setembro de 2025, posterior à aprovação daquele trabalho, em março de 2025. \citeonline{Oliveira2026obitoTBpulmonar}, monografia de graduação em Saúde Coletiva do IESC-UFRJ sobre o \textit{linkage} para vigilância do óbito por tuberculose pulmonar no município do Rio de Janeiro, descreveu a revisão manual da área cinza, definida pelo escore do OpenRecLink, e o uso do Comparador de Registros para identificar pares não encontrados nas etapas anteriores. \citeonline{Lages2026trauma} desenvolveram, no projeto TRAUMA, método de \textit{linkage} com aprendizado de máquina supervisionado para automatizar a classificação dos potenciais pares que exigiriam revisão manual, validado com registros do SIM, do SIH-SUS, do Sinan e do e-SUS VS do Espírito Santo entre julho de 2018 e junho de 2025, com blocagem em múltiplos passos. Seis algoritmos foram avaliados em 106.363 pares rotulados, o LightGBM apresentou o melhor desempenho (F$_1$ de 0,996 no teste e de 0,993 na validação independente) e a comparação com o CIDACS-RL sobre um padrão-ouro de 11.662 pares mostrou estimativas maiores de sensibilidade, F$_1$ e especificidade para o método TRAUMA, com precisão semelhante. No subconjunto comum de 5.037 pares, a diferença de F$_1$ (0,01236; IC 95\% de 0,00921 a 0,01590) foi estatisticamente significativa por \textit{bootstrap} pareado; as demais diferenças foram interpretadas de forma descritiva. Na versão publicada pela \textit{Scientific Reports}, disponibilizada como \textit{Article in Press}\footnote{Os procedimentos metodológicos e os valores de desempenho citados neste texto foram conferidos na versão da editora, DOI 10.1038/s41598-026-73928-3, e não no \textit{preprint} do Research Square, DOI 10.21203/rs.3.rs-10229230/v1.}, todos os potenciais pares com similaridade igual ou superior a 65\% foram revisados manualmente, inclusive os de similaridade igual ou superior a 90\%, que na rotina são classificados automaticamente e também foram incluídos no treinamento; a razão entre vínculos e não pares é de aproximadamente 1:5 (17.362 e 89.001), o treinamento empregou pesos de classe balanceados, sem descrição de reamostragem para balanceamento, e o ajuste de hiperparâmetros empregou validação cruzada em três partições. Nas seções e tabelas consultadas dessa versão, a análise de desempenho considera a variação do limiar de classificação, sem apresentar estimativas por faixa de similaridade. No Rio de Janeiro, \citeonline{Coeli2021suboptimal} realizaram \textit{linkage} em condições subótimas na atenção primária e afirmaram ter gerado, por revisão clerical extensiva, um conjunto de treinamento que pretendem usar para avaliar classificadores de aprendizado de máquina, precedente do mesmo grupo e da mesma cidade em que se situa esta tese. \citeonline{Duarte2019scoring} e \citeonline{Brustulin2018posprocessamento} constituem precedentes brasileiros de pós-processamento do \textit{linkage} probabilístico sem classificador supervisionado, o primeiro por seleção do escore de corte em amostras revisadas manualmente, seguida da classificação automatizada dos pares, e o segundo por etapa determinística posterior ao \textit{linkage}.
```

### `text/latex/capitulo3.tex`

Antes:
```text
A necessidade dessa abordagem é acentuada em crises sanitárias, nas quais a informação oportuna é requisito para a tomada de decisão no cuidado em saúde \cite{Werneck2020covid, Maia2022covid_tb}. A pandemia de COVID-19 atingiu de modo heterogêneo a população brasileira \cite{Hallal2020covid} e sobrecarregou os serviços hospitalares \cite{Ranzani2021covid}. No caso da TB, associou-se a redução documentada no número de notificações, interrupção de tratamentos e aumento de desfechos desfavoráveis \cite{Maia2022covid_tb}, e a queda na detecção de casos pode refletir a retração do acesso ao diagnóstico e a desarticulação das rotinas de vigilância, e não redução da incidência \cite{Maia2022covid_tb}. Cenários semelhantes podem ocorrer em crises climáticas e epidêmicas futuras, o que reforça a importância de protocolos de \textit{linkage} executáveis de forma ágil, padronizada e reprodutível, sem dependência exclusiva de revisão manual, em linha com diretrizes metodológicas internacionais que recomendam integrar técnicas de aprendizado de máquina a dados vinculados \cite{Haneef2022guidelines}. O trabalho insere-se no programa de pós-graduação do Instituto de Estudos em Saúde Coletiva (IESC) da UFRJ, na linha de pesquisa em Ciência de Dados em Saúde Pública, e no Laboratório de Linkage e Análise de Dados Populacionais, que desenvolve há mais de duas décadas metodologias de vinculação de bases para a vigilância epidemiológica e a avaliação de serviços de saúde \cite{Camargo2000reclink, Coeli2002blocking, Oliveira2016accuracy}, o que oferece a experiência acumulada com o CdR e as bases do município do Rio de Janeiro para a aplicação e a validação do método.
```

Depois:
```text
A necessidade dessa abordagem é acentuada em crises sanitárias, nas quais a informação oportuna é requisito para a tomada de decisão no cuidado em saúde \cite{Werneck2020covid, Maia2022covid_tb}. A pandemia de COVID-19 atingiu de modo heterogêneo a população brasileira \cite{Hallal2020covid} e sobrecarregou os serviços hospitalares \cite{Ranzani2021covid}. No caso da TB, associou-se a redução documentada no número de notificações, interrupção de tratamentos e aumento de desfechos desfavoráveis \cite{Maia2022covid_tb}, e a queda na detecção de casos pode refletir a retração do acesso ao diagnóstico e a desarticulação das rotinas de vigilância, e não redução da incidência \cite{Maia2022covid_tb}. Cenários semelhantes podem ocorrer em crises climáticas e epidêmicas futuras, o que reforça a importância de protocolos de \textit{linkage} executáveis de forma ágil, padronizada e reprodutível, sem dependência exclusiva de revisão manual, em linha com diretrizes metodológicas internacionais que recomendam integrar técnicas de aprendizado de máquina a dados vinculados \cite{Haneef2022guidelines}. O trabalho insere-se no programa de pós-graduação do Instituto de Estudos em Saúde Coletiva (IESC) da UFRJ, na linha de pesquisa em Ciência de Dados em Saúde Pública, e no Laboratório de Linkage de Dados Populacionais (LinkDataPop), que desenvolve há mais de duas décadas metodologias de vinculação de bases para a vigilância epidemiológica e a avaliação de serviços de saúde \cite{Camargo2000reclink, Coeli2002blocking, Oliveira2016accuracy}, o que oferece a experiência acumulada com o CdR e as bases do município do Rio de Janeiro para a aplicação e a validação do método.
```

### `text/latex/capitulo_dados.tex`

Antes:
```text
\item \textbf{SIM:} sistema que registra os óbitos ocorridos no território nacional a partir das Declarações de Óbito (DO), contendo variáveis demográficas (nome, data de nascimento, nome da mãe, sexo), geográficas (município de residência, endereço) e relativas à causa do óbito codificada pela Classificação Internacional de Doenças (CID-10) \cite{Jorge2007qualidade}; estimativas por pareamento com o registro civil situam sua completude em 98\% em 2016, com aumento em relação a 2015 e menor completude no norte do país \cite{Costa2020completeness}. Revisão de escopo recente relata, no resumo, cobertura abaixo de 90\% no Acre e no Maranhão, sem indicar o período dessas estimativas \cite{Reboucas2025sim}, diferindo dos 97,5\% e 93,7\%, respectivamente, estimados para 2016 na Tabela~1 de \citeonline{Costa2020completeness}.
```

Depois:
```text
\item \textbf{SIM:} o extrato contém somente óbitos com tuberculose como causa básica. O sistema registra os óbitos ocorridos no território nacional a partir das Declarações de Óbito (DO), contendo variáveis demográficas (nome, data de nascimento, nome da mãe, sexo), geográficas (município de residência, endereço) e relativas à causa do óbito codificada pela Classificação Internacional de Doenças (CID-10) \cite{Jorge2007qualidade}; estimativas por pareamento com o registro civil situam sua completude em 98\% em 2016, com aumento em relação a 2015 e menor completude no norte do país \cite{Costa2020completeness}. Revisão de escopo recente relata, no resumo, cobertura abaixo de 90\% no Acre e no Maranhão, sem indicar o período dessas estimativas \cite{Reboucas2025sim}, diferindo dos 97,5\% e 93,7\%, respectivamente, estimados para 2016 na Tabela~1 de \citeonline{Costa2020completeness}.
```

Antes:
```text
Os registros correspondem ao município do Rio de Janeiro, com notificações do Sinan-TB de 2013 a 2022 e óbitos do SIM de 2019 a 2022, e foram previamente submetidos a \textit{linkage} probabilístico por meio do programa OpenRecLink \cite{Camargo2000reclink}, gerando uma base de potenciais pares que constitui o objeto de análise do presente estudo. A escolha dessas bases justifica-se pelas razões detalhadas na Seção \ref{sec:justificativas-especificas}, em particular a relevância da tuberculose como condição marcadora da qualidade do cuidado em saúde e a experiência acumulada do grupo de pesquisa no \textit{linkage} dessas fontes de dados \cite{Oliveira2012uso, Bartholomay2014improved, Oliveira2016accuracy}.
```

Depois:
```text
O extrato provém do processo de \textit{linkage} descrito por \citeonline{Oliveira2026obitoTBpulmonar}, entre os óbitos de residentes no município do Rio de Janeiro com tuberculose como causa básica (SIM, 2019 a 2022) e as notificações de tuberculose (Sinan, 2013 a 2022), em quatro etapas: \textit{linkage} determinístico, 29 passos probabilísticos no OpenRecLink, um passo aberto e deduplicação (p.~33 a~39). Os vínculos encontrados na etapa determinística foram retirados antes do \textit{linkage} probabilístico, de modo que os potenciais pares do extrato correspondem ao que restou dessa etapa. O extrato reúne os potenciais pares de três dos 29 passos probabilísticos (6, 9 e 11), cujos parâmetros, pontos de corte (7,0240; 7,8444; 7,9019) e contagens de vínculos pelo corte e após a revisão coincidem com os descritos na monografia (Quadro~5, p.~36 a~37). Esses três passos contribuíram com 247 dos 2.549 pares do processo completo (p.~35); a proporção de 1:249 refere-se, portanto, à base exportada. A correspondência documental e as decisões de proveniência informadas pelo autor estão registradas em \texttt{data/proveniencia\_oliveira\_meta.json}. O órgão fornecedor das bases, a data da extração e o critério de escolha dos três passos exportados não constam da documentação disponível ao autor. A escolha dessas bases justifica-se pelas razões detalhadas na Seção \ref{sec:justificativas-especificas}, em particular a relevância da tuberculose como condição marcadora da qualidade do cuidado em saúde e a experiência acumulada do grupo de pesquisa no \textit{linkage} dessas fontes de dados \cite{Oliveira2012uso, Bartholomay2014improved, Oliveira2016accuracy}.
```

Antes:
```text
O procedimento de \textit{linkage} probabilístico no OpenRecLink seguiu o modelo de decisão de Fellegi e Sunter \citeyearpar{Fellegi1969theory}. Na etapa de padronização, os campos textuais de ambas as bases foram convertidos para caixa alta, com remoção de acentos diacríticos, preposições e artigos, e as datas de nascimento foram normalizadas para formato numérico contínuo (AAAAMMDD). A etapa de comparação no OpenRecLink utilizou três campos de identificação: nome próprio e nome da mãe, comparados por algoritmo aproximado (tipo APROX, com limiar de similaridade de 85\%); e data de nascimento, comparada caractere a caractere (tipo CARACTERE, com limiar de 65\%). A variável sexo não integrou a etapa de comparação, tendo sido utilizada exclusivamente como chave de blocagem em passos específicos. Os parâmetros de concordância e discordância atribuídos a cada campo estão detalhados na Tabela~\ref{tab:parametros-orl}. Para cada potencial par, o programa calculou um escore probabilístico correspondente à soma das razões de log-verossimilhança (\textit{log-likelihood ratios}) dos campos comparados, no qual campos concordantes contribuem positivamente e campos discordantes contribuem negativamente para o escore total \cite{Fellegi1969theory, Christen2012book}. Na base utilizada, os escores probabilísticos do OpenRecLink variaram de $-12{,}85$ (discordância em todos os campos) a $+15{,}58$ (concordância plena), distribuindo-se em 116 valores distintos. Para evitar ambiguidade, o presente trabalho distingue duas grandezas: o escore do \textit{linkage} probabilístico, calculado pelo OpenRecLink como soma das razões de log-verossimilhança, e a nota final do Comparador de Registros (CdR; Seção~\ref{sec:comparador-registros}), escore composto derivado dos 29 subescores de similaridade campo a campo. A área cinza refere-se, ao longo da tese, à nota final do CdR no intervalo $[5,9)$, e não ao escore probabilístico do OpenRecLink.
```

Depois:
```text
O procedimento de \textit{linkage} probabilístico no OpenRecLink seguiu o modelo de decisão de Fellegi e Sunter \citeyearpar{Fellegi1969theory}. Na etapa de padronização, os campos textuais de ambas as bases foram convertidos para caixa alta, com remoção de acentos diacríticos, preposições e artigos, e as datas de nascimento foram normalizadas para formato numérico contínuo (AAAAMMDD). A etapa de comparação no OpenRecLink utilizou três campos de identificação: nome próprio e nome da mãe, comparados por algoritmo aproximado (tipo APROX, com limiar de similaridade de 85\%); e data de nascimento, comparada caractere a caractere (tipo CARACTERE, com limiar de 65\%). A variável sexo não integrou a etapa de comparação, tendo sido utilizada exclusivamente como chave de blocagem em passos específicos. Os parâmetros de concordância e discordância atribuídos a cada campo estão detalhados na Tabela~\ref{tab:parametros-orl}. Para cada potencial par, o programa calculou um escore probabilístico correspondente à soma das razões de log-verossimilhança (\textit{log-likelihood ratios}) dos campos comparados, no qual campos concordantes contribuem positivamente e campos discordantes contribuem negativamente para o escore total \cite{Fellegi1969theory, Christen2012book}. Na base utilizada, os escores probabilísticos do OpenRecLink variaram de $-12{,}85$ (discordância em todos os campos) a $+15{,}58$ (concordância plena), distribuindo-se em 116 valores distintos. Para evitar ambiguidade, o presente trabalho distingue duas grandezas: o escore do \textit{linkage} probabilístico, calculado pelo OpenRecLink como soma das razões de log-verossimilhança, e a nota final do Comparador de Registros (CdR; Seção~\ref{sec:comparador-registros}), escore composto derivado dos 29 subescores de similaridade campo a campo. Na origem, a revisão manual incidiu sobre a área cinza do escore do OpenRecLink \cite{Oliveira2026obitoTBpulmonar}; nesta tese, a área cinza é definida pela nota final do CdR no intervalo $[5,9)$ (Seção~\ref{sec:comparador-registros}). As duas regiões não coincidem.
```

Antes:
```text
O \textit{linkage} probabilístico e a produção da base de potenciais pares foram realizados pela equipe do IESC-UFRJ, cujos profissionais são nomeados na Seção~\ref{sec:padrao-ouro}. O estudo integra o projeto de pesquisa ``Protocolos para \textit{linkage} de bases de dados, geocodificação e visualização aplicadas à tuberculose e COVID-19'', do Laboratório de \textit{Linkage} e Análise de Dados Populacionais (LinkDataPop) do IESC-UFRJ, aprovado pelo Comitê de Ética em Pesquisa do IESC-UFRJ (CAAE 43856720.1.0000.5286) e pelo da Secretaria Municipal de Saúde do Rio de Janeiro (CAAE 43856720.1.3001.5279). A revisão clerical que originou o campo PAR foi feita pela epidemiologista nomeada na Seção~\ref{sec:padrao-ouro}. A base de potenciais pares desta tese não deriva da base analisada por \citeonline{Paiva2025comparador}, que utilizou a versão em Java do CdR. O autor recebeu um extrato sem nomes, nome da mãe ou endereço nominal, mas que não é anônimo. Para cada potencial par (61.696 linhas e 60 campos), o extrato contém os 29 subescores de similaridade e a nota final do Comparador de Registros, o escore do \textit{linkage} probabilístico, o passo de blocagem e PAR; datas completas de nascimento, óbito, notificação, diagnóstico e encerramento, além da situação de encerramento; sexo, bairro e códigos de município; e identificadores internos dos registros (REFREC, COMPREC, R\_IDLINHA, C\_IDLINHA e C\_IDPESSOA). REFREC e COMPREC são usados para agrupar partições de treino e teste. O autor não teve acesso às bases nominais nem aos registros individuais de origem. O extrato é pseudonimizado e contém quase-identificadores; seu processamento local deve ser distinguido do envio de dossiês a APIs externas descrito adiante. Os passos de blocagem, os parâmetros de comparação e os critérios de revisão do \textit{linkage} de origem são reportados conforme a documentação disponível ao autor e não puderam ser auditados além do que o extrato permite verificar, limitação retomada no Capítulo~\ref{cap:conclusao}.
```

Depois:
```text
O \textit{linkage} probabilístico e a produção da base de potenciais pares foram realizados pela equipe do IESC-UFRJ, cujos profissionais são nomeados na Seção~\ref{sec:padrao-ouro}. O estudo integra o projeto de pesquisa ``Protocolos para \textit{linkage} de bases de dados, geocodificação e visualização aplicadas à tuberculose e COVID-19'', do Laboratório de \textit{Linkage} de Dados Populacionais (LinkDataPop) do IESC-UFRJ, aprovado pelo Comitê de Ética em Pesquisa do IESC-UFRJ (CAAE 43856720.1.0000.5286) e pelo da Secretaria Municipal de Saúde do Rio de Janeiro (CAAE 43856720.1.3001.5279). A revisão clerical que originou o campo PAR foi feita pela epidemiologista nomeada na Seção~\ref{sec:padrao-ouro}. A base de potenciais pares desta tese não deriva da base analisada por \citeonline{Paiva2025comparador}, que utilizou a versão em Java do CdR. O autor recebeu um extrato sem nomes, nome da mãe ou endereço nominal, mas que não é anônimo. Para cada potencial par (61.696 linhas e 60 campos), o extrato contém os 29 subescores de similaridade e a nota final do Comparador de Registros, o escore do \textit{linkage} probabilístico, o passo de blocagem e PAR; datas completas de nascimento, óbito, notificação, diagnóstico e encerramento, além da situação de encerramento; sexo, bairro e códigos de município; e identificadores internos dos registros (REFREC, COMPREC, R\_IDLINHA, C\_IDLINHA e C\_IDPESSOA). REFREC e COMPREC são usados para agrupar partições de treino e teste. O autor não teve acesso às bases nominais nem aos registros individuais de origem. O extrato é pseudonimizado e contém quase-identificadores; seu processamento local deve ser distinguido do envio de dossiês a APIs externas descrito adiante. Os passos de blocagem, os parâmetros de comparação e os critérios de revisão do \textit{linkage} de origem são reportados conforme a documentação disponível ao autor e não puderam ser auditados além do que o extrato permite verificar, limitação retomada no Capítulo~\ref{cap:conclusao}.
```

Antes:
```text
A geração de potenciais pares no OpenRecLink empregou a técnica de blocagem (\textit{blocking}), que particiona os registros em subgrupos definidos por chaves comuns e restringe as comparações aos pares que compartilham a mesma chave, reduzindo substancialmente o custo computacional \cite{Christen2012book}. No ORL, os parâmetros de blocagem foram definidos por combinações de variáveis fonéticas e demográficas (código \textit{Soundex} do primeiro e do último bloco do nome, sexo, ano de nascimento, primeiro e último bloco do nome da mãe e código do município de residência), organizadas em passos sequenciais. Adotou-se, ainda, a estratégia de exclusão progressiva dos registros do Sinan-TB já pareados em cada passo, de modo que, uma vez identificado um potencial par, o respectivo registro do Sinan não retornasse ao universo de busca dos passos subsequentes. Foram utilizados múltiplos passos de blocagem (\textit{blocking steps}), estratégia recomendada para maximizar a sensibilidade do processo ao permitir que pares não capturados por uma chave restritiva fossem recuperados por chaves alternativas \cite{Coeli2002blocking}. O procedimento de blocagem no OpenRecLink foi configurado com 30 passos (29 de blocagem convencional e 1 passo aberto sem restrição de chave), dos quais três foram selecionados para exportação com base na contagem de pares identificados e na viabilidade de revisão manual dos candidatos. Os três passos exportados, com suas respectivas chaves de blocagem, foram:
```

Depois:
```text
A geração de potenciais pares no OpenRecLink empregou a técnica de blocagem (\textit{blocking}), que particiona os registros em subgrupos definidos por chaves comuns e restringe as comparações aos pares que compartilham a mesma chave, reduzindo substancialmente o custo computacional \cite{Christen2012book}. No ORL, os parâmetros de blocagem foram definidos por combinações de variáveis fonéticas e demográficas (código \textit{Soundex} do primeiro e do último bloco do nome, sexo, ano de nascimento, primeiro e último bloco do nome da mãe, código do município de residência e bairro de residência (BAIRES)), organizadas em passos sequenciais. Adotou-se, ainda, a estratégia de exclusão progressiva dos registros do Sinan-TB já pareados em cada passo, de modo que, uma vez identificado um potencial par, o respectivo registro do Sinan não retornasse ao universo de busca dos passos subsequentes. Foram utilizados múltiplos passos de blocagem (\textit{blocking steps}), estratégia recomendada para maximizar a sensibilidade do processo ao permitir que pares não capturados por uma chave restritiva fossem recuperados por chaves alternativas \cite{Coeli2002blocking}. O procedimento incluiu 29 passos probabilísticos e um passo aberto, cuja chave de blocagem foi o \textit{Soundex} do primeiro fragmento do nome do paciente (PBLOCO), conforme a descrição metodológica e o Quadro~6 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~34 e~39). Três dos passos probabilísticos integram o extrato, sem informação sobre o critério de escolha para exportação. Os três passos exportados, com suas respectivas chaves de blocagem, foram:
```

Antes:
```text
Cada potencial par foi subsequentemente submetido ao comparador de registros descrito na Seção~\ref{sec:comparador-registros}, que calculou escores de similaridade campo a campo para as variáveis de identificação disponíveis em ambas as bases:
```

Depois:
```text
Na origem, o CdR foi usado na revisão do passo aberto, que recuperou 5 pares \cite{Oliveira2026obitoTBpulmonar} (p.~39); o cálculo da nota final sobre os potenciais pares desta tese foi outra execução, com a implementação em Python. Cada potencial par foi submetido ao comparador de registros descrito na Seção~\ref{sec:comparador-registros}, que calculou escores de similaridade campo a campo para as variáveis de identificação disponíveis em ambas as bases:
```

Antes:
```text
Cada par possui uma classificação de referência (padrão-ouro) atribuída por revisão manual, codificada no campo PAR como: 0 (não-par), 1 (par identificado pelo ponto de corte do escore probabilístico) ou 2 (par identificado em revisão manual abaixo do ponto de corte). Para fins de modelagem, os pares com PAR igual a 1 ou 2 foram agrupados em uma única classe positiva, resultando em uma variável-alvo binária. A disponibilidade dessa classificação de referência possibilita o treinamento e a avaliação dos classificadores supervisionados, pois fornece as observações rotuladas necessárias para o aprendizado.
```

Depois:
```text
Cada par possui uma classificação de referência (padrão-ouro) atribuída pelo corte do OpenRecLink e pela revisão manual abaixo dele, codificada no campo PAR como: 0 (não-par), 1 (par identificado pelo ponto de corte do escore probabilístico) ou 2 (par identificado em revisão manual abaixo do ponto de corte). Para fins de modelagem, os pares com PAR igual a 1 ou 2 foram agrupados em uma única classe positiva, resultando em uma variável-alvo binária. A disponibilidade dessa classificação de referência possibilita o treinamento e a avaliação dos classificadores supervisionados, pois fornece as observações rotuladas necessárias para o aprendizado.
```

Antes:
```text
O comparador de referência operacional desta tese é o ponto de corte do OpenRecLink aplicado na rotina do serviço: dos 247 vínculos, 211 (85,4\%) foram aceitos pelo corte (PAR igual a 1) e 36 (14,6\%) foram recuperados por revisão manual abaixo dele (PAR igual a 2). No extrato, o corte é recuperável por passo de blocagem e separa exatamente os pares PAR igual a 1 dos demais: o corte situa-se entre $6{,}28$ e $7{,}02$ no passo~6, entre $7{,}71$ e $7{,}84$ no passo~9 e entre $7{,}84$ e $7{,}90$ no passo~11, e nenhum par rotulado como não par tem escore acima do corte do seu passo (0 de 61.449). Esse resultado é uma consequência da definição do rótulo, não uma medida de acerto: como PAR igual a 1 é definido pelo próprio corte, sua precisão é de 100\% por construção, e a sensibilidade do corte (85,4\%) tem o denominador formado pelo mesmo critério de revisão (viés de verificação). O limite inferior de escore até o qual a revisão manual foi conduzida não está documentado; o escore dos 36 vínculos recuperados manualmente varia de $-5{,}65$ a $7{,}84$. O corte 9 da nota final do Comparador de Registros, que daria sensibilidade de 37,7\% e 2 falsos positivos, é tratado como referência hipotética (Tabela~\ref{tab:cap8-referencia}), e o ganho do GZ-CMD é medido na área cinza da nota do CdR, onde o OpenRecLink não decide.
```

Depois:
```text
O comparador de referência operacional desta tese é o ponto de corte do OpenRecLink aplicado na rotina do serviço: dos 247 vínculos, 211 (85,4\%) foram aceitos pelo corte (PAR igual a 1) e 36 (14,6\%) foram recuperados por revisão manual abaixo dele (PAR igual a 2). No extrato, o corte por passo de blocagem coincide com o descrito por \citeonline{Oliveira2026obitoTBpulmonar} (Quadro~5, p.~36 a~37): 7,0240 no passo~6, 7,8444 no passo~9 e 7,9019 no passo~11, separando exatamente os pares PAR igual a 1 dos demais, e nenhum par rotulado como não par tem escore acima do corte do seu passo (0 de 61.449). Esse resultado é uma consequência da definição do rótulo, não uma medida de acerto: como PAR igual a 1 é definido pelo próprio corte, sua precisão é de 100\% por construção, e a sensibilidade do corte (85,4\%) tem o denominador formado pelo mesmo critério de revisão (viés de verificação). O limite inferior de escore até o qual a revisão manual foi conduzida não está documentado; o escore dos 36 vínculos recuperados manualmente varia de $-5{,}65$ a $7{,}84$. O corte 9 da nota final do Comparador de Registros, que daria sensibilidade de 37,7\% e 2 falsos positivos, é tratado como referência hipotética (Tabela~\ref{tab:cap8-referencia}), e a comparação na área cinza da nota do CdR considera 151 vínculos, dos quais 123 foram aceitos pelo corte do OpenRecLink (PAR igual a 1) e 28 recuperados por revisão manual (PAR igual a 2). Nessa faixa, a cascata v3.1 aceitou automaticamente 113 vínculos na vigilância (105 com PAR igual a 1 e 8 com PAR igual a 2) e 65 na confirmação (61 e 4, respectivamente), menos que os 123 aceitos pelo corte do OpenRecLink. As contagens constam de \texttt{data/area\_cinza\_por\_par.csv} e \texttt{data/sensibilidade\_gzcmd\_por\_par.csv}; não sustentam ganho de recuperação automática sobre esse corte nessa faixa.
```

Antes:
```text
O padrão-ouro (\textit{gold standard}) foi construído em processo conduzido por dois profissionais vinculados ao IESC-UFRJ: um epidemiologista (Luís Guilherme Buteri Alves), responsável pelo pré-processamento dos dados (limpeza de registros com nomes inconsistentes, como ``homem pardo'', ``morador de rua'' e ``recém nascido'', e criação do campo DTNASC2 para normalização de datas de nascimento), e uma epidemiologista (Daniela Medina Coeli Bastos), responsável pela configuração do \textit{linkage}, execução dos passos de blocagem e revisão clerical dos potenciais pares. A revisão clerical (\textit{clerical review}) foi feita pela epidemiologista. A mesma epidemiologista executou o Comparador de Registros sobre os potenciais pares, com a implementação em Python desenvolvida pelo autor \cite{Jardim2024comparador}; a ordem entre essa execução e a revisão clerical que originou o campo PAR não está documentada. A classificação seguiu protocolo sequencial, no qual os potenciais pares foram ordenados em ordem decrescente do escore probabilístico do OpenRecLink e avaliados do escore mais alto para o mais baixo, de modo que os casos de concordância mais evidente fossem resolvidos primeiro. Para cada par, a revisora consultou os sistemas de origem (SIM e Sinan) e avaliou o status de vínculo ou não-par com base na concordância das variáveis de identificação utilizadas pelo OpenRecLink (nome, nome da mãe e data de nascimento) e em evidências complementares disponíveis nos registros originais, em particular o endereço de residência (logradouro, número e bairro), que não integrou o conjunto de campos comparados na etapa probabilística mas constitui informação discriminante para a confirmação manual dos pares. Embora dois profissionais tenham participado do processo, não foram produzidos registros independentes de classificação por cada avaliador, o que impede o cálculo de concordância interavaliadores (\textit{kappa} de Cohen). Essa limitação, frequente em estudos operacionais de \textit{linkage} no contexto brasileiro \cite{Coeli2021suboptimal}, é discutida na Seção~\ref{sec:concl-limitacoes}.
```

Depois:
```text
O padrão-ouro (\textit{gold standard}) foi construído em processo conduzido por dois profissionais vinculados ao IESC-UFRJ: um epidemiologista (Luís Guilherme Buteri Alves), responsável pelo pré-processamento dos dados (limpeza de registros com nomes inconsistentes, como ``homem pardo'', ``morador de rua'' e ``recém nascido'', e criação do campo DTNASC2 para normalização de datas de nascimento), e uma epidemiologista (Daniela Medina Coeli Bastos), responsável pela configuração do \textit{linkage}, execução dos passos de blocagem e revisão clerical dos potenciais pares. A revisão clerical (\textit{clerical review}) foi feita pela epidemiologista. A mesma epidemiologista executou o Comparador de Registros sobre os potenciais pares, com a implementação em Python desenvolvida pelo autor \cite{Jardim2024comparador}, antes da revisão clerical que originou o campo PAR, de modo que a nota final e os subescores estavam disponíveis durante a revisão. A classificação seguiu protocolo sequencial, no qual os potenciais pares foram ordenados em ordem decrescente do escore probabilístico do OpenRecLink e avaliados do escore mais alto para o mais baixo, de modo que os casos de concordância mais evidente fossem resolvidos primeiro. Para cada par, a revisora consultou os sistemas de origem (SIM e Sinan) e avaliou o status de vínculo ou não-par com base na concordância das variáveis de identificação utilizadas pelo OpenRecLink (nome, nome da mãe e data de nascimento) e em evidências complementares disponíveis nos registros originais, em particular o endereço de residência (logradouro, número e bairro), que não integrou o conjunto de campos comparados na etapa probabilística mas constitui informação discriminante para a confirmação manual dos pares. Como a revisão clerical foi feita por uma única revisora, sem segunda classificação independente, não é possível calcular a concordância interavaliadores (\textit{kappa} de Cohen). Essa limitação, frequente em estudos operacionais de \textit{linkage} no contexto brasileiro \cite{Coeli2021suboptimal}, é discutida na Seção~\ref{sec:concl-limitacoes}. Essa ordem pode levar o padrão-ouro a incorporar parte da informação avaliada nesta tese (viés de incorporação). Os 211 vínculos com PAR igual a 1 não dependem do CdR, pois correspondem exatamente aos potenciais pares com escore do OpenRecLink acima do corte do seu passo, sem nenhum não par acima dele (0 de 61.449). Podem ter considerado a nota e os subescores os 36 vínculos com PAR igual a 2, recuperados abaixo do corte, e as decisões de não par na faixa revisada, cujo limite inferior não está documentado. Por isso, podem estar otimistas, em magnitude não estimada, a distribuição dos vínculos por faixa da nota (Tabela~\ref{tab:score-bands}), os classificadores que utilizam a nota e os subescores (Capítulos~\ref{cap:estudo2} e~\ref{cap:estudo3}), a discriminação e a calibração do estimador do GZ-CMD e as sensibilidades da cascata (Capítulo~\ref{cap:estudo4}); a sensibilidade do corte do OpenRecLink (85,4\%) também depende desse critério, pois os 36 vínculos com PAR igual a 2 compõem seu denominador.
```

Antes:
```text
A auditoria da blocagem (M8) identificou 61.696 linhas no extrato, correspondentes a 61.684 potenciais pares distintos, com 12 pares presentes em mais de um passo, todos classificados como não-pares. A Tabela~\ref{tab:blocagem-rendimento} apresenta o rendimento dos três passos exportados. O passo~9 foi o mais produtivo, tanto pelo número de vínculos (127) quanto pela proporção de vínculos (8,8\%), mas também apresentou a maior fração de vínculos na área cinza definida pela nota final do comparador (82,7\%), indicando que a produtividade da blocagem não elimina a necessidade de pós-processamento dos potenciais pares.
```

Depois:
```text
A monografia registra dois critérios de deduplicação das notificações: maior data de notificação e data mais próxima do óbito \cite{Oliveira2026obitoTBpulmonar} (p.~41 e resumo). A base exportada não foi deduplicada por óbito; essa operação é distinta da repetição de um mesmo potencial par entre passos. A auditoria da blocagem (M8) identificou 61.696 linhas no extrato, correspondentes a 61.684 potenciais pares distintos, com 12 pares presentes em mais de um passo, todos classificados como não-pares. A Tabela~\ref{tab:blocagem-rendimento} apresenta o rendimento dos três passos exportados. O passo~9 foi o mais produtivo, tanto pelo número de vínculos (127) quanto pela proporção de vínculos (8,8\%), mas também apresentou a maior fração de vínculos na área cinza definida pela nota final do comparador (82,7\%), indicando que a produtividade da blocagem não elimina a necessidade de pós-processamento dos potenciais pares.
```

### `text/latex/conclusao.tex`

Antes:
```text
A principal limitação deste estudo reside na construção do padrão-ouro. O processo de construção do conjunto de referência envolveu dois profissionais do IESC-UFRJ: um epidemiologista (Luís Guilherme Buteri Alves), responsável pelo pré-processamento dos dados (limpeza de registros com nomes inconsistentes e normalização de campos), e uma epidemiologista (Daniela Medina Coeli Bastos), responsável pela configuração do \textit{linkage} probabilístico, execução dos passos de blocagem e revisão clerical dos potenciais pares. A revisão clerical foi feita pela epidemiologista, sem registros independentes de classificação por outro avaliador, o que impede o cálculo de concordância interavaliadores (\textit{kappa} de Cohen) e a resolução de casos ambíguos por consenso formal. Essa configuração, frequente em estudos operacionais de \textit{linkage} \cite{Gupta2022framework, Gupta2024manual}, pode introduzir viés na rotulagem, especialmente na área cinza, onde a ambiguidade dos registros é maior \cite{Harron2017linkagequality}. Estudos futuros que incorporem dupla revisão independente e métricas de concordância poderão estimar a magnitude dessa incerteza. Se a nota do CdR tiver apoiado a revisão clerical, cuja ordem em relação à execução do comparador não está documentada, o padrão-ouro pode incorporar a própria nota avaliada (viés de incorporação), o que favoreceria a nota final e os classificadores que a utilizam.
```

Depois:
```text
A principal limitação deste estudo reside na construção do padrão-ouro. O processo de construção do conjunto de referência envolveu dois profissionais do IESC-UFRJ: um epidemiologista (Luís Guilherme Buteri Alves), responsável pelo pré-processamento dos dados (limpeza de registros com nomes inconsistentes e normalização de campos), e uma epidemiologista (Daniela Medina Coeli Bastos), responsável pela configuração do \textit{linkage} probabilístico, execução dos passos de blocagem e revisão clerical dos potenciais pares. A revisão clerical foi feita pela epidemiologista, sem registros independentes de classificação por outro avaliador, o que impede o cálculo de concordância interavaliadores (\textit{kappa} de Cohen) e a resolução de casos ambíguos por consenso formal. Essa configuração, frequente em estudos operacionais de \textit{linkage} \cite{Gupta2022framework, Gupta2024manual}, pode introduzir viés na rotulagem, especialmente na área cinza, onde a ambiguidade dos registros é maior \cite{Harron2017linkagequality}. Estudos futuros que incorporem dupla revisão independente e métricas de concordância poderão estimar a magnitude dessa incerteza. O Comparador de Registros foi executado antes da revisão clerical, com a nota final e os subescores disponíveis à revisora. O padrão-ouro pode incorporar parte dessa informação (viés de incorporação), de modo que a avaliação da nota, dos classificadores, da calibração e das sensibilidades da cascata pode estar otimista, em magnitude não estimada; uma revisão cega à nota, em amostra de potenciais pares abaixo do corte, permitiria quantificá-lo.
```

### `text/latex/estudo2_classificadores.tex`

Antes:
```text
(trecho inexistente)
```

Depois:
```text
O Comparador de Registros foi executado antes da revisão clerical que originou PAR, com a nota final e os subescores disponíveis à revisora. O padrão-ouro pode incorporar parte dessa informação (viés de incorporação), de modo que as métricas dos classificadores que a utilizam podem estar otimistas, em magnitude não estimada (Seção~\ref{sec:padrao-ouro}).
```

### `text/latex/estudo3_estrategias.tex`

Antes:
```text
(trecho inexistente)
```

Depois:
```text
O Comparador de Registros foi executado antes da revisão clerical que originou PAR, com a nota final e os subescores disponíveis à revisora. O padrão-ouro pode incorporar parte dessa informação (viés de incorporação), de modo que as métricas dos classificadores que a utilizam podem estar otimistas, em magnitude não estimada (Seção~\ref{sec:padrao-ouro}).
```

### `text/latex/estudo4_gzcmd.tex`

Antes:
```text
O GZ-CMD transforma a área cinza de um problema de volume em um problema de decisão. Na saída do Comparador de Registros posterior ao OpenRecLink, os 21.620 potenciais pares com nota final em $[5,9)$, que concentram 151 dos 247 vínculos, ou seguiriam integralmente ao revisor humano ou seriam descartados por um corte hipotético da nota em 9, que aceitaria 93 vínculos (37,7\%) com 2 falsos positivos, de modo que a escolha se daria entre uma carga de revisão incompatível com a rotina do serviço e a perda de 62,3\% dos vínculos (Tabela~\ref{tab:cap8-referencia}). Ao comparar, para cada potencial par, o \textit{expected loss} de vínculo, não-vínculo, revisão por LLM e revisão humana, a política v3.1 resolveu a maior parte dessa área sem revisão humana no modo vigilância, aceitando automaticamente 78,5\% dos vínculos, ao custo de 34 falsos positivos automáticos, e encaminhando 111 potenciais pares ao profissional do serviço. No modo confirmação, em que o falso vínculo custa cinco vezes a perda de um vínculo, a mesma política concentrou o esforço humano em 127 potenciais pares de probabilidade intermediária, sem falso positivo automático, porém com sensibilidade automática de 53,4\%, e deixou documentado por que não avança além desse ponto: a restrição de falsos vínculos esperados a 0,2\% mostrou-se inviável com as probabilidades disponíveis, e a calibração absoluta e o erro humano permanecem a validar no serviço.
```

Depois:
```text
O GZ-CMD transforma a área cinza de um problema de volume em um problema de decisão. Nos cenários hipotéticos de referência da saída do Comparador de Registros posterior ao OpenRecLink, os 21.620 potenciais pares com nota final em $[5,9)$, que concentram 151 dos 247 vínculos, ou seguiriam integralmente ao revisor humano ou seriam descartados por um corte hipotético da nota em 9, que aceitaria 93 vínculos (37,7\%) com 2 falsos positivos, contrastando-se a revisão integral com a perda de 62,3\% dos vínculos (Tabela~\ref{tab:cap8-referencia}). Ao comparar, para cada potencial par, o \textit{expected loss} de vínculo, não-vínculo, revisão por LLM e revisão humana, a política v3.1, aplicada à base inteira, aceitou automaticamente 194 dos 247 vínculos (78,5\%) no modo vigilância, ao custo de 34 falsos positivos automáticos, e encaminhando 111 potenciais pares ao profissional do serviço. No modo confirmação, em que o falso vínculo custa cinco vezes a perda de um vínculo, a mesma política concentrou o esforço humano em 127 potenciais pares de probabilidade intermediária, sem falso positivo automático, porém com sensibilidade automática de 53,4\%, e deixou documentado por que não avança além desse ponto: a restrição de falsos vínculos esperados a 0,2\% mostrou-se inviável com as probabilidades disponíveis, e a calibração absoluta e o erro humano permanecem a validar no serviço.
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
O Comparador de Registros foi executado antes da revisão clerical que originou PAR, de modo que a nota final e os subescores estavam disponíveis à revisora. O padrão-ouro pode incorporar parte da informação avaliada (viés de incorporação); a discriminação, a calibração e as sensibilidades da cascata podem, por isso, estar otimistas, em magnitude não estimada. Os vínculos com PAR igual a 1 decorrem do corte do OpenRecLink, sem depender do CdR; a revisão abaixo do corte pode ter considerado suas saídas (Seção~\ref{sec:padrao-ouro}). Entre os vínculos com PAR igual a 1, a cascata v3.1 aceitou automaticamente 184 de 211 na vigilância e 124 de 211 na confirmação; entre os com PAR igual a 2, aceitou 10 de 36 e 8 de 36, respectivamente. Na área cinza, aceitou 113 de 151 na vigilância e 65 de 151 na confirmação, frente aos 123 aceitos pelo corte do OpenRecLink nessa faixa (\texttt{data/sensibilidade\_gzcmd\_por\_par.csv}). Essas contagens não demonstram recuperação automática superior à do corte na área cinza.
```

### `text/latex/glossario.tex`

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item[API] \textit{Application Programming Interface}. Interface de programação que define um conjunto padronizado de rotinas, protocolos e ferramentas para comunicação entre componentes de \textit{software}, permitindo integração modular entre sistemas.
```

Antes:
```text

\item[API] \textit{Application Programming Interface}. Interface de programação que define um conjunto padronizado de rotinas, protocolos e ferramentas para comunicação entre componentes de \textit{software}, permitindo integração modular entre sistemas.
```

Depois:
```text
(trecho removido)
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item[Classificação] Tarefa de aprendizado supervisionado que atribui rótulos discretos (vínculo ou não-par) a instâncias com base em atributos preditores.
```

Antes:
```text
\item[Classificação] Tarefa de aprendizado supervisionado que atribui rótulos discretos (vínculo ou não-par) a instâncias com base em atributos preditores.
```

Depois:
```text
\item[Comparador de Registros (CdR)] Ferramenta que calcula escores de similaridade campo a campo entre potenciais pares, produzindo subescores individuais e uma nota final agregada. Nesta tese, foi utilizada uma implementação em Python \cite{Jardim2024comparador} que parte dos critérios de nome e de data de \citeonline{Lucena2013algoritmos} e inclui subescores de endereço derivados dos critérios de \citeonline{Paiva2025comparador}, além dos de município.
```

Antes:
```text
\item[Comparador de Registros (CdR)] Ferramenta que calcula escores de similaridade campo a campo entre potenciais pares, produzindo subescores individuais e uma nota final agregada. Nesta tese, foi utilizada uma implementação em Python \cite{Jardim2024comparador} do algoritmo proposto por Lucena \cite{Lucena2013algoritmos}.

\item[Decisão de limiar] Classificação de potenciais pares pela comparação do escore composto com um ou dois pontos de corte fixos, sem uso de atributos adicionais. Com dois cortes, resultam três regiões (vínculo, revisão e não-vínculo), como na regra de Fellegi e Sunter; a política v3.1 do GZ-CMD substitui os cortes por quatro ações (vínculo, não-vínculo, revisão por modelo de linguagem e revisão humana). Nesta tese, o corte do OpenRecLink usado pelo serviço é o comparador operacional de referência; o corte 9 da nota final do CdR, com revisão integral dos pares abaixo dele, é comparador hipotético; e o corte 6,0 da nota é a referência histórica da análise de equidade.
```

Depois:
```text
\item[Decisão de limiar] Classificação de potenciais pares pela comparação do escore composto com um ou dois pontos de corte fixos, sem uso de atributos adicionais. Com dois cortes, resultam três regiões (vínculo, revisão e não-vínculo), como na regra de Fellegi e Sunter; a política v3.1 do GZ-CMD substitui os cortes por quatro ações (vínculo, não-vínculo, revisão por modelo de linguagem e revisão humana). Nesta tese, o corte do OpenRecLink usado pelo serviço é o comparador operacional de referência; o corte 9 da nota final do CdR é comparador hipotético; a revisão integral de $[5,9)$, aceitando-se automaticamente as notas $\geq9$ e descartando-se as notas $<5$, é outro cenário hipotético; e o corte 6,0 da nota é a referência histórica da análise de equidade.
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item[EVR] Valor Esperado da Revisão (\textit{Expected Value of Review}). Métrica da versão v3 que quantifica o ganho esperado de submeter um potencial par à revisão por modelo de linguagem em comparação com a decisão automática de custo mínimo.
```

Antes:
```text
\item[EVR] Valor Esperado da Revisão (\textit{Expected Value of Review}). Métrica da versão v3 que quantifica o ganho esperado de submeter um potencial par à revisão por modelo de linguagem em comparação com a decisão automática de custo mínimo.
```

Depois:
```text
\item[F$_1$-Score] Média harmônica entre precisão e sensibilidade, utilizada como métrica-síntese do desempenho de classificadores.
```

Antes:
```text

\item[F$_1$-Score] Média harmônica entre precisão e sensibilidade, utilizada como métrica-síntese do desempenho de classificadores.
```

Depois:
```text
(trecho removido)
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item[\textit{Linkage} determinístico] Estratégia de \textit{linkage} baseada em regras exatas de concordância entre campos identificadores.

\item[\textit{Linkage} probabilístico] Estratégia de \textit{linkage} fundamentada na teoria de Fellegi e Sunter, que atribui pesos aos campos comparados e calcula um escore composto para classificar pares.
```

Antes:
```text
\item[Potencial par] Combinação de dois registros, provenientes de bases distintas, que foram selecionados pela etapa de blocagem para comparação detalhada (potencial par).
```

Depois:
```text
\item[\textit{Platt scaling}] Técnica de pós-processamento que ajusta os escores brutos de um classificador por meio de regressão logística (função sigmoide), convertendo-os em probabilidades calibradas. Na execução histórica v3, foi aplicada de forma supervisionada à nota final; o ajuste por momentos nas âncoras é uma variante avaliada como alternativa.
```

Antes:
```text
\item[Vínculo] Par de registros que se refere ao mesmo indivíduo, confirmado por revisão manual ou padrão-ouro (vínculo).

\item[\textit{Platt scaling}] Técnica de pós-processamento que ajusta os escores brutos de um classificador por meio de regressão logística (função sigmoide), convertendo-os em probabilidades calibradas. Na execução histórica v3, foi aplicada de forma supervisionada à nota final; o ajuste por momentos nas âncoras é uma variante avaliada como alternativa.
```

Depois:
```text
\item[Potencial par] Combinação de dois registros, provenientes de bases distintas, que foram selecionados pela etapa de blocagem para comparação detalhada.
```

Antes:
```text
\item[\textit{Linkage} determinístico] Estratégia de \textit{linkage} baseada em regras exatas de concordância entre campos identificadores.

\item[\textit{Linkage} probabilístico] Estratégia de \textit{linkage} fundamentada na teoria de Fellegi e Sunter, que atribui pesos aos campos comparados e calcula um escore composto para classificar pares.
```

Depois:
```text
(trecho removido)
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item[Vínculo] Par de registros que se refere ao mesmo indivíduo, confirmado pelo corte do escore ou por revisão manual no padrão-ouro.
```

### `text/latex/tables/tab_cap8_referencia.tex`

Antes:
```text
\parbox{\linewidth}{\footnotesize \textsuperscript{a}\,Prática do serviço: 211 pares aceitos pelo ponto de corte do escore do OpenRecLink (PAR~=~1) e 36 vínculos recuperados por revisão manual abaixo do corte (PAR~=~2), totalizando 247 (85,4\% e 14,6\%). O corte é inferido por passo de blocagem no extrato (menor escore entre os PAR~=~1 do passo) e separa exatamente PAR~=~1 dos demais pares nos três passos. O número de pares examinados na revisão manual não está documentado (n.d.). \textsuperscript{b}\,Zero por construção: como PAR~=~1 é definido pelo corte do OpenRecLink e a revisão manual só acrescenta vínculos, nenhum não vínculo acima do corte aparece no extrato e a sensibilidade com revisão é 100\% pelo próprio rótulo. Esses valores não medem a acurácia do corte (viés de verificação), e o ganho do GZ-CMD se mede na área cinza da nota do Comparador de Registros (CdR), onde o OpenRecLink não decide. \textsuperscript{c}\,Hipotético: pares com nota final em $[5,9)$ (21.620 pares, 151 vínculos) ao revisor humano, aceitando-se automaticamente os de nota $\geq 9$ (95 pares, 93 vínculos, 2 falsos positivos) e descartando-se os de nota $<5$; sensibilidade com humano $(93+0{,}95\times151)/247$, sob erro humano hipotético de 5\%, não medido; sob erro simétrico de 5\% sobre os 21.469 não vínculos, a revisão integral produziria cerca de 1.073 falsos positivos humanos, hipótese provavelmente pessimista, apresentada apenas para dimensionar a carga. A sensibilidade automática mantém os 247 vínculos no denominador. CdR: Comparador de Registros. n.a.: não se aplica. n.d.: não documentado. Fontes: \texttt{data/referencia\_comparador\_pos\_orl.csv}, \texttt{data/funil\_gzcmd\_final.csv} e \texttt{data/funil\_gzcmd\_final\_metricas.csv} \cite{Jardim2026phdrepo}.}
```

Depois:
```text
\parbox{\linewidth}{\footnotesize \textsuperscript{a}\,Prática do serviço: 211 pares aceitos pelo ponto de corte do escore do OpenRecLink (PAR~=~1) e 36 vínculos recuperados por revisão manual abaixo do corte (PAR~=~2), totalizando 247 (85,4\% e 14,6\%). O corte coincide com o descrito por \citeonline{Oliveira2026obitoTBpulmonar} (Quadro~5, p.~36 a~37) e separa exatamente PAR~=~1 dos demais pares nos três passos. O número de pares examinados na revisão manual não está documentado (n.d.). \textsuperscript{b}\,Zero por construção: como PAR~=~1 é definido pelo corte do OpenRecLink e a revisão manual só acrescenta vínculos, nenhum não vínculo acima do corte aparece no extrato e a sensibilidade com revisão é 100\% pelo próprio rótulo. Esses valores não medem a acurácia do corte (viés de verificação), e a área cinza da nota do Comparador de Registros (CdR) contém 151 vínculos: 123 aceitos pelo corte do OpenRecLink e 28 recuperados por revisão manual. Nessa faixa, a cascata v3.1 aceitou automaticamente 113 na vigilância e 65 na confirmação, abaixo dos 123 aceitos pelo corte (\texttt{data/area\_cinza\_por\_par.csv} e \texttt{data/sensibilidade\_gzcmd\_por\_par.csv}). \textsuperscript{c}\,Hipotético: pares com nota final em $[5,9)$ (21.620 pares, 151 vínculos) ao revisor humano, aceitando-se automaticamente os de nota $\geq 9$ (95 pares, 93 vínculos, 2 falsos positivos) e descartando-se os de nota $<5$; sensibilidade com humano $(93+0{,}95\times151)/247$, sob erro humano hipotético de 5\%, não medido; sob erro simétrico de 5\% sobre os 21.469 não vínculos, a revisão integral produziria cerca de 1.073 falsos positivos humanos, hipótese provavelmente pessimista, apresentada apenas para dimensionar a carga. A sensibilidade automática mantém os 247 vínculos no denominador. CdR: Comparador de Registros. n.a.: não se aplica. n.d.: não documentado. Fontes: \texttt{data/referencia\_comparador\_pos\_orl.csv}, \texttt{data/funil\_gzcmd\_final.csv} e \texttt{data/funil\_gzcmd\_final\_metricas.csv} \cite{Jardim2026phdrepo}.}
```

### `text/latex/tese.tex`

Antes:
```text
\parbox{10.3cm}{Tese de Doutorado apresentada ao Programa de P\'os-Gradua\c{c}\~ao em Sa\'ude Coletiva do Instituto de Estudos de Sa\'ude Coletiva da Universidade Federal do Rio de Janeiro (UFRJ), como parte dos requisitos necess\'arios \`a obten\c{c}\~ao do t\'itulo de Doutor em Ci\^encias (Sa\'ude Coletiva).\par
```

Depois:
```text
\parbox{10.3cm}{Tese de Doutorado apresentada ao Programa de P\'os-Gradua\c{c}\~ao em Sa\'ude Coletiva do Instituto de Estudos em Sa\'ude Coletiva da Universidade Federal do Rio de Janeiro (UFRJ), como parte dos requisitos necess\'arios \`a obten\c{c}\~ao do t\'itulo de Doutor em Ci\^encias (Sa\'ude Coletiva).\par
```

Antes:
```text
\hspace{1,59cm} Aprendizado de m\'aquina aplicado ao p\'os-processamento do relacionamento probabil\'{\i}stico de bases de dados de sa\'ude / Marco Elisio Oliveira Jardim. -- Rio de Janeiro, 2026.
```

Depois:
```text
\hspace{1,59cm} Aprendizado de m\'aquina aplicado ao p\'os-processamento do relacionamento probabil\'{\i}stico de bases de dados de sa\'ude / Marco Elisio Oliveira Jardim. -{}- Rio de Janeiro, 2026.
```

Antes:
```text
% Contagem de folhas conforme as 199 páginas do PDF no build final de 2026-11-04.
```

Depois:
```text
(trecho removido)
```

Antes:
```text
\parbox{12cm}{\centering Tese de Doutorado apresentada ao Programa de P\'os-Gradua\c{c}\~ao em Sa\'ude Coletiva do Instituto de Estudos de Sa\'ude Coletiva da Universidade Federal do Rio de Janeiro (UFRJ), como parte dos requisitos necess\'arios \`a obten\c{c}\~ao do t\'itulo de Doutor em Ci\^encias (Sa\'ude Coletiva).}
```

Depois:
```text
\parbox{12cm}{\centering Tese de Doutorado apresentada ao Programa de P\'os-Gradua\c{c}\~ao em Sa\'ude Coletiva do Instituto de Estudos em Sa\'ude Coletiva da Universidade Federal do Rio de Janeiro (UFRJ), como parte dos requisitos necess\'arios \`a obten\c{c}\~ao do t\'itulo de Doutor em Ci\^encias (Sa\'ude Coletiva).}
```

### `text/slides/defesa.tex`

Antes:
```text
\itens{\item A nota final do CdR não é o escore probabilístico do OpenRecLink
```

Depois:
```text
\itens{\item Origem: processo de \textit{linkage} descrito por Oliveira (2026), p.~33 a~39
\item A nota final do CdR não é o escore probabilístico do OpenRecLink
```

Antes:
```text
(trecho inexistente)
```

Depois:
```text
\item CdR executado antes da revisão: risco de viés de incorporação
```

Antes:
```text
\itens{\item Padrão-ouro com uma revisora e viés de verificação pelo corte
```

Depois:
```text
\itens{\item Padrão-ouro com uma revisora, viés de verificação pelo corte e risco de incorporação da nota do CdR
```

Antes:
```text
\item Projeto aprovado no CEP do IESC-UFRJ e da SMS-Rio (CAAE 43856720.1.0000.5286)
```

Depois:
```text
\item Projeto: CEP IESC-UFRJ, CAAE 43856720.1.0000.5286; CEP SMS-Rio, CAAE 43856720.1.3001.5279
```

Antes:
```text
\item Projeto aprovado no CEP do IESC-UFRJ e da SMS-Rio (CAAE 43856720.1.0000.5286)
```

Depois:
```text
\item Projeto: CEP IESC-UFRJ, CAAE 43856720.1.0000.5286; CEP SMS-Rio, CAAE 43856720.1.3001.5279
```

### `text/slides/roteiro_defesa.md`

Antes:
```text
O estudo integra projeto aprovado no CEP do IESC-UFRJ e da SMS-Rio (CAAE 43856720.1.0000.5286). A ausência de nomes no dossiê não transforma o extrato em dado anônimo nem elimina risco de reidentificação. O hash determinístico das execuções históricas pode ser recalculado por quem detém os identificadores de origem. A exportação restrita com `pair_id` por HMAC foi implementada, sem depósito da matriz nem saneamento retroativo dos artefatos históricos.
```

Depois:
```text
O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). A ausência de nomes no dossiê não transforma o extrato em dado anônimo nem elimina risco de reidentificação. O hash determinístico das execuções históricas pode ser recalculado por quem detém os identificadores de origem. A exportação restrita com `pair_id` por HMAC foi implementada, sem depósito da matriz nem saneamento retroativo dos artefatos históricos.

A origem do extrato é o processo de linkage descrito por Oliveira (2026, p. 33 a 39), após retirada dos vínculos determinísticos e limitado a três passos probabilísticos. Na área cinza do CdR, o corte aceitou 123 vínculos; a cascata aceitou 113 na vigilância e 65 na confirmação. Não há ganho de recuperação automática sobre esse corte nessa faixa.
```

Antes:
```text
Luís Guilherme Buteri Alves realizou o pré-processamento, e Daniela Medina Coeli Bastos configurou a vinculação e revisou os potenciais pares. Dos 247 vínculos, 211 vieram do corte e 36 da revisão abaixo dele. Não houve duas classificações independentes nem kappa humano. A referência incorpora o corte, tem viés de verificação e limita a sensibilidade aos potenciais pares exportados.
```

Depois:
```text
Luís Guilherme Buteri Alves realizou o pré-processamento, e Daniela Medina Coeli Bastos configurou a vinculação e revisou os potenciais pares. Dos 247 vínculos, 211 vieram do corte e 36 da revisão abaixo dele. O CdR foi executado antes da revisão, com risco de viés de incorporação. Não houve segunda classificação independente nem kappa. A referência incorpora o corte e limita a sensibilidade aos potenciais pares exportados.
```

Antes:
```text
O estudo integra projeto aprovado no CEP do IESC-UFRJ e da SMS-Rio (CAAE 43856720.1.0000.5286). O autor recebeu apenas extrato pseudonimizado, sem registros nominais nem correspondência com pessoas. A base legal é a LGPD (arts. 7º, IV, 11, II, c, e 13). Dossiês enviados a APIs externas continham nota, atributos derivados e pseudônimo; nas execuções auditadas, retiraram-se as saídas do motor. Persiste risco de reidentificação.
```

Depois:
```text
O estudo integra projeto aprovado no CEP do IESC-UFRJ (CAAE 43856720.1.0000.5286) e no CEP da SMS-Rio (CAAE 43856720.1.3001.5279). O autor recebeu apenas extrato pseudonimizado, sem registros nominais nem correspondência com pessoas. A base legal é a LGPD (arts. 7º, IV, 11, II, c, e 13). Dossiês enviados a APIs externas continham nota, atributos derivados e pseudônimo; nas execuções auditadas, retiraram-se as saídas do motor. Persiste risco de reidentificação.
```
