# Fatia A: execução e fidelidade

- [x] Ler decisões D1 a D6, reconferência, lista da banca e guia de estilo integral
- [x] Corrigir capítulos, deslocar linkage de origem e conferir fontes locais
- [x] Registrar antes/depois das frases alteradas
- [x] Compilar com jobname separado e verificar diff
- Commit e push: identificador e resultado comunicados na entrega, após as verificações abaixo

## Pendências e limites de evidência

1. **Zenodo (D2): ação externa pendente do autor.** O texto adota o estado de acesso restrito decidido no plano, mediante solicitação e acordo de uso sob CAAE 43856720.1.0000.5286 e 43856720.1.3001.5279. Esta fatia não alterou o registro remoto. `data/zenodo_301_relatorio.md` comprova a publicação histórica pública com MIT, não a futura restrição. As notas de três entradas existentes do `.bib` foram alinhadas à decisão de acesso, para não manter uma declaração contraditória de MIT para dados. Não se criou referência nova nem se mudou DOI, autoria ou resultado.
2. **D6/K5:** não encontrada fonte citável de módulo Python do OpenRecLink nas fontes locais pesquisadas, no `.bib` ou em `data/proveniencia_oliveira_meta.json`. Não se confundiu o CdR Python com um módulo do OpenRecLink. Nenhuma frase sobre a existência desse módulo foi inserida.
3. **C14/McNutt:** incluída a comparação nos EUA, mas não se inventou achado. `.local/fontes/CONFERENCIA_2026-11-04.md:391` registra que nem a atribuição a dois padrões-ouro foi validada em texto integral; o `.bib` confirma apenas identidade/título. Resultado de McNutt permanece sem evidência local suficiente. Quezada-Sánchez tem texto local, com sensibilidade de 90,72% e VPP de 97,10%, cerca de um ponto percentual acima da classificação sem revisão.
4. **C1/C2/lacuna:** explicitada a preservação do corte como prática do serviço e definição de PAR=1, sem superioridade presumida, com remissão à discussão do Estudo 3. A escassez foi delimitada ao levantamento do apêndice, que reconhece cobertura incompleta. Não se atribuiu a Schnell (2023), artigo de microssimulação de registro educacional, evidência não verificada de escassez brasileira.
5. **Objetivo 2/fatia B:** ajustado ao executado: balanceamento em Random Forest e sensibilidade, VPP e F1; retirada a promessa de especificidade e de balanceamento no melhor estimador. Não se inventou especificidade para o Estudo 2.
6. **C10/pré-processamento:** `git show qualificacao-2026:text/latex/capitulo5.tex` confirma o relato da limpeza de nomes e DTNASC2 e sua atribuição a Luís Guilherme. Restaurado somente como informação da qualificação, não documentada em fonte primária de execução; não se restaurou como fato consulta a SIM/Sinan/endereço nem ordenação histórica.
7. **K3/Modal:** os roteiros `scripts/qw1_modal_runner.py:9-11,95-103` e `scripts/qw1_modal_unified_pipeline.py:59,107-108` transferem/leem COMPARADORSEMIDENT.csv. O texto não diz que foi apenas o benchmark de 31 colunas. Provedores do piloto conferidos em `scripts/pilot_c2_517.py:63-84`: Fireworks AI para Kimi, Qwen, DeepSeek e GLM; OpenAI para GPT-4o.
8. **Apêndice de busca e tabela editorial:** não se localizou fixação de hash desses fontes; renomeadas as designações editoriais das consultas para Q1-Q5, com correspondência explícita às chaves históricas O1-O5, preservadas nos dados. Corrigida a nota da tabela de blocagem, sem alterar células numéricas. Nenhum estudo, conclusão, resumo ou slide foi editado por esta fatia.
9. **Introdução/conclusão:** a antiga síntese numérica duplicada foi convertida em anúncio das análises, com remissão ao Estudo 4; não se suprimiram os resultados dos estudos ou da conclusão. Mantidas as ressalvas de exploração, post hoc, verificação seletiva e validação em dados novos. A ressalva de treinamento na prevalência original deixa de ser prescrição na introdução e permanece assunto da discussão/conclusão, não foi negada.

## Verificação

- `python scripts/build_pdf.py --jobname tese_banca_a --outname tese_banca_a`: compilação estabilizada; PDF alternativo, não oficial e não versionado.
- Log de `tese_banca_a`: sem erros LaTeX, citações indefinidas ou referências indefinidas na verificação local.
- Verificação editorial direcionada: ausência de travessões e de palavra taxa nos capítulos e listas de A; presença das definições e do apêndice; horas conferidas em `data/carga_abaixo_corte_horas.csv`; 2.521 conferido no resumo de mesma carga e 2.522 no limiar incluindo empates.
- Busca `rg -n -e '\x{2014}' -e '\x{2013}'` nos sete capítulos/listas de A: sem ocorrências (código 1 normal do ripgrep). A forma com os caracteres literais foi recusada pelo analisador do shell; a busca por códigos Unicode é equivalente.
- `git diff --check`: sem erros na execução anterior; repetido antes do commit.
- Não executada suíte completa; build/PDF oficiais e integração final pertencem à fatia F.

## Atendimento item a item (linhas finais)

- **D1, revisão e viés de incorporação:** `text/latex/capitulo_dados.tex:75`.
- **D2/B9, membro, CEP, acesso e licença:** `text/latex/capitulo_dados.tex:27`.
- **C10, origem em apêndice:** `text/latex/apendice_linkage_origem.tex:1`.
- **C10, pré-processamento da qualificação:** `text/latex/apendice_linkage_origem.tex:40`.
- **K3, provedores e ambiente local:** `text/latex/capitulo_dados.tex:47`.
- **K3, entrada do Modal:** `text/latex/capitulo_dados.tex:115`.
- **C14, experiências de fora e limite McNutt:** `text/latex/capitulo2.tex:81`.
- **K4, raridade/comunalidade:** `text/latex/capitulo2.tex:129`.
- **Objetivo geral e arranjo:** `text/latex/capitulo3.tex:16`.
- **Objetivo 2, alinhamento ao RF:** `text/latex/capitulo3.tex:24`.
- **Objetivo 3, teste inteiro:** `text/latex/capitulo3.tex:26`.
- **Objetivo 4, política e duas fronteiras:** `text/latex/capitulo3.tex:28`.
- **Objetivo 5, raça/cor no extrato:** `text/latex/capitulo3.tex:30`.
- **C1/C2, justificativa do corte:** `text/latex/capitulo3.tex:7`.
- **Introdução, horas hipotéticas:** `text/latex/capitulo1.tex:16`.
- **Introdução, GZ-CMD/P/O/k e modos:** `text/latex/capitulo1.tex:25`.
- **Introdução, anúncio sem síntese duplicada:** `text/latex/capitulo1.tex:31`.
- **Dados, protocolo completo de métricas:** `text/latex/capitulo_dados.tex:97`.
- **Dados, 2.522 vs 2.521:** `text/latex/capitulo_dados.tex:73`.
- **Dados, 120/127 já são PAR=1:** `text/latex/capitulo_dados.tex:86`.
- **Dados, inventário restrito:** `text/latex/capitulo_dados.tex:123`.
- **Siglas, VPN:** `text/latex/siglas.tex:58`.
- **Siglas, GZ-CMD política e F-S removida:** `text/latex/siglas.tex:26`.
- **Glossário, curva de esforço:** `text/latex/glossario.tex:32`.
- **Glossário, corte 8:** `text/latex/glossario.tex:34`.
- **Glossário, P-Platt:** `text/latex/glossario.tex:100`.
- **Glossário, resultado negativo:** `text/latex/glossario.tex:114`.
- **Consultas sem colisão O1-O4:** `text/latex/apendice_busca.tex:16`.
- **Nota da tabela, área cinza da nota:** `text/latex/tables/tab_blocagem_rendimento.tex:19`.
- **Inclusão do apêndice:** `text/latex/tese.tex:674`.

## Fidelidade frase a frase

Base: `53c4a13`. Cada registro contém o trecho anterior e o posterior integral, ou indica inclusão/retirada. A segmentação mantém juntas orações separadas por ponto e vírgula e preserva números decimais. Linhas estruturais LaTeX são registradas quando alteradas. Deslocamentos literais para o novo apêndice não são perdas; as frases restauradas da qualificação são qualificadas expressamente como relato histórico não comprovado. Os vereditos abaixo se referem ao conteúdo da redação, não à execução da ação remota do Zenodo.


### capitulo1.tex

#### A001: `text/latex/capitulo1.tex:9`

**Antes:**

A avaliação da qualidade do cuidado compreende as dimensões de estrutura, processo e resultado \cite{Donabedian1988quality}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A002: `text/latex/capitulo1.tex:16`

**Antes:**

Sua conversão em horas utiliza um parâmetro temporal e fornece ordem de grandeza, sem medir esforço observado.

**Depois:**

Adotando a hipótese de 20 minutos por potencial par, e não tempo observado, esses cenários correspondem a 718\,h\,40\,min e 840\,h\,40\,min, respectivamente (\path{data/carga_abaixo_corte_horas.csv}); o parâmetro é uma hipótese de planejamento, não uma medição da revisão histórica.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A003: `text/latex/capitulo1.tex:18`

**Antes:**

No \textit{linkage} entre o SIM e o Sinan-TB, condição marcadora (\textit{tracer condition}) da qualidade do cuidado em saúde, os indicadores de mortalidade evidenciam subnotificação significativa \cite{Oliveira2012uso, Sousa2011obitos, Pinheiro2012subnotificacao}.

**Depois:**

No \textit{linkage} entre o SIM e o Sinan-TB, condição marcadora (\textit{tracer condition}) da qualidade do cuidado em saúde, os indicadores de mortalidade evidenciam subnotificação expressiva de óbitos por tuberculose \cite{Oliveira2012uso, Sousa2011obitos, Pinheiro2012subnotificacao}.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A004: `text/latex/capitulo1.tex:20`

**Antes:**

Embora a aplicação de técnicas de aprendizado de máquina (\textit{machine learning}) ao \textit{linkage} venha sendo investigada em contextos internacionais \cite{Binette2022entity, Vo2019ensemble, Jiao2021hybrid, Almadani2026linking}, a literatura brasileira disponível sobre o tema sugere que a produção nacional é ainda incipiente, restringindo-se predominantemente a abordagens determinísticas e probabilísticas tradicionais \cite{Coeli2021suboptimal}.

**Depois:**

Embora a aplicação de técnicas de aprendizado de máquina (\textit{machine learning}) ao \textit{linkage} venha sendo investigada em contextos internacionais \cite{Binette2022entity, Vo2019ensemble, Jiao2021hybrid, Almadani2026linking}, há experiências brasileiras de classificação supervisionada, entre elas as de \citeonline{Pita2017trainable}, \citeonline{Pereira2024supervised}, \citeonline{Paiva2025comparador} e \citeonline{Lages2026trauma}, discutidas na Seção~\ref{sec:ml-classificacao-pares}, sem que sua existência dispense a avaliação do pós-processamento sob a referência e o desbalanceamento deste extrato.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A005: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O GZ-CMD é uma política de decisão por \textit{expected loss} que compara custos de falso negativo ($C_{FN}$), falso positivo ($C_{FP}$), revisão por modelo de linguagem de grande porte (LLM; $C_{LLM}$) e revisão humana ($C_H$), com guardas determinísticas, calibração fora da amostra (OOF) agrupada por REFREC e revisão assistida.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A006: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

P designa o arranjo registrado antes da avaliação, com preservação do corte do OpenRecLink e recalibração isotônica restrita à região abaixo dele.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A007: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O corte é preservado por representar a prática do serviço e definir PAR igual a 1 na referência, não por uma demonstração de superioridade sobre regras determinísticas; essa escolha é retomada na discussão do Estudo~3.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A008: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Vigilância prioriza sensibilidade e redução das perdas de vínculos; confirmação prioriza valor preditivo positivo e redução de falsos vínculos.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A009: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O1 ordena pelo escore do OpenRecLink; O2, pela nota do CdR; O3, pela probabilidade calibrada OOF do classificador; O4, pela probabilidade recalibrada de P.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A010: `text/latex/capitulo1.tex:25`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O orçamento $k$ é o número de potenciais pares encaminhados à revisão na ordenação considerada (Glossário).

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A011: `text/latex/capitulo1.tex:31`

**Antes:**

As contribuições compreendem a decomposição por PAR, que distingue a preservação dos vínculos definidos pelo corte do ganho sobre os recuperados pela revisão; a curva de esforço aplicada a este fluxo como instrumento de escolha da ordenação e do ponto de parada, sem reivindicar originalidade da ordenação por pesos ou da revisão clerical descritas por \citeonline{Sariyar2010recordlinkage}; e a investigação do LLM como aceitador e como filtro de descarte antes do revisor humano, associada à investigação de sua ancoragem.

**Depois:**

Os estudos examinam a decomposição por PAR, a curva de esforço para escolha da ordenação e do ponto de parada e o LLM como aceitador e como filtro antes da revisão humana, incluindo a investigação de sua ancoragem.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A012: `text/latex/capitulo1.tex:31`

**Antes:**

Nesta última, o VPP de 16,0\% e os 9,06 FP previstos contra 21 observados são resultados \textit{post hoc}; como filtro de \texttt{NONMATCH}, o mesmo revisor apresentou VPN de 87,8\% na vigilância, em $k=96$; em outro orçamento, $k=125$, descartou 10 dos 23 vínculos.

**Depois:**

Não se reivindica originalidade da ordenação por pesos ou da revisão clerical descritas por \citeonline{Sariyar2010recordlinkage}.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A013: `text/latex/capitulo1.tex:31`

**Antes:**

A utilidade do filtro é exploratória, não validação de sua adoção.

**Depois:**

O Estudo~4 apresenta o VPP do LLM, a comparação entre falsos positivos previstos e observados e os resultados do filtro de \texttt{NONMATCH} em diferentes orçamentos, preservando seu caráter exploratório e \textit{post hoc}, sem validar sua adoção (Seções~\ref{sec:e4-mesma-carga} e~\ref{sec:e4-filtro}).

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A014: `text/latex/capitulo1.tex:31`

**Antes:**

Completam as contribuições a política auditável, o CdR em Python, o pacote e o \textit{benchmark} abertos e o resultado negativo com registro interno (Seções~\ref{sec:e4-mesma-carga} e~\ref{sec:e4-filtro}) \cite{Jardim2026phdrepo}.

**Depois:**

O percurso inclui a política auditável, o CdR em Python, o pacote de código e os resultados agregados abertos, o \textit{benchmark} de acesso restrito e o resultado negativo com registro interno, isto é, o arranjo P, registrado antes da avaliação, não superou a referência do serviço \cite{Jardim2026phdrepo}.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A015: `text/latex/capitulo1.tex:35`

**Antes:**

A curva de esforço relaciona o orçamento humano aos vínculos PAR~=~2 recuperados, permitindo escolher a ordenação e o ponto de parada sem presumir uma carga histórica.

**Depois:**

O Estudo~4 examina a curva de esforço como instrumento de escolha da ordenação e do ponto de parada segundo o orçamento humano, incluindo O1 a O4, a investigação de subamostragem e as comparações na mesma carga.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A016: `text/latex/capitulo1.tex:35`

**Antes:**

O3 utiliza a probabilidade calibrada fora da amostra, e O4 a probabilidade recalibrada de P; ambas provêm do mesmo XGBoost.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A017: `text/latex/capitulo1.tex:35`

**Antes:**

A recomendação gerada pela avaliação é preservar o corte e, abaixo dele, revisar em ordem decrescente de uma probabilidade ou escore, escolhendo a ordenação e o ponto de parada pela curva de esforço conforme o orçamento humano.

**Depois:**

As análises geraram a hipótese de reduzir esforço pela ordenação, não uma demonstração de superioridade da política; a subamostragem não melhorou de forma estável o topo da fila e a calibração (Seção~\ref{sec:e4-subamostragem}).

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A018: `text/latex/capitulo1.tex:35`

**Antes:**

Nenhuma ordenação domina em todas as cargas.

**Depois:**

A discussão das implicações operacionais e da necessidade de validação registrada em dados novos é retomada na conclusão \cite{Jardim2026phdrepo}.

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A019: `text/latex/capitulo1.tex:35`

**Antes:**

Se a ordenação usar probabilidade aprendida, o treinamento deve manter a prevalência original.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.

#### A020: `text/latex/capitulo1.tex:35`

**Antes:**

A subamostragem não melhorou de forma estável o topo da fila e a calibração (Seção~\ref{sec:e4-subamostragem}); a adoção depende de validação registrada em dados novos \cite{Jardim2026phdrepo}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: definição/qualificação ou anúncio no lugar de síntese; resultados e limites remetidos ao Estudo 4/conclusão, sem reivindicar superioridade. Horas são hipótese, não tempo observado.


### capitulo2.tex

#### A021: `text/latex/capitulo2.tex:61`

**Antes:**

A técnica apresenta robustez ao sobreajuste (\textit{overfitting}) e mecanismos intrínsecos para avaliação da importância de variáveis.

**Depois:**

A agregação de árvores reduz a variância, sem eliminar o risco de sobreajuste, e permite avaliar a importância das variáveis \cite{Breiman2001rf}.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A022: `text/latex/capitulo2.tex:81`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Nos Estados Unidos, \citeonline{McNutt2020comparison} compararam aprendizado supervisionado e métodos probabilísticos de \textit{linkage}; a referência bibliográfica e o levantamento de busca permitem identificar essa comparação, mas não sustentam aqui a descrição de seus resultados nem a atribuição a dois padrões-ouro, não conferida em texto integral.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A023: `text/latex/capitulo2.tex:81`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

No México, \citeonline{QuezadaSanchez2022probabilistic} implementaram e validaram o modelo de Fellegi e Sunter em registros de altas hospitalares e óbitos, obtendo sensibilidade de 90,72\% e VPP de 97,10\% na validação, cerca de um ponto percentual acima da classificação automática sem revisão clerical.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A024: `text/latex/capitulo2.tex:81`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Esses resultados se referem à classificação probabilística com e sem revisão, não a um classificador supervisionado.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A025: `text/latex/capitulo2.tex:90`

**Antes:**

A base desta tese tem a proporção natural de cerca de 1:249 entre vínculos e não pares, mantida no treinamento dos classificadores sem reamostragem e modificada nas estratégias de balanceamento, como o Random Forest com SMOTE (Seção~\ref{sec:analise-comparativa}), e a contribuição proposta reside na combinação desses classificadores com uma cascata auditável de regras de guarda, calibração de probabilidades, decisão por \textit{expected loss} e revisão assistida por modelo de linguagem (Capítulo~\ref{cap:estudo4}), que dialoga com a decisão com opção de abstenção em \textit{linkage} \cite{Sadinle2017}.

**Depois:**

A base desta tese tem a proporção natural de cerca de 1:249 entre vínculos e não pares, mantida no treinamento dos classificadores sem reamostragem e modificada nas estratégias de balanceamento, como o Random Forest com SMOTE (Seção~\ref{sec:analise-comparativa}), e a contribuição proposta reside na combinação desses classificadores com uma política auditável com regras de guarda, calibração de probabilidades, decisão por \textit{expected loss} e revisão assistida por modelo de linguagem (Capítulo~\ref{cap:estudo4}), que dialoga com a decisão com opção de abstenção em \textit{linkage} \cite{Sadinle2017}.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A026: `text/latex/capitulo2.tex:112`

**Antes:**

Esta tese & SIM e Sinan-TB & Conjunto comparado no Capítulo~\ref{cap:estudo2} & Resultados nos Capítulos~\ref{cap:estudo2} a~\ref{cap:estudo4} & Cascata com regras de guarda, \textit{expected loss} e revisão assistida por modelo de linguagem \\

**Depois:**

Esta tese & SIM e Sinan-TB & Conjunto comparado no Capítulo~\ref{cap:estudo2} & Resultados nos Capítulos~\ref{cap:estudo2} a~\ref{cap:estudo4} & Política com regras de guarda, \textit{expected loss} e revisão assistida por modelo de linguagem \\

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.

#### A027: `text/latex/capitulo2.tex:129`

**Antes:**

Os critérios contemplam medidas que quantificam o grau de igualdade e de similaridade entre os nomes, a frequência dos fragmentos nas tabelas de nomes, as abreviações, as diferenças entre as datas de nascimento e a igualdade dos endereços de residência.

**Depois:**

Os critérios contemplam medidas que quantificam o grau de igualdade e de similaridade entre os nomes, a raridade ou comunalidade dos fragmentos, segundo sua frequência nas tabelas de nomes, as abreviações, as diferenças entre as datas de nascimento e a igualdade dos endereços de residência.

**Veredito:** Fiel às fontes locais citadas: não se atribuem resultados a McNutt sem texto; Quezada-Sánchez tem números conferidos; ajustes conceituais não criam resultado desta tese.


### capitulo3.tex

#### A028: `text/latex/capitulo3.tex:5`

**Antes:**

Estudos anteriores demonstraram que o \textit{linkage} (vinculação de registros) entre essas duas bases permite identificar óbitos por TB não notificados ao sistema de vigilância, evidenciando subnotificação significativa \cite{Sousa2011obitos, Pinheiro2012subnotificacao, Rocha2015causas}, com impacto direto sobre a análise de causas múltiplas de óbito em coortes de pacientes com TB \cite{Rocha2015causas}.

**Depois:**

Estudos anteriores demonstraram que o \textit{linkage} (vinculação de registros) entre essas duas bases permite identificar óbitos por TB não notificados ao sistema de vigilância, evidenciando subnotificação de óbitos por tuberculose \cite{Sousa2011obitos, Pinheiro2012subnotificacao, Rocha2015causas}, com impacto direto sobre a análise de causas múltiplas de óbito em coortes de pacientes com TB \cite{Rocha2015causas}.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A029: `text/latex/capitulo3.tex:7`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Preserva-se o corte por representar a prática do serviço e definir PAR igual a 1 na referência, sem presumir sua superioridade sobre uma estratégia determinística; a comparação é retomada na discussão do Estudo~3.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A030: `text/latex/capitulo3.tex:9`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

No levantamento documentado no Apêndice~\ref{ap:protocolo-busca}, são escassos os estudos brasileiros de classificadores supervisionados como pós-processamento do \textit{linkage} probabilístico; essa constatação se limita ao corpus recuperado, cuja cobertura é incompleta, e não implica ausência de experiências nacionais.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A031: `text/latex/capitulo3.tex:16`

**Antes:**

O presente estudo tem como objetivo geral avaliar o pós-processamento dos potenciais pares não aceitos pelo corte do OpenRecLink no \textit{linkage}, também denominado relacionamento de registros (\textit{record linkage}), entre SIM e Sinan-TB, comparando ordenação, calibração e política auditável de decisão quanto a recuperação de vínculos, erros e revisão humana.

**Depois:**

O presente estudo tem como objetivo geral avaliar o pós-processamento do \textit{linkage}, também denominado relacionamento de registros (\textit{record linkage}), entre SIM e Sinan-TB, abrangendo a reimplementação do comparador, os classificadores supervisionados, as estratégias operacionais e a equidade das decisões na base completa de potenciais pares, e a ordenação, a calibração e a política auditável de decisão na região não aceita pelo corte do OpenRecLink, quanto à recuperação de vínculos, aos erros e à revisão humana.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A032: `text/latex/capitulo3.tex:16`

**Antes:**

A avaliação distingue resultados do arranjo sequencial, comparações exploratórias de mesma carga e cenários de sensibilidade humana, sem pressupor superioridade da política nem transferência dos parâmetros a outras bases.

**Depois:**

A avaliação distingue resultados do arranjo sequencial, que preserva os vínculos aceitos pelo corte e aplica o GZ-CMD aos demais potenciais pares, comparações exploratórias de mesma carga e cenários de sensibilidade humana, sem pressupor superioridade da política nem transferência dos parâmetros a outras bases.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A033: `text/latex/capitulo3.tex:24`

**Antes:**

\item Avaliar o desempenho de classificadores supervisionados treinados na base completa de potenciais pares, sob o desbalanceamento extremo próprio do \textit{linkage}, com desempenho estratificado por faixa do escore, e comparar estratégias de balanceamento de classes aplicadas ao melhor estimador, quanto ao efeito sobre a sensibilidade e a especificidade, de modo a escolher e a testar a estabilidade do estimador de probabilidade que sustenta a classificação.

**Depois:**

\item Avaliar o desempenho de classificadores supervisionados treinados na base completa de potenciais pares, sob o desbalanceamento extremo próprio do \textit{linkage}, com desempenho estratificado por faixa do escore, e comparar estratégias de balanceamento de classes aplicadas ao Random Forest, quanto ao efeito sobre a sensibilidade, o valor preditivo positivo e o F$_1$-Score, descrevendo a estabilidade interna dos classificadores avaliados.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A034: `text/latex/capitulo3.tex:26`

**Antes:**

\item Derivar e comparar duas estratégias operacionais de pós-processamento, uma orientada à sensibilidade, voltada à vigilância e à recuperação exaustiva de pares, e outra orientada à precisão, voltada à confirmação e à construção de conjuntos analíticos de alta confiabilidade, por meio da otimização de limiares e de regras de domínio, caracterizando a fronteira de Pareto entre elas e estimando, sob cada estratégia, o impacto epidemiológico da recuperação de vínculos da área cinza sobre a detecção de óbitos em pessoas com tuberculose no município do Rio de Janeiro e o custo da revisão manual.

**Depois:**

\item Derivar e comparar duas estratégias operacionais de pós-processamento na base completa de potenciais pares, uma orientada à sensibilidade, voltada à vigilância e à recuperação exaustiva de pares, e outra orientada ao valor preditivo positivo, voltada à confirmação e à construção de conjuntos analíticos de alta confiabilidade, por meio da otimização de limiares e de regras de domínio, caracterizando a fronteira de Pareto entre elas e estimando, no conjunto de teste inteiro, o impacto epidemiológico da recuperação de vínculos sobre a detecção de óbitos em pessoas com tuberculose no município do Rio de Janeiro e o custo da revisão manual.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A035: `text/latex/capitulo3.tex:28`

**Antes:**

\item Desenvolver e avaliar o GZ-CMD, cascata de decisão por \textit{expected loss} com custos assimétricos, calibração fora da amostra e revisão assistida por modelo de linguagem de grande porte, aplicada aos potenciais pares não aceitos pelo corte do OpenRecLink.

**Depois:**

\item Desenvolver e avaliar o GZ-CMD, política de decisão por \textit{expected loss} com custos assimétricos, calibração fora da amostra e revisão assistida por modelo de linguagem de grande porte, aplicada aos potenciais pares não aceitos pelo corte do OpenRecLink.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A036: `text/latex/capitulo3.tex:28`

**Antes:**

Comparar vínculos recuperados, falsos positivos, encaminhamentos ao revisor humano, custo e carga de revisão com a prática do serviço (corte seguido de revisão manual) e com estratégias de corte único e de duas fronteiras, incluindo comparação exploratória com ordenação calibrada na mesma carga e curva de esforço.

**Depois:**

Comparar vínculos recuperados, falsos positivos, encaminhamentos ao revisor humano, custo e carga de revisão com a prática do serviço (corte seguido de revisão manual) e com estratégias de corte único e de duas fronteiras, estas definidas por um limiar inferior de rejeição e um superior de aceitação, com revisão entre ambos, incluindo comparação exploratória com ordenação calibrada na mesma carga e curva de esforço.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.

#### A037: `text/latex/capitulo3.tex:30`

**Antes:**

\item Avaliar o desempenho das decisões por subgrupo, quanto à paridade de sensibilidade e de valor preditivo positivo, explicitando a incompatibilidade geral desses critérios sob prevalências desiguais, distinguindo subgrupos demográficos de subgrupos operacionais, como o quartil de concordância do nome, o passo de blocagem e a situação de encerramento da notificação, e registrando a ausência da variável raça/cor nas bases utilizadas.

**Depois:**

\item Avaliar o desempenho das decisões por subgrupo na base completa de potenciais pares, quanto à paridade de sensibilidade e de valor preditivo positivo, explicitando a incompatibilidade geral desses critérios sob prevalências desiguais, distinguindo subgrupos demográficos de subgrupos operacionais, como o quartil de concordância do nome, o passo de blocagem e a situação de encerramento da notificação, e registrando a ausência da variável raça/cor no extrato utilizado.

**Veredito:** Fiel ao escopo executado: base inteira nos objetivos 1/2/3/5, região abaixo do corte no 4; RF e métricas disponíveis no 2; não promete especificidade não reportada.


### capitulo_dados.tex

#### A038: `text/latex/capitulo_dados.tex:5`

**Antes:**

O estudo utiliza dados secundários provenientes de sistemas nacionais de informação em saúde, vinculados por meio de técnicas probabilísticas, e propõe protocolos computacionais para melhorar a acurácia na classificação de potenciais pares, com vistas a qualificar os dados vinculados e produzir indicadores para a vigilância epidemiológica.

**Depois:**

O estudo utiliza dados secundários provenientes de sistemas nacionais de informação em saúde, vinculados por meio de técnicas probabilísticas, e avalia protocolos computacionais quanto à acurácia na classificação de potenciais pares, com vistas a qualificar os dados vinculados e produzir indicadores para a vigilância epidemiológica.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A039: `text/latex/capitulo_dados.tex:19`

**Antes:**

O extrato provém do processo de \textit{linkage} descrito por \citeonline{Oliveira2026obitoTBpulmonar}, entre os óbitos de residentes no município do Rio de Janeiro com tuberculose como causa básica (SIM, 2019 a 2022) e as notificações de tuberculose (Sinan, 2013 a 2022), em quatro etapas: \textit{linkage} determinístico, 29 passos probabilísticos no OpenRecLink, um passo aberto e deduplicação (p.~33 a~39).

**Depois:**

O extrato provém do processo descrito por \citeonline{Oliveira2026obitoTBpulmonar}, entre óbitos de residentes no Rio de Janeiro com tuberculose como causa básica (SIM, 2019 a 2022) e notificações de tuberculose (Sinan, 2013 a 2022).

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A040: `text/latex/capitulo_dados.tex:19`

**Antes:**

Os vínculos encontrados na etapa determinística foram retirados antes do \textit{linkage} probabilístico, de modo que os potenciais pares do extrato correspondem ao que restou dessa etapa.

**Depois:**

Reúne três passos probabilísticos exportados (6, 9 e 11), após a retirada dos vínculos determinísticos, e não representa todo o processo.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A041: `text/latex/capitulo_dados.tex:19`

**Antes:**

O extrato reúne os potenciais pares de três dos 29 passos probabilísticos (6, 9 e 11), cujos parâmetros, pontos de corte (7,0240; 7,8444; 7,9019) e contagens de vínculos pelo corte e após a revisão coincidem com os descritos na monografia (Quadro~5, p.~36 a~37).

**Depois:**

A proveniência, os parâmetros, os cortes e os procedimentos de origem são apresentados no Apêndice~\ref{ap:linkage-origem}; o órgão fornecedor, a data de extração e o critério de escolha dos passos não estão documentados.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A042: `text/latex/capitulo_dados.tex:19`

**Antes:**

Esses três passos contribuíram com 247 dos 2.549 pares do processo completo (p.~35); a proporção de 1:249 refere-se, portanto, à base exportada.

**Depois:**

A escolha das bases decorre da tuberculose como condição marcadora e da experiência do grupo com essas fontes (Seção~\ref{sec:justificativas-especificas}) \cite{Oliveira2012uso,Bartholomay2014improved,Oliveira2016accuracy}.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A043: `text/latex/capitulo_dados.tex:19`

**Antes:**

A correspondência documental e as decisões de proveniência informadas pelo autor estão registradas em \texttt{data/proveniencia\_oliveira\_meta.json}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A044: `text/latex/capitulo_dados.tex:19`

**Antes:**

O órgão fornecedor das bases, a data da extração e o critério de escolha dos três passos exportados não constam da documentação disponível ao autor.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A045: `text/latex/capitulo_dados.tex:19`

**Antes:**

A escolha dessas bases justifica-se pelas razões detalhadas na Seção \ref{sec:justificativas-especificas}, em particular a relevância da tuberculose como condição marcadora da qualidade do cuidado em saúde e a experiência acumulada do grupo de pesquisa no \textit{linkage} dessas fontes de dados \cite{Oliveira2012uso, Bartholomay2014improved, Oliveira2016accuracy}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A046: `text/latex/capitulo_dados.tex:23`

**Antes:**

O procedimento de \textit{linkage} probabilístico no OpenRecLink seguiu o modelo de decisão de Fellegi e Sunter \citeyearpar{Fellegi1969theory}.

**Depois:**

O modelo de Fellegi e Sunter constitui o referencial do escore probabilístico \cite{Fellegi1969theory}; o termo operacional adotado para a classificação por corte é decisão de limiar.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A047: `text/latex/capitulo_dados.tex:23`

**Antes:**

Na etapa de padronização, os campos textuais de ambas as bases foram convertidos para caixa alta, com remoção de acentos diacríticos, preposições e artigos, e as datas de nascimento foram normalizadas para formato numérico contínuo (AAAAMMDD).

**Depois:**

Distinguem-se o escore do OpenRecLink e a nota final do CdR, derivada de 29 subescores de similaridade.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A048: `text/latex/capitulo_dados.tex:23`

**Antes:**

A etapa de comparação no OpenRecLink utilizou três campos de identificação: nome próprio e nome da mãe, comparados por algoritmo aproximado (tipo APROX, com limiar de similaridade de 85\%); e data de nascimento, comparada caractere a caractere (tipo CARACTERE, com limiar de 65\%).

**Depois:**

Área cinza da nota designa o intervalo $[5,9)$ do CdR, que não coincide com a região abaixo do corte do OpenRecLink.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A049: `text/latex/capitulo_dados.tex:23`

**Antes:**

A variável sexo não integrou a etapa de comparação, tendo sido utilizada exclusivamente como chave de blocagem em passos específicos.

**Depois:**

Os parâmetros de comparação e a padronização de origem encontram-se no Apêndice~\ref{ap:linkage-origem}.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A050: `text/latex/capitulo_dados.tex:23`

**Antes:**

Os parâmetros de concordância e discordância atribuídos a cada campo estão detalhados na Tabela~\ref{tab:parametros-orl}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A051: `text/latex/capitulo_dados.tex:23`

**Antes:**

Para cada potencial par, o programa calculou um escore probabilístico correspondente à soma das razões de log-verossimilhança (\textit{log-likelihood ratios}) dos campos comparados, no qual campos concordantes contribuem positivamente e campos discordantes contribuem negativamente para o escore total \cite{Fellegi1969theory, Christen2012book}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A052: `text/latex/capitulo_dados.tex:23`

**Antes:**

Na base utilizada, os escores probabilísticos do OpenRecLink variaram de $-12{,}85$ (discordância em todos os campos) a $+15{,}58$ (concordância plena), distribuindo-se em 116 valores distintos.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A053: `text/latex/capitulo_dados.tex:23`

**Antes:**

Para evitar ambiguidade, o presente trabalho distingue duas grandezas: o escore do \textit{linkage} probabilístico, calculado pelo OpenRecLink como soma das razões de log-verossimilhança, e a nota final do Comparador de Registros (CdR; Seção~\ref{sec:comparador-registros}), escore composto derivado dos 29 subescores de similaridade campo a campo.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A054: `text/latex/capitulo_dados.tex:23`

**Antes:**

Na origem, a revisão manual incidiu sobre a região abaixo do corte do escore OpenRecLink \cite{Oliveira2026obitoTBpulmonar}; nos resultados desta tese, área cinza da nota designa o intervalo da nota final do CdR em $[5,9)$ (Seção~\ref{sec:comparador-registros}).

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A055: `text/latex/capitulo_dados.tex:23`

**Antes:**

As duas regiões não coincidem.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A056: `text/latex/capitulo_dados.tex:23`

**Antes:**

\begin{table}[!ht]

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A057: `text/latex/capitulo_dados.tex:23`

**Antes:**

\centering

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A058: `text/latex/capitulo_dados.tex:23`

**Antes:**

\caption{Parâmetros de comparação configurados no OpenRecLink para os campos de identificação.}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A059: `text/latex/capitulo_dados.tex:23`

**Antes:**

\label{tab:parametros-orl}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A060: `text/latex/capitulo_dados.tex:23`

**Antes:**

\begin{tabular}{

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A061: `text/latex/capitulo_dados.tex:23`

**Antes:**

p{\dimexpr0.32\linewidth-2\tabcolsep\relax}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A062: `text/latex/capitulo_dados.tex:23`

**Antes:**

>{\centering\arraybackslash}p{\dimexpr0.16\linewidth-2\tabcolsep\relax}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A063: `text/latex/capitulo_dados.tex:23`

**Antes:**

*{2}{>{\centering\arraybackslash}p{\dimexpr0.20\linewidth-2\tabcolsep\relax}}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A064: `text/latex/capitulo_dados.tex:23`

**Antes:**

>{\centering\arraybackslash}p{\dimexpr0.12\linewidth-2\tabcolsep\relax}}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A065: `text/latex/capitulo_dados.tex:23`

**Antes:**

\hline

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A066: `text/latex/capitulo_dados.tex:23`

**Antes:**

\textbf{Campo} & \textbf{Tipo} & \textbf{Concordância (\%)} & \textbf{Discordância (\%)} & \textbf{Limiar (\%)} \\

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A067: `text/latex/capitulo_dados.tex:23`

**Antes:**

\hline

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A068: `text/latex/capitulo_dados.tex:23`

**Antes:**

Nome (NOME) & Aproximado & 99 & 1 & 85 \\

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A069: `text/latex/capitulo_dados.tex:23`

**Antes:**

Nome da mãe (NOMEMAE) & Aproximado & 75 & 2 & 85 \\

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A070: `text/latex/capitulo_dados.tex:23`

**Antes:**

Data de nascimento (DTNASC) & Caractere & 95 & 5 & 65 \\

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A071: `text/latex/capitulo_dados.tex:23`

**Antes:**

\hline

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A072: `text/latex/capitulo_dados.tex:23`

**Antes:**

\end{tabular}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A073: `text/latex/capitulo_dados.tex:23`

**Antes:**

\end{table}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A074: `text/latex/capitulo_dados.tex:27`

**Antes:**

O \textit{linkage} probabilístico e a produção da base de potenciais pares foram realizados pela equipe do IESC-UFRJ, cujos profissionais são nomeados na Seção~\ref{sec:padrao-ouro}.

**Depois:**

O \textit{linkage} probabilístico e a produção da base de potenciais pares foram realizados no IESC-UFRJ; a autoria e os procedimentos de origem são apresentados no Apêndice~\ref{ap:linkage-origem}.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A075: `text/latex/capitulo_dados.tex:27`

**Antes:**

O autor confirmou que o parecer do CEP abrange o \textit{benchmark} público.

**Depois:**

O autor integra o LinkDataPop (IESC/UFRJ) no projeto desses CAAE.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A076: `text/latex/capitulo_dados.tex:27`

**Antes:**

A revisão clerical que originou o campo PAR foi feita pela epidemiologista nomeada na Seção~\ref{sec:padrao-ouro}.

**Depois:**

Os dados por potencial par não são publicizados: o depósito Zenodo é de acesso restrito, mediante solicitação e acordo de uso, sob os pareceres vinculados aos CAAE 43856720.1.0000.5286 e 43856720.1.3001.5279.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A077: `text/latex/capitulo_dados.tex:27`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

A divulgação pública limita-se aos resultados agregados nos CSVs do repositório; os dados estão sujeitos a termo de uso, e a licença MIT se aplica somente ao código a que estiver atribuída.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A078: `text/latex/capitulo_dados.tex:27`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

A revisão clerical que originou PAR foi realizada pela única revisora identificada no Apêndice~\ref{ap:linkage-origem}.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A079: `text/latex/capitulo_dados.tex:27`

**Antes:**

O autor recebeu um extrato sem nomes, nome da mãe ou endereço nominal, mas que não é anônimo.

**Depois:**

O autor não participou do \textit{linkage} de origem, da execução do CdR ou da revisão clerical e, como membro do laboratório, recebeu o extrato pós-CdR, sem nomes, nome da mãe ou endereço nominal, mas não anônimo.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A080: `text/latex/capitulo_dados.tex:29`

**Antes:**

O autor trabalhou exclusivamente com o extrato pseudonimizado produzido pela equipe do laboratório, sem acesso aos registros nominais nem à correspondência entre os identificadores internos e as pessoas.

**Depois:**

O autor trabalhou exclusivamente com o extrato pseudonimizado preparado no laboratório, sem acesso aos registros nominais nem à correspondência entre os identificadores internos e as pessoas.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A081: `text/latex/capitulo_dados.tex:35`

**Antes:**

O \texttt{pair\_id} interno, obtido por SHA-256 sem sal e truncado em 16 caracteres hexadecimais, foi mantido para compatibilidade com os artefatos publicados na versão 3.0.0 do depósito, incluindo os \textit{labels} por potencial par \cite{Jardim2026zenodoGzcmd}.

**Depois:**

O \texttt{pair\_id} interno, obtido por SHA-256 sem sal e truncado em 16 caracteres hexadecimais, foi mantido para compatibilidade com os artefatos históricos da versão 3.0.0 do depósito, agora de acesso restrito, incluindo os \textit{labels} por potencial par \cite{Jardim2026zenodoGzcmd}.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A082: `text/latex/capitulo_dados.tex:47`

**Antes:**

O processamento dos dossiês pelos modelos de linguagem não ocorreu em ambiente local: os dossiês foram enviados a interfaces de programação (APIs) externas (Anthropic, OpenCode Go e Levanto).

**Depois:**

O processamento dos dossiês pelos modelos de linguagem não ocorreu em ambiente local: os dossiês foram enviados a interfaces de programação (APIs) externas (Anthropic, OpenCode Go, Levanto, Fireworks AI e OpenAI).

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A083: `text/latex/capitulo_dados.tex:47`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

No piloto, Kimi K2.5, DeepSeek V3.2, Qwen3-VL-30B e GLM-5 foram acessados via Fireworks AI, e GPT-4o via OpenAI, conforme \path{scripts/pilot_c2_517.py}; o Kimi K2.5 da execução histórica também utilizou Fireworks AI (Seção~\ref{sec:e4-protocolo-llm}).

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A084: `text/latex/capitulo_dados.tex:47`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Recomenda-se ambiente local e controlado para dados identificados, sem envio a APIs externas.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A085: `text/latex/capitulo_dados.tex:51`

**Antes:**

A geração de potenciais pares no OpenRecLink empregou a técnica de blocagem (\textit{blocking}), que particiona os registros em subgrupos definidos por chaves comuns e restringe as comparações aos pares que compartilham a mesma chave, reduzindo substancialmente o custo computacional \cite{Christen2012book}.

**Depois:**

Os potenciais pares do extrato foram gerados por blocagem no OpenRecLink, nos passos 6, 9 e 11, com exclusão progressiva dos registros do Sinan-TB já pareados.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A086: `text/latex/capitulo_dados.tex:51`

**Antes:**

No ORL, os parâmetros de blocagem foram definidos por combinações de variáveis fonéticas e demográficas (código \textit{Soundex} do primeiro e do último bloco do nome, sexo, ano de nascimento, primeiro e último bloco do nome da mãe, código do município de residência e bairro de residência (BAIRES)), organizadas em passos sequenciais.

**Depois:**

As chaves e a sequência de origem estão no Apêndice~\ref{ap:linkage-origem}; o rendimento dos passos exportados é apresentado na Tabela~\ref{tab:blocagem-rendimento}.

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A087: `text/latex/capitulo_dados.tex:51`

**Antes:**

Adotou-se, ainda, a estratégia de exclusão progressiva dos registros do Sinan-TB já pareados em cada passo, de modo que, uma vez identificado um potencial par, o respectivo registro do Sinan não retornasse ao universo de busca dos passos subsequentes.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A088: `text/latex/capitulo_dados.tex:51`

**Antes:**

Foram utilizados múltiplos passos de blocagem (\textit{blocking steps}), estratégia recomendada para maximizar a sensibilidade do processo ao permitir que pares não capturados por uma chave restritiva fossem recuperados por chaves alternativas \cite{Coeli2002blocking}.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A089: `text/latex/capitulo_dados.tex:51`

**Antes:**

O procedimento incluiu 29 passos probabilísticos e um passo aberto, cuja chave de blocagem foi o \textit{Soundex} do primeiro fragmento do nome do paciente (PBLOCO), conforme a descrição metodológica e o Quadro~6 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~34 e~39).

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A090: `text/latex/capitulo_dados.tex:51`

**Antes:**

Três dos passos probabilísticos integram o extrato, sem informação sobre o critério de escolha para exportação.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A091: `text/latex/capitulo_dados.tex:51`

**Antes:**

Os três passos exportados, com suas respectivas chaves de blocagem, foram:

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A092: `text/latex/capitulo_dados.tex:51`

**Antes:**

\begin{enumerate}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A093: `text/latex/capitulo_dados.tex:51`

**Antes:**

\item \textbf{Passo por nome e nome da mãe (passo~6):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) e do primeiro e do último bloco do nome da mãe (PBLOCOMAE, UBLOCOMAE), gerando 2.649 potenciais pares, dos quais 80 foram classificados como pares (76 pelo ponto de corte e 4 em revisão manual);

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A094: `text/latex/capitulo_dados.tex:51`

**Antes:**

\item \textbf{Passo restritivo (passo~9):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + ano de nascimento (ANONASC) + sexo (SEXO), gerando 1.449 potenciais pares, dos quais 127 foram classificados como pares (120 pelo ponto de corte e 7 em revisão manual);

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A095: `text/latex/capitulo_dados.tex:51`

**Antes:**

\item \textbf{Passo ampliado (passo~11):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + sexo (SEXO) + código do município de residência (CODMUNRES), gerando 57.598 potenciais pares, dos quais 40 foram classificados como pares (15 pelo ponto de corte e 25 em revisão manual).

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A096: `text/latex/capitulo_dados.tex:51`

**Antes:**

\end{enumerate}

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A097: `text/latex/capitulo_dados.tex:55`

**Antes:**

Na origem, o CdR foi usado na revisão do passo aberto, que recuperou 5 pares \cite{Oliveira2026obitoTBpulmonar} (p.~39); o cálculo da nota final sobre os potenciais pares desta tese foi outra execução, com a implementação em Python.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: conteúdo preservado no novo apêndice; o corpo recebe síntese/remissão.

#### A098: `text/latex/capitulo_dados.tex:73`

**Antes:**

Os cortes, 7,0240; 7,8444; 7,9019, são os menores valores do escore do OpenRecLink entre PAR igual a 1 nos passos 6, 9 e 11, respectivamente, e coincidem até a quarta casa decimal com o Quadro~5 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~36 a~38).

**Depois:**

Os cortes e a correspondência com o Quadro~5 de \citeonline{Oliveira2026obitoTBpulmonar} estão no Apêndice~\ref{ap:linkage-origem}.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A099: `text/latex/capitulo_dados.tex:73`

**Antes:**

A decomposição desse quadro, total após revisão menos corte, fornece 4, 7 e 25 PAR igual a 2 nesses passos (\path{data/par_por_passo.csv}).

**Depois:**

A separação exata de PAR igual a 1 decorre do critério do \textit{label}, não de validação independente.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A100: `text/latex/capitulo_dados.tex:73`

**Antes:**

A separação exata dos PAR igual a 1 decorre do critério do \textit{label}, não de validação independente.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A101: `text/latex/capitulo_dados.tex:73`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O mínimo de 2.521 na ordenação corresponde à posição do último vínculo, enquanto 2.522 inclui todos os candidatos empatados no limiar global; não são duas contagens da mesma regra de parada.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A102: `text/latex/capitulo_dados.tex:75`

**Antes:**

O padrão-ouro foi construído por dois profissionais do IESC-UFRJ: o epidemiologista Luís Guilherme Buteri Alves realizou o pré-processamento, e a epidemiologista Daniela Medina Coeli Bastos configurou o \textit{linkage}, executou a blocagem e fez a revisão clerical.

**Depois:**

O \textit{linkage} de origem, o CdR e a revisão clerical foram executados por uma única epidemiologista, sem participação do autor e sem dupla classificação independente (Apêndice~\ref{ap:linkage-origem}).

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A103: `text/latex/capitulo_dados.tex:75`

**Antes:**

Houve revisora única, sem classificação independente, impossibilitando estimar concordância interavaliadores.

**Depois:**

O CdR precedeu a revisão que originou PAR, e sua nota e seus subescores estavam disponíveis à revisora; o que ela efetivamente consultou, inclusive SIM, Sinan e endereço, não está documentado.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A104: `text/latex/capitulo_dados.tex:75`

**Antes:**

A documentação de \citeonline{Oliveira2026obitoTBpulmonar} informa revisão manual abaixo do corte, mas não sustenta afirmar ordenação decrescente nem regra de parada (p.~34).

**Depois:**

A ausência de revisão independente impede estimar concordância interavaliadores.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A105: `text/latex/capitulo_dados.tex:75`

**Antes:**

Nela, o CdR foi utilizado no passo aberto, que recuperou cinco pares (Quadro~6), não nos passos 6, 9 e 11.

**Depois:**

A disponibilidade das mesmas evidências usadas pelos classificadores na construção da referência constitui potencial de viés de incorporação, cuja magnitude não foi estimada.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A106: `text/latex/capitulo_dados.tex:75`

**Antes:**

A mesma epidemiologista executou o CdR em Python sobre o extrato utilizado nesta tese \cite{Jardim2024comparador}; não se deve transferir esse uso ao protocolo histórico dos três passos.

**Depois:**

A avaliação supervisionada contra essa referência e a seleção dos casos verificados podem produzir otimismo, exigindo revisão independente e cega para sua quantificação.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A107: `text/latex/capitulo_dados.tex:75`

**Antes:**

O potencial de viés de incorporação depende de quanto as informações comparadas participaram das decisões de referência, exposição que não foi documentada de modo independente.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A108: `text/latex/capitulo_dados.tex:75`

**Antes:**

A avaliação supervisionada contra esse padrão-ouro e a seleção dos casos verificados podem produzir otimismo de magnitude não estimada, exigindo revisão independente e cega para sua quantificação.

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A109: `text/latex/capitulo_dados.tex:86`

**Antes:**

O passo~9 foi o mais produtivo, tanto pelo número de vínculos (127) quanto pela proporção de vínculos (8,8\%), mas também apresentou a maior fração de vínculos na área cinza definida pela nota final do comparador (82,7\%), indicando que a produtividade da blocagem não elimina a necessidade de pós-processamento dos potenciais pares.

**Depois:**

O passo~9 foi o mais produtivo, tanto pelo número de vínculos (127) quanto pela proporção de vínculos (8,8\%), mas também apresentou a maior fração de vínculos na área cinza definida pela nota final do comparador (82,7\%), sem que essa faixa da nota se confunda com a região abaixo do corte do OpenRecLink: 120 dos 127 vínculos desse passo são PAR igual a 1, já aceitos pelo corte, e sete são PAR igual a 2.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A110: `text/latex/capitulo_dados.tex:94`

**Antes:**

O desempenho dos classificadores foi avaliado por meio das seguintes métricas, reconhecidas na literatura de \textit{linkage} e de aprendizado de máquina \cite{Hand2018fmeasure, Hastie2009elements}:

**Depois:**

O protocolo distingue as seguintes métricas, reportadas conforme a avaliação de cada estudo e reconhecidas na literatura de \textit{linkage} e de aprendizado de máquina \cite{Hand2018fmeasure, Hastie2009elements}:

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A111: `text/latex/capitulo_dados.tex:97`

**Antes:**

\item \textbf{Precisão} (\textit{Precision}): proporção de pares classificados como verdadeiros que são efetivamente vínculos.

**Depois:**

\item \textbf{Valor preditivo positivo (VPP), também denominado precisão}: proporção de pares classificados como verdadeiros que são efetivamente vínculos.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A112: `text/latex/capitulo_dados.tex:99`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item \textbf{Valor preditivo negativo (VPN)}: proporção de não vínculos verdadeiros entre os classificados como não vínculos.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A113: `text/latex/capitulo_dados.tex:100`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item \textbf{Especificidade}: proporção de não vínculos corretamente classificados entre todos os não vínculos da referência.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A114: `text/latex/capitulo_dados.tex:101`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item \textbf{Brier}: média do erro quadrático entre a probabilidade estimada e a classe binária de referência.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A115: `text/latex/capitulo_dados.tex:102`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item \textbf{ECE}: erro de calibração esperado, calculado pela média das diferenças absolutas entre probabilidade média e frequência observada nas bandas, ponderadas pelo número de observações; depende da discretização adotada.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A116: `text/latex/capitulo_dados.tex:103`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item \textbf{Sensibilidade automática e com humano}: a primeira considera somente vínculos aceitos automaticamente; a segunda acrescenta os recuperados pelo revisor sob a hipótese explicitada em cada cenário, sem tratar encaminhamento como recuperação observada.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A117: `text/latex/capitulo_dados.tex:109`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

No GZ-CMD, as probabilidades fora da amostra (OOF) são produzidas com partições agrupadas por REFREC, de modo que registros de referência não atravessem treino e avaliação da mesma dobra.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A118: `text/latex/capitulo_dados.tex:109`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Esse procedimento não deve ser atribuído retroativamente às partições históricas dos demais experimentos.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A119: `text/latex/capitulo_dados.tex:115`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Os roteiros QW1 transferem \path{data/COMPARADORSEMIDENT.csv} para a infraestrutura Modal, conforme \path{scripts/qw1_modal_runner.py} e \path{scripts/qw1_modal_unified_pipeline.py}; trata-se do extrato pseudonimizado descrito na Seção~\ref{sec:dados-acesso-etica}, não apenas do \textit{benchmark} de 31 colunas.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A120: `text/latex/capitulo_dados.tex:115`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O envio não equivale à anonimização do extrato.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A121: `text/latex/capitulo_dados.tex:117`

**Antes:**

O ambiente de execução foi definido declarativamente como imagem de contêiner (\textit{container image}), incluindo todas as dependências de \textit{software} e o código-fonte do projeto, o que elimina variações decorrentes da configuração da máquina local e possibilita a reexecução a partir do extrato depositado no Zenodo (versão 3.0.0, utilizada nos experimentos) \cite{Jardim2026zenodoGzcmd}, que não inclui o extrato bruto; os passos que dependem de campos ausentes do depósito não são reexecutáveis por terceiros.

**Depois:**

O ambiente de execução foi definido declarativamente como imagem de contêiner (\textit{container image}), incluindo todas as dependências de \textit{software} e o código-fonte do projeto, o que reduz variações de configuração.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A122: `text/latex/capitulo_dados.tex:117`

**Antes:**

Essa abordagem pode ser particularmente adequada a grupos de pesquisa em saúde coletiva que, em geral, não dispõem de servidores com aceleradores gráficos dedicados, permitindo o acesso pontual a recursos computacionais de alto desempenho com custo proporcional ao tempo efetivo de processamento.

**Depois:**

O \textit{benchmark} do depósito Zenodo de acesso restrito (versão 3.0.0, utilizada nos experimentos) não é o extrato de entrada do QW1 \cite{Jardim2026zenodoGzcmd}; permite reexecutar somente os passos cujas entradas estão nele presentes, mediante autorização de acesso.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A123: `text/latex/capitulo_dados.tex:117`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Os passos que dependem dos campos ausentes não são reexecutáveis apenas com esse depósito.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A124: `text/latex/capitulo_dados.tex:117`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

A viabilidade computacional não afasta as exigências de governança: para dados identificados, recomenda-se execução local em ambiente controlado.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A125: `text/latex/capitulo_dados.tex:123`

**Antes:**

O depósito público no Zenodo reúne, na versão 3.0.1, 11 arquivos: \path{benchmark_sim_sinan_tb.csv}, com 61.696 linhas e 31 colunas (28 subescores de similaridade, nota final, PAR e PASSO); \path{splits.csv}, com índices de linha e partições; \path{baselines.json}, com resultados de referência; \path{manifest.json} e \path{zenodo_metadata.json}, com metadados; \path{README.md}, com a documentação do depósito; \path{k_anonymity_report.json}, com a avaliação de esquemas generalizados; \path{qw2.zip}, com decisões e respostas dos revisores em CSV, JSON e JSONL; \path{qw3.zip}, com o frame analítico de equidade, métricas por grupo, testes e documentação; \path{resultados.zip}, com métricas, probabilidades de teste, valores SHAP, figuras e tabelas; e \path{v2_gzcmd_v31.zip}, com os artefatos do GZ-CMD v3.1 descritos abaixo \cite{Jardim2026zenodo,Jardim2026zenodoGzcmd301}.

**Depois:**

O depósito de acesso restrito no Zenodo reúne, na versão 3.0.1, 11 arquivos: \path{benchmark_sim_sinan_tb.csv}, com 61.696 linhas e 31 colunas (28 subescores de similaridade, nota final, PAR e PASSO); \path{splits.csv}, com índices de linha e partições; \path{baselines.json}, com resultados de referência; \path{manifest.json} e \path{zenodo_metadata.json}, com metadados; \path{README.md}, com a documentação do depósito; \path{k_anonymity_report.json}, com a avaliação de esquemas generalizados; \path{qw2.zip}, com decisões e respostas dos revisores em CSV, JSON e JSONL; \path{qw3.zip}, com o frame analítico de equidade, métricas por grupo, testes e documentação; \path{resultados.zip}, com métricas, probabilidades de teste, valores SHAP, figuras e tabelas; e \path{v2_gzcmd_v31.zip}, com os artefatos do GZ-CMD v3.1 descritos abaixo \cite{Jardim2026zenodo,Jardim2026zenodoGzcmd301}.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A126: `text/latex/capitulo_dados.tex:123`

**Antes:**

A versão 3.0.1 declara licença MIT e corrige documentação e metadados da versão 3.0.0, sem substituir os resultados experimentais.

**Depois:**

A versão 3.0.1 corrige documentação e metadados da versão 3.0.0, sem substituir os resultados experimentais; os dados por potencial par ficam sujeitos a acordo de uso sob os CAAE citados, e MIT se limita ao código, não aos dados.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A127: `text/latex/capitulo_dados.tex:125`

**Antes:**

Na conferência do arquivo publicado, o agrupamento por faixa etária, sexo e indicador de encerramento apresentou 23 células observadas, com tamanho mínimo 6 e nenhuma célula com menos de 5 linhas.

**Depois:**

Na conferência do arquivo depositado, o agrupamento por faixa etária, sexo e indicador de encerramento apresentou 23 células observadas, com tamanho mínimo 6 e nenhuma célula com menos de 5 linhas.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A128: `text/latex/capitulo_dados.tex:127`

**Antes:**

O relatório \path{k_anonymity_report.json} avalia esquemas de generalização sobre o extrato local; não certifica a anonimização dos CSVs publicados.

**Depois:**

O relatório \path{k_anonymity_report.json} avalia esquemas de generalização sobre o extrato local; não certifica a anonimização dos CSVs depositados.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A129: `text/latex/capitulo_dados.tex:127`

**Antes:**

Na conferência direta das combinações de PASSO, 28 subescores e nota final do CSV publicado, observaram-se 24.735 grupos, com mínimo 1, dos quais 23.358 reuniam menos de 5 linhas, somando 30.141 linhas.

**Depois:**

Na conferência direta das combinações de PASSO, 28 subescores e nota final do CSV depositado, observaram-se 24.735 grupos, com mínimo 1, dos quais 23.358 reuniam menos de 5 linhas, somando 30.141 linhas.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A130: `text/latex/capitulo_dados.tex:131`

**Antes:**

Os artefatos do GZ-CMD v3.1, compreendendo funis por par, probabilidades fora da amostra (OOF) agrupadas por REFREC, respostas dos revisores identificadas por \texttt{pair\_id}, proporções de erro por banda e instruções de revisão (\textit{prompts}) v3 e v4, foram publicados como a versão 3.0.0 do mesmo depósito, utilizada nos experimentos (DOI 10.5281/zenodo.23928226) \cite{Jardim2026zenodoGzcmd}, incluindo os \textit{labels} por par do padrão-ouro pseudonimizados por \texttt{pair\_id}, sem identificadores diretos.

**Depois:**

Os artefatos do GZ-CMD v3.1, compreendendo funis por par, probabilidades fora da amostra (OOF) agrupadas por REFREC, respostas dos revisores identificadas por \texttt{pair\_id}, proporções de erro por banda e instruções de revisão (\textit{prompts}) v3 e v4, integram a versão 3.0.0 do mesmo depósito, de acesso restrito, utilizada nos experimentos (DOI 10.5281/zenodo.23928226) \cite{Jardim2026zenodoGzcmd}, incluindo os \textit{labels} por par do padrão-ouro pseudonimizados por \texttt{pair\_id}, sem identificadores diretos.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.


### apendice_linkage_origem.tex

#### A131: `text/latex/apendice_linkage_origem.tex:1`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\chapter{Proveniência e procedimentos do \textit{linkage} de origem}\label{ap:linkage-origem}

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A132: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O \textit{linkage} (vinculação de registros) de origem antecede a avaliação desta tese.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A133: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

A epidemiologista Daniela Medina Coeli Bastos, do IESC-UFRJ, executou sozinha o \textit{linkage}, o Comparador de Registros (CdR) e a revisão clerical que originou PAR.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A134: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O autor não participou dessas etapas e recebeu o extrato pós-CdR como membro do LinkDataPop.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A135: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Houve revisora única, sem dupla classificação independente.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A136: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Conforme a informação de proveniência confirmada pelo autor e registrada em \path{data/proveniencia_oliveira_meta.json}, o CdR em Python foi executado antes da revisão, com nota e subescores disponíveis à revisora.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A137: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Não está documentado o que ela efetivamente consultou durante a classificação, inclusive eventual consulta ao SIM, ao Sinan ou ao endereço.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A138: `text/latex/apendice_linkage_origem.tex:3`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Essa distinção não elimina o potencial de viés de incorporação.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A139: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

O extrato provém do processo de \textit{linkage} descrito por \citeonline{Oliveira2026obitoTBpulmonar}, entre os óbitos de residentes no município do Rio de Janeiro com tuberculose como causa básica (SIM, 2019 a 2022) e as notificações de tuberculose (Sinan, 2013 a 2022), em quatro etapas: \textit{linkage} determinístico, 29 passos probabilísticos no OpenRecLink, um passo aberto e deduplicação (p.~33 a~39). (antes no capítulo de dados)

**Depois:**

O extrato provém do processo de \textit{linkage} descrito por \citeonline{Oliveira2026obitoTBpulmonar}, entre os óbitos de residentes no município do Rio de Janeiro com tuberculose como causa básica (SIM, 2019 a 2022) e as notificações de tuberculose (Sinan, 2013 a 2022), em quatro etapas: \textit{linkage} determinístico, 29 passos probabilísticos no OpenRecLink, um passo aberto e deduplicação (p.~33 a~39).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A140: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

Os vínculos encontrados na etapa determinística foram retirados antes do \textit{linkage} probabilístico, de modo que os potenciais pares do extrato correspondem ao que restou dessa etapa. (antes no capítulo de dados)

**Depois:**

Os vínculos encontrados na etapa determinística foram retirados antes do \textit{linkage} probabilístico, de modo que os potenciais pares do extrato correspondem ao que restou dessa etapa.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A141: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

O extrato reúne os potenciais pares de três dos 29 passos probabilísticos (6, 9 e 11), cujos parâmetros, pontos de corte (7,0240; 7,8444; 7,9019) e contagens de vínculos pelo corte e após a revisão coincidem com os descritos na monografia (Quadro~5, p.~36 a~37). (antes no capítulo de dados)

**Depois:**

O extrato reúne os potenciais pares de três dos 29 passos probabilísticos (6, 9 e 11), cujos parâmetros, pontos de corte (7,0240; 7,8444; 7,9019) e contagens de vínculos pelo corte e após a revisão coincidem com os descritos na monografia (Quadro~5, p.~36 a~37).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A142: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

Esses três passos contribuíram com 247 dos 2.549 pares do processo completo (p.~35); a proporção de 1:249 refere-se, portanto, à base exportada. (antes no capítulo de dados)

**Depois:**

Esses três passos contribuíram com 247 dos 2.549 pares do processo completo (p.~35); a proporção de 1:249 refere-se, portanto, à base exportada.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A143: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

A correspondência documental e as decisões de proveniência informadas pelo autor estão registradas em \texttt{data/proveniencia\_oliveira\_meta.json}. (antes no capítulo de dados)

**Depois:**

A correspondência documental e as decisões de proveniência informadas pelo autor estão registradas em \texttt{data/proveniencia\_oliveira\_meta.json}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A144: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

O órgão fornecedor das bases, a data da extração e o critério de escolha dos três passos exportados não constam da documentação disponível ao autor. (antes no capítulo de dados)

**Depois:**

O órgão fornecedor das bases, a data da extração e o critério de escolha dos três passos exportados não constam da documentação disponível ao autor.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A145: `text/latex/apendice_linkage_origem.tex:5`

**Antes:**

A escolha dessas bases justifica-se pelas razões detalhadas na Seção \ref{sec:justificativas-especificas}, em particular a relevância da tuberculose como condição marcadora da qualidade do cuidado em saúde e a experiência acumulada do grupo de pesquisa no \textit{linkage} dessas fontes de dados \cite{Oliveira2012uso, Bartholomay2014improved, Oliveira2016accuracy}. (antes no capítulo de dados)

**Depois:**

A escolha dessas bases justifica-se pelas razões detalhadas na Seção \ref{sec:justificativas-especificas}, em particular a relevância da tuberculose como condição marcadora da qualidade do cuidado em saúde e a experiência acumulada do grupo de pesquisa no \textit{linkage} dessas fontes de dados \cite{Oliveira2012uso, Bartholomay2014improved, Oliveira2016accuracy}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A146: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

O procedimento de \textit{linkage} probabilístico no OpenRecLink seguiu o modelo de decisão de Fellegi e Sunter \citeyearpar{Fellegi1969theory}. (antes no capítulo de dados)

**Depois:**

O procedimento de \textit{linkage} probabilístico no OpenRecLink seguiu o modelo de decisão de Fellegi e Sunter \citeyearpar{Fellegi1969theory}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A147: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Na etapa de padronização, os campos textuais de ambas as bases foram convertidos para caixa alta, com remoção de acentos diacríticos, preposições e artigos, e as datas de nascimento foram normalizadas para formato numérico contínuo (AAAAMMDD). (antes no capítulo de dados)

**Depois:**

Na etapa de padronização, os campos textuais de ambas as bases foram convertidos para caixa alta, com remoção de acentos diacríticos, preposições e artigos, e as datas de nascimento foram normalizadas para formato numérico contínuo (AAAAMMDD).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A148: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

A etapa de comparação no OpenRecLink utilizou três campos de identificação: nome próprio e nome da mãe, comparados por algoritmo aproximado (tipo APROX, com limiar de similaridade de 85\%); e data de nascimento, comparada caractere a caractere (tipo CARACTERE, com limiar de 65\%). (antes no capítulo de dados)

**Depois:**

A etapa de comparação no OpenRecLink utilizou três campos de identificação: nome próprio e nome da mãe, comparados por algoritmo aproximado (tipo APROX, com limiar de similaridade de 85\%); e data de nascimento, comparada caractere a caractere (tipo CARACTERE, com limiar de 65\%).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A149: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

A variável sexo não integrou a etapa de comparação, tendo sido utilizada exclusivamente como chave de blocagem em passos específicos. (antes no capítulo de dados)

**Depois:**

A variável sexo não integrou a etapa de comparação, tendo sido utilizada exclusivamente como chave de blocagem em passos específicos.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A150: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Os parâmetros de concordância e discordância atribuídos a cada campo estão detalhados na Tabela~\ref{tab:parametros-orl}. (antes no capítulo de dados)

**Depois:**

Os parâmetros de concordância e discordância atribuídos a cada campo estão detalhados na Tabela~\ref{tab:parametros-orl}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A151: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Para cada potencial par, o programa calculou um escore probabilístico correspondente à soma das razões de log-verossimilhança (\textit{log-likelihood ratios}) dos campos comparados, no qual campos concordantes contribuem positivamente e campos discordantes contribuem negativamente para o escore total \cite{Fellegi1969theory, Christen2012book}. (antes no capítulo de dados)

**Depois:**

Para cada potencial par, o programa calculou um escore probabilístico correspondente à soma das razões de log-verossimilhança (\textit{log-likelihood ratios}) dos campos comparados, no qual campos concordantes contribuem positivamente e campos discordantes contribuem negativamente para o escore total \cite{Fellegi1969theory, Christen2012book}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A152: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Na base utilizada, os escores probabilísticos do OpenRecLink variaram de $-12{,}85$ (discordância em todos os campos) a $+15{,}58$ (concordância plena), distribuindo-se em 116 valores distintos. (antes no capítulo de dados)

**Depois:**

Na base utilizada, os escores probabilísticos do OpenRecLink variaram de $-12{,}85$ (discordância em todos os campos) a $+15{,}58$ (concordância plena), distribuindo-se em 116 valores distintos.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A153: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Para evitar ambiguidade, o presente trabalho distingue duas grandezas: o escore do \textit{linkage} probabilístico, calculado pelo OpenRecLink como soma das razões de log-verossimilhança, e a nota final do Comparador de Registros (CdR; Seção~\ref{sec:comparador-registros}), escore composto derivado dos 29 subescores de similaridade campo a campo. (antes no capítulo de dados)

**Depois:**

Para evitar ambiguidade, o presente trabalho distingue duas grandezas: o escore do \textit{linkage} probabilístico, calculado pelo OpenRecLink como soma das razões de log-verossimilhança, e a nota final do Comparador de Registros (CdR; Seção~\ref{sec:comparador-registros}), escore composto derivado dos 29 subescores de similaridade campo a campo.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A154: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

Na origem, a revisão manual incidiu sobre a região abaixo do corte do escore OpenRecLink \cite{Oliveira2026obitoTBpulmonar}; nos resultados desta tese, área cinza da nota designa o intervalo da nota final do CdR em $[5,9)$ (Seção~\ref{sec:comparador-registros}). (antes no capítulo de dados)

**Depois:**

Na origem, a revisão manual incidiu sobre a região abaixo do corte do escore OpenRecLink \cite{Oliveira2026obitoTBpulmonar}; nos resultados desta tese, área cinza da nota designa o intervalo da nota final do CdR em $[5,9)$ (Seção~\ref{sec:comparador-registros}).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A155: `text/latex/apendice_linkage_origem.tex:7`

**Antes:**

As duas regiões não coincidem. (antes no capítulo de dados)

**Depois:**

As duas regiões não coincidem.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A156: `text/latex/apendice_linkage_origem.tex:9`

**Antes:**

\begin{table}[!ht] (antes no capítulo de dados)

**Depois:**

\begin{table}[!ht]

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A157: `text/latex/apendice_linkage_origem.tex:10`

**Antes:**

\centering (antes no capítulo de dados)

**Depois:**

\centering

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A158: `text/latex/apendice_linkage_origem.tex:11`

**Antes:**

\caption{Parâmetros de comparação configurados no OpenRecLink para os campos de identificação.} (antes no capítulo de dados)

**Depois:**

\caption{Parâmetros de comparação configurados no OpenRecLink para os campos de identificação.}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A159: `text/latex/apendice_linkage_origem.tex:12`

**Antes:**

\label{tab:parametros-orl} (antes no capítulo de dados)

**Depois:**

\label{tab:parametros-orl}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A160: `text/latex/apendice_linkage_origem.tex:13`

**Antes:**

\begin{tabular}{ (antes no capítulo de dados)

**Depois:**

\begin{tabular}{

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A161: `text/latex/apendice_linkage_origem.tex:14`

**Antes:**

p{\dimexpr0.32\linewidth-2\tabcolsep\relax} (antes no capítulo de dados)

**Depois:**

p{\dimexpr0.32\linewidth-2\tabcolsep\relax}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A162: `text/latex/apendice_linkage_origem.tex:15`

**Antes:**

>{\centering\arraybackslash}p{\dimexpr0.16\linewidth-2\tabcolsep\relax} (antes no capítulo de dados)

**Depois:**

>{\centering\arraybackslash}p{\dimexpr0.16\linewidth-2\tabcolsep\relax}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A163: `text/latex/apendice_linkage_origem.tex:16`

**Antes:**

*{2}{>{\centering\arraybackslash}p{\dimexpr0.20\linewidth-2\tabcolsep\relax}} (antes no capítulo de dados)

**Depois:**

*{2}{>{\centering\arraybackslash}p{\dimexpr0.20\linewidth-2\tabcolsep\relax}}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A164: `text/latex/apendice_linkage_origem.tex:17`

**Antes:**

>{\centering\arraybackslash}p{\dimexpr0.12\linewidth-2\tabcolsep\relax}} (antes no capítulo de dados)

**Depois:**

>{\centering\arraybackslash}p{\dimexpr0.12\linewidth-2\tabcolsep\relax}}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A165: `text/latex/apendice_linkage_origem.tex:18`

**Antes:**

\hline (antes no capítulo de dados)

**Depois:**

\hline

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A166: `text/latex/apendice_linkage_origem.tex:19`

**Antes:**

\textbf{Campo} & \textbf{Tipo} & \textbf{Concordância (\%)} & \textbf{Discordância (\%)} & \textbf{Limiar (\%)} \\ (antes no capítulo de dados)

**Depois:**

\textbf{Campo} & \textbf{Tipo} & \textbf{Concordância (\%)} & \textbf{Discordância (\%)} & \textbf{Limiar (\%)} \\

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A167: `text/latex/apendice_linkage_origem.tex:20`

**Antes:**

\hline (antes no capítulo de dados)

**Depois:**

\hline

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A168: `text/latex/apendice_linkage_origem.tex:21`

**Antes:**

Nome (NOME) & Aproximado & 99 & 1 & 85 \\ (antes no capítulo de dados)

**Depois:**

Nome (NOME) & Aproximado & 99 & 1 & 85 \\

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A169: `text/latex/apendice_linkage_origem.tex:22`

**Antes:**

Nome da mãe (NOMEMAE) & Aproximado & 75 & 2 & 85 \\ (antes no capítulo de dados)

**Depois:**

Nome da mãe (NOMEMAE) & Aproximado & 75 & 2 & 85 \\

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A170: `text/latex/apendice_linkage_origem.tex:23`

**Antes:**

Data de nascimento (DTNASC) & Caractere & 95 & 5 & 65 \\ (antes no capítulo de dados)

**Depois:**

Data de nascimento (DTNASC) & Caractere & 95 & 5 & 65 \\

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A171: `text/latex/apendice_linkage_origem.tex:24`

**Antes:**

\hline (antes no capítulo de dados)

**Depois:**

\hline

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A172: `text/latex/apendice_linkage_origem.tex:25`

**Antes:**

\end{tabular} (antes no capítulo de dados)

**Depois:**

\end{tabular}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A173: `text/latex/apendice_linkage_origem.tex:26`

**Antes:**

\end{table} (antes no capítulo de dados)

**Depois:**

\end{table}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A174: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

A geração de potenciais pares no OpenRecLink empregou a técnica de blocagem (\textit{blocking}), que particiona os registros em subgrupos definidos por chaves comuns e restringe as comparações aos pares que compartilham a mesma chave, reduzindo substancialmente o custo computacional \cite{Christen2012book}. (antes no capítulo de dados)

**Depois:**

A geração de potenciais pares no OpenRecLink empregou a técnica de blocagem (\textit{blocking}), que particiona os registros em subgrupos definidos por chaves comuns e restringe as comparações aos pares que compartilham a mesma chave, reduzindo substancialmente o custo computacional \cite{Christen2012book}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A175: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

No ORL, os parâmetros de blocagem foram definidos por combinações de variáveis fonéticas e demográficas (código \textit{Soundex} do primeiro e do último bloco do nome, sexo, ano de nascimento, primeiro e último bloco do nome da mãe, código do município de residência e bairro de residência (BAIRES)), organizadas em passos sequenciais. (antes no capítulo de dados)

**Depois:**

No ORL, os parâmetros de blocagem foram definidos por combinações de variáveis fonéticas e demográficas (código \textit{Soundex} do primeiro e do último bloco do nome, sexo, ano de nascimento, primeiro e último bloco do nome da mãe, código do município de residência e bairro de residência (BAIRES)), organizadas em passos sequenciais.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A176: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

Adotou-se, ainda, a estratégia de exclusão progressiva dos registros do Sinan-TB já pareados em cada passo, de modo que, uma vez identificado um potencial par, o respectivo registro do Sinan não retornasse ao universo de busca dos passos subsequentes. (antes no capítulo de dados)

**Depois:**

Adotou-se, ainda, a estratégia de exclusão progressiva dos registros do Sinan-TB já pareados em cada passo, de modo que, uma vez identificado um potencial par, o respectivo registro do Sinan não retornasse ao universo de busca dos passos subsequentes.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A177: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

Foram utilizados múltiplos passos de blocagem (\textit{blocking steps}), estratégia recomendada para maximizar a sensibilidade do processo ao permitir que pares não capturados por uma chave restritiva fossem recuperados por chaves alternativas \cite{Coeli2002blocking}. (antes no capítulo de dados)

**Depois:**

Foram utilizados múltiplos passos de blocagem (\textit{blocking steps}), estratégia recomendada para maximizar a sensibilidade do processo ao permitir que pares não capturados por uma chave restritiva fossem recuperados por chaves alternativas \cite{Coeli2002blocking}.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A178: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

O procedimento incluiu 29 passos probabilísticos e um passo aberto, cuja chave de blocagem foi o \textit{Soundex} do primeiro fragmento do nome do paciente (PBLOCO), conforme a descrição metodológica e o Quadro~6 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~34 e~39). (antes no capítulo de dados)

**Depois:**

O procedimento incluiu 29 passos probabilísticos e um passo aberto, cuja chave de blocagem foi o \textit{Soundex} do primeiro fragmento do nome do paciente (PBLOCO), conforme a descrição metodológica e o Quadro~6 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~34 e~39).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A179: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

Três dos passos probabilísticos integram o extrato, sem informação sobre o critério de escolha para exportação. (antes no capítulo de dados)

**Depois:**

Três dos passos probabilísticos integram o extrato, sem informação sobre o critério de escolha para exportação.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A180: `text/latex/apendice_linkage_origem.tex:28`

**Antes:**

Os três passos exportados, com suas respectivas chaves de blocagem, foram: (antes no capítulo de dados)

**Depois:**

Os três passos exportados, com suas respectivas chaves de blocagem, foram:

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A181: `text/latex/apendice_linkage_origem.tex:30`

**Antes:**

\begin{enumerate} (antes no capítulo de dados)

**Depois:**

\begin{enumerate}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A182: `text/latex/apendice_linkage_origem.tex:31`

**Antes:**

\item \textbf{Passo por nome e nome da mãe (passo~6):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) e do primeiro e do último bloco do nome da mãe (PBLOCOMAE, UBLOCOMAE), gerando 2.649 potenciais pares, dos quais 80 foram classificados como pares (76 pelo ponto de corte e 4 em revisão manual); (antes no capítulo de dados)

**Depois:**

\item \textbf{Passo por nome e nome da mãe (passo~6):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) e do primeiro e do último bloco do nome da mãe (PBLOCOMAE, UBLOCOMAE), gerando 2.649 potenciais pares, dos quais 80 foram classificados como pares (76 pelo ponto de corte e 4 em revisão manual);

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A183: `text/latex/apendice_linkage_origem.tex:32`

**Antes:**

\item \textbf{Passo restritivo (passo~9):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + ano de nascimento (ANONASC) + sexo (SEXO), gerando 1.449 potenciais pares, dos quais 127 foram classificados como pares (120 pelo ponto de corte e 7 em revisão manual); (antes no capítulo de dados)

**Depois:**

\item \textbf{Passo restritivo (passo~9):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + ano de nascimento (ANONASC) + sexo (SEXO), gerando 1.449 potenciais pares, dos quais 127 foram classificados como pares (120 pelo ponto de corte e 7 em revisão manual);

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A184: `text/latex/apendice_linkage_origem.tex:33`

**Antes:**

\item \textbf{Passo ampliado (passo~11):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + sexo (SEXO) + código do município de residência (CODMUNRES), gerando 57.598 potenciais pares, dos quais 40 foram classificados como pares (15 pelo ponto de corte e 25 em revisão manual). (antes no capítulo de dados)

**Depois:**

\item \textbf{Passo ampliado (passo~11):} código \textit{Soundex} do primeiro e do último bloco do nome (PBLOCO, UBLOCO) + sexo (SEXO) + código do município de residência (CODMUNRES), gerando 57.598 potenciais pares, dos quais 40 foram classificados como pares (15 pelo ponto de corte e 25 em revisão manual).

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A185: `text/latex/apendice_linkage_origem.tex:34`

**Antes:**

\end{enumerate} (antes no capítulo de dados)

**Depois:**

\end{enumerate}

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A186: `text/latex/apendice_linkage_origem.tex:36`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O Quadro~5 de \citeonline{Oliveira2026obitoTBpulmonar} (p.~36 a~38) registra cortes de 7,0240, 7,8444 e 7,9019 para os passos 6, 9 e 11.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A187: `text/latex/apendice_linkage_origem.tex:36`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

As contagens pelo corte são 76, 120 e 15, e, após revisão, 80, 127 e 40, respectivamente; as diferenças correspondem a 4, 7 e 25 vínculos PAR igual a 2 (\path{data/par_por_passo.csv}).

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A188: `text/latex/apendice_linkage_origem.tex:36`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

A coincidência até a quarta casa decimal com o menor escore entre PAR igual a 1 no extrato decorre da definição do \textit{label}, não de validação independente.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A189: `text/latex/apendice_linkage_origem.tex:36`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Não se documentaram a ordenação dos candidatos nem a regra de parada da revisão.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A190: `text/latex/apendice_linkage_origem.tex:38`

**Antes:**

Na origem, o CdR foi usado na revisão do passo aberto, que recuperou 5 pares \cite{Oliveira2026obitoTBpulmonar} (p.~39); o cálculo da nota final sobre os potenciais pares desta tese foi outra execução, com a implementação em Python. (antes no capítulo de dados)

**Depois:**

Na origem, o CdR foi usado na revisão do passo aberto, que recuperou 5 pares \cite{Oliveira2026obitoTBpulmonar} (p.~39); o cálculo da nota final sobre os potenciais pares desta tese foi outra execução, com a implementação em Python.

**Veredito:** Fiel: deslocamento literal para material suplementar, sem alteração de conteúdo.

#### A191: `text/latex/apendice_linkage_origem.tex:40`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

O texto da qualificação, preservado em \path{qualificacao-2026:text/latex/capitulo5.tex}, atribuía a Luís Guilherme Buteri Alves o pré-processamento, incluindo limpeza de nomes inconsistentes, como ``homem pardo'', ``morador de rua'' e ``recém-nascido'', e criação de DTNASC2 para normalização das datas de nascimento.

**Veredito:** Fiel ao testemunho da tag, não comprovação de execução: informação da qualificação explicitamente distinguida da fonte primária e da decisão D1.

#### A192: `text/latex/apendice_linkage_origem.tex:40`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Essa é informação da qualificação, não documentada em fonte primária de execução acessível ao autor; sua recuperação preserva o histórico do relato, sem comprovar os procedimentos.

**Veredito:** Fiel ao testemunho da tag, não comprovação de execução: informação da qualificação explicitamente distinguida da fonte primária e da decisão D1.

#### A193: `text/latex/apendice_linkage_origem.tex:40`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Essa atribuição ao pré-processamento não corresponde a uma segunda revisão clerical nem altera a autoria única do \textit{linkage}, do CdR e da revisão confirmada acima.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A194: `text/latex/apendice_linkage_origem.tex:40`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

As afirmações da qualificação sobre ordenação decrescente e consulta ao SIM, ao Sinan e ao endereço não são retomadas como fatos, pois não dispõem de documentação independente.

**Veredito:** Fiel ao testemunho da tag, não comprovação de execução: informação da qualificação explicitamente distinguida da fonte primária e da decisão D1.


### tese.tex

#### A195: `text/latex/tese.tex:475`

**Antes:**

A Daniela Medina Coeli Bastos, do LinkDataPop, agradeço pela cessão do

**Depois:**

A Daniela Medina Coeli Bastos, do LinkDataPop, agradeço pela preparação do

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A196: `text/latex/tese.tex:476`

**Antes:**

extrato e pela revisão clerical que deu origem ao padrão-ouro deste

**Depois:**

extrato pós-CdR e pela revisão clerical que deu origem ao padrão-ouro deste

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A197: `text/latex/tese.tex:674`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\input{apendice_linkage_origem.tex}

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.


### siglas.tex

#### A198: `text/latex/siglas.tex:21`

**Antes:**

FDR      & \textit{False Discovery Rate}: taxa de falsas descobertas \\[6pt]

**Depois:**

FDR      & \textit{False Discovery Rate}: proporção de falsas descobertas \\[6pt]

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A199: `text/latex/siglas.tex:25`

**Antes:**

F-S      & Fellegi e Sunter (teoria de decis\~{a}o para \textit{record linkage}) \\[6pt]

**Depois:**

(retirada deste ponto; ver veredito)

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A200: `text/latex/siglas.tex:26`

**Antes:**

GZ-CMD   & \textit{Grey-Zone Cost-based Mixture Deferral}: cascata de decis\~{a}o audit\'{a}vel para a regi\~{a}o abaixo do corte \\[6pt]

**Depois:**

GZ-CMD   & \textit{Grey-Zone Cost-based Mixture Deferral}: política de decisão por \textit{expected loss} com custos assimétricos para a regi\~{a}o abaixo do corte \\[6pt]

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A201: `text/latex/siglas.tex:58`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

VPN      & Valor Preditivo Negativo \\[6pt]

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.


### glossario.tex

#### A202: `text/latex/glossario.tex:32`

**Antes:**

É instrumento exploratório para discutir orçamento humano; sua conversão em horas depende de parâmetro temporal, não de tempo medido.

**Depois:**

É instrumento exploratório para escolha da ordenação e do ponto de parada segundo o orçamento humano; sua conversão em horas depende de parâmetro temporal, não de tempo medido.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A203: `text/latex/glossario.tex:34`

**Antes:**

O corte 9 da nota do CdR e a revisão integral de $[5,9)$ são cenários hipotéticos; o corte 6,0 é referência histórica da análise de equidade.

**Depois:**

O corte 9 da nota do CdR e a revisão integral de $[5,9)$ são cenários hipotéticos; o corte 8 da nota pertence ao Estudo~3, e o corte 6,0 é referência histórica da análise de equidade.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A204: `text/latex/glossario.tex:68`

**Antes:**

\textit{Framework} operacional proposto nesta tese para governança da incerteza no \textit{linkage} probabilístico.

**Depois:**

Política de decisão por \textit{expected loss} com custos assimétricos proposta nesta tese para governança da incerteza no \textit{linkage} probabilístico.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A205: `text/latex/glossario.tex:88`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item[O1 a O4] Ordenações exploratórias dos potenciais pares abaixo do corte: O1 pelo escore do OpenRecLink, O2 pela nota final do CdR, O3 pela probabilidade calibrada OOF do classificador e O4 pela probabilidade recalibrada de P.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A206: `text/latex/glossario.tex:88`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Não se confundem com as consultas de busca bibliográfica, identificadas por Q1 a Q5 no Apêndice~\ref{ap:protocolo-busca}.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A207: `text/latex/glossario.tex:90`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item[Orçamento $k$] Número de potenciais pares selecionados para revisão em uma ordenação.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A208: `text/latex/glossario.tex:100`

**Antes:**

\item[P-Platt] Sensibilidade de P que substitui a isotônica restrita por \textit{Platt scaling} sobre o logit da probabilidade bruta.

**Depois:**

\item[P-Platt] Análise de sensibilidade de P que substitui a isotônica restrita por \textit{Platt scaling} sobre o logit da probabilidade bruta.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A209: `text/latex/glossario.tex:114`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item[Resultado negativo com registro interno] O arranjo P, registrado antes da avaliação, não superou a referência do serviço; o registro não protege a escolha posterior do enquadramento nem transforma comparações exploratórias em confirmatórias.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A210: `text/latex/glossario.tex:116`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

\item[Vigilância e confirmação] Modos operacionais que priorizam, respectivamente, a sensibilidade para evitar perdas de vínculos e o valor preditivo positivo para evitar falsos vínculos.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.


### apendice_busca.tex

#### A211: `text/latex/apendice_busca.tex:16`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

As consultas do OpenAlex são designadas Q1 a Q5 neste apêndice, correspondendo às chaves históricas O1 a O5 dos arquivos de coleta, que foram preservadas.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A212: `text/latex/apendice_busca.tex:16`

**Antes:**

(inclusão; sem frase anterior neste arquivo)

**Depois:**

Essa distinção evita confundi-las com as ordenações O1 a O4 do Estudo~4.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A213: `text/latex/apendice_busca.tex:67`

**Antes:**

OpenAlex & O1, cadeia principal, título e resumo & 7 & 3\textsuperscript{d} \\

**Depois:**

OpenAlex & Q1, cadeia principal, título e resumo & 7 & 3\textsuperscript{d} \\

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A214: `text/latex/apendice_busca.tex:68`

**Antes:**

OpenAlex & O2, variante com \textit{classifier} e SUS, título e resumo & 13 & 3\textsuperscript{d} \\

**Depois:**

OpenAlex & Q2, variante com \textit{classifier} e SUS, título e resumo & 13 & 3\textsuperscript{d} \\

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A215: `text/latex/apendice_busca.tex:69`

**Antes:**

OpenAlex & O3, cadeia principal, parâmetro \texttt{search} & 2.115 & 3\textsuperscript{d} \\

**Depois:**

OpenAlex & Q3, cadeia principal, parâmetro \texttt{search} & 2.115 & 3\textsuperscript{d} \\

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A216: `text/latex/apendice_busca.tex:70`

**Antes:**

OpenAlex & O4, termos novos da cadeia ampliada, parâmetro \texttt{search} & 1.153 & 3\textsuperscript{d} \\

**Depois:**

OpenAlex & Q4, termos novos da cadeia ampliada, parâmetro \texttt{search} & 1.153 & 3\textsuperscript{d} \\

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A217: `text/latex/apendice_busca.tex:71`

**Antes:**

OpenAlex & O5, cadeia ampliada, título e resumo & 11 & 3\textsuperscript{d} \\

**Depois:**

OpenAlex & Q5, cadeia ampliada, título e resumo & 11 & 3\textsuperscript{d} \\

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A218: `text/latex/apendice_busca.tex:91`

**Antes:**

Nas consultas em título e resumo (O1, O2 e O5), as triagens por título resultaram em três estudos que atendem ao critério, os mesmos recuperados pelas consultas em \texttt{search} (O3 e O4).

**Depois:**

Nas consultas em título e resumo (Q1, Q2 e Q5), as triagens por título resultaram em três estudos que atendem ao critério, os mesmos recuperados pelas consultas em \texttt{search} (Q3 e Q4).

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.

#### A219: `text/latex/apendice_busca.tex:91`

**Antes:**

Das 2.115 recuperações de O3, 1.894 foram excluídas por não terem termo de \textit{linkage} no título, 196 pelo filtro de palavras-chave sobre o resumo e 21 por leitura do título, e o mesmo padrão se repetiu em O4.

**Depois:**

Das 2.115 recuperações de Q3, 1.894 foram excluídas por não terem termo de \textit{linkage} no título, 196 pelo filtro de palavras-chave sobre o resumo e 21 por leitura do título, e o mesmo padrão se repetiu em Q4.

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.


### tables/tab_blocagem_rendimento.tex

#### A220: `text/latex/tables/tab_blocagem_rendimento.tex:19`

**Antes:**

A área cinza corresponde, em toda a tese, à nota final do Comparador de Registros (CdR) em $[5,9)$, com a fração desta tabela calculada entre os vínculos do passo.}

**Depois:**

A área cinza da nota corresponde à nota final do Comparador de Registros (CdR) em $[5,9)$, com a fração desta tabela calculada entre os vínculos do passo; não corresponde à região abaixo do corte do OpenRecLink.}

**Veredito:** Fiel: ajuste de definição, escopo ou remissão conforme os itens acima; sem novo resultado experimental.


### referencias.bib

#### A221: `text/latex/referencias.bib:667`

**Antes:**

note = {Licen{\c{c}}a MIT.

**Depois:**

note = {Dados de acesso restrito, mediante solicita{\c{c}}{\~a}o e acordo de uso; MIT somente para c{\'o}digo.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A222: `text/latex/referencias.bib:677`

**Antes:**

note = {Licen{\c{c}}a MIT.

**Depois:**

note = {Dados de acesso restrito, mediante solicita{\c{c}}{\~a}o e acordo de uso; MIT somente para c{\'o}digo.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.

#### A223: `text/latex/referencias.bib:687`

**Antes:**

note = {Licen{\c{c}}a MIT; dados id{\^e}nticos {\`a} vers{\~a}o 3.0.0.

**Depois:**

note = {Dados de acesso restrito, mediante solicita{\c{c}}{\~a}o e acordo de uso; MIT somente para c{\'o}digo; dados id{\^e}nticos {\`a} vers{\~a}o 3.0.0.

**Veredito:** Fiel à decisão D1/D2 e às fontes de proveniência/execução: corrige atribuição, acesso, entrada computacional ou métrica; ressalvas de privacidade e viés permanecem. Estado remoto restrito depende da ação externa explicitada acima.
