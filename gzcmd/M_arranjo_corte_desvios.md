# Arranjo sequencial: desvios e limites do registro

Este documento complementa o registro histórico, sem alterar seus arquivos ou hashes. P permanece o braço principal. P-Platt é análise de sensibilidade. O comparador a posteriori não é registrado.

Na S8c, os arquivos com hashes fixados são preservados, e a apresentação passa a ser gerada separadamente em `text/latex/tables/editorial/`. Isso não significa que as tabelas e os testes tenham permanecido intactos ao longo de todo o histórico: as alterações anteriores e as restrições à restauração estão discriminadas abaixo.

## Cronologia e desvios

- O registro inicial, commit `8c16a41`, fixou hashes de código que só entrou no git em `83c7842`. A versão originalmente registrada não existe no histórico do git. A primeira S1 utilizou essa versão. O histórico disponível não permite reconstruí-la apenas por checkout.
- O suplemento, emenda 3, autorizou 69 chamadas adicionais, apesar da regra inicial `no sensitivity API calls`. A declaração `parameter_change: false` não elimina o desvio de protocolo. O suplemento foi registrado antes da S3.
- O comparador a posteriori aplica o corte seguido da cascata publicada, sem recalibração. Sua análise foi feita antes do registro inicial, de 5 de novembro de 2026, às 01:22:17 UTC. Os metadados originais da S3 registram 02:11:52 UTC; seu commit `16d5876` consta no git log às 02:17:13 UTC do mesmo dia. A referência anterior a 04:13 UTC foi corrigida pelo histórico. Portanto, os resultados de PAR=2 da cascata já eram conhecidos quando se registraram recalibração, braços e avaliação. O enquadramento sequencial foi escolhido com conhecimento desses resultados.
- A especificidade constava dos desfechos registrados. Sua ausência na apresentação inicial é uma omissão de relato. O texto revisto distingue esse desfecho da taxa esperada de falsos vínculos.

## Reprodutibilidade e interpretação

A reprodução numérica deve ser descrita como concordância dentro de `1e-14`, com partições idênticas, não como identidade bit a bit. A análise de robustez, com registro prévio próprio, examina threads (`n_jobs`), sementes e partições. É não confirmatória e mede estabilidade nos mesmos dados, não generalização.

Os cortes do OpenRecLink são o menor SCORE entre os PAR=1 de cada passo. Coincidem com Oliveira (2026) até a quarta casa decimal. São derivados do rótulo, não estimativas independentes dele.

As taxas do revisor têm denominadores pequenos. Em grey_low, e_fn foi 1,0 em quatro folds na vigilância (0, 1, 3 e 4), conforme `data/arranjo_corte_taxas_crossfit.csv`, linhas 2, 8, 20 e 26. Essa contagem corrige os dois folds mencionados na instrução editorial. P-PAR2-grey tem apenas dois ou três PAR=2 em algumas estimativas. Sua transferência para os pares selecionados abaixo do corte não foi validada em dados novos.

## Fontes preservadas

Consultar `prereg_arranjo_corte.yaml`, `M_arranjo_corte.md`, `prereg_robustez_arranjo.yaml`, `M_robustez_arranjo*.md` e os agregados `data/arranjo_corte_*` e `data/robustez_arranjo_*`. Este complemento não corrige retroativamente os registros, não altera fingerprints e não transforma análises de sensibilidade em resultado principal.

## S8c: testes registrados, reprodução e apresentação

### M1: alteração do teste registrado

O arquivo `tests/test_arranjo_corte.py`, cujo hash registrado começa por `8d30e41`, foi alterado no commit `893a2bb`; o hash da versão posterior começa por `1f5a0ef`. Não se trata, portanto, de preservação integral do código registrado. Na reprodução posterior da S3, `scripts/arranjo_paths.py` confere os bytes dos testes recuperados do commit de registro, em vez de exigir que os testes atuais sejam idênticos aos registrados. Os dados e o código numérico continuam sujeitos às verificações de identidade, mas essa opção distingue a reprodução da avaliação histórica de uma nova execução da S1. Para repetir a S1 com as identidades registradas deve-se utilizar o checkout de `d526c3f`, com os dados privados e as dependências correspondentes, não o estado atual da branch.

### M2: contratos substituídos na auditoria posterior

Em `tests/test_arranjo_corte_s3.py`, a comparação exata do comparador a posteriori com a cópia local foi substituída pela verificação de que o script está no git. Essa verificação assegura a disponibilidade do código, não sua identidade com a versão anteriormente utilizada. No mesmo arquivo, a conferência das taxas passou a aceitar os bytes equivalentes em LF e CRLF. A tolerância não permite mudança nos valores, mas é menos estrita que a igualdade do hash original. O `.gitattributes` agora permite exigir a igualdade exata em um teste complementar. O teste da S3 está fixado em `data/arranjo_corte_metadata.json` e no registro da robustez; não foi editado na S8c. Um teste novo confere diretamente os hashes do comparador e das taxas constantes dos metadados, sem normalização dos fins de linha.

### M3: separação editorial e limites da restauração

As tabelas de comparação de equidade em `text/latex/tables/` já continham alterações de apresentação introduzidas por `scripts/tabelas_equidade_comparacao.py`; não eram saídas literais do gerador original de equidade. O teste de regeneração dessas versões, em `tests/test_equidade_arranjo.py`, está fixado no manifesto de equidade. A restauração das tabelas às saídas anteriores quebraria esse contrato congelado. Preservam-se essas versões de regressão e documenta-se a impossibilidade de restaurá-las nesta fatia sem modificar o teste fixado. O gerador mantém a interface de reprodução exigida por esse contrato, mas sua execução editorial padrão grava apenas em `tables/editorial/`, com substituições verificadas uma a uma.

A tabela `tab_cap8_cascata_v31.tex` também recebeu alteração manual de legenda. O gerador `fechar_refrec_final.py` ainda procurava a legenda anterior, sem verificar se a substituição ocorrera. Como essa tabela não está fixada por hash, a S8c restaura sua legenda histórica e confere a reprodução pelo gerador, que passa a recusar legenda ausente ou ambígua. A apresentação de comparação é gerada em arquivo editorial separado. Os testes novos conferem a legenda esperada, as contagens regeneradas dos CSVs e a falha diante de alvos ausentes ou repetidos. Nos capítulos são alteradas somente as linhas de inclusão das tabelas correspondentes.

### m5: teste da robustez congelado

`tests/test_arranjo_robustez.py` está fixado no pré-registro e nos metadados da robustez. O laço que reaplica o estimador às mesmas entradas não foi alterado. O teste complementar da S8c observa as entradas efetivamente recebidas pelo estimador durante execuções sintéticas do caminho de produção, conferindo a exclusão das respostas novas e as partições utilizadas. O contrato de texto em `tests/test_s6a.py`, que não está fixado, passa a exigir a qualificação completa do cenário, sem modificar o resumo da tese nesta fatia.

### m8: origem da hipótese de duração

Os 20 minutos por potencial par aparecem em comentários da configuração `gzcmd/gzcmd_v3_config.yaml`, como denominador da equivalência de custo entre revisão humana e LLM; não constituem medida de duração do serviço. O gerador histórico `scripts/carga_abaixo_corte_horas.py` identifica esse valor por expressão regular nos comentários e recusa sua ausência ou ambiguidade. Essa dependência é documentada, sem alterar a configuração nem as saídas históricas. O novo gerador de esforço lê o valor de um campo explícito de `data/esforco_arranjo_parametros.json`, com a mesma origem e a natureza hipotética declaradas.

### Análises exploratórias e interpretação do esforço

O cenário de 2.156 potenciais pares pressupõe revisão em ordem decrescente de SCORE dentro de cada passo. Não há fonte para afirmar que essa tenha sido a ordem histórica do serviço; o número não é carga observada. O comparador de mesma carga, os diagnósticos de carga, VPP e mecanismos de perda, a decomposição de esforço e os FP por região são análises exploratórias posteriores à avaliação, não desfechos confirmatórios registrados. As horas expressam ordem de grandeza sob a hipótese de 20 minutos por par. A referência de acurácia do comparador O3 pressupõe revisor perfeito; a projeção com 5% de erro é apresentada separadamente. Não se demonstra superioridade em dados novos nem se avalia aqui a recomendação de encaminhar o MATCH do LLM ao revisor humano.
