# Nota técnica da execução da robustez

O pré-registro e seu código foram enviados em
`3ade1c73eb0cea5c36b0cfa19693d49d198c2ec9`, antes do primeiro treino.
A fila foi enviada em `47b5b59f46cde8e6047d5a90ce87fcdd35180b26`, antes
da primeira chamada à API. As 40 réplicas terminaram, com os gates de R0
aprovados e sem necessidade do fallback de execução com A3.

Depois das fases A, B e C, o pytest completo encontrou uma incompatibilidade
administrativa no reprodutor de equidade: sua checagem de proveniência exigia
os bytes históricos do validador do suplemento e do teste S3, embora a
correção da identidade do dossiê e a atualização do teste estivessem registradas
antes dos treinos. Foram 412 testes aprovados, um com falha e um ignorado.

A tentativa inicial de compatibilização foi descartada. Foram restaurados
integralmente os bytes de `scripts/arranjo_corte_s2_suplementar.py`,
`scripts/equidade_arranjo.py` e `tests/test_arranjo_corte_s3.py` do commit
`893a2bb`. Nenhuma exceção ou relaxamento permanece nos fingerprints
históricos. A conferência da identidade do dossiê foi transferida para
`gzcmd/arranjo_robustez.py`, chamada como wrapper pelo script novo, com os
mesmos testes de rejeição de identidade divergente.

A emenda técnica 1 do pré-registro da robustez documenta a transferência,
os hashes anteriores e atuais e a identidade original da execução. O histórico
não foi reescrito. Não há mudança de parâmetro, desenho, taxa, política, treino,
resposta ou resultado. Os resultados originais continuam vinculados ao registro
enviado antes do treino; a emenda permite auditar a correção sem repetir a API.

Os logs de execução e do pytest, com o HEAD correspondente, permanecem em
`.local/arranjo_robustez/` e `.local/robustez/`. Os artefatos agregados das
fases constam em `data/robustez_arranjo_*.csv` e no JSON de metadados.

## Auditoria após a transferência

`data/robustez_arranjo_auditoria_emenda.json` registra a conferência do wrapper
novo: 2.080 respostas em cache na vigilância, incluindo os 230 pares da fase B,
e 488 na confirmação, incluindo os 12 pares da fase B. As decisões são iguais
às usadas na execução original. Todos os fingerprints históricos são exatos e
todos os hashes dos resultados da robustez e da tabela permaneceram iguais.
Não houve novo treino ou chamada à API. Os hashes dos resultados de referência
continuam `d5f9f3c790802c8dcf7bb2d50c333390c5523f002aafdd7315e136e1c9945d16`
para métricas e `2f762158f44c9d5ceaa04c5788878e51e6d0c22ea65700e064a0a82d652b9f0e`
para PAR=2.

## Resultados não confirmatórios

Fonte: `data/robustez_arranjo_resumo.csv` e os CSVs completos de diferenças,
PAR=2 e estabilidade. Cada bloco contém 20 réplicas válidas.

| Modo | P-Platt domina P, N | P-Platt domina P, P | Inversa, N/P | Classificação |
|---|---:|---:|---:|---|
| Vigilância | 7/20 | 4/20 | 0/20; 0/20 | não sustentada |
| Confirmação | 7/20 | 5/20 | 0/20; 0/20 | não sustentada |

O critério secundário de FP na vigilância foi atendido em 20/20 réplicas de
cada bloco. P permanece o braço principal. A ausência de dominância inversa
não equivale à dominância de P-Platt: há compensações entre desfechos.

Na base inteira, as diferenças P menos P-Platt têm mediana [mínimo; máximo]:

| Modo/bloco | Aceitos | FP abaixo | Perdidos | Pares ao humano | Pares ao LLM |
|---|---:|---:|---:|---:|---:|
| Vigilância/N | -1 [-2; 0] | 18 [14; 20] | -1 [-3; 2] | 16,5 [3; 26] | 221 [135; 289] |
| Vigilância/P | 0 [-4; 3] | 13 [8; 19] | -2 [-6; 2] | 17 [5; 28] | 170,5 [104; 209] |
| Confirmação/N | 0 [0; 0] | 1 [1; 1] | 0 [-2; 2] | 2 [-4; 13] | 21 [13; 30] |
| Confirmação/P | 0 [-2; 4] | 1 [-3; 4] | 0 [-5; 3] | 3,5 [-11; 15] | 4 [-10; 13] |

Nos 36 vínculos PAR=2, a proporção de réplicas com perdas maiores que o
a posteriori fixo foi 20/20 para ambos os braços em todos os blocos/modos,
exceto P na vigilância/bloco P, com 19/20. P-Platt aceitou pelo menos tantos
quanto P em 20/20 no bloco N de ambos os modos, 15/20 na vigilância/bloco P
e 12/20 na confirmação/bloco P. Os resumos são marginais por desfecho, não
partições de uma réplica hipotética formada pelas medianas.

Não houve fold com fallback. As proporções medianas de mudanças de ação abaixo
do corte em relação a R0 foram 0,001838 e 0,000390 (P e P-Platt) na
vigilância/N; 0,004269 e 0,001488 na vigilância/P; 0,000268 e 0,000057 na
confirmação/N; e 0,000724 e 0,000423 na confirmação/P. Os Jaccards medianos
dos conjuntos LLM_REVIEW foram, na mesma ordem, 0,726002 e 0,818566;
0,426895 e 0,474321; 0,784615 e 0,943732; 0,336667 e 0,475733.

## Tempo, custo e testes

A fase A durou 1.882,48 s, a B 1.257,15 s e a C 63,21 s. R0 exigiu
20,31 s de treino e a réplica com xgb_seed 42/n_jobs 2, 32,91 s.
Os 242 pares novos custaram US$ 0,6254985 pelas tarifas históricas aplicadas
aos tokens registrados, não por fatura. Os dois modos terminaram completos,
sem erros de pares, HTTP ou análise. A auditoria encontrou zero segredos em
494 arquivos. As respostas novas não reestimaram taxas.

Após a restauração integral dos arquivos herdados e a transferência da
validação, `python -m pytest tests -q -rs` apresentou 415 testes aprovados,
um ignorado por dependência opcional do QW1 e dois avisos preexistentes de
statsmodels, sem falhas, em 110,99 s. O HEAD e a saída são gravados em
`.local/robustez/` a cada execução. Nenhum teste histórico foi afrouxado.
