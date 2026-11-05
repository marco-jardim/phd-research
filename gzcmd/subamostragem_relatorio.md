# Subamostragem no treino — análise exploratória post hoc

Análise autorizada pelo autor, não registrada. Nenhum texto da tese é alterado.
Sem chamadas a LLM ou APIs. A avaliação mantém a prevalência real.

## Fonte da correção de prior

Dal Pozzolo, Caelen, Johnson e Bontempi (2015), *Calibrating Probability
with Undersampling for Unbalanced Classification*, IEEE SSCI, pp. 159–166.
DOI: 10.1109/SSCI.2015.33.

PDF: <https://dalpozz.github.io/static/pdf/SSCI_calib_final_noCC.pdf>.
SHA-256: `263540d93a6854c0bcc59d5d86396b4e030d435683b7ff83f57327ea1a9a7055`.
O PDF e a extração integral ficam somente em `.local/fontes/` do repositório
principal; não são versionados.

Na página física 3, seção III-A:

> Let us denote β = p(s = 1|−) as the probability of selecting a negative
> instance with undersampling

Na mesma página, seção III-C:

> Let us call p′ the bias-corrected probability obtained from ps using (4):

A equação (9) é `p′ = β ps / (β ps − ps + 1)`. A relação inversa aparece
na equação (2), página física 2, e na forma simplificada da equação (3),
página física 3: `ps = p / (p + β (1 − p))`. O teste sintético verifica a
ida e volta, inclusive nas probabilidades 0 e 1.

## Desenho e interpretação

- Mesmos 30 atributos e `model_factory` do original; semente do XGBoost 42.
- Mesma função original de OOF aninhado REFREC 5×5, sem editar código congelado.
- A remoção atua somente em `fit` dos modelos internos e externos, mantendo
  todos os positivos. A isotônica usa todos os rótulos do OOF interno.
- Razões positivos:negativos 1:1, 1:5, 1:20 e 1:50, sementes de remoção
  1000–1009; referência sem remoção, executada uma vez.
- `scale_pos_weight` continua calculado pela fábrica original, sobre os
  negativos e positivos efetivamente usados naquele ajuste. Não é fixado
  artificialmente na razão da base inteira.
- Avaliação abaixo do corte: 61.485 pares, 36 PAR=2; base inteira: 61.696,
  247 vínculos. Nenhum subconjunto de avaliação é subamostrado.
- Ordenação decrescente da probabilidade, desempate por hash lexical interno,
  exatamente k pares, como no comparador de mesma carga; hashes não são exportados.
- AUC-PR é average precision, não área trapezoidal. ECE usa os decis de
  probabilidade, remove bordas duplicadas e não divide empates entre bins.
- Sensibilidades por passo/sexo referem-se aos membros do **top-96 global**
  abaixo do corte, não a uma nova seleção de 96 por subgrupo. Sexo segue
  `_recode_sex`, usado pela análise de equidade existente; grupos MISSING
  e com menos de 30 pares têm sensibilidade indefinida, com cobertura registrada.

As variantes são `raw`, `raw_prior`, `isotonic` e `isotonic_prior`.
`raw_prior` aplica a equação (9) à probabilidade bruta, usando a retenção
efetiva dos negativos no treino externo. Como XGBoost usa pesos de classe,
a probabilidade bruta não é garantidamente a posterior sob o prior amostrado.
`isotonic_prior` é **somente um diagnóstico de dupla correção**: a isotônica
já foi ajustada à prevalência real. Não se presume que aplicar novamente a
correção melhore sua calibração. Os betas diferem entre folds; portanto a
transformação monotônica por fold não garante invariância da ordenação OOF
quando os folds são reunidos.

Os agregados incluem mediana, mínimo–máximo e diferenças para a referência
da mesma variante. Não são intervalos de confiança. O gabarito compartilha
os vieses de verificação/incorporação das análises anteriores. A análise
é pós-hoc e contém apenas 36 positivos abaixo do corte; não permite conclusão
causal ou confirmatória.

## Execução e resultados

Executar `pwsh -NoProfile -File scripts/subamostragem_treino.ps1`.
O executor verifica versões e RAM, encerra somente loky comprovadamente
órfãos, inicia um processo por réplica em sequência, registra stdout/stderr,
espera explicitamente e usa checkpoints atômicos com backoff.

A projeção conservadora inicial é 41 vezes o tempo medido da referência.
Se exceder três horas, o executor para, preservando o checkpoint, sem
publicar uma análise parcial como completa. Há também limite de três horas
de tempo de parede para a execução.

### Gate e execução observada

Referência reproduzida: erro absoluto máximo **1,1102230246251565e-16**,
inferior à tolerância 1e-14; folds iguais aos do CSV original. Foram concluídas
41 réplicas: uma referência e dez sementes por proporção. Versões verificadas:
sklearn 1.8.0 e xgboost 3.2.0. Soma dos tempos medidos nos workers:
1.371,25 segundos (22,85 minutos). A projeção conservadora inicial foi
3.332,52 segundos, inferior a três horas. O primeiro launcher foi interrompido
pelo timeout externo de 900 segundos; a retomada com timeout de três horas
reutilizou os checkpoints válidos e concluiu sem sobreposição de réplicas.

Os três CSVs finais são agregados e não contêm `pair_id`. A base inteira
também está registrada nos CSVs; as tabelas abaixo se referem exclusivamente
aos 61.485 pares abaixo do corte.

### Ordenação isotônica (comparação principal com O3)

Células: **mediana [mínimo; máximo]; Δ da mediana para a referência**.
Medianas fracionárias de contagens são possíveis porque há dez sementes.
Δ de sensibilidade é fração, não ponto percentual. A referência não tem
variação de remoção e foi executada uma vez.

| Métrica | Referência | 1:1 | 1:5 | 1:20 | 1:50 |
|---|---:|---:|---:|---:|---:|
| AUC | 0,984917 | 0,987740 [0,983299; 0,991905]; +0,002823 | 0,994952 [0,988613; 0,995847]; +0,010035 | 0,991778 [0,989117; 0,995105]; +0,006861 | 0,991334 [0,978155; 0,993375]; +0,006417 |
| AUC-PR/AP | 0,283119 | 0,148406 [0,103306; 0,188314]; −0,134713 | 0,203833 [0,163832; 0,329452]; −0,079285 | 0,249598 [0,213202; 0,303347]; −0,033521 | 0,299798 [0,203751; 0,385763]; +0,016679 |
| PAR=2 top-17 | 6 | 4,5 [2; 6]; −1,5 | 6 [4; 8]; 0 | 5,5 [4; 8]; −0,5 | 7,5 [4; 10]; +1,5 |
| PAR=2 top-29 | 9 | 5 [4; 8]; −4 | 8 [6; 11]; −1 | 9 [7; 10]; 0 | 9,5 [7; 14]; +0,5 |
| PAR=2 top-96 | 22 | 13 [8; 17]; −9 | 18,5 [16; 21]; −3,5 | 21 [18; 22]; −1 | 22,5 [20; 24]; +0,5 |
| PAR=2 top-125 | 23 | 14 [10; 18]; −9 | 21 [17; 23]; −2 | 23 [22; 25]; 0 | 24 [20; 25]; +1 |
| k mínimo para 9 | 29 | 54 [33; 105]; +25 | 36 [19; 41]; +7 | 28,5 [18; 39]; −0,5 | 23,5 [11; 34]; −5,5 |
| k mínimo para 23 | 125 | 413 [279; 547]; +288 | 148,5 [109; 191]; +23,5 | 117,5 [98; 149]; −7,5 | 96 [75; 159]; −29 |
| k mínimo para 36 | 23.443 | 7.884,5 [2.426; 11.774]; −15.558,5 | 2.436,5 [1.390; 7.359]; −21.006,5 | 8.491,5 [1.824; 15.297]; −14.951,5 | 7.178,5 [5.179; 51.474]; −16.264,5 |

### Calibração: bruto, isotônico e bruto com correção de prior

Δ é em relação à referência da **mesma variante**, sem remoção. A referência
bruta e `raw_prior` coincide porque β=1. Menor Brier/ECE é melhor.

| Variante / métrica | Referência | 1:1 | 1:5 | 1:20 | 1:50 |
|---|---:|---:|---:|---:|---:|
| raw / Brier | 0,002293174 | 0,018151874 [0,014525440; 0,021617456]; +0,015858699 | 0,006843101 [0,005381972; 0,008564697]; +0,004549927 | 0,003684825 [0,003368786; 0,003983740]; +0,001391651 | 0,002834477 [0,002634761; 0,003028176]; +0,000541302 |
| raw / ECE | 0,003493360 | 0,034627941 [0,029621929; 0,037760274]; +0,031134581 | 0,012294021 [0,010431945; 0,013825525]; +0,008800661 | 0,006171194 [0,005981911; 0,006730431]; +0,002677834 | 0,004614621 [0,004253008; 0,004795654]; +0,001121261 |
| isotonic / Brier | 0,000629278 | 0,000999415 [0,000749517; 0,001233926]; +0,000370137 | 0,000806474 [0,000623237; 0,001016081]; +0,000177196 | 0,000698104 [0,000607893; 0,000845064]; +0,000068826 | 0,000647747 [0,000526626; 0,000806252]; +0,000018469 |
| isotonic / ECE | 0,000587550 | 0,001160620 [0,000614016; 0,001973145]; +0,000573070 | 0,000811367 [0,000474027; 0,001225011]; +0,000223817 | 0,000684948 [0,000549562; 0,000780885]; +0,000097398 | 0,000623345 [0,000545880; 0,000806231]; +0,000035795 |
| raw_prior / Brier | 0,002293174 | 0,000777799 [0,000674127; 0,001094731]; −0,001515375 | 0,001314872 [0,000920194; 0,001559355]; −0,000978302 | 0,001346028 [0,001231736; 0,001477782]; −0,000947146 | 0,001414748 [0,001353112; 0,001577084]; −0,000878427 |
| raw_prior / ECE | 0,003493360 | 0,001062782 [0,000736852; 0,002095352]; −0,002430578 | 0,001726732 [0,001136256; 0,002320229]; −0,001766628 | 0,001796847 [0,001698178; 0,001953221]; −0,001696513 | 0,002080280 [0,001940087; 0,002291684]; −0,001413080 |

### Sensibilidade no top-96 global, por passo e sexo

Valores em fração; mesmo formato mediana [mínimo; máximo]; Δ. Denominadores
positivos abaixo do corte: passo 6 = 4, passo 9 = 7, passo 11 = 25;
sexo F = 10, sexo M = 26. Ausências e cobertura constam no CSV.

| Variante / grupo | Referência | 1:1 | 1:5 | 1:20 | 1:50 |
|---|---:|---:|---:|---:|---:|
| isotonic / passo 6 | 0,5 | 0,25 [0; 0,25]; −0,25 | 0,25 [0,25; 0,5]; −0,25 | 0,5 [0,5; 0,75]; 0 | 0,5 [0,5; 0,75]; 0 |
| isotonic / passo 9 | 0,571429 | 0,285714 [0,142857; 0,428571]; −0,285714 | 0,5 [0,428571; 0,571429]; −0,071429 | 0,428571 [0,428571; 0,571429]; −0,142857 | 0,571429 [0,428571; 0,571429]; 0 |
| isotonic / passo 11 | 0,64 | 0,38 [0,24; 0,52]; −0,26 | 0,54 [0,48; 0,64]; −0,10 | 0,64 [0,52; 0,64]; 0 | 0,66 [0,56; 0,72]; +0,02 |
| isotonic / sexo F | 0,7 | 0,6 [0,4; 0,7]; −0,1 | 0,7 [0,5; 0,7]; 0 | 0,7 [0,5; 0,7]; 0 | 0,7 [0,7; 0,7]; 0 |
| isotonic / sexo M | 0,576923 | 0,25 [0,076923; 0,384615]; −0,326923 | 0,461538 [0,346154; 0,538462]; −0,115385 | 0,538462 [0,5; 0,576923]; −0,038462 | 0,596154 [0,5; 0,653846]; +0,019231 |
| raw / passo 6 | 0,5 | 0,25 [0; 0,25]; −0,25 | 0,25 [0,25; 0,25]; −0,25 | 0,5 [0,5; 0,75]; 0 | 0,75 [0,5; 0,75]; +0,25 |
| raw / passo 9 | 0,571429 | 0,285714 [0,142857; 0,428571]; −0,285714 | 0,428571 [0,428571; 0,571429]; −0,142857 | 0,5 [0,285714; 0,571429]; −0,071429 | 0,571429 [0,428571; 0,571429]; 0 |
| raw / passo 11 | 0,64 | 0,42 [0,28; 0,48]; −0,22 | 0,54 [0,44; 0,60]; −0,10 | 0,62 [0,52; 0,72]; −0,02 | 0,66 [0,56; 0,68]; +0,02 |
| raw / sexo F | 0,7 | 0,6 [0,4; 0,7]; −0,1 | 0,7 [0,4; 0,7]; 0 | 0,7 [0,5; 0,7]; 0 | 0,7 [0,7; 0,7]; 0 |
| raw / sexo M | 0,576923 | 0,288462 [0,153846; 0,346154]; −0,288462 | 0,423077 [0,384615; 0,5]; −0,153846 | 0,538462 [0,461538; 0,692308]; −0,038462 | 0,634615 [0,461538; 0,653846]; +0,057692 |
| raw_prior / passo 6 | 0,5 | 0,25 [0; 0,25]; −0,25 | 0,25 [0,25; 0,25]; −0,25 | 0,5 [0,5; 0,75]; 0 | 0,75 [0,5; 0,75]; +0,25 |
| raw_prior / passo 9 | 0,571429 | 0,285714 [0,142857; 0,428571]; −0,285714 | 0,428571 [0,428571; 0,571429]; −0,142857 | 0,5 [0,285714; 0,571429]; −0,071429 | 0,571429 [0,428571; 0,571429]; 0 |
| raw_prior / passo 11 | 0,64 | 0,42 [0,28; 0,48]; −0,22 | 0,54 [0,44; 0,60]; −0,10 | 0,62 [0,52; 0,72]; −0,02 | 0,68 [0,56; 0,68]; +0,04 |
| raw_prior / sexo F | 0,7 | 0,6 [0,4; 0,7]; −0,1 | 0,7 [0,4; 0,7]; 0 | 0,7 [0,5; 0,7]; 0 | 0,7 [0,7; 0,7]; 0 |
| raw_prior / sexo M | 0,576923 | 0,288462 [0,153846; 0,346154]; −0,288462 | 0,423077 [0,384615; 0,5]; −0,153846 | 0,538462 [0,461538; 0,692308]; −0,038462 | 0,653846 [0,461538; 0,653846]; +0,076923 |

### Leitura exploratória

Subamostrar mais agressivamente (1:1 e 1:5) não melhora a ordenação principal
nos primeiros 96 pares, embora a mediana da AUC aumente. AUC isolada mascara
a perda na região operacional: no isotônico, 1:1 recupera mediana 13 em vez de
22 PAR=2 no top-96. Em 1:50 há pequeno ganho mediano (22,5), mas amplitude
20–24 e AUC-PR 0,204–0,386; não é melhora uniforme entre sementes.

O alcance dos 36 PAR=2 pode melhorar sem melhorar o início da fila: 1:5
tem k mínimo mediano 2.436,5, mas recupera apenas 18,5 no top-96. A variabilidade
de k mínimo para 36 em 1:50 (5.179–51.474) desaconselha escolher uma semente
favorável como se fosse uma regra geral. O SCORE histórico (k=2.521 para os
36, na comparação anterior) é contexto descritivo, não critério de seleção
de razão ou semente.

A correção analítica reduz bastante Brier/ECE em relação ao bruto
subamostrado, mas não supera a referência isotônica em Brier mediano.
A própria isotônica subamostrada tem Brier/ECE medianos piores que a
referência em todas as proporções. Não há evidência descritiva de uma razão
que melhore simultaneamente e de forma estável a região operacional e a
calibração da referência. Não foram feitos testes inferenciais nem seleção
de hiperparâmetros a partir desses resultados.

Os CSVs contêm ainda todos os top-k solicitados, métricas das quatro variantes,
diferenças de mínimo/mediana/máximo para suas referências, denominadores de
subgrupos, base inteira e curvas até 2.522 com média e amplitude entre sementes.

### Verificação reproduzível

`pwsh -NoProfile -File scripts/verificar_subamostragem.ps1` executa
`python -m pytest tests -q -rs` no Windows e em clone limpo, registrando HEAD,
status inicial/final e logs em `.local/subamostragem/`. A execução Windows
inclui o status da árvore compartilhada e não é apresentada como árvore limpa;
o clone limpo fornece a evidência atribuível ao commit. Resultados efetivos,
hash do commit e checks do PR são reportados separadamente após a execução,
evitando alterar o próprio commit apenas para inserir seu hash no arquivo.
