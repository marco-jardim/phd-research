# Robustez não confirmatória do arranjo

P permanece o braço principal. A comparação P versus P-Platt não confirma um
braço alternativo, não mede generalização e não quantifica variabilidade amostral.
As respostas do modelo de linguagem são compartilhadas entre réplicas. Os vieses
do padrão-ouro permanecem.

## Ordem de execução

1. Registrar hashes do código, testes, fontes e desenho aprovado, commit e push.
2. Executar R0 em processo filho. Conferir probabilidades, folds, taxas, políticas
   e reprodução dos desfechos previamente publicados, sem divulgar novos desfechos.
3. Executar as demais 39 réplicas, sequencialmente, sem API e sem novas métricas
   com rótulo. Os únicos rótulos empregados nesta fase pertencem ao treinamento,
   à calibração e à estimação das taxas nas três coortes congeladas.
4. Congelar a fila única de cada modo, commit e push da nota e das filas.
5. Executar o modelo Claude Haiku 5.5, instruções v4, no máximo quatro workers,
   modos sequenciais. Auditar identidade do dossiê, erros e segredos.
6. Avaliar os desfechos apenas na fase C. Respostas novas nunca estimam taxas.

## Grade e critérios

Bloco N: fold_seed 42, xgb_seed 42 e 101 a 109, cada qual com n_jobs 8 e 2,
nessa ordem. Bloco P: xgb_seed 42, n_jobs 8 e fold_seed 1000 a 1190 de dez em
dez. Cada bloco tem 20 réplicas, incluindo R0 no bloco N.

Dominância fraca com ao menos uma desigualdade estrita e tolerância zero no
vetor (aceitos maior, FP abaixo do corte menor, perdidos menor, potenciais pares
ao humano menor, potenciais pares ao LLM menor). Estável: pelo menos 18/20 em
ambos os blocos e dominância inversa em no máximo 1/20. Não sustentada: menos
de 10/20 em ambos, ou inversa em pelo menos 2/20 em algum bloco. Mista: demais
casos. Réplicas inválidas permanecem no denominador como não dominantes.
Clopper-Pearson de 95% tem interpretação descritiva, não confirmatória.

Secundário na vigilância: P menos P-Platt em FP maior que zero em pelo menos
18/20 de cada bloco. PAR=2: partição dos 36 vínculos; aceitação de P-Platt pelo
menos igual a P; perdas maiores que o a posteriori fixo (9/17 por modo).
Estabilidade sem rótulo: diferenças de probabilidade acima de 1e-14, máximo,
mudanças de ação abaixo do corte, Jaccard do LLM e folds com fallback.

## Segurança e retomada

Completar a grade, sem truncar. Parar acima de 2.000 potenciais pares novos,
US$ 6,50 ou projeção de oito horas. Custo estimado inclui margem de 20%.
Processos filhos sequenciais com Start-Process, logs locais e espera explícita.
Abaixo de 6 GiB livres, encerrar apenas processos loky órfãos; abaixo de 3 GiB,
parar. Checkpoint local com identidade completa e substituição atômica com
backoff. Identidade divergente exige novo cálculo. Nenhum dado de campo ou
dossiê é gravado nos artefatos públicos.
