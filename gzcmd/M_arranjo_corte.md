# Arranjo corte OpenRecLink + GZ-CMD abaixo do corte

## Pré-registro e separação temporal

`gzcmd/prereg_arranjo_corte.yaml` fixa o protocolo autorizado, os SHA-256 do
extrato, configuração, prompt, coortes e código, antes da simulação S1. O script
`scripts/arranjo_corte_s1_simular.py --preregister` apenas registra fontes e
parâmetros, sem ajustar modelos. O YAML deve ser commitado e enviado à branch
antes de executar S1 com seu `--prereg-sha256`. Divergências no YAML, código,
fontes ou commit interrompem a simulação. Emendas exigem consulta ao autor.

Os cortes são o menor SCORE observado entre os PAR=1 de cada passo e,
portanto, são derivados do rótulo, não estimativas independentes dele.
SCORE ≥ 7,02408276169 no passo 6,
≥ 7,84444860082 no passo 9 e ≥ 7,90191406713 no passo 11 aceita o potencial par.
A conferência autorizada exige exatamente 211 PAR=1 acima e 61.485 abaixo,
incluindo 36 PAR=2. O corte não usa a nota final do comparador.

## Classificador, calibração e braços

O novo módulo `gzcmd/arranjo_corte.py` importa `_classifier` e `numeric_features`
de `ml_oof.py`, sem alterar esse arquivo ou seu fingerprint. Mantém os 30
atributos, hiperparâmetros, seed 42 e cinco folds externos por REFREC, com cinco
folds internos, construídos na base inteira. A emenda técnica 2 restaura
`n_jobs=8`, sem sobrescrever nenhum parâmetro de `_classifier`, em todos os
braços. O treino dos braços não é executado em paralelo.
O braço principal P conserva os classificadores da base inteira e ajusta a
isotônica nas probabilidades OOF internas dos pares de treino abaixo do corte.
Nem rótulos nem grupos do teste externo entram no ajuste do calibrador.

P-Platt substitui a isotônica por Platt no logit da probabilidade bruta do
XGBoost, truncada em [0,000001; 0,999999]. Após a emenda técnica 1, usa os
alvos suavizados de Platt (1999): t+=(N++1)/(N++2) e t-=1/(N-+2). O novo
módulo ajusta a regressão logística com Newton e backtracking, recorrendo a
BFGS se necessário. Conserva a penalização da inclinação l2=0,001, o intercepto
sem penalização, max_iter=100 e tol=1e-10, sem alterar `calibration.py`.
Se um conjunto
de calibração tem menos de dez PAR=2, P usa Platt nesse fold e registra a
substituição. B1 usa a base inteira e a probabilidade REFREC original, mas
troca as taxas do revisor pelas taxas cross-fitted. A simulação confere folds
e probabilidades B1 contra `pcal_ml_oof.csv` antes de construir os funis.
Os testes de reprodução com máscara integral desligam P-Platt explicitamente
e impedem qualquer ajuste Platt com um gancho de teste. P e B1 seguem apenas
pela isotônica nesse teste; Platt é ajustado somente para P-Platt ou quando
a regra pré-registrada de poucos positivos exige sua substituição.

A3 restringe também os ajustes do classificador aos pares abaixo do corte,
mantendo os mesmos folds externos e internos. A outra sensibilidade conserva
P e estima e_fn somente com PAR=2, reunindo as quatro bandas [5,9). Nenhuma
sensibilidade gera chamadas adicionais na S2.

## Confiabilidade e política

As fontes congeladas são exatamente as três coortes Claude Haiku 5.5 v4 por
modo declaradas em `error_rates_by_band_v4.yaml`: histórica, suplemento por
linha e suplemento REFREC. Não entram coortes de fechamento ou respostas S2.
Cada fold recebe taxas estimadas somente com respostas de outros folds.
O estimador M3 usa FP/não vínculos, FN/vínculos e UNSURE/n, restaurando valores
fixos quando n<10 ou o denominador correspondente é zero. O caso de um único
fold serve apenas ao teste de equivalência com M3, nunca à execução real.
O manifesto publicado de taxas contém hashes anteriores à regravação final
de quatro checkpoints suplementares. O pré-registro preserva esses hashes
declarados e fixa separadamente os hashes dos arquivos efetivamente disponíveis,
com o mesmo modelo, prompt, campos removidos, tamanho e completude. O teste
M3 exige que suas respostas reproduzam exatamente as taxas publicadas por modo.

O motor é construído por `build_engine_from_config`, com os custos, orçamento
humano ilimitado e erro humano de 5% já configurados. As taxas cross-fitted
substituem o mapping do motor em memória antes da triagem por fold. A guarda
de nota ≥9 exige a probabilidade recalibrada ≥0,9. O orçamento total por modo
é conferido após reunir os folds; ultrapassagem exige decisão do autor.

Não há restrição implementada de cardinalidade N:1 nem resolução de conflitos
por margem entre probabilidades. A seção `consistency` da configuração é
reservada, não implementada; seus valores não controlam decisões do motor.
As taxas esperadas de FP não constituem uma restrição de unicidade de registros.

## Cache e dados privados

O cache exige modo, pair_id, modelo exato `claude-haiku-5-5`, SHA-256 do prompt
v4, SHA-256 do dossiê e os seis campos removidos. O dossiê é reconstruído com
`run_v3(p_cal="fit_platt", policy_version="v3")`, `build_dossiers` e
`DossierView`, sem `model_outputs`. Cada hash é comparado à identidade gravada
no checkpoint completo e ao `identity.json` quando existente. As duas coortes
anteriores ao journal de identidade conservam a identidade por par embutida
nos próprios checkpoints, que são integralmente fixados por SHA-256; nenhum
hash ausente é inventado ou aceito. `load_responses` valida decisões,
completude e proveniência; qualquer fonte adicional ou colisão interrompe.

Os CSVs pendentes contêm apenas pair_id em `.local/arranjo_corte`, ignorado.
Escores e ações por par para a S3 também permanecem nesse diretório privado.
Nenhum dossiê, dado identificador ou rótulo por par é versionado.

## S1, parada e S2

S1 grava somente contagens modo × braço × banda × ação, taxas congeladas por
fold e metadados de proveniência. Não calcula desfechos de acerto, FP/FN da
cascata, sensibilidade ou especificidade. A estimativa de confiabilidade é
parte do ajuste previamente especificado, não uma avaliação da cascata.

Pendentes são a união, por modo, dos pares selecionados por P e B1 sem cache
válido. Mais de 500 pendentes ou custo estimado maior que US$ 2 interrompe o
protocolo e exige consulta. O custo usa 6.847 tokens de entrada e 3.904 de
saída por par, US$ 0,10/0,50 por milhão e margem de 20% para tentativas.

Dentro da regra, S2 usa o runner existente com `--pares` e `--allow-new-cohort`,
prompt v4, Anthropic e até quatro workers, com os modos em sequência. Mantém
checkpoint incremental, backoff na substituição de arquivos e journal de
tokens por chamada. A execução só é aceita com `complete: true`, zero erros
e auditoria de segredos com zero ocorrências. As novas respostas nunca
atualizam as taxas congeladas. Se não há pendentes, S2 é dispensada.

Os desfechos primários ficam para S3: entre os 36 PAR=2 abaixo do corte,
aceitos automaticamente, encaminhados ao revisor humano e perdidos, com IC
de Wilson de 95%; métricas na base inteira e em [5,9) por modo. Encaminhamento
humano não é acerto observado; a taxa humana de 5% é somente um cenário.

## Emenda técnica 1: diagnóstico anterior à repetição da S1

O primeiro treino da S1 parou antes de produzir ações ou avaliar desfechos da
cascata. No fold externo 0, o conjunto de calibração abaixo do corte continha
49.189 potenciais pares: 32 positivos e 49.157 negativos. Nenhuma probabilidade
bruta era zero ou um. Todos os logits recortados eram finitos e sua variância
era 2,9673083611. Os intervalos por classe se sobrepunham: positivos de
-11,6973123081 a 10,7274274605 e negativos de -13,8155095580 a 9,9969190053.
Não havia separação linear perfeita nem variância nula.

O Newton sem controle de passo passou de intercepto -7,3215 e inclinação zero
para intercepto 10.796.318,0483 e inclinação 1.082.103,7950 na iteração 3.
Todos os 49.189 valores previstos saturaram em zero ou um; a curvatura do
intercepto e o determinante da Hessiana se tornaram zero. Portanto, a
singularidade foi uma falha numérica de otimização sob desbalanceamento,
não evidência de separação perfeita no conjunto de calibração.

A emenda foi autorizada antes de repetir S1. Não foram examinados desfechos
com rótulos da cascata, métricas de acerto ou resultados S2. Rótulos foram
usados exclusivamente no treinamento e no diagnóstico autorizado. Objetivo
científico, braços, folds, seed, custos, regras de parada e parâmetros
registrados permanecem fixos; a formulação suavizada autorizada estabiliza o
ajuste Platt. Os hashes novos ficam registrados no YAML da emenda.

## Emenda técnica 2: reprodução do OOF e número de threads

Após estabilizar Platt, a repetição da S1 parou na conferência das
probabilidades OOF contra `data/pcal_ml_oof.csv`, antes de gerar ações,
pendentes ou métricas de acerto. O diagnóstico com `nested_oof_probabilities`
e os mesmos dados, atributos, folds e seed encontrou:

| n_jobs | Folds iguais | Probabilidades divergentes a 1e-14 | Maior diferença absoluta |
|---:|:---:|---:|---:|
| 8, classificador original | sim | 0 | 1,1102230246251565e-16 |
| 2, sobrescrita inicial | sim | 41.194 | 0,44052287581699345 |

O `hist` do XGBoost neste ambiente exige o mesmo número de threads para
reproduzir os resultados publicados. A emenda autorizada remove a sobrescrita
de `n_jobs` exclusivamente do módulo novo e usa `_classifier` sem alterações,
restaurando suas oito threads em todos os braços. A comparação é de
probabilidades e folds, não de acerto contra rótulos. Nenhum desfecho de
avaliação da cascata foi examinado; S2 ainda não havia sido iniciada.

Antes de treinar, a RAM livre é conferida com
`Get-CimInstance Win32_OperatingSystem`. Abaixo de 6 GiB, somente workers
Python do miniconda com `joblib.externals.loky` e pai inexistente podem ser
encerrados; a quantidade é registrada localmente. Os demais processos são
preservados e os braços permanecem sequenciais. Os novos hashes e o
diagnóstico agregado ficam na seção de emendas do pré-registro antes da S1.
