# Sensibilidade dos classificadores do QW1

Somente agregados. O extrato local e os artefatos originais em `data/qw1/`
nao integram este diretorio nem sao alterados pelo executor.

Execucao a partir da raiz do repositorio, em ambiente com PyTorch CUDA,
pytorch-tabnet, XGBoost, scikit-learn, imbalanced-learn e pandas 2:

```pwsh
python scripts/qw1_sensibilidade_gpu.py --reference-dir D:/git/phd-research/data/qw1 --holdout-only
```

`--holdout-only` reproduz o escopo desta entrega, sem nova validacao cruzada.
Sem essa opcao, o executor tambem realiza a CV opcional, retomando os
agregados existentes. Nao reutilizar um diretorio para
outro extrato ou outras versoes das bibliotecas. Para refazer todos os treinos,
indicar um novo diretorio com `--output`.

## Protocolo

* A/B: particao estratificada por linha (`split_data`), 58/56 atributos.
* C/D: particao agrupada por REFREC, 58/56 atributos, sem grupos compartilhados.
* Retirada exclusiva de `obito_tb` e `obito_sinan` em B/D.
* Teste nominal de 30%, semente 42 e probabilidade inclusiva de 0,5.
* Random Forest em CPU. MLP: produtos matriciais e retropropagacao CUDA em
  float64, com inicializacao, Adam, embaralhamento e parada do scikit-learn
  em CPU. TabNet e XGBoost treinados em CUDA, sem fallback silencioso.
* A0: controle adicional com a particao por linha do script comparativo
  original. Nao e um quinto cenario de sensibilidade. A/B empregam outro
  gerador aleatorio, apesar da mesma semente. Divergencias de F1 superiores
  a 0,01 sao registradas, nao substituidas nem ocultadas.
* CV 5x5 somente na parcela de treinamento. Em C/D, cada fold externo da CV
  preserva REFREC; sementes 42 a 46. Nas MLP, a validacao interna de parada
  continua por linha, conforme o algoritmo historico, sem acesso ao teste
  externo. TabNet preserva 100 epocas sem conjunto interno de validacao.
* Nesta entrega, os quatro testes independentes estao completos (20 ajustes),
  mas nao foi executada nova CV. A CV 5x5 historica e mantida como referencia
  em `_meta.json`, nao como resultado desta reexecucao. O executor suporta CV
  opcional: se a estimativa de 25 ajustes TabNet exceder 30 minutos por cenario,
  a CV fica restrita a A e D. Esse ramo nao foi executado nesta entrega.

## Arquivos

`holdout.csv` contem uma linha por modelo/cenario, com F1, precisao,
sensibilidade, AUC, precisao media (AP), VP/FP/FN/VN, denominadores e tempo.
`controle_particao_original.csv` documenta A0 e a comparacao historica.
Na execucao opcional de CV, `cv_folds.csv` contem somente agregados por fold,
nao predicoes individuais, e `cv_summary.csv` resume os 25 folds por
modelo/cenario, com desvio padrao populacional. Esses dois arquivos nao foram
produzidos nesta entrega. `_meta.json` registra GPU, driver, versoes, parametros,
hashes, ausencia de nova CV e tempos. Os tempos nao pressupõem GPU exclusiva
nem permitem comparacao de velocidade com a L4 historica.

O checkpoint recuperado continha os 20 ajustes A/B/C/D completos e tres dos
cinco ajustes A0, mas seus metadados ainda nao estavam finalizados. A retomada
executou somente TabNet e XGBoost em A0. A soma dos tempos dos ajustes completos
e recuperavel; o tempo integral de parede nao e, pois falta o registro de uma
retomada anterior. `contabilidade_tempos` explicita essa restricao, sem atribuir
ao total integral os 900 segundos da primeira chamada interrompida.

O manifesto historico registra Python 3.12.10, scikit-learn 1.8.0,
PyTorch 2.10.0, XGBoost 3.2.0 e NumPy 1.26.4. A execucao local utiliza
Python 3.13.11, scikit-learn 1.9.1, PyTorch 2.11.0+cu128, XGBoost 3.4.1
e NumPy 2.5.4. O teste A emprega outro gerador de particao que o QW1 original;
A0 controla essa diferenca, mas nao isola versoes e backend. As duas MLP em A0
reproduzem exatamente o F1 historico. As demais comparacoes, inclusive as
divergencias acima de 0,01, constam dos agregados e metadados.

Em A0, XGBoost apresentou F1 0,911765 contra 0,919708 historico, diferenca
-0,007943, dentro da tolerancia. Random Forest (0,890511 contra 0,854962)
e TabNet (0,711864 contra 0,610169) permaneceram fora dela, com diferencas
+0,035549 e +0,101695. A e A0 compartilham 5.606 linhas de teste e 19 vinculos.
Nao se atribuem as diferencas remanescentes exclusivamente a versoes ou GPU.

O script gera `text/latex/tables/tab_sensibilidade_qw1.tex`. Nenhuma dessas
saidas contem identificadores, campos individuais ou linhas de potenciais pares.
