# M14: seleção opt-in de atributos na ablação do Experimento 1

`python -m gzcmd eval --feature-allowlist --calibration ml_rf` restringe o RF
aos 29 subescores do comparador e `nota final`, mais as colunas `macd_*` quando
MACD está habilitada. A lista explícita não inclui aliases duplicados,
identificadores, datas brutas, situação de encerramento nem o escore bruto.

`get_feature_columns` e `GZCMDClassifier` aceitam `feature_set="allowlist"` e
`macd_enabled=False` exclui as MACD mesmo em um DataFrame previamente enriquecido.
A seleção histórica permanece inalterada no default, inclusive sua ordenação.
Modelos salvos antes desta opção continuam carregáveis.

O experimento mantém o loader histórico, as sementes de divisão 42 a 46,
a semente interna 42 do RF e a calibração sigmoide de cinco partições no treino.
Não altera as taxas inline usadas por `evaluate_v3_dataframe`, nem o perfil
histórico da política. A avaliação usa apenas probabilidades no conjunto de teste.
`eval --policy-version v3` expõe o mesmo override histórico já oferecido pelo
runner, necessário porque o YAML atual seleciona v3.1. O default continua o YAML.
No override de v3.1 para v3, os limiares 3/10 são restaurados, assim como o custo
LLM 0,25 já implementado em `build_engine_from_config`.
A exclusão desses atributos não constitui prova de ausência de todos os tipos
de vazamento: `nota final` continua permitida, conforme o desenho solicitado.

## Reprodução dos artefatos

Executar `python scripts/exp1_refazer_allowlist.py`. O script chama a CLI real
(`python -m gzcmd eval`) para os dois braços históricos e compara cada coluna
numérica com o CSV versionado, por modo e seed. A tolerância máxima é 0,0005;
qualquer divergência maior interrompe antes da avaliação saneada. A única
diferença de esquema admitida é a coluna posterior `guardrails_enabled`,
obrigatoriamente verdadeira. O metadado registra as diferenças por coluna.

Somente depois dessa etapa são produzidos os CSVs saneados, resumos, testes t,
comparação e importâncias históricas. `exp_f2` é calculado das contagens esperadas
em ambos os modos; `exp_fbeta` na confirmação corresponde a beta 0,5, não a F2.
O percentual de redução de revisões usa a razão das médias; seu DP é o das cinco
reduções individuais e o p compara on/off por t pareado bilateral.

Os testes t também podem ser reproduzidos isoladamente, sem alterar o default:

```powershell
python scripts/exp1_testes_t.py --on data/ablation_comprec_macd_allowlist.csv --off data/ablation_comprec_nomacd_allowlist.csv --out data/exp1_testes_t_allowlist.csv
```

As importâncias usam as cinco RFs internas da calibração e as cinco seeds dos
splits, somente no treino histórico. Trata-se de diagnóstico de Gini,
potencialmente enviesado por cardinalidade e correlação, não de efeito causal.
Nenhum artefato contém linhas individuais ou valores dos campos dos registros.

## Verificação direcionada

```powershell
python -m pytest tests/test_exp1_poder.py tests/test_exp1_allowlist.py -q
```

Os testes fixam o snapshot histórico em 100/91 atributos, comparam os 30 nomes
permitidos com o JSON de referência, verificam a opção pela CLI e pelo eval,
o isolamento treino/teste, a exclusão das MACD mesmo em frame já enriquecido,
a persistência de modelos novos e a leitura do formato antigo.
