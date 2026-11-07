# Sensibilidade post hoc de O3 com MACD

## Escopo

`scripts/sensibilidade_o3_macd.py` compara o O3 original com uma variante que
acrescenta as nove MACD do Experimento 1 saneado. Não modifica a hipótese
fixada de O3 versus O1, a margem de 25%, o arranjo P nem qualquer cache original.
Não faz chamadas a LLM e não usa GPU.

## Procedimento

- Importa `numeric_features`, `_classifier` e `nested_oof_probabilities` de
  `gzcmd.ml_oof`, sem duplicar o algoritmo de treinamento/calibração.
- Lê o extrato com `dtype=str` pelo carregador do comparador existente e
  converte explicitamente os atributos numéricos. As datas e os identificadores
  permanecem strings em memória.
- Calcula as MACD por `gzcmd.loader.feature_engineer` com
  `LoadConfig(macd_enabled=True)`. Valida os nove nomes contra os metadados da
  ablação saneada. Mantém zero para ausências nos 30 atributos originais;
  nas MACD, usa -1, como a imputação constante de `GZCMDClassifier`.
- Reutiliza partições aninhadas 5×5 agrupadas por REFREC, seed 42 para modelos
  e partição externa e 43 a 47 para as partições internas. Mantém os
  hiperparâmetros originais do XGBoost e a calibração isotônica.
- Antes dos ajustes completos, estima o tempo com dois ajustes-piloto,
  extrapolação para 60 ajustes e fator de segurança 3. Interrompe se a
  estimativa superar 60 minutos.
- Ajusta primeiro o controle. Confere diferença máxima absoluta até 1e-6,
  correlação, igualdade dos folds e da fila inteira abaixo do corte.
  Qualquer divergência interrompe a análise antes do ajuste da variante MACD.
- Usa `below_cutoff` e `cumulative_counts` originais. Os empates são resolvidos
  pelo hash lexical do par, apenas em memória, sem expandir o top-k.
- AUC/AP da base inteira usam PAR em {1, 2} como positivo; abaixo do corte,
  os positivos são os PAR=2. Para O1, AUC/AP descrevem SCORE, não uma
  probabilidade ajustada fora da amostra.

## Execução e verificação

No worktree, com o cache original disponível no repositório principal:

```powershell
C:\Users\Marquinho\miniconda3\python.exe scripts/sensibilidade_o3_macd.py --data-root D:/git/phd-research
C:\Users\Marquinho\miniconda3\python.exe -m pytest tests/test_sensibilidade_o3_macd.py -q
```

`--data-root` seleciona somente as entradas privadas e o cache original.
O destino padrão dos agregados continua sendo o `data/` do worktree.
`--estimate-only` executa somente preparação, validações e ajustes-piloto.

O teste de reprodução usa `tests.local_data.requires_files`, permitindo
verificar o contrato dos agregados sem distribuir o extrato privado.

## Saídas e interpretação

- `data/sensibilidade_o3_macd.csv`: variante, métrica e valor.
- `data/sensibilidade_o3_macd_curva.csv`: variante, k e PAR=2 acumulados,
  para toda a fila abaixo do corte.
- `data/sensibilidade_o3_macd_metadata.json`: versões, seeds, parâmetros,
  atributos, hashes, estimativa, tempo e evidência de reprodução.
- Relatório privado em `docs/plans/RELATORIO_O3_MACD_2026-11-07.md`.

Nenhuma saída contém identificador nem valores de campos por par. As curvas
são contagens agregadas. A avaliação assume revisor perfeito e conserva a
limitação de viés de verificação dos PAR=2 definidos pelo serviço.
