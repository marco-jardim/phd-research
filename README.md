# Maximizando Recall ou Precisão: Abordagens de Machine Learning para Pareamento de Registros entre SIM e SINAN-TB

Repositório de notebooks para exploração e comparação de estratégias de linkage/classificação de pares entre o Sistema de Informações sobre Mortalidade (SIM) e o Sistema de Informação de Agravos de Notificação - Tuberculose (SINAN-TB).

## Objetivo

Avaliar e comparar algoritmos de classificação de aprendizado de máquina para linkage probabilístico entre bases de dados de saúde pública (SIM e SINAN-TB), desenvolvendo estratégias que permitam maximizar recall ou precisão conforme a necessidade operacional, sob condições de forte desbalanceamento de classes (~0,4% de pares verdadeiros).

## Status do Projeto

| Fase | Status |
|------|--------|
| Coleta e preparação de dados | ✓ Concluída |
| Implementação dos experimentos | ✓ Concluída (3 notebooks) |
| Execução e geração de resultados | ✓ Concluída (métricas, CSVs, JSONs) |
| Documentação técnica | ✓ Parcial |
| Escrita da tese | ~30% em andamento |

### Resultados preliminares

- **Melhor equilíbrio (F1)**: Random Forest + SMOTE (Precision=0,89, Recall=0,99, F1=0,94)
- **Melhor precisão**: Gradient Boosting (Precision=0,94)
- **Melhor recall**: Stacking Ensemble (Recall=1,0)

### Próximos passos

Os experimentos requerem refinamento e expansão:

- Ampliação do espaço de hiperparâmetros com busca sistemática (GridSearch/RandomizedSearch)
- Validação cruzada mais robusta (k-fold estratificado, repeated k-fold)
- Inclusão de técnicas adicionais (redes neurais, CatBoost, AutoML)
- Análise de sensibilidade dos thresholds
- Avaliação de custo-benefício operacional (trade-off recall vs precisão)
- Testes com diferentes estratégias de feature engineering
- Consolidação metodológica para escrita final da tese

## Notebooks

- `notebooks/01_analise_comparativa_tecnicas.ipynb`
  - Compara modelos classicos (LogReg, RandomForest, GradientBoosting, SVM, MLP) sob desbalanceamento.
  - Usa `class_weight='balanced'`, opcionalmente SMOTE, e varredura de *threshold* para otimizar F1/recall/precisao.

- `notebooks/02_estrategia_maximo_recall.ipynb`
  - Foco em **maximizar recall** (capturar o maximo possivel de pares verdadeiros).
  - Combina *feature engineering* mais permissivo + reamostragem (SMOTE/ADASYN/SMOTETomek) + modelos/ensembles com pesos altos para a classe positiva.

- `notebooks/03_estrategia_maxima_precisao.ipynb`
  - Foco em **maximizar precisao** (pares de alta confianca).
  - Feature engineering mais conservador + selecao de features + modelos regularizados (XGBoost/LightGBM) + calibracao/stacking e regras de negocio.

## Tecnicas (explicacao breve)

- `class_weight` (balanceamento por peso)
  - Ajusta a funcao de perda para penalizar mais erros na classe positiva (rara), reduzindo vies por desbalanceamento.

- Reamostragem (SMOTE e variantes)
  - `SMOTE`: cria amostras sinteticas da classe minoritaria.
  - `BorderlineSMOTE`: foca em amostras perto da fronteira de decisao.
  - `ADASYN`: gera mais exemplos onde a classe minoritaria e mais dificil.
  - `SMOTETomek`: combina oversampling com limpeza de pares (Tomek links).

- Otimizacao de threshold
  - Em vez de usar 0.5, varre limiares para alinhar a decisao com o objetivo (max recall, max precisao, ou melhor F1).

- Ensemble / soft voting
  - Combina probabilidades de varios modelos com pesos; tende a suavizar erros e melhorar estabilidade.

- Stacking
  - Treina um meta-modelo (ex: regressao logistica) sobre as saidas dos modelos base para combinar sinais de forma supervisionada.

- Calibracao de probabilidades (isotonic)
  - Reajusta scores para probabilidades mais confiaveis, ajudando quando thresholds altos sao importantes.

- Regras de negocio (alta confianca)
  - Heuristicas deterministicas (ex: nome/nascimento/mae/municipio) para reforcar decisoes de alta certeza e reduzir falsos positivos.

## Como executar

### Local (recomendado)

1. Crie um ambiente:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Instale dependencias (base):

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Se voce estiver usando `pyproject.toml` apenas como referencia de deps, instale com:

```bash
python -m pip install -U numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyterlab ipykernel papermill nbconvert jupyter xgboost lightgbm
```

3. Rode Jupyter:

```bash
python -m ipykernel install --user --name phd-research
jupyter lab
```

### Execucao headless (CI-friendly)

```bash
mkdir -p notebooks/_executed
papermill notebooks/01_analise_comparativa_tecnicas.ipynb notebooks/_executed/01_analise_comparativa_tecnicas.executed.ipynb
```

## Resultados e saidas

Durante a execucao, alguns notebooks podem gerar CSVs (ex: `comparacao_modelos.csv`, `candidatos_*.csv`).
Evite commitar arquivos derivados grandes; prefira artifacts/outputs fora do git.
