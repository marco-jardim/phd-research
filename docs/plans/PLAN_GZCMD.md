# Plano de Implementacao — GZ-CMD v3 (Grey-Zone Cost-based Mixture Deferral)

**Data**: 2026-02-09
**Status**: Rascunho para revisao
**Autor**: Marco Jardim (com apoio de IA)

---

## 0. Contexto e posicionamento

### 0.1. O que a tese ja demonstrou

A tese mostra que o pos-processamento supervisionado (RF+SMOTE) supera o limiar
ingenuo com folga absoluta (F1 0.947 vs 0.610 no hold-out; +24 obitos recuperados,
+55.8%). O ganho esta concentrado inteiramente na zona cinzenta (escores 5-8), que
contem 47.4% dos pares verdadeiros (117/247) diluidos em 21.518 candidatos.

A tese tambem identifica lacunas operacionais:
- Limiares nao sao portaveis entre contextos (prevalencia muda PPV/NPV)
- Revisao clerical manual e cara e nao-escalavel
- Nao ha mecanismo de auto-calibracao ou monitoramento de drift
- Trabalhos futuros apontam: active learning, pipeline operacional, monitoramento

### 0.2. O que o GZ-CMD propoe

O GZ-CMD e um **framework operacional** que integra componentes conhecidos em uma
arquitetura nova para o dominio de record linkage em saude:

1. Triagem deterministica nos extremos (ancoras)
2. Modelo global unico + calibracao por ancoras (sem labels humanos)
3. Decisao por perda esperada com custos assimetricos FP/FN
4. LLM como revisor clerical automatizado (dual-agent + arbitro)
5. Monitoramento de drift com auto-ajuste

### 0.3. Posicionamento honesto (CRITICO para a defesa)

O GZ-CMD **NAO** e um algoritmo inedito no sentido estrito:
- Triagem 3-vias e o proprio framework Fellegi-Sunter (1969)
- Decision-theoretic linkage existe (Sadinle 2017)
- Guardrails sao regras de negocio (ja na tese)
- Drift monitoring e MLOps padrao

O ineditismo real esta na **integracao operacional**:
- Expected-loss triage com orcamento de LLM (nao existe em linkage)
- LLM como clerical reviewer com protocolo 2-agentes (nao publicado)
- Auto-calibracao por ancoras empiricas derivadas do escore agregado
- Modelagem de erro do LLM por banda alimentando a funcao de custo

**Na tese**: posicionar como "Proposta de Framework Operacional Auto-Calibravel"
ou como capitulo de contribuicao adicional. NAO alegar "algoritmo revolucionario
que supera o estado da arte" — a banca questionara.

---

## 1. Fundamentacao teorica (pre-implementacao)

### 1.1. Revisao bibliografica adicional

| Referencia obrigatoria | Motivo |
|---|---|
| Fellegi & Sunter (1969) | Framework 3-vias original — mostrar que GZ-CMD e extensao moderna |
| Sadinle (2017) — Bayesian decision theory for linkage | Decisao por custo em linkage — estabelecer precedente |
| Enamorado et al. (2019) — fastlink | EM para calibracao semi-supervisionada — contrastar com ancoras |
| Christen (2012) — *Data Matching* | Referencia canonica para quality-aware linkage; taxonomia de erros |
| Larsen & Rubin (2001) — Hierarchical Bayesian record linkage | Framework hierarquico que fundamenta calibracao por ancoras |
| Murray (2016) — Probabilistic record linkage with classification errors | Modelagem explicita de erros de classificacao — base para e_fp/e_fn |
| Steorts et al. (2016) — Entity resolution with empirically motivated priors | Entity resolution vs. pairwise matching — fundamenta constraint N-para-1 |
| Peeters & Bizer (2023) — Entity matching with LLMs | LLMs em tarefas de entity matching — estado da arte e limitacoes |
| Vovk et al. (2005) — Conformal prediction | Citar como alternativa avaliada e descartada (violacao de exchangeability) |
| Sadinle & Fienberg (2013) — 1-para-1 assignment | Matching bipartido em entity resolution |
| Wei et al. (2025) ou similar — LLMs for data integration | LLMs em tarefas de matching/entity resolution |

**Entregavel**: secao de revisao no capitulo do GZ-CMD (~3-4 paginas)

### 1.2. Formalizacao matematica

Formalizar rigorosamente:

1. **Funcao de perda esperada**:
   - L_match(p) = (1-p) * C_fp
   - L_nonmatch(p) = p * C_fn
   - L_review(p) = C_llm + (1-p) * e_fp * C_fp + p * e_fn * C_fn
   - Decisao: argmin{L_match, L_nonmatch, L_review}

2. **EVR (Expected Value of Review)**:
   - EVR = min(L_match, L_nonmatch) - L_review
   - Enviar para LLM se EVR > 0, respeitando orcamento B

3. **Calibracao por ancoras**:
   - p_cal = sigmoid(a * logit(p_base) + b)
   - Restricoes: mean(p_cal | A+) ~= r+, mean(p_cal | A-) ~= r-
   - Resolver para (a, b) por otimizacao (2 eq, 2 incognitas)

4. **Prova de que a triagem 3-vias com custo relaxa as hipoteses de Fellegi-Sunter**:
   - F-S assume independencia condicional dos campos de comparacao e limiares
     derivados de razao de verossimilhanca; GZ-CMD relaxa a independencia via
     modelo discriminativo (RF/Stacking) e substitui limiares por perda esperada
     com orcamento — mostrar equivalencia no caso especial (C_fp=C_fn, e_fp=e_fn=0,
     modelo = naive Bayes ⟹ decisao GZ-CMD coincide com F-S)
   - NAO alegar "generaliza": a relacao e de relaxacao de hipoteses, nao subsuncao

**Entregavel**: secao formal no capitulo (~2-3 paginas)

---

## 2. Implementacao

### 2.1. Estrutura de modulos

```
gzcmd/
  __init__.py
  __main__.py            # CLI entry point
  config.py              # Carrega e valida config.yaml
  bands.py               # Atribuicao de banda por nota_final
  calibration.py         # [NOVO] Anchor-based Platt scaling
  guardrails.py          # [NOVO] Regras deterministicas
  policy_engine.py       # [CORRIGIR] Expected-loss triage
  llm_review.py          # [NOVO] Orquestrador dual-agent + arbitro
  llm_dossier.py         # [NOVO] Montagem do dossie JSON
  consistency.py         # [NOVO] N-para-1 COMPREC->REFREC
  drift_monitor.py       # [NOVO] PSI/KS por banda
  runner.py              # [NOVO] Pipeline completo
  schemas/
    gzcmd_v3_llm_output.schema.json
  prompts/
    gzcmd_v3_llm_prompt.md
  tests/
    test_calibration.py
    test_policy_engine.py
    test_guardrails.py
    test_consistency.py
    test_drift_monitor.py
    test_integration.py
```

### 2.2. Fases de implementacao (paralelizaveis)

O plano esta organizado em **sprints de 2-3 dias**, com tarefas paralelizaveis
dentro de cada sprint. Dependencias entre sprints sao explicitas.

```
SPRINT 1 (Fundacao)          SPRINT 2 (Pipeline)       SPRINT 3 (LLM + Exp)
  [A] config+bands ────────┐                             
  [B] loader+features ─────┼──▶ [E] runner.py ──────────▶ [H] Exp 1-4
  [C] guardrails ──────────┤     (integra A-D)             (usa E+F+G)
  [D] policy_engine fix ───┘                             
                           ┌──▶ [F] calibration ─────────▶ [H]
  (dados CSV explorados)───┘                             
                                [G] consistency ─────────▶ [H]
                                                         
  [I] llm_dossier ─────────────▶ [J] llm_review ────────▶ [K] Exp 2 (LLM)
  [L] prompt engineering ───────▶ [J]                      
                                                         
                                                           [M] drift_monitor
                                                           [N] texto tese
```

#### Sprint 1 — Fundacao [dias 1-3] — 4 tarefas PARALELAS

Nenhuma dependencia entre si. Podem ser implementadas simultaneamente.

| ID | Modulo | Descricao | Complexidade | Deps |
|----|--------|-----------|-------------|------|
| **A** | `config.py` + `bands.py` | Parse/validacao do YAML (dataclasses tipadas). Atribuicao de banda por nota_final. | Baixa | nenhuma |
| **B** | `loader.py` | Carrega CSV, clean_col(), conversao numerica, cria TARGET, feature engineering NB03 + **MACD** (features continuas de DTNASC via R_DTNASC/C_DTNASC: diff_dias, match ano/mes/dia, close<=30/365, interacoes com nome), regras_alta_confianca(). Parse de datas (R_DTOBITO DDMMYYYY, C_* YYYYMMDD). Retorna DataFrame pronto para modelo e permite ablation (MACD on/off). | Media | nenhuma |
| **C** | `guardrails.py` | Regras deterministicas com base nos achados empiricos (secao 2B): **temporal_filter** (dt_obito < dt_diag - 180d → NONMATCH, zero perda de TP comprovada), **homonimia_risk** (dtnasc_all_zero + diff_ano>5 + endereco_zero → forcar REVIEW, NAO filtrar), always_match (nota>=10 + criterios perfeitos), always_nonmatch (nota<3). | Media | nenhuma |
| **D** | `policy_engine.py` | **CORRIGIR bug noop** (linhas ~81-84) + testes. Bug: `p = np.where(p >= th, p, p)` (identidade). Fix: usar caps como guardrails no action final, nao alterar p. | Baixa | nenhuma |

**Bug a corrigir (policy_engine.py)**:
```python
# ANTES (noop):
if self.min_auto_match is not None:
    p = np.where(p >= self.min_auto_match, p, p)  # p = p sempre

# DEPOIS (funcional):
# Nao alterar p; usar caps como override no action final:
# Se p < min_auto_match e base_choice == MATCH → forcar LLM_REVIEW
# Se p > max_auto_nonmatch e base_choice == NONMATCH → forcar LLM_REVIEW
```

**Testes unitarios**: cada modulo DEVE ter tests/ com cobertura >= 80%.

#### Sprint 2 — Pipeline core [dias 3-6] — 3 tarefas PARALELAS (apos Sprint 1)

| ID | Modulo | Descricao | Complexidade | Deps |
|----|--------|-----------|-------------|------|
| **E** | `runner.py` | Pipeline completo: load model → predict p_base → assign band → guardrails → triage → output CSVs. Modos: `vigilancia` (C_fn >> C_fp), `confirmacao` (C_fp >> C_fn). | Media | A, B, C, D |
| **F** | `calibration.py` | Anchor-based Platt scaling: fit(a,b) usando conjuntos ancora empiricos (nota>=9 como A+, nota<5 como A-). p_cal = sigmoid(a * logit(p_base) + b). Resolver sistema 2x2. Validar monotonicidade e coerencia com ancoras. | Media | A (config) |
| **G** | `consistency.py` | **N-para-1 COMPREC→REFREC**: cada COMPREC → max 1 REFREC (keep best p_cal). Se delta entre top-2 < margem → ambos para REVIEW. NAO restringir REFREC (1-to-many valido em TB, 29 REFRECs com multiplos TP). Impacto empirico: -56% pares no grey zone, -4 TP. | Media | nenhuma (interface: recebe DataFrame com COMPREC, REFREC, p_cal) |

**Nota**: F e G podem ser implementados em paralelo com E, desde que
as interfaces (assinaturas de funcao, formatos de entrada/saida) sejam
definidas ANTES no Sprint 1 como contratos em `__init__.py`.

**Entregavel Sprint 2**: CLI executavel:
```
python -m gzcmd run --config config.yaml --data pairs.csv --mode vigilancia --out results/
```

#### Sprint 3A — LLM clerical review [dias 4-8] — PARALELO com Sprint 2

Pode iniciar assim que Sprint 1 terminar (depende apenas de B para o formato
dos dossies). Execucao em paralelo com Sprint 2.

| ID | Modulo | Descricao | Complexidade | Deps |
|----|--------|-----------|-------------|------|
| **I** | `llm_dossier.py` | ✅ DONE. Monta dossie JSON por par: 29 sub-escores, p_cal, banda, reason codes do guardrails, SHAP top-5 (se disponivel), contexto da policy engine. Formato validado contra schema. | Media | B (formato features) |
| **J** | `llm_review.py` | ✅ DONE. Orquestrador dual-agent + arbitro: Agent-A (objetivo) + Agent-B (cetico), consenso ou arbitragem, retry com backoff, fallback automatico. Validacao contra `gzcmd_v3_llm_output.schema.json`. | Alta | I, L |
| **L** | Prompt engineering | ✅ DONE. Prompt v3.1 band-calibrado em `gzcmd_v3_llm_prompt.md`. 3 modelos testados, Kimi K2.5 selecionado. | Media | nenhuma (usa dados do teste 2B.7) |

**Decisao sobre PII**: RESOLVIDA — sem acesso a PII.
O LLM opera apenas com sub-escores numericos. Isso o torna essencialmente
um segundo classificador com overhead de latencia/custo, POREM com
capacidade de raciocinio sobre combinacoes de evidencias que o modelo
nao captura (confirmado empiricamente: 4/5 FP identificados, secao 2B.7).
Discutir honestamente esta limitacao no texto. Propor modo com PII em
enclave seguro como extensao para ambientes que o permitam.

**API LLM**: fireworks.ai (chave em .env FIREWORKS_API_KEY).
Modelo unico: **kimi-k2p5** (Kimi K2.5, $3/M tok, ~4s latencia).
Fallback (indisponibilidade apenas): **qwen3-235b-a22b-instruct-2507** (~22s latencia).
DeepSeek R1 descartado definitivamente (vies estrutural UNSURE — ver secao 2B.8).
Custo estimado: ~$0.003/caso. Atualizar config.yaml para kimi-k2p5 como default.

#### Sprint 3B — Experimentacao [dias 7-10] — apos Sprints 2 e 3A

| ID | Tarefa | Descricao | Deps |
|----|--------|-----------|------|
| **H** | Exp 1-4 (core) | Ablacao (incl. MACD on/off), 5-fold CV, sensibilidade (ver secao 3.3) | E, F, G |
| **K** | Exp 2 (LLM) | LLM review nos pares REVIEW do Exp 1, batch ~74 pares (~$0.22) | J, H |

#### Sprint 4 — Monitoramento + Texto [dias 9-14] — PARALELO

| ID | Tarefa | Descricao | Complexidade | Deps |
|----|--------|-----------|-------------|------|
| **M** | `drift_monitor.py` | PSI e KS por campo, por banda e step. Acoes: tighten_thresholds, freeze_calibration, emit_alert. Logging/auditoria. | Media | E |
| **N** | Texto da tese | Redacao Cap 8 + ajustes nos demais capitulos (secao 4). | Alta | H, K (resultados) |

#### Resumo de paralelismo

```
Dia:  1    2    3    4    5    6    7    8    9    10   11   12   13   14
      ├─ Sprint 1 (A║B║C║D) ─┤
                    ├─ Sprint 2 (E + F║G) ───┤
                    ├─ Sprint 3A (I║L → J) ──────────┤
                                        ├─ Sprint 3B (H → K) ──┤
                                                  ├─ Sprint 4 (M║N) ──────┤
```

**Caminho critico**: Sprint 1 → Sprint 2 (E) → Sprint 3B (H) → Sprint 4 (N)
**Tempo total estimado**: 14 dias uteis (2.8 semanas)
**Economia vs sequencial**: ~40% (sequencial seria ~24 dias)

---

## 2B. Achados empiricos — Investigacao pre-implementacao

Esta secao documenta a investigacao conduzida sobre os dados reais da tese
ANTES de implementar o GZ-CMD, para informar decisoes de design.

### 2B.1. Performance do baseline (reproducao)

Modelo: `modelo_precisao.joblib` (StackingClassifier com XGB+LGB+RF+GB, meta=LogReg).
Hold-out 70/30, random_state=42, stratificado.

| Config (ml_th, regras_th) | TP | FP | FN | F1     |
|---------------------------|----|----|-----|--------|
| 0.94, >=1.5               | 70 | 5  | 4   | 0.9396 |
| 0.85, >=1.0               | 71 | 11 | 3   | 0.9103 |
| Melhor reportado na tese  | 71 | 5  | 3   | 0.947  |

Nota: a reproducao exata (F1=0.947) nao foi atingida porque os top-25
features selecionados por RF importance variam entre execucoes.

### 2B.2. Investigacao 1 — Competicao COMPREC/REFREC

| Achado | Dado |
|--------|------|
| COMPREC (SINAN) unicos | 19.388 de 61.696 pares; 53.4% aparecem >1 vez; max 23 |
| TP que compartilham COMPREC | **0** — cada notificacao SINAN tem no maximo 1 match |
| REFREC (SIM) com multiplos TP | **29** (ate 5 TP/REFREC) — esperado em TB (renotificacao) |
| REFREC unicos | 1.515; 82% aparecem >1 vez; max 630 comparacoes por REFREC |
| FP no grey zone competindo com TP | 2.039 pares (49 via COMPREC, 1.994 via REFREC) |
| COMPREC onde FP nota >= TP | 4 casos perigosos |

**Implicacao**: O constraint correto e **N-para-1** (muitos SINAN → 1 SIM por COMPREC),
NAO 1-para-1 bipartido classico. Cada notificacao SINAN origina no maximo 1 obito,
mas cada obito pode ter multiplas notificacoes.

**Dedup COMPREC (keep best)**: removeria 56.2% dos pares no grey zone (12.210),
perdendo apenas 4 TP onde o FP tinha nota igual ou superior.

### 2B.3. Investigacao 2 — Consistencia temporal

| Achado | Dado |
|--------|------|
| TP com obito antes do diagnostico | 10 de 247 (4%), minimo = -122 dias |
| FP (grey zone) com obito antes do diag | 3.772 de 21.471 (17.6%) |
| FP com obito >1 ano antes do diag | 1.992 |
| Mediana intervalo obito-diag nos TP | 73 dias |

**Margens seguras para filtro temporal** (zero TP perdidos):

| Corte (dias) | FP eliminados | TP perdidos |
|-------------|---------------|-------------|
| < -122      | 3.116         | 0           |
| < -180      | 2.822         | 0           |
| < -365      | 1.992         | 0           |

### 2B.4. Impacto no F1 do baseline

**Resultado central**: no ponto otimo de F1 (th=0.94), os filtros tem impacto ZERO.

| TP | FP baseline | FP c/ filtros | F1 baseline | F1 c/ filtros | delta |
|----|-------------|--------------|-------------|--------------|-------|
| 74 | 178 | 138 | 0.4540 | 0.5175 | +0.064 |
| 73 | 113 | 87 | 0.5615 | 0.6239 | +0.062 |
| 72 | 19 | 17 | 0.8727 | 0.8834 | +0.011 |
| 71 | 6 | 6 | 0.9404 | 0.9404 | 0.000 |
| 70 | 2 | 2 | 0.9589 | 0.9589 | 0.000 |

Os filtros ajudam no regime de alto-recall (thresholds baixos), que e exatamente
onde o GZ-CMD opera (grey zone). No ponto de precisao maxima, sao inuteis.

### 2B.5. Investigacao 3 — Os 5 FP irredutiveis (analise de homonimia)

Os 5 FP no ponto otimo sao **homonimias duras**:

| # | COMPREC | REFREC | nota | proba | NOME | MAE | DTNASC(5) | ENDER | diff_ano | dias |
|---|---------|--------|------|-------|------|-----|-----------|-------|----------|------|
| 1 | 60693 | 2389 | 8.8 | 0.999 | 1.00 | 0.50 | 0/0/0/0/0 | ~0.12 | 0 | 1501 |
| 2 | 50043 | 1864 | 8.6 | 0.975 | 1.00 | 1.00 | 0/0/0/0/0 | 0 | **44** | 531 |
| 3 | 4862 | 1864 | 8.6 | 0.983 | 1.00 | 1.00 | 0/0/0/0/0 | 0 | **41** | 2609 |
| 4 | 19628 | 2477 | 7.7 | 0.958 | 1.00 | 0.67 | 0/0/0/0/0 | 0 | **13** | 2140 |
| 5 | 14369 | 1063 | 6.7 | 0.985 | 1.00 | 0 | 0/0/0/0/0 | ~0.12 | 0 | 2210 |

**Padrao**: nome perfeito + nenhuma concordancia de data + nenhum endereco.
FP #2 e #3 sao notavel: nome E mae identicos, mas nascidos 44 e 41 anos
de diferenca (pessoas completamente diferentes com mesmo nome e mesmo nome de mae).

**Contexto epidemiologico**: homonimia e um problema documentado em bases de saude
brasileiras. Estimativas do DATASUS/SVS indicam 2-5% de registros duplicados por
homonimia em bases nacionais (SIM, SINASC, SIH). No contexto SIM-SINAN, o problema
e agravado por: (a) a populacao-alvo (TB) tem concentracao em areas de alta densidade
demografica, onde nomes comuns sao mais frequentes; (b) o sistema OpenRecLink compara
blocos de registros por soundex/fonetica, gerando inevitavelmente pares homonimos;
(c) campos de data de nascimento podem estar ausentes ou inconsistentes entre os
sistemas (SIM recebe a informacao do atestado de obito; SINAN, da ficha de notificacao).
E fundamental distinguir tres fenomenos: **homonimia verdadeira** (pessoas diferentes
com mesmo nome), **dados faltantes** (mesma pessoa, porem data nao preenchida no
sistema-fonte), e **inconsistencia inter-sistemas** (mesma pessoa, porem datas divergentes
por erro de digitacao ou fonte diferente).

**Tentativa de filtro**: `dtnasc_all_zero` captura os 5 FP, mas tambem
61 TP (24.7%) — dados faltantes no sistema-fonte confundem. O flag nao
e seguro como filtro binario.

**Combinacoes testadas**:

| Filtro | TP perdidos | FP capturados | Viavel? |
|--------|-------------|---------------|---------|
| dtnasc_all_zero | 61 (24.7%) | 5/5 | NAO |
| dtnasc_zero + diff_ano > 2 | 41 (16.6%) | 3/5 | NAO |
| dtnasc_zero + ender_zero + nota >= 6 | 21 (8.5%) | 5/5 | NAO |

**Conclusao**: estes 5 FP sao irredutiveis com features atuais. Opcoes:
1. **Novas features**: flags de missingness (R_DTNASC/C_DTNASC nulos vs presentes-mas-discordantes)
2. **Modelo interno**: deixar o modelo aprender a interacao, nao usar como filtro
3. **LLM review**: estes sao candidatos ideais para revisao — alta probabilidade, baixa evidencia corroborante
4. **Aceitar**: F1=0.947 pode ja ser o teto pratico com estas features

### 2B.6. Implicacoes para o design do GZ-CMD

1. **Filtro temporal (-180d)**: implementar como guardrail. Ganho no grey zone, zero risco.
   Adicionar a `guardrails.py`.

2. **Dedup COMPREC (keep best)**: implementar em `consistency.py` como constraint
   N-para-1 (nao bipartido classico). Impacto no grey zone: -56% pares.
   Constraint: cada COMPREC → no maximo 1 REFREC. Quando multiplos candidatos
   existem para mesmo COMPREC, manter o de maior probabilidade calibrada.
   Se delta < margem, enviar ambos para LLM review.

3. **Homonimia**: NAO usar como filtro. Incluir `dtnasc_all_zero` e `diff_ano`
   como features de entrada no modelo. O modelo pode aprender a interacao.

4. **LLM review**: os 5 FP irredutiveis sao o caso de uso ideal — alta
   probabilidade ML mas evidencia corroborante fraca. O GZ-CMD deve
   direcionar estes para revisao via EVR (Expected Value of Review).

5. **Meta revisada**: GZ-CMD core provavelmente NAO superara F1=0.947.
   O ganho esta em: (a) reducao de FP no grey zone via filtros,
   (b) flexibilidade operacional via custos assimetricos,
   (c) LLM como ultima linha para homonimias.

### 2B.6b. Achados MACD — implementacao e ablacao (fev/2026)

**Contexto**: MACD (Medidas Continuas de Diferenca de Datas) foi implementado em
`gzcmd/loader.py` como features continuas derivadas de R_DTNASC/C_DTNASC:
`macd_nasc_diff_capped`, `macd_nasc_year/month/day_match`, `macd_nasc_close/very_close`,
e interacoes com `nome_perf` (nome perfeito + DOB longe). Toggle via `LoadConfig.macd_enabled`.

#### Fase 1 — Ablacao com Platt scaling (resultado nulo)

Ablacao v3 (40 runs: 2 modos x 2 splits x 2 MACD x 5 seeds) com calibracao Platt
produziu resultados **identicos** em todos os 40 runs.

**Causa raiz**: Platt scaling opera exclusivamente sobre `nota_final` (escore agregado
do OpenRecLink). As features MACD sao computadas e armazenadas no DataFrame, mas
nenhum componente downstream as consome. Para que MACD tenha efeito mensuravel, e
necessario um classificador ML que consuma as features engineered.

#### Fase 2 — Classificador ML (RF + Platt) — CONCLUIDO

Implementado `gzcmd/classifier.py`: wrapper sklearn com RandomForestClassifier +
CalibratedClassifierCV (method="sigmoid", Platt on top). Feature set = toda saida
numerica do loader (29 sub-escores + agregados + MACD quando habilitado).
Modo `--calibration ml_rf` no CLI/eval, com fit no train e predict_proba no test.

#### Fase 3 — Ablacao definitiva com ML (Comprec split, 5 seeds) — CONCLUIDO

**Resultado (media ± desvio-padrao sobre 5 seeds, split por COMPREC)**:

| Modo         | MACD | Precision     | Recall        | F1            | F-beta        | Auto-cov | LLM calls |
|-------------|------|---------------|---------------|---------------|---------------|----------|-----------|
| confirmacao | OFF  | 0.949 ± 0.006 | 0.937 ± 0.019 | 0.943 ± 0.009 | 0.946 ± 0.005 | 99.2%    | 138       |
| confirmacao | ON   | 0.957 ± 0.004 | 0.938 ± 0.023 | 0.947 ± 0.011 | 0.953 ± 0.004 | 99.4%    | 114       |
| vigilancia  | OFF  | 0.934 ± 0.009 | 0.964 ± 0.013 | 0.949 ± 0.004 | 0.958 ± 0.009 | 99.3%    | 128       |
| vigilancia  | ON   | 0.947 ± 0.006 | 0.962 ± 0.013 | 0.954 ± 0.005 | 0.959 ± 0.010 | 99.4%    | 109       |

**Delta (MACD ON - OFF)**:

| Modo         | ΔPrecision | ΔRecall | ΔF1    | ΔF-beta | ΔLLM calls |
|-------------|-----------|---------|--------|---------|------------|
| confirmacao | **+0.8pp** | +0.1pp  | +0.4pp | **+0.7pp** | **-24**    |
| vigilancia  | **+1.3pp** | -0.2pp  | **+0.5pp** | +0.1pp  | **-19**    |

**Interpretacao**:
- MACD melhora precision ~1pp em ambos os modos, com recall praticamente inalterado
- F1 sobe ~0.5pp consistentemente
- LLM calls caem ~20 (menos revisao humana necessaria)
- Resultado robusto: 5 seeds com Comprec split (previne leakage entre pares do mesmo COMPREC)

**Nota sobre ablacao parcial anterior (Row split, 2 seeds)**: resultado preliminar sugeria
que MACD degradava vigilancia F1/F2. Contradição resolvida: row split permite leakage
entre pares do mesmo COMPREC, inflando baseline artificialmente. Comprec split e o
protocolo correto e confirma beneficio do MACD.

**Nota para a tese**: MACD nao e contribuicao standalone — o ganho so se materializa
quando ha um classificador ML que consome as features. A representacao binaria do
OpenRecLink descarta informacao recuperavel (diferencas continuas de datas), e essa
informacao flui para a decisao apenas via classificador, nao via calibracao direta
do escore agregado. O achado reforça a narrativa de que a codificacao lossy do
comparador e uma limitacao pratica superavel.

### 2B.7. Teste empirico — LLM clerical review (pre-implementacao)

Teste realizado com 14 casos (5 FP + 5 TP + 4 FN) usando fireworks/kimi-k2p5.
Prompt em portugues, especialista SIM-SINAN TB. Sem PII — apenas sub-escores numericos.

> **Nota metodologica**: n=14 e prova de conceito, NAO evidencia estatistica.
> Para 4/5 FP corretos: IC95% binomial exato = [36%, 98%] (Clopper-Pearson).
> Para 4/5 TP corretos: IC95% = [36%, 98%].
> O n necessario para IC width < 10pp com 80% de poder seria ~90 casos por classe.
> Resultados devem ser interpretados como **sinal direcional**, nao como estimativa
> pontual de desempenho.

**Resultados por classe**:

| Classe | Corretos | UNSURE | Errados | Acuracia pontual | IC95% Clopper-Pearson |
|--------|----------|--------|---------|------------------|----------------------|
| FP (5) | 4 NONMATCH | 1 | 0 | 80% | [36%, 98%] |
| TP (5) | 4 MATCH | 1 | 0 | 80% | [36%, 98%] |
| FN (4) | 0 | 1 | 3 NONMATCH | 0% | [0%, 60%] |

**Impacto no F1**:

| Cenario | TP | FP | FN | F1 | Delta |
|---------|----|----|-----|-------|-------|
| Baseline (sem LLM) | 70 | 5 | 4 | 0.9396 | — |
| Com LLM (UNSURE→humano) | 70 | 1 | 4 | **0.9655** | **+0.026** |

**Achados-chave**:
1. LLM excelente para FP: 4/5 homonimias identificadas (HOMONYMY_RISK, TEMPORAL_ANOMALY)
2. LLM seguro para TP: 0/5 pares verdadeiros rejeitados
3. LLM **NAO recupera FN**: 0/4 — mesmo ponto cego do modelo (DOB=0 ambiguo)
4. 3 decisoes UNSURE (conservador) — carga de revisao humana gerenciavel
5. FP_2 (diff_ano=44, nome+mae=100%) foi UNSURE, nao NONMATCH — caso mais dificil

**Limitacao fundamental**: sem PII (datas reais), o LLM nao distingue "data ausente
no sistema-fonte" de "data presente mas discordante". Isso impede recuperacao de FN
e limita ganho a precisao.

**Reposicionamento conceitual do LLM**: os resultados mostram que o LLM NAO funciona
como classificador superior — opera sobre os mesmos sub-escores numericos do modelo ML
e tem os mesmos pontos cegos. O valor real do LLM e como **gerador de justificativas
estruturadas para decisoes marginais**:
- Produz razoes auditaveis em linguagem natural (e.g., "HOMONYMY_RISK: nome identico
  mas nascimento incompativel sugere homonimia")
- Decisoes UNSURE identificam casos que requerem arbitragem humana
- Metricas de avaliacao adequadas: **taxa de UNSURE** (quanto filtra para humano),
  **concordancia humano-LLM** (kappa), **cobertura de razoes** (% dos 18 codigos usados)
  — NAO acuracia bruta, que e ilusoria com n=14
- Para a tese: posicionar como "assistente de revisao clerical", nao como "revisor autonomo"

**Implicacao para design**: LLM review e viavel como modulo de precisao (+2.6pp F1)
E como gerador de justificativas auditaveis para vigilancia epidemiologica.
Para ganho em recall, seria necessario acesso a PII em enclave seguro (proposta para
trabalhos futuros). O custo por caso (~$0.003 no kimi-k2p5) e negligivel.

### 2B.8 PoC comparativo 3 modelos LLM (Sprint 3A, task L)

**Objetivo**: selecionar modelo LLM para revisao clerical via Fireworks API.

**Modelos testados**:

| Modelo | Tier | Output $/M tok | Contexto | Fireworks ID |
|--------|------|-----------------|----------|--------------|
| Qwen3 235B A22B | Mid-range | $0.88 | 262K | `qwen3-235b-a22b-instruct-2507` |
| Kimi K2.5 | Premium | $3.00 | 262K | `kimi-k2p5` |
| DeepSeek R1 0528 | Frontier | $5.40 | 164K | `deepseek-r1-0528` |

**Setup**: 4 pares grey-zone (nota 4.38 / 5.35 / 6.45 / 7.33), dossier com 30 subscores
de linkage (sem PII), prompt v3.0, temperature=0.0, `response_format=json_object`.

**Resultados**:

| Par | Nota | Banda | Qwen3 235B | Kimi K2.5 | DeepSeek R1 |
|-----|------|-------|------------|-----------|-------------|
| 1 | 4.38 | grey_low | NONMATCH(0.35) | NONMATCH(0.75) | NONMATCH(0.70) |
| 2 | 5.35 | grey_mid | UNSURE(0.50) | UNSURE(0.50) | UNSURE(0.50) |
| 3 | 6.45 | grey_mid | UNSURE(0.50) | UNSURE(0.55) | UNSURE(0.50) |
| 4 | 7.33 | near_high | MATCH(0.85) | MATCH(0.75) | UNSURE(0.50) |

**Concordancia**: Qwen3 vs Kimi = 100%. Ambos vs DeepSeek = 75%.

**Latencia media**: Kimi ~3s, Qwen3 ~15s, DeepSeek ~8s.

**Analise**:
1. DeepSeek R1 conservador demais — UNSURE no par near_high (7.33) onde Qwen3/Kimi
   acertaram MATCH. Estilo chain-of-thought gera excesso de duvidas.
2. Qwen3 e Kimi idênticos em decisão, mas Kimi 5x mais rapido.
3. Kimi K2.5 melhor custo-beneficio: decisoes alinhadas, latencia baixa,
   custo intermediario ($3/M vs $0.88 e $5.40).

**Decisao**: Kimi K2.5 (`kimi-k2p5`) selecionado como modelo primario.

#### Rodada 2 — prompt v3.1

Prompt reescrito com: politica calibrada por banda (near_high/grey_mid/grey_low),
limiares objetivos por feature, instrucao anti-vies ("dados faltantes nao sao contradicao"),
e regra de modelo calibrado (p_cal). Mesmos 4 pares, mesmas condicoes.

| Par | Nota | Banda | Qwen3 235B | Kimi K2.5 | DeepSeek R1 |
|-----|------|-------|------------|-----------|-------------|
| 1 | 4.38 | grey_low | NONMATCH(0.65) | NONMATCH(0.70) | UNSURE(0.50) |
| 2 | 5.35 | grey_mid | UNSURE(0.55) | UNSURE(0.55) | UNSURE(0.50) |
| 3 | 6.45 | grey_mid | UNSURE(0.55) | UNSURE(0.55) | UNSURE(0.50) |
| 4 | 7.33 | near_high | MATCH(0.75) | MATCH(0.75) | UNSURE(0.50) |

**Concordancia v3.1**: Qwen3 vs Kimi = 100%. Ambos vs DeepSeek = 50% (piorou).

**Latencia v3.1**: Kimi ~4s, Qwen3 ~22s, DeepSeek ~6s.

**Analise v3.1**:
1. Qwen3 e Kimi mantiveram alinhamento perfeito. Decisoes identicas ao v3.0
   exceto confiancas mais calibradas (Qwen3 grey_low 0.35→0.65, alinhado com Kimi).
2. DeepSeek R1 **piorou**: UNSURE em 4/4 pares (vs 3/4 no v3.0). O modelo reasoning
   argumenta dos dois lados e converge sistematicamente para UNSURE(0.50).
   Esse vies e estrutural do estilo chain-of-thought, nao corrigivel via prompt.
3. DeepSeek R1 **descartado** para revisao clerical. Adequado para tarefas
   analiticas (explicacao, auditoria de codigo) mas nao para decisao binaria.

**Decisao final**: usar **apenas** Kimi K2.5 (`kimi-k2p5`) como modelo de producao.
Qwen3 235B (`qwen3-235b-a22b-instruct-2507`) como fallback exclusivo — acionado
apenas se Kimi estiver indisponivel ou retornar erro apos retry.
DeepSeek R1 **descartado definitivamente** — nao sera usado em nenhuma etapa do pipeline.
Prompt v3.1 como baseline para Sprint 3A.

**Justificativa consolidada**:
- Kimi e Qwen3 concordam em 100% dos casos (8/8 decisoes em 2 rodadas)
- Kimi e 5x mais rapido (~4s vs ~22s) e suficiente como modelo unico
- DeepSeek R1 tem vies estrutural: modelos reasoning/CoT argumentam dos dois lados
  e convergem para UNSURE(0.50) sistematicamente, tornando-os inadequados para
  decisoes binarias em revisao clerical
- Custo operacional com Kimi: ~$0.003/caso ($3/M tok output)

**Script**: `scripts/poc_llm_3model_test.py` — reproduzivel com `--n-pairs N`.

### 2B.9. Pipeline E2E — Sprint 3A completa (llm_dossier + llm_review)

Implementados os dois modulos finais do Sprint 3A e testados end-to-end:

**`gzcmd/llm_dossier.py`**: Monta dossie JSON por par a partir do DataFrame apos triagem.
Conteudo: 29 sub-escores agrupados por dominio (nome:4, mae:4, nascimento:4,
endereco:6, municipio:1) + scores compostos (5) + flags (3) + MACD (9, se habilitado)
+ saidas do modelo (p_cal, band, base_choice, evr). Sem PII: apenas scores numericos.
Identificador: SHA-256[:16] de COMPREC+REFREC+PASSO.

**`gzcmd/llm_review.py`**: Orquestrador dual-agent + arbitro.
- Agent-A (objetivo) e Agent-B (cetico) recebem o mesmo dossie com system prompts diferenciados
- Se concordam → consenso direto; se discordam → Arbitro decide
- API: Fireworks REST. Modelo primario: Kimi K2.5 (`kimi-k2p5`), fallback: Qwen3 235B
- Retry com backoff exponencial (3 tentativas), fallback automatico por indisponibilidade
- Validacao de schema: decision ∈ {MATCH, NONMATCH, UNSURE}, confidence [0,1], reason_codes validos
- Extracao JSON robusta: trata markdown fences e blocos `<think>` de modelos reasoning

**Teste E2E** (3 pares amostrais, modo vigilancia, prompt v3.1):

| Pair ID  | Band      | Nota | p_cal | Decisao  | Conf | Protocolo | Tempo |
| -------- | --------- | ---- | ----- | -------- | ---- | --------- | ----- |
| e02833fc | high      | 9.77 | 0.908 | MATCH    | 0.92 | consensus | 5.8s  |
| 3312f130 | near_high | 8.93 | 0.618 | MATCH    | 0.85 | consensus | 7.4s  |
| b0f99dae | grey_high | 7.98 | 0.173 | NONMATCH | 0.75 | consensus | 5.0s  |

**Observacoes**:
1. 100% consenso Agent-A/B — arbitro nao foi acionado em nenhum caso.
2. Decisoes coerentes com os scores: high + p_cal alto → MATCH; grey_high + p_cal baixo → NONMATCH.
3. Latencia media 6.1s/par (aceitavel para batch de ~70-80 pares em producao).
4. Pipeline completo funcional: load → features → calibrate → band → guardrails → triage → dossier → LLM review.

**Script E2E**: `scripts/test_sprint3a_e2e.py` — reproduzivel com `--n-pairs N --mode MODE`.

**Status Sprint 3A**: COMPLETO (tarefas I, J, L todas concluidas).

---

## 3. Protocolo experimental (avaliacao e validacao)

### 3.1. Objetivo epidemiologico e metricas

- **Modo vigilancia** (prioriza recall): metrica primaria **F2**; reportar tambem Precision/Recall/F1,
  e custo operacional (revisoes por 1.000 pares / tempo humano estimado).
- **Modo confirmacao** (prioriza precision): metrica primaria **F0.5**; reportar tambem Precision/Recall/F1.
- **Incerteza**: reportar IC (Wilson para proporcoes; bootstrap estratificado para F_beta quando aplicavel).

### 3.2. Split e prevencao de vazamento (leakage)

- **Split base (comparabilidade com a tese)**: hold-out 70/30 estratificado (`random_state=42`).
- **Tuning sem leakage**: hiperparametros, features e limiares sao escolhidos somente em
  treinamento/validacao (nested CV ou validacao fixa). O conjunto de teste e usado **uma unica vez**.
- **Sensibilidade a drift**: split temporal (treina em anos anteriores, testa em anos posteriores)

### 3.3. Experimentos (Exp 1-4)

- **Exp 1 (Ablacao do GZ-CMD)**: medir impacto incremental de (i) **MACD** (features continuas de DTNASC)
  vs sub-escores binarios, (ii) calibracao por ancoras, (iii) guardrails temporais,
  (iv) consistencia N-para-1, (v) orcamento B e EVR, (vi) LLM review.
- **Exp 2 (LLM clerical review)**: avaliar por banda e modo; metrica primaria = reducao de FP dado
  recall fixo; relatar taxa de UNSURE, concordancia humano-LLM (kappa) quando houver, e custo.
- **Exp 3 (Robustez)**: 5-fold CV estratificado (com cuidado para nao vazar tuning); relatar variancia.
- **Exp 4 (Sensibilidade)**: varrer `C_fp/C_fn` e orcamento B; gerar fronteira de Pareto
  (revisoes vs recall; FP vs FN) para vigilancia vs confirmacao.

### 3.4. Limitacao single-site (e mitigacoes)

Validacao e single-site (uma UF/periodo; 247 positivos). Isso limita validade externa.
Mitigacoes no escopo do estudo: (i) split temporal; (ii) stress tests de prevalencia por reweighting;
(iii) analise de sensibilidade por custos e por bandas.

### 3.5. Criterios de sucesso (praticos)

- Vigilancia: manter/elevar recall vs baseline com custo de revisao controlado.
- Confirmacao: reduzir FP sem derrubar recall abaixo do baseline.
- Transparencia: thresholds e custos fixados antes de avaliar no teste; resultados com IC.

---

## 4. Revisao do texto da tese

### 4.1. Enquadramento narrativo (banca de epidemiologia)

O Capitulo 8 deve ser defendido como contribuicao **operacional** para vigilancia
(governanca de incerteza, custo assimetrico, triagem 3-vias), e nao como "modelo melhor".

### 4.2. Estrutura do novo Capitulo 8 (GZ-CMD)

```
8.1. Motivacao e problema operacional (vigilancia vs confirmacao)
     - Por que FN e mais grave que FP em vigilancia
     - Por que F_beta (F2/F0.5) e mais coerente que F1 nesse contexto

8.2. Dados e representacao
     - Dados secundarios; sem acesso a identificadores nominais (nome/endereco/documentos)
     - Definicoes OpenRecLink: C_ = candidato (SINAN), R_ = referencia (SIM)
     - 29 sub-escores + nota final; limites do vetor binario

8.3. Metodos: pipeline GZ-CMD
     - Bandas por nota final + ancoras
     - Calibracao por ancoras
     - Guardrails temporais e consistencia N-para-1
     - Policy engine: perda esperada + EVR + orcamento

8.4. Revisao clerical assistida (sem PII)
     - Protocolo dual-agent + arbitro: Agent-A (objetivo) + Agent-B (cetico)
     - Se consenso → decisao direta; se discordancia → Arbitro com contexto de ambos
     - LLM como gerador de justificativas (reason_codes + evidence_summary)
     - UNSURE → encaminhamento humano (conservador por design)
     - Modelo unico: Kimi K2.5 (100% concordancia com Qwen3 em 8/8 decisoes)
     - DeepSeek R1 descartado: vies estrutural CoT→UNSURE em decisoes binarias
     - Logging e auditabilidade: dossie JSON + resposta LLM arquivados

8.5. Resultados e avaliacao
     - Exp 1-4, comparacao com baselines, custos
     - Curvas revisoes-vs-recall e Pareto

8.6. Nota sobre perda de informacao em comparacoes binarias (MACD)
     - OpenRecLink: sub-escores binarios de DTNASC sao projecao lossy
     - Usar datas brutas (R_DTNASC/C_DTNASC) para features continuas melhora recall
     - NAO constitui contribuicao standalone; relatar como achado empirico

8.7. Discussao
     - Ganho operacional vs ganho em F1
     - Limitacoes: LLM sem PII, 247 positivos, single-site
     - Implicacoes para a vigilancia em saude
     - Trabalhos futuros: active learning, validacao externa, novos blocking

     8.7.x. Consideracao sobre embeddings para similaridade semantica
     - Avaliar se modelos de embedding (sentence-transformers, etc.) poderiam
       substituir ou complementar metricas de string no pipeline de linkage
     - Conclusao: descartado. Justificativas:
       (a) Os campos comparados sao curtos e estruturados (nome, DOB, mae,
           municipio) — metricas classicas (Jaro-Winkler, token overlap) ja
           capturam a variacao relevante (typos, abreviacoes, apelidos)
       (b) Embeddings generalistas diluem a granularidade que distancias de
           string oferecem em campos curtos; cosine similarity nao supera
           Jaro-Winkler para nomes de 2-3 tokens
       (c) Os erros da zona cinzenta derivam de dados faltantes ou conflitantes
           (mae ausente, DOB parcial, homonimia), nao de falta de compreensao
           semantica — embedding nenhum recupera informacao que nao existe
       (d) Custo computacional (vetorizacao de milhoes de registros + indice
           ANN) desproporcional ao ganho esperado (<1pp F1)
       (e) Perda de interpretabilidade: sub-escores de string sao rastreáveis
           ("JW(nome)=0.82"), embeddings sao caixas-pretas
     - Cenarios onde faria sentido (nao aplicaveis aqui): campos de texto
       livre extensos, fontes multilingues, entidades complexas (empresas)
```

#### 4.2.1. Texto sugerido (tese statement do Capitulo 8)

Neste capitulo, proponho um protocolo operacional e auditavel para linkage probabilistico
SIM-SINAN em vigilancia de obitos por TB, onde o erro nao e simetrico: perder um vinculo
verdadeiro (FN) tem custo epidemiologico maior do que revisar um falso positivo (FP).
A contribuicao central nao e "um classificador novo", e sim uma politica de decisao em 3 vias
(vincular / nao vincular / revisar) que explicita custos, orcamento de revisao e metas de
confiabilidade, com dois modos de uso: vigilancia (F2 como metrica primaria) e confirmacao
(F0.5).
Mostro empiricamente que parte do teto observado vem de uma representacao lossy de
comparacoes binarias; ao extrair medidas continuas de diferenca de datas ja presentes no
arquivo (MACD), o sistema recupera casos perdidos e melhora o desempenho de forma
consistente com a finalidade de vigilancia.
Para um subconjunto pequeno de casos ambiguos (homonimia), avalio revisao assistida por IA
como apoio a triagem e geracao de justificativas, mantendo uma saida conservadora
"UNSURE" para encaminhamento humano.
Por desenho metodologico, este estudo utiliza dados secundarios e nao acessa identificadores
nominais (nome, endereco, documentos), nem ha contato com sujeitos.
Finalmente, discuto limitacoes de generalizacao (base unica, poucos positivos) e como o
protocolo pode ser revalidado ao mudar periodo/estado sem depender de um unico limiar fixo.

#### 4.2.2. Perguntas/objecoes provaveis da banca (com respostas curtas)

1. "Por que trocar F1 por F2? Nao e forcar o resultado?"
   - Em vigilancia, FN e mais grave que FP (subestima obitos/casos). F2 torna esse custo explicito.
     Eu ainda reporto P/R/F1 e apresento dois modos (vigilancia vs confirmacao) para transparenciar
     o trade-off.

2. "Isso e so ajustar limiar/heuristica. Onde esta a contribuicao?"
   - A contribuicao e transformar limiar ad hoc em politica reprodutivel de 3 vias com orcamento,
     criterios de encaminhamento e auditoria; e demonstrar que a codificacao binaria das
     comparacoes perde informacao recuperavel (MACD), o que impacta recall.

3. "Como voce sabe que isso generaliza para outros estados/periodos?"
   - Nao sei; eu declaro como limitacao (single-site, poucos positivos). A proposta e um protocolo
     de revalidacao: splits temporais, simulacao de shift, re-calibracao por ancoras e re-estimativa
     de limiares por custo/objetivo.

4. "IA na revisao clerical: por que confiar? E se ela errar?"
   - Eu nao uso IA como verdade; uso como apoio para triagem e justificativa, com saida "UNSURE"
     para casos de risco e logging para auditoria. O piloto indica potencial para reduzir FP em
     homonimia, mas com amostra pequena (IC amplo).

5. "Qual e o ganho pratico para vigilancia? O que muda no dia a dia?"
   - O ganho e operacional: reduzir carga de revisao e tornar decisoes mais consistentes. Eu reporto
     impacto em (i) FP evitados para o mesmo recall, (ii) FN recuperados com MACD, e (iii) volume
     em "revisar" vs "auto", permitindo estimar custo humano e escolher o modo adequado.

#### 4.2.3. Nota metodologica (selecao de limiares e metricas sem vazamento)

Para evitar vazamento de informacao, toda escolha de limiares (p.ex., `t_ml`, `t_rules`, bandas,
orcamento de revisao e pesos de custo) e feita somente com dados de treinamento/validacao
(p.ex., nested CV ou validacao fixa), e o conjunto de teste fica reservado para avaliacao final
uma unica vez. No texto, deixe explicito que (i) beta e custos sao fixados antes de olhar o teste,
(ii) o teste nao e usado para "ajustar limiar", e (iii) os resultados sao reportados com IC
(bootstrap estratificado) e analise de sensibilidade aos custos.

### 4.3. Ajustes nos capitulos existentes

| Capitulo | Ajuste |
|----------|--------|
| Cap 1 (Introducao) | Adicionar paragrafo: "Alem da avaliacao comparativa, esta tese propoe um framework operacional..." |
| Cap 2 (Revisao) | Adicionar subsecao sobre decision-theoretic linkage (Sadinle 2017) e LLMs para data integration |
| Cap 4 (Objetivos) | Adicionar objetivo especifico: "Propor e avaliar um framework operacional auto-calibravel..." |
| Cap 6 (Resultados) | Sem mudanca — manter intacto como baseline |
| Cap 7 (Discussao) | Referenciar o cap 8 onde apropriado; na subsecao de limitacoes, citar que o GZ-CMD endereça parcialmente X e Y |
| Cap 9 (Conclusoes) | Expandir contribuicoes: adicionar o framework GZ-CMD como contribuicao metodologica-operacional. Atualizar trabalhos futuros: separar o que o GZ-CMD ja resolve vs o que permanece em aberto |

### 4.4. Tabelas e figuras novas necessarias

| ID | Tipo | Conteudo |
|----|------|----------|
| T1 | Tabela | Comparacao unificada: Naive vs Rules vs RF+SMOTE vs GZ-CMD (F1, P, R, revisoes/par) |
| T2 | Tabela | Ablacao do GZ-CMD (E1a-E1e) |
| T3 | Tabela | Resultados LLM por modo (sem PII, com PII, por orcamento) |
| T4 | Tabela | Sensibilidade a custos (C_fp/C_fn) |
| F1 | Figura | Diagrama de fluxo do pipeline GZ-CMD |
| F2 | Figura | Curva revisoes-vs-recall: GZ-CMD vs baselines |
| F3 | Figura | Fronteira de Pareto: GZ-CMD sobreposto a tese |
| F4 | Figura | Heatmap de decisoes por banda e modo operacional |

---

## 5. Cronograma sugerido

**Nota**: o cronograma original de 6 semanas era otimista. Considerando a complexidade
da calibracao empirica, iteracao de prompts LLM, e redacao academica, o prazo realista
e de **10-12 semanas** (ver riscos R-07).

| Semana | Fase | Entregavel |
|--------|------|-----------|
| 1-2 | Fundamentacao + Sprint 1 | Revisao bibliografica; modulos core paralelos (config, loader+MACD, guardrails, policy fix) |
| 3-4 | Sprint 2 + Sprint 3A inicio | Runner, calibration, consistency; inicio dossier + prompt LLM |
| 5-6 | Sprint 3A + Sprint 3B | Iteracao de prompts LLM; Exp 1 (ablacao, incl. MACD on/off) + Exp 3 (5-fold CV) |
| 7-8 | Experimentos LLM + Analise | Exp 2 (LLM, 74 pares); Exp 4 (sensibilidade); compilacao resultados |
| 9-10 | Texto | Redacao cap 8 (GZ-CMD) + ajustes caps 1-7,9; tabelas/figuras |
| 11-12 | Revisao + Buffer | Revisao cruzada; correcoes; margem para imprevistos |

**Justificativa da extensao**: (a) iteracao de prompts LLM requer multiplos ciclos de
teste-avaliacao-ajuste, nao paralelizavel com experimentacao; (b) redacao academica em
portugues de capitulo novo (8) + revisao de 6 capitulos existentes demanda tempo
proporcional; (c) buffer de 2 semanas para riscos nao antecipados (ver R-07).

---

## 6. Riscos e mitigacoes

| Risco | Probabilidade | Impacto | Mitigacao |
|-------|-------------|---------|-----------|
| GZ-CMD core nao supera F1 da tese | **Confirmado** | Medio | Posicionar ganho como operacional (custo), nao como F1. Investigacao empirica (secao 2B.4) demonstra que filtros temporais e dedup COMPREC nao impactam F1 no ponto otimo — os 5 FP irredutiveis sao homonimias que requerem features novas ou LLM |
| LLM sem PII nao melhora decisao | Alta | Alto | Implementar modo com PII (enclave); discutir honestamente no texto |
| Taxas de erro LLM muito diferentes do estimado | Media | Alto | Calibrar empiricamente nos 74 pares; iterar prompt |
| Banca questiona ineditismo | Media | Alto | Posicionar como integracao operacional; citar Fellegi-Sunter e Sadinle como fundacao |
| 247 positivos insuficientes para validar por banda | Alta | Medio | Manter modelo global; so usar bandas para calibracao e politica, nao para treino |
| Ancoras degradam com mudanca de comparador | Baixa (single-run) | Baixo | Monitoramento de drift (Fase 3) cobre operacao futura |
| R-07: Reproducibilidade LLM | Media | Alto | Modelos comerciais mudam sem aviso (fine-tuning, routing, deprecation). **Mitigacao**: (a) logar TODAS as respostas com hash do prompt, model_id, timestamp, e versao da API; (b) fixar temperatura=0.1 e seed quando suportado; (c) reportar intervalo de variacao entre runs repetidos (3-5x); (d) manter fallback para modelo local (e.g. Llama 3.1 quantizado) |
| R-08: Prazo excede 12 semanas | Media | Medio | Buffer de 2 semanas ja incluso no cronograma. Priorizar entregaveis minimos: pipeline core + 1 experimento + cap 8 draft |

---

## 7. Decisoes — resolvidas

1. **Estrutura**: Capitulo separado (Cap 8) — CONFIRMADO
2. **PII**: NAO tera acesso. LLM opera apenas com sub-escores numericos (prova de conceito)
3. **API LLM**: fireworks.ai — modelo unico **kimi-k2p5** (Kimi K2.5).
   Fallback (apenas em caso de indisponibilidade): **qwen3-235b-a22b-instruct-2507** (Qwen3 235B).
   DeepSeek R1 descartado definitivamente por vies estrutural UNSURE em decisoes binarias (secao 2B.8).
   API key em `.env FIREWORKS_API_KEY`.
4. **Objetivos**: SIM — adicionar como objetivo especifico formal no capitulo 4
5. **Colunas do CSV**: mapeadas a partir de `data/COMPARADORSEMIDENT.csv` (ver Apendice A abaixo)

---

## Apendice A — Mapeamento de colunas do CSV de pares

**Arquivo**: `data/COMPARADORSEMIDENT.csv`
**Separador**: `;` | **Decimal**: `,`

### A.1. Campos de identificacao do par

| Coluna | Descricao |
|--------|-----------|
| PAR | Identificador unico do par candidato |
| PASSO | Etapa do blocking em que o par foi gerado |
| COMPREC | ID do registro no SINAN (sistema de notificacao) |
| REFREC | ID do registro no SIM (sistema de mortalidade) |
| SCORE | Escore agregado original do comparador |

### A.2. Campos da base Referencia (prefixo R_ — convencao OpenRecLink)

No linkage desta tese: Referencia = SIM (Sistema de Informacoes sobre Mortalidade).

| Coluna | Descricao |
|--------|-----------|
| R_DTNASC | Data de nascimento (Referencia/SIM) |
| R_SEXO | Sexo (Referencia/SIM) |
| R_CODMUNRES | Codigo municipio de residencia (Referencia/SIM) |
| R_BAIRES | Bairro de residencia (Referencia/SIM) |
| R_DTOBITO | Data do obito (Referencia/SIM) |
| R_IDLINHA | ID de linha (Referencia/SIM) |
| R_DTNASC2 | Data de nascimento alternativa (Referencia/SIM) |

### A.3. Campos da base Candidato (prefixo C_ — convencao OpenRecLink)

No linkage desta tese: Candidato = SINAN (Sistema de Informacao de Agravos de Notificacao).

| Coluna | Descricao |
|--------|-----------|
| C_DTNASC | Data de nascimento (Candidato/SINAN) |
| C_SEXO | Sexo (Candidato/SINAN) |
| C_CODMUNRES | Codigo municipio de residencia (Candidato/SINAN) |
| C_BAIRES | Bairro de residencia (Candidato/SINAN) |
| C_SITUENCE | Situacao do encerramento (Candidato/SINAN) |
| C_IDLINHA | ID de linha (Candidato/SINAN) |
| C_IDPESSOA | ID da pessoa (Candidato/SINAN) |
| C_NOVOIDPESSOA | Novo ID da pessoa (Candidato/SINAN) |
| C_DTNASC2 | Data de nascimento alternativa (Candidato/SINAN) |
| C_BAIRES2 | Bairro alternativo (Candidato/SINAN) |
| C_IDUNID | ID unidade de saude (Candidato/SINAN) |
| C_IDUNIDAT | ID unidade de atendimento (Candidato/SINAN) |
| C_IDMUNI2 | ID municipio secundario (Candidato/SINAN) |
| C_IDMUNIAT | ID municipio de atendimento (Candidato/SINAN) |
| C_DTENCE | Data de encerramento (Candidato/SINAN) |
| C_DTNOTI | Data de notificacao (Candidato/SINAN) |
| C_DTDIAG | Data de diagnostico (Candidato/SINAN) |
| C_DTNOTIAT | Data de notificacao atualizada (Candidato/SINAN) |

### A.4. Sub-escores de comparacao (29 features para o modelo ML)

#### NOME (7 sub-escores)
| Coluna | Descricao |
|--------|-----------|
| nome prim frag igual | Primeiro fragmento do nome identico |
| nome ult frag igual | Ultimo fragmento do nome identico |
| nome qtd frag iguais | Quantidade de fragmentos identicos |
| nome qtd frag raros | Quantidade de fragmentos raros identicos |
| nome qtd frag comuns | Quantidade de fragmentos comuns identicos |
| nome qtd frag muito parec | Quantidade de fragmentos muito parecidos |
| nome qtd frag abrev | Quantidade de fragmentos abreviados |

#### NOMEMAE (7 sub-escores — mesma estrutura que NOME)
| Coluna | Descricao |
|--------|-----------|
| nomemae prim frag igual | Primeiro fragmento do nome da mae identico |
| nomemae ult frag igual | Ultimo fragmento do nome da mae identico |
| nomemae qtd frag iguais | Quantidade de fragmentos identicos (mae) |
| nomemae qtd frag raros | Quantidade de fragmentos raros identicos (mae) |
| nomemae qtd frag comuns | Quantidade de fragmentos comuns identicos (mae) |
| nomemae qtd frag muito parec | Quantidade de fragmentos muito parecidos (mae) |
| nomemae qtd frag abrev | Quantidade de fragmentos abreviados (mae) |

#### DTNASC (5 sub-escores)
| Coluna | Descricao |
|--------|-----------|
| dtnasc dt iguais | Datas de nascimento identicas |
| dtnasc dt ap 1digi | Datas diferem em 1 digito |
| dtnasc dt inv dia | Dia invertido |
| dtnasc dt inv mes | Mes invertido |
| dtnasc dt inv ano | Ano invertido |

#### CODMUNRES (4 sub-escores)
| Coluna | Descricao |
|--------|-----------|
| codmunres uf igual | UF identica |
| codmunres uf prox | UF proxima |
| codmunres local igual | Municipio identico |
| codmunres local prox | Municipio proximo |

#### ENDERECO (6 sub-escores)
| Coluna | Descricao |
|--------|-----------|
| endereco via igual | Logradouro identico |
| endereco via prox | Logradouro proximo |
| endereco numero igual | Numero identico |
| endereco compl prox | Complemento proximo |
| endereco texto prox | Texto proximo (fuzzy) |
| endereco tokens jacc | Jaccard de tokens do endereco |

### A.5. Escore agregado

| Coluna | Descricao |
|--------|-----------|
| nota final | Escore agregado calculado pelo comparador (usado para bandas e ancoras) |

---
