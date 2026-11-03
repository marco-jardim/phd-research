# M4 — Ablação do dossiê (vigilância)

Seleção: 18 vínculos ouro classificados NONMATCH por ≥2 revisores; 9 pelos três.
O tamanho decorre do critério explícito NONMATCH por pelo menos dois revisores; não foi ampliado para atingir a expectativa aproximada do despacho.

## Método e fronteira de informação

Gabarito reconstruído do extrato dtype=str, PAR ∈ {1,2}; CSV de comparação conferido com os três JSONs originais. Pipeline v3/fit_platt e dossiês originais reconstruídos; ambos os braços usam os mesmos prompts, parser e protocolo A+B+árbitro. Removido o grupo inteiro model_outputs: model_outputs.p_cal, model_outputs.base_choice, model_outputs.band, model_outputs.evr, model_outputs.guardrail, model_outputs.guardrail_reason. band, EVR e guardrails também podem denunciar probabilidade/decisão. Subescores, nota e MACD mantidos. Sem alterar o prompt: instruções sobre campos ausentes permanecem (ablação do dossiê, não do prompt).

DeepSeek: T=0, top_k=1, thinking disabled/reasoning_effort none, 800 tokens. Haiku: sem temperature na API, 2500 tokens, sem pensamento estendido. Ordem dos braços contrabalanceada por modelo; até quatro tarefas simultâneas. Retries originais com backoff; HTTP 400/401/402/403/404 interrompem sem insistir. Checkpoint por par; retomada não repete sucessos. Não se salvam raw, evidência livre, dossiês ou PII.

McNemar exato bilateral: endpoint primário MATCH vs demais; secundário NONMATCH vs demais. UNSURE não é correção estrita, mas deixa de ser FN explícito. Sem ajuste de multiplicidade; análise selecionada em FN compartilhados, não generalizável a todos os pares.

## Resultados

| Modelo | N | Controle M/U/N | Sem p_cal M/U/N | FN controle → MATCH | FN controle → UNSURE | McNemar MATCH p | Controle=original |
|---|---:|---|---|---:|---:|---:|---:|
| deepseek-v4.1-flash | 18 | 0/0/18 | 9/9/0 | 9 | 9 | 0.00390625 | 18/18 |
| claude-haiku-5-5 | 18 | 0/9/9 | 7/11/0 | 2 | 7 | 0.015625 | 13/18 |

Totais dos dois modelos principais: {"calls": 159, "http_errors": 0, "parse_errors": 0, "review_errors": 0, "note": "Main two models only; optional Kimi probe reported separately."}

### deepseek-v4.1-flash

```json
{
  "model": "deepseek-v4.1-flash",
  "n": 18,
  "selected_n": 18,
  "complete": true,
  "controle": {
    "MATCH": 0,
    "UNSURE": 0,
    "NONMATCH": 18
  },
  "sem_pcal": {
    "MATCH": 9,
    "UNSURE": 9,
    "NONMATCH": 0
  },
  "transitions": {
    "MATCH->MATCH": 0,
    "MATCH->UNSURE": 0,
    "MATCH->NONMATCH": 0,
    "UNSURE->MATCH": 0,
    "UNSURE->UNSURE": 0,
    "UNSURE->NONMATCH": 0,
    "NONMATCH->MATCH": 9,
    "NONMATCH->UNSURE": 9,
    "NONMATCH->NONMATCH": 0
  },
  "gained_match": 9,
  "lost_match": 0,
  "mcnemar_match_exact_p": 0.00390625,
  "fn_to_nonfn": 18,
  "nonfn_to_fn": 0,
  "mcnemar_nonmatch_exact_p": 7.62939453125e-06,
  "original_fn_to_match": 9,
  "original_fn_to_match_or_unsure": 18,
  "original_fn_n": 18,
  "control_fn_n": 18,
  "control_fn_to_match": 9,
  "control_fn_to_unsure": 9,
  "control_original_agree": 18,
  "control_original_agreement": 1.0,
  "transport": {
    "controle": {
      "calls": 38,
      "http_errors": 0,
      "error_statuses": {},
      "input_tokens": 60909,
      "output_tokens": 12881,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0
    },
    "sem_pcal": {
      "calls": 40,
      "http_errors": 0,
      "error_statuses": {},
      "input_tokens": 63176,
      "output_tokens": 12710,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0
    }
  },
  "elapsed_s": {
    "controle": 151.32,
    "sem_pcal": 151.59
  },
  "review_errors": 0,
  "cost_usd": null,
  "cost_estimate_usd": null
}
```

Interpretação: a ablação melhora MATCH no conjunto pareado, compatível com ancoragem nos outputs do motor; não isola p_cal de seus proxies.

### claude-haiku-5-5

```json
{
  "model": "claude-haiku-5-5",
  "n": 18,
  "selected_n": 18,
  "complete": true,
  "controle": {
    "MATCH": 0,
    "UNSURE": 9,
    "NONMATCH": 9
  },
  "sem_pcal": {
    "MATCH": 7,
    "UNSURE": 11,
    "NONMATCH": 0
  },
  "transitions": {
    "MATCH->MATCH": 0,
    "MATCH->UNSURE": 0,
    "MATCH->NONMATCH": 0,
    "UNSURE->MATCH": 5,
    "UNSURE->UNSURE": 4,
    "UNSURE->NONMATCH": 0,
    "NONMATCH->MATCH": 2,
    "NONMATCH->UNSURE": 7,
    "NONMATCH->NONMATCH": 0
  },
  "gained_match": 7,
  "lost_match": 0,
  "mcnemar_match_exact_p": 0.015625,
  "fn_to_nonfn": 9,
  "nonfn_to_fn": 0,
  "mcnemar_nonmatch_exact_p": 0.00390625,
  "original_fn_to_match": 2,
  "original_fn_to_match_or_unsure": 10,
  "original_fn_n": 10,
  "control_fn_n": 9,
  "control_fn_to_match": 2,
  "control_fn_to_unsure": 7,
  "control_original_agree": 13,
  "control_original_agreement": 0.7222222222222222,
  "transport": {
    "controle": {
      "calls": 40,
      "http_errors": 0,
      "error_statuses": {},
      "input_tokens": 95243,
      "output_tokens": 82562,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 3393
    },
    "sem_pcal": {
      "calls": 41,
      "http_errors": 0,
      "error_statuses": {},
      "input_tokens": 95421,
      "output_tokens": 73376,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 3257
    }
  },
  "elapsed_s": {
    "controle": 359.38,
    "sem_pcal": 333.06
  },
  "review_errors": 0,
  "cost_usd": null,
  "cost_estimate_usd": {
    "controle": 0.0508053,
    "sem_pcal": 0.0462301
  }
}
```

Interpretação: a ablação melhora MATCH no conjunto pareado, compatível com ancoragem nos outputs do motor; não isola p_cal de seus proxies.

## Custo, tempo e limites

Tokens de todas as respostas HTTP bem-sucedidas (inclusive tentativas com JSON inválido) registrados por braço. USD efetivamente faturado não é devolvido por estes transportes; Go consome quota de assinatura. Para Haiku, cost_estimate_usd usa tarifa pública standard/global por milhão: entrada $0.10, saída $0.50, cache read $0.01, cache write 5m $0.125 (prompts ≤100k, sem cache solicitado). Não é fatura. Fonte consultada: https://platform.claude.com/docs/en/about-claude/pricing (2026-11-02). elapsed_s mede tempo de parede por braço, inclui retries; braços paralelos não devem ter seus tempos somados como duração de parede total. Concordância controle-original mede repetibilidade no subconjunto selecionado, não acurácia geral. Campos ausentes podem induzir UNSURE.

## Auditoria da seleção

```json
{
  "common_reviewed_n": 1410,
  "gold_positive_n": 104,
  "original_fn_counts": {
    "Kimi K2.5": 19,
    "DeepSeek V4.1 Flash": 42,
    "Haiku 5.5": 10
  },
  "fn_union_at_least_one": 44,
  "fn_shared_at_least_two": 18,
  "fn_all_three": 9
}
```

## Fumaça prévia e inspeção

{"two_pairs_each_arm": true, "same_pairs": true, "http_errors": 0, "wall_seconds_including_kimi": 83.57, "kimi_successful_pairs": 3}
Os dois pares iniciais por braço foram inspecionados antes da execução completa; sucessos reaproveitados sem novas chamadas.

Última execução (retomada após fumaça): {
  "wall_seconds": 216.21,
  "limit": 0,
  "workers": 4,
  "arms": "ambos",
  "note": "resumed successes are not called again"
}

## Interrupção operacional e limites da contagem

{
  "reason": "background executor terminated first continuation at 120 seconds; no API error observed",
  "executor_limit_s": 120,
  "before_resume": {
    "deepseek-v4.1-flash_controle": {
      "completed_pairs": 15,
      "checkpoint_calls": 31
    },
    "deepseek-v4.1-flash_sem_pcal": {
      "completed_pairs": 14,
      "checkpoint_calls": 32
    },
    "claude-haiku-5-5_controle": {
      "completed_pairs": 7,
      "checkpoint_calls": 17
    },
    "claude-haiku-5-5_sem_pcal": {
      "completed_pairs": 8,
      "checkpoint_calls": 18
    }
  },
  "historical_call_count_exact": false,
  "possible_uncheckpointed_calls": "unknown; one in-progress review per arm, up to three API attempts per unfinished review without retries",
  "successful_pairs_reused": true
}
As estatísticas preservadas são um limite inferior das chamadas/tokens, pois a primeira retomada foi terminada pelo limite de tempo do executor durante pares em andamento. Os resultados completos já checkpointados não foram repetidos; pares incompletos foram retomados. A retomada final usa journaling por chamada e executor sem limite de tempo. Isso não altera o teste pareado dos resultados finais, mas impede afirmar custo faturado ou número histórico exato de chamadas não checkpointadas.

## Sondagem opcional Kimi K3

{"n": 3, "errors": [], "transport": {"calls": 7, "http_errors": 0, "error_statuses": {}, "input_tokens": 11790, "output_tokens": 7724, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, "elapsed_s": 39.88}
reasoning_effort=low; no máximo três pares, interrupção imediata em HTTP 400/quota; exploratório, sem inferência pareada.
