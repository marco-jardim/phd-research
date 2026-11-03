# Exp2 — Haiku 5.5 sem model_outputs

Gerado em 2026-11-03T00:29:30.147522+00:00.

Modelo exato: claude-haiku-5-5. Mesmos pares históricos por modo, reconstrução v3/fit_platt; mesmo prompt, parser e protocolo A+B+árbitro. Removido todo o bloco model_outputs (p_cal, base_choice, band, evr, guardrail, guardrail_reason). Subescores, nota e MACD mantidos. O prompt não foi alterado e pode mencionar os campos ausentes.

max_tokens=2500; sem temperature e sem extended thinking; até quatro pares concorrentes (modos sequenciais). Retries com backoff exponencial, cinco tentativas; checkpoint por par e journal de tokens por chamada. Três pares de fumaça por modo reaproveitados; nenhum sucesso é chamado novamente.

Gabarito PAR ∈ {1,2}; identificadores dtype=str aparados, SHA256(COMPREC|REFREC|PASSO)[:16]. Métricas old/new nos pares com resposta em ambos. UNSURE explicitamente separado; são mostradas taxas excluindo UNSURE e duas políticas inclusivas (UNSURE como MATCH e como NONMATCH). McNemar exato bilateral compara acerto estrito contra gabarito, contando UNSURE como não acerto. Não há ajuste de multiplicidade; execução estocástica única, comparação histórica não controla variação temporal. Amostras condicionadas à triagem não representam todo o extrato.

## Matriz e taxas

| Modo | Braço | N | VP | FP | FN | VN | UNSURE (+/−) | Sens. sem U | Esp. sem U |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vigilancia | old | 1410 | 35 | 0 | 10 | 1103 | 262 (59/203) | 0.777778 | 1.000000 |
| vigilancia | new | 1410 | 67 | 6 | 2 | 77 | 1258 (35/1223) | 0.971014 | 0.927711 |
| confirmacao | old | 435 | 55 | 1 | 5 | 203 | 171 (41/130) | 0.916667 | 0.995098 |
| confirmacao | new | 435 | 72 | 3 | 0 | 13 | 347 (29/318) | 1.000000 | 0.812500 |

Taxas inclusivas e todos os contadores estão em `exp2_haiku_sem_mo_resumo.csv`.

## Comparação pareada, chamadas, tokens, custo e tempo

```json
[
  {
    "mode": "vigilancia",
    "n_common": 1410,
    "fn_old_to_match": 2,
    "fn_old_to_unsure": 7,
    "fp_new_not_old": 6,
    "mcnemar_old_correct_new_not": 1031,
    "mcnemar_old_not_new_correct": 37,
    "mcnemar_exact_p": 2.8924830759196795e-253,
    "transitions": {
      "MATCH->MATCH": 32,
      "UNSURE->UNSURE": 223,
      "UNSURE->MATCH": 36,
      "MATCH->UNSURE": 3,
      "NONMATCH->UNSURE": 1032,
      "UNSURE->NONMATCH": 3,
      "NONMATCH->NONMATCH": 76,
      "NONMATCH->MATCH": 5
    },
    "transport": {
      "calls": 2930,
      "http_errors": 0,
      "response_parse_errors": 1,
      "error_statuses": {},
      "input_tokens": 6565396,
      "output_tokens": 4955739,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 3462
    },
    "review_errors": 0,
    "cost_estimate_usd": 3.1344091,
    "elapsed_s": 5731.28,
    "failed_pair_ids": [],
    "provider_pii_flags": 0
  },
  {
    "mode": "confirmacao",
    "n_common": 435,
    "fn_old_to_match": 0,
    "fn_old_to_unsure": 5,
    "fp_new_not_old": 3,
    "mcnemar_old_correct_new_not": 193,
    "mcnemar_old_not_new_correct": 20,
    "mcnemar_exact_p": 1.0238999124749158e-36,
    "transitions": {
      "MATCH->MATCH": 52,
      "UNSURE->UNSURE": 149,
      "UNSURE->MATCH": 22,
      "NONMATCH->UNSURE": 194,
      "NONMATCH->NONMATCH": 13,
      "NONMATCH->MATCH": 1,
      "MATCH->UNSURE": 4
    },
    "transport": {
      "calls": 904,
      "http_errors": 0,
      "response_parse_errors": 0,
      "error_statuses": {},
      "input_tokens": 2023425,
      "output_tokens": 1562677,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 3439
    },
    "review_errors": 0,
    "cost_estimate_usd": 0.983681,
    "elapsed_s": 1831.68,
    "failed_pair_ids": [],
    "provider_pii_flags": 0
  }
]
```

Custo estimado, não fatura: tarifas standard/global já documentadas na ablação, USD/M entrada 0.10, saída 0.50, cache leitura 0.01 e escrita 0.125 (até 100k). Fonte: https://platform.claude.com/docs/en/about-claude/pricing (2026-11-02). Tempo elapsed_s soma sessões de API incluindo fumaça/retries; exclui reconstrução do pipeline. O arquivo execucao.json registra duração de parede da execução completa final.

Execução final: {"smoke_successes_reused": true, "wall_seconds": 7861.4305294, "resume_wall_seconds": 3061.6874718, "finished_at": "2026-11-03T00:29:20.8869796+00:00", "started_at": "2026-11-02T22:18:19.4564502+00:00", "modes_sequential": true, "workers": 4, "resumed_at": "2026-11-02T23:38:19.1995078+00:00", "wall_includes_interruption": true}

## Fumaça e recuperação de checkpoint

```json
{
  "smoke": {
    "timestamp": "2026-11-02T22:18:14.111407+00:00",
    "inspected_before_full_run": true,
    "modes": {
      "vigilancia": {
        "n": 3,
        "decisions": [
          "MATCH",
          "MATCH",
          "MATCH"
        ],
        "protocols": [
          "consensus",
          "consensus",
          "consensus"
        ],
        "calls": 6,
        "http_errors": 0,
        "parse_errors": 0,
        "pii_flags": 0,
        "free_response_fields": false,
        "elapsed_s": 20.8
      },
      "confirmacao": {
        "n": 3,
        "decisions": [
          "MATCH",
          "MATCH",
          "MATCH"
        ],
        "protocols": [
          "consensus",
          "consensus",
          "consensus"
        ],
        "calls": 6,
        "http_errors": 0,
        "parse_errors": 0,
        "pii_flags": 0,
        "free_response_fields": false,
        "elapsed_s": 18.29
      }
    }
  },
  "windows_checkpoint_recovery": {
    "timestamp": "2026-11-02T23:37:48.526698+00:00",
    "reason": "Windows PermissionError replacing tmp checkpoint; no HTTP failure",
    "checkpoint_counts": {
      "exp2_llm_sync_vigilancia.json": 1148,
      "exp2_llm_sync_vigilancia.tmp": 1149
    },
    "journal_complete_results": 1152,
    "canonical_not_in_journal": 0,
    "tmp_not_in_journal": 0,
    "journal_extra_vs_canonical": 4,
    "transport_recovered": {
      "calls": 2383,
      "http_errors": 0,
      "response_parse_errors": 1,
      "error_statuses": {},
      "input_tokens": 5328345,
      "output_tokens": 4034665,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 3462
    },
    "recovery_policy": "journal successful results authoritative; reuse all successes; canonical/tmp/fallback latest usage; no full-cohort repeat"
  }
}
```

A troca atômica bloqueada pelo Windows recebeu cinco retries com backoff de 0,2 até 3 segundos; bloqueio persistente produz checkpoint alternativo sem abortar inferência. Na retomada, o journal recuperou sucessos além do JSON/tmp; nenhum sucesso foi reexecutado. elapsed_s acumula as sessões de API; wall_seconds inclui reconstrução e intervalo de recuperação desde o primeiro início do lote completo, excluindo a fumaça prévia.

Persistência restrita a IDs pseudônimos, decisões estruturadas, códigos enumerados, flags e contadores; sem dossiês, evidências livres ou respostas brutas. Ver auditoria.json para auditoria de segredo e esquema.

## Interpretação dos resultados

- Vigilância: os FN explícitos caíram de 10 para 2. Dos 10 FN antigos, somente 2 passaram a MATCH (correção estrita); 7 passaram a UNSURE (abstenção). Os FP totais aumentaram de 0 para 6, todos novos. UNSURE aumentou de 262 para 1.258 (89,22% do lote).
- Confirmação: os FN explícitos caíram de 5 para 0. Nenhum dos 5 FN antigos passou a MATCH; os 5 passaram a UNSURE. Os FP totais aumentaram de 1 para 3, com 3 FP novos (o FP antigo deixou de ser MATCH). UNSURE aumentou de 171 para 347 (79,77% do lote).

A redução de NONMATCH entre vínculos verdadeiros não equivale a recuperação automática dos vínculos: a maior parte dos FN antigos virou abstenção. A sensibilidade calculada somente nos casos resolvidos aumenta, mas condiciona-se a um subconjunto muito menor. Houve aumento de FP e forte aumento da carga potencial de revisão humana. O McNemar de acerto estrito favorece o dossiê completo, pois conta abstenção como não acerto; não deve ser interpretado como teste de erro binário com custo clínico equivalente para UNSURE e decisão errada. Não se sustenta uma melhoria global do revisor com base apenas na redução dos FN explícitos.

Os números desta interpretação estão nas linhas old/new de `data/exp2_haiku_sem_mo_resumo.csv` e nos contadores/transições de `data/sprint3b/anthropic_sem_mo/comparacao.json`, gerados por `python scripts/run_exp2_sem_model_outputs.py --summarize-only`.

## Totais, tentativas adicionais e verificação final

Foram concluídas 1.845 revisões, com 3.834 chamadas, 8.588.821 tokens de entrada e 6.518.416 tokens de saída. Custo estimado total: US$ 4,1180901, não faturamento confirmado. Não houve erro HTTP nem par sem resposta ao final. Houve 1 erro de extração de JSON registrado, recuperado por retry.

| Modo | Chamadas | Mínimo A+B+árbitros finais | Chamadas adicionais | Erros de parsing registrados | Adicionais sem classificação específica |
|---|---:|---:|---:|---:|---:|
| Vigilância | 2.930 | 2.925 | 5 | 1 | 4 |
| Confirmação | 904 | 899 | 5 | 0 | 5 |

O mínimo decorre de `2 × n_reviewed + metrics.protocols.arbiter` nos JSONs finais (105 árbitros em vigilância e 29 em confirmação). Há 10 chamadas além desse mínimo: 1 associada ao erro de parsing instrumentado e 9 sem classificação específica no medidor. O transporte original também repete respostas truncadas em max_tokens, mas não registrou essa causa separadamente; não é possível atribuir retrospectivamente as 9 tentativas a truncamento com certeza. Portanto, zero erros finais de revisão não significa zero tentativas malsucedidas. Os tokens e o custo incluem todas as chamadas contabilizadas, inclusive essas tentativas.

O tempo acumulado registrado nas sessões de API foi 7.562,96 s (5.731,28 s em vigilância e 1.831,68 s em confirmação). A duração de parede desde o primeiro início do lote completo até o fim dos dois modos foi 7.861,43 s, aproximadamente 2 h 11 min 1 s, incluindo reconstrução e interrupção para recuperação do checkpoint. Esses tempos têm fronteiras diferentes: a duração de parede exclui a fumaça prévia, incluída nos tempos acumulados de API.

Verificação final realizada nesta execução:

- As 1.410 e 435 respostas têm IDs únicos e reproduzem exatamente as coortes históricas.
- As 1.845 linhas finais são iguais aos respectivos resultados do journal; não há pares pendentes.
- Auditoria de 3.710 arquivos encontrou zero ocorrências da chave Anthropic, comparada somente em memória, e validou os campos permitidos e códigos enumerados dos resultados. Nenhum resultado sinalizou vazamento de PII; não foram persistidos dossiês ou texto livre de resposta.
- `python -m pytest tests -q`: **144 passed, 2 warnings**, em 11,52 s. Os avisos são de divisão por zero/perfect separation no teste existente `tests/test_m8_blocagem.py::test_loglinear_known_missing_cell`; não houve falhas.
