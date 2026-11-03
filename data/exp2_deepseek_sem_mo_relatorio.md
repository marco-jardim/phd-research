# Exp2: DeepSeek V4.1 Flash sem model_outputs

Gerado em 2026-11-03T01:35:07.664693+00:00.

Modelo deepseek-v4.1-flash via OpenCode Go. Mesmos pares históricos por modo, pipeline v3/fit_platt, mesmos prompts, parser e protocolo Agent-A + Agent-B + árbitro. Removido todo o grupo model_outputs (p_cal, base_choice, band, evr, guardrail, guardrail_reason). Subescores, nota e MACD mantidos. O prompt não foi alterado e pode mencionar campos ausentes.

Parâmetros do run original: temperature=0, top_p=1, top_k=1, presence_penalty=0, frequency_penalty=0, max_tokens=800, thinking disabled, reasoning_effort none; timeout por chamada=120 s, quatro tentativas com backoff inicial de 2 s, sem fallback distinto. Até quatro pares concorrentes; modos sequenciais. O processo do lote não tem timeout de 120 s.

Gabarito PAR ∈ {1,2}, identificadores dtype=str aparados, SHA256(COMPREC|REFREC|PASSO)[:16]. Métricas nos pares com resposta nos dois braços. UNSURE separado, taxas excluindo UNSURE e duas políticas inclusivas no CSV (UNSURE como MATCH ou NONMATCH). McNemar exato bilateral compara acerto estrito; UNSURE conta como não acerto. Sem ajuste de multiplicidade. Comparação histórica, execução única: não separa o efeito da ablação da variação temporal. Coortes condicionadas à triagem não representam todo o extrato.

## Matriz e taxas

| Modo | Braço | N | VP | FP | FN | VN | UNSURE (+/−) | Sens. sem U | Esp. sem U |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vigilancia | old | 1410 | 49 | 3 | 42 | 1256 | 60 (13/47) | 0.538462 | 0.997617 |
| vigilancia | new | 1410 | 89 | 67 | 0 | 38 | 1216 (15/1201) | 1.000000 | 0.361905 |
| confirmacao | old | 435 | 69 | 2 | 8 | 287 | 69 (24/45) | 0.896104 | 0.993080 |
| confirmacao | new | 435 | 86 | 25 | 0 | 5 | 319 (15/304) | 1.000000 | 0.166667 |

Taxas inclusivas: `exp2_deepseek_sem_mo_resumo.csv`. Comparações individuais nos CSVs `_pares_<modo>`.

## Comparação pareada e execução

```json
[
  {
    "mode": "vigilancia",
    "n_common": 1410,
    "fn_old_to_match": 32,
    "fn_old_to_unsure": 10,
    "fp_new_not_old": 64,
    "mcnemar_old_correct_new_not": 1220,
    "mcnemar_old_not_new_correct": 42,
    "mcnemar_exact_p": 1.6352289985607056e-301,
    "transitions": {
      "MATCH->MATCH": 50,
      "MATCH->UNSURE": 2,
      "UNSURE->UNSURE": 31,
      "UNSURE->MATCH": 29,
      "NONMATCH->UNSURE": 1183,
      "NONMATCH->MATCH": 77,
      "NONMATCH->NONMATCH": 38
    },
    "transport": {
      "calls": 2917,
      "http_errors": 0,
      "response_parse_errors": 0,
      "error_statuses": {},
      "input_tokens": 4458693,
      "output_tokens": 917189,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 0
    },
    "review_errors": 0,
    "cost_estimate_usd": null,
    "elapsed_s": 2553.38,
    "failed_pair_ids": [],
    "provider_pii_flags": 0
  },
  {
    "mode": "confirmacao",
    "n_common": 435,
    "fn_old_to_match": 4,
    "fn_old_to_unsure": 4,
    "fp_new_not_old": 25,
    "mcnemar_old_correct_new_not": 287,
    "mcnemar_old_not_new_correct": 22,
    "mcnemar_exact_p": 5.163909064887603e-60,
    "transitions": {
      "MATCH->MATCH": 64,
      "MATCH->UNSURE": 7,
      "UNSURE->UNSURE": 32,
      "UNSURE->MATCH": 37,
      "NONMATCH->UNSURE": 280,
      "NONMATCH->MATCH": 10,
      "NONMATCH->NONMATCH": 5
    },
    "transport": {
      "calls": 905,
      "http_errors": 0,
      "response_parse_errors": 0,
      "error_statuses": {},
      "input_tokens": 1386415,
      "output_tokens": 285003,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "max_request_input_tokens": 0
    },
    "review_errors": 0,
    "cost_estimate_usd": null,
    "elapsed_s": 807.57,
    "failed_pair_ids": [],
    "provider_pii_flags": 0
  }
]
```

Tempo elapsed_s inclui chamadas/retries, exclui reconstrução. Go consome quota de assinatura; não há valor faturado disponível. execucao.json registra tempo de parede do lote. Sondagem de quota separada em quota_probe.json; suas chamadas não entram no transporte dos pares.

Execução: {"timestamp": "2026-11-03T01:34:38.358889+00:00", "wall_seconds": 3515.04, "exit_codes": {"vigilancia": 0, "confirmacao": 0}, "workers": 4, "modes_sequential": true}

Persistência restrita a IDs pseudônimos, escores, decisões estruturadas, códigos enumerados, flags e contadores. Nenhum dossiê, resposta bruta ou evidência livre salvo. Checkpoint por par com retry na troca atômica e fallback recuperável no Windows; sucessos não são repetidos. Ver auditoria.json.

## Interpretação

FN explícitos devem ser interpretados junto com UNSURE e FP. Sensibilidade de 100% após excluir UNSURE não significa recuperação de todos os vínculos, pois retira do denominador os positivos sem decisão. O teste de McNemar acima usa acerto estrito, não apenas ausência de FN.

vigilancia: FN 42 para 0, FP 3 para 67, UNSURE 60 para 1216; acertos estritos 1305 para 127. A redução de FN não deve ser apresentada isoladamente como melhora global.

confirmacao: FN 8 para 0, FP 2 para 25, UNSURE 69 para 319; acertos estritos 356 para 91. A redução de FN não deve ser apresentada isoladamente como melhora global.
