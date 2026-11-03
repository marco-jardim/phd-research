# Exp2 — Sage (Levanto)

## Método

Coortes e ordem dos JSONs Anthropic originais: 1.410 vigilância, 435 confirmação. Gabarito reconstruído de COMPARADORSEMIDENT.csv (dtype=str), PAR∈{1,2}; SHA256(COMPREC|REFREC|PASSO)[:16], campos aparados. Dossiês originais gzcmd/llm_dossier.py; sem_mo remove o bloco model_outputs inteiro. A seleção é a coorte histórica fixa, não a triagem recalculada hoje. Os identificadores do pipeline são alinhados aos campos string originais por junção normalizada validada um-para-um; somente o hash string original é persistido. Não se salvam dossiês, PII, chaves ou raciocínio livre.

Uma decisão choice por par (não o protocolo A+B+árbitro); opções fixas match/nonmatch. reasoning=auto, grounding.trigger=never. Instruções originais + adaptação por modo registradas nos JSONs. Vigilância tolera incerteza moderada a favor do vínculo; confirmação exige evidência forte. chosen=null é UNSURE, sem retry. Checkpoint a cada par, hash da requisição na retomada; 429/500/502/503/504 têm backoff/Retry-After; 404 recebe até três tentativas com backoff conforme diagnóstico solicitado; 402 interrompe imediatamente a sequência. Concorrência padrão 1 para parada sem outras chamadas em voo.

Compatibilidade de transporte: após HTTP400 do deployment confidencial, a retomada pode representar grounding desligado como null (campo nullable, default null no OpenAPI oficial), em vez de {trigger: never}. A mudança é registrada em wire_encoding_changes, com hash das respostas preservadas; os hashes canônicos dos dossiês/instruções não são alterados. Cada nova resposta é verificada para ausência de ativação de grounding.

## Probabilidades e denominadores

Segundo https://docs.levanto.ai/decision-model/choice.md, probability=P(chosen correto), não P(match); pode ser null na abstenção. O valor bruto é preservado. AUC/Brier e confiabilidade principais usam P(match) em probabilities[match], inclusive UNSURE. Os campos *_raw_chosen_vs_y atendem ao diagnóstico literal solicitado, mas NÃO são métricas válidas de risco de vínculo, pois a classe-alvo muda conforme chosen. Limiares 0,8/0,9/0,95/0,98 aplicam-se à confiança da escolha (probability), com abstenção abaixo; não são limiares P(match). Curva por decis fixos [0,0.1), ..., [0.9,1].

sens/espec_sem_unsure excluem abstenções; *_unsure_como_erro mantêm todos os gabaritos nos denominadores sem dar crédito por abstenção. Também são reportadas taxas com UNSURE como vínculo. Denominador zero = NA. Execuções parciais não representam a coorte completa. Comparação dos outros revisores é restrita aos mesmos pares concluídos; diferenças incluem transporte/protocolo e não isolam modelo.

## Resultados (NA quando não há observações)

| Modo | Braço | Modelo observado | Status | N/esperado | VP | FP | FN | VN | UNSURE | AUC P(match) | Brier |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vigilancia | sem_mo | levanto-sage-2 | complete | 1410/1410 | 65 | 164 | 16 | 719 | 446 | 0.8180881140299211 | 0.18053607560567503 |
| confirmacao | sem_mo | levanto-sage-2 | complete | 435/435 | 38 | 7 | 50 | 302 | 38 | 0.745479338352997 | 0.13674765169970454 |
| vigilancia | com_mo | levanto-sage-2 | complete | 1410/1410 | 49 | 9 | 21 | 1224 | 107 | 0.9202423724820354 | 0.04085352461795126 |

### Sensibilidade e especificidade (proporções)

| Modo/braço | Sens. sem U | Espec. sem U | Sens. U=erro | Espec. U=erro |
|---|---:|---:|---:|---:|
| vigilancia/sem_mo | 0.802469 | 0.81427 | 0.625 | 0.550536 |
| confirmacao/sem_mo | 0.431818 | 0.977346 | 0.376238 | 0.904192 |
| vigilancia/com_mo | 0.7 | 0.992701 | 0.471154 | 0.937213 |

### Limiar próprio na confiança da escolha

| Modo/braço | Limiar | VP | FP | FN | VN | U | Sens. sem U | Espec. sem U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vigilancia/sem_mo | 0.8 | 62 | 132 | 12 | 516 | 688 | 0.837838 | 0.796296 |
| vigilancia/sem_mo | 0.9 | 62 | 121 | 8 | 444 | 775 | 0.885714 | 0.785841 |
| vigilancia/sem_mo | 0.95 | 47 | 77 | 3 | 278 | 1005 | 0.94 | 0.783099 |
| vigilancia/sem_mo | 0.98 | 16 | 0 | 2 | 86 | 1306 | 0.888889 | 1 |
| confirmacao/sem_mo | 0.8 | 36 | 5 | 42 | 289 | 63 | 0.461538 | 0.982993 |
| confirmacao/sem_mo | 0.9 | 35 | 5 | 34 | 277 | 84 | 0.507246 | 0.98227 |
| confirmacao/sem_mo | 0.95 | 19 | 0 | 27 | 149 | 240 | 0.413043 | 1 |
| confirmacao/sem_mo | 0.98 | 0 | 0 | 12 | 94 | 329 | 0 | 1 |
| vigilancia/com_mo | 0.8 | 35 | 4 | 17 | 1166 | 188 | 0.673077 | 0.996581 |
| vigilancia/com_mo | 0.9 | 31 | 4 | 14 | 1025 | 336 | 0.688889 | 0.996113 |
| vigilancia/com_mo | 0.95 | 21 | 3 | 6 | 796 | 584 | 0.777778 | 0.996245 |
| vigilancia/com_mo | 0.98 | 3 | 0 | 1 | 237 | 1169 | 0.75 | 1 |

### Comparação nos mesmos pares concluídos

| Modo/braço | Revisor original | N comum | Concordância | VP/FP/FN/VN/U original |
|---|---|---:|---:|---|
| vigilancia/sem_mo | Haiku 5.5 | 1410 | 0.508511 | 35/0/10/1103/262 |
| vigilancia/sem_mo | DeepSeek V4.1 Flash | 1410 | 0.532624 | 49/3/42/1256/60 |
| vigilancia/sem_mo | Kimi K2.5 | 1410 | 0.549645 | 79/6/19/1265/41 |
| confirmacao/sem_mo | Haiku 5.5 | 435 | 0.498851 | 55/1/5/203/171 |
| confirmacao/sem_mo | DeepSeek V4.1 Flash | 435 | 0.682759 | 69/2/8/287/69 |
| confirmacao/sem_mo | Kimi K3 | 343 | 0.804665 | 57/3/16/248/19 |
| vigilancia/com_mo | Haiku 5.5 | 1410 | 0.804965 | 35/0/10/1103/262 |
| vigilancia/com_mo | DeepSeek V4.1 Flash | 1410 | 0.895745 | 49/3/42/1256/60 |
| vigilancia/com_mo | Kimi K2.5 | 1410 | 0.904965 | 79/6/19/1265/41 |

O CSV fornecido de confirmação identifica Kimi K3, não K2.5; a comparação mantém esse rótulo e restringe-se aos pares com decisão disponível. Não se fundem versões Kimi nem se imputam ausências. As decisões pareadas, gabarito e P(match) estão em data/exp2_sage_comparacao_pares.csv; data/exp2_sage_comparacao.csv contém os agregados por revisor/modo/braço.

## Ancoragem — vigilância pareada

Pares em comum: 1410. Transições com_mo → sem_mo: {"NONMATCH->NONMATCH": 711, "NONMATCH->UNSURE": 403, "UNSURE->NONMATCH": 24, "MATCH->UNSURE": 6, "NONMATCH->MATCH": 131, "MATCH->MATCH": 52, "UNSURE->MATCH": 46, "UNSURE->UNSURE": 37}. CSV pareado: data/exp2_sage_ancoragem.csv. A ordem fixa (sem_mo primeiro) e uma única realização por braço limitam interpretação causal/variabilidade.
McNemar exato bilateral: MATCH vs demais; endpoint secundário FN explícito vs demais somente nos vínculos ouro. UNSURE não é correção estrita; reversões FN→MATCH e FN→UNSURE separadas. Sem ajuste de multiplicidade; efeito não isola p_cal de seus proxies.
```json
{
  "n_paired": 1410,
  "gained_match_sem_mo": 177,
  "lost_match_sem_mo": 6,
  "mcnemar_match_exact_p": 8.105074935312312e-45,
  "fn_com_mo": 21,
  "fn_sem_mo": 16,
  "fn_com_to_sem_match": 5,
  "fn_com_to_sem_unsure": 6,
  "fn_reverted_to_nonfn": 11,
  "fn_introduced": 6,
  "mcnemar_fn_exact_p": 0.332305908203125
}
```

## Latência, consumo e limites

Distribuição min/p50/p90/p95/p99/max/média de latência da API e tempo do cliente no resumo CSV. Tokens/custo quantitativos são preservados se devolvidos, em telemetria.json e nos metadados por par; somas por campo e número de decisões com o campo informado constam no resumo CSV (api_*). campos ausentes não significam consumo zero. A estimativa de um dossiê fica em preflight.json, não é medição de todo o lote nem fatura. Não extrapolar custo sem tarifa/consumo observados. Resumo, curva de confiabilidade, limiares e comparação estão nos CSVs data/exp2_sage_*.csv.

- vigilancia/sem_mo: latência API p50=7535.95 ms, p95=13630.1 ms; input_tokens=3.50374e+06, output_tokens=757091. Totais das decisões checkpointadas, não uma fatura incluindo sondagens e eventuais falhas. Preço do plano/custo USD não foi informado pela API; nenhum custo em dólares é inferido.
  Chamadas /decide rastreadas após inclusão do contador: 1105; limite inferior histórico anterior: 309. O histórico anterior não tinha journaling por tentativa; não se afirma total exato faturado.
- confirmacao/sem_mo: latência API p50=3780.8 ms, p95=9412.73 ms; input_tokens=888635, output_tokens=141061. Totais das decisões checkpointadas, não uma fatura incluindo sondagens e eventuais falhas. Preço do plano/custo USD não foi informado pela API; nenhum custo em dólares é inferido.
  Chamadas /decide rastreadas após inclusão do contador: 432; limite inferior histórico anterior: 3. O histórico anterior não tinha journaling por tentativa; não se afirma total exato faturado.
- vigilancia/com_mo: latência API p50=183.8 ms, p95=9223.45 ms; input_tokens=2.48107e+06, output_tokens=321094. Totais das decisões checkpointadas, não uma fatura incluindo sondagens e eventuais falhas. Preço do plano/custo USD não foi informado pela API; nenhum custo em dólares é inferido.
  Chamadas /decide rastreadas após inclusão do contador: 1410; limite inferior histórico anterior: 0. O histórico anterior não tinha journaling por tentativa; não se afirma total exato faturado.
## Preflight / interrupção

```json
{
  "started_at": "2026-11-02T22:14:49.546970+00:00",
  "ready": {
    "status": "ready",
    "load": "low"
  },
  "estimate": {
    "usage": {
      "min_input_tokens": 1288.0,
      "max_input_tokens": 2576.0,
      "exact": false,
      "min_output_tokens": 0.0,
      "max_output_tokens": 800.0,
      "sequences": 1.0
    }
  },
  "estimate_pair_id": "9552589dccf390f6",
  "estimate_raw_keys": [
    "usage"
  ],
  "updated_at": "2026-11-03T01:00:02.143390+00:00",
  "quota_exhausted": false,
  "calls_this_process": {
    "/ready": 2,
    "/decide": 2053
  },
  "completed": {
    "vigilancia": 1410,
    "confirmacao": 435
  },
  "http_diagnostics_this_process": [
    {
      "endpoint": "/ready",
      "attempt": 1,
      "status": 503,
      "final_url": "https://sage.levanto.ai/ready",
      "body_literal": "{\"status\":\"unavailable\"}"
    }
  ],
  "execution_history": [
    {
      "finished_at": "2026-11-02T23:38:52.251330+00:00",
      "resume_after_quota": true,
      "quota_exhausted": false,
      "calls": {
        "/ready": 1,
        "/decide": 382
      },
      "completed": {
        "vigilancia": 688,
        "confirmacao": 3
      }
    },
    {
      "finished_at": "2026-11-02T23:41:09.694224+00:00",
      "resume_after_quota": true,
      "quota_exhausted": false,
      "calls": {
        "/ready": 1,
        "/decide": 7
      },
      "completed": {
        "vigilancia": 694,
        "confirmacao": 3
      }
    },
    {
      "finished_at": "2026-11-03T00:24:14.886962+00:00",
      "resume_after_quota": true,
      "quota_exhausted": false,
      "calls": {
        "/ready": 1,
        "/decide": 271
      },
      "completed": {
        "vigilancia": 1199,
        "confirmacao": 3
      }
    },
    {
      "finished_at": "2026-11-03T01:00:02.094205+00:00",
      "resume_after_quota": true,
      "quota_exhausted": false,
      "calls": {
        "/ready": 2,
        "/decide": 2053
      },
      "completed": {
        "vigilancia": 1410,
        "confirmacao": 435
      }
    }
  ]
}
```

## Sondagem mínima antes da retomada

```json
{
  "timestamp": "2026-11-02T22:56:00.531910+00:00",
  "method": "POST",
  "final_url": "https://sage.levanto.ai/decide",
  "status": 200,
  "requests": 1,
  "body_literal": "{\"id\":\"quota_probe\",\"kind\":\"yesno\",\"result\":{\"answer\":\"yes\",\"probability\":0.9968938942454479},\"meta\":{\"model\":\"levanto-sage-2\",\"latency_ms\":67.2,\"question_count\":1,\"usage\":{\"billed_input_tokens\":57,\"rendered_tokens\":57,\"image_count\":0,\"image_tokens\":0,\"input_tokens\":57,\"output_tokens\":0},\"reasoning\":{\"ran\":false,\"finished\":null,\"tokens\":null,\"limited\":null}},\"grounding_meta\":null}",
  "checkpoint_counts": {
    "vigilancia": {
      "completed": 307,
      "expected": 1410,
      "status": "quota_exhausted"
    },
    "confirmacao": {
      "completed": 3,
      "expected": 435,
      "status": "smoke_complete"
    }
  }
}
```

Uma sondagem curta HTTP 200 apenas confirma que essa requisição foi aceita; não garante cota suficiente para o dossiê real. Respostas existentes são preservadas na retomada.
## Diagnóstico HTTP isolado

Quickstart oficial confirma https://sage.levanto.ai/decide e Authorization: Bearer. A chave do .env tem prefixo lv_live_ (somente os oito caracteres públicos foram exibidos). Sondagens sintéticas yesno e choice com reasoning=auto retornaram 200, modelo levanto-sage-2; não foi identificada divergência de host/caminho. O 404 inicial não prova rota inexistente. Status, URL final e respostas literais das sondagens estão em diagnostico_http.json. Variantes /v1/decide são testadas apenas se a sondagem /decide retornar 404.
