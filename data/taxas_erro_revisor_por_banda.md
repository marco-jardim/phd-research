# Taxas de erro do revisor LLM — Exp2

## Método e limites

Gabarito: PAR ∈ {1,2}; pair_id = SHA256(COMPREC|REFREC|PASSO)[:16], campos aparados. Extrato lido com dtype=str e conversão explícita to_numeric. Decisões aparadas, maiúsculas e validadas em MATCH/NONMATCH/UNSURE; não se usa confiança do LLM.

e_fp = FP/n_não_vínculos; e_fn = FN/n_vínculos; u = UNSURE/n. UNSURE permanece nos denominadores de classe, não é contado como FP/FN. IC 95% Wilson bilateral (z=1,9599639845), sem correção. NA significa denominador zero, não erro zero.

Pooled inclui apenas K2.5, DeepSeek e Haiku completos; K3 é reportado à parte. As coortes completas são verificadas idênticas por modo: pooling de contagens equivale à média das taxas de modelo ponderada por n por banda. Não há deduplicação entre revisores: n mede decisões, não pares independentes. IC pooled é descritivo, pois revisões do mesmo par são correlacionadas. Os modos também podem sobrepor pares.

K3: seleção parcial por interrupção de quota, 343/435 de confirmação; não extrapolar para os 92 não revisados. Ignorados *.parcial336.json e resume_sem_erros.json. Haiku é estocástico (sem temperature); uma execução não estima variabilidade entre execuções. Amostra condicionada à triagem Exp2: não generalizar taxas para todos os pares.

**Divergência de localização:** a confirmação K2.5 solicitada em `data/sprint3b/exp2_llm_sync_confirmacao.json` não existe neste checkout. Quando ausente, o script usa explicitamente `data/qw2/qw2_confirmacao_kimi.json` (435 pares, modelo accounts/fireworks/models/kimi-k2p5), valida a mesma coorte e registra `alternate_source: true`. Esse artefato QW2 pode ter protocolo/retries diferentes do run original: não interpretar as diferenças como efeito causal de modelo. O run opencode_server de 148 pares sem modelo não é utilizado. Se o artefato original for restaurado no caminho solicitado, ele terá precedência.

## Artefatos incluídos

| Modelo | Modo | n | Parcial | Arquivo |
|---|---|---:|---|---|
| kimi-k2.5 | vigilancia | 1410 | False | `data/sprint3b/exp2_llm_sync_vigilancia.json` |
| kimi-k2.5 | confirmacao | 435 | False | `data/qw2/qw2_confirmacao_kimi.json` |
| deepseek-v4.1-flash | vigilancia | 1410 | False | `data/sprint3b/opencode_go/exp2_llm_sync_vigilancia.json` |
| deepseek-v4.1-flash | confirmacao | 435 | False | `data/sprint3b/opencode_go/exp2_llm_sync_confirmacao.json` |
| claude-haiku-5-5 | vigilancia | 1410 | False | `data/sprint3b/anthropic/exp2_llm_sync_vigilancia.json` |
| claude-haiku-5-5 | confirmacao | 435 | False | `data/sprint3b/anthropic/exp2_llm_sync_confirmacao.json` |
| kimi-k3 | confirmacao | 343 | True | `data/sprint3b/opencode_go_kimi/exp2_llm_sync_confirmacao.json` |

## Referência fixa atual

| Banda | e_fp fixo | e_fn fixo |
|---|---:|---:|
| grey_low | 0.0800 | 0.1200 |
| grey_mid | 0.0600 | 0.1000 |
| grey_high | 0.0400 | 0.0800 |
| near_high | 0.0200 | 0.0600 |
| low | 0.1000 | 0.1500 |
| high | 0.0100 | 0.0300 |

## Estimativas por modo × modelo × banda

Taxas e IC em proporções; Δ é diferença absoluta em relação ao config, não diferença com sinal. ⚠ sinaliza n<10; denominadores de classe pequenos também são sinalizados no CSV.

| Modo | Modelo | Banda | n | Vínculos | e_fp [IC95%] | e_fn [IC95%] | u [IC95%] | Δ e_fp | Δ e_fn |
|---|---|---|---:|---:|---|---|---|---:|---:|
| confirmacao | claude-haiku-5-5 | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | claude-haiku-5-5 | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | claude-haiku-5-5 | grey_mid | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | claude-haiku-5-5 | grey_high | 281 | 10 | 0.0000 [0.0000, 0.0140] | 0.3000 [0.1078, 0.6032] | 0.3630 [0.3090, 0.4207] | 0.0400 | 0.2200 |
| confirmacao | claude-haiku-5-5 | near_high | 96 | 34 | 0.0000 [0.0000, 0.0583] | 0.0588 [0.0163, 0.1909] | 0.6354 [0.5357, 0.7248] | 0.0200 | 0.0012 |
| confirmacao | claude-haiku-5-5 | high | 58 | 57 | 1.0000 [0.2065, 1.0000] | 0.0000 [0.0000, 0.0631] | 0.1379 [0.0716, 0.2493] | 0.9900 | 0.0300 |
| confirmacao | deepseek-v4.1-flash | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | deepseek-v4.1-flash | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | deepseek-v4.1-flash | grey_mid | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | deepseek-v4.1-flash | grey_high | 281 | 10 | 0.0000 [0.0000, 0.0140] | 0.5000 [0.2366, 0.7634] | 0.0463 [0.0272, 0.0775] | 0.0400 | 0.4200 |
| confirmacao | deepseek-v4.1-flash | near_high | 96 | 34 | 0.0161 [0.0029, 0.0859] | 0.0882 [0.0305, 0.2296] | 0.5833 [0.4834, 0.6769] | 0.0039 | 0.0282 |
| confirmacao | deepseek-v4.1-flash | high | 58 | 57 | 1.0000 [0.2065, 1.0000] | 0.0000 [0.0000, 0.0631] | 0.0000 [0.0000, 0.0621] | 0.9900 | 0.0300 |
| confirmacao | kimi-k2.5 | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k2.5 | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k2.5 | grey_mid | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k2.5 | grey_high | 281 | 10 | 0.0000 [0.0000, 0.0140] | 0.4000 [0.1682, 0.6873] | 0.0000 [0.0000, 0.0135] | 0.0400 | 0.3200 |
| confirmacao | kimi-k2.5 | near_high | 96 | 34 | 0.0323 [0.0089, 0.1102] | 0.1765 [0.0835, 0.3351] | 0.0000 [0.0000, 0.0385] | 0.0123 | 0.1165 |
| confirmacao | kimi-k2.5 | high | 58 | 57 | 1.0000 [0.2065, 1.0000] | 0.0000 [0.0000, 0.0631] | 0.0000 [0.0000, 0.0621] | 0.9900 | 0.0300 |
| confirmacao | kimi-k3 (parcial) | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k3 (parcial) | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k3 (parcial) | grey_mid | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | kimi-k3 (parcial) | grey_high | 215 | 10 | 0.0098 [0.0027, 0.0349] | 0.5000 [0.2366, 0.7634] | 0.0140 [0.0048, 0.0402] | 0.0302 | 0.4200 |
| confirmacao | kimi-k3 (parcial) | near_high | 80 | 30 | 0.0000 [0.0000, 0.0713] | 0.3667 [0.2187, 0.5449] | 0.1375 [0.0785, 0.2297] | 0.0200 | 0.3067 |
| confirmacao | kimi-k3 (parcial) | high | 48 | 47 | 1.0000 [0.2065, 1.0000] | 0.0000 [0.0000, 0.0756] | 0.1042 [0.0453, 0.2217] | 0.9900 | 0.0300 |
| vigilancia | claude-haiku-5-5 | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | claude-haiku-5-5 | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | claude-haiku-5-5 | grey_mid | 362 | 5 | 0.0000 [0.0000, 0.0106] | 0.4000 [0.1176, 0.7693] | 0.0138 [0.0059, 0.0319] | 0.0600 | 0.3000 |
| vigilancia | claude-haiku-5-5 | grey_high | 930 | 43 | 0.0000 [0.0000, 0.0043] | 0.1860 [0.0974, 0.3262] | 0.1892 [0.1654, 0.2157] | 0.0400 | 0.1060 |
| vigilancia | claude-haiku-5-5 | near_high | 96 | 34 | 0.0000 [0.0000, 0.0583] | 0.0000 [0.0000, 0.1015] | 0.8125 [0.7230, 0.8780] | 0.0200 | 0.0600 |
| vigilancia | claude-haiku-5-5 | high | 22 | 22 | NA | 0.0000 [0.0000, 0.1487] | 0.1364 [0.0475, 0.3333] | NA | 0.0300 |
| vigilancia | deepseek-v4.1-flash | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | deepseek-v4.1-flash | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | deepseek-v4.1-flash | grey_mid | 362 | 5 | 0.0000 [0.0000, 0.0106] | 1.0000 [0.5655, 1.0000] | 0.0000 [0.0000, 0.0105] | 0.0600 | 0.9000 |
| vigilancia | deepseek-v4.1-flash | grey_high | 930 | 43 | 0.0000 [0.0000, 0.0043] | 0.7907 [0.6479, 0.8858] | 0.0043 [0.0017, 0.0110] | 0.0400 | 0.7107 |
| vigilancia | deepseek-v4.1-flash | near_high | 96 | 34 | 0.0484 [0.0166, 0.1329] | 0.0882 [0.0305, 0.2296] | 0.5833 [0.4834, 0.6769] | 0.0284 | 0.0282 |
| vigilancia | deepseek-v4.1-flash | high | 22 | 22 | NA | 0.0000 [0.0000, 0.1487] | 0.0000 [0.0000, 0.1487] | NA | 0.0300 |
| vigilancia | kimi-k2.5 | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | kimi-k2.5 | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | kimi-k2.5 | grey_mid | 362 | 5 | 0.0000 [0.0000, 0.0106] | 0.4000 [0.1176, 0.7693] | 0.0028 [0.0005, 0.0155] | 0.0600 | 0.3000 |
| vigilancia | kimi-k2.5 | grey_high | 930 | 43 | 0.0000 [0.0000, 0.0043] | 0.3721 [0.2438, 0.5214] | 0.0000 [0.0000, 0.0041] | 0.0400 | 0.2921 |
| vigilancia | kimi-k2.5 | near_high | 96 | 34 | 0.0968 [0.0451, 0.1955] | 0.0294 [0.0052, 0.1492] | 0.4167 [0.3231, 0.5166] | 0.0768 | 0.0306 |
| vigilancia | kimi-k2.5 | high | 22 | 22 | NA | 0.0000 [0.0000, 0.1487] | 0.0000 [0.0000, 0.1487] | NA | 0.0300 |
| vigilancia | pooled | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | pooled | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| vigilancia | pooled | grey_mid | 1086 | 15 | 0.0000 [0.0000, 0.0036] | 0.6000 [0.3575, 0.8018] | 0.0055 [0.0025, 0.0120] | 0.0600 | 0.5000 |
| vigilancia | pooled | grey_high | 2790 | 129 | 0.0000 [0.0000, 0.0014] | 0.4496 [0.3665, 0.5357] | 0.0645 [0.0560, 0.0742] | 0.0400 | 0.3696 |
| vigilancia | pooled | near_high | 288 | 102 | 0.0484 [0.0257, 0.0894] | 0.0392 [0.0154, 0.0965] | 0.6042 [0.5467, 0.6589] | 0.0284 | 0.0208 |
| vigilancia | pooled | high | 66 | 66 | NA | 0.0000 [0.0000, 0.0550] | 0.0455 [0.0156, 0.1253] | NA | 0.0300 |
| confirmacao | pooled | low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | pooled | grey_low | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | pooled | grey_mid | 0 ⚠ | 0 | NA | NA | NA | NA | NA |
| confirmacao | pooled | grey_high | 843 | 30 | 0.0000 [0.0000, 0.0047] | 0.4000 [0.2459, 0.5768] | 0.1364 [0.1149, 0.1612] | 0.0400 | 0.3200 |
| confirmacao | pooled | near_high | 288 | 102 | 0.0161 [0.0055, 0.0463] | 0.1078 [0.0613, 0.1829] | 0.4062 [0.3511, 0.4638] | 0.0039 | 0.0478 |
| confirmacao | pooled | high | 174 | 171 | 1.0000 [0.4385, 1.0000] | 0.0000 [0.0000, 0.0220] | 0.0460 [0.0235, 0.0881] | 0.9900 | 0.0300 |

## Contradições MATCH ↔ NONMATCH: Kimi K2.5 × DeepSeek

| Modo | Banda | Contradições | Vínculos | Kimi MATCH / DS NONMATCH | Kimi NONMATCH / DS MATCH |
|---|---|---:|---:|---:|---:|
| vigilancia | low | 0 | 0 | 0 | 0 |
| vigilancia | grey_low | 0 | 0 | 0 | 0 |
| vigilancia | grey_mid | 2 | 2 | 2 | 0 |
| vigilancia | grey_high | 22 | 22 | 20 | 2 |
| vigilancia | near_high | 0 | 0 | 0 | 0 |
| vigilancia | high | 0 | 0 | 0 | 0 |
| confirmacao | low | 0 | 0 | 0 | 0 |
| confirmacao | grey_low | 0 | 0 | 0 | 0 |
| confirmacao | grey_mid | 0 | 0 | 0 | 0 |
| confirmacao | grey_high | 3 | 3 | 3 | 0 |
| confirmacao | near_high | 0 | 0 | 0 | 0 |
| confirmacao | high | 0 | 0 | 0 | 0 |

vigilancia: 24 contradições; 24 vínculos verdadeiros.

confirmacao: 3 contradições; 3 vínculos verdadeiros.

## Uso pelo motor (sem alteração do config principal)

A chave atual é `llm_review.reliability.error_rates_by_band`, um mapping `banda: {e_fp: float, e_fn: float}`; o motor utiliza esses dois números no custo esperado. Não há suporte implícito a caminho de arquivo nessa chave. Para adotar, carregar `gzcmd/error_rates_by_band_estimadas.yaml` externamente e atribuir o mapping de `by_mode.<modo>.models.<modelo>.error_rates_by_band` ou `by_mode.<modo>.pooled.error_rates_by_band` à chave atual antes de construir a política. O bloco `llm_review.reliability.error_rates_by_band` do arquivo novo fornece um default pooled dos modos completos; não substitua o mapping por uma string de caminho. Os blocos `models` agregam modos por modelo; `pooled` agrega somente modelos completos.

Para denominador zero, somente os mappings destinados ao motor retêm o valor fixo do config (`fallback_fixed` identifica cada campo); estimativas/CSV mantêm NA e IC ausente. Isso evita vender falta de evidência como taxa zero. Bandas vazias não foram calibradas. Estes mappings de confiabilidade fornecem somente e_fp/e_fn; u é diagnóstico e não é embutido nessas duas taxas nem configura o custo humano de UNSURE.

Artefatos persistidos são exclusivamente agregados, sem identificadores por par, evidências ou PII.
