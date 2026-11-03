# Exp2: Claude Sonnet 5.5, dossiê sem model_outputs, prompt v4

Modelo claude-sonnet-5-5 (API Anthropic), mesmo protocolo Agent-A + Agent-B + árbitro, mesmo dossiê sem `model_outputs`, mesmo prompt v4 e mesmas coortes históricas do Haiku v4. Configuração idêntica: sem temperature, `max_tokens` 2500, sem extended thinking, 4 workers. Executor: `scripts/run_exp2_sonnet_sem_mo.py` (reutiliza `run_exp2_sem_model_outputs.py`, só troca a tabela de preço). Avaliação: `scripts/avaliar_sonnet_v4.py`.

## vigilancia

Respostas do Sonnet: 1410 de 1410 (completo).

| Braço | N | VP | FP | FN | VN | UNSURE | % UNSURE | Sens. sem U (IC95) | Esp. sem U (IC95) | Acerto estrito |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| sonnet-5.5 | 1410 | 82 | 52 | 12 | 1100 | 164 | 11,6% | 87,2% [79,0%; 92,5%] | 95,5% [94,1%; 96,5%] | 83,8% |
| haiku-5.5 (mesmos pares) | 1410 | 73 | 34 | 19 | 1223 | 61 | 4,3% | 79,3% [70,0%; 86,4%] | 97,3% [96,2%; 98,1%] | 91,9% |
| haiku-5.5 (coorte completa) | 1410 | 73 | 34 | 19 | 1223 | 61 | 4,3% | 79,3% [70,0%; 86,4%] | 97,3% [96,2%; 98,1%] | 91,9% |

McNemar exato (acerto estrito, UNSURE conta como não acerto), 1410 pares em comum: Sonnet acerta e Haiku erra = 23; Haiku acerta e Sonnet erra = 137; p < 0,0001.

Discordância por banda de nota:

| Banda | N | Sonnet certo/Haiku errado | Haiku certo/Sonnet errado | Ambos certos | Ambos errados | N da coorte na banda |
|---|---:|---:|---:|---:|---:|---:|
| low | 0 | 0 | 0 | 0 | 0 | 0 |
| grey_low | 0 | 0 | 0 | 0 | 0 | 0 |
| grey_mid | 362 | 4 | 28 | 317 | 13 | 362 |
| grey_high | 930 | 12 | 101 | 759 | 58 | 930 |
| near_high | 96 | 4 | 8 | 69 | 15 | 96 |
| high | 22 | 3 | 0 | 14 | 5 | 22 |

Taxas de erro do Sonnet por banda (IC95 de Wilson; e_FP = FP/não vínculos, e_FN = FN/vínculos, u = UNSURE/N):

| Banda | N | vínculos | e_FP | e_FN | u |
|---|---:|---:|---|---|---|
| low | 0 | 0 | NA (FP 0) | NA (FN 0) | NA (U 0) |
| grey_low | 0 | 0 | NA (FP 0) | NA (FN 0) | NA (U 0) |
| grey_mid | 362 | 5 | 2,5% [1,3%; 4,7%] (FP 9) | 0,0% [0,0%; 43,4%] (FN 0) | 8,8% [6,3%; 12,2%] (U 32) |
| grey_high | 930 | 43 | 4,5% [3,3%; 6,1%] (FP 40) | 2,3% [0,4%; 12,1%] (FN 1) | 12,7% [10,7%; 15,0%] (U 118) |
| near_high | 96 | 34 | 4,8% [1,7%; 13,3%] (FP 3) | 20,6% [10,3%; 36,8%] (FN 7) | 13,5% [8,1%; 21,8%] (U 13) |
| high | 22 | 22 | NA (FP 0) | 18,2% [7,3%; 38,5%] (FN 4) | 4,5% [0,8%; 21,8%] (U 1) |

Latência mediana por par: 17,2 s; tokens por par (mediana): 8.476; fração decidida por árbitro: 19,4%; chamadas: 3.102; tokens de entrada: 10.054.284; saída: 3.182.909; erros de API: 0; custo estimado: US$ 51,94.

## confirmacao

Respostas do Sonnet: 435 de 435 (completo).

| Braço | N | VP | FP | FN | VN | UNSURE | % UNSURE | Sens. sem U (IC95) | Esp. sem U (IC95) | Acerto estrito |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| sonnet-5.5 | 435 | 73 | 6 | 16 | 284 | 56 | 12,9% | 82,0% [72,8%; 88,6%] | 97,9% [95,6%; 99,0%] | 82,1% |
| haiku-5.5 (mesmos pares) | 435 | 64 | 6 | 20 | 320 | 25 | 5,7% | 76,2% [66,1%; 84,0%] | 98,2% [96,0%; 99,2%] | 88,3% |
| haiku-5.5 (coorte completa) | 435 | 64 | 6 | 20 | 320 | 25 | 5,7% | 76,2% [66,1%; 84,0%] | 98,2% [96,0%; 99,2%] | 88,3% |

McNemar exato (acerto estrito, UNSURE conta como não acerto), 435 pares em comum: Sonnet acerta e Haiku erra = 14; Haiku acerta e Sonnet erra = 41; p = 0,0004.

Discordância por banda de nota:

| Banda | N | Sonnet certo/Haiku errado | Haiku certo/Sonnet errado | Ambos certos | Ambos errados | N da coorte na banda |
|---|---:|---:|---:|---:|---:|---:|
| low | 0 | 0 | 0 | 0 | 0 | 0 |
| grey_low | 0 | 0 | 0 | 0 | 0 | 0 |
| grey_mid | 0 | 0 | 0 | 0 | 0 | 0 |
| grey_high | 281 | 7 | 33 | 234 | 7 | 281 |
| near_high | 96 | 2 | 8 | 70 | 16 | 96 |
| high | 58 | 5 | 0 | 39 | 14 | 58 |

Taxas de erro do Sonnet por banda (IC95 de Wilson; e_FP = FP/não vínculos, e_FN = FN/vínculos, u = UNSURE/N):

| Banda | N | vínculos | e_FP | e_FN | u |
|---|---:|---:|---|---|---|
| low | 0 | 0 | NA (FP 0) | NA (FN 0) | NA (U 0) |
| grey_low | 0 | 0 | NA (FP 0) | NA (FN 0) | NA (U 0) |
| grey_mid | 0 | 0 | NA (FP 0) | NA (FN 0) | NA (U 0) |
| grey_high | 281 | 10 | 1,1% [0,4%; 3,2%] (FP 3) | 10,0% [1,8%; 40,4%] (FN 1) | 12,8% [9,4%; 17,2%] (U 36) |
| near_high | 96 | 34 | 4,8% [1,7%; 13,3%] (FP 3) | 20,6% [10,3%; 36,8%] (FN 7) | 14,6% [8,9%; 23,0%] (U 14) |
| high | 58 | 57 | 0,0% [0,0%; 79,3%] (FP 0) | 14,0% [7,3%; 25,3%] (FN 8) | 10,3% [4,8%; 20,8%] (U 6) |

Latência mediana por par: 17,6 s; tokens por par (mediana): 8.546; fração decidida por árbitro: 17,0%; chamadas: 944; tokens de entrada: 3.054.115; saída: 974.715; erros de API: 0; custo estimado: US$ 15,86.

## Custo

Estimativa tarifária (não é fatura): Claude Sonnet 5.5 a US$ 2 por milhão de tokens de entrada e US$ 10 por milhão de saída (https://platform.claude.com/docs/en/about-claude/pricing, consultado em 2026-11-03). O Sonnet 5.5 custa 20 vezes o Haiku 5.5 por token e gerou mais tokens de saída por chamada.

| Execução | Modo | Pares | Chamadas | Tokens de entrada | Tokens de saída | US$ |
|---|---|---:|---:|---:|---:|---:|
| coorte histórica | vigilancia | 1410 | 3.102 | 10.054.284 | 3.182.909 | 51,94 |
| coorte histórica | confirmacao | 435 | 944 | 3.054.115 | 974.715 | 15,86 |
| lote v3.1 REFREC | vigilancia | 155 | 313 | 986.866 | 274.466 | 4,72 |
| lote v3.1 REFREC | confirmacao | 17 | 35 | 111.293 | 34.293 | 0,57 |
| complemento do lote v3.1 (pares do lote em que o Haiku usou a execução por linha) | vigilancia | 100 | 208 | 662.773 | 180.669 | 3,13 |
| complemento do lote v3.1 (pares do lote em que o Haiku usou a execução por linha) | confirmacao | 3 | 6 | 18.860 | 7.282 | 0,11 |
| **Total** | | | | | | **76,32** |

O primeiro teto (US$ 40) interrompeu a vigilância em 651 pares; com autorização, foi retomada do checkpoint sem repetir pares e concluída (teto acumulado de US$ 80, não atingido). Houve um erro HTTP 529 recuperado por retry; nenhum par falhou.

## Cascata v3.1 com Sonnet como revisor

`scripts/funil_gzcmd_v2_sonnet.py` reexecuta o pipeline v3 e v3.1 sobre o extrato inteiro (61.696 pares, 247 vínculos) e troca apenas as respostas do revisor LLM: Sonnet 5.5 v4 contra Haiku 5.5 v4, ambos sem `model_outputs`. A seleção do lote LLM usa as taxas de erro por banda do Haiku, sem reestimar para o Sonnet (por isso o lote é o mesmo). Execução offline, sem chamadas LLM. Saídas: `data/funil_sonnet_v4_cascata.csv` e `funil_sonnet_v4_cobertura.json`.

Cobertura do lote v3.1 pelas respostas do Sonnet: vigilancia: 347 de 347 (coorte histórica 1410, REFREC e complemento cobrem o restante); confirmacao: 44 de 44 (coorte histórica 435, REFREC e complemento cobrem o restante). A lista de pendentes REFREC cobria só os pares sem resposta do Haiku na execução REFREC; os pares do lote v3.1 que o Haiku tinha respondido na execução por linha (100 na vigilância, 3 na confirmação) não tinham resposta do Sonnet e foram executados à parte (complemento).

Definições: VP e FP são decisões MATCH aceitas automaticamente (motor ou LLM); perdidos são vínculos finalizados como NONMATCH; humano soma UNSURE do LLM e encaminhamento direto do motor; sens. automática = VP/vínculos; sens. com humano = (VP + 0,95 × vínculos ao humano)/vínculos, hipótese de erro humano de 5%, sem decisões humanas observadas. IC95 de Wilson para a sensibilidade automática.

### Versão v3.1

| Modo | Revisor | VP | FP | Perdidos | Humano (pares / vínculos) | Sens. automática (IC95) | Sens. com humano | LLM: MATCH / NONMATCH / UNSURE |
|---|---|---:|---:|---:|---|---|---:|---|
| vigilancia | Claude Sonnet 5.5 | 196 | 35 | 12 | 134 / 39 | 79,4% [73,9%; 83,9%] | 94,4% | 36 / 257 / 54 |
| vigilancia | Claude Haiku 5.5 | 194 | 34 | 15 | 111 / 38 | 78,5% [73,0%; 83,2%] | 93,2% | 33 / 283 / 31 |
| confirmacao | Claude Sonnet 5.5 | 136 | 0 | 27 | 129 / 84 | 55,1% [48,8%; 61,1%] | 87,4% | 18 / 18 / 8 |
| confirmacao | Claude Haiku 5.5 | 132 | 0 | 28 | 127 / 87 | 53,4% [47,2%; 59,6%] | 86,9% | 14 / 24 / 6 |

### Versão v3

| Modo | Revisor | VP | FP | Perdidos | Humano (pares / vínculos) | Sens. automática (IC95) | Sens. com humano | LLM: MATCH / NONMATCH / UNSURE |
|---|---|---:|---:|---:|---|---|---:|---|
| vigilancia | Claude Sonnet 5.5 | 153 | 53 | 84 | 164 / 10 | 61,9% [55,7%; 67,8%] | 65,8% | 134 / 1112 / 164 |
| vigilancia | Claude Haiku 5.5 | 144 | 35 | 91 | 61 / 12 | 58,3% [52,1%; 64,3%] | 62,9% | 107 / 1242 / 61 |
| confirmacao | Claude Sonnet 5.5 | 109 | 6 | 126 | 56 / 12 | 44,1% [38,1%; 50,4%] | 48,7% | 79 / 300 / 56 |
| confirmacao | Claude Haiku 5.5 | 100 | 6 | 130 | 25 / 17 | 40,5% [34,6%; 46,7%] | 47,0% | 70 / 340 / 25 |

## Ressalvas

- Execução única por revisor, sem ajuste de multiplicidade; UNSURE conta como não acerto no McNemar.
- O Sonnet usa mais UNSURE que o Haiku: isso reduz perdidos e eleva a sensibilidade, mas desloca pares para o humano.
- Os conjuntos de vínculos do lote LLM são pequenos (28 na vigilância e 26 na confirmação no lote v3.1); diferenças de poucos pares não são conclusivas.
- A seleção do lote usa taxas estimadas com o Haiku; um lote selecionado com taxas do Sonnet poderia diferir.
- Extrato anonimizado; nenhum dado por par nos artefatos.
