# Arranjo corte OpenRecLink + GZ-CMD: S1 e S2

## Registro, reprodução e recursos

- Pré-registro inicial enviado antes da primeira S1: `8c16a41`.
- Emenda 1, estabilização de Platt: `83c7842`.
- Emenda 2, classificador original sem sobrescrita de threads:
  `7b40cfb3821b46e0eb402feb44f8e529dba801a3`, enviada antes da S1 concluída.
- SHA-256 do pré-registro vigente:
  `b55899f0e5ab114bf89a00c3e529bd9df14f48b49cc1b10ae7ec7e9e01644601`.
- Conferência autorizada da partição: 211 acima, 61.485 abaixo e 36 PAR=2 abaixo.
- OOF REFREC reproduzido: mesmos folds e zero divergências acima da tolerância
  absoluta de 1e-14 contra `data/pcal_ml_oof.csv`. O teste com máscara integral
  exercitou somente a isotônica, proibindo qualquer ajuste Platt.
- Nenhum fold precisou substituir a isotônica principal por Platt.
- A conferência de RAM por CIM encerrou somente três workers loky órfãos do
  miniconda, após confirmar a ausência dos pais. Uma segunda conferência não
  encontrou outros órfãos. Nenhum outro processo foi encerrado.
- O script registrou 3,88 GiB disponíveis imediatamente antes do treino.
  `_classifier` manteve suas oito threads originais; braços sem paralelismo.

## S1: somente contagens de ações

P, P-Platt, A3 e P-PAR2-grey incluem os 61.485 pares abaixo do corte.
B1 é a ponte na base inteira, com 61.696 pares. Os 211 aceitos pelo corte
OpenRecLink não integram as contagens dos braços restritos abaixo do corte.
As tabelas não avaliam acerto nem o desfecho da cascata contra rótulos.

| Modo | Braço | MATCH | NONMATCH | LLM_REVIEW | DEFERRED |
|---|---|---:|---:|---:|---:|
| Vigilância | P | 0 | 61081 | 344 | 60 |
| Vigilância | P-Platt | 2 | 61316 | 110 | 57 |
| Vigilância | B1 | 195 | 61048 | 374 | 79 |
| Vigilância | A3 | 1 | 61043 | 367 | 74 |
| Vigilância | P-PAR2-grey | 0 | 61127 | 265 | 93 |
| Confirmação | P | 0 | 61428 | 41 | 16 |
| Confirmação | P-Platt | 0 | 61445 | 25 | 15 |
| Confirmação | B1 | 118 | 61385 | 70 | 123 |
| Confirmação | A3 | 1 | 61430 | 32 | 22 |
| Confirmação | P-PAR2-grey | 0 | 61445 | 18 | 22 |

### Contagens por modo, braço, banda e ação

Zeros completam células ausentes do CSV longo, sem acrescentar observações.

| Modo | Braço | Banda | MATCH | NONMATCH | LLM_REVIEW | DEFERRED |
|---|---|---|---:|---:|---:|---:|
| Vigilância | P | low | 0 | 39979 | 0 | 0 |
| Vigilância | P | grey_low | 0 | 14195 | 30 | 30 |
| Vigilância | P | grey_mid | 0 | 5965 | 162 | 0 |
| Vigilância | P | grey_high | 0 | 909 | 132 | 1 |
| Vigilância | P | near_high | 0 | 33 | 18 | 22 |
| Vigilância | P | high | 0 | 0 | 2 | 7 |
| Vigilância | P-Platt | low | 0 | 39979 | 0 | 0 |
| Vigilância | P-Platt | grey_low | 0 | 14221 | 6 | 28 |
| Vigilância | P-Platt | grey_mid | 0 | 6076 | 51 | 0 |
| Vigilância | P-Platt | grey_high | 2 | 1004 | 35 | 1 |
| Vigilância | P-Platt | near_high | 0 | 36 | 15 | 22 |
| Vigilância | P-Platt | high | 0 | 0 | 3 | 6 |
| Vigilância | B1 | low | 0 | 39981 | 0 | 0 |
| Vigilância | B1 | grey_low | 12 | 14187 | 26 | 38 |
| Vigilância | B1 | grey_mid | 38 | 5953 | 184 | 0 |
| Vigilância | B1 | grey_high | 29 | 902 | 140 | 9 |
| Vigilância | B1 | near_high | 35 | 25 | 24 | 18 |
| Vigilância | B1 | high | 81 | 0 | 0 | 14 |
| Vigilância | A3 | low | 0 | 39979 | 0 | 0 |
| Vigilância | A3 | grey_low | 1 | 14173 | 25 | 56 |
| Vigilância | A3 | grey_mid | 0 | 5941 | 186 | 0 |
| Vigilância | A3 | grey_high | 0 | 913 | 129 | 0 |
| Vigilância | A3 | near_high | 0 | 37 | 22 | 14 |
| Vigilância | A3 | high | 0 | 0 | 5 | 4 |
| Vigilância | P-PAR2-grey | low | 0 | 39979 | 0 | 0 |
| Vigilância | P-PAR2-grey | grey_low | 0 | 14195 | 31 | 29 |
| Vigilância | P-PAR2-grey | grey_mid | 0 | 5993 | 113 | 21 |
| Vigilância | P-PAR2-grey | grey_high | 0 | 927 | 101 | 14 |
| Vigilância | P-PAR2-grey | near_high | 0 | 33 | 18 | 22 |
| Vigilância | P-PAR2-grey | high | 0 | 0 | 2 | 7 |
| Confirmação | P | low | 0 | 39979 | 0 | 0 |
| Confirmação | P | grey_low | 0 | 14249 | 0 | 6 |
| Confirmação | P | grey_mid | 0 | 6125 | 0 | 2 |
| Confirmação | P | grey_high | 0 | 1015 | 27 | 0 |
| Confirmação | P | near_high | 0 | 60 | 5 | 8 |
| Confirmação | P | high | 0 | 0 | 9 | 0 |
| Confirmação | P-Platt | low | 0 | 39979 | 0 | 0 |
| Confirmação | P-Platt | grey_low | 0 | 14249 | 0 | 6 |
| Confirmação | P-Platt | grey_mid | 0 | 6126 | 0 | 1 |
| Confirmação | P-Platt | grey_high | 0 | 1031 | 11 | 0 |
| Confirmação | P-Platt | near_high | 0 | 60 | 5 | 8 |
| Confirmação | P-Platt | high | 0 | 0 | 9 | 0 |
| Confirmação | B1 | low | 0 | 39981 | 0 | 0 |
| Confirmação | B1 | grey_low | 2 | 14227 | 0 | 34 |
| Confirmação | B1 | grey_mid | 20 | 6126 | 0 | 29 |
| Confirmação | B1 | grey_high | 21 | 1010 | 41 | 8 |
| Confirmação | B1 | near_high | 12 | 41 | 15 | 34 |
| Confirmação | B1 | high | 63 | 0 | 14 | 18 |
| Confirmação | A3 | low | 0 | 39979 | 0 | 0 |
| Confirmação | A3 | grey_low | 1 | 14239 | 0 | 15 |
| Confirmação | A3 | grey_mid | 0 | 6126 | 0 | 1 |
| Confirmação | A3 | grey_high | 0 | 1020 | 22 | 0 |
| Confirmação | A3 | near_high | 0 | 63 | 4 | 6 |
| Confirmação | A3 | high | 0 | 3 | 6 | 0 |
| Confirmação | P-PAR2-grey | low | 0 | 39979 | 0 | 0 |
| Confirmação | P-PAR2-grey | grey_low | 0 | 14249 | 0 | 6 |
| Confirmação | P-PAR2-grey | grey_mid | 0 | 6125 | 0 | 2 |
| Confirmação | P-PAR2-grey | grey_high | 0 | 1032 | 7 | 3 |
| Confirmação | P-PAR2-grey | near_high | 0 | 60 | 2 | 11 |
| Confirmação | P-PAR2-grey | high | 0 | 0 | 9 | 0 |

Fonte: `arranjo_corte_s1_resumo.csv`. Taxas de ajuste:
`arranjo_corte_taxas_crossfit.csv`. Proveniência e fronteiras de ajuste:
`arranjo_corte_s1_metadata.json`.

## Regra de parada e S2

A união P+B1 por modo encontrou dez pendentes na vigilância e 14 na
confirmação, totalizando 24. A estimativa prévia foi US$ 0,07593696,
incluindo a margem de 20% para tentativas. Nem o limite de 500 pares nem
o limite de US$ 2 foi atingido. Sensibilidades não geraram chamadas adicionais.

S2 usou Anthropic, modelo `claude-haiku-5-5`, prompt v4, dossiê sem
`model_outputs`, `--allow-new-cohort` e quatro workers, com modos sequenciais.
Cada processo foi iniciado com `Start-Process` e aguardado com `Wait-Process`.

| Modo | Pares concluídos | Chamadas API | Tokens de entrada | Tokens de saída | Custo pelos tokens observados, US$ |
|---|---:|---:|---:|---:|---:|
| Vigilância | 10 | 22 | 70898 | 42058 | 0,0281188 |
| Confirmação | 14 | 30 | 96482 | 50256 | 0,0347762 |
| Total | 24 | 52 | 167380 | 92314 | 0,0628950 |

Os dois runs terminaram com `complete: true`, zero erros por par, zero erros
HTTP e zero erros de parsing. A auditoria verificou 59 arquivos e encontrou
zero ocorrências de segredos. O custo usa os tokens efetivamente registrados
e as tarifas históricas do runner; é estimativa tarifária, não valor de fatura.

As taxas permaneceram congeladas antes e depois da S2, com SHA-256:
`d0876836394b2513135995df3fc6e21621a2e162806bd8dd88b6fe54fb931172`.
Nenhuma resposta nova foi usada para reestimar a confiabilidade do revisor.

Metadados agregados e auditorias estão em
`sprint3b/anthropic_sem_mo_v4_arranjo_corte/`. Resultados por par, journals e
listas de pendentes permanecem locais e não são versionados. Nenhum dossiê
ou resposta livre foi salvo. S3 e S4 não foram executadas.

## Verificação

```text
python -m pytest tests/test_arranjo_corte.py -q
18 passed in 29.47s

python -m pytest tests -q
338 passed, 1 skipped, 2 warnings in 51.61s
```

Os dois arquivos ignorados exigidos pela suíte histórica foram copiados do
clone para o worktree sem versionamento. Os dois warnings vêm dos testes
de modelagem log-linear de blocagem, não do arranjo corte.
