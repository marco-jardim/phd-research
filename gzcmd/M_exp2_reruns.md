# Avaliação consolidada dos reruns do Exp2

Executar `python scripts/avaliar_reruns_exp2.py`. Não há chamadas de rede ou de LLM.
O manifesto explícito usa os arquivos canônicos de checkpoint, sem incluir backups.
A confirmação de Kimi K2.5 usa QW2 somente na ausência da fonte sprint3b, com
proveniência e SHA256 do conteúdo lido. A coorte original define os pares esperados
por modo; respostas duplicadas, estranhas à coorte ou sem gabarito são rejeitadas.

## Métricas

`VP/FP/FN/VN` contam apenas decisões definitivas. `UNSURE_pos/neg` preservam as
abstenções por classe. `sens/espec` excluem abstenções de seus denominadores;
`sens_unsure_nonmatch/espec_unsure_nonmatch` tratam todas como NONMATCH.
`sens_incl_unsure=(VP+UNSURE_pos)/(VP+FN+UNSURE_pos)` mede vínculos reconhecidos
ou encaminhados ao revisor humano, não acerto humano observado. Todos esses
indicadores, a fração de abstenção e o acerto estrito têm IC de 95% de Wilson.
Denominador zero permanece ausente no CSV. Sem resposta não equivale a UNSURE.

Comparações usam a interseção de pares de cada par de condições, com contagem
dos exclusivos. A direção é completo/v3 para sem MO/v3, completo/v3 para sem MO/v4
e sem MO/v3 para sem MO/v4. `FN_revertidos` exige NONMATCH para MATCH em vínculo;
`FN_para_UNSURE` é reportado separadamente. FP novos e UNSURE novos são contados
relativamente à primeira condição. McNemar exato bilateral compara acerto estrito,
com UNSURE incorreto em ambas as classes; não há ajuste de multiplicidade. Kappa
usa as três categorias e fica ausente quando o denominador é zero.

## Artefatos e parcialidade

Os CSVs `data/exp2_reruns_{resumo,pareado,por_banda}.csv` são exclusivamente
agregados. Cada fonte inclui `completo`, n respondido/esperado/sem resposta e hash.
Bandas seguem o YAML principal, incluindo bandas vazias, além dos recortes s<9 e
s>=9. `n` é o denominador local; `n_respondido` é o total do run. A tabela
`text/latex/tables/tab_cap8_reruns.tex` separa os modos e identifica parciais em nota.
`--only-complete` exclui parciais somente dessa tabela, não dos CSVs ou YAML.

`gzcmd/error_rates_by_band_v4.yaml` reutiliza `summarize` e `engine_profile` de
`estimar_taxas_erro_revisor.py`, mantendo o esquema de perfis por modo/modelo e o
mapping `llm_review.reliability.error_rates_by_band` consumido por `--rates-yaml`.
Inclui apenas Haiku e DeepSeek sem MO/v4, inclusive checkpoints, sem modificar o
config histórico. FP e FN são condicionais à classe, com UNSURE nos denominadores
mas fora dos numeradores. Denominadores zero usam fallback fixo explícito somente
nos mappings do motor; as estimativas permanecem ausentes. `provisional` e
`partial` sinalizam execuções incompletas ou ausência de modelo/modo no pooling.
Pares repetidos entre revisores e modos não são observações independentes; IC
pooled é descritivo. Checkpoints não são amostras aleatórias e não permitem
extrapolação aos pares ainda sem resposta. Reexecutar após a conclusão atualiza
todos os agregados, sem alterar os JSONs de origem.
