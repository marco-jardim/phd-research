# Revisão PR7: OOF por grupo e persistência

O padrão passou a ser `refrec` (ver `M_refrec_fechamento.md`); `row` e `comprec` ficam como análises de sensibilidade. `fit_xgb_oof` aceita `split_by` como
`row`, `refrec` ou `comprec`; `run_v3` honra `evaluation.split_by` quando não
recebe uma escolha explícita. O fingerprint do cache inclui a escolha do
split, além de hashes do código e da fonte e versões dos pacotes.

Nos modos agrupados, StratifiedGroupKFold é usado tanto nos cinco folds externos
quanto nos cinco internos, com seed 42 e seeds internos 43 a 47. As fronteiras
de treino/predição são verificadas por grupo. Os grupos não são preditores
nem colunas persistidas. Identificadores ausentes são recusados, sem converter
ausência em um grupo artificial. Cabeçalhos OpenRecLink são normalizados por
`split(',')[0]` antes de resolver os grupos. O comparador Platt histórico
permanece identificado e não substitui as predições OOF nas métricas novas.

```powershell
python scripts/avaliar_oof_grupos.py
python scripts/avaliar_bandas_sem_dado.py
python scripts/auditar_privacidade_pr7.py
python scripts/retirar_arquivos_m3v3.py
python -m pytest tests -q
```

O primeiro comando compara métricas no total e na área cinza e reexecuta a
política de custo com probabilidades por linha e REFREC, sem substituir as
tabelas e figuras publicadas. A cascata completa em agregados discrimina
as respostas LLM ainda ausentes. Os limites automáticos de pendências não
transformam encaminhamentos humanos em aceitação automática.

O segundo comando testa taxas das bandas sem respostas, separando a
substituição de e_fp/e_fn daquela que inclui também UNSURE. Não pressupõe
invariância da seleção: se houver mudanças, elas são reportadas.

O saneamento de privacidade remove rótulos dos CSVs novos solicitados, mas
conserva os agregados. Os geradores não voltam a gravá-los. Arquivos legados
dos PRs anteriores são apenas inventariados, com hashes, para decisão do
usuário. Os snapshots `_m3v3` mantêm todos os bytes locais; são retirados
somente do índice git após confirmar ausência de citações em `.tex`.

O journal do executor sem saídas do motor recebe `identity.json` antes de
recuperar resultados ou iniciar chamadas. Sem checkpoint, resultados órfãos
sem identidade são recusados. Identidade divergente, incluindo prompt,
modelo e coorte, é recusada. Um journal antigo sem identidade só pode ser
migrado quando há checkpoint completo ou parcial cuja identidade já foi
validada. A escrita usa os retries e alternativos recuperáveis existentes.

Os quatro executores históricos e a reconstrução do dossiê da ablação passam
`p_cal="fit_platt", policy_version="v3"` explicitamente. A regressão verifica
os lotes 1.410/435, a igualdade dos identificadores com a coorte histórica
e quatro guardas ALWAYS_MATCH por modo, sem chamadas LLM.

O teste de taxas foi movido de `scripts/` para `tests/`. Os testes de
persistência usam o `tmp_path` do pytest, não diretórios temporários na raiz
do repositório. O monitor de deriva usa seed 42, restaurando o estado aleatório
após cada teste. As três execuções completas e a análise de materialidade
estão documentadas em `data/funil_gzcmd_oof_grupo_relatorio.md`.
