# Auditoria A1/A3/A5/A6

Execução offline, sem alterar a política publicada nem as taxas vigentes:

```powershell
python scripts/auditar_funil_adversarial.py
python scripts/fig_cascata_v31.py
python -m pytest tests -q
```

O script reestima as taxas M3 por modo removendo os identificadores antigos
reutilizados pela cascata v3.1. Antes disso, reproduz numericamente todas as
taxas vigentes a partir das respostas antigas, incluindo o fallback fixo em
bandas com menos de dez respostas ou denominador zero. Os suplementos não
entram na reestimação. O replay utiliza somente respostas já existentes;
se a política alternativa selecionar pares sem resposta, reporta projeções
e limites, sem imputar desempenho observado. O funil observado original não
é sobrescrito. As bandas e o universo mantêm a calibração OOF existente.

As métricas dos suplementos excluem UNSURE dos denominadores resolvidos e
também registram a sensibilidade com todos os vínculos no denominador.
A expectativa que substitui respostas reutilizadas mantém os suplementos
observados. Trata-se de análise de sensibilidade, não de intervalo de confiança
nem de validação independente de toda seleção adaptativa da política.
A variante é uma exclusão única dos reutilizados originais, não um procedimento
iterativo nem validação cruzada. Novos selecionados podem estar na coorte
restante que estimou taxas; `llm_ainda_nas_taxas` torna explícita essa sobreposição
residual para evitar interpretar o leave-out como avaliação independente.

A reconciliação distingue a cascata pré-lote da pós-lote: a primeira observa
somente os reutilizados; a segunda observa também os suplementos. A varredura
usa expectativa por banda em todos os pares LLM. O CSV discrimina reutilização,
orçamento, atualização das respostas e arredondamento de apresentação.
O custo exato de 1/240 é um contrafactual separado, pois ambas as saídas
publicadas empregam 0,005. A identidade aditiva é verificada numericamente.

Os intervalos de Wilson são descritivos e não ajustados por dependência entre
pares nem seleção adaptativa. A geração do CSV final incorpora as colunas de
IC para evitar perdê-las em regenerações. As tabelas são atualizadas de forma
idempotente, preservando as demais notas editoriais existentes.

As figuras mantêm cores para erro humano, linha tracejada exclusivamente
para sensibilidade automática, traço-ponto para custo 0,05 e pontilhado para
custo 0,25. Há uma legenda conjunta. Os formatadores usam vírgula decimal.
A cascata mostra apenas contagens e vínculos, sem custo ou sensibilidade
humana: a legenda do capítulo deve deixar de citar esses indicadores.
Nenhum capítulo foi editado por esta auditoria.

`tab_score_bands` é gerada a partir de `nota final`, não de `SCORE` do
OpenRecLink. A evidência reproduz as contagens por banda e os recortes do
funil. Retira-se a explicação de razões de verossimilhança dessa tabela,
pois descrevia outra variável. Os agregados e hashes estão em
`data/funil_gzcmd_adversarial_metadata.json`; os resultados completos e a
frase proposta para o capítulo estão em `data/funil_gzcmd_adversarial_relatorio.md`.
