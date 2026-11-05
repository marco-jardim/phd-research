# Prompt caching: transporte Anthropic para runs novos

## Motivo e escopo

O e-mail da Anthropic alertou sobre baixa taxa de acerto do cache. Segundo o
diagnóstico dos runs anteriores, todos registram cache 0: as requisições enviavam
strings sem `cache_control`, e quatro workers começavam juntos. Esta alteração
não executa inferência nem altera resultados anteriores.

O novo executor é `scripts/run_exp2_cached.py`, somente Anthropic, prompt v4,
dossiês sem saídas do motor e protocolo Agent-A + Agent-B + árbitro preservado.
Use uma pasta nova por modo e execução. A retomada rejeita checkpoints históricos
sem o metadado de cache e não mistura políticas de aquecimento.

```text
python scripts/run_exp2_cached.py --modo vigilancia --pares pares.json --saida-dir data/sprint3b/anthropic_cached_v4/vigilancia --allow-new-cohort
```

Esse comando faz inferência e é apenas documentação: não foi executado nesta
mudança. Sem `--allow-new-cohort`, exige a mesma coorte histórica.
Credenciais vêm exclusivamente do ambiente; o novo caminho não carrega `.env`.

## Estrutura antes e depois

Antes: `system = system_v4 + papel`; `user = template_v4 + "\n\n" + dossiê`
(no árbitro, cabeçalho do dossiê e opiniões A/B seguem o template).

Depois:

```json
{
  "system": [
    {"type": "text", "text": "<system v4 completo extraído pelo loader histórico>", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "\n\n<papel Agent-A, Agent-B ou árbitro>"}
  ],
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "<template fixo v4>", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "\n\n<dossiê e, no árbitro, opiniões na ordem histórica>"}
  ]}]
}
```

Não se adiciona nem remove texto: a concatenação direta dos blocos é idêntica
byte a byte em UTF-8 às strings anteriores, inclusive separadores `\n\n`.
“System completo” significa o trecho system efetivamente extraído, não o arquivo
Markdown inteiro com as instruções de user. Nenhum prompt, dossiê, parser, decisão
ou chave de identidade do protocolo muda. `prompt_sha256` continua sendo o hash
do arquivo v4 inteiro; `dossier_sha256` usa a serialização histórica ordenada.

O primeiro breakpoint permite compartilhar o system entre papéis; o segundo
inclui também o papel e compartilha o template entre pares do mesmo papel.
O cache Anthropic é cumulativo, não um cache isolado de cada bloco. Não se marca
o dossiê variável. TTL padrão: cinco minutos; o mínimo do Haiku 5.5 é 512 tokens.
O prefixo estimado de 1.300–1.700 tokens ultrapassa esse mínimo; não houve medição
via API nesta mudança.

## Aquecimento, uso e custo

`--cache-warmup` vem ligado. O primeiro par pendente termina sozinho (incluindo
Agent-A, Agent-B e árbitro se necessário) antes de criar/liberar o pool. Não há
chamada artificial extra. Falha nesse par impede liberar workers. Em retomadas,
aquece o primeiro par ainda não concluído; runs completos não chamam a API.
`--no-cache-warmup` desliga essa barreira. Árbitro só é aquecido se houver
divergência no primeiro par; não se força uma divergência para preencher cache.

Os contadores existentes `cache_read_input_tokens` e
`cache_creation_input_tokens` são preservados por chamada no journal, agregados
em `transport`, copiados para `metadata` e para `execucao.json` em cada checkpoint.
O JSON registra `prompt_caching: {enabled: true, breakpoints:
["system[0]", "messages[0].content[0]"], warmup: true}` (ou `false` sem aquecimento).
`execucao.json` descreve o último modo da pasta: use pastas distintas por modo.

`gzcmd/llm_usage_cost.py` já calcula entrada, saída, leitura e escrita separadamente:
USD/M 0,10; 0,50; 0,01; 0,125, respectivamente (tier até 100 mil tokens).
A economia esperada é cerca de 10% do custo total, não 90%: a saída responde por
cerca de 74% do custo e não recebe desconto de cache. Acertos, TTL, gravações e
prefixo realmente reutilizado determinam a economia observada. Não é fatura
nem resultado medido; conferir os contadores nos próximos runs aprovados.

## Reprodutibilidade e Fireworks

Antes de editar, foram procurados nomes e SHA256 dos quatro arquivos indicados
nos `prereg_*.yaml`, metadados/manifests de `data/` e literais dos testes.
`scripts/run_exp2_sem_model_outputs.py` e `gzcmd/llm_dossier.py` estão fixados e
permanecem intactos. `scripts/run_exp2_anthropic.py` também foi preservado.
O runner novo deriva do histórico para manter a persistência Windows-safe,
coortes e identidade, sem mudar o arquivo pré-registrado. O caminho histórico
Anthropic continua disponível; runs anteriores e artefatos não foram reescritos.

`gzcmd/llm_review.py` não tem hash fixado nas fontes verificadas. Seu transporte
padrão agora recusa URL/modelo Fireworks antes de qualquer POST com o erro
“Fireworks é proibido neste projeto”. Transportes Anthropic sobrescrevem esse
método; não há fallback para Fireworks. O runner novo aceita somente Anthropic.

## Verificação offline

`tests/test_anthropic_prompt_caching.py` simula
`anthropic.Anthropic().messages.create`, bloqueia POSTs e cobre os três papéis,
bytes das strings históricas, identidade, retomada, contadores, custo, ordem do
executor falso, opção de aquecimento e proibição do Fireworks. Nenhuma chamada
real à API é necessária ou foi autorizada para essa validação.

Fonte: https://platform.claude.com/docs/pt-BR/build-with-claude/prompt-caching
(consultada em 05/11/2026).
