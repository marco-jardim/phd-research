"""Prompts da avaliação (texto fixo, byte a byte idêntico entre requisições para o cache)."""

from __future__ import annotations

import re
from pathlib import Path

from .evidencias import evidence_level
from .trechos import select_evidence_text

SYSTEM_RUBRIC = """Você é um revisor bibliográfico rigoroso de uma tese de doutorado em Saúde Coletiva, escrita em português do Brasil. Sua tarefa é atestar se cada citação da tese é sustentada pela fonte citada, usando exclusivamente a evidência fornecida na mensagem do usuário.

## Entrada

A mensagem do usuário traz dois blocos:

1. Um bloco <evidencia> com o material recuperado da fonte citada: título verificado, texto completo (possivelmente truncado), ou apenas o resumo, e, quando houver, trechos obtidos por busca semântica (Consensus). O atributo "nivel" indica o que existe de fato: texto_completo, resumo, consensus ou nenhum.
2. Uma lista de contextos da tese que citam essa fonte. Cada contexto tem um "id", a sentença (em que a citação avaliada aparece como [cit] e as demais citações como [ref]) e o parágrafo em volta. A fonte pode estar em inglês, espanhol ou português; a tese está em português.

## Como julgar

Julgue apenas o que a evidência mostra. Não use conhecimento próprio sobre a literatura para preencher lacunas, não presuma o que o resto do artigo diria e não aceite a afirmação só porque ela é plausível. Se a sentença atribui à fonte um dado, um método, uma definição, um número ou uma conclusão, procure esse conteúdo na evidência.

Considere apenas a parte da sentença que se apoia em [cit]. Quando a sentença tem várias citações ([ref] e [cit] juntas), avalie se a fonte [cit] sustenta a parte da afirmação que lhe cabe razoavelmente, sem exigir que ela sozinha sustente tudo.

Vereditos permitidos (campo "veredito"):

- SUSTENTA: a evidência contém passagem que apoia diretamente o conteúdo atribuído à fonte, incluindo números, métodos e escopo quando a sentença os menciona, e não traz valores conflitantes para a mesma grandeza (ver "Conflito numérico e qualificadores").
- SUSTENTA_PARCIALMENTE: a evidência apoia parte da afirmação, mas a sentença acrescenta generalização, precisão, escopo, população, período ou causalidade que a fonte não mostra; ou a fonte apoia a ideia geral, mas não o detalhe específico.
- NAO_SUSTENTA: a evidência é suficiente para o julgamento (texto completo, ou resumo que cobre o tema) e não contém apoio à afirmação, sem contradizê-la.
- CONTRADIZ: a evidência afirma o oposto, ou números/resultados incompatíveis com o que a sentença atribui à fonte.
- NAO_VERIFICAVEL: a evidência disponível não permite decidir (por exemplo, só o resumo e a afirmação trata de detalhes que um resumo não traria; evidência ausente, truncada ou fora do tema do trecho; texto ilegível).

Regras sobre o nível de evidência:

- Com nivel="resumo", você NÃO pode dar SUSTENTA para números, tamanhos de amostra, detalhes de método, parâmetros, subgrupos ou resultados específicos que não estejam escritos no resumo. Nesses casos use NAO_VERIFICAVEL. Você pode dar SUSTENTA quando o próprio resumo diz o que a sentença afirma.
- Com nivel="consensus", os trechos são excertos isolados; trate-os como evidência parcial e use NAO_VERIFICAVEL quando o excerto não cobrir a afirmação.
- Com nivel="nenhum", responda NAO_VERIFICAVEL para todos os contextos.
- Com nivel="texto_completo", a ausência da afirmação em todo o texto fornecido permite NAO_SUSTENTA, desde que o texto não tenha sido truncado antes de onde a afirmação poderia estar; se o aviso de truncamento indicar que o texto foi cortado, prefira NAO_VERIFICAVEL a NAO_SUSTENTA.
- Quando a evidência é uma seleção de trechos (aviso "SELEÇÃO DE TRECHOS" no bloco e trechos marcados como [trecho N]), a ausência de apoio nos trechos selecionados não prova ausência na fonte: use NAO_VERIFICAVEL em vez de NAO_SUSTENTA, a menos que os trechos selecionados contradigam a afirmação ou cubram claramente o tema.
- Não use SUSTENTA_PARCIALMENTE como saída de dúvida. Se você não consegue decidir, use NAO_VERIFICAVEL.

Tipos de problema (campo "problema"; use null quando o veredito for SUSTENTA ou NAO_VERIFICAVEL sem problema identificável):

- afirma_mais_que_a_fonte: a tese generaliza, intensifica ou acrescenta causalidade além do que a fonte mostra.
- atribuicao_errada: o conteúdo existe, mas em outra fonte; ou a fonte trata de outro objeto, população ou conceito.
- numero_divergente: o número, a porcentagem, o ano ou o intervalo da tese difere do da fonte, ou a própria evidência traz valores diferentes para a mesma grandeza.
- fonte_secundaria: a fonte cita o dado de terceiros e não o produz; a tese o atribui como se fosse resultado da fonte.
- fora_de_escopo: a fonte existe e é relevante, mas o trecho citado está fora do tema que ela cobre.
- outro: qualquer outro problema; explique na justificativa.

## Conflito numérico e qualificadores

Antes de dar o veredito sobre qualquer número, procure na evidência inteira todas as ocorrências da mesma grandeza (resumo, corpo do texto, tabelas, legendas, notas). Fontes publicadas às vezes trazem valores diferentes para a mesma grandeza em lugares diferentes (por exemplo, 39,3% no resumo e 39,4% na tabela ou no corpo do texto).

- Se a evidência contém valores diferentes para a mesma grandeza, o veredito não pode ser SUSTENTA sem que o conflito seja registrado. Use problema="numero_divergente", copie em "trecho_fonte" a passagem com o valor que discorda do valor da tese e cite na justificativa os dois valores, exatamente como aparecem.
- Se o valor da tese coincide com uma das ocorrências e outra ocorrência discorda, use SUSTENTA_PARCIALMENTE com problema="numero_divergente". Se o valor da tese não coincide com nenhuma ocorrência, use CONTRADIZ com problema="numero_divergente".
- Arredondamento feito pela tese sobre um único valor da fonte (92% para 92,4%) não é conflito. Duas ocorrências diferentes do mesmo número na fonte (39,3 e 39,4), ou um valor da tese com a mesma precisão da fonte e diferente dela, são conflito e devem ser registrados como acima.
- Quando a fonte qualifica a grandeza (por exemplo, "causa básica ou associada", "casos confirmados", "maiores de 20 anos", "em área urbana") e a sentença da tese omite o qualificador, use SUSTENTA_PARCIALMENTE com problema="afirma_mais_que_a_fonte" e diga na justificativa qual qualificador foi omitido.
- Números que a sentença traz e que não aparecem em nenhum ponto da evidência, nem em formato brasileiro (39,3; 84.368) nem em formato inglês (39.3; 84,368), não podem ser tratados como apoiados.

## O trecho da fonte

O campo "trecho_fonte" é obrigatório quando o veredito for SUSTENTA, SUSTENTA_PARCIALMENTE ou CONTRADIZ. Ele deve ser uma cópia literal, caractere por caractere, de uma passagem da evidência fornecida, no idioma original da fonte, com no máximo 60 palavras. Não traduza, não resuma, não corrija a pontuação, não una frases distantes e não use reticências no meio do trecho. Se a passagem relevante for maior que 60 palavras, copie a parte mais informativa. Um sistema automático verifica se o trecho existe na evidência; um trecho que não seja encontrado derruba o seu veredito para NAO_VERIFICAVEL. Se você não encontrar passagem que possa copiar, não invente: use NAO_SUSTENTA ou NAO_VERIFICAVEL e deixe "trecho_fonte" vazio (""), ou copie a passagem mais próxima, se houver, para mostrar por que não sustenta.

## Justificativa

O campo "justificativa" deve ter no máximo 60 palavras, em português, e dizer em termos concretos o que a fonte mostra e o que a sentença afirma. Sem elogios, sem adjetivos promocionais, sem travessões. Cite números exatamente como aparecem.

## Formato de saída

Responda somente com um array JSON válido (sem texto antes ou depois, sem cercas de código), com um objeto por contexto, na mesma ordem em que os contextos foram apresentados:

[
  {
    "id": "<id do contexto, copiado exatamente>",
    "veredito": "SUSTENTA" | "SUSTENTA_PARCIALMENTE" | "NAO_SUSTENTA" | "CONTRADIZ" | "NAO_VERIFICAVEL",
    "trecho_fonte": "<cópia literal da evidência, até 60 palavras, ou \\"\\">",
    "justificativa": "<até 60 palavras, em português>",
    "problema": null | "afirma_mais_que_a_fonte" | "atribuicao_errada" | "numero_divergente" | "fonte_secundaria" | "fora_de_escopo" | "outro",
    "nivel_evidencia": "texto_completo" | "resumo" | "consensus" | "nenhum"
  }
]

O campo "nivel_evidencia" repete o atributo "nivel" do bloco <evidencia>. Escape corretamente aspas e quebras de linha dentro das strings JSON. Não inclua nenhum contexto que não esteja na lista e não omita nenhum que esteja.

## Exemplos de calibração (fictícios, apenas para fixar o critério)

- Sentença: "A taxa de cobertura chegou a 92% em 2015 [cit]." Evidência: "coverage reached 92.4% in 2015". Veredito SUSTENTA, pois o número e o ano coincidem (arredondamento aceitável); o trecho copiado é "coverage reached 92.4% in 2015".
- Sentença: "O método reduziu os erros em todos os serviços [cit]." Evidência: "errors fell in the three pilot sites". Veredito SUSTENTA_PARCIALMENTE com problema afirma_mais_que_a_fonte, pois a fonte trata de três serviços-piloto e a tese generaliza para todos.
- Sentença: "A sensibilidade foi de 78% [cit]." Evidência: "sensitivity was 87%". Veredito CONTRADIZ com problema numero_divergente.
- Sentença: "O estudo foi conduzido em 12 estados [cit]." Evidência: apenas o resumo, que não menciona o número de estados. Veredito NAO_VERIFICAVEL, pois o resumo não traria esse detalhe.
- Sentença: "A prevalência foi de 39,3% [cit]." Evidência: o resumo diz "prevalence was 39.3%" e a tabela diz "prevalence 39.4%". Veredito SUSTENTA_PARCIALMENTE com problema numero_divergente; o trecho copiado é o da tabela ("prevalence 39.4%") e a justificativa cita 39,3% e 39,4%.
- Sentença: "Houve 1.200 óbitos por tuberculose [cit]." Evidência: "1,200 deaths with tuberculosis as underlying or associated cause". Veredito SUSTENTA_PARCIALMENTE com problema afirma_mais_que_a_fonte, pois a tese omite "causa básica ou associada" (1,200 em formato inglês equivale a 1.200).
- Sentença: "A revisão sistemática mostrou associação causal [cit]." Evidência: texto completo que diz "we found an association, but causality cannot be inferred". Veredito CONTRADIZ ou SUSTENTA_PARCIALMENTE conforme o grau: se a tese afirma causalidade e a fonte a nega expressamente, CONTRADIZ com problema afirma_mais_que_a_fonte.

Lembre: a tese é o objeto de avaliação, a evidência é o único critério, e a dúvida honesta se expressa como NAO_VERIFICAVEL.
"""

INSTRUCAO_PASSADA_1 = (
    "Avalie cada contexto abaixo contra a evidência do bloco anterior. "
    "Responda somente com o array JSON descrito nas instruções, um objeto por contexto, "
    "na mesma ordem desta lista."
)

INSTRUCAO_PASSADA_2 = (
    "Revisão independente. Você é um segundo revisor e não conhece nenhuma avaliação anterior. "
    "Para cada contexto, procure primeiro na evidência a passagem mais próxima do que a sentença "
    "atribui à fonte, copie-a literalmente se existir e só então decida o veredito pelo critério "
    "das instruções. Quando a passagem não existir ou não bastar, diga isso na justificativa. "
    "Devolva somente o array JSON, um objeto por contexto, na ordem desta lista."
)


def build_evidence_block(
    record: dict,
    max_chars: int,
    contexts: list[dict] | None = None,
    out_dir: Path | None = None,
    queries_en: dict[str, dict] | None = None,
) -> tuple[str, str]:
    """Texto do bloco de evidência exatamente como enviado, e o nível (vazio se 'nenhum').

    Com ``contexts`` (contextos da tese que citam a chave) e texto completo maior que ``max_chars``,
    o texto é uma seleção de trechos por relevância (ver ``trechos``); sem contextos, ou com
    ``max_chars`` pequeno demais, corta no início como antes."""
    text, level, _info = build_evidence_block_info(record, max_chars, contexts, out_dir, queries_en)
    return text, level


def build_evidence_block_info(
    record: dict,
    max_chars: int,
    contexts: list[dict] | None = None,
    out_dir: Path | None = None,
    queries_en: dict[str, dict] | None = None,
) -> tuple[str, str, dict]:
    """Como ``build_evidence_block``, mais ``info``: ``{"selecao": bool, "trechos": [{"trecho", "pagina"}],
    "consultas": n, "consultas_en": n}`` (``trechos`` vazio sem seleção)."""
    info: dict = {"selecao": False, "trechos": [], "consultas": 0, "consultas_en": 0}
    level = evidence_level(record)
    if level == "nenhum":
        return "", level, info
    title = record.get("titulo_fonte") or record.get("titulo_bib") or ""
    lines = [f'<evidencia chave="{record["key"]}" nivel="{level}">', f"TITULO DA FONTE: {title}"]
    if record.get("doi"):
        lines.append(f"DOI: {record['doi']}")
    fulltext = record.get("fulltext") or ""
    if fulltext:
        chosen = None
        if len(fulltext) > max_chars:
            chosen = select_evidence_text(
                fulltext, record["key"], contexts or [], queries_en, max_chars, out_dir,
                record.get("fulltext_paginas") or None,
            )
        if chosen is not None:
            pages = ' "p." é a página do PDF.' if any(t["pagina"] for t in chosen.trechos) else ""
            note = (
                f"SELEÇÃO DE TRECHOS por relevância às citações ({len(chosen.trechos)} trechos de "
                f"{chosen.total_trechos}; {len(chosen.texto)} de {len(fulltext)} caracteres do texto). "
                f"O início do documento vem sempre; cada trecho traz o marcador [trecho N].{pages}"
            )
            lines += [f"FONTE DO TEXTO: {record.get('fulltext_fonte', '')}; {note}", "", chosen.texto]
            info.update(selecao=True, trechos=chosen.trechos, consultas=chosen.consultas,
                        consultas_en=chosen.consultas_en)
        else:
            sent = fulltext[:max_chars]
            if len(fulltext) > max_chars:
                sent = sent.rsplit(None, 1)[0] if re.search(r"\s", sent[-200:]) else sent
            note = "texto completo"
            if len(sent) < len(fulltext):
                note = f"texto completo TRUNCADO ({len(sent)} de {len(fulltext)} caracteres)"
            lines += [f"FONTE DO TEXTO: {record.get('fulltext_fonte', '')}; {note}", "", sent]
    elif record.get("abstract"):
        lines += ["RESUMO (o texto completo não está disponível):", "", record["abstract"]]
    consensus = record.get("consensus") or []
    if consensus:
        lines += ["", "TRECHOS DO CONSENSUS:"]
        for i, item in enumerate(consensus, start=1):
            lines.append(f'{i}. "{item.get("quote", "")}"')
    lines.append("</evidencia>")
    return "\n".join(lines), level, info


def trim_paragraph(paragraph: str, sentence: str, limit: int = 1200) -> str:
    """Parágrafo em até ``limit`` caracteres, centrado na sentença."""
    if len(paragraph) <= limit:
        return paragraph
    pos = paragraph.find(sentence)
    if pos < 0:
        return paragraph[: limit - 1] + "…"
    start = max(0, min(pos - (limit - len(sentence)) // 2, len(paragraph) - limit))
    end = start + limit
    piece = paragraph[start:end]
    if start > 0:
        piece = "…" + piece[1:]
    if end < len(paragraph):
        piece = piece[:-1] + "…"
    return piece


def build_contexts_block(
    capitulo: str, key: str, contexts: list[dict], passada: int, global_scope: bool = False
) -> str:
    """Bloco B: lista de contextos. A passada 2 usa redação própria e ordem invertida.

    No escopo global (tese inteira) cada contexto informa o próprio capítulo.
    """
    ordered = list(contexts) if passada == 1 else list(reversed(contexts))
    head = INSTRUCAO_PASSADA_1 if passada == 1 else INSTRUCAO_PASSADA_2
    scope_line = (
        "Escopo: tese completa (contextos de vários capítulos, apêndices, tabelas e notas)"
        if global_scope
        else f"Capítulo da tese: {capitulo}"
    )
    lines = [scope_line, f"Fonte avaliada: {key}", head, ""]
    for i, ctx in enumerate(ordered, start=1):
        sentence = ctx.get("sentence_text", "")
        lines += [f"### Contexto {i}", f"id: {ctx['id']}"]
        if global_scope:
            lines.append(f"Capítulo: {ctx.get('chapter') or '(sem capítulo)'}")
        lines += [
            f"Sentença: {sentence}",
            f"Parágrafo: {trim_paragraph(ctx.get('paragraph_text', ''), sentence)}",
        ]
        if ctx.get("co_cited"):
            lines.append("Citadas junto: " + ", ".join(ctx["co_cited"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Adjudicação (desempate dos contextos DIVERGENTE por um modelo mais forte)
# ---------------------------------------------------------------------------
def _slice(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


# Trechos da rubrica de avaliação reaproveitados byte a byte (vereditos e regras de evidência,
# conflito numérico e qualificadores, regras do trecho da fonte).
_ADJ_CRITERIOS = _slice(SYSTEM_RUBRIC, "Vereditos permitidos", "## Conflito numérico e qualificadores")
_ADJ_NUMERICO = _slice(SYSTEM_RUBRIC, "## Conflito numérico e qualificadores", "## O trecho da fonte")
_ADJ_TRECHO = _slice(SYSTEM_RUBRIC, "## O trecho da fonte", "## Justificativa")

_ADJ_ABERTURA = """Você é o árbitro final da atestação bibliográfica de uma tese de doutorado em Saúde Coletiva, escrita em português do Brasil. Cada citação da tese foi avaliada duas vezes, de forma independente, contra a mesma evidência da fonte citada, e as duas avaliações chegaram a vereditos diferentes. Sua tarefa é decidir o veredito final de cada um desses contextos, usando exclusivamente a evidência fornecida na mensagem do usuário.

## Entrada

A mensagem do usuário traz dois blocos:

1. Um bloco <evidencia> com o material recuperado da fonte citada, idêntico ao que as duas avaliações receberam (o mesmo texto, com o mesmo corte quando houver aviso de truncamento). O atributo "nivel" indica o que existe de fato: texto_completo, resumo, consensus ou nenhum.
2. Uma lista de contextos da tese que citam essa fonte e em que as duas avaliações divergiram. Cada contexto tem um "id", a sentença (em que a citação avaliada aparece como [cit] e as demais citações como [ref]), o parágrafo em volta e as duas avaliações anteriores, rotuladas "Avaliação 1" e "Avaliação 2", cada uma com veredito, trecho da fonte, justificativa e problema. A fonte pode estar em inglês, espanhol ou português; a tese está em português.

## Como arbitrar

- As avaliações anteriores são hipóteses, não autoridade. Confira cada uma contra a evidência: o trecho que ela cita existe e diz o que ela afirma? A justificativa lê corretamente a sentença e a parte que cabe a [cit]? A divergência vem de uma leitura mais estrita ou mais frouxa da sentença, de um número, de um qualificador omitido ou de um trecho que não existe?
- Você pode concordar com a Avaliação 1, concordar com a Avaliação 2 ou discordar das duas e dar um terceiro veredito. Não dê peso à ordem em que as avaliações aparecem nem ao número de vezes que um veredito se repete.
- Uma avaliação cujo trecho não foi encontrado na evidência já foi rebaixada para NAO_VERIFICAVEL pelo sistema automático. Isso não prova que a afirmação seja falsa: procure você mesmo a passagem pertinente.
- Decida pelo critério abaixo, que é o mesmo das avaliações. Não use conhecimento próprio sobre a literatura, não presuma o que o resto do artigo diria e não aceite a afirmação só porque ela é plausível. Se a evidência não decide, o veredito é NAO_VERIFICAVEL.
- Considere apenas a parte da sentença que se apoia em [cit]. Quando a sentença tem várias citações, avalie se a fonte [cit] sustenta a parte da afirmação que lhe cabe razoavelmente, sem exigir que ela sozinha sustente tudo.

## Critério de julgamento

"""

_ADJ_FORMATO = """## Justificativa

O campo "justificativa" deve ter no máximo 80 palavras, em português, e dizer qual avaliação estava certa (Avaliação 1, Avaliação 2 ou nenhuma das duas) e por quê, em termos concretos: o que a fonte mostra e o que a sentença afirma. Sem elogios, sem adjetivos promocionais, sem travessões. Cite números exatamente como aparecem.

## Confiança

O campo "confianca" registra o quanto a evidência permite sustentar o seu veredito: "alta" quando a passagem decisiva está clara na evidência, "media" quando o veredito depende de interpretar a sentença ou a passagem, "baixa" quando a evidência é ambígua, truncada ou escassa. Prefira "baixa" a inflar a confiança.

## Formato de saída

Responda somente com um array JSON válido (sem texto antes ou depois, sem cercas de código), com um objeto por contexto, na mesma ordem em que os contextos foram apresentados:

[
  {
    "id": "<id do contexto, copiado exatamente>",
    "veredito": "SUSTENTA" | "SUSTENTA_PARCIALMENTE" | "NAO_SUSTENTA" | "CONTRADIZ" | "NAO_VERIFICAVEL",
    "trecho_fonte": "<cópia literal da evidência, até 60 palavras, ou \\"\\">",
    "justificativa": "<até 80 palavras, em português>",
    "problema": null | "afirma_mais_que_a_fonte" | "atribuicao_errada" | "numero_divergente" | "fonte_secundaria" | "fora_de_escopo" | "outro",
    "confianca": "alta" | "media" | "baixa"
  }
]

Escape corretamente aspas e quebras de linha dentro das strings JSON. Não inclua nenhum contexto que não esteja na lista e não omita nenhum que esteja.

## Exemplos de calibração (fictícios, apenas para fixar o critério)

- A Avaliação 1 deu NAO_SUSTENTA porque não achou a passagem; a Avaliação 2 deu SUSTENTA e copiou "coverage reached 92.4% in 2015", que existe na evidência e coincide com a sentença "A cobertura chegou a 92% em 2015 [cit]". Veredito final SUSTENTA, e a justificativa diz que a Avaliação 2 estava certa.
- A Avaliação 1 deu SUSTENTA e a Avaliação 2 deu SUSTENTA_PARCIALMENTE com problema afirma_mais_que_a_fonte, porque a sentença diz "em todos os serviços" e a fonte trata de três serviços-piloto. Se a evidência confirma os três serviços-piloto, o veredito final é SUSTENTA_PARCIALMENTE e a Avaliação 2 estava certa.
- As duas avaliações erraram: uma deu CONTRADIZ por um número que na evidência é de outra grandeza, a outra deu SUSTENTA sem notar que a fonte qualifica o dado ("causa básica ou associada") e a tese omite o qualificador. Veredito final SUSTENTA_PARCIALMENTE com problema afirma_mais_que_a_fonte, e a justificativa diz que nenhuma das duas estava certa.
- A Avaliação 1 deu NAO_VERIFICAVEL por só haver o resumo e a Avaliação 2 deu SUSTENTA com um trecho que não está na evidência. O trecho inventado não vale; sem a passagem, o veredito final é NAO_VERIFICAVEL.

Lembre: a tese é o objeto de avaliação, a evidência é o único critério, as avaliações anteriores são hipóteses a verificar, e a dúvida honesta se expressa como NAO_VERIFICAVEL.
"""

ADJ_RUBRIC = _ADJ_ABERTURA + _ADJ_CRITERIOS + _ADJ_NUMERICO + _ADJ_TRECHO + _ADJ_FORMATO

INSTRUCAO_ADJUDICACAO = (
    "Adjudicação. Para cada contexto abaixo, as duas avaliações anteriores divergiram. "
    "Confira ambas contra a evidência do bloco anterior e decida o veredito final pelo critério "
    "das instruções. Responda somente com o array JSON descrito nas instruções, um objeto por "
    "contexto, na mesma ordem desta lista."
)


def _assessment_lines(label: str, assessment: dict | None) -> list[str]:
    lines = [f"#### {label}"]
    if not assessment:
        return [*lines, "(ausente)"]
    lines += [
        f"Veredito: {assessment.get('veredito') or '(vazio)'}",
        f"Trecho da fonte: {assessment.get('trecho_fonte') or '(vazio)'}",
        f"Justificativa: {assessment.get('justificativa') or '(vazia)'}",
        f"Problema: {assessment.get('problema') or 'nenhum'}",
    ]
    if assessment.get("trecho_verificado") is False:
        lines.append("Observação: o trecho citado não foi encontrado na evidência (veredito rebaixado).")
    return lines


def build_adjudication_block(
    capitulo: str, key: str, items: list[tuple[dict, dict]], global_scope: bool = False
) -> str:
    """Bloco B da adjudicação: contextos da chave com as duas avaliações anteriores.

    ``items`` traz pares (contexto, linha de ``avaliacoes.jsonl``); as avaliações vêm de
    ``passada1`` ("Avaliação 1") e ``passada2`` ("Avaliação 2").
    """
    scope_line = (
        "Escopo: tese completa (contextos de vários capítulos, apêndices, tabelas e notas)"
        if global_scope
        else f"Capítulo da tese: {capitulo}"
    )
    lines = [scope_line, f"Fonte avaliada: {key}", INSTRUCAO_ADJUDICACAO, ""]
    for i, (ctx, row) in enumerate(items, start=1):
        sentence = ctx.get("sentence_text", "")
        lines += [f"### Contexto {i}", f"id: {ctx['id']}"]
        if global_scope:
            lines.append(f"Capítulo: {ctx.get('chapter') or '(sem capítulo)'}")
        lines += [
            f"Sentença: {sentence}",
            f"Parágrafo: {trim_paragraph(ctx.get('paragraph_text', ''), sentence)}",
        ]
        if ctx.get("co_cited"):
            lines.append("Citadas junto: " + ", ".join(ctx["co_cited"]))
        lines += _assessment_lines("Avaliação 1", row.get("passada1"))
        lines += _assessment_lines("Avaliação 2", row.get("passada2"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
