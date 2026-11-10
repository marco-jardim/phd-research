"""Etapa 4: relatório em Markdown a partir de ``avaliacoes.jsonl`` e dos artefatos da piloto."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .common import truncate_words
from .evidencias import LEVELS

VERDICT_ORDER = (
    "SUSTENTA", "SUSTENTA_PARCIALMENTE", "NAO_SUSTENTA", "CONTRADIZ",
    "NAO_VERIFICAVEL", "DIVERGENTE", "PENDENTE",
)
QUOTE_WORDS = 25


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fmt_dec(value: float, digits: int = 1) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def cell(text: object) -> str:
    return str(text if text not in (None, "") else "-").replace("|", "/").replace("\n", " ").strip()


def nv_reason(row: dict) -> str:
    """Motivo de um NAO_VERIFICAVEL final."""
    flags = set(row.get("flags") or [])
    if "sem_evidencia" in flags:
        return "sem evidência recuperada"
    if {"trecho_nao_encontrado", "trecho_ausente"} & flags:
        return "trecho citado pelo modelo não encontrado na evidência"
    if {"veredito_invalido", "sem_resposta_modelo", "resposta_truncada", "json_invalido"} & flags:
        return "resposta do modelo inválida ou incompleta"
    if row.get("nivel_evidencia") == "resumo":
        return "apenas o resumo estava disponível"
    if row.get("nivel_evidencia") == "consensus":
        return "apenas trechos do Consensus estavam disponíveis"
    return "decisão do modelo (evidência insuficiente)"


def estimate_cost(uso: dict[str, dict[str, int]], precos: dict) -> dict[str, float]:
    """Custo estimado em USD por modo; preços em USD por milhão de tokens."""
    out: dict[str, float] = {}
    discount = float(precos.get("batch_desconto", 0.0))
    for modo, tokens in uso.items():
        if modo not in ("batch", "sync"):  # "adjudicacao" usa outro modelo e tem seção própria
            continue
        total = (
            tokens.get("input_tokens", 0) * float(precos["input"])
            + tokens.get("output_tokens", 0) * float(precos["output"])
            + tokens.get("cache_creation_input_tokens", 0) * float(precos["cache_write"])
            + tokens.get("cache_read_input_tokens", 0) * float(precos["cache_read"])
        ) / 1_000_000
        out[modo] = total * (1 - discount) if modo == "batch" else total
    out["total"] = sum(out.values())
    return out


def load_precos(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in ("input", "output", "cache_write", "cache_read") if k not in data]
    if missing:
        raise ValueError(f"preços ausentes no YAML: {', '.join(missing)}")
    return data


def _location(row: dict) -> str:
    return f"{row.get('file', '')}:{row.get('line', 0)}"


def _sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r.get("file", ""), r.get("line", 0), r["key"], r["id"]))


PROBLEM_RANK = {
    "CONTRADIZ": 0, "NAO_SUSTENTA": 1, "DIVERGENTE": 2, "SUSTENTA_PARCIALMENTE": 3,
    "NAO_VERIFICAVEL": 4, "PENDENTE": 5, "SUSTENTA": 6,
}
NO_CHAPTER = "(sem capítulo)"


def _distribution(rows: list[dict]) -> str:
    counts = Counter(r["veredito_final"] for r in rows)
    return "; ".join(f"{v} {counts[v]}" for v in VERDICT_ORDER if counts.get(v))


def chapter_table(rows: list[dict]) -> list[str]:
    """Contextos por capítulo (ordem de aparição no texto) e por veredito."""
    order: list[str] = []
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        chapter = row.get("chapter") or NO_CHAPTER
        if chapter not in by_chapter:
            order.append(chapter)
        by_chapter[chapter].append(row)
    present = [v for v in VERDICT_ORDER if any(r["veredito_final"] == v for r in rows)]
    lines = ["", "### Por capítulo", "",
             "| Capítulo | Contextos | " + " | ".join(present) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in present) + " |"]
    for chapter in order:
        group = by_chapter[chapter]
        counts = Counter(r["veredito_final"] for r in group)
        lines.append(f"| {cell(chapter)} | {len(group)} | " + " | ".join(str(counts.get(v, 0)) for v in present) + " |")
    return lines


def key_table(rows: list[dict]) -> list[str]:
    """Uma linha por chave, com os problemas primeiro."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_key[row["key"]].append(row)

    def sort_key(item: tuple[str, list[dict]]):
        key, group = item
        worst = min(PROBLEM_RANK.get(r["veredito_final"], 4) for r in group)
        n_problems = sum(1 for r in group if r["veredito_final"] != "SUSTENTA")
        return (worst, -n_problems, key)

    lines = ["", "## Por chave", "",
             "| Chave | Contextos | Vereditos | Nível de evidência |", "| --- | ---: | --- | --- |"]
    for key, group in sorted(by_key.items(), key=sort_key):
        levels = ", ".join(sorted({r.get("nivel_evidencia", "?") for r in group}))
        lines.append(f"| `{key}` | {len(group)} | {_distribution(group)} | {levels} |")
    return lines


def adjudication_agreement(row: dict) -> str:
    """Com qual avaliação anterior o veredito adjudicado coincide: avaliacao_1, avaliacao_2 ou nenhuma."""
    verdict = (row.get("adjudicacao") or {}).get("veredito")
    if verdict == (row.get("passada1") or {}).get("veredito"):
        return "avaliacao_1"
    if verdict == (row.get("passada2") or {}).get("veredito"):
        return "avaliacao_2"
    return "nenhuma"


def adjudication_section(rows: list[dict], meta: dict) -> list[str]:
    """Seção "Adjudicação": vereditos adjudicados, concordância com as avaliações e lista crítica."""
    adjudicated = [r for r in rows if r.get("adjudicacao")]
    if not adjudicated:
        return []
    info = meta.get("adjudicacao") or {}
    models = sorted({r["adjudicacao"].get("modelo") or "?" for r in adjudicated})
    lines = ["", "## Adjudicação", "",
             f"Contextos DIVERGENTE adjudicados: {len(adjudicated)} (modelo: {', '.join(f'`{m}`' for m in models)}; "
             f"status: {info.get('status', '?')}; ainda divergentes: {info.get('contextos_ainda_divergentes', '?')}).",
             "", "### Veredito adjudicado", "", "| Veredito | Contextos |", "| --- | ---: |"]
    counts = Counter(r["adjudicacao"]["veredito"] for r in adjudicated)
    lines += [f"| {v} | {counts[v]} |" for v in VERDICT_ORDER if counts.get(v)]
    agreement = Counter(adjudication_agreement(r) for r in adjudicated)
    lines += ["", "### Concordância com as avaliações anteriores", "", "| Coincide com | Contextos |",
              "| --- | ---: |",
              f"| Avaliação 1 | {agreement.get('avaliacao_1', 0)} |",
              f"| Avaliação 2 | {agreement.get('avaliacao_2', 0)} |",
              f"| Nenhuma das duas | {agreement.get('nenhuma', 0)} |"]
    confidence = Counter((r["adjudicacao"].get("confianca") or "n/a") for r in adjudicated)
    lines += ["", "Confiança declarada: " + "; ".join(f"{c} {n}" for c, n in sorted(confidence.items())) + "."]

    lines += ["", "### NAO_SUSTENTA e CONTRADIZ após adjudicação", ""]
    critical = [r for r in adjudicated if r["veredito_final"] in ("NAO_SUSTENTA", "CONTRADIZ")]
    if not critical:
        lines.append("Nenhum.")
    for row in critical:
        adj = row["adjudicacao"]
        lines.append(
            f"- {_location(row)} `{row['key']}` ({row['veredito_final']}; coincide com "
            f"{adjudication_agreement(row)}; confiança {adj.get('confianca') or 'n/a'}): "
            f"{cell(adj.get('justificativa'))}"
        )

    tokens = (meta.get("uso") or {}).get("adjudicacao")
    if tokens:
        lines += ["", "### Uso de tokens da adjudicação", "",
                  "| Modo | Entrada | Saída | Escrita de cache | Leitura de cache |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for modo in ("batch", "sync"):
            t = tokens.get(modo, {})
            lines.append(
                f"| {modo} | {fmt_int(t.get('input_tokens', 0))} | {fmt_int(t.get('output_tokens', 0))} | "
                f"{fmt_int(t.get('cache_creation_input_tokens', 0))} | "
                f"{fmt_int(t.get('cache_read_input_tokens', 0))} |"
            )
    return lines


def build_report(
    rows: list[dict],
    meta: dict,
    artefatos: list[dict],
    classificacao: dict[str, dict],
    capitulo: str,
    precos: dict | None = None,
    escopo: str = "capitulo",
) -> str:
    rows = _sorted(rows)
    global_scope = escopo == "tese"
    lines = [f"# Relatório da atestação bibliográfica: {capitulo}", ""]
    lines += [
        f"Modelo: `{meta.get('modelo', '?')}`; data (UTC): {meta.get('data_utc', '?')}; "
        f"modo: {meta.get('parametros', {}).get('modo', '?')}; status: {meta.get('status', '?')}.",
        "",
        "## Contagens",
        "",
        "### Por veredito final",
        "",
        "| Veredito | Contextos | % |",
        "| --- | ---: | ---: |",
    ]
    total = len(rows)
    by_verdict = Counter(r["veredito_final"] for r in rows)
    for verdict in VERDICT_ORDER:
        if by_verdict.get(verdict):
            lines.append(f"| {verdict} | {by_verdict[verdict]} | {fmt_dec(100 * by_verdict[verdict] / total)} |")
    lines += ["", f"Total de contextos avaliados pelo modelo ou sem evidência: {total}.", ""]

    by_level = Counter(r["nivel_evidencia"] for r in rows)
    lines += ["### Por nível de evidência", "", "| Nível | Contextos |", "| --- | ---: |"]
    lines += [f"| {level} | {by_level.get(level, 0)} |" for level in LEVELS]

    reused = [r for r in rows if r.get("reaproveitado_de")]
    if reused:
        origins = Counter(r["reaproveitado_de"] for r in reused)
        lines += ["", f"Vereditos reaproveitados de execuções anteriores (evidência inalterada): {len(reused)} "
                  f"({', '.join(f'{o} {n}' for o, n in sorted(origins.items()))})."]
    if global_scope:
        lines += chapter_table(rows)
        lines += key_table(rows)

    lines += ["", "## Contextos", "",
              "| Local | Chave | Veredito | Trecho verificado | Problema | Números não encontrados | "
              "Justificativa | Trecho da fonte |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        verified = {True: "sim", False: "não", None: "n/a"}[row.get("trecho_verificado")]
        quote = truncate_words(row.get("trecho_fonte") or "", QUOTE_WORDS)
        numbers = ", ".join(row.get("numeros_nao_encontrados") or [])
        lines.append(
            f"| {cell(_location(row))} | `{row['key']}` | {row['veredito_final']} | {verified} | "
            f"{cell(row.get('problema'))} | {cell(numbers)} | {cell(row.get('justificativa'))} | {cell(quote)} |"
        )

    def listing(title: str, selected: list[dict]) -> None:
        lines.extend(["", f"## {title}", ""])
        if not selected:
            lines.append("Nenhum.")
            return
        for row in selected:
            lines.append(f"- {_location(row)} `{row['key']}`: {cell(row.get('justificativa'))}")

    listing("DIVERGENTE (revisão humana)", [r for r in rows if r["veredito_final"] == "DIVERGENTE"])
    listing("NAO_SUSTENTA e CONTRADIZ", [r for r in rows if r["veredito_final"] in ("NAO_SUSTENTA", "CONTRADIZ")])
    lines += adjudication_section(rows, meta)

    lines += ["", "## Números da sentença não encontrados na evidência (sinal mecânico, não é veredito)", ""]
    flagged = [r for r in rows if r.get("numeros_nao_encontrados")]
    if not flagged:
        lines.append("Nenhum.")
    for row in flagged:
        lines.append(f"- {_location(row)} `{row['key']}` ({row['veredito_final']}): "
                     f"{', '.join(row['numeros_nao_encontrados'])}")

    lines += ["", "## NAO_VERIFICAVEL por motivo", ""]
    reasons: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["veredito_final"] == "NAO_VERIFICAVEL":
            reasons[nv_reason(row)].append(row)
    if not reasons:
        lines.append("Nenhum.")
    for reason in sorted(reasons):
        lines += [f"### {reason} ({len(reasons[reason])})", ""]
        lines += [f"- {_location(r)} `{r['key']}`" for r in reasons[reason]]
        lines.append("")

    lines += ["## Obras próprias: conferência de artefatos", ""]
    if artefatos:
        counts = Counter(a["veredito"] for a in artefatos)
        lines += [f"- {verdict}: {n}" for verdict, n in sorted(counts.items())]
        lines.append("")
        for item in sorted(artefatos, key=lambda a: (a.get("file", ""), a.get("line", 0))):
            missing = f"; ausentes: {', '.join(item['ausentes'])}" if item.get("ausentes") else ""
            lines.append(f"- {item['file']}:{item['line']} `{item['key']}`: {item['veredito']}{missing}")
    else:
        lines.append("Nenhuma citação de obra própria no capítulo.")

    grey = sorted(k for k, v in classificacao.items() if v.get("classe") == "cinzenta")
    lines += ["", "## Literatura cinzenta (conferência manual)", ""]
    if grey:
        lines += [f"- `{k}`: nível de evidência {classificacao[k].get('nivel', 'nenhum')}" for k in grey]
    else:
        lines.append("Nenhuma.")

    lines += ["", "## Uso de tokens", "", "| Modo | Entrada | Saída | Escrita de cache | Leitura de cache |",
              "| --- | ---: | ---: | ---: | ---: |"]
    uso = meta.get("uso", {})
    for modo in ("batch", "sync"):
        t = uso.get(modo, {})
        lines.append(
            f"| {modo} | {fmt_int(t.get('input_tokens', 0))} | {fmt_int(t.get('output_tokens', 0))} | "
            f"{fmt_int(t.get('cache_creation_input_tokens', 0))} | {fmt_int(t.get('cache_read_input_tokens', 0))} |"
        )
    if precos is not None:
        cost = estimate_cost(uso, precos)
        lines += [
            "",
            f"Custo estimado (USD, tabela de preços informada; não é fatura): "
            f"batch {fmt_dec(cost.get('batch', 0.0), 2)}; sync {fmt_dec(cost.get('sync', 0.0), 2)}; "
            f"total {fmt_dec(cost['total'], 2)}.",
        ]
    return "\n".join(lines) + "\n"
