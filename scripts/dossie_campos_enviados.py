"""Audit the 29 extract sub-scores against both frozen v4 dossier field audits.

The municipality is serialized as the scalar ``municipality``. The remaining
thirteen sub-scores retain their extract names. Nothing is added to the dossier.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.llm_dossier import build_dossier

PREFIXES = ("NOME ", "NOMEMAE ", "DTNASC ", "CODMUNRES ", "ENDERECO ")
CATEGORIES = {"NOME": "nome", "NOMEMAE": "nome da mãe", "DTNASC": "nascimento",
              "CODMUNRES": "município", "ENDERECO": "endereço"}


def field_rows(extract: Path, audits: Path) -> list[dict]:
    header = pd.read_csv(extract, sep=";", dtype=str, nrows=0).columns
    scores = [c for c in header if c.startswith(PREFIXES)]
    if len(scores) != 29:
        raise ValueError(f"Expected 29 sub-scores, found {len(scores)}")
    # Synthetic sentinels prove the scalar municipality's origin as well as the
    # dictionary keys; no extract row is read or saved.
    features = build_dossier(pd.Series(dict.fromkeys(scores, 1.0)),
                             pair_id="synthetic").to_dict()["features"]
    sent = set().union(*(features[group] for group in ("name", "mother", "dob", "address")))
    if features["municipality"] != 1.0:
        raise ValueError("Municipality scalar did not preserve the extract score")
    sent.add("CODMUNRES local igual")
    if len(sent) != 14:
        raise ValueError(f"Historical boundary changed: {len(sent)} sub-scores")
    for mode in ("vigilancia", "confirmacao"):
        path = audits / f"dossier_field_audit_{mode}.json"
        if not path.exists() and mode == "confirmacao":
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        names = set(audit["field_names"])
        expected = (sent - {"CODMUNRES local igual"}) | {"municipality"}
        if not expected <= names or (set(scores) & names) != sent - {"CODMUNRES local igual"}:
            raise ValueError(f"Frozen field audit differs: {mode}")
    return [{"campo": c, "categoria": CATEGORIES[c.split()[0]],
             "enviado": "sim" if c in sent else "não"} for c in scores]


def render(rows: list[dict]) -> str:
    lines = [r"% Gerada por scripts/dossie_campos_enviados.py; não editar.",
             r"\begin{table}[htbp]", r"\centering\scriptsize",
             r"\caption{Subescores do extrato e inclusão no dossiê histórico: 14 de 29.}",
             r"\label{tab:dossie-campos}", r"\begin{tabular}{lll}", r"\toprule",
             r"Campo & Categoria & Enviado \\", r"\midrule"]
    lines.extend(" & ".join(row[k] for k in ("campo", "categoria", "enviado")) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\footnotesize Fonte: \path{data/dossie_campos_enviados.csv}, cabeçalho do extrato e auditorias históricas v4 dos dois modos. O escore municipal é enviado como \texttt{municipality}. Compostos, indicadores e nota final não integram os 29 subescores deste quadro.}",
              r"\end{table}", ""]
    from scripts.tabelas_persona_editorial import transform
    return transform("tab_dossie_campos", "\n".join(lines))


def generate(extract: Path, audits: Path, output: Path, table: Path) -> None:
    rows = field_rows(extract, audits)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["campo", "categoria", "enviado"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    table.write_text(render(rows), encoding="utf-8", newline="\n")
    print(json.dumps({"subescores": len(rows), "enviados": sum(r["enviado"] == "sim" for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--audits", type=Path, default=ROOT / "data/sprint3b/anthropic_sem_mo_v4")
    args = parser.parse_args()
    generate(args.extract, args.audits, ROOT / "data/dossie_campos_enviados.csv",
             ROOT / "text/latex/tables/editorial/tab_dossie_campos.tex")
