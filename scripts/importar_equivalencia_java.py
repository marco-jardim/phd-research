"""Importa o agregado do CdR e gera a tabela de equivalência da tese."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(r"D:\git\Comparador-de-Registros\tests\equivalencia")
FIELDS = ["indice", "subescore", "n", "concordancia", "diferenca_maxima"]


def mecanismo(row: dict[str, str]) -> str:
    """Resume os mecanismos descritos na causa, sem atribuir efeito isolado."""
    name = row["subescore"]
    cause = row["causa"].lower()
    if "sem divergencia" in cause:
        return "sem divergência"
    if name == "nota_final":
        return "soma dos componentes"
    if "inverte" in name:
        return "peso da inversão; reconhecimento"
    if "raros" in name or "comuns" in name:
        return "tabelas de frequência; nome único; contador"
    if "soundex" in name:
        return "Soundex; contador; precisão"
    if "abreviacoes" in name or "iguais" in name:
        return "contador; precisão; dígitos"
    return "contador; dígitos"


def decimal_pt(value: str) -> str:
    return value.replace(".", ",")


def label(name: str) -> str:
    if name == "nota_final":
        return "Nota final"
    if name.startswith("data_"):
        return {
            "data_igual": "Data: igualdade",
            "data_um_digito": "Data: um dígito",
            "data_inverte_dia": "Data: inversão do dia",
            "data_inverte_mes": "Data: inversão do mês",
            "data_inverte_ano": "Data: inversão do ano",
        }[name]
    who, criterion = name.split("_", 1)
    return ("Paciente" if who == "paciente" else "Mãe") + ": " + {
        "primeiro": "primeiro fragmento",
        "ultimo": "último fragmento",
        "iguais": "fragmentos iguais",
        "raros": "fragmentos raros",
        "comuns": "fragmentos comuns",
        "soundex": "Soundex",
        "abreviacoes": "abreviações",
    }[criterion]


def render_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"% Gerada por scripts/importar_equivalencia_java.py. Não editar à mão.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Concordância entre o protótipo Java e a implementação Python do Comparador de Registros em 500 pares sintéticos construídos para exercitar casos de fronteira; proporção de pares com subescore idêntico (tolerância de $10^{-9}$) e maior diferença absoluta}",
        r"\label{tab:equivalencia-java}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Subescore & Concordância (\%) & Diferença máxima \\",
        r"\midrule",
    ]
    for row in rows:
        if row["indice"] in {"14", "19"}:
            lines.append(r"\midrule")
        percent = f"{Decimal(row['concordancia']) * 100:.1f}"
        lines.append(
            f"{label(row['subescore'])} & {decimal_pt(percent)} & "
            + decimal_pt(row["diferenca_maxima"])
            + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize Nota: o lote não estima a concordância em dados reais. As tabelas de frequência são sintéticas e idênticas nos dois lados. Fonte: \path{tests/equivalencia/resultado_agregado.csv} do repositório do CdR, commit \texttt{0d5b411}, PR \#25 \cite{Jardim2025comparador}. Importação e geração: \path{scripts/importar_equivalencia_java.py} desta tese.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source_dir / "resultado_agregado.csv"
    report = args.source_dir / "relatorio_equivalencia.md"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream, delimiter=";"))
    if len(raw) != 20 or [row["indice"] for row in raw] != [str(i) for i in range(20)]:
        raise ValueError("Esperadas 20 linhas ordenadas, com índices de 0 a 19")
    if {row["n"] for row in raw} != {"500"}:
        raise ValueError("O lote documentado contém 500 pares")
    rows = [{**{key: row[key] for key in FIELDS}, "mecanismo": mecanismo(row)} for row in raw]
    report_text = report.read_text(encoding="utf-8")
    seed = re.search(r"\bSeed\s+(\d+)", report_text)
    date = re.search(r'"data_utc":\s*"([^"]+)"', report_text)
    if date is None:
        raise ValueError("Data da execução ausente do relatório")
    meta = {
        "repositorio": "https://github.com/marco-jardim/Comparador-de-Registros",
        "commit": "0d5b411",
        "pr": 25,
        "branch": "test/equivalencia-java",
        "jdk": "Temurin 21.0.13",
        "n": 500,
        "sublotes": {"ascii": 250, "acentos": 250},
        "seed": int(seed.group(1)) if seed else None,
        "data_utc": date.group(1),
        "tolerancia": 1e-9,
        "fonte": "tests/equivalencia/resultado_agregado.csv",
        "relatorio": "tests/equivalencia/relatorio_equivalencia.md",
        "sha256_fonte": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256_relatorio": hashlib.sha256(report.read_bytes()).hexdigest(),
        "limite": "Lote sintético de fronteira; não estima concordância em dados reais.",
    }
    data = args.output_root / "data"
    tables = args.output_root / "text" / "latex" / "tables"
    data.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    with (data / "equivalencia_java_python.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*FIELDS, "mecanismo"])
        writer.writeheader()
        writer.writerows(rows)
    (data / "equivalencia_java_python_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (tables / "tab_equivalencia_java.tex").write_text(render_table(rows), encoding="utf-8")
    print("Importadas 20 linhas; gerados CSV, metadados e tabela LaTeX.")


if __name__ == "__main__":
    main()
