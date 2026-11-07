"""Tabela editorial dos modelos, a partir de agregados públicos, sem API.

V1 a V4 são designações editoriais. As chaves históricas S1 a S4 nos
artefatos permanecem inalteradas para preservar a rastreabilidade.
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "text/latex/tables/editorial/tab_cap8_tamanho_modelo.tex"
MODELS = (
    ("claude-haiku-5-5", "Claude Haiku 5.5", "historico"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", "historico"),
    ("claude-opus-5-5", "Claude Opus 5.5", "principal"),
    ("claude-fable-5-1", "Claude Fable 5.1", "principal"),
)


def rows(data_dir, name):
    with (data_dir / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def one(data, **criteria):
    found = [row for row in data if all(row[key] == value for key, value in criteria.items())]
    if len(found) != 1:
        raise ValueError(f"Esperada uma linha para {criteria}; encontradas {len(found)}")
    return found[0]


def pt(value, decimals=0):
    return f"{float(value):,.{decimals}f}".translate(str.maketrans(",.", ".,"))


def render(data_dir=ROOT / "data"):
    data = rows(data_dir, "probe_frontier_agregado.csv")
    costs = rows(data_dir, "probe_frontier_custos.csv")
    metadata = json.loads((data_dir / "probe_frontier_metadata.json").read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        r"\caption{Comparação exploratória \textit{post hoc} de quatro modelos como revisores, relativa à referência PAR.}",
        r"\label{tab:cap8-tamanho-modelo}", r"\small", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Modelo & V1: $n$ & MATCH & FP & VPP (\%) & V2: $n$ & Recuperados/descartados/UNSURE \\",
        r"\midrule",
    ]
    for model, name, passada in MODELS:
        s1 = one(data, model=model, modo="vigilancia", estrato="S1", passada=passada)
        s2 = one(data, model=model, modo="vigilancia", estrato="S2", passada=passada)
        cells = [pt(s1[key]) for key in ("n", "MATCH", "FP")]
        cells += [pt(float(s1["VPP"]) * 100, 1), pt(s2["n"])]
        cells += ["/".join(pt(s2[key]) for key in ("PAR2_recuperados", "PAR2_descartados", "PAR2_unsure"))]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}}", r"\medskip", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Modelo & Confirmação: $n$ & VPP (\%) & $\kappa$ réplica V/C & US\$/par probe & US\$/par histórico \\",
        r"\midrule",
    ]
    for model, name, passada in MODELS:
        confirm = one(data, model=model, modo="confirmacao", estrato="C", passada=passada)
        total = one(data, model=model, modo="vigilancia", estrato="TOTAL_AMOSTRA", passada=passada)
        kappas = "/".join(pt(row["kappa_replica"], 3) if row["kappa_replica"] else "ND"
                          for row in (total, confirm))
        probe = one(costs, model=model, escopo="probe")
        historic = [row for row in costs if row["model"] == model and row["escopo"] == "coorte_historica"]
        probe_cost = pt(probe["usd_por_par_revisado"], 4) if probe["usd_por_par_revisado"] else "ND"
        historic_cost = pt(one(historic, model=model)["usd_por_par_revisado"], 4) if historic else "ND"
        cells = [pt(confirm["n"]), pt(float(confirm["VPP"]) * 100, 1), kappas, probe_cost, historic_cost]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    sizes = metadata["strata"]
    lines += [
        r"\bottomrule", r"\end{tabular}}",
        r"\par\footnotesize Fonte: \path{data/probe_frontier_agregado.csv}, "
        r"\path{data/probe_frontier_custos.csv} e \path{data/probe_frontier_metadata.json}. "
        f"V1: {sizes['S1']} aceitações históricas do Haiku na vigilância; "
        f"V2: {sizes['S2']} PAR~=~2 não aceitos por ele; "
        f"V3: {sizes['S3']} PAR~=~0 com decisões discordantes entre Haiku e Sonnet; "
        f"V4: amostra de {sizes['S4']} dos {metadata['s4_population']} potenciais pares restantes da vigilância. "
        r"As chaves S1 a S4 nos artefatos são históricas, não as etapas do protocolo. "
        r"$n$ indica resultados disponíveis, não tamanho original do estrato; ausências não são UNSURE. "
        r"Haiku e Sonnet: passada histórica; Opus e Fable: passada principal. "
        r"$\kappa$ entre a passada de referência e uma réplica, nos casos completos comuns: "
        r"vigilância/confirmação (V/C), com 137/38 pares para Haiku e 62/41 para Opus e Fable; sem réplica Sonnet. "
        r"ND: não disponível. Custos estimados por decisão final válida, incluindo pilotos, falhas e réplicas "
        r"no numerador e contando novamente as decisões de réplicas no denominador; Haiku probe por lote, "
        r"Opus/Fable por lote e acesso direto, históricos por acesso direto. "
        r"Não são comparação controlada de tarifas nem fatura.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate(destination=OUTPUT, data_dir=ROOT / "data"):
    destination.write_text(render(data_dir), encoding="utf-8", newline="\n")
    return destination


if __name__ == "__main__":
    print(generate())
