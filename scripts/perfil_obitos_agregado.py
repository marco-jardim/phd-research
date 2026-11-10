"""Perfil agregado, sem identificadores, datas ou valores por registro.

REFREC identifica SIM, COMPREC identifica Sinan. O conjunto COMPREC não é
denominado óbitos: atributos de R só são conservados quando invariantes entre
candidatos do grupo; a nota desse conjunto é a média das notas dos candidatos.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablacao_teste_par2_agregado import load_raw, numeric_frame
from scripts.epidemiological_impact import extract_clinical


def invariant(series):
    values = series.dropna().unique()
    return values[0] if len(values) == 1 and series.notna().all() else None


def summarize(raw):
    numeric = numeric_frame(raw)
    clinical = extract_clinical(numeric)
    clinical["idade"] = clinical["idade_obito"].where(clinical["idade_obito"].between(0, 120))
    clinical["sexo"] = raw["R_SEXO"].str.strip().str.upper() if "R_SEXO" in raw else None
    clinical["sexo"] = clinical["sexo"].where(clinical["sexo"].isin(["M", "F"]))
    clinical["nota"] = numeric["nota final"]
    groups = {"pares_candidatos": clinical,
              "vinculos_par1": clinical.loc[numeric.PAR.eq(1)]}
    for identifier, name in (("COMPREC", "registros_comprec"), ("REFREC", "obitos_refrec")):
        # Nenhum identificador é serializado. Nota é escore de par, não do óbito.
        groups[name] = clinical.groupby(raw[identifier], sort=True).agg(
            idade=("idade", invariant), sexo=("sexo", invariant), nota=("nota", "mean"))
    rows = []
    for name, group in groups.items():
        row = dict(conjunto=name, n=len(group),
                   idade_origem="R_DTOBITO - R_DTNASC; dias/365.25; suporte 0 a 120"
                   if {"R_DTOBITO", "R_DTNASC"} <= set(raw) else "não disponível no extrato",
                   sexo_origem="R_SEXO" if "R_SEXO" in raw else "não disponível no extrato",
                   nota_unidade="media_por_registro" if name in {"registros_comprec", "obitos_refrec"} else "par")
        for field in ("idade", "nota"):
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            row.update({f"{field}_n": len(values), f"{field}_media": values.mean(),
                        f"{field}_dp": values.std(ddof=1), f"{field}_mediana": values.median()})
        for sex, label in (("M", "masculino"), ("F", "feminino"), (None, "ausente_ou_ambiguo")):
            n = int(group.sexo.isna().sum() if sex is None else group.sexo.eq(sex).sum())
            row[f"sexo_{label}_n"] = n
            row[f"sexo_{label}_pct"] = 100 * n / len(group) if len(group) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def table(frame):
    labels = {"pares_candidatos": "Potenciais pares", "vinculos_par1": "Pares verdadeiros PAR=1",
              "registros_comprec": "Registros COMPREC", "obitos_refrec": "Óbitos REFREC"}
    def number(v):
        return f"{v:.1f}".replace(".", "{,}") if pd.notna(v) else "n.d."
    lines = [r"\begin{table}[htbp]", r"\centering\scriptsize",
             r"\caption{Perfil agregado do extrato por unidade de contagem.}",
             r"\label{tab:perfil-obitos-agregado}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Conjunto & $n$ & Idade: $n$ & Média (DP); mediana & Nota: média (DP); mediana & M: $n$ (\%) & F: $n$ (\%) \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        if row.conjunto not in labels:
            continue
        lines.append(f"{labels[row.conjunto]} & {row.n:,} & {row.idade_n:,}".replace(",", ".")
                     + f" & {number(row.idade_media)} ({number(row.idade_dp)}); {number(row.idade_mediana)}"
                     + f" & {number(row.nota_media)} ({number(row.nota_dp)}); {number(row.nota_mediana)}"
                     + f" & {row.sexo_masculino_n:,}".replace(",", ".") + f" ({number(row.sexo_masculino_pct)})"
                     + f" & {row.sexo_feminino_n:,}".replace(",", ".") + f" ({number(row.sexo_feminino_pct)}) " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\par\footnotesize Fonte: \path{data/perfil_obitos_agregado.csv}, gerado por "
              r"\path{scripts/perfil_obitos_agregado.py}. Idade derivada das datas do SIM, em anos, "
              r"restrita a 0 a 120; DP amostral. M e F usam o total do conjunto como denominador. "
              r"Nos grupos por registro, idade e sexo só são mantidos quando invariantes e completos "
              r"entre candidatos; a nota é a média por registro das notas dos pares, não atributo do óbito. "
              r"COMPREC identifica o Sinan, REFREC o SIM. Os óbitos REFREC têm ao menos um potencial par nos passos exportados. Ausências e ambiguidades constam do CSV. "
              r"Este perfil da base inteira não substitui o perfil dos vínculos recuperados no teste.",
              r"\end{table}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    args = parser.parse_args()
    frame = summarize(load_raw(args.input))
    historical = json.loads((ROOT / "data/epidemiological_profile.json").read_text(encoding="utf-8"))
    rows = []
    for name in ("recuperados", "ja_encontrados"):
        values = historical[name]
        row = dict(conjunto=name, n=values["n"], idade_n=values["idade_n_validos"],
                   idade_media=values["idade_media"], idade_dp=values["idade_dp"],
                   nota_n=values["n"], nota_media=values["nota_media"],
                   idade_origem="agregado histórico arredondado: data/epidemiological_profile.json",
                   sexo_origem="agregado histórico: R_SEXO", nota_unidade="par")
        for key, label in (("sexo_masculino", "masculino"), ("sexo_feminino", "feminino"),
                           ("sexo_outro", "ausente_ou_ambiguo")):
            row[f"sexo_{label}_n"] = values[key]
            row[f"sexo_{label}_pct"] = 100 * values[key] / values["n"]
        rows.append(row)
    # Os dois grupos do teste são importados de agregado já versionado, não
    # reconstruídos por novo ajuste do RF. Medianas e DP da nota não registrados
    # permanecem ausentes; nenhum valor é inferido das células históricas TeX.
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    output = ROOT / "data/perfil_obitos_agregado.csv"
    frame.to_csv(output, index=False, lineterminator="\n")
    # A tabela usa o CSV público como única entrada editorial.
    (ROOT / "text/latex/tables/editorial/tab_perfil_obitos_agregado.tex").write_text(
        table(pd.read_csv(output)), encoding="utf-8", newline="\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
