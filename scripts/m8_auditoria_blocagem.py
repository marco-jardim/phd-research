"""M8, parte A: auditoria offline; publica somente tabelas agregadas, nunca pares.

Uso: python scripts/m8_auditoria_blocagem.py [--input arquivo --out-dir pasta]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("NOME ", "NOMEMAE ", "DTNASC ", "CODMUNRES ", "ENDERECO ")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_extract(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    aliases = {c: c.split(",")[0] for c in raw if c.startswith(("COMPREC,", "REFREC,"))}
    raw = raw.rename(columns=aliases)
    for c in ["COMPREC", "REFREC", "PASSO", "PAR", "nota final"]:
        if c not in raw:
            raise ValueError(f"Coluna obrigatória ausente: {c}")
    if raw[["COMPREC", "REFREC", "PASSO"]].isna().any().any():
        raise ValueError("Identificador/passo ausente; contagens não seriam confiáveis")
    out = raw[["COMPREC", "REFREC", "PASSO"]].copy()
    out["y"] = pd.to_numeric(raw["PAR"], errors="raise").isin([1, 2])
    out["nota final"] = pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False), errors="raise")
    for c in raw:
        if c.startswith(PREFIXES):
            out[c] = pd.to_numeric(raw[c].str.replace(",", ".", regex=False), errors="coerce")
    # Somente igualdade local: nenhum valor demográfico individual sai da memória.
    if "R_SEXO,C,1,0" in raw and "C_SEXO,C,1,0" in raw:
        r, c = raw["R_SEXO,C,1,0"].str.strip(), raw["C_SEXO,C,1,0"].str.strip()
        valid = r.isin(["1", "2", "M", "F"]) & c.isin(["1", "2", "M", "F"])
        out["SEXO igualdade"] = r.eq(c).astype(float).where(valid)
    out["pair_id"] = [hashlib.sha256(f"{a}|{b}|{s}".encode()).hexdigest()[:16]
                      for a, b, s in out[["COMPREC", "REFREC", "PASSO"]].itertuples(index=False, name=None)]
    return out


def pair_sets(df: pd.DataFrame, positive: bool = False) -> dict[str, set]:
    source = df[df["y"]] if positive else df
    return {str(s): set(zip(g["COMPREC"], g["REFREC"])) for s, g in source.groupby("PASSO")}


def precondition(df: pd.DataFrame) -> dict:
    per_pair = df.groupby(["COMPREC", "REFREC"])["PASSO"].nunique()
    linked = df[df["y"]].groupby(["COMPREC", "REFREC"])["PASSO"].nunique()
    return {"linhas": len(df), "pares_distintos": len(per_pair),
            "pares_multiplos_passos": int(per_pair.gt(1).sum()),
            "linhas_vinculo": int(df["y"].sum()), "vinculos_distintos": len(linked),
            "vinculos_multiplos_passos": int(linked.gt(1).sum()),
            "captura_recaptura_viavel": bool(linked.gt(1).any()),
            "observacao": "Sobreposição é condição necessária, não suficiente: exclusão progressiva invalida listas de captura independentes."}


def yield_by_step(df: pd.DataFrame) -> pd.DataFrame:
    links = pair_sets(df, True)
    rows = []
    for s, g in df.groupby("PASSO"):
        mine = links.get(str(s), set())
        others = set().union(*(v for k, v in links.items() if k != str(s)))
        unique = g.drop_duplicates(["COMPREC", "REFREC"])
        # A nota é descritiva por linha; as frações positivas usam pares distintos.
        positive = unique[unique["y"]]
        row = {"passo": str(s), "linhas": len(g), "pares_distintos": len(unique),
               "linhas_vinculo": int(g["y"].sum()), "vinculos": len(mine),
               "vinculos_so_neste_passo": len(mine - others),
               "vinculos_tambem_outro_passo": len(mine & others),
               "taxa_vinculo_pares": len(mine) / len(unique),
               "fracao_vinculos_cinza_nota": float(positive["nota final"].between(5, 9, inclusive="left").mean()) if len(positive) else np.nan,
               "fracao_vinculos_nota_ge9": float(positive["nota final"].ge(9).mean()) if len(positive) else np.nan}
        for q in [0, .25, .5, .75, .9, .95, 1]:
            row[f"nota_q{q:g}"] = g["nota final"].quantile(q)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("passo", key=lambda x: pd.to_numeric(x))


def overlap(df: pd.DataFrame) -> pd.DataFrame:
    candidates, links = pair_sets(df), pair_sets(df, True)
    return pd.DataFrame([{"passo_a": a, "passo_b": b,
                          "pares_em_comum": len(candidates[a] & candidates[b]),
                          "vinculos_em_comum": len(links.get(a, set()) & links.get(b, set()))}
                         for a, b in itertools.product(sorted(candidates, key=int), repeat=2)])


def chapman(n_a: int, n_b: int, n_ab: int) -> dict:
    """Chapman + IC normal aproximado, condicionado às duas listas válidas."""
    if not 0 < n_ab <= min(n_a, n_b):
        raise ValueError("Exige sobreposição positiva e contagens coerentes")
    observed = n_a + n_b - n_ab
    estimate = (n_a + 1) * (n_b + 1) / (n_ab + 1) - 1
    variance = ((n_a + 1) * (n_b + 1) * (n_a - n_ab) * (n_b - n_ab)
                / ((n_ab + 1) ** 2 * (n_ab + 2)))
    return {"metodo": "Chapman", "observados_listas": observed, "total_estimado": estimate,
            "nao_capturados": max(0, estimate - observed),
            "ic95_inferior": max(observed, estimate - 1.96 * np.sqrt(variance)),
            "ic95_superior": max(observed, estimate + 1.96 * np.sqrt(variance)),
            "variancia": variance}


def loglinear(captures: dict[str, set]) -> dict:
    """Fienberg: efeitos principais + interações entre pares de listas, sem 3-vias.

    IC delta-log do intercepto: célula não observada = exp(intercepto).
    Exige listas sem exclusão progressiva e população fechada, nunca inferidas aqui.
    """
    import statsmodels.api as sm

    keys = sorted(captures)
    k = len(keys)
    if not 3 <= k <= 10:
        raise ValueError("Modelo exige de 3 a 10 listas")
    union = set().union(*captures.values())
    counts = {}
    for pair in union:
        profile = tuple(int(pair in captures[s]) for s in keys)
        counts[profile] = counts.get(profile, 0) + 1
    profiles = list(itertools.product([0, 1], repeat=k))[1:]
    x = np.array([[1, *p, *(p[a] * p[b] for a, b in itertools.combinations(range(k), 2))]
                  for p in profiles], dtype=float)
    y = np.array([counts.get(p, 0) for p in profiles])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Design não identificável")
    fit = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    if not fit.converged or not np.isfinite(fit.params).all():
        raise ValueError("GLM não convergiu")
    missing = float(np.exp(fit.params[0]))
    se = float(np.sqrt(fit.cov_params()[0, 0]))
    return {"metodo": "Fienberg_Poisson_principais_e_interacoes_duplas", "observados_listas": len(union),
            "total_estimado": len(union) + missing, "nao_capturados": missing,
            "ic95_inferior": len(union) + float(np.exp(fit.params[0] - 1.96 * se)),
            "ic95_superior": len(union) + float(np.exp(fit.params[0] + 1.96 * se)),
            "graus_liberdade_residuais": int(fit.df_resid)}


def capture_table(df: pd.DataFrame) -> pd.DataFrame:
    # O extrato é sequencial: mesmo alguma recaptura não autoriza estimar o universo.
    return pd.DataFrame([{"metodo": "não estimado", "status": "inviável",
                          "vinculos_observados": precondition(df)["vinculos_distintos"],
                          "total_estimado": np.nan, "nao_capturados": np.nan,
                          "ic95_inferior": np.nan, "ic95_superior": np.nan,
                          "motivo": "Exclusão progressiva documentada; vínculos sem recapturas no extrato. Zero sobreposição não significa zero perdidos."}])


def dictionary(df: pd.DataFrame) -> dict:
    result = {}
    for c in df:
        if not (c.startswith(PREFIXES) or c == "SEXO igualdade"):
            continue
        diagnostic = any(t in c for t in ["raros", "comuns", "abrev", "inv ", "ap 1digi"])
        result[c] = {"descricao": c, "tipo": "diagnóstico de composição/erro, não igualdade" if diagnostic else "proxy de concordância",
                     "limiar_exato": 1.0, "limiar_alto": .8,
                     "nota": "Fração >=1 e >=0.8 não equivale a igualdade de Soundex; alguns subescores têm teto <1. Abreviação indica erro, não concordância."}
    return result


def aggregate_fields(df: pd.DataFrame, fields: dict) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("PASSO"):
        for c, info in fields.items():
            x = g[c]
            rows.append({"passo": str(s), "campo": c, "n": len(g), "n_disponivel": int(x.notna().sum()),
                         "fracao_ausente": float(x.isna().mean()), "fracao_exata_proxy": float(x.ge(1).mean()),
                         "fracao_alta_proxy": float(x.ge(.8).mean()), "media": float(x.mean()) if x.notna().any() else None,
                         "chave_provavel_proxy": bool(x.ge(.8).mean() >= .99 and info["tipo"] == "proxy de concordância")})
    return pd.DataFrame(rows)


def link_profiles(df: pd.DataFrame, fields: dict) -> pd.DataFrame:
    links = df[df["y"]].drop_duplicates(["COMPREC", "REFREC"])
    sets = pair_sets(df, True)
    early = min(df["PASSO"].unique(), key=int)
    all_other = {s: set().union(*(v for t, v in sets.items() if t != s)) for s in sets}
    exclusive = pd.Series([((a, b) not in all_other[str(s)]) and int(s) > int(early)
                          for a, b, s in links[["COMPREC", "REFREC", "PASSO"]].itertuples(index=False, name=None)], index=links.index)
    groups = {"todos_vinculos": links, "so_passos_tardios": links[exclusive],
              "nome_baixo_lt_0.5": links[links["NOME qtd frag iguais"].lt(.5)]}
    groups.update({f"passo_{s}": g for s, g in links.groupby("PASSO")})
    rows = []
    for label, g in groups.items():
        for c in fields:
            x = g[c]
            rows.append({"grupo": label, "n_vinculos": len(g), "campo": c,
                         "fracao_abaixo_0.8": float(x.lt(.8).mean()) if len(g) else None,
                         "fracao_zero": float(x.eq(0).mean()) if len(g) else None,
                         "fracao_ausente": float(x.isna().mean()) if len(g) else None})
    return pd.DataFrame(rows)


def run(input_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    df = load_extract(input_path)
    pre = precondition(df)
    print("PRÉ-CONDIÇÃO:", json.dumps(pre, ensure_ascii=False))
    write_json(output / "precondicao.json", pre)
    fields = dictionary(df)
    write_json(output / "dicionario_campos.json", fields)
    tables = {"blocagem_rendimento_por_passo.csv": yield_by_step(df),
              "blocagem_sobreposicao.csv": overlap(df),
              "blocagem_captura_recaptura.csv": capture_table(df),
              "concordancia_por_passo.csv": aggregate_fields(df, fields),
              "perfil_vinculos.csv": link_profiles(df, fields)}
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding="utf-8")
    write_json(output / "proveniencia_a.json", {"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "pandas": pd.__version__, "numpy": np.__version__,
               "nota_usada": "nota final do comparador, NÃO SCORE ORL; faixas solicitadas [5,9) e >=9"})
    print(tables["blocagem_rendimento_por_passo.csv"].to_string(index=False))
    return pre


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/m8")
    args = parser.parse_args()
    run(args.input, args.out_dir)


if __name__ == "__main__":
    main()
