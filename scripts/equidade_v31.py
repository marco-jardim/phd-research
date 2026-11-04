"""Estudo 5: equidade avaliada sobre o classificador supervisionado e a cascata v3.1.

Reaproveita, sem alterá-las, as rotinas de ``scripts/qw3_fairness_analysis.py``
(fatiamento, IC de Wilson, qui-quadrado/Fisher, razão de chances, Benjamini-Hochberg)
e avalia quatro decisões binárias sobre os mesmos 61.696 potenciais pares:

* ``corte6``: ``nota final >= 6,0`` (análise histórica, deve reproduzir 7.452 preditos positivos);
* ``xgb_sens6``: XGBoost com isotônica OOF agrupada por REFREC (``data/pcal_ml_oof.csv``),
  com o limiar que mais se aproxima da sensibilidade global do corte 6,0;
* ``xgb_p090``: o mesmo estimador com ``p_cal_ml >= 0,9`` (limiar da guarda);
* ``cascata_v31_vigilancia`` e ``cascata_v31_confirmacao``: aceitação automática da
  cascata v3.1 (``final == MATCH``, motor ou LLM); HUMANO, UNSURE e DEFERRED contam como
  não aceitos.

Subgrupos demográficos (sexo, faixa etária) e operacionais (quartil do nome, origem,
município, passo de blocagem, ano do óbito, completude dos campos) ficam separados.
Não há variável de raça/cor na base (verificado nas colunas do extrato).

Saídas agregadas (sem linhas por par) em ``data/qw3`` (não versionado) e em
``data/equidade_v31_*.csv``; tabelas LaTeX em ``text/latex/tables``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_qw3():
    path = ROOT / "scripts" / "qw3_fairness_analysis.py"
    spec = importlib.util.spec_from_file_location("qw3_fairness_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qw3_fairness_analysis"] = module
    spec.loader.exec_module(module)
    return module


qw3 = _load_qw3()

ALPHA = 0.05
N_MIN = 30
CAPITAL_RJ = 330455

DEMOGRAFICAS = ["sexo", "faixa_idade"]
OPERACIONAIS = [
    "nome_qtd_quartis",
    "bandeira",
    "municipio",
    "passo",
    "ano_obito",
    "completude",
]
DIMENSOES = DEMOGRAFICAS + OPERACIONAIS
TIPO = {d: "demografica" for d in DEMOGRAFICAS} | {d: "operacional" for d in OPERACIONAIS}

FIELDS_COMPLETUDE = [
    "R_DTNASC",
    "R_BAIRES",
    "C_DTNASC",
    "C_BAIRES",
]

DECISOES = [
    "corte6",
    "xgb_sens6",
    "xgb_p090",
    "cascata_v31_vigilancia",
    "cascata_v31_confirmacao",
]


def make_pair_id(comprec: str, refrec: str, passo: str) -> str:
    raw = f"{comprec}|{refrec}|{passo}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_frame(input_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = qw3.Config(
        input_csv=input_csv,
        output_dir=ROOT / "data" / "qw3",
        sep=";",
        seed=2026,
        target_col="PAR",
        target_pos_values=(1, 2),
        score_col="nota final",
        threshold=6.0,
        prediction_col=None,
        score_band_col="SCORE,C,15,0",
        alpha=ALPHA,
        n_min_cell=N_MIN,
    )
    frame, _ = qw3._prepare_fairness_frame(cfg)
    raw = pd.read_csv(input_csv, sep=";", dtype=str, low_memory=False)

    comprec = raw["COMPREC,C,12,0"]
    refrec = raw["REFREC,C,12,0"]
    passo = raw["PASSO"]
    frame["pair_id"] = [make_pair_id(c, r, p) for c, r, p in zip(comprec, refrec, passo)]
    frame["refrec"] = refrec.to_numpy()

    codmun = qw3._parse_int(raw["R_CODMUNRES,C,6,0"]).combine_first(
        qw3._parse_int(raw["C_CODMUNRES,C,6,0"])
    )
    municipio = pd.Series("MISSING", index=raw.index, dtype="string")
    municipio = municipio.mask(codmun.notna() & (codmun == CAPITAL_RJ), "capital_RJ")
    municipio = municipio.mask(codmun.notna() & (codmun != CAPITAL_RJ), "outros_municipios")
    frame["municipio"] = municipio.to_numpy()

    frame["passo"] = ("passo_" + raw["PASSO"].astype("string").str.strip()).to_numpy()

    obito = qw3._parse_date_ddmmyyyy(raw["R_DTOBITO,C,10,0"])
    frame["ano_obito"] = obito.dt.year.astype("Int64").astype("string").fillna("MISSING").to_numpy()

    ausentes = sum(
        (raw[col if col in raw.columns else qw3._resolve_column(raw, col)].isna()).astype(int)
        for col in FIELDS_COMPLETUDE
    )
    frame["completude"] = np.where(ausentes == 0, "completo", "incompleto")

    frame["nota"] = qw3._parse_decimal(raw["nota final"]).astype("float64").to_numpy()
    meta = {"rows": int(len(frame)), "raca_cor_presente": any("RACA" in c.upper() for c in raw.columns)}
    return frame, meta


def attach_decisions(frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    oof = pd.read_csv(ROOT / "data" / "pcal_ml_oof.csv", dtype={"pair_id": str})
    merged = frame.merge(oof[["pair_id", "p_cal_ml"]], on="pair_id", how="left", validate="one_to_one")
    if merged["p_cal_ml"].isna().any():
        raise RuntimeError("pair_id sem p_cal_ml: chave COMPREC|REFREC|PASSO não reproduz o OOF")
    merged["corte6"] = (merged["nota"] >= 6.0).astype("int64")
    if int(merged["corte6"].sum()) != int(merged["y_pred"].sum()):
        raise RuntimeError("corte6 diverge do y_pred do script qw3")

    y = merged["y_true"].to_numpy()
    p = merged["p_cal_ml"].to_numpy()
    sens_alvo = merged.loc[merged["corte6"] == 1, "y_true"].sum() / y.sum()
    candidatos = np.unique(p[y == 1])
    sens_cand = np.array([(p[y == 1] >= t).mean() for t in candidatos])
    melhor = np.abs(sens_cand - sens_alvo).min()
    t_sens = float(candidatos[np.where(np.isclose(np.abs(sens_cand - sens_alvo), melhor))[0].max()])
    merged["xgb_sens6"] = (p >= t_sens).astype("int64")
    merged["xgb_p090"] = (p >= 0.9).astype("int64")
    meta["limiar_xgb_sens6"] = t_sens
    meta["sens_alvo_corte6"] = float(sens_alvo)

    for modo in ("vigilancia", "confirmacao"):
        funil = pd.read_csv(
            ROOT / "data" / f"funil_gzcmd_{modo}.csv",
            dtype={"pair_id": str},
            usecols=["pair_id", "y", "final"],
            low_memory=False,
        )
        junta = merged[["pair_id", "y_true"]].merge(funil, on="pair_id", how="left", validate="one_to_one")
        if junta["final"].isna().any() or not (junta["y"].to_numpy() == junta["y_true"].to_numpy()).all():
            raise RuntimeError(f"funil {modo} não alinha com o extrato por pair_id")
        merged[f"cascata_v31_{modo}"] = (junta["final"].to_numpy() == "MATCH").astype("int64")
        merged[f"humano_{modo}"] = junta["final"].isin(["DEFERRED", "UNSURE"]).astype("int64").to_numpy()
    return merged


def wilson(num: int, den: int) -> tuple[float, float]:
    return qw3._wilson_ci(int(num), int(den), alpha=ALPHA)


def contagens(sub: pd.DataFrame, col: str) -> dict[str, int]:
    y = sub["y_true"].to_numpy()
    d = sub[col].to_numpy()
    return {
        "n_total": int(len(sub)),
        "n_true": int((y == 1).sum()),
        "n_pred_pos": int((d == 1).sum()),
        "tp": int(((y == 1) & (d == 1)).sum()),
        "fp": int(((y == 0) & (d == 1)).sum()),
        "fn": int(((y == 1) & (d == 0)).sum()),
        "tn": int(((y == 0) & (d == 0)).sum()),
    }


def rates(c: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for nome, num, den in (
        ("sens", c["tp"], c["tp"] + c["fn"]),
        ("ppv", c["tp"], c["tp"] + c["fp"]),
        ("espec", c["tn"], c["tn"] + c["fp"]),
    ):
        lo, hi = wilson(num, den)
        out[nome] = num / den if den > 0 else math.nan
        out[f"{nome}_low"] = lo
        out[f"{nome}_high"] = hi
    out["prevalencia"] = (c["tp"] + c["fn"]) / c["n_total"] if c["n_total"] > 0 else math.nan
    den_neg = c["tn"] + c["fp"]
    out["fpr"] = c["fp"] / den_neg if den_neg > 0 else math.nan
    return out


def elegiveis(frame: pd.DataFrame, dim: str) -> list[str]:
    cont = frame[dim].astype("string").value_counts(dropna=False)
    return sorted(str(g) for g, n in cont.items() if pd.notna(g) and str(g) != "MISSING" and int(n) >= N_MIN)


def analisar_decisao(frame: pd.DataFrame, decisao: str):
    work = frame.copy()
    work["y_pred"] = work[decisao]
    humano_col = None
    if decisao.startswith("cascata_v31_"):
        humano_col = "humano_" + decisao.removeprefix("cascata_v31_")

    glob = contagens(work, "y_pred")
    glob.update(rates(glob))
    glob["decisao"] = decisao
    glob["n_humano"] = int(work[humano_col].sum()) if humano_col else 0
    glob["n_humano_true"] = int(work.loc[work["y_true"] == 1, humano_col].sum()) if humano_col else 0

    grupos, testes = [], []
    for dim in DIMENSOES:
        gs = elegiveis(work, dim)
        sub_dim = work[work[dim].astype("string").isin(gs)]
        for g in gs:
            sub = sub_dim[sub_dim[dim].astype("string") == g]
            c = contagens(sub, "y_pred")
            row = {"decisao": decisao, "tipo": TIPO[dim], "dimension": dim, "group": g, **c, **rates(c)}
            row["n_humano"] = int(sub[humano_col].sum()) if humano_col else 0
            row["n_humano_true"] = int(sub.loc[sub["y_true"] == 1, humano_col].sum()) if humano_col else 0
            grupos.append(row)
        for metric in ("tpr", "ppv"):
            for g1, g2 in combinations(gs, 2):
                t = qw3._build_pairwise_test(sub_dim, dim, metric, g1, g2, alpha=ALPHA)
                t.update({"decisao": decisao, "tipo": TIPO[dim]})
                testes.append(t)

    tests = pd.DataFrame(testes)
    tests["p_value_fdr"] = qw3._benjamini_hochberg([float(v) for v in tests["p_value_raw"]], alpha=ALPHA)
    tests["reject_fdr"] = tests["p_value_fdr"].le(ALPHA)
    return pd.DataFrame([glob]), pd.DataFrame(grupos), tests


def diferencas_maximas(grupos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (decisao, dim), sub in grupos.groupby(["decisao", "dimension"], sort=False):
        for nome, den_fn in (
            ("sens", lambda r: r["tp"] + r["fn"]),
            ("ppv", lambda r: r["tp"] + r["fp"]),
            ("espec", lambda r: r["tn"] + r["fp"]),
        ):
            sub = sub.copy()
            sub["den"] = sub.apply(den_fn, axis=1)
            ok = sub[(sub["den"] >= N_MIN) & sub[nome].notna()]
            base = {
                "decisao": decisao,
                "tipo": TIPO[dim],
                "dimension": dim,
                "metric": nome,
                "n_grupos_elegiveis": int(len(sub)),
                "n_grupos_den_ge_30": int(len(ok)),
            }
            if len(ok) < 2:
                rows.append(base | {"max_diff": math.nan})
                continue
            alto = ok.loc[ok[nome].idxmax()]
            baixo = ok.loc[ok[nome].idxmin()]
            num = {"sens": "tp", "ppv": "tp", "espec": "tn"}[nome]
            diff, lo, hi = qw3._difference_ci(
                float(alto[nome]), int(alto["den"]), float(baixo[nome]), int(baixo["den"]), alpha=ALPHA
            )
            rows.append(
                base
                | {
                    "max_diff": diff,
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "grupo_alto": alto["group"],
                    "grupo_baixo": baixo["group"],
                    "rate_alto": float(alto[nome]),
                    "rate_baixo": float(baixo[nome]),
                    "den_alto": int(alto["den"]),
                    "den_baixo": int(baixo["den"]),
                    "num_col": num,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------------

NOMES_DIM = {
    "sexo": "Sexo",
    "faixa_idade": "Faixa et\\'aria",
    "nome_qtd_quartis": "Nome (quartis)",
    "bandeira": "Encerramento da notifica\\c{c}\\~ao",
    "municipio": "Munic\\'ipio",
    "passo": "Passo",
    "ano_obito": "Ano do \\'obito",
    "completude": "Completude",
}
NOMES_GRUPO = {
    "demais_situacoes": "Demais situa\\c{c}\\~oes",
    "encerrada_obito": "Encerrada por \\'obito",
    "capital_RJ": "Capital (RJ)",
    "outros_municipios": "Demais",
    "completo": "Completo",
    "incompleto": "Incompleto",
    "passo_6": "6",
    "passo_9": "9",
    "passo_11": "11",
    "0-17": "0 a 17",
    "18-39": "18 a 39",
    "40-59": "40 a 59",
    "60-79": "60 a 79",
    "80+": "80+",
}


def fmt_int(v: int) -> str:
    s = f"{int(v):,}".replace(",", ".")
    return s


def fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n.a."
    return f"{100 * v:.1f}".replace(".", "{,}")


def fmt_pp(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n.a."
    sinal = "$+$" if v >= 0 else "$-$"
    return sinal + f"{abs(100 * v):.1f}".replace(".", "{,}")


def fmt_p(v: float) -> str:
    if v < 0.001:
        return "$<$0{,}001"
    return f"{v:.3f}".replace(".", "{,}")


def fmt_or(v: float) -> str:
    if v is None or math.isnan(v):
        return "n.a."
    if v >= 100:
        return f"{v:.0f}".replace(".", "{,}")
    return f"{v:.2f}".replace(".", "{,}")


def nome_grupo(g: str) -> str:
    return NOMES_GRUPO.get(g, g)


def tabela_grupos(grupos: pd.DataFrame, path: Path) -> None:
    a = grupos[grupos["decisao"] == "xgb_sens6"].set_index(["dimension", "group"])
    b = grupos[grupos["decisao"] == "cascata_v31_vigilancia"].set_index(["dimension", "group"])
    linhas = []
    for tipo, dims in (("Demogr\\'aficos", DEMOGRAFICAS), ("Operacionais", OPERACIONAIS)):
        linhas.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Subgrupos {tipo.lower()}}}}} \\\\")
        for dim in dims:
            for (d, g), r in a.loc[a.index.get_level_values(0) == dim].iterrows():
                rb = b.loc[(d, g)]
                linhas.append(
                    f"{NOMES_DIM[dim]} & {nome_grupo(g)} & {fmt_int(r['n_total'])} & {fmt_int(r['n_true'])} & "
                    f"{fmt_pct(r['sens'])} & {fmt_pct(r['ppv'])} & {fmt_pct(rb['sens'])} & {fmt_pct(rb['ppv'])} \\\\"
                )
            linhas.append("\\midrule")
    linhas.pop()
    corpo = "\n".join(linhas)
    path.write_text(
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Sensibilidade e valor preditivo positivo (VPP), em porcentagem, por subgrupo: XGBoost com limiar de sensibilidade equivalente ao corte 6{,}0 e decis\\~ao autom\\'atica da cascata v3.1 (vigil\\^ancia).}\n"
        "\\label{tab:qw3-group-metrics}\n\\small\n"
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        " & & & & \\multicolumn{2}{c}{XGBoost} & \\multicolumn{2}{c}{Cascata v3.1} \\\\\n"
        "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        "Dimens\\~ao & Grupo & $n$ & Vínculos & Sens. & VPP & Sens. & VPP \\\\\n\\midrule\n"
        f"{corpo}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def tabela_rejeicoes(
    tests: pd.DataFrame,
    path: Path,
    decisao: str = "xgb_sens6",
    label: str = "tab:qw3-fairness-rejections",
    descr: str = "XGBoost com limiar de sensibilidade equivalente ao corte 6{,}0",
) -> None:
    t = tests[(tests["decisao"] == decisao) & (tests["reject_fdr"])].copy()
    t["ord"] = t["p_value_fdr"]
    t = t.sort_values(["tipo", "dimension", "metric", "ord"])
    linhas = []
    for _, r in t.iterrows():
        metrica = "Sens." if r["metric"] == "tpr" else "VPP"
        linhas.append(
            f"{NOMES_DIM[r['dimension']]} & {'Demogr.' if r['tipo'] == 'demografica' else 'Operac.'} & {metrica} & {nome_grupo(r['group_a'])} & "
            f"{nome_grupo(r['group_b'])} & {fmt_p(r['p_value_fdr'])} & {fmt_pp(r['rate_diff'])} & {fmt_or(r['odds_ratio'])} \\\\"
        )
    corpo = "\n".join(linhas) if linhas else "\\multicolumn{8}{c}{Nenhuma rejei\\c{c}\\~ao} \\\\"
    n_total = len(tests[tests["decisao"] == decisao])
    path.write_text(
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{Rejei\\c{{c}}\\~oes da hip\\'otese de paridade (Benjamini-Hochberg, $\\alpha = 0{{,}}05$, {n_total} testes): {descr}. $\\Delta$ em pontos percentuais (grupo A menos grupo B).}}\n"
        f"\\label{{{label}}}\n\\small\n"
        "\\begin{tabular}{lllllrrr}\n\\toprule\n"
        "Dimens\\~ao & Tipo & M\\'etrica & Grupo A & Grupo B & $p_{\\mathrm{FDR}}$ & $\\Delta$ & OR \\\\\n\\midrule\n"
        f"{corpo}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def tabela_resumo(glob: pd.DataFrame, maxd: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    rotulos = {
        "corte6": "Corte 6{,}0 da nota",
        "xgb_sens6": "XGBoost, sens. do corte",
        "xgb_p090": "XGBoost, $p \\geq 0{,}9$",
        "cascata_v31_vigilancia": "Cascata v3.1 (vigil.)",
        "cascata_v31_confirmacao": "Cascata v3.1 (confirm.)",
    }
    linhas = []
    for dec in DECISOES:
        g = glob[glob["decisao"] == dec].iloc[0]
        rej = tests[(tests["decisao"] == dec)]
        linhas.append(
            f"{rotulos[dec]} & {fmt_int(g['n_pred_pos'])} & {fmt_int(g['tp'])} & {fmt_int(g['fp'])} & {fmt_int(g['fn'])} & "
            f"{fmt_pct(g['sens'])} & {fmt_pct(g['ppv'])} & {fmt_pct(g['espec'])} & {int(rej['reject_fdr'].sum())}/{len(rej)} \\\\"
        )
    corpo_a = "\n".join(linhas)

    def cel(dec: str, dim: str, metric: str) -> str:
        r = maxd[(maxd["decisao"] == dec) & (maxd["dimension"] == dim) & (maxd["metric"] == metric)]
        if r.empty or pd.isna(r.iloc[0]["max_diff"]):
            return "n.a."
        return fmt_pct(float(r.iloc[0]["max_diff"]))

    linhas_b = []
    for tipo, dims in (("Demogr\\'aficos", DEMOGRAFICAS), ("Operacionais", OPERACIONAIS)):
        linhas_b.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Subgrupos {tipo.lower()}}}}} \\\\")
        for dim in dims:
            for metric, nm in (("sens", "Sens."), ("ppv", "VPP")):
                celulas = " & ".join(cel(dec, dim, metric) for dec in DECISOES)
                linhas_b.append(f"{NOMES_DIM[dim]} & {nm} & {celulas} \\\\")
        linhas_b.append("\\midrule")
    linhas_b.pop()
    corpo_b = "\n".join(linhas_b)
    path.write_text(
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Equidade por decis\\~ao: desempenho global (acima) e diferen\\c{c}a m\\'axima entre subgrupos, em pontos percentuais, considerando grupos com denominador $\\geq 30$ (abaixo). A \\'ultima coluna da parte superior conta rejei\\c{c}\\~oes/testes ap\\'os Benjamini-Hochberg.}\n"
        "\\label{tab:equidade-v31}\n\\footnotesize\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        "Decis\\~ao & Pred.$+$ & VP & FP & FN & Sens. & VPP & Espec. & Rejei\\c{c}\\~oes \\\\\n\\midrule\n"
        f"{corpo_a}\n\\bottomrule\n\\end{{tabular}}}}\n\\par\\medskip\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "\\multicolumn{7}{l}{\\textit{Diferen\\c{c}a m\\'axima entre subgrupos (pontos percentuais)}} \\\\\n\\midrule\n"
        "Dimens\\~ao & M\\'etrica & Corte 6{,}0 & XGB sens. & XGB $p \\geq 0{,}9$ & Cascata vig. & Cascata conf. \\\\\n\\midrule\n"
        f"{corpo_b}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", type=Path, default=ROOT / "data" / "COMPARADORSEMIDENT.csv")
    ap.add_argument("--qw3-dir", type=Path, default=ROOT / "data" / "qw3")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--tables-dir", type=Path, default=ROOT / "text" / "latex" / "tables")
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()

    frame, meta = build_frame(args.input_csv)
    if meta["raca_cor_presente"]:
        raise RuntimeError("coluna de raça/cor encontrada: revisar o capítulo")
    # C_SITUENCE 3 ou 4 (notificação encerrada por óbito) vs. demais situações
    frame["bandeira"] = frame["bandeira"].astype("string").replace(
        {"controle": "demais_situacoes", "tb_origem": "encerrada_obito"}
    )
    frame = attach_decisions(frame, meta)

    globs, grupos, testes = [], [], []
    for dec in DECISOES:
        g, gr, te = analisar_decisao(frame, dec)
        globs.append(g)
        grupos.append(gr)
        testes.append(te)
    glob = pd.concat(globs, ignore_index=True)
    grupos_df = pd.concat(grupos, ignore_index=True)
    testes_df = pd.concat(testes, ignore_index=True)
    maxd = diferencas_maximas(grupos_df)

    cols_teste = [
        "decisao", "tipo", "dimension", "metric", "group_a", "group_b", "test", "n1", "n2", "num_a", "num_b",
        "p_value_raw", "p_value_fdr", "reject_fdr", "odds_ratio", "or_ci_low", "or_ci_high",
        "rate_a", "rate_b", "rate_diff", "rate_diff_ci_low", "rate_diff_ci_high",
    ]
    testes_df = testes_df[cols_teste]

    saidas = {
        "global": glob,
        "group_metrics": grupos_df,
        "pairwise_tests": testes_df,
        "maxdiff": maxd,
    }
    args.qw3_dir.mkdir(parents=True, exist_ok=True)
    for nome, df in saidas.items():
        df.to_csv(args.qw3_dir / f"equidade_xgb_{nome}.csv", index=False)
        df.to_csv(args.data_dir / f"equidade_v31_{nome}.csv", index=False)

    manifest = {
        "analysis_name": "estudo5_equidade_v31",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "seed": "n/a (sem aleatoriedade; OOF e funil lidos de artefatos)",
        "input_csv": "data/COMPARADORSEMIDENT.csv (nao versionado)",
        "oof": "data/pcal_ml_oof.csv",
        "funil": ["data/funil_gzcmd_vigilancia.csv", "data/funil_gzcmd_confirmacao.csv"],
        "decisoes": DECISOES,
        "limiar_xgb_sens6": meta["limiar_xgb_sens6"],
        "sens_alvo_corte6": meta["sens_alvo_corte6"],
        "limiar_xgb_p090": 0.9,
        "raca_cor_presente": meta["raca_cor_presente"],
        "alpha": ALPHA,
        "n_min": N_MIN,
        "dimensoes_demograficas": DEMOGRAFICAS,
        "dimensoes_operacionais": OPERACIONAIS,
        "rows": meta["rows"],
    }
    (args.qw3_dir / "equidade_xgb_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.data_dir / "equidade_v31_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_tables:
        tabela_grupos(grupos_df, args.tables_dir / "tab_qw3_group_metrics.tex")
        tabela_rejeicoes(testes_df, args.tables_dir / "tab_qw3_fairness_rejections.tex")
        tabela_rejeicoes(
            testes_df,
            args.tables_dir / "tab_equidade_v31_rejeicoes_cascata.tex",
            decisao="cascata_v31_vigilancia",
            label="tab:equidade-v31-rejeicoes-cascata",
            descr="cascata v3.1 (vigil\\^ancia), aceitação automática",
        )
        tabela_resumo(glob, maxd, testes_df, args.tables_dir / "tab_equidade_v31.tex")

    pd.set_option("display.width", 250)
    print(glob[["decisao", "n_pred_pos", "tp", "fp", "fn", "tn", "sens", "ppv", "espec", "n_humano", "n_humano_true"]].to_string())
    print(json.dumps({k: manifest[k] for k in ("limiar_xgb_sens6", "sens_alvo_corte6")}))


if __name__ == "__main__":
    main()
