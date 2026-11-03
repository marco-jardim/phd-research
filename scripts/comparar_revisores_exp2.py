"""Compara dois revisores LLM (Exp2) par a par contra o padrão-ouro.

Uso:
    python scripts/comparar_revisores_exp2.py \
        --a data/sprint3b/exp2_llm_sync_vigilancia.json --nome-a "Kimi K2.5" \
        --b data/sprint3b/opencode_go/exp2_llm_sync_vigilancia.json --nome-b "DeepSeek V4.1 Flash"
Saída: data/exp2_comparacao_revisores_<modo>.csv e impressão das métricas e da concordância.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]


def load_run(p: Path, tag: str) -> pd.DataFrame:
    r = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame([{"pair_id": x["pair_id"], f"dec_{tag}": x["decision"], f"conf_{tag}": x.get("confidence")}
                       for x in r["results"]])
    return df.set_index("pair_id"), r.get("mode"), r.get("model")


def gold() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "COMPARADORSEMIDENT.csv", sep=";", low_memory=False, decimal=",", dtype=str)
    col = lambda n: next(c for c in df.columns if c == n or c.startswith(n + ","))
    pid = [hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
           for a, b, c in zip(df[col("COMPREC")].str.strip(), df[col("REFREC")].str.strip(), df[col("PASSO")].str.strip())]
    y = pd.to_numeric(df["PAR"], errors="coerce").fillna(0).astype(int).isin([1, 2]).astype(int)
    nota = pd.to_numeric(df[col("nota final")].str.replace(",", "."), errors="coerce")
    return pd.DataFrame({"pair_id": pid, "y": y.to_numpy(), "nota_final": nota.to_numpy()}).set_index("pair_id")


def metrics(dec: pd.Series, y: pd.Series) -> dict:
    vp = int(((dec == "MATCH") & (y == 1)).sum()); fp = int(((dec == "MATCH") & (y == 0)).sum())
    fn = int(((dec == "NONMATCH") & (y == 1)).sum()); vn = int(((dec == "NONMATCH") & (y == 0)).sum())
    u1 = int(((dec == "UNSURE") & (y == 1)).sum()); u0 = int(((dec == "UNSURE") & (y == 0)).sum())
    return {"VP": vp, "FP": fp, "FN": fn, "VN": vn, "UNSURE": u1 + u0, "UNSURE_pos": u1,
            "acuracia": round((vp + vn) / max(vp + fp + fn + vn, 1), 4), "precisao": round(vp / max(vp + fp, 1), 4),
            "sensibilidade": round(vp / max(vp + fn, 1), 4),
            "sens_incl_unsure_como_vinculo": round((vp + u1) / max(vp + fn + u1, 1), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--nome-a", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--nome-b", default="B")
    args = ap.parse_args()
    a, mode_a, model_a = load_run(Path(args.a), "a"); b, mode_b, model_b = load_run(Path(args.b), "b")
    g = gold()
    d = a.join(b, how="outer").join(g, how="left")
    print(f"A = {args.nome_a} ({model_a}, {mode_a}) n={len(a)} | B = {args.nome_b} ({model_b}, {mode_b}) n={len(b)}")
    print(f"pares em comum: {int(d.dec_a.notna().sum() & d.dec_b.notna().sum() if False else (d.dec_a.notna() & d.dec_b.notna()).sum())} | só A: {int((d.dec_a.notna() & d.dec_b.isna()).sum())} | só B: {int((d.dec_a.isna() & d.dec_b.notna()).sum())}")
    both = d[d.dec_a.notna() & d.dec_b.notna()]
    print("\nmétricas contra o padrão-ouro (pares em comum):")
    for tag, nome in (("a", args.nome_a), ("b", args.nome_b)):
        print(f"  {nome:22s}", metrics(both[f"dec_{tag}"], both.y))
    print(f"\nconcordância bruta: {(both.dec_a == both.dec_b).mean():.4f} | kappa de Cohen: {cohen_kappa_score(both.dec_a, both.dec_b):.4f}")
    print("tabela cruzada (linhas A, colunas B):\n", pd.crosstab(both.dec_a, both.dec_b))
    fn_a = set(both[(both.dec_a == "NONMATCH") & (both.y == 1)].index)
    fn_b = set(both[(both.dec_b == "NONMATCH") & (both.y == 1)].index)
    print(f"\nfalsos negativos: A={len(fn_a)} B={len(fn_b)} em comum={len(fn_a & fn_b)} só A={len(fn_a - fn_b)} só B={len(fn_b - fn_a)}")
    vinc = both[both.y == 1]
    print("vínculos verdadeiros (n=%d): ambos MATCH=%d | A MATCH e B não=%d | B MATCH e A não=%d | nenhum MATCH=%d" % (
        len(vinc), ((vinc.dec_a == "MATCH") & (vinc.dec_b == "MATCH")).sum(), ((vinc.dec_a == "MATCH") & (vinc.dec_b != "MATCH")).sum(),
        ((vinc.dec_b == "MATCH") & (vinc.dec_a != "MATCH")).sum(), ((vinc.dec_a != "MATCH") & (vinc.dec_b != "MATCH")).sum()))
    out = ROOT / "data" / f"exp2_comparacao_revisores_{mode_a or 'modo'}.csv"
    d.to_csv(out); print("salvo:", out)


if __name__ == "__main__":
    main()
