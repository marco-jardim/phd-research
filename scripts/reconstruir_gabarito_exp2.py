"""Reconstrói o gabarito (VP/FP/FN/VN) da revisão assistida por LLM (Exp2) a partir do
artefato persistido da execução e do padrão-ouro do extrato sem identificadores.

O artefato data/sprint3b/exp2_llm_sync_*.json registra, para cada par revisado, o
`pair_id` (SHA-256 de ``COMPREC|REFREC|PASSO``, truncado a 16 caracteres) e a decisão
(MATCH / NONMATCH / UNSURE). O `pair_id` é recomputado sobre data/COMPARADORSEMIDENT.csv
e cruzado com a coluna PAR (1 ou 2 = vínculo verdadeiro).

Uso:
    python scripts/reconstruir_gabarito_exp2.py [--json data/sprint3b/exp2_llm_sync_vigilancia.json]
Saídas:
    data/exp2_gabarito_reconstruido.csv       matriz decisão x rótulo
    data/exp2_gabarito_por_par.csv            pair_id, decisão, confiança, PAR, nota final
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if c == name or c.startswith(name + ","):
            return c
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "data" / "sprint3b" / "exp2_llm_sync_vigilancia.json"))
    ap.add_argument("--csv", default=str(ROOT / "data" / "COMPARADORSEMIDENT.csv"))
    args = ap.parse_args()

    run = json.loads(Path(args.json).read_text(encoding="utf-8"))
    res = pd.DataFrame([{"pair_id": r["pair_id"], "decision": r["decision"], "confidence": r.get("confidence")}
                        for r in run["results"]])
    print(f"artefato: {Path(args.json).name} | modelo={run.get('model')} | revisados={len(res)} | "
          f"decisões={res.decision.value_counts().to_dict()}")

    df = pd.read_csv(args.csv, sep=";", low_memory=False, decimal=",", dtype=str)
    c_comp, c_ref, c_passo = resolve(df, "COMPREC"), resolve(df, "REFREC"), resolve(df, "PASSO")
    df["pair_id"] = [hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
                     for a, b, c in zip(df[c_comp].str.strip(), df[c_ref].str.strip(), df[c_passo].str.strip())]
    assert df["pair_id"].is_unique, "pair_id não é único no extrato"
    df["PAR"] = pd.to_numeric(df["PAR"], errors="coerce").fillna(0).astype(int)
    df["y"] = df["PAR"].isin([1, 2]).astype(int)
    df["nota_final"] = pd.to_numeric(df[resolve(df, "nota final")].str.replace(",", "."), errors="coerce")

    m = res.merge(df[["pair_id", "PAR", "y", "nota_final"]], on="pair_id", how="left")
    n_miss = int(m["y"].isna().sum())
    print(f"pareados com o extrato: {len(m) - n_miss}/{len(m)} (não encontrados: {n_miss})")
    m["y"] = m["y"].fillna(-1).astype(int)

    tab = pd.crosstab(m["decision"], m["y"].map({1: "vínculo", 0: "não vínculo", -1: "sem rótulo"}), margins=True)
    print("\nmatriz decisão x padrão-ouro:\n", tab)

    vp = int(((m.decision == "MATCH") & (m.y == 1)).sum()); fp = int(((m.decision == "MATCH") & (m.y == 0)).sum())
    fn = int(((m.decision == "NONMATCH") & (m.y == 1)).sum()); vn = int(((m.decision == "NONMATCH") & (m.y == 0)).sum())
    u1 = int(((m.decision == "UNSURE") & (m.y == 1)).sum()); u0 = int(((m.decision == "UNSURE") & (m.y == 0)).sum())
    resolved = vp + fp + fn + vn
    print(f"\nresolvidos={resolved}: VP={vp} FP={fp} FN={fn} VN={vn} | acurácia={(vp+vn)/resolved:.4f} "
          f"precisão={vp/(vp+fp):.4f} sensibilidade={vp/(vp+fn):.4f}")
    print(f"inconclusivos={u1+u0}: vínculos={u1} não vínculos={u0} | prevalência={u1/(u1+u0):.3f}")
    print(f"vínculos verdadeiros entre os {len(m)} revisados: {vp+fn+u1}")
    print("nota final dos revisados: min=%.2f max=%.2f" % (m.nota_final.min(), m.nota_final.max()))

    out = ROOT / "data"
    tab.to_csv(out / "exp2_gabarito_reconstruido.csv")
    m.to_csv(out / "exp2_gabarito_por_par.csv", index=False)


if __name__ == "__main__":
    main()
