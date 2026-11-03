"""Funil de ponta a ponta do GZ-CMD (run_v3 real: calibração, guardas, motor de custo, orçamento)
cruzado com o padrão-ouro e, quando disponível, com as decisões do revisor LLM.

Para cada modo, por faixa de nota final e por etapa, conta pares e vínculos verdadeiros e
registra a ação final. Responde: quantos vínculos da área cinza (5 <= s < 9) o framework perde
antes do revisor humano, e onde.

Uso:
    python scripts/funil_gzcmd.py [--llm-vigilancia data/sprint3b/exp2_llm_sync_vigilancia.json]
                                  [--llm-confirmacao data/sprint3b/opencode_go/exp2_llm_sync_confirmacao.json]
Saídas:
    data/funil_gzcmd_<modo>.csv          por par: nota, faixa, y, guardrail, p_cal, evr, action, llm_decision, final
    data/funil_gzcmd_resumo.csv          por modo x faixa x etapa: n e vínculos
    text/latex/tables/tab_cap8_funil.tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2_llm_sync as exp2  # noqa: E402
from gzcmd.runner import run_v3  # noqa: E402

FAIXAS = [0, 5, 9, 10, 99]
LABELS = ["[0,5)", "[5,9)", "[9,10)", "[10,+)"]


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def funil(mode: str, llm_json: str | None) -> pd.DataFrame:
    df, _ = run_v3(input_csv=str(exp2.DATA_CSV), config_path=str(exp2.CONFIG_YAML), mode=mode, p_cal="fit_platt", policy_version="v3")
    col = lambda n: next(c for c in df.columns if c == n or c.startswith(n + ","))
    nota = pd.to_numeric(df[col("nota_final")] if any(c.startswith("nota_final") for c in df.columns) else df[col("nota final")], errors="coerce")
    y = pd.to_numeric(df["PAR"], errors="coerce").fillna(0).astype(int).isin([1, 2]).astype(int)
    pid = [hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
           for a, b, c in zip(df[col("COMPREC")].astype(str).str.strip(), df[col("REFREC")].astype(str).str.strip(), df[col("PASSO")].astype(str).str.strip())]
    out = pd.DataFrame({"pair_id": pid, "nota": nota.to_numpy(), "faixa": pd.cut(nota, FAIXAS, right=False, labels=LABELS).astype(str),
                        "y": y.to_numpy(), "guardrail": df["guardrail"].fillna("none").to_numpy(), "p_cal": df["p_cal"].to_numpy(),
                        "evr": df["evr"].to_numpy(), "base_choice": df["base_choice"].to_numpy(), "action": df["action"].to_numpy()}).set_index("pair_id")
    out["llm_decision"] = np.nan
    if llm_json and Path(llm_json).exists():
        r = json.loads(Path(llm_json).read_text(encoding="utf-8"))
        dec = {x["pair_id"]: x["decision"] for x in r["results"]}
        out["llm_decision"] = out.index.map(dec)
    out["final"] = np.where(out.action == "LLM_REVIEW", out.llm_decision.fillna("LLM_PENDENTE"), out.action)
    out["final"] = out["final"].replace({"MATCH": "MATCH", "NONMATCH": "NONMATCH", "UNSURE": "HUMANO"})
    return out


def resumo(out: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for fx in LABELS:
        s = out[out.faixa == fx]
        g = s[s.guardrail != "none"]
        rows.append({"modo": mode, "faixa": fx, "n": len(s), "vinc": int(s.y.sum()),
                     "guarda_nonmatch_n": int((s.guardrail == "ALWAYS_NONMATCH").sum()), "guarda_nonmatch_vinc": int(s.y[s.guardrail == "ALWAYS_NONMATCH"].sum()),
                     "guarda_match_n": int((s.guardrail == "ALWAYS_MATCH").sum()),
                     "auto_nonmatch_n": int((s.action == "NONMATCH").sum()), "auto_nonmatch_vinc": int(s.y[s.action == "NONMATCH"].sum()),
                     "auto_match_n": int((s.action == "MATCH").sum()), "auto_match_naovinc": int((1 - s.y[s.action == "MATCH"]).sum()),
                     "llm_n": int((s.action == "LLM_REVIEW").sum()), "llm_vinc": int(s.y[s.action == "LLM_REVIEW"].sum()),
                     "llm_match_vinc": int(s.y[s.llm_decision == "MATCH"].sum()), "llm_match_naovinc": int((1 - s.y[s.llm_decision == "MATCH"]).sum()),
                     "llm_nonmatch_vinc": int(s.y[s.llm_decision == "NONMATCH"].sum()),
                     "humano_n": int((s.llm_decision == "UNSURE").sum()), "humano_vinc": int(s.y[s.llm_decision == "UNSURE"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-vigilancia", default=str(ROOT / "data/sprint3b/exp2_llm_sync_vigilancia.json"))
    ap.add_argument("--llm-confirmacao", default=str(ROOT / "data/sprint3b/opencode_go/exp2_llm_sync_confirmacao.json"))
    a = ap.parse_args()
    res = []
    for mode, lj in (("vigilancia", a.llm_vigilancia), ("confirmacao", a.llm_confirmacao)):
        out = funil(mode, lj)
        out.to_csv(ROOT / "data" / f"funil_gzcmd_{mode}.csv")
        r = resumo(out, mode); res.append(r)
        print(f"\n===== {mode} (LLM: {Path(lj).name if lj else 'nenhum'})")
        print(r.drop(columns=["modo"]).to_string(index=False))
        cz = out[out.faixa == "[5,9)"]; n_v = int(cz.y.sum())
        enc = int(cz.y[cz.action == "LLM_REVIEW"].sum()); vp = int(cz.y[cz.llm_decision == "MATCH"].sum()); hum = int(cz.y[cz.llm_decision == "UNSURE"].sum())
        print(f"ÁREA CINZA [5,9): vínculos={n_v} | descartados automaticamente antes do LLM={n_v-enc} ({(n_v-enc)/n_v:.1%}) | encaminhados ao LLM={enc}"
              f" | LLM MATCH={vp} | LLM UNSURE (humano)={hum} | LLM NONMATCH={enc-vp-hum}")
        print(f"   sensibilidade ponta a ponta (automática)={vp/n_v:.1%} | com humano perfeito nos UNSURE={(vp+hum)/n_v:.1%}")
        tot_v = int(out.y.sum()); vp_all = int(out.y[(out.final == "MATCH")].sum()); hum_all = int(out.y[out.final == "HUMANO"].sum())
        fp_all = int((1 - out.y[out.final == "MATCH"]).sum())
        print(f"BASE COMPLETA: vínculos={tot_v} | MATCH final={vp_all} ({vp_all/tot_v:.1%}) | humano={hum_all} | FP final={fp_all}"
              f" | revisões humanas={int((out.final=='HUMANO').sum())}")
    R = pd.concat(res); R.to_csv(ROOT / "data" / "funil_gzcmd_resumo.csv", index=False)


if __name__ == "__main__":
    main()
