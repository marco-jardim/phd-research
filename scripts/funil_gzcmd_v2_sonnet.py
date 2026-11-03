"""Cascata v3.1 com Claude Sonnet 5.5 como revisor LLM, lado a lado com o Haiku v4.

Reutiliza as funções de scripts/funil_gzcmd_v2.py (sem alterá-lo) e as taxas de erro por
banda do Haiku (que definem a seleção do lote LLM; nada é reestimado). Escreve somente
arquivos novos `data/funil_sonnet_v4_*`; não toca nos artefatos `funil_gzcmd_*`.

Uso:
  python scripts/funil_gzcmd_v2_sonnet.py --listar-faltantes   # lote v3.1 sem resposta do Sonnet
  python scripts/funil_gzcmd_v2_sonnet.py                       # cascata e métricas
Sem chamadas LLM, sem PII (somente agregados).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gzcmd.config import load_config  # noqa: E402
from gzcmd.final_funnel import acceptance_metrics  # noqa: E402
from gzcmd.intervals import wilson95  # noqa: E402
from gzcmd.runner import run_v3  # noqa: E402
import funil_gzcmd_v2 as funil  # noqa: E402

SONNET_DIRS = ("anthropic_sonnet_sem_mo_v4", "anthropic_sonnet_sem_mo_v4_lote31_refrec",
               "anthropic_sonnet_sem_mo_v4_lote31_complemento")
MODES = ("vigilancia", "confirmacao")


def sonnet_responses(mode: str, *, include_v31: bool) -> tuple[dict[str, str], list[dict]]:
    decisions, sources = {}, []
    for position, directory in enumerate(SONNET_DIRS):
        if position and not include_v31:
            break
        path = ROOT / "data/sprint3b" / directory / f"exp2_llm_sync_{mode}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["mode"] == mode and run["model"] == "claude-sonnet-5-5" and run["prompt_version"] == "v4"
        for row in run["results"]:
            if row["pair_id"] in decisions:
                raise ValueError(f"Resposta duplicada em {directory}")
            decisions[row["pair_id"]] = row["decision"].strip().upper()
        sources.append({"dir": directory, "n": len(run["results"]), "complete": bool(run.get("complete")),
                        "cost_usd": run.get("cost_estimate_usd")})
    return decisions, sources


def build(mode: str, config_path: Path, input_csv: Path, rates_path: Path):
    old_df, _ = run_v3(input_csv=input_csv, config_path=config_path, mode=mode, p_cal="platt", policy_version="v3")
    new_df, _ = run_v3(input_csv=input_csv, config_path=config_path, mode=mode, p_cal="ml_xgb",
                       policy_version="v3.1", ml_oof_path=ROOT / "data/pcal_ml_oof.csv", reuse_ml_oof=True,
                       policy_kind="cost", rates_yaml=rates_path)
    old, new = funil.safe_funnel(old_df), funil.safe_funnel(new_df)
    return old, new, set(old.index[old.action.eq("LLM_REVIEW")]), set(new.index[new.action.eq("LLM_REVIEW")])


def cascade_row(mode, version, reviewer, frame, responses, batch):
    replayed = funil.replay(frame, responses, batch)
    missing = int(replayed.final.eq("SEM_RESPOSTA_LLM").sum())
    if missing:
        raise ValueError(f"{reviewer}/{mode}/{version}: {missing} pares do lote sem resposta")
    m = acceptance_metrics(replayed)
    vinc = m["VP_cadeia"] + m["FN_cadeia"]
    low, high = wilson95(m["VP_cadeia"], vinc)
    llm = replayed.loc[replayed.action.eq("LLM_REVIEW")]
    return {"modo": mode, "versao": version, "revisor": reviewer, "pares_extrato": len(frame),
            "vinculos": int(frame.y.sum()), "pares_llm": len(llm), "vinculos_no_lote_llm": int(llm.y.sum()),
            "VP": m["VP_cadeia"], "FP": m["FP_cadeia"], "perdidos_FN_nonmatch": m["FN_nonmatch_observado"],
            "humano_pares": m["pares_ao_humano"], "humano_vinculos": m["vinculos_ao_humano"],
            "humano_nao_vinculos": m["nao_vinculos_ao_humano"], "humano_direto": m["humano_direto"],
            "humano_unsure": m["humano_unsure"],
            "sens_automatica": m["sens_cadeia_observada"], "sens_auto_ic95_low": low, "sens_auto_ic95_high": high,
            "sens_com_humano_erro_05": (m["VP_cadeia"] + .95 * m["vinculos_ao_humano"]) / vinc,
            "espec_aceitacao_automatica": m["espec_cadeia_observada"],
            "llm_match": int(llm.llm_decision.eq("MATCH").sum()),
            "llm_nonmatch": int(llm.llm_decision.eq("NONMATCH").sum()),
            "llm_unsure": int(llm.llm_decision.eq("UNSURE").sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listar-faltantes", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "gzcmd/gzcmd_v3_config.yaml")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rates_path = Path(cfg.llm_review.estimated_rates_path)
    rows, coverage = [], []
    for mode in MODES:
        old, new, old_batch, new_batch = build(mode, args.config, args.input_csv, rates_path)
        son_old, _ = sonnet_responses(mode, include_v31=False)
        son_all, sources = sonnet_responses(mode, include_v31=True)
        assert set(son_old) == old_batch, f"coorte antiga do Sonnet incompleta em {mode}"
        missing = sorted(new_batch - set(son_all))
        coverage.append({"modo": mode, "lote_v3.1": len(new_batch), "lote_antigo": len(old_batch),
                         "novos_no_lote": len(new_batch - old_batch),
                         "com_resposta_sonnet": len(new_batch & set(son_all)),
                         "faltantes_sonnet": len(missing), "fontes": sources})
        if args.listar_faltantes:
            out = ROOT / f"data/lote_llm_v31_sonnet_complemento_{mode}_pendentes.csv"
            pd.DataFrame({"pair_id": missing}).to_csv(out, index=False)
            print("FALTANTES", mode, len(missing), out.name, flush=True)
            continue
        haiku_old, _ = funil.load_responses(mode, funil.HAIKU_V4)
        haiku_all, _ = funil.load_responses(mode, funil.HAIKU_V4, include_v31=True)
        for reviewer, old_resp, all_resp in (("Claude Sonnet 5.5 v4 sem MO", son_old, son_all),
                                            ("Claude Haiku 5.5 v4 sem MO", haiku_old, haiku_all)):
            rows.append(cascade_row(mode, "v3", reviewer, old, old_resp, old_batch))
            rows.append(cascade_row(mode, "v3.1", reviewer, new, all_resp, old_batch | new_batch))
    if args.listar_faltantes:
        print(json.dumps(coverage, ensure_ascii=False))
        return
    pd.DataFrame(rows).to_csv(ROOT / "data/funil_sonnet_v4_cascata.csv", index=False)
    (ROOT / "data/funil_sonnet_v4_cobertura.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2),
                                                              encoding="utf-8")
    print(pd.DataFrame(rows).T.to_string())
    print(json.dumps(coverage, ensure_ascii=False))


if __name__ == "__main__":
    main()

