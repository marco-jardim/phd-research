"""Sensitivity of row-OOF v3.1 selection to fixed reliability in unmeasured bands."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.config import load_config
from gzcmd.reliability import load_band_rates
from scripts.auditar_funil_adversarial import engine_run, endpoint_projection
from scripts.funil_gzcmd_v2 import HAIKU_V4, load_responses, replay


def main():
    cfg = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml")
    rows = []
    for mode, band, neighbor in (("vigilancia", "grey_low", "grey_mid"), ("confirmacao", "grey_mid", "grey_high")):
        frame = pd.read_csv(ROOT / f"data/funil_gzcmd_{mode}.csv", low_memory=False).set_index("pair_id")
        frame["TARGET"] = frame.y
        frame["nota final"] = frame.nota
        rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode,
                                source=ROOT / "gzcmd/error_rates_by_band_v4.yaml")
        baseline, costs, human = engine_run(frame, cfg, mode, rates)
        assert baseline.action.equals(frame.action)
        responses, _ = load_responses(mode, HAIKU_V4, include_v31=True)
        old_ids, _ = load_responses(mode, HAIKU_V4)
        assert not set(old_ids).intersection(frame.index[frame.band.eq(band)])
        for scenario in ("fixas", "vizinha_e_fp_e_fn", "vizinha_todas_taxas"):
            used = dict(rates)
            if scenario == "vizinha_e_fp_e_fn":
                used[band] = replace(rates[neighbor], e_unsure=rates[band].e_unsure)
            elif scenario == "vizinha_todas_taxas":
                used[band] = rates[neighbor]
            out, _, _ = engine_run(frame, cfg, mode, used)
            b_ids = set(baseline.index[baseline.action.eq("LLM_REVIEW")])
            selected = set(out.index[out.action.eq("LLM_REVIEW")])
            evaluated = replay(out, responses, set(out.index))
            rows.append({"modo": mode, "cenario": scenario, "banda_sem_dado": band, "banda_vizinha": neighbor,
                         "n_respostas_haiku_banda": 0, "llm_banda_original": int((baseline.action.eq("LLM_REVIEW") & baseline.band.eq(band)).sum()),
                         "lote_llm_original": len(b_ids), "lote_llm": len(selected),
                         "llm_adicionados": len(selected - b_ids), "llm_removidos": len(b_ids - selected),
                         "acoes_alteradas": int(out.action.ne(baseline.action).sum()),
                         "selecao_identica": selected == b_ids, "e_fp": used[band].e_fp, "e_fn": used[band].e_fn,
                         "u": used[band].e_unsure, **endpoint_projection(evaluated, costs, used, human)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "data/funil_gzcmd_bandas_sem_dado.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
