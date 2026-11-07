"""Preserve row outputs and extract unreviewed REFREC pairs, without labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.ml_oof import fit_xgb_oof
from gzcmd.runner import run_v3
from scripts.funil_gzcmd_v2 import HAIKU_V4, load_responses, safe_funnel


def preserve_row_outputs():
    manifest = ROOT / "data/arquivos_preservados_row.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        for row in saved:
            assert hashlib.sha256((ROOT / row["archive"]).read_bytes()).hexdigest() == row["sha256"]
        return saved
    paths = set()
    for pattern in ("funil_gzcmd*.csv", "funil_gzcmd*.json", "funil_gzcmd*.md", "perda_total_por_modelo.csv",
                    "varredura*.csv", "varredura*.json", "varredura*.md", "np_confirmacao_relatorio.md",
                    "comparacao_np_confirmacao.csv", "variante_guarda_confirmacao.csv", "exp2_reruns*.csv",
                    "lote_llm_v31*.csv", "pcal_ml_oof.csv", "pcal_ml_oof.json"):
        paths.update((ROOT / "data").glob(pattern))
    for stem in ("fig_cap8_varredura_custos", "fig_cap8_np_confirmacao", "fig_cap8_cascata_v31"):
        paths.update((ROOT / "text/latex/figures").glob(stem + ".*"))
    for name in ("tab_cap8_reruns.tex", "tab_cap8_cascata_v31.tex"):
        paths.add(ROOT / "text/latex/tables" / name)
    saved = []
    for source in sorted(paths):
        if any(tag in source.stem for tag in ("_row", "_m3v3", "oof_grupo")):
            continue
        target = source.with_name(source.stem + "_row" + source.suffix)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Different row archive already exists: {target.name}")
        shutil.copy2(source, target)
        saved.append({"source": source.relative_to(ROOT).as_posix(), "archive": target.relative_to(ROOT).as_posix(), "sha256": digest})
    original = ROOT / "gzcmd/error_rates_by_band_v4.yaml"
    backup = ROOT / "gzcmd/error_rates_by_band_v4_lote_antigo.yaml"
    if backup.exists() and backup.read_bytes() != original.read_bytes():
        raise ValueError("Different previous-rates snapshot exists")
    shutil.copy2(original, backup)
    manifest.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved


def main():
    saved = preserve_row_outputs()
    source = ROOT / "data/COMPARADORSEMIDENT.csv"
    # Existing REFREC cache is verified by the fitter before promoting it.
    cached = ROOT / "data/pcal_ml_oof_refrec.csv"
    fit_xgb_oof(source, output_path=cached, reuse=True, split_by="refrec")
    for suffix in (".csv", ".json"):
        shutil.copy2(cached.with_suffix(suffix), (ROOT / "data/pcal_ml_oof.csv").with_suffix(suffix))
    rates_path = ROOT / "gzcmd/error_rates_by_band_v4_lote_antigo.yaml"
    profile = yaml.safe_load(rates_path.read_text(encoding="utf-8"))
    for mode, expected in (("vigilancia", 155), ("confirmacao", 17)):
        out, _ = run_v3(input_csv=source, config_path=ROOT / "gzcmd/gzcmd_v3_config.yaml", mode=mode,
                         p_cal="ml_xgb", policy_version="v3.1", reuse_ml_oof=True, policy_kind="cost", rates_yaml=rates_path)
        frame = safe_funnel(out)
        responses, _ = load_responses(mode, HAIKU_V4, include_v31=True)
        pending = frame.loc[frame.action.eq("LLM_REVIEW") & ~frame.index.isin(responses),
                            ["nota", "p_cal", "band", "decisao_motor"]].copy()
        empty_bands = {b for b, row in profile["by_mode"][mode]["models"]["claude-haiku-5-5"]["estimates_by_band"].items() if row["n"] == 0}
        pending["motivo"] = pending.band.map(lambda b: "banda_sem_dado" if b in empty_bands else "sem_resposta")
        pending = pending.rename(columns={"p_cal": "p_cal_ml", "band": "banda"}).reset_index()
        if len(pending) != expected:
            raise ValueError(f"Unexpected initial REFREC pending cohort: {mode}/{len(pending)}")
        path = ROOT / f"data/lote_llm_v31_refrec_{mode}_pendentes.csv"
        if path.exists() and path.read_text(encoding="utf-8") != pending.to_csv(index=False):
            raise ValueError("Refusing to change an already exported cohort")
        pending.to_csv(path, index=False)
        print("REFREC_PENDING", mode, len(pending), "BANDA_SEM_DADO", int(pending.motivo.eq("banda_sem_dado").sum()))
    print("ROW_ARCHIVES", len(saved))


if __name__ == "__main__":
    main()
