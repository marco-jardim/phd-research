"""A figura de importância global do XGBoost tem fonte versionada e coerente."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "xgboost_feature_importance_global.csv"
META = ROOT / "data" / "xgboost_feature_importance_global_metadata.json"
PGF = ROOT / "text" / "latex" / "fig_doc" / "feature_importance_heatmap.pgf"
TEX = ROOT / "text" / "latex" / "estudo1_comparador.tex"


def _rows():
    with CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_tem_so_atributo_e_importancia():
    rows = _rows()
    assert list(rows[0].keys()) == ["feature", "importance_gain"]
    assert len(rows) == 58
    values = [float(r["importance_gain"]) for r in rows]
    assert values == sorted(values, reverse=True)


def test_metadata_confere_com_o_arquivo():
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert meta["entrada_bytes"] == len(CSV.read_bytes())
    assert "23948459" in meta["origem"]


def test_rotulos_da_figura_sao_os_15_primeiros_do_csv_na_ordem():
    pgf = PGF.read_text(encoding="utf-8")
    labels = re.findall(r"\\pgftext\[x=[^,]+, y=([\d.]+)in, left, base\]\{.*?\\def%\{\\%\}([^{}]+)\}\}%", pgf)
    nomes = [nome.replace(r"\_", "_") for _, nome in sorted(labels, key=lambda t: -float(t[0]))]
    esperados = [r["feature"] for r in _rows()[:15]]
    assert [n for n in nomes if n in esperados] == esperados


def test_legenda_cita_a_fonte():
    assert r"\protect\path{data/xgboost_feature_importance_global.csv}" in TEX.read_text(encoding="utf-8")
