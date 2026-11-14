import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts.comparador_p_match_humano import generate
from scripts.esforco_arranjo import generate as effort
from scripts.tabelas_s8_editorial import same_load

ROOT = Path(__file__).resolve().parents[1]


def test_comparator_regenerates_all_outputs_without_private_data(tmp_path):
    table = generate(output=tmp_path)
    assert len(table) == 2
    for name in ("comparador_p_match_humano.csv", "comparador_p_match_humano_metadata.json",
                 "comparador_mesma_carga_erro05.csv"):
        assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes()
    for row, expected in zip(table.itertuples(), [(96, 25, 121, 23, 22, 125), (17, 12, 29, 9, 9, 29)]):
        assert (row.P_revisoes_humanas_originais, row.P_MATCH_llm_ao_humano,
                row.P_revisoes_humanas_total, row.P_par2_recuperados,
                row.O3_par2_mesma_carga, row.O3_k_min_igualar_P) == expected
        assert row.P_FP_revisor_perfeito == row.O3_FP_revisor_perfeito == 0
        assert row.P_revisoes_humanas_total == row.O3_k_mesma_carga
        assert row.exploratorio_post_hoc and row.revisor_perfeito_ambos
    meta = json.loads((tmp_path / "comparador_p_match_humano_metadata.json").read_text(encoding="utf-8"))
    assert not meta["registered"] and meta["analysis"] == "exploratory_post_hoc"
    assert "BOTH" in meta["perfect_reviewer"] and "Does not model human false positives" in meta["error05"]
    assert not {"pair_id", "PAR", "REFREC", "COMPREC", "NOME"}.intersection(table.columns)


def copy_inputs(tmp_path, names):
    (tmp_path / "data").mkdir()
    for name in names:
        shutil.copyfile(ROOT / "data" / name, tmp_path / "data" / name)


def test_effort_uses_CSV_counts_and_minutes_not_the_legacy_JSON_constants(tmp_path):
    copy_inputs(tmp_path, ["carga_abaixo_corte.csv", "carga_abaixo_corte_horas.csv",
                           "esforco_arranjo_parametros.json", "comparador_mesma_carga.csv"])
    counts = pd.read_csv(tmp_path / "data/carga_abaixo_corte.csv")
    mask = counts.limiar.eq("por_passo") & counts.passo.eq("TOTAL")
    reference = int(counts.loc[mask, "pares_acima_limiar"].item()) + 10
    counts.loc[mask, "pares_acima_limiar"] = reference
    counts.to_csv(tmp_path / "data/carga_abaixo_corte.csv", index=False)
    hours = pd.read_csv(tmp_path / "data/carga_abaixo_corte_horas.csv")
    mask_hours = hours.limiar.eq("por_passo")
    minutes = float(hours.loc[mask_hours, "minutos_por_par"].item()) + 7
    hours.loc[mask_hours, "pares"] = reference
    hours.loc[mask_hours, "minutos_por_par"] = minutes
    hours.to_csv(tmp_path / "data/carga_abaixo_corte_horas.csv", index=False)
    result = effort(tmp_path)
    assert result.revisoes_referencia.eq(reference).all()
    assert result.minutos_por_par.eq(minutes).all()
    assert result.horas.equals(result.revisoes * minutes / 60)
    assert result.revisoes_poupadas.equals(reference - result.revisoes)
    hours.loc[mask_hours, "pares"] = reference + 1
    hours.to_csv(tmp_path / "data/carga_abaixo_corte_horas.csv", index=False)
    with pytest.raises(ValueError, match="diverge"):
        effort(tmp_path)


def test_editorial_5_percent_column_is_driven_by_materialized_CSV():
    data = pd.read_csv(ROOT / "data/comparador_mesma_carga.csv")
    projected = pd.read_csv(ROOT / "data/comparador_mesma_carga_erro05.csv")
    pd.testing.assert_series_equal(projected.par2_topk_erro05,
                                   projected.par2_topk * .95, check_names=False, rtol=1e-14)
    assert not projected.modela_FP_humano.any()
    selected = projected.modo.eq("vigilancia") & projected.ordenacao.eq("O1_SCORE") & projected.carga.eq("K1_humana")
    projected.loc[selected, "par2_topk_erro05"] = 13.37
    text = same_load(data, projected)
    assert "13{,}37" in text
    assert "não modela falsos positivos do revisor humano" in text
    with pytest.raises(ValueError, match="ausente"):
        same_load(data, projected.loc[~selected])
    with pytest.raises(ValueError, match="ambígua"):
        same_load(data, pd.concat([projected, projected.loc[selected]]))


@pytest.mark.deck_qualificacao
def test_line_endings_are_explicit_for_byte_compared_outputs():
    attributes = (ROOT / ".gitattributes").read_text()
    for name in ("data/esforco_arranjo.csv", "data/fp_arranjo_por_regiao.csv",
                 "text/latex/figures/curva_esforco.pgf", "text/slides/figures/curva_esforco.pgf"):
        assert name + " text eol=lf" in attributes
        assert b"\r\n" not in (ROOT / name).read_bytes()
