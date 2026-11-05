import json
from pathlib import Path

import pandas as pd
import pytest

from local_data import DATA_ROOT, requires_files, requires_reviewer_sources
from scripts.cenarios_sensibilidade_humana import (
    aggregate_agreement, audit_filter_modes, compare_order, generate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scenarios_regenerate_from_versioned_aggregates(tmp_path):
    table, amplitude, order = generate(output=tmp_path)
    for stem in ("cenarios_sensibilidade_humana", "cenarios_sensibilidade_humana_amplitude",
                 "cenarios_sensibilidade_humana_ordem"):
        assert (tmp_path / (stem + ".csv")).read_bytes() == (ROOT / "data" / (stem + ".csv")).read_bytes()
    name = "cenarios_sensibilidade_humana_metadata.json"
    assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes()
    assert len(table) == 78 and len(amplitude) == 26
    assert table.especificidade_humana_assumida.eq(1.0).all()
    assert set(table.s) == {.95, .99, 1.0}
    assert table.fonte_s.eq("cenario (config gzcmd_v3: 0,05; nao medido)").all()
    assert not {"pair_id", "COMPREC", "REFREC", "NOME"}.intersection(table.columns)
    for row in table.itertuples():
        assert row.sensibilidade_total == pytest.approx(
            (row.vinculos_aceitos_sem_humano + row.s * row.vinculos_ao_humano) / row.vinculos_total)
        assert row.vinculos_perdidos_cobertura == row.vinculos_total - row.vinculos_aceitos_sem_humano - row.vinculos_ao_humano
    assert not order.inverte_ordem_estrita.any()
    assert order.empate_varia.sum() == 3
    assert not order.loc[order.mesma_carga_humana, "empate_varia"].any()
    assert amplitude.amplitude_pp.max() == pytest.approx(100 * .05 * 36 / 247)
    assert amplitude.loc[amplitude.estrategia.ne("humano_integral"), "amplitude_pp"].max() == pytest.approx(100 * .05 * 26 / 247)


def test_scenarios_take_counts_from_the_input_CSVs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    from scripts.cenarios_sensibilidade_humana import SOURCES, PARAMS
    for name in (*SOURCES, PARAMS):
        (data / name).write_bytes((ROOT / "data" / name).read_bytes())
    metrics = pd.read_csv(data / "arranjo_corte_metricas.csv")
    selected = metrics.modo.eq("vigilancia") & metrics.braco.eq("P") & metrics.faixa.eq("TODAS")
    metrics.loc[selected, "aceitos"] -= 1
    metrics.to_csv(data / "arranjo_corte_metricas.csv", index=False)
    actual, _, _ = generate(tmp_path)
    row = actual.loc[actual.modo.eq("vigilancia") & actual.estrategia.eq("P") & actual.s.eq(1)].iloc[0]
    assert row.vinculos_aceitos_sem_humano == metrics.loc[selected, "aceitos"].item()
    assert row.sensibilidade_total == pytest.approx((row.vinculos_aceitos_sem_humano + row.vinculos_ao_humano) / row.vinculos_total)


def test_order_comparison_can_detect_a_strict_reversal_not_just_ties():
    synthetic = pd.DataFrame({"modo": ["test"] * 4, "estrategia": ["A", "A", "B", "B"],
                              "k": [10] * 4, "revisoes_humanas": [10] * 4,
                              "s": [.95, 1, .95, 1], "sensibilidade_total": [.1, .1, .09, .11]})
    order = compare_order(synthetic)
    assert order.inverte_ordem_estrita.item()
    assert not order.empate_varia.item()


def test_agreement_counts_decision_switches_without_writing_pair_rows():
    frame = pd.DataFrame({"par": [2, 0, 0], "vigilancia": ["MATCH", "NONMATCH", "UNSURE"],
                          "confirmacao": ["MATCH", "UNSURE", "NONMATCH"]})
    out = aggregate_agreement(frame, [3]).set_index("classe")
    assert out.loc["TODOS", ["coincidentes", "diferentes"]].tolist() == [1, 2]
    assert out.loc["PAR2", "diferentes"] == 0
    assert out.loc["nao_vinculo", ["NONMATCH_UNSURE", "UNSURE_NONMATCH"]].tolist() == [1, 1]
    frame.loc[0, "confirmacao"] = "invalid"
    with pytest.raises(ValueError, match="desconhecida"):
        aggregate_agreement(frame, [3])


def test_versioned_mode_audit_proves_shared_queue_not_shared_response_file():
    table = pd.read_csv(ROOT / "data/filtro_llm_concordancia_modos.csv")
    total = table.loc[table.classe.eq("TODOS")].set_index("k")
    assert total.loc[125, ["coincidentes", "diferentes"]].tolist() == [119, 6]
    assert total.loc[[17, 29], "diferentes"].eq(2).all()
    assert table.loc[table.classe.eq("PAR2"), "diferentes"].eq(0).all()
    meta = json.loads((ROOT / "data/filtro_llm_concordancia_modos_metadata.json").read_text(encoding="utf-8"))
    assert meta["queue_same_by_design"] and meta["distinct_response_files"]
    assert meta["declared_modes_validated"] and meta["historical_response_hashes_reproduced"]
    assert not meta["historical_filter_metadata_edited"]
    assert meta["api_calls"] == meta["model_fits"] == 0


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                ROOT / "data/sprint3b/anthropic_sem_mo_v4_filtro_llm/exp2_llm_sync_vigilancia.json",
                ROOT / "data/sprint3b/anthropic_sem_mo_v4_filtro_llm/exp2_llm_sync_confirmacao.json")
@requires_reviewer_sources()
def test_private_mode_audit_regenerates_count_only_outputs(tmp_path):
    audit_filter_modes(DATA_ROOT, output=tmp_path)
    for name in ("filtro_llm_concordancia_modos.csv", "filtro_llm_concordancia_modos_metadata.json"):
        assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes()
