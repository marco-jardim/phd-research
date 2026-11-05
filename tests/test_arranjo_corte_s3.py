import json
from pathlib import Path

import pandas as pd
import pytest

from gzcmd.arranjo_corte import sha256
from gzcmd.intervals import wilson95
from scripts.arranjo_corte_s2_suplementar import responses
from scripts.arranjo_corte_s3_avaliar import ARMS, metrics, render_tex, replay
from local_data import DATA_ROOT, assert_frozen_csv_hash, requires_files, requires_reviewer_sources

ROOT = Path(__file__).resolve().parents[1]


def load_tables():
    return (pd.read_csv(ROOT / "data/arranjo_corte_metricas.csv"),
            pd.read_csv(ROOT / "data/arranjo_corte_par2.csv"))


def test_wilson_known_values_and_boundaries():
    assert wilson95(50, 100) == pytest.approx((.4038315303659956, .5961684696340044))
    assert wilson95(0, 36)[0] == 0
    assert wilson95(36, 36)[1] == 1
    with pytest.raises(ValueError):
        wilson95(37, 36)


def test_llm_unsure_is_one_human_referral_not_an_imputed_result():
    frame = pd.DataFrame({"y": [1, 1, 1, 0], "abaixo": [True] * 4,
                          "action": ["LLM_REVIEW", "LLM_REVIEW", "DEFERRED", "LLM_REVIEW"]},
                         index=["a", "b", "c", "d"])
    known = {"a": "MATCH", "b": "UNSURE", "d": "NONMATCH"}
    final = replay(frame, known)
    usage = {pid: {"calls": 2, "input_tokens": 100, "output_tokens": 20} for pid in known}
    cost = {pid: .001 for pid in known}
    row = metrics(final, "teste", "P", "TODAS", usage, cost)
    assert (row["aceitos"], row["vinc_humano"], row["perdidos"]) == (1, 2, 0)
    assert row["pares_humano"] == 2
    assert row["pares_llm"] == 3
    assert row["chamadas_llm"] == 6
    assert row["custo_llm_usd"] == pytest.approx(.003)
    assert row["sens_humano_05"] == pytest.approx((1 + .95 * 2) / 3)
    subset = metrics(final.loc[["a", "c"]], "teste", "P", "[5,9)", usage, cost)
    assert subset["pares_llm"] == 1
    assert subset["chamadas_llm"] == 2
    assert subset["custo_llm_usd"] == pytest.approx(.001)
    with pytest.raises(ValueError, match="lacks an observed response"):
        replay(frame, {"a": "MATCH"})


def test_generated_partitions_close_for_all_arms_and_scopes():
    table, primary = load_tables()
    assert len(table) == 36 and len(primary) == 18
    assert set(table.braco) == set(ARMS)
    assert table.aceitos.add(table.vinc_humano).add(table.perdidos).equals(table.vinculos)
    assert primary.aceitos.add(primary.humano).add(primary.perdidos).eq(36).all()
    assert table.loc[table.faixa.eq("TODAS"), "vinculos"].eq(247).all()
    assert table.loc[table.faixa.eq("[5,9)"), "vinculos"].eq(151).all()
    assert table.fp_abaixo.le(table.fp).all()
    assert table.sem_resposta_llm.eq(0).all()
    for row in table.itertuples():
        assert (row.ic95_inf, row.ic95_sup) == pytest.approx(wilson95(row.aceitos, row.vinculos))
        assert row.sens_humano_05 == pytest.approx((row.aceitos + .95 * row.vinc_humano) / row.vinculos)


@pytest.mark.parametrize("mode,expected", [("vigilancia", (194, 34, 111, 347)),
                                           ("confirmacao", (132, 0, 127, 44))])
def test_published_cascade_reproduces_reference(mode, expected):
    table, _ = load_tables()
    row = table.loc[table.modo.eq(mode) & table.braco.eq("publicada") & table.faixa.eq("TODAS")].iloc[0]
    assert tuple(row[k] for k in ("aceitos", "fp", "pares_humano", "pares_llm")) == expected


@pytest.mark.parametrize("mode,expected", [("vigilancia", (221, 34, 9, 90, 330)),
                                           ("confirmacao", (219, 0, 17, 51, 25))])
def test_aposteriori_reproduces_reference(mode, expected):
    table, _ = load_tables()
    row = table.loc[table.modo.eq(mode) & table.braco.eq("aposteriori") & table.faixa.eq("TODAS")].iloc[0]
    assert tuple(row[k] for k in ("aceitos", "fp", "perdidos", "pares_humano", "pares_llm")) == expected
    regenerated = pd.read_csv(ROOT / "data/arranjo_corte_aposteriori.csv")
    original = regenerated.loc[regenerated.modo.eq(mode) & regenerated.faixa.eq("TODAS")
                               & regenerated.arranjo.eq("corte_mais_cascata")].iloc[0]
    assert original.vinculos_automaticos == row.aceitos
    assert original.custo_llm_usd == pytest.approx(row.custo_llm_usd)


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
@requires_reviewer_sources()
@requires_files(ROOT / ".local/arranjo_corte/policy_vigilancia_P.csv",
                ROOT / ".local/arranjo_corte/policy_confirmacao_P.csv",
                DATA_ROOT / "data/funil_gzcmd_vigilancia.csv")
def test_P_has_no_unanswered_llm_pair(mode):
    known, _ = responses(DATA_ROOT, mode, include_supplement=True)
    policy = pd.read_csv(ROOT / f".local/arranjo_corte/policy_{mode}_P.csv")
    assert set(policy.loc[policy.action.eq("LLM_REVIEW"), "pair_id"]) <= known.keys()
    table, _ = load_tables()
    assert table.loc[table.modo.eq(mode) & table.braco.eq("P"), "sem_resposta_llm"].eq(0).all()


def test_cut_and_integral_human_primary_endpoints():
    table, primary = load_tables()
    cut = primary.loc[primary.braco.eq("corte_sozinho")]
    human = primary.loc[primary.braco.eq("humano_integral")]
    assert cut.aceitos.eq(0).all() and cut.perdidos.eq(36).all()
    assert human.humano.eq(36).all() and human.perdidos.eq(0).all()
    assert table.loc[table.braco.isin(["corte_sozinho", "humano_integral"]), "pares_llm"].eq(0).all()
    assert table.loc[table.braco.eq("corte_sozinho"), "pares_humano"].eq(0).all()


def test_render_identifies_comparisons_and_has_no_trailing_midrule(tmp_path):
    table, primary = load_tables()
    target = tmp_path / "arranjo.tex"
    render_tex(table, primary, target)
    tex = target.read_text(encoding="utf-8")
    assert "Cascata sobre a base inteira" in tex
    assert "A posteriori (sem registro prévio)" in tex
    assert "PAR~=~1" in tex and "PAR=1" not in tex
    assert "\\midrule\n\\bottomrule" not in tex
    assert "\\midrule\n\\end{longtable}" not in tex
    assert "(ii) Humano integral" in tex and "(iii)" not in tex
    assert "reutiliza a cascata sobre a base inteira" in tex
    caption = tex.split(r"\caption{", 1)[1].split(r"\label{", 1)[0]
    assert "corte do OpenRecLink" in caption and "padrão-ouro" in caption
    # O contrato cobre ambas as legendas e todas as notas, não só a primeira.
    assert "gabarito" not in tex.lower()
    assert "vieses do padrão-ouro" in tex
    assert "ressalvas sobre o padrão-ouro" in tex
    assert tex == (ROOT / "text/latex/tables/tab_cap8_arranjo_corte.tex").read_text(encoding="utf-8")


def test_aggregate_privacy_latex_and_auditable_comparator():
    table, primary = load_tables()
    forbidden = {"pair_id", "PAR", "TARGET", "y", "COMPREC", "REFREC"}
    assert not forbidden.intersection(table.columns)
    assert not forbidden.intersection(primary.columns)
    tex = (ROOT / "text/latex/tables/tab_cap8_arranjo_corte.tex").read_text(encoding="utf-8")
    assert tex.splitlines()[2:4] == [r"\scriptsize", r"\setlength{\tabcolsep}{2pt}"]
    assert all(value in tex for value in (r"\toprule", r"\bottomrule",
                                          r"\label{tab:cap8-arranjo-corte}", "verificação", "incorporação", "0{,}95"))
    import subprocess
    assert subprocess.check_output(["git", "ls-files", "scripts/arranjo_corte_orl_gzcmd.py"], cwd=ROOT)


def test_supplement_complete_and_rates_frozen():
    meta = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))
    supplement = json.loads((ROOT / "data/sprint3b/anthropic_sem_mo_v4_arranjo_corte_supl/execucao.json").read_text(encoding="utf-8"))
    assert supplement["total_pairs"] == 69
    assert all(r["complete"] and r["n_errors"] == 0 for r in supplement["runs"])
    assert meta["frozen_rates_sha256"] == supplement["frozen_rates_sha256"]
    assert_frozen_csv_hash(ROOT / "data/arranjo_corte_taxas_crossfit.csv", meta["frozen_rates_sha256"])
    assert meta["new_responses_reestimate_rates"] is False
    assert "chapter_tex_modified" not in meta
    for name, value in meta["code_sha256"].items():
        assert sha256(ROOT / name) == value


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", ROOT / ".local/arranjo_corte/oof.npz",
                ROOT / ".local/arranjo_corte/policy_vigilancia_P.csv")
@requires_reviewer_sources()
def test_S3_regenerates_exact_csvs_and_table(tmp_path):
    from scripts.arranjo_corte_s3_avaliar import evaluate
    prereg = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))["prereg_sha256"]
    evaluate(DATA_ROOT, prereg, output_root=tmp_path)
    for name in ("arranjo_corte_metricas.csv", "arranjo_corte_par2.csv", "arranjo_corte_aposteriori.csv"):
        assert (tmp_path / "data" / name).read_bytes() == (ROOT / "data" / name).read_bytes()
    table = "text/latex/tables/tab_cap8_arranjo_corte.tex"
    assert (tmp_path / table).read_bytes() == (ROOT / table).read_bytes()


def test_comparator_selftest_does_not_require_TEMP(monkeypatch):
    import os
    import subprocess
    import sys
    monkeypatch.delenv("TEMP", raising=False)
    result = subprocess.run([sys.executable, "scripts/arranjo_corte_orl_gzcmd.py", "--self-test"],
                            cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=True)
    assert "SELF-TEST OK" in result.stdout
