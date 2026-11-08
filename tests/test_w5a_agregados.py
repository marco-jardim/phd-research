"""Proveniência e agregados W5a, sem acesso portátil a valores por par."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import ablacao_dossie_proveniencia as provenance
from scripts import ablacao_teste_par2_agregado as composition
from scripts import perfil_obitos_agregado as profile
from local_data import DATA_ROOT, requires_files

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = DATA_ROOT / "data/COMPARADORSEMIDENT.csv"
SPRINT = DATA_ROOT / "data/sprint3b"


def test_metadata_allowlist_drops_pair_data_and_prompt_text(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(dict(model="test", prompt_version="v4", system_prompt="secret prompt",
                                    params=dict(temperature=0, pair_id="secret"), results=["private"],
                                    selected_pair_ids=["private"])), encoding="utf-8")
    result = provenance.metadata(path, tmp_path)
    assert result["system_prompt_sha256"] == hashlib.sha256(b"secret prompt").hexdigest()
    assert result["params"] == {"temperature": 0}
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)
    assert result["date"] is None


def test_public_provenance_records_missing_arm_prompt_without_imputation():
    result = json.loads((ROOT / "data/sprint3b/ablacao/ablacao_dossie_proveniencia.json").read_text(encoding="utf-8"))
    assert len(result["arms"]) == 5 and len(result["cohorts"]) == 6
    for arm in result["arms"]:
        assert arm["prompt_version"] is None and arm["prompt_sha"] is None
        assert arm["date"] is None and arm["params"] == {}
        aggregate = ROOT / "data/sprint3b" / arm["source"].replace(".json", "_agregado.json")
        data = json.loads(aggregate.read_text(encoding="utf-8"))
        assert arm["source_sha256"] in json.dumps(data)
    v4 = result["cohorts"]["anthropic_sem_mo_v4"]
    runs = [r for r in v4 if r["source"].split("/")[-1].startswith("exp2_llm_sync_")]
    assert len(runs) == 2 and all(r["prompt_version"] == "v4" for r in runs)
    assert all(r["params"]["max_tokens"] == 2500 for r in runs)
    assert not {"selected_pair_ids", "results", "dossier_sha256"} & set(result)


def test_absent_sources_do_not_overwrite_provenance_with_empty_evidence(tmp_path):
    with pytest.raises(FileNotFoundError, match="Originais privados ausentes"):
        provenance.generate(tmp_path)


@requires_files(SPRINT / "ablacao/claude-haiku-5-5_controle.json")
def test_private_arm_metadata_reproduces_public():
    result = json.loads((ROOT / "data/sprint3b/ablacao/ablacao_dossie_proveniencia.json").read_text(encoding="utf-8"))
    arm = next(r for r in result["arms"] if r["source"].endswith("claude-haiku-5-5_controle.json"))
    assert arm == provenance.metadata(SPRINT / arm["source"], SPRINT)


def test_composition_closes_without_pair_fields():
    frame = pd.read_csv(ROOT / "data/ablacao_teste_composicao.csv")
    assert frame.iloc[0].to_dict() == dict(particao="classificadores_linha", seed=42, test_size=.3,
                                          n_teste=18509, par1=66, par2=8, vinculos=74)
    assert not {"pair_id", "COMPREC", "REFREC", "PAR"} & set(frame)


@requires_files(PRIVATE)
def test_composition_reuses_canonical_split():
    result = composition.summarize(composition.load_raw(PRIVATE))
    pd.testing.assert_frame_equal(result, pd.read_csv(ROOT / "data/ablacao_teste_composicao.csv"))


def test_profile_aggregate_denominators_and_editorial():
    frame = pd.read_csv(ROOT / "data/perfil_obitos_agregado.csv")
    assert not {"pair_id", "COMPREC", "REFREC", "R_DTNASC", "R_DTOBITO", "R_SEXO"} & set(frame)
    assert frame.n.tolist() == [61696, 211, 19388, 1515, 27, 40]
    columns = [f"sexo_{sex}_n" for sex in ("masculino", "feminino", "ausente_ou_ambiguo")]
    assert frame[columns].sum(axis=1).eq(frame.n).all()
    assert frame.idade_n.le(frame.n).all() and frame.nota_n.eq(frame.n).all()
    assert profile.table(frame) == (ROOT / "text/latex/tables/editorial/tab_perfil_obitos_agregado.tex").read_text(encoding="utf-8")
    for name in ("recuperados", "ja_encontrados"):
        row = frame.set_index("conjunto").loc[name]
        assert pd.isna(row.idade_mediana) and pd.isna(row.nota_dp)


def test_profile_does_not_choose_arbitrary_candidate_for_demography():
    raw = pd.DataFrame({"PAR": ["1", "0"], "COMPREC": ["c", "c"], "REFREC": ["r1", "r2"],
                        "R_DTNASC": ["19800101", "19900101"], "R_DTOBITO": ["01012020", "01012020"],
                        "R_SEXO": ["M", "F"], "nota final": ["9,0", "3,0"]})
    row = profile.summarize(raw).set_index("conjunto").loc["registros_comprec"]
    assert row.n == 1 and row.idade_n == 0 and row.sexo_ausente_ou_ambiguo_n == 1
    assert row.nota_media == 6
    raw = raw.drop(columns=["R_DTNASC", "R_DTOBITO", "R_SEXO"])
    missing = profile.summarize(raw)
    assert missing.idade_n.eq(0).all()
    assert missing.idade_origem.eq("não disponível no extrato").all()
    assert missing.sexo_origem.eq("não disponível no extrato").all()


@requires_files(PRIVATE)
def test_profile_from_private_extract_matches_public_base_sets():
    actual = profile.summarize(composition.load_raw(PRIVATE))
    expected = pd.read_csv(ROOT / "data/perfil_obitos_agregado.csv").iloc[:4]
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_text_uses_test_par2_denominator_and_profile_source():
    text = (ROOT / "text/latex/estudo3_estrategias.tex").read_text(encoding="utf-8")
    # Revisão 2: o perfil agregado dos óbitos foi para o apêndice das estratégias.
    appendix = (ROOT / "text/latex/apendice_estrategias.tex").read_text(encoding="utf-8")
    assert "cinco dos oito PAR" in text
    assert r"\path{data/ablacao_teste_composicao.csv}" in text
    assert r"\input{tables/editorial/tab_perfil_obitos_agregado.tex}" in appendix
    assert "não foi identificado agregado versionado" not in text + appendix
