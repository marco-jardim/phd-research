"""Contratos W11a: agregação portável, rastreabilidade e limites editoriais."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.probe_dossie_v5_agregado import FIELDS, generate, utility
from scripts.v4_candidata_registro import FILES, build_registry
from terminologia import contem


ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def one(data, **criteria):
    found = [r for r in data if all(r[k] == str(v) for k, v in criteria.items())]
    assert len(found) == 1, criteria
    return found[0]


def text():
    return (ROOT / "text/latex/estudo4_gzcmd.tex").read_text(encoding="utf-8")


def pt(value):
    return f"{float(value):,.0f}".translate(str.maketrans(",.", ".,"))


@pytest.fixture
def source(tmp_path):
    """Reconstrói apenas o schema agregado, nunca consulta o diretório privado."""
    items = rows("probe_dossie_v5_agregado.csv")
    raw = []
    for item in items:
        item = dict(item)
        item["n"] = item.pop("n_comparados")
        item["n_expected"] = item.pop("n_originais")
        item["ponderado"] = "False"
        item["coluna_individual_nao_publicavel"] = "sentinela_privada"
        raw.append(item)
    with (tmp_path / "resultados_agregados.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw)
    metadata = json.loads((ROOT / "data/probe_dossie_v5_metadata.json").read_text(encoding="utf-8"))
    transport = {"stages": metadata["transport_by_stage"], "spent_usd": metadata["cost_usd"],
                 "gate": {"useful": False}}
    (tmp_path / "transporte_agregado.json").write_text(json.dumps(transport), encoding="utf-8")
    (tmp_path / "PREREG.md").write_text("Pré-registro sintético\n", encoding="utf-8")
    (tmp_path / "prompt_v5.md").write_text("Prompt sintético não publicado\n", encoding="utf-8")
    return tmp_path


def test_aggregate_portable_and_allowlisted(source, tmp_path):
    output = tmp_path / "publico"
    meta = generate(source, output)
    assert (output / "probe_dossie_v5_agregado.csv").read_bytes() == (ROOT / "data/probe_dossie_v5_agregado.csv").read_bytes()
    public = "\n".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert "sentinela_privada" not in public
    assert "Prompt sintético" not in public
    assert "pair_id" not in public
    assert tuple(rows("probe_dossie_v5_agregado.csv")[0]) == FIELDS
    assert meta["prompt_v5_sha256"] == hashlib.sha256((source / "prompt_v5.md").read_bytes()).hexdigest()
    for name, digest in meta["source_sha256"].items():
        assert digest == hashlib.sha256((source / name).read_bytes()).hexdigest()


def test_aggregate_rejects_inconsistent_denominators(source, tmp_path):
    path = source / "resultados_agregados.csv"
    contents = path.read_text(encoding="utf-8")
    # S1 contém 18 MATCH, 0 NONMATCH e 3 UNSURE, não 19 MATCH.
    path.write_text(contents.replace(",18,0,3,", ",19,0,3,", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Denominadores inconsistentes"):
        generate(source, tmp_path / "invalid")


def test_utility_gate_uses_paired_cases_and_does_not_count_missing():
    data = rows("probe_dossie_v5_agregado.csv")
    gate = utility(data)
    assert gate["FP_S1_retirados_validos"] == 3  # Não 21 históricos menos 14 observados.
    assert gate["PAR2_S2_adicionais"] == 2
    assert gate["VPP_S1"] == pytest.approx(4 / 18)
    assert gate["useful"] is False
    s1 = one(data, modo="vigilancia", estrato="S1", passada="v5")
    assert (int(s1["n_comparados"]), int(s1["n_originais"])) == (21, 25)
    assert (int(s1["NONMATCH"]), int(s1["UNSURE"])) == (0, 3)
    s2 = one(data, modo="vigilancia", estrato="S2", passada="v5")
    assert int(s2["n_sem_resultado"]) == 6
    assert int(s2["n_originais"]) == 13
    assert int(one(data, modo="vigilancia", estrato="S3", passada="v5")["FP"]) == 2
    for arm in ("v5", "v4_pareado", "v4_todos"):
        assert float(one(data, modo="confirmacao", estrato="C", passada=arm)["VPP"]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("field,value", [("FP", "12"), ("VPP", "0.35")])
def test_utility_thresholds_are_not_relaxed(field, value):
    data = rows("probe_dossie_v5_agregado.csv")
    one(data, modo="vigilancia", estrato="S1", passada="v5")[field] = value
    assert utility(data)["useful"] is True


def test_utility_recovery_threshold_and_missing_boundary():
    data = rows("probe_dossie_v5_agregado.csv")
    s1 = one(data, modo="vigilancia", estrato="S1", passada="v5")
    s1["FP"] = "13"  # Quatro retirados, ainda insuficiente.
    assert utility(data)["useful"] is False
    one(data, modo="vigilancia", estrato="S2", passada="v5")["PAR2_recuperados"] = "3"
    assert utility(data)["useful"] is True


def test_transport_cost_and_truncation():
    meta = json.loads((ROOT / "data/probe_dossie_v5_metadata.json").read_text(encoding="utf-8"))
    assert meta["cost_usd"] == pytest.approx(0.31989855, abs=1e-12)
    assert sum(s["cost_usd"] for s in meta["transport_by_stage"]) == pytest.approx(meta["cost_usd"], abs=1e-12)
    assert meta["cancellation"]["late_success_responses_used"] == 299
    assert meta["cancellation"]["canceled_without_generation"] == 1
    assert meta["truncadas_passada_principal"] == 25
    assert meta["pares_sem_decisao"] == 20
    assert len(meta["campos_acrescentados"]) == 15
    assert meta["opus_executado"] is False
    assert meta["utility"]["resultado"] == "não atingido"
    assert "25 respostas truncadas" in text() and "20 pares sem decisão" in text()


@pytest.mark.parametrize("variant,label", [
    ("O1", "O1"), ("O3", "O3"), ("F4", "F4"),
    ("O3_contexto", "C"), ("O3_contexto_MACD", "CM"),
])
def test_context_table_matches_versioned_aggregate(variant, label):
    data = rows("contexto_obito.csv")
    counts = [one(data, variante=variant, metrica=f"revisoes_{n}")["valor"] for n in (9, 23, 36)]
    assert label + " & " + " & ".join(pt(n) for n in counts) in text()


def test_context_bootstrap_and_band_limits_match_sources():
    data = rows("contexto_robustez_bootstrap.csv")
    entirely_negative = [(r["comparacao"], r["meta_nominal"]) for r in data
                         if r["comparacao"].startswith("CM-") and float(r["diferenca_revisoes_ic95_sup"]) < 0]
    assert entirely_negative == [("CM-O1", "36")]
    r = one(data, comparacao="CM-O1", meta_nominal=36)
    for key in ("diferenca_revisoes_mediana", "diferenca_revisoes_ic95_inf", "diferenca_revisoes_ic95_sup"):
        assert f"${pt(r[key])}$" in text()
    band = rows("contexto_robustez_faixa.csv")
    counts = [one(band, fila=q, meta=36)["revisoes"] for q in ("CM", "F4", "O3", "O1")]
    assert [int(c) for c in counts] == [416, 840, 1330, 2521]
    assert all(pt(n) in text() for n in counts)
    assert "a recomendação permanece F4" in text()


def test_propagation_and_bayesian_evidence_limits():
    r = one(rows("propagacao_pessoa.csv"), regra="R1_UNIAO", passo="TOTAL")
    assert [int(r[k]) for k in ("par2", "par0", "fp_P_vigilancia_propagados", "fp_P_confirmacao_propagados")] == [2, 0, 0, 0]
    assert "não altera as conclusões sobre as perdas" in text()
    data = rows("llm_evidencia.csv")
    post = lambda metric: float(one(data, modo="vigilancia", variante="O3-post", tipo="aceitacao", metrica=metric)["valor"])
    assert (post("VP"), post("aceitacoes")) == (11, 26)
    assert post("VPP") == pytest.approx(11 / 26)
    assert float(one(data, modo="vigilancia", variante="P-LLM", tipo="aceitacao", metrica="VPP")["valor"]) == .16
    assert contem(text(), "42,3\\% (11 vínculos em 26 aceitações)")  # migrado: terminologia 3.6
    assert "58,3\\% dos PAR~=~2" in text()
    assert "não constitui validação aninhada integral" in text()


def test_registry_hashes_and_portable_root(tmp_path):
    saved = json.loads((ROOT / "data/v4_candidata_registro.json").read_text(encoding="utf-8"))
    assert saved == build_registry(ROOT, saved["date_utc"])
    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    assert build_registry(tmp_path, saved["date_utc"]) == saved
    (tmp_path / FILES[0]).write_bytes(b"alterado\n")
    assert build_registry(tmp_path, saved["date_utc"])["sha256"] != saved["sha256"]


def test_candidate_primary_features_and_training_are_fixed():
    spec = yaml.safe_load((ROOT / "gzcmd/v4_candidata.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "data/contexto_obito_metadata.json").read_text(encoding="utf-8"))
    assert spec["primary"]["comparison"] == "O3_vs_O1"
    assert spec["primary"]["margin_reduction"] == .25
    assert spec["targets"]["new_cohort_fractions"] == [.25, .64, 1.0]
    assert spec["secondary"]["CM"]["features"] == metadata["features"]["O3_contexto_MACD"]
    for key in ("objective", "eval_metric", "n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "n_jobs", "random_state", "verbosity"):
        assert spec["training"][key] == metadata["classifier_parameters"][key]
    assert spec["training"]["split_by"] == "REFREC"
    assert spec["training"]["inner_seeds"] == metadata["inner_seeds"]
    assert spec["training"]["versions"] == metadata["versions"]
    assert spec["reference"]["reviewers"] == 2 and spec["reference"]["arbitration"]
    assert spec["stopping"]["cohort_start_blocked_until_resolved"]
    assert set(spec["excluded"]) == {"LLM_como_evidencia", "dossie_v5", "encaminhamento_conformal"}
    assert r"pelo menos 25\% menos revisões que O1, pelo limite inferior do intervalo de 95\% entre partições predefinidas" in text()
    assert "gzcmd/M_v4_candidata.md" in text()
    assert "método de parada com garantia de recuperação permanece pendente" in text()


def test_new_table_has_consistent_columns_and_no_prose_dashes():
    import re

    body = text()
    assert not re.search("[—–]", body)
    table = body.split(r"\label{tab:cap8-contexto-posthoc}", 1)[1].split(r"\end{table}", 1)[0]
    for line in table.splitlines():
        if " & " in line:
            assert line.count(" & ") == 3
    assert "acrescentar campos não bastou" in body
    assert "não foi confirmada nem refutada" in body
