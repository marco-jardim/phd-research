"""Portão da atestação bibliográfica: todo contexto de citação da tese precisa estar no ledger.

* ``test_*_tese_real`` rodam sobre ``text/latex/tese.tex`` e ``docs/atestado/`` (pulam só se o
  ledger não existir).
* Os demais testam o exportador e a lógica do portão com dados sintéticos.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts import atestar_citacoes as cli
from scripts import citacoes_contexto as cc
from scripts.atestado import ledger as led
from scripts.atestado.common import strip_accents

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "atestado" / led.LEDGER_NAME
PENDENCIAS = ROOT / "docs" / "atestado" / led.PENDENCIAS_NAME


# ---------------------------------------------------------------------------
# Tese real
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tese():
    if not LEDGER.is_file():
        pytest.skip(
            "docs/atestado/ledger.csv não existe: rode "
            "`python scripts/atestar_citacoes.py exportar-ledger --escopo tese`"
        )
    doc = cc.build_document(ROOT / "text" / "latex" / "tese.tex", ROOT)
    records = cc.extract_citations(doc)
    entries, _files = cc.load_bib(doc)
    own = {
        key
        for key, entry in entries.items()
        if "jardim" in strip_accents(entry.fields.get("author", "")).casefold()
    }
    return {
        "ids": [r["id"] for r in records],
        "locais": {r["id"]: f"{r['key']} ({r['file']}:{r['line']})" for r in records},
        "chaves": {r["id"]: r["key"] for r in records},
        "own": own,
        "ledger": led.load_ledger(LEDGER),
        "pendencias": led.load_pendencias(PENDENCIAS),
    }


def test_todo_contexto_da_tese_esta_atestado(tese):
    faltam = led.contextos_sem_atestamento(tese["ids"], tese["ledger"])
    detalhe = "\n".join(f"  {cid}  {tese['locais'][cid]}" for cid in faltam[:50])
    assert not faltam, (
        f"contextos sem atestamento: {len(faltam)} de {len(tese['ids'])}. Frases novas ou reescritas "
        "precisam ser atestadas (ver docstring de scripts/atestar_citacoes.py).\n" + detalhe
    )


def test_nenhum_problema_novo_fora_das_pendencias(tese):
    novos = led.problemas_fora_das_pendencias(tese["ledger"], tese["pendencias"])
    detalhe = "\n".join(f"  {r['id']}  {r['chave']}  {r['arquivo']}  {r['veredito_final']}" for r in novos)
    assert not novos, "NAO_SUSTENTA/CONTRADIZ fora de docs/atestado/pendencias.yaml:\n" + detalhe


def test_pendencias_tem_decisao_ou_origem_de_linha_de_base(tese):
    abertas = led.pendencias_sem_decisao(tese["pendencias"])
    detalhe = "\n".join(f"  {p['id']}  {p.get('chave')}  {p.get('arquivo')}" for p in abertas)
    assert not abertas, (
        f"pendências sem origem {led.ORIGEM_BASELINE} e com decisao 'pendente' (o autor precisa decidir):\n" + detalhe
    )


def test_chave_do_ledger_coincide_com_a_do_contexto(tese):
    errados = led.chaves_divergentes(tese["ledger"], tese["chaves"])
    detalhe = "\n".join(f"  {r['id']}  ledger={r['chave']}  contexto={tese['chaves'][r['id']]}" for r in errados[:50])
    assert not errados, "chave do ledger diferente da chave do contexto extraído:\n" + detalhe


def test_ledger_tem_hash_de_avaliacao(tese):
    sem_hash = led.linhas_sem_hash(tese["ledger"])
    assert not sem_hash, f"linhas sem hash_avaliacao: {[r['id'] for r in sem_hash[:20]]}"


def test_ledger_so_aceita_proprio_para_obra_propria(tese):
    indevidos = led.proprios_indevidos(tese["ledger"], tese["own"])
    assert not indevidos, f"PROPRIO em chave que não é de autoria 'Jardim': {[r['chave'] for r in indevidos]}"


def test_ledger_sem_veredito_vazio_pendente_ou_divergente(tese):
    assert not led.linhas_sem_veredito(tese["ledger"])


# ---------------------------------------------------------------------------
# Lógica do portão (sintético)
# ---------------------------------------------------------------------------
def _linha(cid, chave="A2020", veredito="SUSTENTA"):
    return {
        "id": cid, "id_legado": "", "chave": chave, "arquivo": "x.tex", "capitulo": "C", "veredito_final": veredito,
        "nivel_evidencia": "resumo", "adjudicado": "nao", "modelo": "m", "data": "2026-11-01",
        "hash_avaliacao": led.hash_avaliacao("m", veredito, "trecho"),
    }


def test_portao_rejeita_divergente_vazio_e_pendente():
    ledger = [_linha("a1"), _linha("a2", veredito="DIVERGENTE"), _linha("a3", veredito=""), _linha("a4", veredito="PENDENTE")]
    assert [r["id"] for r in led.linhas_sem_veredito(ledger)] == ["a2", "a3", "a4"]
    assert led.linhas_sem_veredito([_linha("a1"), _linha("a5", veredito="NAO_SUSTENTA")]) == []


def test_portao_chave_do_ledger_deve_ser_a_do_contexto():
    ledger = [_linha("a1", chave="A2020"), _linha("a2", chave="B2021"), _linha("a3", chave="C2022")]
    contextos = {"a1": "A2020", "a2": "OUTRA2021"}
    assert [r["id"] for r in led.chaves_divergentes(ledger, contextos)] == ["a2"]  # a3 não está nos contextos


def test_portao_exige_hash_de_avaliacao_exceto_proprio():
    ok = _linha("a1")
    sem = {**_linha("a2"), "hash_avaliacao": ""}
    curto = {**_linha("a3"), "hash_avaliacao": "abc"}
    proprio = {**_linha("p1", veredito="PROPRIO"), "hash_avaliacao": ""}
    assert [r["id"] for r in led.linhas_sem_hash([ok, sem, curto, proprio])] == ["a2", "a3"]


def test_hash_avaliacao_depende_de_modelo_veredito_e_trecho():
    base = led.hash_avaliacao("m", "SUSTENTA", "trecho")
    assert len(base) == 16 and base == led.hash_avaliacao("m", "SUSTENTA", "trecho")
    assert len({base, led.hash_avaliacao("m2", "SUSTENTA", "trecho"), led.hash_avaliacao("m", "CONTRADIZ", "trecho"),
                led.hash_avaliacao("m", "SUSTENTA", "outro trecho")}) == 4


def test_portao_pendencia_pendente_so_com_origem_de_linha_de_base():
    base = {"id": "a1", "chave": "A", "arquivo": "x.tex", "decisao": "pendente", "origem": led.ORIGEM_BASELINE}
    nova = {"id": "a2", "chave": "A", "arquivo": "x.tex", "decisao": "pendente"}
    decidida = {"id": "a3", "chave": "A", "arquivo": "x.tex", "decisao": "aceita, erro de citação"}
    assert [p["id"] for p in led.pendencias_sem_decisao([base, nova, decidida])] == ["a2"]


def test_portao_id_ausente_do_ledger_falha():
    ledger = [_linha("a1"), _linha("a2")]
    assert led.contextos_sem_atestamento(["a1", "a2"], ledger) == []
    assert led.contextos_sem_atestamento(["a1", "novo"], ledger) == ["novo"]


def test_portao_problema_novo_fora_das_pendencias_falha():
    ledger = [_linha("a1", veredito="NAO_SUSTENTA"), _linha("a2", veredito="CONTRADIZ"), _linha("a3")]
    pendencias = [{"id": "a1", "chave": "A2020", "arquivo": "x.tex", "decisao": "pendente"}]
    fora = led.problemas_fora_das_pendencias(ledger, pendencias)
    assert [r["id"] for r in fora] == ["a2"]
    pendencias.append({"id": "a2", "chave": "A2020", "arquivo": "x.tex", "decisao": "pendente"})
    assert led.problemas_fora_das_pendencias(ledger, pendencias) == []


def test_portao_proprio_aceito_so_para_obra_propria():
    ledger = [_linha("p1", chave="Jardim2025", veredito="PROPRIO"), _linha("p2", chave="Outro2020", veredito="PROPRIO")]
    assert [r["id"] for r in led.proprios_indevidos(ledger, {"Jardim2025"})] == ["p2"]
    assert led.problemas_fora_das_pendencias(ledger[:1], []) == []


# ---------------------------------------------------------------------------
# Exportador (sintético)
# ---------------------------------------------------------------------------
def _avaliacao(cid, chave, veredito, **extra):
    return {
        "id": cid, "key": chave, "file": "text/latex/cap.tex", "line": 3, "chapter": "Introdução",
        "classe": "aberta", "nivel_evidencia": "texto_completo", "veredito_final": veredito, "flags": [],
        "trecho_fonte": "TRECHO SECRETO", "justificativa": "JUSTIFICATIVA SECRETA", **extra,
    }


def _piloto(tmp_path, avaliacoes, artefatos=None, contextos=None):
    out = tmp_path / ".local" / "biblio"
    pdir = out / "piloto" / "tese"
    pdir.mkdir(parents=True)
    (pdir / "avaliacoes.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in avaliacoes) + "\n", encoding="utf-8"
    )
    (pdir / "metadados.json").write_text(
        json.dumps({"modelo": "claude-sonnet-5-5", "data_utc": "2026-11-05T10:00:00+00:00"}), encoding="utf-8"
    )
    (pdir / "artefatos.json").write_text(json.dumps(artefatos or []), encoding="utf-8")
    contextos = contextos or [{"id": "p1", "key": "Jardim2025", "chapter": "Métodos"}]
    (out / "citacoes.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in contextos) + "\n", encoding="utf-8"
    )
    return out, pdir


def test_exportador_grava_ledger_sem_citacoes_nem_justificativas(tmp_path):
    avaliacoes = [
        _avaliacao("a1", "Silva2020", "SUSTENTA"),
        _avaliacao(
            "a2", "Lima2019", "NAO_SUSTENTA",
            adjudicacao={"modelo": "claude-opus-5-5", "em": "2026-11-06T09:00:00+00:00", "veredito": "NAO_SUSTENTA"},
        ),
        _avaliacao("a3", "Pend2019", "PENDENTE", flags=["pendente"]),
    ]
    artefatos = [{"id": "p1", "key": "Jardim2025", "file": "text/latex/m.tex", "line": 9, "veredito": "ARTEFATO_OK"}]
    out, pdir = _piloto(tmp_path, avaliacoes, artefatos)
    dest = tmp_path / "docs" / "atestado"

    summary = led.exportar_ledger(pdir, out, dest, aceitar_novas=True)

    texto = (dest / "ledger.csv").read_text(encoding="utf-8")
    assert "SECRET" not in texto and '"' not in texto
    assert texto.splitlines()[0] == (
        "id;id_legado;chave;arquivo;capitulo;veredito_final;nivel_evidencia;adjudicado;modelo;data;hash_avaliacao"
    )
    rows = {r["id"]: r for r in led.load_ledger(dest / "ledger.csv")}
    assert set(rows) == {"a1", "a2", "p1"}
    assert dict(rows["a1"]) == {
        "id": "a1", "id_legado": "", "chave": "Silva2020", "arquivo": "text/latex/cap.tex", "capitulo": "Introdução",
        "veredito_final": "SUSTENTA", "nivel_evidencia": "texto_completo", "adjudicado": "nao",
        "modelo": "claude-sonnet-5-5", "data": "2026-11-05",
        "hash_avaliacao": led.hash_avaliacao("claude-sonnet-5-5", "SUSTENTA", "TRECHO SECRETO"),
    }
    assert "TRECHO" not in rows["a1"]["hash_avaliacao"] and rows["p1"]["hash_avaliacao"] == ""
    assert (rows["a2"]["adjudicado"], rows["a2"]["modelo"], rows["a2"]["data"]) == (
        "sim", "claude-opus-5-5", "2026-11-06",
    )
    assert (rows["p1"]["veredito_final"], rows["p1"]["capitulo"], rows["p1"]["modelo"]) == ("PROPRIO", "Métodos", "")
    assert summary["pendentes_sem_veredito"] == ["a3"]

    pend = yaml.safe_load((dest / "pendencias.yaml").read_text(encoding="utf-8"))["pendencias"]
    assert pend == [{"id": "a2", "chave": "Lima2019", "arquivo": "text/latex/cap.tex", "decisao": "pendente"}]
    assert "SECRET" not in (dest / "pendencias.yaml").read_text(encoding="utf-8")


def _primeira_exportacao(tmp_path):
    """Ledger com a1 CONTRADIZ aceito (decisão registrada) em ``dest``."""
    out, pdir = _piloto(tmp_path, [_avaliacao("a1", "A2020", "CONTRADIZ")])
    dest = tmp_path / "docs" / "atestado"
    led.exportar_ledger(pdir, out, dest, aceitar_novas=True)
    texto = (dest / "pendencias.yaml").read_text(encoding="utf-8").replace("decisao: pendente", "decisao: aceita, erro de citação")
    (dest / "pendencias.yaml").write_text(texto, encoding="utf-8")
    return dest


def test_reexportar_preserva_decisoes_e_nao_acrescenta_pendencias_novas(tmp_path):
    dest = _primeira_exportacao(tmp_path)
    out, pdir = _piloto(
        tmp_path / "2", [_avaliacao("a1", "A2020", "CONTRADIZ"), _avaliacao("a9", "B2021", "NAO_SUSTENTA")]
    )
    summary = led.exportar_ledger(pdir, out, dest)

    pend = {p["id"]: p for p in led.load_pendencias(dest / "pendencias.yaml")}
    assert pend["a1"]["decisao"] == "aceita, erro de citação"
    assert "a9" not in pend  # sem --aceitar-novas o id novo não entra em pendencias.yaml
    assert summary["pendencias_novas"] == ["a9"] and summary["novas_aceitas"] is False
    ledger = {r["id"]: r for r in led.load_ledger(dest / "ledger.csv")}
    assert ledger["a9"]["veredito_final"] == "NAO_SUSTENTA"  # o ledger é gravado e o portão falha
    assert [r["id"] for r in led.problemas_fora_das_pendencias(list(ledger.values()), list(pend.values()))] == ["a9"]


def test_aceitar_novas_acrescenta_com_decisao_pendente(tmp_path):
    dest = _primeira_exportacao(tmp_path)
    out, pdir = _piloto(
        tmp_path / "2", [_avaliacao("a1", "A2020", "CONTRADIZ"), _avaliacao("a9", "B2021", "NAO_SUSTENTA")]
    )
    summary = led.exportar_ledger(pdir, out, dest, aceitar_novas=True)

    pend = {p["id"]: p for p in led.load_pendencias(dest / "pendencias.yaml")}
    assert pend["a1"]["decisao"] == "aceita, erro de citação"
    assert pend["a9"] == {"id": "a9", "chave": "B2021", "arquivo": "text/latex/cap.tex", "decisao": "pendente"}
    assert summary["pendencias_novas"] == ["a9"] and summary["novas_aceitas"] is True
    assert [p["id"] for p in led.pendencias_sem_decisao(list(pend.values()))] == ["a9"]  # o portão ainda exige decisão


def test_marcar_baseline_marca_so_as_pendencias_existentes(tmp_path):
    out, pdir = _piloto(tmp_path, [_avaliacao("a1", "A2020", "CONTRADIZ"), _avaliacao("a2", "B2021", "NAO_SUSTENTA")])
    dest = tmp_path / "docs" / "atestado"
    dest.mkdir(parents=True)
    (dest / "pendencias.yaml").write_text(
        yaml.safe_dump({"pendencias": [{"id": "a1", "chave": "A2020", "arquivo": "x.tex", "decisao": "pendente"}]}),
        encoding="utf-8",
    )
    summary = led.exportar_ledger(pdir, out, dest, marcar_baseline=True)

    pend = {p["id"]: p for p in led.load_pendencias(dest / "pendencias.yaml")}
    assert pend["a1"]["origem"] == led.ORIGEM_BASELINE and pend["a1"]["decisao"] == "pendente"
    assert "a2" not in pend and summary["pendencias_novas"] == ["a2"]
    assert led.pendencias_sem_decisao(list(pend.values())) == []
    # idempotente: a marca é preservada nas exportações seguintes
    led.exportar_ledger(pdir, out, dest)
    assert led.load_pendencias(dest / "pendencias.yaml")[0]["origem"] == led.ORIGEM_BASELINE


def test_avaliacoes_com_id_legado_sao_mapeadas_ao_id_novo(tmp_path):
    contextos = [
        {"id": "n1", "id_legado": "old1", "key": "Silva2020", "file": "text/latex/novo.tex", "chapter": "Cap"},
        {"id": "n2", "id_legado": "old1", "key": "Silva2020", "file": "text/latex/novo.tex", "chapter": "Cap"},
        {"id": "n3", "id_legado": "old3", "key": "Lima2019", "file": "text/latex/novo.tex", "chapter": "Cap"},
    ]
    avaliacoes = [
        _avaliacao("old1", "Silva2020", "SUSTENTA"),
        _avaliacao("n3", "Lima2019", "CONTRADIZ"),  # já com id novo
        _avaliacao("orfao", "Zeta2000", "SUSTENTA"),  # id que não existe mais no texto
    ]
    out, pdir = _piloto(tmp_path, avaliacoes, contextos=contextos)
    dest = tmp_path / "docs" / "atestado"
    # pendência antiga (id legado) migra para o id novo sem contar como problema novo
    dest.mkdir(parents=True)
    (dest / "pendencias.yaml").write_text(
        yaml.safe_dump({"pendencias": [{"id": "old3", "chave": "Lima2019", "arquivo": "x.tex", "decisao": "ok",
                                         "origem": led.ORIGEM_BASELINE}]}),
        encoding="utf-8",
    )
    summary = led.exportar_ledger(pdir, out, dest)

    rows = {r["id"]: r for r in led.load_ledger(dest / "ledger.csv")}
    assert set(rows) == {"n1", "n2", "n3", "orfao"}
    assert rows["n1"]["id_legado"] == rows["n2"]["id_legado"] == "old1"
    assert rows["n1"]["arquivo"] == "text/latex/novo.tex" and rows["n1"]["capitulo"] == "Cap"
    assert rows["n1"]["hash_avaliacao"] == rows["n2"]["hash_avaliacao"]
    assert led.contextos_sem_atestamento(["n1", "n2", "n3"], list(rows.values())) == []
    pend = led.load_pendencias(dest / "pendencias.yaml")
    assert pend == [{"id": "n3", "chave": "Lima2019", "arquivo": "text/latex/novo.tex", "decisao": "ok",
                     "origem": led.ORIGEM_BASELINE}]
    assert summary["pendencias_novas"] == [] and summary["pendencias_resolvidas"] == []


def test_avaliacao_com_id_atual_prevalece_sobre_a_mapeada_por_id_legado(tmp_path):
    contextos = [{"id": "n1", "id_legado": "old1", "key": "A2020", "file": "f.tex", "chapter": "C"}]
    avaliacoes = [_avaliacao("old1", "A2020", "SUSTENTA"), _avaliacao("n1", "A2020", "CONTRADIZ")]
    out, pdir = _piloto(tmp_path, avaliacoes, contextos=contextos)
    led.exportar_ledger(pdir, out, tmp_path / "dest", aceitar_novas=True)
    (row,) = led.load_ledger(tmp_path / "dest" / "ledger.csv")
    assert row["id"] == "n1" and row["veredito_final"] == "CONTRADIZ"


def test_exportar_ledger_sem_avaliacoes_e_erro(tmp_path):
    with pytest.raises(ValueError):
        led.exportar_ledger(tmp_path / "vazio", tmp_path, tmp_path / "dest")


def test_subcomando_exportar_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("scripts.atestado.common.ROOT", tmp_path)
    _piloto(tmp_path, [_avaliacao("a1", "A2020", "SUSTENTA")])
    assert cli.main(["exportar-ledger", "--escopo", "tese"]) == 0
    assert (tmp_path / "docs" / "atestado" / "ledger.csv").is_file()
    assert "1 linhas" in capsys.readouterr().out
    with (tmp_path / "docs" / "atestado" / "ledger.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle, delimiter=";"))[0]["id"] == "a1"


def test_subcomando_sai_com_codigo_3_para_problema_novo_sem_aceitar_novas(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr("scripts.atestado.common.ROOT", tmp_path)
    _piloto(tmp_path, [_avaliacao("a1", "A2020", "SUSTENTA"), _avaliacao("a2", "B2021", "NAO_SUSTENTA")])
    assert cli.main(["exportar-ledger", "--escopo", "tese"]) == 3
    captured = capsys.readouterr()
    assert "problemas NOVOS (1): a2" in captured.out and "--aceitar-novas" in captured.err
    assert (tmp_path / "docs" / "atestado" / "ledger.csv").is_file()
    assert led.load_pendencias(tmp_path / "docs" / "atestado" / "pendencias.yaml") == []

    assert cli.main(["exportar-ledger", "--escopo", "tese", "--aceitar-novas"]) == 0
    pend = led.load_pendencias(tmp_path / "docs" / "atestado" / "pendencias.yaml")
    assert [(p["id"], p["decisao"]) for p in pend] == [("a2", "pendente")]
    assert cli.main(["exportar-ledger", "--escopo", "tese"]) == 0  # agora a2 já consta
