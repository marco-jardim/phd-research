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


def test_ledger_so_aceita_proprio_para_obra_propria(tese):
    indevidos = led.proprios_indevidos(tese["ledger"], tese["own"])
    assert not indevidos, f"PROPRIO em chave que não é de autoria 'Jardim': {[r['chave'] for r in indevidos]}"


def test_ledger_sem_veredito_vazio_ou_pendente(tese):
    assert not led.linhas_sem_veredito(tese["ledger"])


# ---------------------------------------------------------------------------
# Lógica do portão (sintético)
# ---------------------------------------------------------------------------
def _linha(cid, chave="A2020", veredito="SUSTENTA"):
    return {
        "id": cid, "chave": chave, "arquivo": "x.tex", "capitulo": "C", "veredito_final": veredito,
        "nivel_evidencia": "resumo", "adjudicado": "nao", "modelo": "m", "data": "2026-11-01",
    }


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


def _piloto(tmp_path, avaliacoes, artefatos=None):
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
    (out / "citacoes.jsonl").write_text(
        json.dumps({"id": "p1", "key": "Jardim2025", "chapter": "Métodos"}) + "\n", encoding="utf-8"
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

    summary = led.exportar_ledger(pdir, out, dest)

    texto = (dest / "ledger.csv").read_text(encoding="utf-8")
    assert "SECRET" not in texto and '"' not in texto
    assert texto.splitlines()[0] == "id;chave;arquivo;capitulo;veredito_final;nivel_evidencia;adjudicado;modelo;data"
    rows = {r["id"]: r for r in led.load_ledger(dest / "ledger.csv")}
    assert set(rows) == {"a1", "a2", "p1"}
    assert dict(rows["a1"]) == {
        "id": "a1", "chave": "Silva2020", "arquivo": "text/latex/cap.tex", "capitulo": "Introdução",
        "veredito_final": "SUSTENTA", "nivel_evidencia": "texto_completo", "adjudicado": "nao",
        "modelo": "claude-sonnet-5-5", "data": "2026-11-05",
    }
    assert (rows["a2"]["adjudicado"], rows["a2"]["modelo"], rows["a2"]["data"]) == (
        "sim", "claude-opus-5-5", "2026-11-06",
    )
    assert (rows["p1"]["veredito_final"], rows["p1"]["capitulo"], rows["p1"]["modelo"]) == ("PROPRIO", "Métodos", "")
    assert summary["pendentes_sem_veredito"] == ["a3"]

    pend = yaml.safe_load((dest / "pendencias.yaml").read_text(encoding="utf-8"))["pendencias"]
    assert pend == [{"id": "a2", "chave": "Lima2019", "arquivo": "text/latex/cap.tex", "decisao": "pendente"}]
    assert "SECRET" not in (dest / "pendencias.yaml").read_text(encoding="utf-8")


def test_reexportar_preserva_decisoes_e_aponta_pendencias_novas(tmp_path):
    out, pdir = _piloto(tmp_path, [_avaliacao("a1", "A2020", "CONTRADIZ")])
    dest = tmp_path / "docs" / "atestado"
    led.exportar_ledger(pdir, out, dest)
    texto = (dest / "pendencias.yaml").read_text(encoding="utf-8").replace("decisao: pendente", "decisao: aceita, erro de citação")
    (dest / "pendencias.yaml").write_text(texto, encoding="utf-8")

    out, pdir = _piloto(
        tmp_path / "2", [_avaliacao("a1", "A2020", "CONTRADIZ"), _avaliacao("a9", "B2021", "NAO_SUSTENTA")]
    )
    summary = led.exportar_ledger(pdir, out, dest)

    pend = {p["id"]: p for p in led.load_pendencias(dest / "pendencias.yaml")}
    assert pend["a1"]["decisao"] == "aceita, erro de citação"
    assert pend["a9"]["decisao"] == "pendente"
    assert summary["pendencias_novas"] == ["a9"]


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
