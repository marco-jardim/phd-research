"""Reconstrução da linha de base da atestação a partir do ledger (fixtures sintéticas, sem API)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reconstruir_baseline_atestado as rb
from scripts.atestado import avaliar as av
from scripts.atestado import ledger as led
from scripts.atestado.prompts import build_evidence_block

MODELO = "claude-sonnet-5-5"
MODELO_ADJ = "claude-opus-5-5"
DATA = "2026-11-10"
ARQUIVO = "text/latex/cap.tex"
CAPITULO = "Capítulo de teste"
CHAVE = "Fonte2020"
REGISTRO = {
    "key": CHAVE,
    "titulo_fonte": "Fonte de teste",
    "fulltext": (
        "A fonte descreve o método de seleção de escore no relacionamento probabilístico de bases de "
        "dados e compara o resultado com a revisão manual de uma amostra."
    ),
    "fulltext_fonte": "local:Fonte2020.txt",
}
TRECHO = "o método de seleção de escore no relacionamento probabilístico de bases de dados"


def _ctx(cid, key=CHAVE, legado=""):
    return {
        "id": cid, "id_legado": legado, "key": key, "file": ARQUIVO, "line": 10, "chapter": CAPITULO,
        "section_path": [], "sentence_text": "Frase que cita a fonte.",
    }


def _aval(cid, veredito, trecho, key=CHAVE, **extra):
    return {
        "id": cid, "key": key, "file": ARQUIVO, "line": 10, "chapter": CAPITULO, "section_path": [],
        "classe": "literatura", "nivel_evidencia": "texto_completo", "evidencia_fingerprint": "fp_antigo",
        "trechos_enviados": [], "veredito_final": veredito, "trecho_fonte": trecho, "justificativa": "",
        "problema": None, "flags": [], "passada1": None, "passada2": None, **extra,
    }


def _linha(ctx, veredito, trecho="", nivel="texto_completo", adjudicado="nao", modelo=MODELO, data=DATA):
    proprio = veredito == led.PROPRIO
    return {
        "id": ctx["id"], "id_legado": ctx["id_legado"], "chave": ctx["key"], "arquivo": ARQUIVO,
        "capitulo": CAPITULO, "veredito_final": veredito, "nivel_evidencia": nivel, "adjudicado": adjudicado,
        "modelo": "" if proprio else modelo, "data": data,
        "hash_avaliacao": "" if proprio else led.hash_avaliacao(modelo, veredito, trecho),
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _cenario(tmp_path, contextos, ledger, atuais, snapshots=None, checkpoint=None, artefatos=None):
    out = tmp_path / "biblio"
    pilot = out / "piloto" / "tese"
    pilot.mkdir(parents=True)
    _jsonl(out / "citacoes.jsonl", contextos)
    (out / "evidencias").mkdir()
    (out / "evidencias" / f"{CHAVE}.json").write_text(json.dumps(REGISTRO, ensure_ascii=False), encoding="utf-8")
    _jsonl(pilot / "avaliacoes.jsonl", atuais)
    for name, rows in (snapshots or {}).items():
        _jsonl(pilot / name, rows)
    meta = {"modelo": MODELO, "data_utc": f"{DATA}T22:06:55+00:00", "adjudicacao": {"modelo": MODELO_ADJ}}
    (pilot / "metadados.json").write_text(json.dumps(meta), encoding="utf-8")
    (pilot / "artefatos.json").write_text(json.dumps(artefatos or []), encoding="utf-8")
    cp = checkpoint or {"versao": 1, "modelo": MODELO, "requisicoes": {}, "lotes": [], "falhas": []}
    (pilot / "checkpoint.json").write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")
    ledger_path = tmp_path / "docs" / "ledger.csv"
    ledger_path.parent.mkdir(parents=True)
    ordenado = sorted(ledger, key=lambda r: (r["arquivo"], r["chave"], r["id"]))
    ledger_path.write_text(led.render_ledger(ordenado), encoding="utf-8", newline="\n")
    return SimpleNamespace(
        out=out, pilot=pilot, ledger=ledger_path, saida=tmp_path / "rec" / "avaliacoes_reconstruido.jsonl",
        backup=tmp_path / "acidental", bib=tmp_path / "sem.bib",
    )


def _rodar(c, *extra):
    return rb.main([
        "--ledger", str(c.ledger), "--out", str(c.out), "--saida", str(c.saida),
        "--backup-dir", str(c.backup), "--bib", str(c.bib), *extra,
    ])


def _ler(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reexportar(c, tmp_path) -> str:
    """O ledger que o exportador gravaria com a saída reconstruída no lugar de avaliacoes.jsonl."""
    verif = tmp_path / "verif"
    verif.mkdir()
    shutil.copy(c.saida, verif / "avaliacoes.jsonl")
    shutil.copy(c.pilot / "metadados.json", verif / "metadados.json")
    shutil.copy(c.pilot / "artefatos.json", verif / "artefatos.json")
    rows, _pendentes = led.build_ledger_rows(verif, c.out)
    return led.render_ledger(rows)


def _cenario_variantes(tmp_path):
    """c1 com três variantes (acidental, trecho errado, certa), n1 sem evidência e p1 obra própria."""
    c1, n1, p1 = _ctx("c1"), _ctx("n1", key="Sem2020"), _ctx("p1", key="Jardim2026x")
    ledger = [
        _linha(c1, "SUSTENTA", TRECHO),
        _linha(n1, "NAO_VERIFICAVEL", "", nivel="nenhum"),
        _linha(p1, led.PROPRIO, nivel="sem_artefato_na_frase"),
    ]
    snapshots = {
        "avaliacoes.pre_e1_lucena.jsonl": [_aval("c1", "SUSTENTA", "outro trecho da fonte, que não é o do ledger")],
        "avaliacoes.pre_cap3.jsonl": [_aval("c1", "SUSTENTA", TRECHO)],
    }
    artefatos = [{"id": "p1", "key": "Jardim2026x", "file": ARQUIVO, "line": 10, "veredito": "SEM_ARTEFATO_NA_FRASE"}]
    atuais = [_aval("c1", "SUSTENTA_PARCIALMENTE", TRECHO)]
    return _cenario(tmp_path, [c1, n1, p1], ledger, atuais, snapshots=snapshots, artefatos=artefatos)


def test_hash_escolhe_a_variante_do_ledger_e_grava_evidencia_atual(tmp_path, capsys):
    c = _cenario_variantes(tmp_path)
    antes = (c.pilot / "avaliacoes.jsonl").read_bytes()

    assert _rodar(c) == 0

    rows = {r["id"]: r for r in _ler(c.saida)}
    assert list(rows) == ["c1", "n1"]  # ordem de citacoes.jsonl; PROPRIO não entra
    c1 = rows["c1"]
    assert (c1["veredito_final"], c1["trecho_fonte"]) == ("SUSTENTA", TRECHO)
    assert c1["reconstruido_de"] == "piloto/tese/avaliacoes.pre_cap3.jsonl"
    assert c1["reaproveitado_de"] == rb.ORIGEM and f"reaproveitado_de={rb.ORIGEM}" in c1["flags"]
    assert c1["evidencia_fingerprint"] == av.evidence_fingerprint(build_evidence_block(REGISTRO, 120000)[0])
    assert c1["evidencia_fingerprint_avaliada"] == "fp_antigo"
    assert c1["fonte_sha256"] == av.source_sha256(REGISTRO, 120000)
    assert (rows["n1"]["veredito_final"], rows["n1"]["flags"]) == ("NAO_VERIFICAVEL", ["sem_evidencia"])
    assert rows["n1"]["reconstruido_de"] == rb.FONTE_PIPELINE
    assert (c.pilot / "avaliacoes.jsonl").read_bytes() == antes  # sem --aplicar nada muda na pasta da tese
    assert not c.backup.exists()
    assert _reexportar(c, tmp_path) == c.ledger.read_text(encoding="utf-8")

    saida = capsys.readouterr().out
    assert "piloto/tese/avaliacoes.pre_cap3.jsonl: 1" in saida
    assert "sem correspondência: 0" in saida


def test_casa_pelo_id_legado_e_grava_o_id_atual(tmp_path):
    c2 = _ctx("novo2", legado="velho2")
    c = _cenario(tmp_path, [c2], [_linha(c2, "SUSTENTA", TRECHO)], [_aval("velho2", "SUSTENTA", TRECHO)])

    assert _rodar(c) == 0

    [row] = _ler(c.saida)
    assert (row["id"], row["id_legado"], row["veredito_final"]) == ("novo2", "velho2", "SUSTENTA")
    assert row["reconstruido_de"] == "piloto/tese/avaliacoes.jsonl"
    assert _reexportar(c, tmp_path) == c.ledger.read_text(encoding="utf-8")


def test_linha_sem_correspondencia_sai_com_codigo_2_e_grava_parcial(tmp_path, capsys):
    c1, n1 = _ctx("c1"), _ctx("n1", key="Sem2020")
    ledger = [
        _linha(c1, "SUSTENTA", "trecho que nenhuma avaliação guardada citou"),
        _linha(n1, "NAO_VERIFICAVEL", "", nivel="nenhum"),
    ]
    c = _cenario(tmp_path, [c1, n1], ledger, [_aval("c1", "SUSTENTA", TRECHO)])
    antes = (c.pilot / "avaliacoes.jsonl").read_bytes()

    assert _rodar(c) == 2

    assert [r["id"] for r in _ler(c.saida)] == ["n1"]
    saida = capsys.readouterr().out
    assert "sem correspondência: 1" in saida
    [linha] = [x for x in saida.splitlines() if x.strip().startswith("c1 ")]
    assert "nenhuma candidata com o mesmo hash_avaliacao" in linha
    assert "vereditos das candidatas: SUSTENTA" in linha

    # --aplicar com reconstrução incompleta não toca a pasta da tese nem faz backup
    assert _rodar(c, "--aplicar") == 2
    assert (c.pilot / "avaliacoes.jsonl").read_bytes() == antes
    assert not c.backup.exists()


def test_aplicar_faz_backup_antes_de_regravar_e_nao_sobrescreve_backup(tmp_path, capsys):
    c = _cenario_variantes(tmp_path)
    originais = {name: (c.pilot / name).read_bytes() for name in rb.ARQUIVOS_BACKUP}

    with pytest.raises(SystemExit):
        _rodar(c, "--saida", str(c.pilot / "avaliacoes.jsonl"))
    assert (c.pilot / "avaliacoes.jsonl").read_bytes() == originais["avaliacoes.jsonl"]

    assert _rodar(c, "--aplicar") == 0

    for name, conteudo in originais.items():
        assert (c.backup / name).read_bytes() == conteudo
    assert _ler(c.pilot / "avaliacoes.jsonl") == _ler(c.saida)
    assert (c.pilot / "checkpoint.json").read_bytes() == originais["checkpoint.json"]
    capsys.readouterr()

    # segunda aplicação: o backup do estado acidental fica intacto e a origem da reconstrução é preservada
    assert _rodar(c, "--aplicar") == 0
    assert (c.backup / "avaliacoes.jsonl").read_bytes() == originais["avaliacoes.jsonl"]
    assert "mantidos) avaliacoes.jsonl, checkpoint.json, metadados.json" in capsys.readouterr().out
    rows = {r["id"]: r for r in _ler(c.pilot / "avaliacoes.jsonl")}
    assert rows["c1"]["reconstruido_de"] == "piloto/tese/avaliacoes.pre_cap3.jsonl"


def test_remonta_passadas_e_adjudicacao_do_checkpoint(tmp_path, capsys):
    c1, c3 = _ctx("c1"), _ctx("c3")
    ledger = [
        _linha(c1, "SUSTENTA", TRECHO),
        _linha(c3, "SUSTENTA_PARCIALMENTE", TRECHO, adjudicado="sim", modelo=MODELO_ADJ, data="2026-11-08"),
    ]
    checkpoint = {
        "versao": 1, "modelo": MODELO, "lotes": [], "falhas": [],
        "requisicoes": {
            "p1-teste": {
                "key": CHAVE, "passada": 1, "ids": ["c1", "c3"], "modo": "batch", "uso": {}, "erro": None,
                "itens": [
                    {"id": "c1", "veredito": "SUSTENTA", "trecho_fonte": TRECHO, "justificativa": "j1",
                     "problema": None, "nivel_evidencia": "texto_completo"},
                    {"id": "c3", "veredito": "NAO_SUSTENTA", "trecho_fonte": "", "justificativa": "j3",
                     "problema": "afirma_mais_que_a_fonte"},
                ],
            },
            "p2-teste": {
                "key": CHAVE, "passada": 2, "ids": ["c3"], "modo": "batch", "uso": {}, "erro": None,
                "itens": [{"id": "c3", "veredito": "SUSTENTA", "trecho_fonte": TRECHO, "justificativa": "j3b",
                           "problema": None}],
            },
        },
        "adjudicacao": {
            "requisicoes": {
                "adj-teste": {
                    "key": CHAVE, "passada": 3, "ids": ["c3"], "modo": "batch", "uso": {}, "erro": None,
                    "itens": [{"id": "c3", "veredito": "SUSTENTA_PARCIALMENTE", "trecho_fonte": TRECHO,
                               "justificativa": "ja", "problema": "afirma_mais_que_a_fonte", "confianca": "alta"}],
                },
            },
            "lotes": [{"id": "msgbatch_teste", "passada": 3, "baixado": True, "custom_ids": ["adj-teste"],
                       "enviado_em": "2026-11-08T03:00:00+00:00"}],
            "falhas": [],
        },
    }
    atuais = [_aval("c1", "NAO_VERIFICAVEL", "")]  # veredito acidental, hash diferente
    c = _cenario(tmp_path, [c1, c3], ledger, atuais, checkpoint=checkpoint)

    assert _rodar(c) == 0

    rows = {r["id"]: r for r in _ler(c.saida)}
    assert rows["c1"]["passada1"]["veredito"] == "SUSTENTA"
    c3_row = rows["c3"]
    assert (c3_row["veredito_final"], c3_row["veredito_pre_adjudicacao"]) == ("SUSTENTA_PARCIALMENTE", "DIVERGENTE")
    assert (c3_row["adjudicacao"]["modelo"], c3_row["adjudicacao"]["batch_id"]) == (MODELO_ADJ, "msgbatch_teste")
    assert {r["reconstruido_de"] for r in rows.values()} == {"checkpoint:piloto/tese/checkpoint.json"}
    assert _reexportar(c, tmp_path) == c.ledger.read_text(encoding="utf-8")
    assert "checkpoint:piloto/tese/checkpoint.json: 2" in capsys.readouterr().out
