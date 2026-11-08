"""Registro de invariantes: números centrais conferem com os artefatos e
continuam aparecendo no texto (sem fixar frases).

Dois grupos parametrizados, para que cada falha seja granular:
- ``test_valor_confere_com_artefato``: um caso por (entrada, valor);
- ``test_ocorrencia_no_texto``: um caso por (entrada, ocorrência).
Ocorrências com ``xfail`` documentam inconsistências reais do texto (strict:
se passarem, o teste falha e o marcador deve ser removido).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import registro_invariantes as ri  # noqa: E402
from local_data import DATA_ROOT  # noqa: E402

ri.RAIZ_DADOS = DATA_ROOT

REG = ri.carregar_registro()
CASOS_VALORES = ri.casos_valores(REG)
CASOS_OCORRENCIAS = ri.casos_ocorrencias(REG)


def _params(casos):
    return [
        pytest.param(
            c, id=c.id,
            marks=([pytest.mark.xfail(reason=c.xfail, strict=True)]
                   if c.xfail else []))
        for c in casos
    ]


def _executar(caso):
    try:
        problemas = ri.executar_caso(REG, caso)
    except ri.FonteIndisponivel as exc:
        pytest.skip(ri.MENSAGEM_INDISPONIVEL + str(exc))
    except ri.ErroRegistro as exc:
        pytest.fail(f"erro de registro: {exc}")
    assert not problemas, "\n".join(problemas)


def test_estrutura_do_registro():
    assert ri.verificar_estrutura(REG) == []


def test_registro_cobre_numeros_e_afirmacoes():
    ids = {e["id"] for e in REG.entradas}
    assert {f"N{i:02d}" for i in range(1, 40)} <= ids
    assert {f"C{i:02d}" for i in range(1, 10)} <= ids


@pytest.mark.parametrize("caso", _params(CASOS_VALORES))
def test_valor_confere_com_artefato(caso):
    _executar(caso)


@pytest.mark.parametrize("caso", _params(CASOS_OCORRENCIAS))
def test_ocorrencia_no_texto(caso):
    _executar(caso)


# --- testes da própria biblioteca -----------------------------------------

def test_formatacao_ptbr():
    assert ri.formatar_pt(61696, 0) == "61.696"
    assert ri.formatar_pt(0.870445, 3) == "0,870"
    assert ri.formatar_pt(-5.64846544302, 4) == "-5,6485"
    assert ri.formatar_valor(0.8704453441295547, "pct1", None) == ("87,0", "87.0")
    assert ri.formatar_valor(4163, None, None) == ("4.163", "4,163")
    assert ri.formatar_valor(61449 / 247, None, "1:{:.0f}") == ("1:249", "1:249")


def test_avaliador_seguro():
    assert ri.avaliar("1 - 125/501", lambda n: 0) == pytest.approx(0.7505, abs=1e-4)
    assert ri.avaliar("a / b", {"a": 6, "b": 3}.__getitem__) == 2
    for ruim in ("__import__('os')", "a.b", "[1]", "2 ** 3", "open('x')"):
        with pytest.raises(ri.ErroRegistro):
            ri.avaliar(ruim, {"a": 1}.__getitem__)
    lo, hi = ri.wilson(13, 22)
    assert (round(lo * 100, 1), round(hi * 100, 1)) == (38.7, 76.7)


def test_normalizacao_e_fronteira_numerica():
    assert ri.normalizar(r"PAR~=~2, $k = 96$, 87{,}0\% e \textit{post hoc}") == (
        "PAR=2, k=96, 87,0% e post hoc")
    assert ri.contem("a 215 de 247", "215")
    assert not ri.contem("a 2.156 de", "156")
    assert not ri.contem("valor 215,5", "215")
    assert ri.contem("valor 215, 247", "215")
    assert ri.contem("Hipótese Post Hoc", "post hoc", dobrar=True)


def test_ocorrencia_exige_coocorrencia_no_mesmo_paragrafo(tmp_path, monkeypatch):
    texto = tmp_path / "t.tex"
    texto.write_text("215 dos casos.\n\nO total é 247 vínculos.\n", encoding="utf-8")
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    reg = ri.Registro([{"id": "X", "tipo": "numero", "valores": [],
                        "ocorrencias": []}])
    entrada = reg.por_id["X"]
    assert ri.verificar_ocorrencia(reg, entrada, {"arquivo": "t.tex", "exige": ["215"]}) == []
    falha = ri.verificar_ocorrencia(
        reg, entrada, {"arquivo": "t.tex", "exige": ["215", "247"]})
    assert falha and "247" in falha[0]
    ri.paragrafos.cache_clear()
