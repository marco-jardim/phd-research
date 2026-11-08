"""Registro de invariantes: números centrais conferem com os artefatos e
continuam aparecendo no texto (sem fixar frases).

Dois grupos parametrizados, para que cada falha seja granular:
- ``test_valor_confere_com_artefato``: um caso por (entrada, valor);
- ``test_ocorrencia_no_texto``: um caso por (entrada, ocorrência).
Ocorrências com ``xfail`` documentam inconsistências reais do texto (strict:
se passarem, o teste falha e o marcador deve ser removido).
"""
import shutil
import subprocess
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


# --- F5: fontes versionadas resolvidas pela raiz do repositório --------------

def test_valores_versionados_rodam_sem_data_root(monkeypatch, tmp_path):
    """No CI não há DATA_ROOT: os artefatos versionados têm de ser lidos do checkout."""
    if not (ri.ROOT / "data" / "extrato_resumo.csv").exists():
        pytest.skip("checkout sem os artefatos versionados de data/")
    monkeypatch.setattr(ri, "RAIZ_DADOS", tmp_path / "inexistente")
    reg = ri.carregar_registro()
    executados = 0
    for caso in ri.casos_valores(reg):
        try:
            ri.executar_caso(reg, caso)
        except ri.FonteIndisponivel:
            continue
        except ri.ErroRegistro:
            pass  # executou (e falhou); a falha é do teste granular, não da cobertura
        executados += 1
    assert executados >= 30, f"só {executados} casos de valor rodaram sem DATA_ROOT"


@pytest.mark.skipif(shutil.which("git") is None, reason="git indisponível")
def test_caminho_dado_versionado_usa_repo_e_privado_usa_data_root(tmp_path, monkeypatch):
    repo, privado = tmp_path / "repo", tmp_path / "privado"
    (repo / "data").mkdir(parents=True)
    (privado / "data").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    for nome in ("rastreado.csv", "rastreado_removido.csv"):
        (repo / "data" / nome).write_text("a\n1\n", encoding="utf-8")
        (privado / "data" / nome).write_text("a\n2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "data/rastreado.csv",
                    "data/rastreado_removido.csv"], check=True, capture_output=True)
    (repo / "data" / "rastreado_removido.csv").unlink()  # continua no índice
    (privado / "data" / "so_privado.csv").write_text("a\n3\n", encoding="utf-8")
    monkeypatch.setattr(ri, "ROOT", repo)
    monkeypatch.setattr(ri, "RAIZ_DADOS", privado)
    ri._rastreado.cache_clear()
    try:
        assert ri.caminho_dado("data/rastreado.csv") == repo / "data" / "rastreado.csv"
        # rastreado mas ausente do checkout: indisponível, sem cair na cópia privada
        assert ri.caminho_dado("data/rastreado_removido.csv") == (
            repo / "data" / "rastreado_removido.csv")
        assert ri.caminho_dado("data/so_privado.csv") == privado / "data" / "so_privado.csv"
    finally:
        ri._rastreado.cache_clear()


# --- F10: afirmações ancoradas em polaridade e termos técnicos ---------------

def _verificar(tmp_path, monkeypatch, texto, entrada_id, arquivo_sufixo, tipo="afirmacao",
               oc=None):
    """Aplica uma ocorrência (do registro ou literal) a um texto sintético."""
    if oc is None:
        oc = next(o for o in REG.por_id[entrada_id]["ocorrencias"]
                  if o["arquivo"].endswith(arquivo_sufixo) and not o.get("secao"))
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    (tmp_path / "t.tex").write_text(texto, encoding="utf-8")
    reg = ri.Registro([{"id": "X", "tipo": tipo, "valores": [], "ocorrencias": []}])
    try:
        return ri.verificar_ocorrencia(reg, reg.por_id["X"], dict(oc, arquivo="t.tex"))
    finally:
        ri.paragrafos.cache_clear()


def test_c07_reescrita_passa_e_inversao_falha(tmp_path, monkeypatch):
    for texto in (
        r"O LLM não aceita vínculos sozinho; atua apenas no filtro de \texttt{NONMATCH}.",
        r"O LLM nunca aceita vínculos por conta própria, só filtra NONMATCH.",
        r"O LLM, sem aceitar vínculos, serve de filtro de NONMATCH.",
    ):
        assert _verificar(tmp_path, monkeypatch, texto, "C07", "conclusao.tex") == [], texto
    inversao = r"O LLM aceita vínculos sozinho e também filtra \texttt{NONMATCH}."
    assert _verificar(tmp_path, monkeypatch, inversao, "C07", "conclusao.tex")


def test_c02_reescrita_passa_e_inversao_falha(tmp_path, monkeypatch):
    base = (r"A hipótese seria refutada se O3 não atingisse a recuperação de P com "
            r"pelo menos 25\% {} revisões que O1, pelo limite inferior do IC de 95\%.")
    assert _verificar(tmp_path, monkeypatch, base.format("menos"), "C02", "conclusao.tex") == []
    assert _verificar(tmp_path, monkeypatch, base.format("mais"), "C02", "conclusao.tex")


def test_afirmacoes_nao_fixam_frases_inteiras():
    """Cada âncora de afirmação tem no máximo três palavras (token ou variante de lema)."""
    for e in REG.entradas:
        if e["tipo"] != "afirmacao":
            continue
        for oc in e["ocorrencias"]:
            for chave in ("exige", "exige_todos", "exige_algum"):
                for t in oc.get(chave) or []:
                    for alt in (t if isinstance(t, list) else [t]):
                        assert len(str(alt).split()) <= 3, (e["id"], chave, alt)


def test_afirmacoes_tem_ao_menos_duas_ancoras_independentes():
    # Cada ocorrência de afirmação tem pelo menos duas âncoras independentes
    # (itens de exige_todos, ou um item mais exige_algum), nunca um único token genérico.
    for e in REG.entradas:
        if e["tipo"] != "afirmacao":
            continue
        for oc in e["ocorrencias"]:
            n = len(oc.get("exige_todos") or []) + (1 if oc.get("exige_algum") else 0)
            assert n >= 2, (e["id"], oc["arquivo"])


def test_yaml_documenta_aprovacao_da_autora():
    texto = ri.REGISTRO.read_text(encoding="utf-8")
    assert "aprovação da autora" in texto  # nota de cabeçalho
    linhas = [ln for ln in texto.splitlines() if ln.startswith("  # aprovado_por_autora:")]
    assert len(linhas) >= 9  # uma linha por afirmação C01-C09 alterada


# --- F16: comentários e separadores de parágrafo -------------------------------

def test_linha_so_de_comentario_nao_separa_paragrafos(tmp_path, monkeypatch):
    oc = {"exige": ["215", "247"]}
    junto = "Há 215 casos\n% comentário solto\nem 247 vínculos.\n"
    assert _verificar(tmp_path, monkeypatch, junto, "X", "", "numero", oc) == []
    # comentário recuado também some; comentário entre linhas em branco não junta
    recuado = "Há 215 casos\n   % comentário recuado\nem 247 vínculos.\n"
    assert _verificar(tmp_path, monkeypatch, recuado, "X", "", "numero", oc) == []
    separado = "Há 215 casos\n\n% comentário\n\nem 247 vínculos.\n"
    assert _verificar(tmp_path, monkeypatch, separado, "X", "", "numero", oc)
    # comentário ao fim da linha continua removido sem juntar o que a linha em branco separa
    fim = "Há 215 casos % nota\n\nem 247 vínculos.\n"
    assert _verificar(tmp_path, monkeypatch, fim, "X", "", "numero", oc)
    # \% (porcentagem escapada) não inicia comentário
    pct = r"Há 215 casos, 12\% do total, em 247 vínculos." + "\n"
    assert _verificar(tmp_path, monkeypatch, pct, "X", "", "numero", oc) == []


def test_item_e_par_separam_paragrafos(tmp_path, monkeypatch):
    oc = {"exige": ["215", "247"]}
    itens = "\\begin{itemize}\n\\item 215 casos\n\\item 247 vínculos\n\\end{itemize}\n"
    assert _verificar(tmp_path, monkeypatch, itens, "X", "", "numero", oc)
    um_item = "\\begin{itemize}\n\\item 215 casos e 247 vínculos\n\\end{itemize}\n"
    assert _verificar(tmp_path, monkeypatch, um_item, "X", "", "numero", oc) == []
    par = "215 casos\\par 247 vínculos\n"
    assert _verificar(tmp_path, monkeypatch, par, "X", "", "numero", oc)
    # \paragraph e \parbox não são \par
    nao_par = "215 casos \\paragraph{t} e \\parbox{3cm}{247 vínculos}\n"
    assert _verificar(tmp_path, monkeypatch, nao_par, "X", "", "numero", oc) == []


def test_slides_beamer_usam_o_quadro_como_unidade(tmp_path, monkeypatch):
    oc = {"exige": ["215", "247"]}
    quadro = ("\\begin{frame}{Título}\n\\itens{\\item 215 casos\n\n\\item 247 vínculos\\par}\n"
              "\\end{frame}\n")
    assert _verificar(tmp_path, monkeypatch, quadro, "X", "", "numero", oc) == []
    # quadros distintos não se misturam
    dois = ("\\begin{frame}\n\\item 215 casos\n\\end{frame}\n\n"
            "\\begin{frame}\n\\item 247 vínculos\n\\end{frame}\n")
    assert _verificar(tmp_path, monkeypatch, dois, "X", "", "numero", oc)
    # fora de quadros continua a separação por \item
    fora = "\\item 215 casos\n\\item 247 vínculos\n\\begin{frame}\nnada\n\\end{frame}\n"
    assert _verificar(tmp_path, monkeypatch, fora, "X", "", "numero", oc)


def test_markdown_continua_separado_por_linha_em_branco(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    (tmp_path / "r.md").write_text("Há 215 \\item e 247 vínculos\n\n- 1.000 itens\n",
                                    encoding="utf-8")
    reg = ri.Registro([{"id": "X", "tipo": "numero", "valores": [], "ocorrencias": []}])
    try:
        assert ri.verificar_ocorrencia(
            reg, reg.por_id["X"], {"arquivo": "r.md", "exige": ["215", "247"]}) == []
        assert len(ri.paragrafos("r.md", None)) == 2
    finally:
        ri.paragrafos.cache_clear()
