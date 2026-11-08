"""Registro de invariantes: números centrais conferem com os artefatos e
continuam aparecendo no texto (sem fixar frases).

Dois grupos parametrizados, para que cada falha seja granular:
- ``test_valor_confere_com_artefato``: um caso por (entrada, valor);
- ``test_ocorrencia_no_texto``: um caso por (entrada, ocorrência).
Ocorrências com ``xfail`` documentam inconsistências reais do texto (strict:
se passarem, o teste falha e o marcador deve ser removido).

Cada ocorrência tem uma ``vigencia`` (``sempre``, ``pre_revisao``, ``pos_revisao``,
``informativa``) que a liga ao estado de ``tests/data/arquivos_revisados.yaml``: fora de
vigor o caso é ignorado (``skip``); ``informativa`` nunca falha (``skip`` com os problemas).
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
    # Informativa nunca falha: sem xfail estrito (um caso que passe não é XPASS).
    return [
        pytest.param(
            c, id=c.id,
            marks=([pytest.mark.xfail(reason=c.xfail, strict=True)]
                   if c.xfail and c.vigencia != "informativa" else []))
        for c in casos
    ]


def _executar(caso):
    try:
        estado, problemas = ri.executar_vigente(REG, caso)
    except ri.FonteIndisponivel as exc:
        pytest.skip(ri.MENSAGEM_INDISPONIVEL + str(exc))
    except ri.ErroRegistro as exc:
        pytest.fail(f"erro de registro: {exc}")
    if estado == "inativa":
        pytest.skip(f"vigência {caso.vigencia}: fora de vigor para o estado atual de "
                    "arquivos_revisados.yaml")
    if estado == "informativa":
        if problemas:
            pytest.skip("informativa (não falha): " + "; ".join(problemas))
        return
    assert not problemas, "\n".join(problemas)


def test_estrutura_do_registro():
    assert ri.verificar_estrutura(REG) == []


ANCORAS_NOVAS = ("C10", "C01r", "C11", "C12", "C13", "C14", "C15", "C16",
                 "C08r", "C02r", "C03r", "C05r", "C09r")


def test_registro_cobre_numeros_e_afirmacoes():
    ids = {e["id"] for e in REG.entradas}
    assert {f"N{i:02d}" for i in range(1, 44)} <= ids
    assert {f"C{i:02d}" for i in range(1, 10)} <= ids
    assert set(ANCORAS_NOVAS) <= ids


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


# --- mesma_frase: coocorrência do valor com a entidade na mesma frase ---------

def _reg_mesma_frase():
    return ri.Registro([{
        "id": "N99", "tipo": "numero", "ocorrencias": [],
        "valores": [
            {"nome": "rec_o3", "fonte": {"arquivo": "x.csv"}, "texto_pt": "125"},
            {"nome": "vpp_o3", "fonte": {"arquivo": "x.csv"}, "texto_pt": "87,5"},
            {"nome": "total", "fonte": {"arquivo": "x.csv"}, "texto_pt": "1.234"},
        ]}])


def _checar_frase(tmp_path, monkeypatch, texto, oc):
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    (tmp_path / "t.tex").write_text(texto, encoding="utf-8")
    reg = _reg_mesma_frase()
    try:
        return ri.verificar_ocorrencia(reg, reg.por_id["N99"], dict(oc, arquivo="t.tex"))
    finally:
        ri.paragrafos.cache_clear()


OC_O3 = {"exige": ["rec_o3"], "mesma_frase": ["O3", "vigilância"]}


def test_mesma_frase_valor_na_frase_certa_passa(tmp_path, monkeypatch):
    texto = "Na vigilância, o O3 recuperou 125 pares. O1 recuperou 87 pares.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, OC_O3) == []


def test_mesma_frase_valor_em_frase_sem_a_entidade_falha(tmp_path, monkeypatch):
    texto = "O O3 opera na vigilância. O total foi de 125 pares.\n"
    falha = _checar_frase(tmp_path, monkeypatch, texto, OC_O3)
    assert falha and "125" in falha[0] and "mesma frase" in falha[0]


def test_mesma_frase_valor_trocado_de_entidade_falha(tmp_path, monkeypatch):
    texto = "Na vigilância, o O1 recuperou 125 pares. O O3 recuperou 87 pares.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, OC_O3)
    # ponto e vírgula também separa: o 125 fica na oração do O1
    texto = "Na vigilância, o O1 recuperou 125 pares; o O3 recuperou 87 pares.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, OC_O3)
    # sem o campo, o registro antigo continua passando com o mesmo texto
    assert _checar_frase(tmp_path, monkeypatch, texto, {"exige": ["rec_o3"]}) == []


def test_mesma_frase_aceita_alternativas(tmp_path, monkeypatch):
    oc = {"exige": ["rec_o3"],
          "mesma_frase": [["O3", "O-3"], ["vigilância", "vigilancia"]]}
    assert _checar_frase(
        tmp_path, monkeypatch, "Na vigilancia, o O-3 recuperou 125 pares.\n", oc) == []
    assert _checar_frase(
        tmp_path, monkeypatch, "Na vigilância, o O3 recuperou 125 pares.\n", oc) == []
    assert _checar_frase(
        tmp_path, monkeypatch, "Na confirmação, o O3 recuperou 125 pares.\n", oc)


def test_mesma_frase_exige_todos_os_termos(tmp_path, monkeypatch):
    texto = "Na confirmação, o O3 recuperou 125 pares. A vigilância é outro modo.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, OC_O3)


def test_mesma_frase_cada_valor_precisa_da_sua_frase(tmp_path, monkeypatch):
    oc = {"exige": ["rec_o3", "vpp_o3"], "mesma_frase": ["O3"]}
    ok = "O O3 recuperou 125 pares. Para o O3, a VPP foi de 87,5\\%.\n"
    assert _checar_frase(tmp_path, monkeypatch, ok, oc) == []
    ruim = "O O3 recuperou 125 pares. A VPP do O1 foi de 87,5\\%.\n"
    falha = _checar_frase(tmp_path, monkeypatch, ruim, oc)
    assert falha and "87,5" in falha[0]


def test_mesma_frase_nao_quebra_em_abreviacao_nem_decimal(tmp_path, monkeypatch):
    texto = "O O3 em vigilância (cf. Tab. 2) recuperou 125 pares com VPP de 87,5\\%.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, OC_O3) == []
    texto = "Na vigilância, o O3 teve 1.234 pares e VPP de 87,5\\%.\n"
    oc = {"exige": ["total", "vpp_o3"], "mesma_frase": ["O3", "vigilância"]}
    assert _checar_frase(tmp_path, monkeypatch, texto, oc) == []


def test_mesma_frase_so_vale_para_valores_do_registro(tmp_path, monkeypatch):
    # token literal (não é nome de valor) continua exigindo só o parágrafo
    oc = {"exige": ["rec_o3", "pares"], "mesma_frase": ["O3"]}
    texto = "O O3 recuperou 125. Foram os pares avaliados.\n"
    assert _checar_frase(tmp_path, monkeypatch, texto, oc) == []


def test_mesma_frase_sem_valor_exigido_e_erro_de_registro(tmp_path, monkeypatch):
    oc = {"exige": ["pares"], "mesma_frase": ["O3"]}
    with pytest.raises(ri.ErroRegistro):
        _checar_frase(tmp_path, monkeypatch, "O O3 recuperou os pares.\n", oc)


def test_estrutura_valida_mesma_frase():
    def reg_com(mf):
        return ri.Registro([{
            "id": "N98", "tipo": "numero",
            "valores": [{"nome": "v", "fonte": {"arquivo": "x.csv"}, "texto_pt": "1"}],
            "ocorrencias": [{"arquivo": "t.tex", "exige": ["v"], "mesma_frase": mf}]}])

    assert ri.verificar_estrutura(reg_com(["O3", ["a", "b"]])) == []
    assert ri.verificar_estrutura(reg_com([]))
    assert ri.verificar_estrutura(reg_com("O3"))
    assert ri.verificar_estrutura(reg_com([["a", 3]]))


def test_segmentar_frases():
    seg = ri.segmentar_frases
    assert seg("Foram 125 pares. Em O3 não houve.") == ["Foram 125 pares.", "Em O3 não houve."]
    assert seg("Ver Fig. 3 e Tab. 2. Fim.") == ["Ver Fig. 3 e Tab. 2.", "Fim."]
    assert seg("Silva et al. mostraram 3,5 vezes. Outro.") == [
        "Silva et al. mostraram 3,5 vezes.", "Outro."]
    assert seg("Total de 1.234,5 pares. Fim") == ["Total de 1.234,5 pares.", "Fim"]
    assert seg("Foi 125. O3 não.") == ["Foi 125.", "O3 não."]
    assert seg("Isso, i.e. aquilo, vale. Sim? Não! Talvez") == [
        "Isso, i.e. aquilo, vale.", "Sim?", "Não!", "Talvez"]
    assert seg("A. B. Silva fez 2 testes etc. e voltou.") == [
        "A. B. Silva fez 2 testes etc. e voltou."]
    assert seg("um; dois") == ["um;", "dois"]
    assert seg("") == []


# --- vigencia: ocorrências ligadas ao estado de arquivos_revisados.yaml -------------

def _e_slide(oc):
    return oc["arquivo"].startswith("text/slides/")


def test_slides_e_roteiro_sao_todos_informativos():
    achadas = 0
    for e in REG.entradas:
        for oc in e["ocorrencias"]:
            if _e_slide(oc):
                achadas += 1
                assert oc.get("vigencia") == "informativa", (e["id"], oc["arquivo"])
    assert achadas > 0


def test_ancoras_antigas_c01_a_c09_sao_pre_revisao_fora_dos_slides():
    for i in range(1, 10):
        for oc in REG.por_id[f"C{i:02d}"]["ocorrencias"]:
            esperado = "informativa" if _e_slide(oc) else "pre_revisao"
            assert oc.get("vigencia") == esperado, (i, oc["arquivo"])


def test_ancoras_novas_sao_aprovadas_afirmacoes_pos_revisao_sem_slides():
    for cid in ANCORAS_NOVAS:
        e = REG.por_id[cid]
        assert e["tipo"] == "afirmacao", cid
        assert str(e["aprovado_por_autora"]).startswith("2026-11-08"), cid
        if cid in ("C01r", "C12", "C08r"):
            assert "revisada em 2026-11-08 (F3)" in str(e["aprovado_por_autora"]), cid
        assert e["ocorrencias"], cid
        for oc in e["ocorrencias"]:
            assert oc.get("vigencia") == "pos_revisao", (cid, oc["arquivo"])
            assert not _e_slide(oc), (cid, oc["arquivo"])
            assert len(oc.get("exige_todos") or []) >= 2, (cid, oc["arquivo"])


def test_ancoras_novas_aparecem_nos_arquivos_previstos():
    def arquivos(cid):
        return {(Path(oc["arquivo"]).name, oc.get("secao")) for oc in REG.por_id[cid]["ocorrencias"]}

    for cid in ("C10", "C01r", "C11", "C12", "C13", "C14", "C08r", "C03r", "C05r", "C09r"):
        assert ("tese.tex", "resumo") in arquivos(cid), cid
        assert ("tese.tex", "abstract") in arquivos(cid), cid
        assert ("conclusao.tex", None) in arquivos(cid), cid
    for cid in ("C10", "C01r", "C03r"):
        assert {("capitulo1.tex", None), ("capitulo3.tex", None)} <= arquivos(cid), cid
    for cid in ("C01r", "C11", "C12", "C13", "C08r", "C02r"):
        assert ("estudo4_gzcmd.tex", None) in arquivos(cid), cid
    assert ("estudo5_equidade.tex", None) in arquivos("C14")
    assert ("estudo1_comparador.tex", None) in arquivos("C15")
    assert ("estudo2_classificadores.tex", None) in arquivos("C16")
    assert {("estudo2_classificadores.tex", None), ("estudo3_estrategias.tex", None)} <= arquivos("C05r")


def test_c11_e_c01r_carregam_a_ressalva_do_ic_acoplada():
    marcas = ("IC", "CI", "zero", "intervalo", "interval")
    for oc in REG.por_id["C11"]["ocorrencias"]:
        algum = [str(t) for t in oc["exige_algum"]]
        assert any(m in t for t in algum for m in marcas), oc["arquivo"]
    for oc in REG.por_id["C01r"]["ocorrencias"]:
        if oc.get("secao") or oc["arquivo"].endswith(("estudo4_gzcmd.tex", "conclusao.tex")):
            tokens = [str(a) for t in oc["exige_todos"] for a in (t if isinstance(t, list) else [t])]
            assert any("F4" in t or "fusões" in t or "fusion" in t for t in tokens) and any("zero" in t or "IC" in t or "CI" in t for t in tokens)
        assert oc["exige_algum"], oc["arquivo"]  # polaridade (ordenação e não automação)


def test_numeros_novos_n40_em_diante_tem_fonte_e_mesma_frase_pos_revisao():
    for i in range(40, 45):
        e = REG.por_id[f"N{i}"]
        assert e["tipo"] == "numero"
        for v in e["valores"]:
            assert v["fonte"]["arquivo"].startswith("data/"), v["nome"]
            assert v["fonte"].get("coluna"), v["nome"]
            assert v.get("texto_pt"), v["nome"]
        assert e["ocorrencias"]
        for oc in e["ocorrencias"]:
            assert oc.get("vigencia") == "pos_revisao"
            assert oc.get("mesma_frase")


def test_ocorrencias_pos_revisao_de_numeros_tem_mesma_frase():
    n = 0
    for e in REG.entradas:
        if e["tipo"] != "numero":
            continue
        for oc in e["ocorrencias"]:
            if oc.get("vigencia") == "pos_revisao":
                n += 1
                assert oc.get("mesma_frase"), (e["id"], oc["arquivo"])
    assert n > 0


def test_numeros_das_ancoras_ficam_na_mesma_frase_da_entidade():
    """Exemplos pedidos: 125 com O3 e vigilância; 501 com O1; 1.203 com F4; 16,0 com VPP e LLM."""
    def termos(entrada, valor):
        achados = [oc["mesma_frase"] for oc in REG.por_id[entrada]["ocorrencias"]
                   if oc.get("vigencia") == "pos_revisao" and valor in (oc.get("exige") or [])]
        assert achados, (entrada, valor)
        return achados

    assert ["O3", "vigilância"] in termos("N20", "k_O3_vig")
    assert ["O1"] in termos("N20", "k_O1_vig")
    assert ["F4"] in termos("N26", "boot_mediana_meta36")
    assert ["VPP", "LLM"] in termos("N16", "vpp_llm_vig")


def test_estado_vigencia_segue_arquivos_revisados():
    rev = frozenset({"text/latex/capitulo1.tex", "text/latex/tese.tex#resumo"})

    def oc(vigencia, arquivo="text/latex/capitulo1.tex", secao=None):
        base = {"arquivo": arquivo, "vigencia": vigencia}
        return dict(base, secao=secao) if secao else base

    assert ri.estado_vigencia({"arquivo": "text/latex/capitulo1.tex"}, rev) == "ativa"
    assert ri.estado_vigencia(oc("sempre"), rev) == "ativa"
    assert ri.estado_vigencia(oc("informativa"), rev) == "informativa"
    # arquivo revisado: pre_revisao deixa de valer e pos_revisao passa a valer
    assert ri.estado_vigencia(oc("pre_revisao"), rev) == "inativa"
    assert ri.estado_vigencia(oc("pos_revisao"), rev) == "ativa"
    # arquivo ainda não revisado: o inverso
    outro = "text/latex/conclusao.tex"
    assert ri.estado_vigencia(oc("pre_revisao", outro), rev) == "ativa"
    assert ri.estado_vigencia(oc("pos_revisao", outro), rev) == "inativa"
    # tese.tex: a vigência vale pela chave tese.tex#resumo / tese.tex#abstract
    tese = "text/latex/tese.tex"
    assert ri.estado_vigencia(oc("pos_revisao", tese, "resumo"), rev) == "ativa"
    assert ri.estado_vigencia(oc("pos_revisao", tese, "abstract"), rev) == "inativa"
    assert ri.estado_vigencia(oc("pre_revisao", tese, "resumo"), rev) == "inativa"
    assert ri.estado_vigencia(oc("pre_revisao", tese, "abstract"), rev) == "ativa"
    # o arquivo inteiro na lista vale para as duas seções
    inteiro = frozenset({tese})
    assert ri.estado_vigencia(oc("pos_revisao", tese, "abstract"), inteiro) == "ativa"
    with pytest.raises(ri.ErroRegistro):
        ri.estado_vigencia(oc("depois"), rev)


def test_chaves_revisadas_aceita_secao_e_terminologia_a_ignora(tmp_path):
    import terminologia

    lista = tmp_path / "revisados.yaml"
    lista.write_text("- text/latex/estudo4_gzcmd.tex\n- text/latex/tese.tex#resumo\n",
                     encoding="utf-8")
    assert ri.chaves_revisadas(lista) == frozenset(
        {"text/latex/estudo4_gzcmd.tex", "text/latex/tese.tex#resumo"})
    # chave com seção não é arquivo: não entra na lista de modo estrito nem no lint
    assert terminologia._revisados(lista) == frozenset(
        {(terminologia.REPO / "text/latex/estudo4_gzcmd.tex").resolve()})
    vazia = tmp_path / "vazia.yaml"
    vazia.write_text("[]\n", encoding="utf-8")
    assert ri.chaves_revisadas(vazia) == frozenset()
    assert isinstance(ri.chaves_revisadas(), frozenset)  # a lista real do repositório


def test_executar_vigente_respeita_a_vigencia(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    (tmp_path / "t.tex").write_text("Texto sem o token procurado.\n", encoding="utf-8")
    lista = tmp_path / "revisados.yaml"
    monkeypatch.setattr(ri, "ARQUIVOS_REVISADOS", lista)
    ocs = [{"arquivo": "t.tex", "vigencia": v, "exige": ["xyz"]}
           for v in ("sempre", "pre_revisao", "pos_revisao", "informativa")]
    reg = ri.Registro([{"id": "V", "tipo": "numero", "valores": [], "ocorrencias": ocs}])
    casos = ri.casos_ocorrencias(reg)
    assert [c.vigencia for c in casos] == ["sempre", "pre_revisao", "pos_revisao", "informativa"]
    try:
        lista.write_text("[]\n", encoding="utf-8")
        r = {c.vigencia: ri.executar_vigente(reg, c) for c in casos}
        assert r["sempre"][0] == "ativa" and r["sempre"][1]
        assert r["pre_revisao"][0] == "ativa" and r["pre_revisao"][1]
        assert r["pos_revisao"] == ("inativa", [])
        assert r["informativa"][0] == "informativa" and r["informativa"][1]
        status = {c.vigencia: st for c, st, _ in ri.executar_tudo(reg)}
        assert status == {"sempre": "falha", "pre_revisao": "falha",
                          "pos_revisao": "inativo", "informativa": "informativa"}

        lista.write_text("- t.tex\n", encoding="utf-8")
        r = {c.vigencia: ri.executar_vigente(reg, c) for c in casos}
        assert r["sempre"][0] == "ativa" and r["sempre"][1]
        assert r["pre_revisao"] == ("inativa", [])
        assert r["pos_revisao"][0] == "ativa" and r["pos_revisao"][1]
        assert r["informativa"][0] == "informativa" and r["informativa"][1]
        status = {c.vigencia: st for c, st, _ in ri.executar_tudo(reg)}
        assert status == {"sempre": "falha", "pre_revisao": "inativo",
                          "pos_revisao": "falha", "informativa": "informativa"}
    finally:
        ri.paragrafos.cache_clear()


def test_informativa_com_xfail_nao_vira_xpass(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "ROOT", tmp_path)
    ri.paragrafos.cache_clear()
    (tmp_path / "t.tex").write_text("Aqui há xyz.\n", encoding="utf-8")
    lista = tmp_path / "revisados.yaml"
    lista.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(ri, "ARQUIVOS_REVISADOS", lista)
    oc = {"arquivo": "t.tex", "vigencia": "informativa", "exige": ["xyz"], "xfail": "I0"}
    reg = ri.Registro([{"id": "V", "tipo": "numero", "valores": [], "ocorrencias": [oc]}])
    try:
        (caso, status, detalhe), = ri.executar_tudo(reg)
    finally:
        ri.paragrafos.cache_clear()
    assert status == "informativa" and detalhe == ""


def test_estrutura_recusa_vigencia_invalida():
    reg = ri.Registro([{
        "id": "N97", "tipo": "numero",
        "valores": [{"nome": "v", "fonte": {"arquivo": "x.csv"}, "texto_pt": "1"}],
        "ocorrencias": [{"arquivo": "t.tex", "exige": ["v"], "vigencia": "depois"}]}])
    assert any("vigencia inválida" in p for p in ri.verificar_estrutura(reg))
    for ok in ri.VIGENCIAS:
        reg_ok = ri.Registro([{
            "id": "N96", "tipo": "numero",
            "valores": [{"nome": "v", "fonte": {"arquivo": "x.csv"}, "texto_pt": "1"}],
            "ocorrencias": [{"arquivo": "t.tex", "exige": ["v"], "vigencia": ok}]}])
        assert ri.verificar_estrutura(reg_ok) == []


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
