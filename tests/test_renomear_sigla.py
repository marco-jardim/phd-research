"""Testes de scripts/renomear_sigla.py (renomeação segura da sigla P) sem rede: o cliente da API é falso."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import renomear_sigla as rs
from scripts import rotulos_apresentacao as rot
from scripts.atestado.common import read_jsonl

AMOSTRA = "text/latex/amostra.tex"
PGF = "text/latex/figures/curva.pgf"
GERADOR = "scripts/gerador.py"
PINO = "tests/test_amostra.py"
SLIDES = "text/slides/defesa.tex"

TEX = r"""\section{Resultados}

\subsection{Arranjo sequencial P}

O braço P recupera 23 pares verdadeiros com 96 revisões na vigilância do estudo.

A sensibilidade de P-Platt foi de 87,0\% e a de P-PAR2-grey, de 88,7\% no mesmo modo.

A união $P\cup B1$ teve 69 potenciais pares pendentes no suplemento.

Com revisor perfeito, P, \texttt{MATCH} ao humano, recupera 22 pares; P com \texttt{MATCH} ao humano, 21.

\begin{itemize}
\item[P] arranjo sequencial recalibrado.
\item[P-Platt] sensibilidade com Platt scaling.
\end{itemize}

Seja $q = P(y=1\mid s)$ a probabilidade condicional; veja o Capítulo~\ref{cap:foo-P-bar} \label{tab:foo-P-bar}.

A palavra P\'os-gradua\c{c}\~ao e o arquivo \path{data/foo-P-bar.csv} somam 12 itens. % comentário com P aqui

Percentil P~90 do tempo e Jardim P. e Silva citados; PAR PPV PRL NP, P10 e P90, Proposição, p minúsculo.

Ambíguos: $P$ e \texttt{P} no apêndice com 5 linhas.
"""

PGF_TXT = "\\pgftext[left]{P=0,95}\n\\pgftext[left]{Curva de P}\n"

GERADOR_TXT = '''"""Gerador de teste: o braço P aparece só aqui na docstring."""
import re

NOTA = "P: arranjo sequencial recalibrado"
ROTULO = {"P": "P"}
SELETOR = re.compile(r"(?P<nome>\\d+)")


def filtra(df):
    a = df[df["braco"] == "P"]
    b = df.braco.eq("P")
    return a, b
'''

PINO_TXT = '''def test_pino():
    texto = "o braço P recupera pares"
    assert "braço P recupera" in texto
'''

SLIDES_TXT = "O braço P nos slides.\n"

BLOCO = ("Texto de preenchimento longo para que o contexto seja idêntico nos dois lados: o braço P "
         "recupera pares verdadeiros nesta coorte de teste, com folga suficiente para que o contexto à direita "
         "também seja idêntico nas duas ocorrências do texto.\n\n")

# (antes, token, classe) das 19 candidatas de TEX
ESPERADO_TEX = [
    ("Arranjo sequencial ", "P", "A7"),
    ("O braço ", "P", "A1"),
    ("sensibilidade de ", "P-Platt", "A2"),
    ("e a de ", "P-PAR2-grey", "A3"),
    ("A união $", "P", "A4"),
    ("Com revisor perfeito, ", "P", "A5"),
    ("recupera 22 pares; ", "P", "A5"),
    ("\\item[", "P", "A6"),
    ("\\item[", "P-Platt", "A6"),
    ("Seja $q = ", "P", "N1"),
    ("\\ref{cap:foo-", "P", "N3"),
    ("\\label{tab:foo-", "P", "N3"),
    ("A palavra ", "P", "N2"),
    ("\\path{data/foo-", "P", "N4"),
    ("% comentário com ", "P", "N9"),
    ("Percentil ", "P", "N6"),
    ("Jardim ", "P", "N11"),
    ("Ambíguos: $", "P", rs.AMBIGUA),
    ("e \\texttt{", "P", rs.AMBIGUA),
]


def _escrever(raiz: Path, rel: str, texto: str, crlf: bool = False) -> None:
    p = raiz / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    _escrever(tmp_path, AMOSTRA, TEX)
    _escrever(tmp_path, PGF, PGF_TXT)
    _escrever(tmp_path, GERADOR, GERADOR_TXT)
    _escrever(tmp_path, PINO, PINO_TXT)
    _escrever(tmp_path, SLIDES, SLIDES_TXT)
    return tmp_path


def _inventario(raiz: Path, arquivos: list[str] | None = None) -> dict:
    return rs.inventariar(raiz, arquivos or [AMOSTRA, PGF, GERADOR, PINO, SLIDES], "abc1234")


def _registro(snap: dict, rel: str, texto: str, antes: str, token: str) -> dict:
    assert texto.count(antes + token) >= 1, (antes, token)
    ini = texto.index(antes + token) + len(antes)
    achados = [o for o in snap["ocorrencias"] if o["arquivo"] == rel and o["offset"] == ini]
    assert len(achados) == 1, (antes, token)
    return achados[0]


def _manifesto(raiz: Path, snap: dict, aprovar=None) -> list[dict]:
    linhas = rs.construir_manifesto(snap, {}, {}, {}, so_regras=True)
    for l in linhas:
        ok = l["classe_final"] in rs.CLASSES_ALVO if aprovar is None else aprovar(l)
        l["aprovado"] = "sim" if ok else ""
    rs.gravar_manifesto(raiz / ".local/renomear_sigla/manifesto.csv", linhas)
    return linhas


def _aplicar(raiz: Path, *extra: str) -> int:
    return rs.main(["aplicar", "--raiz", str(raiz), *extra])


def _ler(raiz: Path, rel: str) -> bytes:
    return (raiz / rel).read_bytes()


def _trocar_uma_vez(texto: str, antigo: str, novo: str) -> str:
    assert texto.count(antigo) == 1, antigo
    return texto.replace(antigo, novo)


def _tex_esperado() -> str:
    t = TEX
    for antigo, novo in [
        ("Arranjo sequencial P}", "Arranjo sequencial ASR}"),
        ("O braço P recupera", "O braço ASR recupera"),
        ("de P-Platt foi", "de ASR-Platt foi"),
        ("a de P-PAR2-grey,", "a de ASR-PAR2-faixa,"),
        ("$P\\cup B1$", "$\\mathrm{ASR}\\cup B1$"),
        ("perfeito, P, \\texttt", "perfeito, ASR, \\texttt"),
        ("pares; P com", "pares; ASR com"),
        ("\\item[P] ", "\\item[ASR] "),
        ("\\item[P-Platt]", "\\item[ASR-Platt]"),
    ]:
        t = _trocar_uma_vez(t, antigo, novo)
    return t


# ---------------------------------------------------------------------------
# Mapeamento único
# ---------------------------------------------------------------------------
def test_mapeamento_unico_de_rotulos():
    assert rot.ROTULO_BRACO == {"P": "ASR", "P-Platt": "ASR-Platt", "P-PAR2-grey": "ASR-PAR2-faixa",
                                "P_MATCH_ao_humano": r"ASR, \texttt{MATCH} ao humano"}
    assert rot.ROTULO_MATH["P"] == r"\mathrm{ASR}"
    assert rot.substituicao_token("P-Platt") == "ASR-Platt"
    assert rot.substituicao_token("P", em_math=True) == r"\mathrm{ASR}"
    with pytest.raises(ValueError):
        rot.substituicao_token("P-Platt", em_math=True)
    with pytest.raises(ValueError):
        rot.substituicao_token("PAR")


# ---------------------------------------------------------------------------
# Classificação por regras: alvos A1-A7 e não alvos N1-N12
# ---------------------------------------------------------------------------
def test_classes_do_texto_latex(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    assert len(snap["ocorrencias"]) == len(ESPERADO_TEX)
    for antes, token, classe in ESPERADO_TEX:
        assert _registro(snap, AMOSTRA, TEX, antes, token)["classe_regra"] == classe, (antes, token)


def test_n5_n6_n10_nao_sao_candidatas(raiz):
    texto = "PAR PPV PRL NP, P10 e P90, Proposição, p minúsculo, conflict\\_P e arranjo_P_x.\n"
    assert rs.candidatos_texto("text/latex/x.tex", texto) == []


def test_python_literais_e_seletores(raiz):
    snap = _inventario(raiz, [GERADOR])
    esperado = [
        ("o braço ", "P", "N9"),                 # docstring
        ('NOTA = "', "P", "A1"),                 # rótulo de apresentação
        ('ROTULO = {"', "P", "N9"),              # chave de dicionário
        ('"P": "', "P", "A1"),                   # valor de rótulo
        ("(?", "P", "N8"),                       # grupo nomeado de regex
        ('df["braco"] == "', "P", "N9"),         # seletor de dados
        (".eq(\"", "P", "N9"),                   # seletor de dados
    ]
    assert len(snap["ocorrencias"]) == len(esperado)
    for antes, token, classe in esperado:
        assert _registro(snap, GERADOR, GERADOR_TXT, antes, token)["classe_regra"] == classe, (antes, token)


def test_pinos_de_teste_sao_apresentacao(raiz):
    snap = _inventario(raiz, [PINO])
    assert [o["classe_regra"] for o in snap["ocorrencias"]] == ["A1", "A1"]
    assert {o["superficie"] for o in snap["ocorrencias"]} == {"testes"}


def test_pgf_precisao_e_legenda(raiz):
    snap = _inventario(raiz, [PGF])
    assert _registro(snap, PGF, PGF_TXT, "{", "P")["classe_regra"] == "N7"
    assert _registro(snap, PGF, PGF_TXT, "Curva de ", "P")["classe_regra"] == "A1"
    assert {o["superficie"] for o in snap["ocorrencias"]} == {"figuras"}


def test_slides_fora_de_escopo_n12(raiz):
    snap = _inventario(raiz, [SLIDES])
    assert [(o["classe_regra"], o["zona"]) for o in snap["ocorrencias"]] == [("N12", "fora_de_escopo")]
    assert rs.fora_de_escopo("text/latex/tese_rev_diff.tex")
    assert not rs.fora_de_escopo("text/latex/tese.tex")


def test_campos_do_snapshot_e_id(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    o = _registro(snap, AMOSTRA, TEX, "O braço ", "P")
    for campo in ("id", "arquivo", "superficie", "zona", "offset", "token", "esq80", "dir80", "hash_frase",
                  "numeros_paragrafo", "sha256_arquivo", "commit", "classe_regra"):
        assert campo in o
    assert o["id"] == rs.hash16("|".join([AMOSTRA, o["esq80"], "P", o["dir80"]]))
    assert len(o["hash_frase"]) == 16 and o["commit"] == "abc1234"
    assert o["superficie"] == "texto" and o["zona"] == "prosa"
    assert "23" in o["numeros_paragrafo"] and "96" in o["numeros_paragrafo"]
    assert snap["arquivos"][AMOSTRA]["sha256"] == o["sha256_arquivo"]


def test_inventariar_cli_grava_snapshot(raiz, capsys):
    cod = rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "texto,geradores",
                   "--arquivos", f"{AMOSTRA},{GERADOR},{PGF}"])
    assert cod == 0
    achados = list((raiz / ".local/renomear_sigla").glob("ancoras_*_*.json"))
    assert len(achados) == 1 and re.fullmatch(r"ancoras_[^_]+_\d{12}\.json", achados[0].name)
    snap = json.loads(achados[0].read_text(encoding="utf-8"))
    assert set(snap["arquivos"]) == {AMOSTRA, GERADOR}  # figuras ficou de fora das superfícies pedidas
    assert rs.main(["classificar", "--regras", "--raiz", str(raiz)]) == 0
    assert "classes pela regra" in capsys.readouterr().out


def test_classificar_regras_recusa_arquivo_alterado(raiz):
    assert rs.main(["inventariar", "--raiz", str(raiz), "--arquivos", AMOSTRA, "--superficies", "texto"]) == 0
    _escrever(raiz, AMOSTRA, TEX + "\nMais um parágrafo.\n")
    assert rs.main(["classificar", "--regras", "--raiz", str(raiz)]) == 2


# ---------------------------------------------------------------------------
# aplicar: troca só os spans aprovados; nunca toca N1-N12
# ---------------------------------------------------------------------------
def test_aplicar_troca_so_alvos_e_preserva_nao_alvos(raiz):
    snap = _inventario(raiz)
    _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    novo = _ler(raiz, AMOSTRA).decode("utf-8")
    assert novo == _tex_esperado()
    # N1-N12 e ambíguas intactas
    for trecho in ["q = P(y=1\\mid s)", "foo-P-bar}", "P\\'os", "data/foo-P-bar.csv", "comentário com P aqui",
                   "Percentil P~90", "Jardim P.", "$P$", "\\texttt{P}"]:
        assert trecho in novo
    assert len(rs.candidatos_texto(AMOSTRA, novo)) == len(ESPERADO_TEX) - 9
    assert _ler(raiz, SLIDES).decode("utf-8") == SLIDES_TXT
    ger = _ler(raiz, GERADOR).decode("utf-8")
    assert 'NOTA = "ASR: arranjo' in ger and '{"P": "ASR"}' in ger
    assert 'df["braco"] == "P"' in ger and '.eq("P")' in ger and "(?P<nome>" in ger
    assert _ler(raiz, PGF).decode("utf-8") == PGF_TXT.replace("Curva de P", "Curva de ASR")
    assert 'assert "braço ASR recupera" in texto' in _ler(raiz, PINO).decode("utf-8")


def test_aplicar_ignora_linhas_nao_aprovadas_e_respeita_lote(raiz):
    snap = _inventario(raiz)
    _manifesto(raiz, snap)
    assert _aplicar(raiz, "--lote", AMOSTRA) == 0
    assert _ler(raiz, AMOSTRA).decode("utf-8") == _tex_esperado()
    assert _ler(raiz, GERADOR).decode("utf-8") == GERADOR_TXT  # fora do lote

    _escrever(raiz, AMOSTRA, TEX)
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap, aprovar=lambda l: l["classe_final"] == "A1")
    assert _aplicar(raiz) == 0
    assert _ler(raiz, AMOSTRA).decode("utf-8") == _trocar_uma_vez(TEX, "O braço P recupera", "O braço ASR recupera")


def test_aplicar_recusa_aprovacao_de_nao_alvo_e_nada_altera(raiz, capsys):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap, aprovar=lambda l: l["classe_final"] in rs.CLASSES_ALVO or l["classe_final"] == "N1")
    assert _aplicar(raiz) == 3
    assert "não é alvo" in capsys.readouterr().err
    assert _ler(raiz, AMOSTRA).decode("utf-8") == TEX


def test_aplicar_recusa_substituicao_fora_do_mapeamento(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap)
    for l in linhas:
        if l["classe_final"] == "A1":
            l["substituicao"] = "AS"
    rs.gravar_manifesto(raiz / ".local/renomear_sigla/manifesto.csv", linhas)
    assert _aplicar(raiz) == 3
    assert _ler(raiz, AMOSTRA).decode("utf-8") == TEX


def test_aplicar_recusa_contexto_ambiguo(raiz, capsys):
    _escrever(raiz, AMOSTRA, BLOCO * 2)
    snap = _inventario(raiz, [AMOSTRA])
    assert len(snap["ocorrencias"]) == 2
    a, b = snap["ocorrencias"]
    assert (a["esq80"], a["dir80"]) == (b["esq80"], b["dir80"]) and a["id"] != b["id"]
    _manifesto(raiz, snap)
    antes = _ler(raiz, AMOSTRA)
    assert _aplicar(raiz) == 3
    assert "contexto_ambiguo" in capsys.readouterr().err
    assert _ler(raiz, AMOSTRA) == antes


def test_aplicar_recusa_arquivo_cujo_contexto_sumiu(raiz, capsys):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap)
    alterado = _trocar_uma_vez(TEX, "O braço P recupera 23 pares verdadeiros com 96 revisões na vigilância do estudo.",
                               "Texto reescrito sem a sigla.")
    _escrever(raiz, AMOSTRA, alterado)
    antes = _ler(raiz, AMOSTRA)
    assert _aplicar(raiz) == 3
    assert "contexto_ausente" in capsys.readouterr().err
    assert _ler(raiz, AMOSTRA) == antes  # tudo ou nada por arquivo


def test_aplicar_relocaliza_pelo_contexto_quando_o_arquivo_mudou(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap)
    prefixo = "Parágrafo novo no início com 7 itens.\n\n"
    _escrever(raiz, AMOSTRA, prefixo + TEX)  # sha256 e offsets mudam
    assert _aplicar(raiz) == 0
    assert _ler(raiz, AMOSTRA).decode("utf-8") == prefixo + _tex_esperado()


def test_aplicar_e_idempotente(raiz):
    snap = _inventario(raiz)
    _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    primeira = {rel: _ler(raiz, rel) for rel in (AMOSTRA, PGF, GERADOR, PINO, SLIDES)}
    assert _aplicar(raiz) == 0
    assert {rel: _ler(raiz, rel) for rel in primeira} == primeira


def test_aplicar_preserva_crlf(raiz):
    _escrever(raiz, AMOSTRA, TEX, crlf=True)
    snap = _inventario(raiz, [AMOSTRA])
    assert snap["arquivos"][AMOSTRA]["eol"] == "crlf"
    _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    dados = _ler(raiz, AMOSTRA)
    assert dados == _tex_esperado().replace("\n", "\r\n").encode("utf-8")
    assert b"\n" not in dados.replace(b"\r\n", b"")


def test_aplicar_dry_run_nao_escreve(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap)
    assert _aplicar(raiz, "--dry-run") == 0
    assert _ler(raiz, AMOSTRA).decode("utf-8") == TEX


def test_aplicar_recusa_arquivo_fora_de_escopo(raiz):
    snap = _inventario(raiz, [SLIDES])
    linhas = rs.construir_manifesto(snap, {}, {}, {}, so_regras=True)
    for l in linhas:
        l.update(aprovado="sim", classe_final="A1", substituicao="ASR")
    rs.gravar_manifesto(raiz / ".local/renomear_sigla/manifesto.csv", linhas)
    assert _aplicar(raiz) == 3
    assert _ler(raiz, SLIDES).decode("utf-8") == SLIDES_TXT


# ---------------------------------------------------------------------------
# verificar: diff por span, resíduo, números por parágrafo
# ---------------------------------------------------------------------------
def _verificar(raiz: Path, linhas: list[dict], base: str = TEX, completo: bool = False) -> dict:
    return rs.verificar(raiz, [AMOSTRA], linhas, lambda rel: base, completo)


def test_verificar_aceita_a_aplicacao_correta(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    rel = _verificar(raiz, linhas, completo=True)
    assert rel["ok"], rel
    arq = rel["arquivos"][0]
    assert arq["residuo"] == len(ESPERADO_TEX) - 9 and arq["alvos_pendentes"] == 0


def test_verificar_detecta_diferenca_fora_do_span(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    atual = _ler(raiz, AMOSTRA).decode("utf-8")
    _escrever(raiz, AMOSTRA, _trocar_uma_vez(atual, "potenciais pares pendentes", "pares candidatos pendentes"))
    rel = _verificar(raiz, linhas)
    assert not rel["ok"]
    tipos = [f["tipo"] for f in rel["arquivos"][0]["falhas"]]
    assert "diferenca_fora_do_span" in tipos
    inesperadas = next(f for f in rel["arquivos"][0]["falhas"] if f["tipo"] == "diferenca_fora_do_span")["inesperadas"]
    assert inesperadas


def test_verificar_detecta_troca_de_nao_alvo(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    atual = _ler(raiz, AMOSTRA).decode("utf-8")
    _escrever(raiz, AMOSTRA, _trocar_uma_vez(atual, "q = P(y=1", "q = ASR(y=1"))
    rel = _verificar(raiz, linhas)
    assert not rel["ok"]
    assert "diferenca_fora_do_span" in [f["tipo"] for f in rel["arquivos"][0]["falhas"]]


def test_verificar_invariancia_numerica(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    atual = _ler(raiz, AMOSTRA).decode("utf-8")
    _escrever(raiz, AMOSTRA, _trocar_uma_vez(atual, "com 96 revisões", "com 97 revisões"))
    rel = _verificar(raiz, linhas)
    tipos = [f["tipo"] for f in rel["arquivos"][0]["falhas"]]
    assert "numeros_por_paragrafo" in tipos and not rel["ok"]
    nums = next(f for f in rel["arquivos"][0]["falhas"] if f["tipo"] == "numeros_por_paragrafo")
    assert nums["total"] == 1


def test_verificar_numeros_iguais_apos_a_troca(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    antes = rs.contar_numeros_texto(TEX)
    depois = rs.contar_numeros_texto(_ler(raiz, AMOSTRA).decode("utf-8"))
    assert antes == depois and sum(antes.values()) > 0


def test_verificar_alvo_pendente_e_aviso_ou_falha(raiz):
    snap = _inventario(raiz, [AMOSTRA])
    linhas = _manifesto(raiz, snap, aprovar=lambda l: l["classe_final"] == "A1")
    assert _aplicar(raiz) == 0
    leve = _verificar(raiz, linhas)
    assert leve["ok"] and leve["arquivos"][0]["avisos"]
    estrito = _verificar(raiz, linhas, completo=True)
    assert not estrito["ok"]


def test_verificar_cli_com_base_git_falsa(raiz, capsys):
    snap = _inventario(raiz, [AMOSTRA])
    _manifesto(raiz, snap)
    assert _aplicar(raiz) == 0
    args = rs.build_parser().parse_args(["verificar", "--raiz", str(raiz), "--arquivos", AMOSTRA, "--completo"])
    assert rs.cmd_verificar(args, leitor=lambda rel: TEX) == 0
    assert (raiz / ".local/renomear_sigla/verificacao.json").is_file()
    assert "verificação: OK" in capsys.readouterr().out
    assert rs.cmd_verificar(args, leitor=lambda rel: TEX + "\nOutro parágrafo.\n") == 3


# ---------------------------------------------------------------------------
# Decisão de classe e manifesto
# ---------------------------------------------------------------------------
def test_decidir_classe():
    o = {"classe_regra": "A1", "zona": "prosa"}
    assert rs.decidir_classe(o, {"classe": "A1", "confianca": 0.95}, None, False) == ("A1", "regra_e_llm")
    assert rs.decidir_classe(o, {"classe": "A1", "confianca": 0.5}, None, False) == (rs.AMBIGUA, "pendente")
    assert rs.decidir_classe(o, {"classe": "N9", "confianca": 0.99}, None, False) == (rs.AMBIGUA, "pendente")
    assert rs.decidir_classe(o, None, None, False) == (rs.AMBIGUA, "pendente")
    assert rs.decidir_classe(o, {"classe": "N9", "confianca": 0.99}, {"classe": "A2"}, False) == ("A2", "adjudicacao")
    assert rs.decidir_classe(o, None, None, True) == ("A1", "so_regras")
    amb = {"classe_regra": rs.AMBIGUA, "zona": "math"}
    assert rs.decidir_classe(amb, {"classe": "A4", "confianca": 0.99}, None, False) == (rs.AMBIGUA, "pendente")
    prot = {"classe_regra": "N3", "zona": "label_ref"}
    assert rs.decidir_classe(prot, {"classe": "A1", "confianca": 0.99}, {"classe": "A1"}, False) == (
        "N3", "zona_protegida_conflito")
    assert rs.decidir_classe(prot, {"classe": "N3", "confianca": 0.99}, None, False) == ("N3", "zona_protegida")


def test_precisa_adjudicar():
    o = {"classe_regra": "A1"}
    assert not rs.precisa_adjudicar(o, {"classe": "A1", "confianca": 0.95})
    assert rs.precisa_adjudicar(o, {"classe": "A1", "confianca": 0.89})
    assert rs.precisa_adjudicar(o, {"classe": "N9", "confianca": 0.99})
    assert rs.precisa_adjudicar({"classe_regra": rs.AMBIGUA}, {"classe": "A1", "confianca": 0.99})
    assert rs.precisa_adjudicar(o, None)
    assert rs.precisa_adjudicar(o, {"erro": "json_invalido", "classe": None})


def test_contexto_do_manifesto_vai_e_volta():
    ctx = rs.fmt_contexto("linha um\nlinha dois ", "P", " fim")
    assert "\n" not in ctx
    assert rs.parse_contexto(ctx) == ("linha um\nlinha dois ", "P", " fim")
    with pytest.raises(ValueError):
        rs.parse_contexto("sem marcadores")
    with pytest.raises(ValueError):
        rs.fmt_contexto("tem ⏎ marcador", "P", "")


def test_manifesto_resumo_e_aprovacao_preservada(raiz):
    assert rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "texto", "--arquivos", AMOSTRA]) == 0
    assert rs.main(["manifesto", "--raiz", str(raiz), "--so-regras"]) == 0
    destino = raiz / ".local/renomear_sigla/manifesto.csv"
    linhas = rs.ler_manifesto(destino)
    assert list(linhas[0]) == rs.COLUNAS[:len(linhas[0])] and len(linhas) == len(ESPERADO_TEX)
    resumo = (raiz / ".local/renomear_sigla/manifesto_resumo.txt").read_text(encoding="utf-8")
    assert "Resumo por arquivo" in resumo and AMOSTRA in resumo and "Casos ambíguos" in resumo
    assert resumo.count("regra=AMB") == 2
    for l in linhas:
        if l["classe_final"] == "A1":
            l["aprovado"] = "sim"
    rs.gravar_manifesto(destino, linhas)
    assert rs.main(["manifesto", "--raiz", str(raiz), "--so-regras"]) == 0
    de_novo = rs.ler_manifesto(destino)
    assert [l["aprovado"] for l in de_novo] == [l["aprovado"] for l in linhas]
    assert sum(1 for l in de_novo if l["aprovado"] == "sim") == 1


def test_manifesto_sem_llm_exige_so_regras(raiz, capsys):
    assert rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "texto", "--arquivos", AMOSTRA]) == 0
    assert rs.main(["manifesto", "--raiz", str(raiz)]) == 2
    assert "dupla checagem" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# LLM com cliente falso
# ---------------------------------------------------------------------------
def _msg(texto: str, stop: str = "end_turn"):
    uso = SimpleNamespace(input_tokens=11, output_tokens=7, cache_creation_input_tokens=5, cache_read_input_tokens=3)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=texto)], usage=uso, stop_reason=stop)


class ClienteFalso:
    def __init__(self, classe_por_id=None, padrao="A1", conf=0.95, texto_fixo=None, truncar_primeiras=0):
        self.truncar_primeiras = truncar_primeiras
        self.chamadas: list[dict] = []
        self.lotes_enviados: list[list[dict]] = []
        self._lotes: dict[str, list[dict]] = {}
        self.classe_por_id = classe_por_id or {}
        self.padrao, self.conf, self.texto_fixo = padrao, conf, texto_fixo
        self.messages = SimpleNamespace(
            create=self._create,
            batches=SimpleNamespace(create=self._b_create, retrieve=self._b_retrieve, results=self._b_results),
        )

    def ids_enviados(self) -> list[str]:
        return [i for p in self.chamadas for i in re.findall(r"### id: (\S+)", p["messages"][0]["content"])]

    def _corpo(self, params: dict) -> str:
        if self.texto_fixo is not None:
            return self.texto_fixo
        ids = re.findall(r"### id: (\S+)", params["messages"][0]["content"])
        return json.dumps([{"id": i, "classe": self.classe_por_id.get(i, self.padrao), "confianca": self.conf,
                            "justificativa": "teste"} for i in ids])

    def _create(self, **params):
        self.chamadas.append(params)
        if len(self.chamadas) <= self.truncar_primeiras:
            return _msg('[{"id": "x", "classe": "A1", "justificativa": "texto cortado', stop="max_tokens")
        return _msg(self._corpo(params))

    def _b_create(self, requests):
        lote_id = f"lote{len(self._lotes) + 1}"
        self._lotes[lote_id] = requests
        self.lotes_enviados.append(requests)
        return SimpleNamespace(id=lote_id, processing_status="in_progress")

    def _b_retrieve(self, lote_id):
        return SimpleNamespace(id=lote_id, processing_status="ended")

    def _b_results(self, lote_id):
        for r in self._lotes[lote_id]:
            yield SimpleNamespace(custom_id=r["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=_msg(self._corpo(r["params"]))))


@pytest.fixture
def chave(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "segredo-teste-123")
    return "segredo-teste-123"


def _preparar(raiz: Path) -> dict:
    assert rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "texto,geradores,figuras",
                    "--arquivos", f"{AMOSTRA},{GERADOR},{PGF}"]) == 0
    return json.loads(next((raiz / ".local/renomear_sigla").glob("ancoras_*.json")).read_text(encoding="utf-8"))


def _classificar(raiz: Path, cliente: ClienteFalso, *extra: str) -> int:
    args = rs.build_parser().parse_args(["classificar", "--llm", "--raiz", str(raiz), "--modo", "sync",
                                         "--por-requisicao", "4", *extra])
    return rs.cmd_classificar_llm(args, fabrica=lambda k: cliente)


def test_llm_classifica_todas_com_checkpoint_e_sem_vazar_chave(raiz, chave, capsys):
    snap = _preparar(raiz)
    total = len(snap["ocorrencias"])
    cli = ClienteFalso()
    assert _classificar(raiz, cli) == 0
    linhas = read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")
    assert len(linhas) == total and {l["id"] for l in linhas} == {o["id"] for o in snap["ocorrencias"]}
    assert all(l["modelo"] == "claude-sonnet-5-5" and l["modo"] == "sync" and l["data"] for l in linhas)
    assert len(cli.chamadas) == -(-total // 4)
    for p in cli.chamadas:
        assert "temperature" not in p
        assert p["model"] == "claude-sonnet-5-5"
        assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
        usuario = p["messages"][0]["content"]
        assert "⟦" in usuario and "classe da regra" not in usuario  # a primeira via é independente da regra
    assert chave not in capsys.readouterr().out
    assert chave not in (raiz / ".local/renomear_sigla/llm.jsonl").read_text(encoding="utf-8")
    # segunda execução: nada a enviar
    n = len(cli.chamadas)
    assert _classificar(raiz, cli) == 0 and len(cli.chamadas) == n


def test_llm_retoma_do_checkpoint(raiz, chave):
    snap = _preparar(raiz)
    total = len(snap["ocorrencias"])
    cli = ClienteFalso()
    assert _classificar(raiz, cli, "--limite", "3") == 0
    primeiros = set(cli.ids_enviados())
    assert len(primeiros) == 3
    cli2 = ClienteFalso()
    assert _classificar(raiz, cli2) == 0
    assert primeiros.isdisjoint(cli2.ids_enviados())
    assert len(read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")) == total


def test_llm_resposta_invalida_vira_erro_e_refaz(raiz, chave):
    _preparar(raiz)
    ruim = ClienteFalso(texto_fixo="isto não é um array")
    assert _classificar(raiz, ruim) == 3
    linhas = read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")
    assert linhas and all(l["erro"].startswith("json_invalido") for l in linhas)
    bom = ClienteFalso()
    assert _classificar(raiz, bom) == 3  # erros não são reenviados sem --refazer-falhas
    assert bom.chamadas == []
    assert _classificar(raiz, bom, "--refazer-falhas") == 0
    assert all(not l["erro"] for l in read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl"))


def test_llm_classe_invalida_e_id_ausente(raiz, chave):
    _preparar(raiz)

    class Torto(ClienteFalso):
        def _corpo(self, params):
            ids = re.findall(r"### id: (\S+)", params["messages"][0]["content"])
            itens = [{"id": ids[0], "classe": "Z9", "confianca": 0.9, "justificativa": "x"}]
            return json.dumps(itens)

    assert _classificar(raiz, Torto()) == 3
    erros = {l["erro"] for l in read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")}
    assert erros <= {"classe_ou_confianca_invalida", "ausente_na_resposta"} and len(erros) == 2


def test_llm_estimar_nao_chama_api(raiz, capsys):
    _preparar(raiz)
    assert rs.main(["classificar", "--llm", "--estimar", "--raiz", str(raiz)]) == 0
    saida = capsys.readouterr().out
    assert "requisições" in saida and "tokens" in saida
    assert not (raiz / ".local/renomear_sigla/llm.jsonl").exists()


def test_llm_sem_chave_cita_so_o_nome(raiz, capsys, monkeypatch):
    _preparar(raiz)
    monkeypatch.delenv("RENOMEAR_SEM_CHAVE_XYZ", raising=False)
    assert rs.main(["classificar", "--llm", "--raiz", str(raiz), "--chave-env", "RENOMEAR_SEM_CHAVE_XYZ"]) == 2
    assert "RENOMEAR_SEM_CHAVE_XYZ ausente" in capsys.readouterr().err


def test_llm_modo_batch_com_checkpoint(raiz, chave):
    snap = _preparar(raiz)
    out = raiz / ".local/renomear_sigla"
    itens = rs.itens_do_snapshot(raiz, snap, rs.RAIO_LLM)
    cli = ClienteFalso()
    arm = rs.Armazem(out / "llm.jsonl")
    kw = dict(itens=itens, sistema=rs.SISTEMA_CLASSIFICAR, modelo="claude-sonnet-5-5", client=cli, modo="batch",
              armazem=arm, etapa="clas", por_req=5, sleep=lambda s: None, estado_lotes=out / "llm.lotes.json")
    res = rs.executar_llm(**kw)
    assert res == {"requisicoes": -(-len(itens) // 5), "pendentes": 0, "erros": 0}
    assert len(cli.lotes_enviados) == 1 and cli.chamadas == []
    assert all(l["modo"] == "batch" for l in rs.Armazem(out / "llm.jsonl").linhas.values())
    assert json.loads((out / "llm.lotes.json").read_text(encoding="utf-8"))["lotes"][0]["baixado"] is True
    res2 = rs.executar_llm(**{**kw, "armazem": rs.Armazem(out / "llm.jsonl")})
    assert res2["requisicoes"] == 0 and len(cli.lotes_enviados) == 1


def test_adjudicar_so_divergentes_e_manifesto_final(raiz, chave, capsys):
    snap = _preparar(raiz)
    out = raiz / ".local/renomear_sigla"
    assert _classificar(raiz, ClienteFalso(padrao="A1", conf=0.95)) == 0  # o primeiro leitor diz A1 para tudo
    llm = {r["id"]: r for r in read_jsonl(out / "llm.jsonl")}
    esperados = {o["id"] for o in snap["ocorrencias"] if rs.precisa_adjudicar(o, llm[o["id"]])}
    assert esperados and len(esperados) < len(snap["ocorrencias"])
    opus = ClienteFalso(padrao="N9", conf=0.97)
    args = rs.build_parser().parse_args(["adjudicar", "--raiz", str(raiz), "--modo", "sync",
                                         "--por-requisicao", "4"])
    assert rs.cmd_adjudicar(args, fabrica=lambda k: opus) == 0
    assert set(opus.ids_enviados()) == esperados
    assert all(p["model"] == "claude-opus-5-5" and "temperature" not in p for p in opus.chamadas)
    assert all("classe da regra" in p["messages"][0]["content"] for p in opus.chamadas)
    adj = read_jsonl(out / "adjudicacao.jsonl")
    assert {a["id"] for a in adj} == esperados and all(a["modelo"] == "claude-opus-5-5" for a in adj)

    assert rs.main(["manifesto", "--raiz", str(raiz)]) == 0
    linhas = {l["id"]: l for l in rs.ler_manifesto(out / "manifesto.csv")}
    assert all(l["classe_final"] != rs.AMBIGUA for l in linhas.values())
    for ident, l in linhas.items():
        if ident in esperados and l["zona"] not in rs.ZONAS_PROTEGIDAS:
            assert l["classe_final"] == "N9" and l["origem_final"] == "adjudicacao" and l["substituicao"] == ""
        if ident not in esperados:
            assert l["classe_final"] == "A1" and l["origem_final"] == "regra_e_llm" and l["substituicao"] == "ASR"
    assert "Resumo por arquivo" in capsys.readouterr().out


def test_adjudicar_exige_llm(raiz, capsys):
    _preparar(raiz)
    assert rs.main(["adjudicar", "--raiz", str(raiz)]) == 2
    assert "classificar --llm" in capsys.readouterr().err


def test_max_tokens_e_por_requisicao_padrao(raiz, chave):
    p = rs.build_parser()
    ad = p.parse_args(["adjudicar"])
    assert ad.max_tokens == 8000 and ad.por_requisicao == 2
    assert p.parse_args(["classificar", "--llm"]).max_tokens == 8000
    assert p.parse_args(["adjudicar", "--max-tokens", "9000", "--por-requisicao", "1"]).max_tokens == 9000
    _preparar(raiz)
    padrao = ClienteFalso()
    assert _classificar(raiz, padrao) == 0
    assert {c["max_tokens"] for c in padrao.chamadas} == {8000}
    (raiz / ".local/renomear_sigla/llm.jsonl").unlink()
    explicito = ClienteFalso()
    assert _classificar(raiz, explicito, "--max-tokens", "1234") == 0
    assert {c["max_tokens"] for c in explicito.chamadas} == {1234}


def test_llm_truncado_e_refeito_so_nos_truncados_um_por_requisicao(raiz, chave):
    snap = _preparar(raiz)
    total = len(snap["ocorrencias"])
    cortado = ClienteFalso(truncar_primeiras=2)
    assert _classificar(raiz, cortado, "--concorrencia", "1") == 3
    linhas = read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")
    truncados = {l["id"] for l in linhas if l["erro"] == "resposta_truncada"}
    assert len(linhas) == total and len(truncados) == 8
    completo = ClienteFalso()
    assert _classificar(raiz, completo, "--refazer-falhas") == 0
    assert set(completo.ids_enviados()) == truncados  # os já respondidos não voltam
    assert len(completo.chamadas) == 8  # truncados reenviados um por requisição
    final = read_jsonl(raiz / ".local/renomear_sigla/llm.jsonl")
    assert len(final) == total and all(not l["erro"] for l in final)


def test_adjudicar_truncado_e_refeito(raiz, chave):
    _preparar(raiz)
    out = raiz / ".local/renomear_sigla"
    assert _classificar(raiz, ClienteFalso(padrao="A1")) == 0
    args = rs.build_parser().parse_args(["adjudicar", "--raiz", str(raiz), "--modo", "sync", "--concorrencia", "1"])
    cortado = ClienteFalso(padrao="N9", truncar_primeiras=3)
    assert rs.cmd_adjudicar(args, fabrica=lambda k: cortado) == 3
    assert all(len(re.findall(r"### id:", p["messages"][0]["content"])) <= 2 for p in cortado.chamadas)
    truncados = {l["id"] for l in read_jsonl(out / "adjudicacao.jsonl") if l["erro"] == "resposta_truncada"}
    assert 0 < len(truncados) <= 6
    bom = ClienteFalso(padrao="N9")
    args2 = rs.build_parser().parse_args(["adjudicar", "--raiz", str(raiz), "--modo", "sync", "--refazer-falhas"])
    assert rs.cmd_adjudicar(args2, fabrica=lambda k: bom) == 0
    assert set(bom.ids_enviados()) == truncados and len(bom.chamadas) == len(truncados)


MARC = "scripts/marcador.py"
MARC_TXT = 'import re\n\nPADRAO = re.compile(r"[\u27e6\u27e7\u23ce](?P<nome>\\d+)")\n'


def test_manifesto_nao_alvo_com_marcadores_entra_escapado_e_inaplicavel(raiz, capsys):
    _escrever(raiz, MARC, MARC_TXT)
    snap = _inventario(raiz, [MARC])
    assert [o["classe_regra"] for o in snap["ocorrencias"]] == ["N8"]
    assert rs.tem_marcador(snap["ocorrencias"][0]["esq80"])
    linha = rs.construir_manifesto(snap, {}, {}, {}, so_regras=True)[0]
    assert linha["classe_final"] == "N8" and linha["aplicavel"] == "nao" and linha["substituicao"] == ""
    ctx = linha["contexto"]
    assert "\\u27e6" in ctx and "\\u27e7" in ctx and "\\u23ce" in ctx  # literais escapados
    assert ctx.count(rs.MARCA_ABRE) == 1 and ctx.count(rs.MARCA_FECHA) == 1  # só o par do token
    assert rs.parse_contexto(ctx)[1] == "P"

    # pela CLI: o manifesto sai (código 0), com a linha listada como inaplicável
    assert rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "geradores", "--arquivos", MARC]) == 0
    assert rs.main(["manifesto", "--raiz", str(raiz), "--so-regras"]) == 0
    destino = raiz / ".local/renomear_sigla/manifesto.csv"
    assert "aplicavel=nao" in capsys.readouterr().out
    linhas = rs.ler_manifesto(destino)
    assert [l["aplicavel"] for l in linhas] == ["nao"]
    # mesmo aprovada à mão, a linha inaplicável é recusada e nada muda
    linhas[0]["aprovado"] = "sim"
    rs.gravar_manifesto(destino, linhas)
    assert _aplicar(raiz) == 3
    assert "aplicavel=nao" in capsys.readouterr().err
    assert _ler(raiz, MARC).decode("utf-8") == MARC_TXT


def test_manifesto_aborta_se_ocorrencia_alvo_tem_marcador(raiz, capsys):
    texto = "O braço P recupera pares verdadeiros \u27e6 com um marcador no contexto da ocorrência.\n"
    _escrever(raiz, AMOSTRA, texto)
    snap = _inventario(raiz, [AMOSTRA])
    assert [o["classe_regra"] for o in snap["ocorrencias"]] == ["A1"]
    with pytest.raises(ValueError, match="ALVO"):
        rs.construir_manifesto(snap, {}, {}, {}, so_regras=True)
    assert rs.main(["inventariar", "--raiz", str(raiz), "--superficies", "texto", "--arquivos", AMOSTRA]) == 0
    assert rs.main(["manifesto", "--raiz", str(raiz), "--so-regras"]) == 2
    assert "ALVO" in capsys.readouterr().err
    assert not (raiz / ".local/renomear_sigla/manifesto.csv").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("comando", [[], ["inventariar"], ["classificar"], ["adjudicar"], ["manifesto"],
                                     ["aplicar"], ["verificar"]])
def test_cli_help(comando, capsys):
    with pytest.raises(SystemExit) as exc:
        rs.build_parser().parse_args([*comando, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_exige_subcomando():
    with pytest.raises(SystemExit) as exc:
        rs.build_parser().parse_args([])
    assert exc.value.code == 2
