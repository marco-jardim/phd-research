"""Testes de scripts/cruzamento_ancoras.py (cruzamento da tese com as âncoras) com a API mockada."""

from __future__ import annotations

import csv
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import cruzamento_ancoras as cli
from scripts.atestado.common import write_jsonl_atomic
from scripts.cruzamento import adjudicar as adj
from scripts.cruzamento import cruzar as crz
from scripts.cruzamento import execucao as ex
from scripts.cruzamento import fatiar as ft
from scripts.cruzamento import leitores as ld
from scripts.cruzamento import mapear as mp
from scripts.cruzamento import opencode as oc
from scripts.cruzamento import pinos as pn
from scripts.cruzamento import relatorio as rel
from scripts.cruzamento.comum import (
    FATIAS_NOME,
    MAPA_NOME,
    MAPA_OPENCODE_NOME,
    PARES_NOME,
    PINOS_NOME,
    carregar_ancoras,
    carregar_ficha,
    extrair_objeto,
    id_curto,
    indexar_fatia,
    localizar_trecho,
    numerar,
)

REPO = Path(__file__).resolve().parents[1]

ANCORAS = [
    {"id": "C01", "origem": "atual", "texto": "Texto da âncora um.", "ressalva": ""},
    {"id": "C02", "origem": "proposta", "texto": "Texto da âncora dois.", "ressalva": "Com ressalva."},
]


# ---------------------------------------------------------------------------
# Projeto LaTeX de teste
# ---------------------------------------------------------------------------
ROOT_TEX = """\\documentclass{book}
\\begin{document}
\\input{cap1.tex}
\\input{cap1_diff.tex}
\\end{document}
"""

CAP1_TEX = """\\begin{chapter}{Introdução}\\label{cap:intro}
Primeiro parágrafo da introdução com algumas palavras de teste.

Segundo parágrafo da introdução, também com palavras.

\\section{Métodos}
Texto de métodos com a meta 23 no passo.

\\input{tables/tab_a.tex}

\\subsection{Detalhe}
Texto do detalhe.
\\end{chapter}
"""

TAB_TEX = """\\begin{table}\\caption{Tabela A}\\label{tab:a}
\\begin{tabular}{l}
x \\\\
\\end{tabular}
\\end{table}
"""

DIFF_TEX = "\\chapter{Capítulo de diff}\nTexto de diff que não deve aparecer.\n"
ORFAO_TEX = "\\chapter{Órfão}\nTexto órfão fora da árvore.\n"

SLIDES_TEX = """\\documentclass{beamer}
\\begin{document}
\\begin{frame}
\\titlepage
\\end{frame}
% \\begin{frame}{Comentado}
\\section{Resultados}
\\begin{frame}{Estudo com \\textit{expected loss}}
Texto do slide um.

Outro bloco do slide.
\\end{frame}
\\end{document}
"""

ROTEIRO_MD = """# Roteiro

## Cronometragem

Tempo total de 30 minutos.

## Fala-guia

### 1. Capa

Fala da capa com texto.

### 2. Dados

Fala dos dados com 23 casos.
"""


def escrever(caminho: Path, texto: str) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8", newline="\n")
    return caminho


def bytes_gerados(caminho: Path) -> bytes:
    """Bytes de uma saída gerada em ``tmp_path``, para comparar duas execuções entre si.

    Lê com ``open("rb")`` de propósito. ``scripts/auditar_eol.py`` trata módulo de teste com leitura de bytes
    como teste de bytes de arquivos versionados e expande todos os literais dos módulos ``scripts.*``
    importados; aqui isso inclui a regex ``_NOME_VARIAVEL`` de ``scripts/cruzamento_ancoras.py``, que como
    glob casa quase qualquer caminho do repositório. Este módulo não compara bytes de arquivo versionado.
    """
    with caminho.open("rb") as arquivo:
        return arquivo.read()


@pytest.fixture()
def projeto(tmp_path: Path) -> Path:
    escrever(tmp_path / "tese.tex", ROOT_TEX)
    escrever(tmp_path / "cap1.tex", CAP1_TEX)
    escrever(tmp_path / "tables" / "tab_a.tex", TAB_TEX)
    escrever(tmp_path / "cap1_diff.tex", DIFF_TEX)
    escrever(tmp_path / "orfao.tex", ORFAO_TEX)
    escrever(tmp_path / "slides" / "defesa.tex", SLIDES_TEX)
    escrever(tmp_path / "slides" / "roteiro.md", ROTEIRO_MD)
    return tmp_path


# ---------------------------------------------------------------------------
# fatiar
# ---------------------------------------------------------------------------
def test_fatiar_tese_e_deterministico(projeto: Path) -> None:
    primeira = ft.fatiar_tese(projeto / "tese.tex", projeto)
    segunda = ft.fatiar_tese(projeto / "tese.tex", projeto)
    assert primeira == segunda and primeira
    ids = [f["id"] for f in primeira]
    assert len(ids) == len(set(ids))
    for f in primeira:
        assert f["id"] == id_curto(f["arquivo"], str(f["linha_inicio"]), f["texto"])
        assert len(f["linhas"]) == len(f["texto"].split("\n"))
        assert f["fonte"] == "tese"


def test_fatiar_tese_ignora_diff_e_arquivos_fora_da_arvore(projeto: Path) -> None:
    fatias = ft.fatiar_tese(projeto / "tese.tex", projeto)
    arquivos = {f["arquivo"] for f in fatias}
    assert "cap1.tex" in arquivos and "tables/tab_a.tex" in arquivos
    assert "cap1_diff.tex" not in arquivos and "orfao.tex" not in arquivos
    juntos = "\n".join(f["texto"] for f in fatias)
    assert "Texto de diff" not in juntos and "Texto órfão" not in juntos


def test_arquivo_excluido() -> None:
    assert ft.arquivo_excluido("text/latex/tese_rev_diff.tex")
    assert ft.arquivo_excluido("text/latex/cap1_diff.tex")
    assert ft.arquivo_excluido("text/latex/tese_diff_work/a.tex")
    assert not ft.arquivo_excluido("text/latex/cap1.tex")


def test_fatiar_tese_secao_e_tabela_na_secao_que_a_inclui(projeto: Path) -> None:
    fatias = ft.fatiar_tese(projeto / "tese.tex", projeto)
    tabela = [f for f in fatias if f["arquivo"] == "tables/tab_a.tex"]
    assert tabela and all(f["secao"] == "Cap. 1 Introdução > 1.1 Métodos" for f in tabela)
    assert fatias[0]["secao"] == "Cap. 1 Introdução"


def test_fatiar_tese_nao_corta_paragrafo(projeto: Path) -> None:
    # com limite de 1 palavra cada parágrafo vira uma fatia só dele (nunca é cortado)
    fatias = ft.fatiar_tese(projeto / "tese.tex", projeto, max_palavras=1)
    assert all("\n\n" not in f["texto"] for f in fatias)
    textos = [f["texto"] for f in fatias]
    assert any("Segundo parágrafo da introdução, também com palavras." == t for t in textos)
    assert any(t.startswith("\\section{Métodos}\nTexto de métodos com a meta 23 no passo.") for t in textos)
    # com o limite padrão os parágrafos da mesma seção e arquivo se juntam
    padrao = ft.fatiar_tese(projeto / "tese.tex", projeto)
    assert len(padrao) < len(fatias)


def _paragrafo(inicio: int, palavras: int) -> tuple[int, list[str]]:
    return inicio, [" ".join(["palavra"] * palavras)]


def test_empacotar_respeita_limite_de_palavras() -> None:
    paragrafos = [_paragrafo(1, 30), _paragrafo(3, 30), _paragrafo(5, 30), _paragrafo(7, 100), _paragrafo(9, 10)]
    grupos = ft.empacotar(paragrafos, 70)
    assert [[p[0] for p in g] for g in grupos] == [[1, 3], [5], [7], [9]]
    for grupo in grupos:
        total = sum(len(p[1][0].split()) for p in grupo)
        assert total <= 70 or len(grupo) == 1  # só um parágrafo sozinho pode passar do limite


def test_fatiar_slides_um_frame_por_fatia(projeto: Path) -> None:
    fatias = ft.fatiar_slides(projeto / "slides" / "defesa.tex", projeto)
    assert len(fatias) == 2  # o frame comentado não conta
    assert fatias[0]["secao"] == "Slides > Slide 1"
    assert fatias[1]["secao"] == "Slides > Resultados > Slide 2: Estudo com expected loss"
    assert (fatias[0]["linha_inicio"], fatias[0]["linha_fim"]) == (3, 5)
    assert (fatias[1]["linha_inicio"], fatias[1]["linha_fim"]) == (8, 12)
    assert fatias[1]["arquivo"] == "slides/defesa.tex"
    assert "Outro bloco do slide." in fatias[1]["texto"] and "Comentado" not in "".join(f["texto"] for f in fatias)


def test_fatiar_roteiro_por_secao(projeto: Path) -> None:
    fatias = ft.fatiar_roteiro(projeto / "slides" / "roteiro.md", projeto)
    assert [f["secao"] for f in fatias] == [
        "Roteiro > Cronometragem", "Roteiro > Fala-guia > 1. Capa", "Roteiro > Fala-guia > 2. Dados",
    ]
    capa = fatias[1]
    assert capa["texto"] == "### 1. Capa\n\nFala da capa com texto."
    assert capa["linhas"] == [9, 0, 11] and (capa["linha_inicio"], capa["linha_fim"]) == (9, 11)
    assert capa["arquivo"] == "slides/roteiro.md"


def test_cli_fatiar_grava_e_e_deterministico(projeto: Path) -> None:
    out = projeto / "saida"
    argv = ["fatiar", "--tese", str(projeto / "tese.tex"), "--slides", str(projeto / "slides" / "defesa.tex"),
            "--roteiro", str(projeto / "slides" / "roteiro.md"), "--base", str(projeto), "--out", str(out),
            "--fontes", "tese,slides,roteiro"]
    assert cli.main(argv) == 0
    primeiro = bytes_gerados(out / FATIAS_NOME)
    assert cli.main(argv) == 0
    assert bytes_gerados(out / FATIAS_NOME) == primeiro
    fontes = {json.loads(linha)["fonte"] for linha in primeiro.decode("utf-8").splitlines()}
    assert fontes == {"tese", "slides", "roteiro"}


def _fontes_gravadas(out: Path) -> set[str]:
    return {json.loads(linha)["fonte"] for linha in (out / FATIAS_NOME).read_text(encoding="utf-8").splitlines()}


def test_cli_fatiar_padrao_so_tese_e_fontes_escolhidas(projeto: Path, capsys) -> None:
    out = projeto / "saida"
    base = ["fatiar", "--tese", str(projeto / "tese.tex"), "--slides", str(projeto / "slides" / "defesa.tex"),
            "--roteiro", str(projeto / "slides" / "roteiro.md"), "--base", str(projeto), "--out", str(out)]
    assert cli.build_parser().parse_args(["fatiar"]).fontes == "tese"
    assert cli.main(base) == 0
    assert _fontes_gravadas(out) == {"tese"}  # o padrão não inclui slides nem roteiro
    assert cli.main([*base, "--fontes", "tese,slides"]) == 0
    assert _fontes_gravadas(out) == {"tese", "slides"}
    assert cli.main([*base, "--fontes", "tese,video"]) == 2
    assert "--fontes inválido" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Âncoras, numeração e verificação literal de trechos
# ---------------------------------------------------------------------------
def test_carregar_ancoras(tmp_path: Path) -> None:
    caminho = escrever(tmp_path / "a.yaml", (
        "- id: C01\n  origem: atual\n  texto: Primeira âncora.\n"
        "- id: C01r\n  origem: proposta\n  texto: Segunda.\n  ressalva: Com ressalva.\n"
    ))
    ancoras = carregar_ancoras(caminho)
    assert [a["id"] for a in ancoras] == ["C01", "C01r"]
    assert ancoras[1]["ressalva"] == "Com ressalva." and ancoras[0]["ressalva"] == ""
    ruim = escrever(tmp_path / "b.yaml", "- id: X\n  origem: inventada\n  texto: t\n")
    with pytest.raises(ValueError):
        carregar_ancoras(ruim)


def fatia_exemplo() -> dict:
    return {"id": "f1", "arquivo": "x.tex", "linha_inicio": 10, "linha_fim": 13, "secao": "S", "fonte": "tese",
            "texto": "Linha um com  espaços\nsegue aqui.\n\nOutro parágrafo.", "linhas": [10, 11, 0, 13]}


def test_numerar_prefixa_o_numero_da_linha() -> None:
    texto = numerar(fatia_exemplo()).split("\n")
    assert texto[0].strip().startswith("10|") and texto[1].strip().startswith("11|")
    assert texto[2] == "" and texto[3].strip().startswith("13|")


def test_localizar_trecho_normaliza_espacos_e_devolve_a_linha() -> None:
    indice = indexar_fatia(fatia_exemplo())
    assert localizar_trecho(indice, "com espaços segue aqui.") == 10
    assert localizar_trecho(indice, "aqui. Outro   parágrafo.") == 11
    assert localizar_trecho(indice, "Outro parágrafo.") == 13
    assert localizar_trecho(indice, "texto inexistente") is None
    assert localizar_trecho(indice, "   ") is None


def test_processar_resultado_verifica_trechos_sem_descartar() -> None:
    fatia = fatia_exemplo()
    resultado = {
        "relacoes": [
            {"ancora_id": "C01", "relacao": "afirma", "trecho": "segue aqui.", "linha": 99, "justificativa": "j"},
            {"ancora_id": "C01", "relacao": "contradiz", "trecho": "frase parafraseada", "linha": 10},
            {"ancora_id": "C77", "relacao": "inventada", "trecho": "Outro parágrafo.", "linha": 13},
            "não é objeto",
        ],
        "afirmacoes": [
            {"trecho": "Outro parágrafo.", "linha": 13, "tipo": "resultado", "topicos": ["P_vigilancia", "xx"],
             "numeros": ["23"], "polaridade": "afirma", "escopo": "e"},
        ],
    }
    saida = mp.processar_resultado(fatia, resultado, {"C01", "C02"})
    valida, invalida, estranha = saida["relacoes"]
    assert valida["trecho_valido"] and valida["linha"] == 11 and "linha_corrigida" in valida["flags"]
    assert not invalida["trecho_valido"] and "trecho_invalido" in invalida["flags"] and invalida["linha"] == 10
    assert "relacao_invalida" in estranha["flags"] and "ancora_desconhecida" in estranha["flags"]
    assert saida["malformados"] == 1
    af = saida["afirmacoes"][0]
    assert af["topicos"] == ["P_vigilancia"] and "topico_fora_da_taxonomia" in af["flags"]
    assert "numero_fora_do_trecho" in af["flags"]


def test_extrair_objeto_aceita_cerca_e_rejeita_lixo() -> None:
    assert extrair_objeto('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extrair_objeto("sem json")
    with pytest.raises(ValueError):
        extrair_objeto("{quebrado")


# ---------------------------------------------------------------------------
# Cliente mockado
# ---------------------------------------------------------------------------
class ErroApi(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status_code = status


class APIConnectionError(Exception):
    """Erro de rede (sem status_code), reconhecido pelo nome da classe."""


def mensagem(conteudo) -> SimpleNamespace:
    texto = conteudo if isinstance(conteudo, str) else json.dumps(conteudo, ensure_ascii=False)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=texto)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                              cache_creation_input_tokens=10, cache_read_input_tokens=5),
        stop_reason="end_turn",
    )


class LotesFalsos:
    def __init__(self, responder, pronto: bool = True) -> None:
        self.responder = responder
        self.pronto = pronto
        self.criados: list[str] = []
        self.guardados: dict[str, list] = {}

    def create(self, requests):
        lote = f"msgbatch_{len(self.criados) + 1}"
        self.criados.append(lote)
        self.guardados[lote] = requests
        return SimpleNamespace(id=lote, processing_status="in_progress")

    def retrieve(self, lote):
        return SimpleNamespace(id=lote, processing_status="ended" if self.pronto else "in_progress")

    def results(self, lote):
        for pedido in self.guardados[lote]:
            yield SimpleNamespace(
                custom_id=pedido["custom_id"],
                result=SimpleNamespace(type="succeeded", message=mensagem(self.responder(pedido["params"]))),
            )


class MensagensFalsas:
    def __init__(self, responder, erros=(), ao_chamar=None, pronto: bool = True) -> None:
        self.responder = responder
        self.erros = list(erros)
        self.ao_chamar = ao_chamar
        self.chamadas: list[dict] = []
        self.trava = threading.Lock()
        self.batches = LotesFalsos(responder, pronto)

    def create(self, **params):
        with self.trava:
            self.chamadas.append(params)
            erro = self.erros.pop(0) if self.erros else None
            if self.ao_chamar is not None:
                self.ao_chamar(len(self.chamadas))
        if erro is not None:
            raise erro
        return mensagem(self.responder(params))


def cliente_falso(responder, **kw) -> SimpleNamespace:
    return SimpleNamespace(messages=MensagensFalsas(responder, **kw))


def fatia_sintetica(i: int) -> dict:
    linhas = [f"Primeira frase da fatia {i} com número {i}.", f"Segunda linha da fatia {i}."]
    texto = "\n".join(linhas)
    base = 10 * i + 1
    return {
        "id": id_curto("a.tex", str(base), texto), "arquivo": "a.tex" if i % 2 == 0 else "b.tex",
        "linha_inicio": base, "linha_fim": base + 1, "secao": f"Cap > Seção {i}", "fonte": "tese",
        "palavras": 12, "texto": texto, "linhas": [base, base + 1],
    }


@pytest.fixture()
def fatias() -> list[dict]:
    return [fatia_sintetica(i) for i in range(6)]


def responder_mapear_de(fatias: list[dict], trecho_ruim_na_fatia: int | None = None):
    por_id = {f["id"]: (n, f) for n, f in enumerate(fatias)}

    def responder(params):
        texto = params["messages"][0]["content"][0]["text"]
        n, fatia = por_id[re.search(r"^Fatia (\S+)$", texto, re.M).group(1)]
        primeira, segunda = fatia["texto"].split("\n")
        trecho = "Texto que não existe na fatia." if n == trecho_ruim_na_fatia else primeira
        return {
            "relacoes": [{"ancora_id": "C01", "relacao": "afirma" if n % 2 == 0 else "contradiz",
                          "trecho": trecho, "linha": fatia["linhas"][0], "justificativa": "j"}],
            "afirmacoes": [{"trecho": segunda, "linha": fatia["linhas"][1], "tipo": "resultado",
                            "topicos": ["P_vigilancia"], "numeros": [], "polaridade": "afirma", "escopo": "esc"}],
        }

    return responder


def rodar_mapear(out: Path, fatias, cliente, modo="sync", concorrencia=1, sleeps=None, settings=None):
    settings = settings or ex.criar_settings("modelo-teste", modo, 1000, concorrencia)
    sleep = sleeps.append if sleeps is not None else (lambda s: None)
    return mp.run_mapear(fatias=fatias, ancoras=ANCORAS, out=out, settings=settings, client=cliente, sleep=sleep)


# ---------------------------------------------------------------------------
# mapear: requisições, sync, retry, retomada
# ---------------------------------------------------------------------------
def test_requisicoes_sem_temperature_e_com_cache_no_sistema(tmp_path: Path, fatias) -> None:
    cliente = cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path / "sync", fatias, cliente)
    cliente_lote = cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path / "lote", fatias, cliente_lote, modo="batch")
    pedidos = [p["params"] for lote in cliente_lote.messages.batches.guardados.values() for p in lote]
    todos = [*cliente.messages.chamadas, *pedidos]
    assert len(todos) == 12
    for params in todos:
        assert "temperature" not in params
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Texto da âncora um." in params["system"][0]["text"]  # âncoras no bloco em cache
        assert "P_vigilancia" in params["system"][0]["text"]  # taxonomia no bloco em cache


def test_concorrencia_maior_que_um_produz_o_mesmo_resultado(tmp_path: Path, fatias) -> None:
    serial = rodar_mapear(tmp_path / "c1", fatias, cliente_falso(responder_mapear_de(fatias)), concorrencia=1)
    paralelo = rodar_mapear(tmp_path / "c4", fatias, cliente_falso(responder_mapear_de(fatias)), concorrencia=4)
    assert serial["linhas"] == paralelo["linhas"]
    assert bytes_gerados(tmp_path / "c1" / MAPA_NOME) == bytes_gerados(tmp_path / "c4" / MAPA_NOME)
    assert serial["pendentes"] == 0 and paralelo["pendentes"] == 0 and not paralelo["incompleto"]
    checkpoint = json.loads((tmp_path / "c4" / "mapear.checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["requisicoes"]) == 6


@pytest.mark.parametrize("erro", [ErroApi(429), ErroApi(529), APIConnectionError()])
def test_erro_transitorio_e_refeito_com_backoff(tmp_path: Path, fatias, erro) -> None:
    cliente = cliente_falso(responder_mapear_de(fatias), erros=[erro])
    sleeps: list[float] = []
    resultado = rodar_mapear(tmp_path, fatias, cliente, concorrencia=3, sleeps=sleeps)
    assert resultado["pendentes"] == 0 and not resultado["incompleto"]
    assert len(cliente.messages.chamadas) == 7  # 6 fatias + a tentativa que falhou
    assert sleeps == [3.0]  # primeiro passo do backoff exponencial
    assert all(linha["erro"] is None for linha in resultado["linhas"])


def test_backoff_exponencial_e_erro_permanente_nao_e_repetido(tmp_path: Path, fatias) -> None:
    cliente = cliente_falso(responder_mapear_de(fatias[:1]), erros=[ErroApi(429), ErroApi(429), ErroApi(529)])
    sleeps: list[float] = []
    resultado = rodar_mapear(tmp_path / "a", fatias[:1], cliente, sleeps=sleeps)
    assert sleeps == [3.0, 6.0, 12.0] and resultado["pendentes"] == 0
    recusado = cliente_falso(responder_mapear_de(fatias[:1]), erros=[ErroApi(400)])
    resultado = rodar_mapear(tmp_path / "b", fatias[:1], recusado, sleeps=[])
    assert len(recusado.messages.chamadas) == 1 and resultado["pendentes"] == 1 and resultado["incompleto"]
    estado = json.loads((tmp_path / "b" / "mapear.checkpoint.json").read_text(encoding="utf-8"))
    assert len(estado["falhas"]) == 1 and estado["requisicoes"] == {}


def test_retomada_nao_reenvia_fatias_ja_respondidas(tmp_path: Path, fatias) -> None:
    primeiro = cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path, fatias[:3], primeiro)
    assert len(primeiro.messages.chamadas) == 3
    segundo = cliente_falso(responder_mapear_de(fatias))
    resultado = rodar_mapear(tmp_path, fatias, segundo, concorrencia=4)
    assert len(segundo.messages.chamadas) == 3  # só as três que faltavam
    enviadas = {re.search(r"^Fatia (\S+)$", c["messages"][0]["content"][0]["text"], re.M).group(1)
                for c in segundo.messages.chamadas}
    assert enviadas == {f["id"] for f in fatias[3:]}
    assert resultado["pendentes"] == 0 and all(linha["erro"] is None for linha in resultado["linhas"])
    terceiro = cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path, fatias, terceiro)
    assert terceiro.messages.chamadas == []


def test_checkpoint_e_gravado_a_cada_resposta(tmp_path: Path, fatias) -> None:
    caminho = tmp_path / "mapear.checkpoint.json"
    vistos: list[int] = []

    def ao_chamar(_n: int) -> None:
        vistos.append(len(json.loads(caminho.read_text(encoding="utf-8"))["requisicoes"]) if caminho.exists() else 0)

    rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias), ao_chamar=ao_chamar))
    assert vistos == [0, 1, 2, 3, 4, 5]  # a resposta anterior já estava no disco antes de cada chamada


def test_json_invalido_fica_registrado_sem_derrubar_a_execucao(tmp_path: Path, fatias) -> None:
    boa = responder_mapear_de(fatias)

    def responder(params):
        texto = params["messages"][0]["content"][0]["text"]
        return "isto não é JSON" if fatias[2]["id"] in texto else boa(params)

    resultado = rodar_mapear(tmp_path, fatias, cliente_falso(responder), concorrencia=2)
    com_erro = [linha for linha in resultado["linhas"] if linha["erro"]]
    assert len(com_erro) == 1 and com_erro[0]["fatia_id"] == fatias[2]["id"]
    assert com_erro[0]["erro"].startswith("json_invalido")
    assert resultado["erros"] == 1 and resultado["pendentes"] == 0
    estado = json.loads((tmp_path / "mapear.checkpoint.json").read_text(encoding="utf-8"))
    registrado = [e for e in estado["requisicoes"].values() if e["erro"]]
    assert len(registrado) == 1 and registrado[0]["resultado"] is None and registrado[0]["amostra"].startswith("isto")
    # --refazer-falhas reenvia só a fatia com erro
    novo = cliente_falso(boa)
    settings = ex.criar_settings("modelo-teste", "sync", 1000, 1, refazer_falhas=True)
    refeito = rodar_mapear(tmp_path, fatias, novo, settings=settings)
    assert len(novo.messages.chamadas) == 1 and refeito["erros"] == 0


def test_resposta_truncada_e_registrada() -> None:
    mensagem_cortada = mensagem('{"relacoes": [')
    mensagem_cortada.stop_reason = "max_tokens"
    req = SimpleNamespace(key="k", passada=1, ids=["k"])
    entrada = ex.nova_entrada(mensagem_cortada, req, "sync")
    assert entrada["erro"] == "resposta_truncada" and entrada["resultado"] is None


def test_trecho_invalido_vai_ao_relatorio_sem_ser_descartado(tmp_path: Path, fatias) -> None:
    resultado = rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias, trecho_ruim_na_fatia=0)))
    assert resultado["trechos_invalidos"] == 1
    linha = resultado["linhas"][0]
    assert linha["relacoes"][0]["trecho_valido"] is False and "trecho_invalido" in linha["relacoes"][0]["flags"]
    write_jsonl_atomic(tmp_path / FATIAS_NOME, fatias)
    saida = rel.gerar_relatorio(tmp_path, ANCORAS)
    texto = saida["relatorio"].read_text(encoding="utf-8")
    assert "### Trechos inválidos (1)" in texto and "Texto que não existe na fatia." in texto
    assert "C01;a.tex;2;0;0;0;0;2" in saida["matriz"].read_text(encoding="utf-8")  # a inválida não conta


# ---------------------------------------------------------------------------
# mapear em lote (Message Batches)
# ---------------------------------------------------------------------------
def test_lote_e_retomavel_sem_reenvio(tmp_path: Path, fatias) -> None:
    cliente = cliente_falso(responder_mapear_de(fatias), pronto=False)
    settings = ex.criar_settings("modelo-teste", "batch", 1000)
    settings.espera_max_s = 0.0  # não esperar o lote: simula interromper a execução
    primeiro = rodar_mapear(tmp_path, fatias, cliente, modo="batch", settings=settings)
    assert primeiro["incompleto"] and primeiro["pendentes"] == 6
    assert cliente.messages.batches.criados == ["msgbatch_1"]
    assert all(linha["erro"] == "pendente" for linha in primeiro["linhas"])
    cliente.messages.batches.pronto = True
    segundo = rodar_mapear(tmp_path, fatias, cliente, modo="batch", settings=settings)
    assert cliente.messages.batches.criados == ["msgbatch_1"]  # o lote registrado foi consultado, não reenviado
    assert not segundo["incompleto"] and all(linha["erro"] is None for linha in segundo["linhas"])
    rodar_mapear(tmp_path, fatias, cliente, modo="batch")
    assert cliente.messages.batches.criados == ["msgbatch_1"]
    assert cliente.messages.chamadas == []  # modo lote não usa messages.create


# ---------------------------------------------------------------------------
# cruzar
# ---------------------------------------------------------------------------
def afirmacao(i: int, fatia_id: str, topicos=("P_vigilancia",), valido: bool = True) -> dict:
    return {
        "id": f"af{i:03d}", "fatia_id": fatia_id, "arquivo": "a.tex" if i % 2 == 0 else "b.tex",
        "trecho": f"Trecho literal número {i}.", "linha": 10 + i, "tipo": "resultado", "topicos": list(topicos),
        "numeros": [str(i)], "polaridade": "afirma", "escopo": "esc", "trecho_valido": valido, "flags": [],
    }


def mapa_sintetico(fatias: list[dict]) -> list[dict]:
    f0, f1 = fatias[0]["id"], fatias[1]["id"]
    relacao = {
        "id": "r1", "fatia_id": f0, "arquivo": "a.tex", "ancora_id": "C01", "relacao": "contradiz",
        "trecho": "Primeira frase da fatia 0 com número 0.", "linha": 1, "justificativa": "j",
        "trecho_valido": True, "flags": [],
    }
    base = {"erro": None, "malformados": 0}
    return [
        {**base, "fatia_id": f0, "arquivo": "a.tex", "afirmacoes": [afirmacao(i, f0) for i in (1, 2, 3)],
         "relacoes": [relacao, {**relacao, "id": "r2", "relacao": "afirma"},
                      {**relacao, "id": "r3", "relacao": "exagera", "trecho_valido": False}]},
        {**base, "fatia_id": f1, "arquivo": "b.tex", "afirmacoes": [afirmacao(i, f1) for i in (4, 5)],
         "relacoes": []},
    ]


def test_agrupamento_por_topico(fatias) -> None:
    mapa = mapa_sintetico(fatias)
    extra = [afirmacao(6, "f", topicos=("P_vigilancia", "custos")), afirmacao(7, "f", topicos=("outro",)),
             afirmacao(8, "f", valido=False)]
    mapa.append({"fatia_id": "f", "afirmacoes": extra, "relacoes": []})
    validas = crz.afirmacoes_validas(mapa)
    assert "af008" not in {a["id"] for a in validas}  # trecho inválido não é cruzado
    grupos = crz.agrupar_por_topico(validas)
    assert list(grupos) == ["P_vigilancia", "custos"]  # "outro" fora; ordem da taxonomia
    assert len(grupos["P_vigilancia"]) == 6 and [a["id"] for a in grupos["custos"]] == ["af006"]
    assert "outro" in crz.agrupar_por_topico(validas, incluir_outro=True)


def test_divisao_de_lotes() -> None:
    assert [len(l) for l in crz.dividir_lotes(list(range(130)), 60)] == [60, 60, 10]
    assert [len(l) for l in crz.dividir_lotes(list(range(5)), 60)] == [5]


def test_planejamento_de_cruzar_divide_topicos_grandes(fatias) -> None:
    mapa = mapa_sintetico(fatias)
    settings = ex.criar_settings("modelo-teste", "sync", 1000)
    um = crz.planejar_cruzar(mapa, settings)
    assert len(um) == 1 and um[0].b is None and len(um[0].a) == 5
    assert "cache_control" not in um[0].req.params["messages"][0]["content"][0]
    # 5 afirmações em lotes de 2: (0,0), (0,1), (0,2), (1,1), (1,2); o lote de 1 não tem par interno
    dividido = crz.planejar_cruzar(mapa, settings, max_afirmacoes=2)
    assert [(len(p.a), len(p.b) if p.b else 0) for p in dividido] == [(2, 0), (2, 2), (2, 1), (2, 0), (2, 1)]
    for plano in dividido:
        assert plano.req.params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "temperature" not in plano.req.params
    assert len({p.req.custom_id for p in dividido}) == 5
    outro = crz.planejar_cruzar(mapa, settings, max_afirmacoes=2)
    assert [p.req.custom_id for p in outro] == [p.req.custom_id for p in dividido]  # estável


def responder_cruzar(params):
    texto = "\n".join(b["text"] for b in params["messages"][0]["content"])
    pares = []
    if "[A1]" in texto and "[A2]" in texto:
        pares.append({"a": "A1", "b": "A2", "tipo": "numero_divergente", "justificativa": "números diferentes"})
        pares.append({"a": "A1", "b": "A9", "tipo": "escopo_trocado", "justificativa": "rótulo inexistente"})
    return {"pares": pares}


def rodar_cruzar(out: Path, mapa, cliente, concorrencia=2, modo="sync", max_afirmacoes=60):
    settings = ex.criar_settings("modelo-teste", modo, 1000, concorrencia)
    return crz.run_cruzar(mapa=mapa, out=out, settings=settings, client=cliente,
                          max_afirmacoes=max_afirmacoes, sleep=lambda s: None)


def test_cruzar_usa_trechos_do_mapa_e_registra_rotulos_invalidos(tmp_path: Path, fatias) -> None:
    mapa = mapa_sintetico(fatias)
    cliente = cliente_falso(responder_cruzar)
    resultado = rodar_cruzar(tmp_path, mapa, cliente)
    assert resultado["requisicoes"] == 1 and not resultado["incompleto"]
    assert len(resultado["pares"]) == 1
    par = resultado["pares"][0]
    # ordem por arquivo e linha: A1 = af002 (a.tex:12), A2 = af004 (a.tex:14)
    assert (par["a"]["arquivo"], par["a"]["linha"], par["a"]["trecho"]) == ("a.tex", 12, "Trecho literal número 2.")
    assert (par["b"]["arquivo"], par["b"]["linha"]) == ("a.tex", 14)
    assert par["tipo"] == "numero_divergente" and par["topico"] == "P_vigilancia"
    assert [p["motivo"] for p in resultado["problemas"]] == ["rotulo_invalido"]
    params = cliente.messages.chamadas[0]
    assert "temperature" not in params and params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert (tmp_path / PARES_NOME).is_file() and (tmp_path / "pares_problemas.jsonl").is_file()


def test_cruzar_concorrencia_e_retomada(tmp_path: Path, fatias) -> None:
    mapa = mapa_sintetico(fatias)
    a = rodar_cruzar(tmp_path / "c1", mapa, cliente_falso(responder_cruzar), concorrencia=1, max_afirmacoes=2)
    b = rodar_cruzar(tmp_path / "c4", mapa, cliente_falso(responder_cruzar), concorrencia=4, max_afirmacoes=2)
    assert a["pares"] == b["pares"] and a["problemas"] == b["problemas"]
    novo = cliente_falso(responder_cruzar)
    rodar_cruzar(tmp_path / "c4", mapa, novo, max_afirmacoes=2)
    assert novo.messages.chamadas == []


# ---------------------------------------------------------------------------
# adjudicar
# ---------------------------------------------------------------------------
def responder_adjudicar(params):
    texto = "\n".join(b["text"] for b in params["messages"][0]["content"])
    if "APONTAMENTO (par" in texto:
        return {"veredito": "CONFIRMADO", "correcao": "Ajustar o número em b.tex:11.", "justificativa": "j"}
    return {"veredito": "DEPENDE_DE_ESCOPO", "correcao": "Explicitar o escopo.", "justificativa": "j"}


def rodar_adjudicar(out: Path, pares, mapa, fatias, cliente, concorrencia=2):
    settings = ex.criar_settings("opus-teste", "sync", 1000, concorrencia)
    return adj.run_adjudicar(pares=pares, mapa=mapa, fatias=fatias, ancoras=ANCORAS, out=out,
                             settings=settings, client=cliente, sleep=lambda s: None)


def test_adjudicar_envia_pares_e_so_relacoes_contradiz_exagera_validas(tmp_path: Path, fatias) -> None:
    mapa = mapa_sintetico(fatias)
    pares = rodar_cruzar(tmp_path / "x", mapa, cliente_falso(responder_cruzar))["pares"]
    cliente = cliente_falso(responder_adjudicar)
    resultado = rodar_adjudicar(tmp_path, pares, mapa, fatias, cliente)
    # um par e a relação r1 (a "afirma" e a "exagera" com trecho inválido ficam de fora)
    assert resultado["requisicoes"] == 2
    assert resultado["vereditos"] == {"CONFIRMADO": 1, "DEPENDE_DE_ESCOPO": 1}
    por_tipo = {linha["alvo_tipo"]: linha for linha in resultado["linhas"]}
    assert por_tipo["par"]["veredito"] == "CONFIRMADO" and por_tipo["par"]["correcao"].startswith("Ajustar")
    assert por_tipo["relacao"]["alvo_id"] == "r1"
    for params in cliente.messages.chamadas:
        assert params["model"] == "opus-teste" and "temperature" not in params
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        # as fatias completas, com número de linha, vão na requisição
        corpo = "\n".join(b["text"] for b in params["messages"][0]["content"])
        assert "Primeira frase da fatia 0" in corpo
    novo = cliente_falso(responder_adjudicar)
    rodar_adjudicar(tmp_path, pares, mapa, fatias, novo)
    assert novo.messages.chamadas == []  # retomada sem reenvio


def test_adjudicar_veredito_invalido_e_registrado(tmp_path: Path, fatias) -> None:
    mapa = mapa_sintetico(fatias)
    resultado = rodar_adjudicar(tmp_path, [], mapa, fatias, cliente_falso(lambda p: {"veredito": "TALVEZ"}))
    assert resultado["requisicoes"] == 1 and resultado["erros"] == 1
    assert resultado["linhas"][0]["erro"] == "veredito_invalido" and resultado["linhas"][0]["veredito"] is None


def test_adjudicar_fatia_ausente_falha_com_mensagem(fatias) -> None:
    mapa = mapa_sintetico(fatias)
    with pytest.raises(ValueError, match="fatiar"):
        adj.planejar_adjudicar([], mapa, fatias[1:], ANCORAS, ex.criar_settings("m", "sync", 10))


# ---------------------------------------------------------------------------
# relatório
# ---------------------------------------------------------------------------
def test_relatorio_conta_custo_a_partir_do_usage(tmp_path: Path, fatias) -> None:
    rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias)))
    write_jsonl_atomic(tmp_path / FATIAS_NOME, fatias)
    saida = rel.gerar_relatorio(tmp_path, ANCORAS)
    texto = saida["relatorio"].read_text(encoding="utf-8")
    assert "| mapear | sync | 6 | 600 | 60 | 30 | 300 |" in texto
    assert "| **Total** | | 6 | 600 | 60 | 30 | 300 |" in texto
    assert "### C01 (atual)" in texto and "`a.tex:1`" in texto  # afirma: a.tex:1, 21, 41
    linhas = saida["matriz"].read_text(encoding="utf-8").splitlines()
    assert linhas[0] == "ancora_id;arquivo;afirma;contradiz;enfraquece;exagera;omite_ressalva;total"
    assert "C01;a.tex;3;0;0;0;0;3" in linhas and "C01;b.tex;0;3;0;0;0;3" in linhas
    assert "C02" in texto and "nenhuma ocorrência" in texto


def test_uso_soma_modos_e_etapas_separados(tmp_path: Path, fatias) -> None:
    rodar_mapear(tmp_path, fatias[:2], cliente_falso(responder_mapear_de(fatias)), modo="sync")
    rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias)), modo="batch")
    uso = rel.uso_por_etapa(tmp_path)
    assert uso["mapear"]["sync"]["requisicoes"] == 2 and uso["mapear"]["batch"]["requisicoes"] == 4
    assert uso["mapear"]["batch"]["output_tokens"] == 200


def test_candidatas_a_ancora_contam_afirmacoes_sem_ancora() -> None:
    def af(n: int, linha: int, numeros=("23",)) -> dict:
        return {"id": f"x{n}", "trecho": f"A meta de 23 aparece aqui {n}.", "linha": linha, "arquivo": f"f{n}.tex",
                "tipo": "resultado", "topicos": ["P_vigilancia"], "numeros": list(numeros), "polaridade": "afirma",
                "escopo": "meta", "trecho_valido": True}

    mapa = [
        {"relacoes": [], "afirmacoes": [af(1, 5), af(2, 6), af(3, 7)]},
        {"relacoes": [{"trecho": "A meta de 23 aparece aqui 4.", "linha": 9, "trecho_valido": True}],
         "afirmacoes": [af(4, 9)]},  # coberta por uma relação com âncora
        {"relacoes": [], "afirmacoes": [af(5, 3, numeros=("12,4%",))]},
    ]
    grupos = rel.candidatas_a_ancora(mapa, min_ocorrencias=3)
    assert len(grupos) == 1 and grupos[0]["ocorrencias"] == 3 and grupos[0]["numeros"] == ["23"]
    assert grupos[0]["locais"] == ["f1.tex:5", "f2.tex:6", "f3.tex:7"]


def test_fluxo_completo_e_relatorio_de_contradicoes(tmp_path: Path, fatias) -> None:
    write_jsonl_atomic(tmp_path / FATIAS_NOME, fatias)
    mapeado = rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias)), concorrencia=3)
    cruzado = rodar_cruzar(tmp_path, mapeado["linhas"], cliente_falso(responder_cruzar))
    assert len(cruzado["pares"]) == 1
    adjudicado = rodar_adjudicar(tmp_path, cruzado["pares"], mapeado["linhas"], fatias,
                                 cliente_falso(responder_adjudicar))
    assert adjudicado["vereditos"] == {"CONFIRMADO": 1, "DEPENDE_DE_ESCOPO": 3}
    texto = rel.gerar_relatorio(tmp_path, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    assert "#### P_vigilancia" in texto and "numero_divergente" in texto and "Ajustar o número em b.tex:11." in texto
    assert "### Depende de escopo" in texto and "Explicitar o escopo." in texto
    assert "| cruzar | sync | 1 |" in texto and "| adjudicar | sync | 4 |" in texto


# ---------------------------------------------------------------------------
# CLI: --estimar, padrões, sem segredos, .gitignore
# ---------------------------------------------------------------------------
def preparar_saida(out: Path, fatias, ancoras_yaml: Path) -> None:
    write_jsonl_atomic(out / FATIAS_NOME, fatias)
    write_jsonl_atomic(out / MAPA_NOME, mapa_sintetico(fatias))
    escrever(ancoras_yaml, "".join(
        f"- id: {a['id']}\n  origem: {a['origem']}\n  texto: {a['texto']}\n" for a in ANCORAS))


def test_estimar_nao_chama_a_api_nem_grava_checkpoint(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    par = {"id": "p1", "tipo": "numero_divergente", "topico": "P_vigilancia", "topicos": ["P_vigilancia"],
           "justificativa": "j", "flags": [],
           "a": {"afirmacao_id": "af002", "fatia_id": fatias[0]["id"], "arquivo": "a.tex", "linha": 12, "trecho": "t"},
           "b": {"afirmacao_id": "af004", "fatia_id": fatias[1]["id"], "arquivo": "a.tex", "linha": 14, "trecho": "u"}}
    write_jsonl_atomic(out / PARES_NOME, [par])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def proibido(chave):
        raise AssertionError("a estimativa não pode criar cliente da API")

    monkeypatch.setattr(cli.av, "make_client", proibido)
    base = ["--out", str(out)]
    assert cli.main(["mapear", "--estimar", "--ancoras", str(yaml_path), *base]) == 0
    assert cli.main(["cruzar", "--estimar", *base]) == 0
    assert cli.main(["adjudicar", "--estimar", "--ancoras", str(yaml_path), *base]) == 0
    saida = capsys.readouterr().out
    assert re.search(r"^mapear: 6 requisições; entrada ≈ [\d.]+ tokens; saída ≈ [\d.]+ tokens", saida, re.M)
    assert re.search(r"^cruzar: 1 requisições; entrada ≈", saida, re.M)
    assert re.search(r"^adjudicar: 2 requisições; entrada ≈", saida, re.M)
    assert not list(out.glob("*.checkpoint.json")) and not (out / "lotes").exists()


def test_estimar_mapear_cresce_com_o_texto(fatias) -> None:
    settings = ex.criar_settings("m", "sync", 4000)
    pequeno = mp.estimar_mapear(fatias[:2], ANCORAS, settings)
    grande = mp.estimar_mapear(fatias, ANCORAS, settings)
    assert pequeno["requisicoes"] == 2 and grande["requisicoes"] == 6
    assert grande["entrada"] > pequeno["entrada"] and grande["saida"] > pequeno["saida"] > 0


def test_cli_mapear_sync_ponta_a_ponta_sem_vazar_a_chave(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    segredo = "chave-falsa-para-teste-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", segredo)
    cliente = cliente_falso(responder_mapear_de(fatias))
    chaves: list[str] = []

    def fabrica(chave):
        chaves.append(chave)
        return cliente

    monkeypatch.setattr(cli.av, "make_client", fabrica)
    argv = ["mapear", "--modo", "sync", "--concorrencia", "3", "--ancoras", str(yaml_path), "--out", str(out)]
    assert cli.main(argv) == 0
    assert chaves == [segredo] and len(cliente.messages.chamadas) == 6
    assert cli.main(argv) == 0 and len(cliente.messages.chamadas) == 6  # retomada: nada reenviado
    capturado = capsys.readouterr()
    assert segredo not in capturado.out and segredo not in capturado.err
    for arquivo in out.rglob("*"):
        if arquivo.is_file():
            assert segredo not in arquivo.read_text(encoding="utf-8", errors="ignore")


def test_cli_sem_chave_e_sem_fatias(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    assert cli.main(["mapear", "--out", str(out), "--ancoras", str(yaml_path)]) == 2  # sem fatias.jsonl
    preparar_saida(out, fatias, yaml_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_env_var", lambda nome: "")
    assert cli.main(["mapear", "--out", str(out), "--ancoras", str(yaml_path)]) == 2
    assert "ANTHROPIC_API_KEY ausente" in capsys.readouterr().err


def test_chave_env_padrao_e_anthropic_api_key() -> None:
    parser = cli.build_parser()
    for etapa in ("mapear", "pinos", "cruzar", "adjudicar"):
        assert parser.parse_args([etapa]).chave_env == "ANTHROPIC_API_KEY"
        assert parser.parse_args([etapa, "--chave-env", "OUTRA"]).chave_env == "OUTRA"


def test_cli_chave_env_alt_cria_o_cliente_com_o_valor_dessa_variavel(
        tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    padrao, alt = "valor-padrao-falso-111", "valor-alt-falso-222"
    monkeypatch.setenv("ANTHROPIC_API_KEY", padrao)
    monkeypatch.setenv("ANTHROPIC_API_KEY_ALT", alt)
    cliente = cliente_falso(responder_mapear_de(fatias))
    chaves: list[str] = []

    def fabrica(chave):
        chaves.append(chave)
        return cliente

    monkeypatch.setattr(cli.av, "make_client", fabrica)
    argv = ["mapear", "--modo", "sync", "--ancoras", str(yaml_path), "--out", str(out)]
    assert cli.main([*argv, "--chave-env", "ANTHROPIC_API_KEY_ALT"]) == 0
    assert chaves == [alt]
    capturado = capsys.readouterr()
    for segredo in (padrao, alt):
        assert segredo not in capturado.out and segredo not in capturado.err


def test_cli_chave_env_chega_a_todas_as_etapas_com_cliente(tmp_path: Path, fatias, monkeypatch) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    par = {"id": "p1", "tipo": "numero_divergente", "topico": "P_vigilancia", "topicos": ["P_vigilancia"],
           "justificativa": "j", "flags": [],
           "a": {"afirmacao_id": "af002", "fatia_id": fatias[0]["id"], "arquivo": "a.tex", "linha": 12, "trecho": "t"},
           "b": {"afirmacao_id": "af004", "fatia_id": fatias[1]["id"], "arquivo": "a.tex", "linha": 14, "trecho": "u"}}
    write_jsonl_atomic(out / PARES_NOME, [par])
    nomes: list[str] = []

    def cliente_recusado(nome="ANTHROPIC_API_KEY"):
        nomes.append(nome)
        return None, 2

    monkeypatch.setattr(cli, "_cliente", cliente_recusado)
    base = ["--chave-env", "NOME_QUALQUER", "--out", str(out)]
    assert cli.main(["mapear", "--ancoras", str(yaml_path), *base]) == 2
    assert cli.main(["cruzar", *base]) == 2
    assert cli.main(["adjudicar", "--ancoras", str(yaml_path), *base]) == 2
    assert nomes == ["NOME_QUALQUER"] * 3


def test_cli_chave_env_ausente_sai_com_2_citando_so_o_nome(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "valor-padrao-falso-333")
    monkeypatch.delenv("ANTHROPIC_API_KEY_ALT", raising=False)
    monkeypatch.setattr(cli, "load_env_var", lambda nome: "" if nome == "ANTHROPIC_API_KEY_ALT" else "x")

    def proibido(chave):
        raise AssertionError("sem a variável escolhida não pode haver cliente")

    monkeypatch.setattr(cli.av, "make_client", proibido)
    argv = ["mapear", "--ancoras", str(yaml_path), "--out", str(out), "--chave-env", "ANTHROPIC_API_KEY_ALT"]
    assert cli.main(argv) == 2
    capturado = capsys.readouterr()
    assert "ANTHROPIC_API_KEY_ALT ausente" in capturado.err
    assert "valor-padrao-falso-333" not in capturado.out + capturado.err


# ---------------------------------------------------------------------------
# Várias chaves: rodízio, concorrência por chave, failover, lote por chave, relatório
# ---------------------------------------------------------------------------
class ErroCredito(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("Your credit balance is too low to access the Anthropic API.")
        self.status_code = status


def fatia_do_pedido(params: dict) -> str:
    return re.search(r"^Fatia (\S+)$", params["messages"][0]["content"][0]["text"], re.M).group(1)


def ler_checkpoint(out: Path, etapa: str = "mapear") -> dict:
    return json.loads((out / f"{etapa}.checkpoint.json").read_text(encoding="utf-8"))


def indice_no_plano(fatias, settings) -> dict[str, int]:
    return {r.custom_id: i for i, r in enumerate(mp.planejar_mapear(fatias, ANCORAS, settings))}


def test_rodizio_entre_duas_chaves_e_deterministico_e_sobrevive_a_retomada(tmp_path: Path, fatias) -> None:
    settings = ex.criar_settings("modelo-teste", "sync", 1000, 2)
    indice = indice_no_plano(fatias, settings)
    a, b = cliente_falso(responder_mapear_de(fatias)), cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path, fatias[:3], {"K_A": a, "K_B": b}, settings=settings)  # interrompida na metade
    a2, b2 = cliente_falso(responder_mapear_de(fatias)), cliente_falso(responder_mapear_de(fatias))
    resultado = rodar_mapear(tmp_path, fatias, {"K_A": a2, "K_B": b2}, settings=settings)
    assert resultado["pendentes"] == 0
    estado = ler_checkpoint(tmp_path)
    assert len(estado["requisicoes"]) == 6
    for cid, entrada in estado["requisicoes"].items():
        assert entrada["chave"] == ("K_A", "K_B")[indice[cid] % 2]
    por_fatia = {f["id"]: n for n, f in enumerate(fatias)}
    enviados = {nome: sorted(por_fatia[fatia_do_pedido(p)] for p in c.messages.chamadas)
                for nome, c in (("A1", a), ("B1", b), ("A2", a2), ("B2", b2))}
    assert enviados == {"A1": [0, 2], "B1": [1], "A2": [4], "B2": [3, 5]}


def test_concorrencia_vale_por_chave(tmp_path: Path) -> None:
    fatias8 = [fatia_sintetica(i) for i in range(8)]
    responder = responder_mapear_de(fatias8)
    trava = threading.Lock()
    ativos = {"K_A": 0, "K_B": 0}
    picos = {"K_A": 0, "K_B": 0}
    pico_total = [0]
    barreira = threading.Barrier(4, timeout=10)  # só passa com 2 chaves x 2 em voo ao mesmo tempo

    def cliente(nome: str) -> SimpleNamespace:
        def create(**params):
            with trava:
                ativos[nome] += 1
                picos[nome] = max(picos[nome], ativos[nome])
                pico_total[0] = max(pico_total[0], sum(ativos.values()))
            try:
                barreira.wait()
                return mensagem(responder(params))
            finally:
                with trava:
                    ativos[nome] -= 1

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    resultado = rodar_mapear(tmp_path, fatias8, {"K_A": cliente("K_A"), "K_B": cliente("K_B")}, concorrencia=2)
    assert resultado["pendentes"] == 0 and not resultado["incompleto"]
    assert picos == {"K_A": 2, "K_B": 2} and pico_total[0] == 4


@pytest.mark.parametrize("erro", [ErroApi(401), ErroApi(403), ErroCredito(400), ErroCredito(402)])
def test_failover_quando_uma_chave_cai(tmp_path: Path, fatias, erro) -> None:
    caida = cliente_falso(responder_mapear_de(fatias), erros=[erro] * 10)
    boa = cliente_falso(responder_mapear_de(fatias))
    logs: list[str] = []
    settings = ex.criar_settings("modelo-teste", "sync", 1000, 1)
    resultado = mp.run_mapear(fatias=fatias, ancoras=ANCORAS, out=tmp_path, settings=settings,
                              client={"K_A": caida, "K_B": boa}, sleep=lambda s: None, log=logs.append)
    assert resultado["pendentes"] == 0 and not resultado["incompleto"]
    assert len(caida.messages.chamadas) == 1  # sem retry na chave caída
    assert len(boa.messages.chamadas) == 6
    estado = ler_checkpoint(tmp_path)
    assert {e["chave"] for e in estado["requisicoes"].values()} == {"K_B"}
    assert [f["motivo"] for f in estado["falhas"]] == [f"chave_indisponivel:K_A status={erro.status_code}"]
    assert any("chave K_A indisponível" in linha and "K_B" in linha for linha in logs)


def test_erro_400_sem_credito_nao_derruba_a_chave(tmp_path: Path, fatias) -> None:
    a = cliente_falso(responder_mapear_de(fatias), erros=[ErroApi(400)])
    b = cliente_falso(responder_mapear_de(fatias))
    resultado = rodar_mapear(tmp_path, fatias, {"K_A": a, "K_B": b})
    assert resultado["pendentes"] == 1 and len(a.messages.chamadas) == 3 and len(b.messages.chamadas) == 3


def test_cli_todas_as_chaves_indisponiveis_sai_com_3_e_checkpoint(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    segredos = {"CHAVE_TESTE_A": "segredo-falso-aaa-111", "CHAVE_TESTE_B": "segredo-falso-bbb-222"}
    for nome, valor in segredos.items():
        monkeypatch.setenv(nome, valor)
    clientes = {segredos["CHAVE_TESTE_A"]: cliente_falso(responder_mapear_de(fatias), erros=[ErroApi(401)] * 10),
                segredos["CHAVE_TESTE_B"]: cliente_falso(responder_mapear_de(fatias), erros=[ErroCredito(400)] * 10)}
    monkeypatch.setattr(cli.av, "make_client", lambda chave: clientes[chave])
    argv = ["mapear", "--modo", "sync", "--concorrencia", "1", "--chave-env", "CHAVE_TESTE_A,CHAVE_TESTE_B",
            "--ancoras", str(yaml_path), "--out", str(out)]
    assert cli.main(argv) == 3
    estado = ler_checkpoint(out)
    assert estado["requisicoes"] == {}
    assert sorted(f["motivo"].split(" ")[0] for f in estado["falhas"]) == [
        "chave_indisponivel:CHAVE_TESTE_A", "chave_indisponivel:CHAVE_TESTE_B"]
    capturado = capsys.readouterr()
    assert "todas as chaves indisponíveis" in capturado.out
    for valor in segredos.values():
        assert valor not in capturado.out and valor not in capturado.err
        for arquivo in out.rglob("*"):
            if arquivo.is_file():
                assert valor not in arquivo.read_text(encoding="utf-8", errors="ignore")


def test_lote_fica_associado_a_chave_que_o_criou(tmp_path: Path, fatias) -> None:
    a = cliente_falso(responder_mapear_de(fatias), pronto=False)
    b = cliente_falso(responder_mapear_de(fatias), pronto=False)
    consultas: list[str] = []
    def espiar(nome: str, original):
        def retrieve(lote):
            consultas.append(nome)
            return original(lote)
        return retrieve

    for nome, c in (("K_A", a), ("K_B", b)):
        c.messages.batches.retrieve = espiar(nome, c.messages.batches.retrieve)
    settings = ex.criar_settings("modelo-teste", "batch", 1000)
    settings.espera_max_s = 0.0
    indice = indice_no_plano(fatias, settings)
    primeiro = rodar_mapear(tmp_path, fatias, {"K_A": a, "K_B": b}, modo="batch", settings=settings)
    assert primeiro["incompleto"] and primeiro["pendentes"] == 6
    lotes = ler_checkpoint(tmp_path)["lotes"]
    assert [lote["chave"] for lote in lotes] == ["K_A", "K_B"]
    for lote in lotes:
        assert {indice[cid] % 2 for cid in lote["custom_ids"]} == {0 if lote["chave"] == "K_A" else 1}
    assert a.messages.batches.criados == ["msgbatch_1"] and b.messages.batches.criados == ["msgbatch_1"]
    a.messages.batches.pronto = b.messages.batches.pronto = True
    segundo = rodar_mapear(tmp_path, fatias, {"K_A": a, "K_B": b}, modo="batch", settings=settings)
    assert not segundo["incompleto"] and segundo["pendentes"] == 0
    assert a.messages.batches.criados == ["msgbatch_1"] and b.messages.batches.criados == ["msgbatch_1"]
    assert consultas == ["K_A", "K_B", "K_A", "K_B"]  # cada lote consultado pela própria chave
    estado = ler_checkpoint(tmp_path)
    for cid, entrada in estado["requisicoes"].items():
        assert entrada["modo"] == "batch" and entrada["chave"] == ("K_A", "K_B")[indice[cid] % 2]
    # sem a chave do lote na execução, o lote não é consultado com outra chave
    c = cliente_falso(responder_mapear_de(fatias), pronto=False)
    logs: list[str] = []
    mp.run_mapear(fatias=fatias, ancoras=ANCORAS, out=tmp_path / "outro", settings=settings,
                  client={"K_A": c}, sleep=lambda s: None, log=logs.append)
    estado_outro = ler_checkpoint(tmp_path / "outro")
    estado_outro["lotes"][0]["chave"] = "K_B"
    (tmp_path / "outro" / "mapear.checkpoint.json").write_text(json.dumps(estado_outro), encoding="utf-8")
    c.messages.batches.pronto = True
    restante = mp.run_mapear(fatias=fatias, ancoras=ANCORAS, out=tmp_path / "outro", settings=settings,
                             client={"K_A": c}, sleep=lambda s: None, log=logs.append)
    assert restante["pendentes"] == 6 and any("pertence à chave K_B" in linha for linha in logs)


def test_lote_com_chave_caida_vai_para_a_restante(tmp_path: Path, fatias) -> None:
    a = cliente_falso(responder_mapear_de(fatias))
    a.messages.batches.create = lambda requests: (_ for _ in ()).throw(ErroApi(401))
    b = cliente_falso(responder_mapear_de(fatias))
    settings = ex.criar_settings("modelo-teste", "batch", 1000)
    resultado = rodar_mapear(tmp_path, fatias, {"K_A": a, "K_B": b}, modo="batch", settings=settings)
    assert resultado["pendentes"] == 0
    lotes = ler_checkpoint(tmp_path)["lotes"]
    assert [lote["chave"] for lote in lotes] == ["K_B", "K_B"]
    assert sorted(len(lote["custom_ids"]) for lote in lotes) == [3, 3]


def test_relatorio_separa_uso_por_chave(tmp_path: Path, fatias) -> None:
    a, b = cliente_falso(responder_mapear_de(fatias)), cliente_falso(responder_mapear_de(fatias))
    rodar_mapear(tmp_path, fatias, {"K_A": a, "K_B": b})
    uso = rel.uso_por_chave(tmp_path)
    assert uso["K_A"]["mapear"]["sync"]["requisicoes"] == 3 and uso["K_B"]["mapear"]["sync"]["output_tokens"] == 150
    write_jsonl_atomic(tmp_path / FATIAS_NOME, fatias)
    texto = rel.gerar_relatorio(tmp_path, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    assert "| mapear | sync | 6 | 600 | 60 | 30 | 300 |" in texto  # total por etapa continua
    assert "| K_A | mapear | sync | 3 | 300 | 30 | 15 | 150 |" in texto
    assert "| K_B | mapear | sync | 3 | 300 | 30 | 15 | 150 |" in texto
    assert "| K_A | **Total** | | 3 | 300 | 30 | 15 | 150 |" in texto


def test_relatorio_marca_checkpoint_antigo_sem_chave(tmp_path: Path, fatias) -> None:
    rodar_mapear(tmp_path, fatias[:2], cliente_falso(responder_mapear_de(fatias)))
    estado = ler_checkpoint(tmp_path)
    for entrada in estado["requisicoes"].values():
        del entrada["chave"]
    (tmp_path / "mapear.checkpoint.json").write_text(json.dumps(estado), encoding="utf-8")
    assert rel.uso_por_chave(tmp_path)[rel.CHAVE_NAO_REGISTRADA]["mapear"]["sync"]["requisicoes"] == 2


def test_cli_duas_chaves_sem_vazar_valores(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    segredos = {"ANTHROPIC_API_KEY": "segredo-falso-principal-1", "ANTHROPIC_API_KEY_ALT": "segredo-falso-alt-2"}
    for nome, valor in segredos.items():
        monkeypatch.setenv(nome, valor)
    clientes = {valor: cliente_falso(responder_mapear_de(fatias)) for valor in segredos.values()}
    criados: list[str] = []

    def fabrica(chave):
        criados.append(chave)
        return clientes[chave]

    monkeypatch.setattr(cli.av, "make_client", fabrica)
    argv = ["mapear", "--modo", "sync", "--chave-env", "ANTHROPIC_API_KEY,ANTHROPIC_API_KEY_ALT",
            "--ancoras", str(yaml_path), "--out", str(out)]
    assert cli.main(argv) == 0
    assert criados == list(segredos.values())
    assert all(len(c.messages.chamadas) == 3 for c in clientes.values())
    assert cli.main(["relatorio", "--ancoras", str(yaml_path), "--out", str(out)]) == 0
    assert "| ANTHROPIC_API_KEY_ALT | mapear | sync | 3 |" in (out / rel.RELATORIO_NOME).read_text(encoding="utf-8")
    capturado = capsys.readouterr()
    for valor in segredos.values():
        assert valor not in capturado.out and valor not in capturado.err
        for arquivo in out.rglob("*"):
            if arquivo.is_file():
                assert valor not in arquivo.read_text(encoding="utf-8", errors="ignore")


def test_cli_chave_env_invalida_ou_parcial_sai_com_2_sem_eco(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    monkeypatch.setattr(cli.av, "make_client", lambda chave: pytest.fail("não pode criar cliente"))
    colado = "sk-ant-valor-colado-por-engano"
    assert cli.main(["mapear", "--chave-env", colado, "--ancoras", str(yaml_path), "--out", str(out)]) == 2
    monkeypatch.setenv("CHAVE_TESTE_A", "segredo-falso-ccc-333")
    monkeypatch.setattr(cli, "load_env_var", lambda nome: "segredo-falso-ccc-333" if nome == "CHAVE_TESTE_A" else "")
    assert cli.main(["mapear", "--chave-env", "CHAVE_TESTE_A,CHAVE_TESTE_B", "--ancoras", str(yaml_path),
                     "--out", str(out)]) == 2
    capturado = capsys.readouterr()
    assert "--chave-env inválido" in capturado.err and "CHAVE_TESTE_B ausente" in capturado.err
    assert colado not in capturado.out + capturado.err and "segredo-falso-ccc-333" not in capturado.out + capturado.err


def test_padroes_da_linha_de_comando() -> None:
    parser = cli.build_parser()
    for etapa in ("mapear", "cruzar", "adjudicar"):
        args = parser.parse_args([etapa])
        assert args.concorrencia == 8 and args.modo == "batch" and args.estimar is False
        assert args.out == ".local/cruzamento"
    assert parser.parse_args(["mapear"]).modelo == "claude-sonnet-5-5"
    assert parser.parse_args(["cruzar"]).modelo == "claude-sonnet-5-5"
    assert parser.parse_args(["adjudicar"]).modelo == "claude-opus-5-5"
    assert parser.parse_args(["cruzar"]).max_afirmacoes == 60
    assert parser.parse_args(["mapear", "--modo", "sync", "--concorrencia", "2"]).concorrencia == 2


def test_saidas_ficam_em_pasta_ignorada_pelo_git() -> None:
    linhas = {linha.strip() for linha in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert ".local/" in linhas
    assert cli.DEFAULT_OUT.startswith(".local/")


# ---------------------------------------------------------------------------
# Ficha de valores conferidos
# ---------------------------------------------------------------------------
REGISTRO_YAML = """\
- id: N01
  descricao: "Total de potenciais pares do extrato"
  tipo: numero
  valores:
    - {nome: n_pares_total, fonte: {arquivo: data/extrato_resumo.csv, filtro: {metrica: n_pares_candidatos}, coluna: valor}, texto_pt: "61.696"}
    - {nome: so_auxiliar, fonte: {derivado: "a / b"}, auxiliar: true}
- id: N02
  descricao: "Razão de desbalanceamento"
  tipo: numero
  valores:
    - {nome: razao, fonte: {derivado: "n_par0 / n_vinculos", formato: "1:{:.0f}"}, texto_pt: "1:249"}
- id: N03
  descricao: "Entrada N com tipo afirmacao"
  tipo: afirmacao
  valores:
    - {nome: valor_de_afirmacao, fonte: {arquivo: data/x.csv, coluna: y}, texto_pt: "999"}
- id: C01
  descricao: "Afirmação central"
  tipo: afirmacao
  ocorrencias: []
"""


def test_carregar_ficha_so_numericas_com_texto_pt(tmp_path: Path) -> None:
    ficha = carregar_ficha(escrever(tmp_path / "registro.yaml", REGISTRO_YAML))
    linhas = ficha.splitlines()
    assert len(linhas) == 2
    assert linhas[0].startswith("- N01 (Total de potenciais pares do extrato) | n_pares_total = 61.696 | ")
    assert "arquivo data/extrato_resumo.csv" in linhas[0] and "coluna valor" in linhas[0]
    assert "filtro metrica=n_pares_candidatos" in linhas[0]
    assert "derivado: n_par0 / n_vinculos" in linhas[1] and "= 1:249" in linhas[1]
    assert "N03" not in ficha and "C01" not in ficha and "so_auxiliar" not in ficha
    assert carregar_ficha(tmp_path / "inexistente.yaml") == ""


def test_ficha_do_registro_real_tem_as_numericas() -> None:
    ficha = carregar_ficha(REPO / "tests" / "data" / "registro_invariantes.yaml")
    assert "- N01 (" in ficha and "n_pares_total = 61.696" in ficha
    assert "- C01" not in ficha


def test_prompts_de_sistema_levam_a_ficha_so_quando_ha_ficha(fatias) -> None:
    ficha = "- N01 (Total) | n_pares_total = 61.696 | arquivo data/x.csv; coluna valor"
    settings = ex.criar_settings("m", "sync", 100)
    mapa = mapa_sintetico(fatias)
    pares = [{"id": "p1", "tipo": "numero_divergente", "topico": "P_vigilancia", "topicos": ["P_vigilancia"],
              "justificativa": "j", "flags": [],
              "a": {"afirmacao_id": "af002", "fatia_id": fatias[0]["id"], "arquivo": "a.tex", "linha": 12, "trecho": "t"},
              "b": {"afirmacao_id": "af004", "fatia_id": fatias[1]["id"], "arquivo": "a.tex", "linha": 14, "trecho": "u"}}]
    com = [
        mp.planejar_mapear(fatias, ANCORAS, settings, ficha)[0].params,
        crz.planejar_cruzar(mapa, settings, ficha=ficha)[0].req.params,
        adj.planejar_adjudicar(pares, mapa, fatias, ANCORAS, settings, ficha)[0].req.params,
    ]
    sem = [
        mp.planejar_mapear(fatias, ANCORAS, settings)[0].params,
        crz.planejar_cruzar(mapa, settings)[0].req.params,
        adj.planejar_adjudicar(pares, mapa, fatias, ANCORAS, settings)[0].req.params,
    ]
    for params in com:
        sistema = params["system"][0]["text"]
        assert "FICHA DE VALORES CONFERIDOS" in sistema and ficha in sistema and "id da ficha" in sistema
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    for params in sem:
        assert "FICHA DE VALORES CONFERIDOS" not in params["system"][0]["text"]
    # o prompt muda com a ficha, então o custom_id também (sem colisão com checkpoints sem ficha)
    assert mp.planejar_mapear(fatias, ANCORAS, settings, ficha)[0].custom_id != \
        mp.planejar_mapear(fatias, ANCORAS, settings)[0].custom_id


def test_cli_mapear_ficha_do_registro_e_sem_ficha(tmp_path: Path, fatias, monkeypatch) -> None:
    yaml_path = tmp_path / "ancoras.yaml"
    registro = escrever(tmp_path / "registro.yaml", REGISTRO_YAML)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-falsa")
    sistemas = {}
    for nome, extra in (("com", []), ("sem", ["--sem-ficha"])):
        out = tmp_path / nome
        preparar_saida(out, fatias, yaml_path)
        cliente = cliente_falso(responder_mapear_de(fatias))
        monkeypatch.setattr(cli.av, "make_client", lambda chave, c=cliente: c)
        argv = ["mapear", "--modo", "sync", "--limite", "1", "--ancoras", str(yaml_path), "--out", str(out),
                "--registro", str(registro), *extra]
        assert cli.main(argv) == 0
        sistemas[nome] = cliente.messages.chamadas[0]["system"][0]["text"]
    assert "n_pares_total = 61.696" in sistemas["com"] and "N01" in sistemas["com"]
    assert "FICHA DE VALORES CONFERIDOS" not in sistemas["sem"] and "61.696" not in sistemas["sem"]


def test_padroes_da_ficha_na_linha_de_comando() -> None:
    parser = cli.build_parser()
    for etapa in ("mapear", "cruzar", "adjudicar"):
        args = parser.parse_args([etapa])
        assert args.sem_ficha is False and args.registro == "tests/data/registro_invariantes.yaml"
        assert parser.parse_args([etapa, "--sem-ficha"]).sem_ficha is True


# ---------------------------------------------------------------------------
# pinos
# ---------------------------------------------------------------------------
COLUNAS_PINOS = ["arquivo_teste", "funcao", "linha", "tipo", "literal", "alvo", "categoria",
                 "afetado_pela_terminologia", "proposta_migracao", "linha_assert", "detalhe", "deteccao"]


def pino_csv(i: int, categoria: str = "frase", alvo: str = "text/latex/capitulo3.tex",
             arquivo: str = "tests/test_a.py") -> dict:
    return {"arquivo_teste": arquivo, "funcao": f"test_{i}", "linha": str(100 + i), "tipo": "contem",
            "literal": f"literal {i}", "alvo": alvo, "categoria": categoria, "afetado_pela_terminologia": "nao",
            "proposta_migracao": "manter", "linha_assert": str(100 + i), "detalhe": "", "deteccao": "direta"}


def escrever_inventario(caminho: Path, linhas: list[dict]) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_PINOS, delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)
    return caminho


def responder_pinos(params):
    texto = params["messages"][0]["content"][0]["text"]
    itens = []
    for pid, resto in re.findall(r"^\[([0-9a-f]{16})\] (.*)$", texto, re.M):
        forte = '"literal 3"' in resto
        itens.append({
            "pino_id": pid, "afirmacao_travada": "afirmação travada", "duplica_ancora": None,
            "conflito": "forte_demais" if forte else "nenhum", "ancora_conflito": "C01" if forte else None,
            "central_sem_ancora": '"literal 4"' in resto, "justificativa": "j",
        })
    return {"pinos": itens}


def rodar_pinos(out: Path, pinos, cliente, tamanho=40, meta=None, concorrencia=1, modo="sync"):
    settings = ex.criar_settings("modelo-teste", modo, 1000, concorrencia)
    return pn.run_pinos(pinos=pinos, ancoras=ANCORAS, out=out, settings=settings, client=cliente,
                        tamanho_lote=tamanho, meta=meta, sleep=lambda s: None)


def test_pinos_filtra_categorias_e_le_inventario_com_bom(tmp_path: Path) -> None:
    categorias = ["frase", "negacao", "ressalva", "numero", "termo", "citacao", "estrutural"]
    caminho = escrever_inventario(tmp_path / "inv.csv", [pino_csv(i, c) for i, c in enumerate(categorias)])
    linhas = pn.ler_inventario(caminho)
    assert len(linhas) == 7 and linhas[0]["arquivo_teste"] == "tests/test_a.py"  # BOM não vaza para a coluna
    pinos, meta = pn.selecionar(linhas)
    assert [p["categoria"] for p in pinos] == ["frase", "negacao", "ressalva"]
    assert meta["linhas_inventario"] == 7 and meta["pinos_de_conteudo"] == 3 and meta["excluidos_slides"] == 0


def test_pinos_agrupa_em_lotes_de_40_com_ancoras_em_cache(tmp_path: Path) -> None:
    pinos, _ = pn.selecionar([pino_csv(i) for i in range(85)])
    assert [len(l) for l in pn.agrupar(pinos, 40)] == [40, 40, 5]
    assert [len(l) for l in pn.agrupar(pinos)] == [40, 40, 5]  # padrão
    cliente = cliente_falso(responder_pinos)
    resultado = rodar_pinos(tmp_path, pinos, cliente)
    assert resultado["requisicoes"] == 3 and len(cliente.messages.chamadas) == 3
    assert len(resultado["linhas"]) == 85 and resultado["erros"] == 0 and not resultado["incompleto"]
    for params in cliente.messages.chamadas:
        assert "temperature" not in params
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Texto da âncora um." in params["system"][0]["text"]
    corpo = cliente.messages.chamadas[0]["messages"][0]["content"][0]["text"]
    assert len(re.findall(r"^\[[0-9a-f]{16}\] ", corpo, re.M)) == 40
    saida = [json.loads(l) for l in (tmp_path / PINOS_NOME).read_text(encoding="utf-8").splitlines()]
    assert len(saida) == 85 and {"pino_id", "afirmacao_travada", "duplica_ancora", "conflito", "ancora_conflito",
                                 "central_sem_ancora", "justificativa"} <= set(saida[0])
    novo = cliente_falso(responder_pinos)
    rodar_pinos(tmp_path, pinos, novo)
    assert novo.messages.chamadas == []  # retomada pelo checkpoint


def test_pino_id_e_estavel(tmp_path: Path) -> None:
    a = pn.pino_id("tests/test_a.py", "test_1", "101", "literal 1")
    assert a == pn.pino_id("tests/test_a.py", "test_1", "101", "literal 1") and len(a) == 16
    assert a != pn.pino_id("tests/test_a.py", "test_1", "102", "literal 1")
    assert a == id_curto("tests/test_a.py", "test_1", "101", "literal 1")
    linhas = [pino_csv(i) for i in range(5)]
    direto, _ = pn.selecionar(linhas)
    invertido, _ = pn.selecionar(list(reversed(linhas)))
    assert [p["pino_id"] for p in direto] == [p["pino_id"] for p in invertido]
    settings = ex.criar_settings("m", "sync", 100)
    ids1 = [r.custom_id for r, _ in pn.planejar_pinos(direto, ANCORAS, settings)]
    ids2 = [r.custom_id for r, _ in pn.planejar_pinos(invertido, ANCORAS, settings)]
    assert ids1 == ids2


def test_pinos_json_invalido_e_registrado_como_falha(tmp_path: Path) -> None:
    pinos, _ = pn.selecionar([pino_csv(i) for i in range(85)])
    boa = responder_pinos

    def responder(params):
        texto = params["messages"][0]["content"][0]["text"]
        return "isto não é JSON" if '"literal 45"' in texto else boa(params)

    resultado = rodar_pinos(tmp_path, pinos, cliente_falso(responder))
    com_erro = [l for l in resultado["linhas"] if l["erro"]]
    assert len(com_erro) == 40 and resultado["erros"] == 40 and resultado["pendentes"] == 0
    assert all(l["erro"].startswith("json_invalido") and l["conflito"] is None for l in com_erro)
    estado = json.loads((tmp_path / "pinos.checkpoint.json").read_text(encoding="utf-8"))
    assert sum(1 for e in estado["requisicoes"].values() if e["erro"]) == 1


def test_pinos_resposta_incompleta_ou_invalida_nao_e_descartada(tmp_path: Path) -> None:
    pinos, _ = pn.selecionar([pino_csv(i) for i in range(3)])

    def responder(params):
        saida = responder_pinos(params)["pinos"]
        saida[0]["conflito"] = "grave"  # fora da lista
        saida[1]["ancora_conflito"] = "C99"  # âncora inexistente
        return {"pinos": saida[:2]}  # o terceiro falta

    linhas = rodar_pinos(tmp_path, pinos, cliente_falso(responder))["linhas"]
    assert [l["erro"] for l in linhas] == ["conflito_invalido", None, "ausente_na_resposta"]
    assert linhas[1]["flags"] == ["ancora_desconhecida"]


def test_pinos_exclui_alvo_so_de_slides_por_padrao() -> None:
    alvos = [
        "text/slides/defesa.tex",  # excluído
        "text/slides/defesa.tex|text/slides/roteiro_defesa.md",  # excluído
        "(inferido) text/slides/roteiro_defesa.md",  # excluído
        "text/latex/capitulo3.tex|text/slides/defesa.tex",  # tem arquivo de text/latex
        "text/slides/defesa.tex|+2",  # truncado: pode haver outros
        "(dinâmico)",
        "text/latex/conclusao.tex",
    ]
    linhas = [pino_csv(i, alvo=a) for i, a in enumerate(alvos)]
    pinos, meta = pn.selecionar(linhas)
    assert meta["excluidos_slides"] == 3 and meta["pinos_enviados"] == 4 and len(pinos) == 4
    assert {p["alvo"] for p in pinos} == set(alvos[3:])
    todos, meta2 = pn.selecionar(linhas, incluir_slides=True)
    assert len(todos) == 7 and meta2["excluidos_slides"] == 0 and meta2["incluir_slides"] is True
    assert pn.alvo_so_slides("text/slides/defesa.tex") and not pn.alvo_so_slides("")


def test_relatorio_secao_pinos(tmp_path: Path, fatias) -> None:
    rodar_mapear(tmp_path, fatias, cliente_falso(responder_mapear_de(fatias)))
    write_jsonl_atomic(tmp_path / FATIAS_NOME, fatias)
    sem_pinos = rel.gerar_relatorio(tmp_path, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    assert "## Pinos" in sem_pinos and "não executado" in sem_pinos
    linhas = [pino_csv(i) for i in range(6)] + [pino_csv(10 + i, alvo="text/slides/defesa.tex") for i in range(2)]
    pinos, meta = pn.selecionar(linhas)
    rodar_pinos(tmp_path, pinos, cliente_falso(responder_pinos), meta=meta)
    texto = rel.gerar_relatorio(tmp_path, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    secao = texto.split("\n## Pinos\n")[1].split("\n## Custo")[0]
    assert "excluídos (slides e roteiro serão refeitos): 2" in secao
    assert "forte_demais 1" in secao and "nenhum 5" in secao
    assert "`tests/test_a.py:103` \"literal 3\" (âncora C01): j" in secao
    assert "### Pinos centrais sem âncora (1)" in secao and "`tests/test_a.py:104` \"literal 4\"" in secao
    assert "| pinos | sync | 1 |" in texto


def test_cli_pinos_ponta_a_ponta_estimar_e_slides(tmp_path: Path, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = escrever(tmp_path / "ancoras.yaml", "".join(
        f"- id: {a['id']}\n  origem: {a['origem']}\n  texto: {a['texto']}\n" for a in ANCORAS))
    linhas = [pino_csv(i) for i in range(5)] + [pino_csv(20, alvo="text/slides/defesa.tex"),
                                                pino_csv(21, categoria="numero")]
    inventario = escrever_inventario(tmp_path / "inv.csv", linhas)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-falsa")
    cliente = cliente_falso(responder_pinos)
    monkeypatch.setattr(cli.av, "make_client", lambda chave: cliente)
    base = ["pinos", "--inventario", str(inventario), "--ancoras", str(yaml_path), "--out", str(out)]
    assert cli.main([*base, "--estimar"]) == 0
    saida = capsys.readouterr().out
    assert "excluídos por alvo só em slides/roteiro: 1; enviados: 5" in saida
    assert re.search(r"^pinos: 1 requisições; entrada ≈", saida, re.M)
    assert cliente.messages.chamadas == [] and not list(out.glob("*.checkpoint.json"))
    assert cli.main([*base, "--modo", "sync"]) == 0
    assert len(cliente.messages.chamadas) == 1
    assert len((out / PINOS_NOME).read_text(encoding="utf-8").splitlines()) == 5
    assert json.loads((out / "pinos_meta.json").read_text(encoding="utf-8"))["excluidos_slides"] == 1
    assert cli.main([*base, "--modo", "sync", "--incluir-slides"]) == 0
    assert len((out / PINOS_NOME).read_text(encoding="utf-8").splitlines()) == 6
    assert cli.main(["pinos", "--inventario", str(tmp_path / "nao_existe.csv"), "--ancoras", str(yaml_path),
                     "--out", str(out)]) == 2


def test_padroes_do_pinos_na_linha_de_comando() -> None:
    args = cli.build_parser().parse_args(["pinos"])
    assert args.inventario == ".local/fase0/inventario_pinos.csv" and args.tamanho_lote == 40
    assert args.incluir_slides is False and args.modelo == "claude-sonnet-5-5" and args.modo == "batch"


# ---------------------------------------------------------------------------
# Segundo leitor: DeepSeek V4.1 Flash via OpenCode Go (mapear --provedor opencode, --mapas, concordância)
# ---------------------------------------------------------------------------
class RespostaHttp:
    def __init__(self, status: int, corpo: dict | None = None, texto: str = "") -> None:
        self.status_code = status
        self._corpo = corpo
        self.text = texto

    def json(self) -> dict:
        if self._corpo is None:
            raise ValueError("sem corpo")
        return self._corpo


class ErroDeRede(OSError):
    """Falha de conexão no transporte (como ``requests.ConnectionError``)."""


class HttpFalso:
    """Dublê de ``requests``: grava as chamadas e responde no formato ``chat/completions``."""

    def __init__(self, responder, erros=()) -> None:
        self.responder = responder
        self.erros = list(erros)
        self.chamadas: list[dict] = []
        self.trava = threading.Lock()

    def post(self, url, headers=None, **kw):
        payload = kw["json"]
        with self.trava:
            self.chamadas.append({"url": url, "headers": headers, "json": payload, "timeout": kw.get("timeout")})
            erro = self.erros.pop(0) if self.erros else None
        if isinstance(erro, int):
            return RespostaHttp(erro, texto="falha do gateway")
        if erro is not None:
            raise erro
        resultado = self.responder({"messages": [{"content": [{"text": payload["messages"][-1]["content"]}]}]})
        corpo = {
            "choices": [{"message": {"content": json.dumps(resultado, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return RespostaHttp(200, corpo)


def test_opencode_monta_a_requisicao_sem_vazar_a_chave(fatias) -> None:
    segredo = "chave-opencode-falsa-777"
    http = HttpFalso(responder_mapear_de(fatias))
    cliente = oc.ClienteOpencode(segredo, http=http)
    settings = ex.criar_settings(oc.MODELO_OPENCODE_PADRAO, "sync", 1000)
    req = mp.planejar_mapear(fatias[:1], ANCORAS, settings)[0]
    resposta = cliente.messages.create(**req.params)
    chamada = http.chamadas[0]
    assert chamada["url"] == "https://opencode.ai/zen/go/v1/chat/completions"
    assert chamada["headers"]["Authorization"] == f"Bearer {segredo}" and chamada["headers"]["x-opencode-session"]
    corpo = chamada["json"]
    assert corpo["model"] == "deepseek-v4.1-flash" and corpo["max_tokens"] == 1000 and corpo["temperature"] == 0.0
    assert corpo["thinking"] == {"type": "disabled"} and corpo["reasoning_effort"] == "none"
    assert [m["role"] for m in corpo["messages"]] == ["system", "user"]
    assert "Texto da âncora um." in corpo["messages"][0]["content"]  # mesmo prompt de sistema do Sonnet
    assert f"Fatia {fatias[0]['id']}" in corpo["messages"][1]["content"]
    assert req.params["system"][0]["text"] == corpo["messages"][0]["content"]
    assert "cache_control" not in json.dumps(corpo) and segredo not in json.dumps(corpo)
    entrada = ex.nova_entrada(resposta, req, "sync", oc.CHAVE_ENV_OPENCODE)
    assert entrada["erro"] is None and entrada["chave"] == "OPENCODE_GO_API_KEY"
    assert entrada["uso"] == {"input_tokens": 100, "output_tokens": 50,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert len(entrada["resultado"]["relacoes"]) == 1 and segredo not in json.dumps(entrada)


def test_opencode_resposta_cortada_e_erros_sem_a_chave() -> None:
    cortada = oc._mensagem({"choices": [{"message": {"content": "{"}, "finish_reason": "length"}], "usage": {}})
    assert cortada.stop_reason == "max_tokens" and cortada.usage.input_tokens == 0
    segredo = "chave-opencode-falsa-888"
    params = ex.montar_params(ex.criar_settings("deepseek-v4.1-flash", "sync", 10), "sistema", [ex.bloco_texto("u")])
    for erro in (401, 429, 500):
        cliente = oc.ClienteOpencode(segredo, http=HttpFalso(lambda p: {}, erros=[erro]))
        with pytest.raises(oc.ErroOpencode) as info:
            cliente.messages.create(**params)
        assert info.value.status_code == erro and segredo not in str(info.value) + info.value.body
    rede = oc.ClienteOpencode(segredo, http=HttpFalso(lambda p: {}, erros=[ErroDeRede(f"falha {segredo}")]))
    with pytest.raises(oc.APIConnectionError) as info:
        rede.messages.create(**params)
    assert segredo not in str(info.value)


def test_opencode_retry_com_backoff_e_retomada_no_checkpoint_proprio(tmp_path: Path, fatias) -> None:
    http = HttpFalso(responder_mapear_de(fatias), erros=[429, ErroDeRede("rede")])
    sleeps: list[float] = []
    settings = ex.criar_settings(oc.MODELO_OPENCODE_PADRAO, "sync", 1000, 1)
    resultado = mp.run_mapear(fatias=fatias[:2], ancoras=ANCORAS, out=tmp_path, settings=settings,
                              client=oc.ClienteOpencode("k", http=http), sleep=sleeps.append,
                              etapa="mapear_opencode", mapa_nome=MAPA_OPENCODE_NOME)
    assert resultado["pendentes"] == 0 and resultado["erros"] == 0
    assert sleeps == [3.0, 6.0]  # 429 e depois erro de rede na primeira requisição: backoff exponencial
    assert (tmp_path / MAPA_OPENCODE_NOME).is_file() and (tmp_path / "mapear_opencode.checkpoint.json").is_file()
    assert not (tmp_path / MAPA_NOME).exists() and not (tmp_path / "mapear.checkpoint.json").exists()
    novo = HttpFalso(responder_mapear_de(fatias))
    mp.run_mapear(fatias=fatias[:2], ancoras=ANCORAS, out=tmp_path, settings=settings,
                  client=oc.ClienteOpencode("k", http=novo), sleep=sleeps.append,
                  etapa="mapear_opencode", mapa_nome=MAPA_OPENCODE_NOME)
    assert novo.chamadas == []


def test_cli_mapear_opencode_grava_em_arquivo_separado_sem_tocar_no_sonnet(
        tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    mapa_sonnet = bytes_gerados(out / MAPA_NOME)
    segredo = "chave-opencode-falsa-999"
    monkeypatch.setenv("OPENCODE_GO_API_KEY", segredo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_env_var", lambda nome: segredo if nome == "OPENCODE_GO_API_KEY" else "")
    http = HttpFalso(responder_mapear_de(fatias, trecho_ruim_na_fatia=2))
    monkeypatch.setattr(cli.av, "make_client", lambda chave: pytest.fail("não pode usar o cliente da Anthropic"))
    monkeypatch.setattr(cli.oc, "criar_cliente", lambda chave: oc.ClienteOpencode(chave, http=http))
    # sem --modo (padrão batch) e sem --modelo: o provedor opencode força sync e o DeepSeek V4.1 Flash
    argv = ["mapear", "--provedor", "opencode", "--concorrencia", "3", "--ancoras", str(yaml_path),
            "--out", str(out)]
    assert cli.main(argv) == 0
    assert len(http.chamadas) == 6
    assert {c["json"]["model"] for c in http.chamadas} == {"deepseek-v4.1-flash"}
    assert all(c["headers"]["Authorization"] == f"Bearer {segredo}" for c in http.chamadas)
    assert bytes_gerados(out / MAPA_NOME) == mapa_sonnet  # mapa.jsonl do Sonnet intocado
    assert not (out / "mapear.checkpoint.json").exists()
    linhas = [json.loads(l) for l in (out / MAPA_OPENCODE_NOME).read_text(encoding="utf-8").splitlines()]
    assert len(linhas) == 6
    invalidos = [r for l in linhas for r in l["relacoes"] if not r["trecho_valido"]]
    assert len(invalidos) == 1 and invalidos[0]["flags"] == ["trecho_invalido"]  # verificação literal vale igual
    checkpoint = json.loads((out / "mapear_opencode.checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["modelo"] == "deepseek-v4.1-flash" and len(checkpoint["requisicoes"]) == 6
    for entrada in checkpoint["requisicoes"].values():
        assert entrada["chave"] == "OPENCODE_GO_API_KEY" and entrada["modo"] == "sync"
        assert entrada["uso"]["input_tokens"] == 100 and entrada["uso"]["output_tokens"] == 50  # usage registrado
    assert cli.main(argv) == 0 and len(http.chamadas) == 6  # retomada: nada reenviado
    capturado = capsys.readouterr()
    assert f"saída: {out / MAPA_OPENCODE_NOME}" in capturado.out
    assert segredo not in capturado.out and segredo not in capturado.err
    for arquivo in out.rglob("*"):
        if arquivo.is_file():
            assert segredo not in arquivo.read_text(encoding="utf-8", errors="ignore")
    texto = rel.gerar_relatorio(out, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    assert "| mapear_opencode | sync | 6 | 600 | 0 | 0 | 300 |" in texto
    assert "| OPENCODE_GO_API_KEY | **Total** |" in texto


def test_cli_mapear_opencode_sem_chave_sai_com_2(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    monkeypatch.setattr(cli, "load_env_var", lambda nome: "")
    monkeypatch.setattr(cli.oc, "criar_cliente", lambda chave: pytest.fail("sem chave não cria cliente"))
    assert cli.main(["mapear", "--provedor", "opencode", "--ancoras", str(yaml_path), "--out", str(out)]) == 2
    assert "OPENCODE_GO_API_KEY ausente" in capsys.readouterr().err
    assert not (out / MAPA_OPENCODE_NOME).exists()


def linha_de_mapa(fatia: dict, resultado: dict) -> dict:
    return {
        "fatia_id": fatia["id"], "arquivo": fatia["arquivo"], "linha_inicio": fatia["linha_inicio"],
        "secao": fatia["secao"], "fonte": fatia["fonte"], "erro": None,
        **mp.processar_resultado(fatia, resultado, {"C01", "C02"}),
    }


def mapas_dos_dois_leitores(fatias: list[dict]) -> tuple[list[dict], list[dict]]:
    """Mapas do Sonnet e do DeepSeek sobre a fatia 0, construídos com a verificação real de trechos."""
    f = fatias[0]
    primeira, segunda = f["texto"].split("\n")

    def relacao(ancora: str, tipo: str, trecho: str, justificativa: str = "j") -> dict:
        return {"ancora_id": ancora, "relacao": tipo, "trecho": trecho, "linha": 0, "justificativa": justificativa}

    def afirmacao_de(trecho: str) -> dict:
        return {"trecho": trecho, "linha": f["linhas"][1], "tipo": "resultado", "topicos": ["P_vigilancia"],
                "numeros": [], "polaridade": "afirma", "escopo": "esc"}

    sonnet = [linha_de_mapa(f, {"relacoes": [relacao("C01", "contradiz", primeira)],
                                "afirmacoes": [afirmacao_de(segunda)]})]
    opencode = [linha_de_mapa(f, {
        "relacoes": [
            relacao("C01", "contradiz", primeira, "outra justificativa"),  # igual à do Sonnet: dedup
            relacao("C01", "exagera", primeira),  # mesmo trecho, outra relação: só DeepSeek
            relacao("C02", "omite_ressalva", segunda),  # só DeepSeek
            relacao("C01", "contradiz", "Texto que não existe na fatia."),  # trecho inválido
        ],
        "afirmacoes": [afirmacao_de(segunda.replace(" ", "  "))],  # mesmo trecho com espaços extras: dedup
    })]
    return sonnet, opencode


def test_uniao_deduplica_por_fatia_e_trecho_guardando_o_leitor(fatias) -> None:
    sonnet, opencode = mapas_dos_dois_leitores(fatias)
    uniao = ld.unir_mapas([("sonnet", sonnet), ("opencode", opencode)])
    assert len(uniao) == 1 and uniao[0]["erro"] is None
    relacoes = {(r["ancora_id"], r["relacao"], r["trecho_valido"]): r for r in uniao[0]["relacoes"]}
    assert len(uniao[0]["relacoes"]) == 4
    assert relacoes[("C01", "contradiz", True)]["leitores"] == ["sonnet", "opencode"]
    assert relacoes[("C01", "contradiz", True)]["justificativa"] == "j"  # fica o item do primeiro leitor
    assert relacoes[("C01", "exagera", True)]["leitores"] == ["opencode"]
    assert relacoes[("C02", "omite_ressalva", True)]["leitores"] == ["opencode"]
    invalida = relacoes[("C01", "contradiz", False)]
    assert invalida["leitores"] == ["opencode"] and invalida["flags"] == ["trecho_invalido"]
    assert len(uniao[0]["afirmacoes"]) == 1 and uniao[0]["afirmacoes"][0]["leitores"] == ["sonnet", "opencode"]
    # ids estáveis: os do Sonnet continuam os mesmos na união
    assert {r["id"] for r in sonnet[0]["relacoes"]} <= {r["id"] for r in uniao[0]["relacoes"]}
    assert sonnet[0]["afirmacoes"][0]["id"] == uniao[0]["afirmacoes"][0]["id"] == opencode[0]["afirmacoes"][0]["id"]
    assert ld.unir_mapas([("opencode", opencode), ("sonnet", sonnet)])[0]["relacoes"][0]["leitores"] == \
        ["opencode", "sonnet"]
    # só o que tem trecho verificado segue para cruzar e adjudicar
    # (a ordem é por arquivo, linha e id, e o id depende do hash: compara-se como conjunto)
    assert {r["relacao"] for r in adj.relacoes_adjudicaveis(uniao, {"C01", "C02"})} == {"contradiz", "exagera"}
    assert len(adj.relacoes_adjudicaveis(uniao, {"C01", "C02"})) == 2
    assert len(crz.afirmacoes_validas(uniao)) == 1


def test_uniao_com_um_mapa_devolve_as_linhas_sem_alteracao(fatias) -> None:
    sonnet, _ = mapas_dos_dois_leitores(fatias)
    assert ld.unir_mapas([("sonnet", sonnet)]) is sonnet
    assert "leitores" not in sonnet[0]["relacoes"][0]


def test_uniao_erro_so_quando_todos_os_leitores_falharam(fatias) -> None:
    sonnet, opencode = mapas_dos_dois_leitores(fatias)
    pendente = {**sonnet[0], "erro": "pendente", "relacoes": [], "afirmacoes": []}
    assert ld.unir_mapas([("sonnet", [pendente]), ("opencode", opencode)])[0]["erro"] is None
    falha = {**opencode[0], "erro": "resposta_truncada", "relacoes": [], "afirmacoes": []}
    assert ld.unir_mapas([("sonnet", [pendente]), ("opencode", [falha])])[0]["erro"] == "pendente"


def test_carregar_mapas_nomes_leitores_e_erros(tmp_path: Path, fatias) -> None:
    sonnet, opencode = mapas_dos_dois_leitores(fatias)
    assert ld.separar_mapas(" mapa.jsonl, mapa_opencode.jsonl,mapa.jsonl ,") == [MAPA_NOME, MAPA_OPENCODE_NOME]
    assert ld.nome_do_leitor(MAPA_NOME) == "sonnet" and ld.nome_do_leitor(MAPA_OPENCODE_NOME) == "opencode"
    assert ld.nome_do_leitor("mapa_kimi.jsonl") == "kimi"
    write_jsonl_atomic(tmp_path / MAPA_NOME, sonnet)
    mapa, nomes, erro = ld.carregar_mapas(tmp_path, [MAPA_NOME])
    assert erro is None and nomes == ["sonnet"] and mapa == sonnet
    mapa, nomes, erro = ld.carregar_mapas(tmp_path, [MAPA_NOME, MAPA_OPENCODE_NOME])
    assert mapa == [] and f"{tmp_path / MAPA_OPENCODE_NOME} ausente ou vazio; rode `mapear` antes" == erro
    write_jsonl_atomic(tmp_path / MAPA_OPENCODE_NOME, opencode)
    mapa, nomes, erro = ld.carregar_mapas(tmp_path, [MAPA_NOME, MAPA_OPENCODE_NOME])
    assert erro is None and nomes == ["sonnet", "opencode"] and len(mapa[0]["relacoes"]) == 4
    assert ld.carregar_mapas(tmp_path, [])[2] is not None


def escrever_dois_mapas(out: Path, fatias: list[dict]) -> None:
    sonnet, opencode = mapas_dos_dois_leitores(fatias)
    write_jsonl_atomic(out / FATIAS_NOME, fatias)
    write_jsonl_atomic(out / MAPA_NOME, sonnet)
    write_jsonl_atomic(out / MAPA_OPENCODE_NOME, opencode)


def test_relatorio_secao_de_concordancia_entre_leitores(tmp_path: Path, fatias) -> None:
    escrever_dois_mapas(tmp_path, fatias)
    padrao = rel.gerar_relatorio(tmp_path, ANCORAS)["relatorio"].read_text(encoding="utf-8")
    assert "Concordância entre leitores" not in padrao  # padrão: só mapa.jsonl, relatório como antes
    texto = rel.gerar_relatorio(tmp_path, ANCORAS, arquivos_mapas=(MAPA_NOME, MAPA_OPENCODE_NOME))[
        "relatorio"].read_text(encoding="utf-8")
    secao = texto.split("\n## Concordância entre leitores\n")[1].split("\n## Contradições confirmadas")[0]
    assert "Leitores: sonnet, opencode." in secao
    c01 = secao.split("### C01 (atual)")[1].split("### C02")[0]
    assert "| Relação | Só sonnet | Só opencode | Ambos |" in c01
    assert "| contradiz | 0 | 0 | 1 |" in c01 and "| exagera | 0 | 1 | 0 |" in c01  # o inválido não conta
    assert "- só opencode: exagera `a.tex:1`" in c01
    c02 = secao.split("### C02 (proposta)")[1]
    assert "| omite_ressalva | 0 | 1 | 0 |" in c02 and "- só opencode: omite_ressalva `a.tex:2`" in c02
    assert "Trechos inválidos (1)" in texto  # a verificação literal do DeepSeek entra no relatório
    linhas = rel.gerar_relatorio(tmp_path, ANCORAS, arquivos_mapas=(MAPA_NOME, MAPA_OPENCODE_NOME))["matriz"]
    assert "C01;a.tex;0;1;0;1;0;2" in linhas.read_text(encoding="utf-8").splitlines()


def test_concordancia_so_do_sonnet_e_so_do_deepseek_por_ancora(fatias) -> None:
    sonnet, opencode = mapas_dos_dois_leitores(fatias)
    # o Sonnet acha ainda uma relação que o DeepSeek não achou
    sonnet[0]["relacoes"].append({**sonnet[0]["relacoes"][0], "id": "so-sonnet", "relacao": "enfraquece"})
    uniao = ld.unir_mapas([("sonnet", sonnet), ("opencode", opencode)])
    linhas = ld.concordancia_por_ancora(uniao, ["sonnet", "opencode"], ["C01", "C02"])
    assert {(l["ancora_id"], l["relacao"]): (l["so_sonnet"], l["so_opencode"], l["ambos"]) for l in linhas} == {
        ("C01", "contradiz"): (0, 0, 1), ("C01", "enfraquece"): (1, 0, 0), ("C01", "exagera"): (0, 1, 0),
        ("C02", "omite_ressalva"): (0, 1, 0),
    }


def test_cli_cruzar_e_adjudicar_aceitam_mapas_e_o_padrao_nao_muda(tmp_path: Path, fatias, monkeypatch, capsys) -> None:
    out = tmp_path / "saida"
    yaml_path = tmp_path / "ancoras.yaml"
    preparar_saida(out, fatias, yaml_path)
    outro = mapa_sintetico(fatias)
    outro[0]["relacoes"] = [{**outro[0]["relacoes"][0], "id": "r9", "relacao": "exagera"}]
    outro[1]["afirmacoes"] = [afirmacao(9, fatias[1]["id"])]
    write_jsonl_atomic(out / MAPA_OPENCODE_NOME, outro)
    par = {"id": "p1", "tipo": "numero_divergente", "topico": "P_vigilancia", "topicos": ["P_vigilancia"],
           "justificativa": "j", "flags": [],
           "a": {"afirmacao_id": "af002", "fatia_id": fatias[0]["id"], "arquivo": "a.tex", "linha": 12, "trecho": "t"},
           "b": {"afirmacao_id": "af004", "fatia_id": fatias[1]["id"], "arquivo": "a.tex", "linha": 14, "trecho": "u"}}
    write_jsonl_atomic(out / PARES_NOME, [par])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-falsa")
    cliente = cliente_falso(lambda params: {"pares": []})
    monkeypatch.setattr(cli.av, "make_client", lambda chave: cliente)
    base = ["--out", str(out)]
    uniao = ["--mapas", f"{MAPA_NOME},{MAPA_OPENCODE_NOME}"]

    def textos() -> list[str]:
        return [ex.texto_dos_params(p) for p in cliente.messages.chamadas]

    # padrão: só mapa.jsonl, mesmo com mapa_opencode.jsonl na pasta
    assert cli.main(["cruzar", "--modo", "sync", *base]) == 0
    assert len(cliente.messages.chamadas) == 1 and "Trecho literal número 9." not in textos()[0]
    write_jsonl_atomic(out / PARES_NOME, [par])  # o cruzar com resposta vazia regravou pares.jsonl sem pares
    assert cli.main(["adjudicar", "--estimar", "--ancoras", str(yaml_path), *base]) == 0
    assert re.search(r"^adjudicar: 2 requisições", capsys.readouterr().out, re.M)  # par + r1
    # com --mapas: afirmações e relações dos dois leitores
    assert cli.main(["cruzar", "--modo", "sync", *uniao, *base]) == 0
    assert len(cliente.messages.chamadas) == 2 and "Trecho literal número 9." in textos()[1]
    assert "Trecho literal número 5." in textos()[1]
    assert cli.main(["cruzar", "--estimar", *uniao, *base]) == 0
    write_jsonl_atomic(out / PARES_NOME, [par])  # o cruzar com resposta vazia regravou pares.jsonl sem pares
    assert cli.main(["adjudicar", "--estimar", "--ancoras", str(yaml_path), *uniao, *base]) == 0
    assert re.search(r"^adjudicar: 3 requisições", capsys.readouterr().out, re.M)  # par + r1 + r9
    # mapa ausente: erro de insumo citando o arquivo
    assert cli.main(["cruzar", "--estimar", "--mapas", "mapa_kimi.jsonl", *base]) == 2
    assert "mapa_kimi.jsonl ausente ou vazio; rode `mapear` antes" in capsys.readouterr().err
    assert cli.main(["adjudicar", "--estimar", "--ancoras", str(yaml_path), "--mapas", "", *base]) == 2


def test_cli_relatorio_mapas_e_padroes_da_linha_de_comando(tmp_path: Path, fatias) -> None:
    parser = cli.build_parser()
    for etapa in ("cruzar", "adjudicar", "relatorio"):
        assert parser.parse_args([etapa]).mapas == "mapa.jsonl"
    assert parser.parse_args(["mapear"]).provedor == "anthropic"
    assert parser.parse_args(["mapear", "--provedor", "opencode"]).provedor == "opencode"
    out = tmp_path / "saida"
    escrever_dois_mapas(out, fatias)
    yaml_path = escrever(tmp_path / "ancoras.yaml", "".join(
        f"- id: {a['id']}\n  origem: {a['origem']}\n  texto: {a['texto']}\n" for a in ANCORAS))
    assert cli.main(["relatorio", "--ancoras", str(yaml_path), "--out", str(out)]) == 0
    assert "Concordância entre leitores" not in (out / "relatorio.md").read_text(encoding="utf-8")
    assert cli.main(["relatorio", "--ancoras", str(yaml_path), "--out", str(out),
                     "--mapas", f"{MAPA_NOME},{MAPA_OPENCODE_NOME}"]) == 0
    assert "## Concordância entre leitores" in (out / "relatorio.md").read_text(encoding="utf-8")


def test_cli_mapear_padrao_continua_gravando_mapa_jsonl(tmp_path: Path, fatias, monkeypatch) -> None:
    out = tmp_path / "saida"
    yaml_path = escrever(tmp_path / "ancoras.yaml", "".join(
        f"- id: {a['id']}\n  origem: {a['origem']}\n  texto: {a['texto']}\n" for a in ANCORAS))
    write_jsonl_atomic(out / FATIAS_NOME, fatias)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-falsa")
    cliente = cliente_falso(responder_mapear_de(fatias))
    monkeypatch.setattr(cli.av, "make_client", lambda chave: cliente)
    monkeypatch.setattr(cli.oc, "criar_cliente", lambda chave: pytest.fail("anthropic não usa o OpenCode Go"))
    assert cli.main(["mapear", "--modo", "sync", "--ancoras", str(yaml_path), "--out", str(out)]) == 0
    assert (out / MAPA_NOME).is_file() and (out / "mapear.checkpoint.json").is_file()
    assert not (out / MAPA_OPENCODE_NOME).exists() and not (out / "mapear_opencode.checkpoint.json").exists()
    assert {c["model"] for c in cliente.messages.chamadas} == {"claude-sonnet-5-5"}
