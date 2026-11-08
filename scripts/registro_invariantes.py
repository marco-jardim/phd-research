#!/usr/bin/env python
"""Registro de invariantes da tese: biblioteca e CLI.

O registro (``tests/data/registro_invariantes.yaml``) lista os números e as
afirmações centrais da tese. Este módulo faz duas verificações independentes:

1. *Artefato*: recalcula cada valor a partir do CSV/JSON de origem, formata no
   padrão pt-BR (vírgula decimal, ponto de milhar) e compara com o texto que o
   registro declara (``texto_pt``/``texto_en``).
2. *Texto*: para cada ocorrência (arquivo e, no caso do Resumo/Abstract, seção),
   exige que exista ao menos um parágrafo contendo todos os tokens exigidos.
   Não se fixa frase alguma: só âncoras de conceito (números e termos técnicos).

Uso::

    python scripts/registro_invariantes.py [--relatorio .local/fase0/registro_relatorio.md]

Código de saída 1 se houver falha inesperada (ou XPASS de uma ocorrência marcada
``xfail``).
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import json
import math
import operator
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRO = ROOT / "tests" / "data" / "registro_invariantes.yaml"
# Raiz dos artefatos de dados (mesma variável de tests/local_data.py).
RAIZ_DADOS = Path(os.environ.get("PHD_RESEARCH_DATA_ROOT", str(ROOT)))

MENSAGEM_INDISPONIVEL = (
    "Private/local inputs unavailable; set PHD_RESEARCH_DATA_ROOT: ")


class ErroRegistro(Exception):
    """O registro ou o artefato não permite calcular/avaliar o item."""


class FonteIndisponivel(Exception):
    """Arquivo ausente neste checkout (dado local ou texto não versionado)."""


# --------------------------------------------------------------------------
# Formatação numérica
# --------------------------------------------------------------------------

def _decimal(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def _arredondar(x, casas: int) -> Decimal:
    d = x if isinstance(x, Decimal) else _decimal(x)
    return d.quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)


def formatar_pt(x, casas: int) -> str:
    """Número (float ou Decimal) no padrão pt-BR: ``1.234,5``."""
    d = _arredondar(x, casas)
    s = format(abs(d), f",.{casas}f").replace(",", "\0").replace(".", ",")
    s = s.replace("\0", ".")
    return ("-" + s) if d < 0 else s


def pt_para_en(texto: str) -> str:
    """``1.234,5`` -> ``1,234.5`` (troca os separadores)."""
    return texto.translate({ord("."): ",", ord(","): "."})


_TRANSFORMACAO = re.compile(r"^(int|dec(\d)|pct(\d))$")


def formatar_valor(bruto: float, transformacao: str | None,
                   formato: str | None) -> tuple[str, str]:
    """Devolve ``(texto_pt, texto_en)`` do valor bruto.

    Transformações: ``int``, ``decN`` (N casas), ``pctN`` (x100, N casas); o
    prefixo ``abs_`` toma o módulo antes (ex.: ``abs_int``).
    """
    if formato is not None:
        t = formato.format(bruto)
        return t, t
    if transformacao is not None and str(transformacao).startswith("abs_"):
        bruto = abs(bruto)
        transformacao = str(transformacao)[4:]
    if transformacao is None:
        if float(bruto).is_integer():
            transformacao = "int"
        else:
            raise ErroRegistro(
                f"valor não inteiro {bruto!r} exige 'transformacao' (decN/pctN)")
    m = _TRANSFORMACAO.match(str(transformacao))
    if not m:
        raise ErroRegistro(f"transformação desconhecida: {transformacao!r}")
    if m.group(1) == "int":
        t = formatar_pt(bruto, 0)
    elif m.group(2) is not None:
        t = formatar_pt(bruto, int(m.group(2)))
    else:
        t = formatar_pt(_decimal(bruto) * 100, int(m.group(3)))
    return t, pt_para_en(t)


# --------------------------------------------------------------------------
# Avaliador aritmético seguro (sem eval)
# --------------------------------------------------------------------------

def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centro - meia, centro + meia


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}
_FUNCOES = {
    "abs": abs, "min": min, "max": max,
    "wilson_inf": lambda k, n: wilson(k, n)[0],
    "wilson_sup": lambda k, n: wilson(k, n)[1],
}


def avaliar(expressao: str, nomes) -> float:
    """Avalia aritmética simples; ``nomes(id)`` resolve identificadores."""
    try:
        arvore = ast.parse(str(expressao), mode="eval")
    except SyntaxError as exc:
        raise ErroRegistro(f"expressão inválida {expressao!r}: {exc}") from exc

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if (isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)):
            return n.value
        if isinstance(n, ast.Name):
            return nomes(n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _FUNCOES and not n.keywords):
            return _FUNCOES[n.func.id](*[ev(a) for a in n.args])
        raise ErroRegistro(f"construção não permitida em {expressao!r}")

    return ev(arvore)


# --------------------------------------------------------------------------
# Leitura de artefatos
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _linhas_csv(caminho: str) -> tuple[dict, ...]:
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        cabecalho = f.readline()
        f.seek(0)
        delim = ";" if cabecalho.count(";") > cabecalho.count(",") else ","
        return tuple(csv.DictReader(f, delimiter=delim))


@functools.lru_cache(maxsize=None)
def _json(caminho: str):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def _numero(s) -> float:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace(",", "."))
        except ValueError as exc:
            raise ErroRegistro(f"célula não numérica: {s!r}") from exc


def _igual(celula, alvo) -> bool:
    if str(celula).strip() == str(alvo).strip():
        return True
    try:
        return float(celula) == float(alvo)
    except (TypeError, ValueError):
        return False


@functools.lru_cache(maxsize=None)
def _rastreado(raiz: str, arquivo: str) -> bool:
    """True se ``arquivo`` está rastreado pelo git em ``raiz`` (uma consulta por caminho)."""
    extra = {}
    if sys.platform == "win32":
        extra["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        r = subprocess.run(
            ["git", "-C", raiz, "ls-files", "--error-unmatch", "--", arquivo],
            capture_output=True, check=False, timeout=30, **extra)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def caminho_dado(arquivo: str) -> Path:
    """Resolve a fonte de dados.

    Arquivo versionado (ou presente) no checkout: raiz do repositório, de modo que
    a verificação rode também no CI. Só arquivo não versionado (privado) usa
    ``RAIZ_DADOS`` (``PHD_RESEARCH_DATA_ROOT``).
    """
    no_repo = ROOT / arquivo
    if no_repo.exists() or _rastreado(str(ROOT), arquivo):
        return no_repo
    return RAIZ_DADOS / arquivo


def ler_fonte(fonte: dict) -> float:
    """Lê o valor bruto de uma fonte de arquivo (CSV ou JSON)."""
    arquivo = fonte["arquivo"]
    caminho = caminho_dado(arquivo)
    if not caminho.exists():
        raise FonteIndisponivel(str(caminho))
    if arquivo.lower().endswith(".json"):
        no = _json(str(caminho))
        for parte in str(fonte["chave"]).split("."):
            try:
                no = no[int(parte)] if isinstance(no, list) else no[parte]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ErroRegistro(
                    f"{arquivo}: chave {fonte['chave']!r} não encontrada") from exc
        if isinstance(no, list) and fonte.get("agregar") == "contagem":
            return float(len(no))
        return _numero(no)
    filtro = fonte.get("filtro") or {}
    linhas = _linhas_csv(str(caminho))
    if linhas and any(c not in linhas[0] for c in [*filtro, fonte.get("coluna")]
                      if c is not None):
        raise ErroRegistro(f"{arquivo}: coluna ausente em {sorted(filtro)} / "
                           f"{fonte.get('coluna')!r}; colunas={list(linhas[0])}")
    sel = [r for r in linhas if all(_igual(r[k], v) for k, v in filtro.items())]
    agregar = fonte.get("agregar", "unico")
    if agregar == "contagem":
        return float(len(sel))
    if agregar == "unico" and len(sel) != 1:
        raise ErroRegistro(
            f"{arquivo}: filtro {filtro} casou {len(sel)} linhas (esperado 1)")
    if not sel:
        raise ErroRegistro(f"{arquivo}: filtro {filtro} sem linhas")
    valores = [_numero(r[fonte["coluna"]]) for r in sel]
    if agregar == "soma":
        return float(sum(valores))
    return valores[0]


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

class Registro:
    def __init__(self, entradas: list[dict]):
        self.entradas = entradas
        self.por_id: dict[str, dict] = {}
        self.valores: dict[str, tuple[dict, dict]] = {}
        for e in entradas:
            if e["id"] in self.por_id:
                raise ErroRegistro(f"id duplicado: {e['id']}")
            self.por_id[e["id"]] = e
            for v in e.get("valores") or []:
                if v["nome"] in self.valores:
                    raise ErroRegistro(f"nome de valor duplicado: {v['nome']}")
                self.valores[v["nome"]] = (e, v)
        self._bruto: dict[str, float] = {}

    def bruto(self, nome: str) -> float:
        if nome in self._bruto:
            return self._bruto[nome]
        if nome not in self.valores:
            raise ErroRegistro(f"valor desconhecido: {nome}")
        _, v = self.valores[nome]
        fonte = v.get("fonte") or {}
        if "derivado" in fonte:
            r = avaliar(fonte["derivado"], self.bruto)
        elif "arquivo" in fonte:
            r = ler_fonte(fonte)
        else:
            raise ErroRegistro(f"{nome}: fonte sem 'arquivo' nem 'derivado'")
        self._bruto[nome] = r
        return r

    def arquivos_ausentes(self, nome: str, _visto=None) -> list[str]:
        """Arquivos de dados necessários ao valor que faltam neste checkout."""
        _visto = _visto if _visto is not None else set()
        if nome in _visto:
            return []
        _visto.add(nome)
        fonte = self.valores[nome][1].get("fonte") or {}
        if "derivado" in fonte:
            ausentes: list[str] = []
            for no in ast.walk(ast.parse(str(fonte["derivado"]), mode="eval")):
                if isinstance(no, ast.Name) and no.id in self.valores:
                    ausentes += self.arquivos_ausentes(no.id, _visto)
            return ausentes
        caminho = caminho_dado(fonte["arquivo"])
        return [] if caminho.exists() else [str(caminho)]

    def formatado(self, nome: str) -> tuple[str, str]:
        _, v = self.valores[nome]
        return formatar_valor(self.bruto(nome), v.get("transformacao"),
                              (v.get("fonte") or {}).get("formato", v.get("formato")))

    def texto(self, nome: str, idioma: str) -> str:
        _, v = self.valores[nome]
        if v.get("texto_pt") is None:
            raise ErroRegistro(f"{nome}: valor auxiliar sem texto_pt não pode ser token")
        if idioma == "en":
            if v.get("texto_en") is not None:
                return str(v["texto_en"])
            return pt_para_en(str(v["texto_pt"]))
        return str(v["texto_pt"])


def carregar_registro(caminho: Path = REGISTRO) -> Registro:
    dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))
    return Registro(dados)


# --------------------------------------------------------------------------
# Verificação de valores (registro x artefato)
# --------------------------------------------------------------------------

def verificar_valor(reg: Registro, nome: str) -> list[str]:
    """Lista de problemas (vazia = ok). Levanta FonteIndisponivel se faltar dado."""
    ausentes = reg.arquivos_ausentes(nome)
    if ausentes:
        raise FonteIndisponivel(", ".join(ausentes))
    _, v = reg.valores[nome]
    pt, en = reg.formatado(nome)
    problemas = []
    if v.get("texto_pt") is not None and str(v["texto_pt"]) != pt:
        problemas.append(
            f"texto do registro diverge do artefato: texto_pt={v['texto_pt']!r}, "
            f"artefato={pt!r} (bruto={reg.bruto(nome)!r})")
    if v.get("texto_en") is not None and str(v["texto_en"]) != en:
        problemas.append(
            f"texto do registro diverge do artefato: texto_en={v['texto_en']!r}, "
            f"artefato={en!r} (bruto={reg.bruto(nome)!r})")
    return problemas


# --------------------------------------------------------------------------
# Verificação de ocorrências (registro x texto)
# --------------------------------------------------------------------------

_MARCADORES = {
    "resumo": (r"\{\\LARGE\\bf Resumo\}", r"\\textbf\{Palavras-chave", "resumo"),
    "abstract": (r"\{\\LARGE\\bf Abstract\}", r"\\textbf\{Keywords", "abstract"),
}
_ABNTEX = {"resumo": ("\\begin{resumo}", "\\end{resumo}"),
           "abstract": ("\\begin{abstract}", "\\end{abstract}")}


def _sem_comentarios(texto: str) -> str:
    # Linha só de comentário some inteira (com a quebra de linha): deixá-la em
    # branco criaria uma separação de parágrafo que o texto compilado não tem.
    texto = re.sub(r"^[ \t]*%.*(?:\n|\Z)", "", texto, flags=re.MULTILINE)
    return re.sub(r"(?<!\\)%.*", "", texto)


# Separadores de parágrafo: linha em branco, \item e \par (comando isolado).
# Quadro Beamer completo (grupo de captura único: re.split devolve os quadros nos
# índices ímpares).
_QUADRO = re.compile(r"(\\begin\{frame\}.*?\\end\{frame\})", re.DOTALL)
_SEPARADOR_PARAGRAFO = re.compile(r"\n\s*\n|\\item(?![A-Za-z])|\\par(?![A-Za-z])")


def _bloco(texto: str, secao: str) -> str:
    ini, fim, _ = _MARCADORES[secao]
    m = re.search(ini, texto)
    if m:
        f = re.search(fim, texto[m.end():])
        if f:
            fim_pos = m.end() + f.end()
            # inclui a linha de palavras-chave
            nl = texto.find("\n", fim_pos)
            return texto[m.start(): nl if nl != -1 else len(texto)]
        raise ErroRegistro(f"fim do bloco {secao} não encontrado")
    abre, fecha = _ABNTEX[secao]
    i = texto.find(abre)
    j = texto.find(fecha)
    if i != -1 and j > i:
        return texto[i:j]
    raise ErroRegistro(f"bloco {secao} não encontrado")


def normalizar(s: str) -> str:
    """Normalização LaTeX mínima para comparar tokens e texto."""
    s = s.replace("{,}", ",").replace("\\%", "%").replace("\\,", " ")
    s = s.replace("~", " ").replace("\\_", "_").replace("\\&", "&")
    s = s.replace("\u00a0", " ").replace("\u2212", "-")
    s = re.sub(r"\\(?:textit|textbf|emph|texttt|textsc|mbox|text|mathrm|num)\*?\{",
                "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s*=\s*", "=", s)
    return re.sub(r"\s+", " ", s).strip()


def _dobrar(s: str) -> str:
    s = unicodedata.normalize("NFD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


@functools.lru_cache(maxsize=None)
def paragrafos(arquivo: str, secao: str | None) -> tuple[str, ...]:
    caminho = ROOT / arquivo
    if not caminho.exists():
        raise FonteIndisponivel(str(caminho))
    texto = caminho.read_text(encoding="utf-8")
    if arquivo.endswith(".tex"):
        texto = _sem_comentarios(texto)
    if secao:
        if secao not in _MARCADORES:
            raise ErroRegistro(f"seção desconhecida: {secao!r}")
        texto = _bloco(texto, secao)
    if not secao and arquivo.endswith(".tex") and r"\begin{frame}" in texto:
        # Slides Beamer: a unidade de coocorrência é o quadro inteiro (os itens de
        # um quadro compõem uma mesma tela); o que fica fora dos quadros segue
        # a separação por parágrafo.
        partes = []
        for i, trecho in enumerate(_QUADRO.split(texto)):
            partes += [trecho] if i % 2 else _SEPARADOR_PARAGRAFO.split(trecho)
    elif arquivo.endswith(".tex"):
        partes = _SEPARADOR_PARAGRAFO.split(texto)
    else:
        partes = re.split(r"\n\s*\n", texto)  # .md: só linha em branco
    return tuple(p for p in (normalizar(x) for x in partes) if p)


def contem(paragrafo: str, token: str, dobrar: bool = False) -> bool:
    if dobrar:
        paragrafo, token = _dobrar(paragrafo), _dobrar(token)
    pad = re.escape(token)
    if token[:1].isdigit():
        pad = r"(?<!\d)(?<!\d[.,])" + pad
    if token[-1:].isdigit():
        pad = pad + r"(?!\d)(?![.,]\d)"
    return re.search(pad, paragrafo) is not None


def _expandir(reg: Registro, token, idioma: str, tipo: str):
    """Token -> lista de alternativas já normalizadas."""
    alternativas = token if isinstance(token, list) else [token]
    saida = []
    for t in alternativas:
        t = str(t)
        if t in reg.valores:
            t = reg.texto(t, idioma)
        saida.append(normalizar(t))
    return saida


def verificar_ocorrencia(reg: Registro, entrada: dict, oc: dict) -> list[str]:
    arquivo, secao = oc["arquivo"], oc.get("secao")
    paras = paragrafos(arquivo, secao)
    idioma = "en" if (secao == "abstract" or oc.get("idioma") == "en") else "pt"
    dobrar = entrada.get("tipo") == "afirmacao" or bool(oc.get("ignorar_caixa"))
    todos = [*(oc.get("exige") or []), *(oc.get("exige_todos") or [])]
    todos = [_expandir(reg, t, idioma, entrada.get("tipo", "numero")) for t in todos]
    algum = [normalizar(str(t)) for t in (oc.get("exige_algum") or [])]
    if not todos and not algum:
        raise ErroRegistro(f"{entrada['id']}: ocorrência sem exigências")

    melhor, melhor_pt, melhor_falta = None, -1, []
    for p in paras:
        falta = [alts[0] if len(alts) == 1 else " | ".join(alts)
                 for alts in todos if not any(contem(p, a, dobrar) for a in alts)]
        pontos = len(todos) - len(falta)
        falta_algum = bool(algum) and not any(contem(p, a, dobrar) for a in algum)
        if not falta and not falta_algum:
            return []
        pontos += 0 if falta_algum else 1
        if pontos > melhor_pt:
            melhor, melhor_pt = p, pontos
            melhor_falta = falta + ([f"algum de {algum}"] if falta_algum else [])
    onde = f"{arquivo}" + (f" [{secao}]" if secao else "")
    trecho = (melhor or "")[:240]
    return [f"{onde}: faltam {melhor_falta}; melhor parágrafo: «{trecho}»"]


# --------------------------------------------------------------------------
# Casos (para testes e relatório)
# --------------------------------------------------------------------------

@dataclass
class Caso:
    id: str
    entrada: str
    tipo: str          # "valor" | "ocorrencia"
    alvo: str
    xfail: str | None = None
    ocorrencia: dict | None = None


def _rotulo_ocorrencia(entrada: str, i: int, oc: dict) -> str:
    base = Path(oc["arquivo"]).name
    sec = f":{oc['secao']}" if oc.get("secao") else ""
    return f"{entrada}:{base}{sec}#{i}"


def casos_valores(reg: Registro) -> list[Caso]:
    return [Caso(f"{e['id']}:{v['nome']}", e["id"], "valor", v["nome"])
            for e in reg.entradas for v in e.get("valores") or []]


def casos_ocorrencias(reg: Registro) -> list[Caso]:
    saida = []
    for e in reg.entradas:
        for i, oc in enumerate(e.get("ocorrencias") or []):
            saida.append(Caso(_rotulo_ocorrencia(e["id"], i, oc), e["id"],
                              "ocorrencia", oc["arquivo"], oc.get("xfail"), oc))
    return saida


def executar_caso(reg: Registro, caso: Caso) -> list[str]:
    if caso.tipo == "valor":
        return verificar_valor(reg, caso.alvo)
    return verificar_ocorrencia(reg, reg.por_id[caso.entrada], caso.ocorrencia)


def verificar_estrutura(reg: Registro) -> list[str]:
    """Consistência interna do registro (referências, tipos, campos)."""
    problemas = []
    for e in reg.entradas:
        eid = e["id"]
        if e.get("tipo") not in {"numero", "afirmacao"}:
            problemas.append(f"{eid}: tipo inválido {e.get('tipo')!r}")
        if e.get("tipo") == "numero" and not e.get("valores"):
            problemas.append(f"{eid}: número sem 'valores'")
        if not e.get("ocorrencias"):
            problemas.append(f"{eid}: sem 'ocorrencias'")
        for v in e.get("valores") or []:
            fonte = v.get("fonte") or {}
            if not ("arquivo" in fonte or "derivado" in fonte):
                problemas.append(f"{eid}.{v['nome']}: fonte sem arquivo/derivado")
            if v.get("texto_pt") is None and not v.get("auxiliar"):
                problemas.append(f"{eid}.{v['nome']}: sem texto_pt")
        for oc in e.get("ocorrencias") or []:
            for chave in ("exige", "exige_todos", "exige_algum"):
                for t in oc.get(chave) or []:
                    for a in (t if isinstance(t, list) else [t]):
                        if a in reg.valores and reg.valores[a][1].get("texto_pt") is None:
                            problemas.append(f"{eid}: token {a} sem texto_pt")
            if not any(oc.get(k) for k in ("exige", "exige_todos", "exige_algum")):
                problemas.append(f"{eid}: ocorrência {oc['arquivo']} sem exigências")
            if oc.get("secao") and oc["secao"] not in _MARCADORES:
                problemas.append(f"{eid}: seção inválida {oc['secao']!r}")
    return problemas


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def executar_tudo(reg: Registro) -> list[tuple[Caso, str, str]]:
    """Executa todos os casos; devolve ``(caso, status, detalhe)``.

    status: ok | falha | xfail | xpass | pulado
    """
    resultados = []
    for caso in [*casos_valores(reg), *casos_ocorrencias(reg)]:
        try:
            problemas = executar_caso(reg, caso)
        except FonteIndisponivel as exc:
            resultados.append((caso, "pulado", MENSAGEM_INDISPONIVEL + str(exc)))
            continue
        except ErroRegistro as exc:
            problemas = [f"erro de registro: {exc}"]
        if caso.xfail:
            if problemas:
                resultados.append((caso, "xfail", f"{caso.xfail} | {problemas[0]}"))
            else:
                resultados.append((caso, "xpass",
                                   f"passou, mas estava marcado xfail: {caso.xfail}"))
        elif problemas:
            resultados.append((caso, "falha", "; ".join(problemas)))
        else:
            resultados.append((caso, "ok", ""))
    return resultados


def relatorio_markdown(reg: Registro, resultados) -> str:
    contagem: dict[str, int] = {}
    for _, st, _ in resultados:
        contagem[st] = contagem.get(st, 0) + 1
    linhas = ["# Relatório do registro de invariantes", "",
              "Resumo: " + ", ".join(f"{k}={v}" for k, v in sorted(contagem.items())),
              "", f"Entradas: {len(reg.entradas)}; valores: {len(reg.valores)}.", ""]
    for titulo, status in (("Falhas", "falha"), ("XPASS (remover xfail)", "xpass"),
                           ("Inconsistências documentadas (xfail)", "xfail"),
                           ("Pulados (dado ausente)", "pulado")):
        sel = [(c, d) for c, st, d in resultados if st == status]
        if not sel:
            continue
        linhas += [f"## {titulo}", ""]
        for c, d in sel:
            linhas.append(f"- `{c.id}`: {d}")
        linhas.append("")
    linhas += ["## Todos os casos", "", "| caso | status |", "|---|---|"]
    linhas += [f"| `{c.id}` | {st} |" for c, st, _ in resultados]
    return "\n".join(linhas) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registro", default=str(REGISTRO))
    ap.add_argument("--relatorio", help="caminho do relatório em markdown")
    args = ap.parse_args(argv)
    reg = carregar_registro(Path(args.registro))
    estrutura = verificar_estrutura(reg)
    resultados = executar_tudo(reg)
    texto = relatorio_markdown(reg, resultados)
    if estrutura:
        texto += "\n## Estrutura\n\n" + "\n".join(f"- {p}" for p in estrutura) + "\n"
    if args.relatorio:
        destino = Path(args.relatorio)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")
    ruins = [r for r in resultados if r[1] in {"falha", "xpass"}]
    contagem: dict[str, int] = {}
    for _, st, _ in resultados:
        contagem[st] = contagem.get(st, 0) + 1
    print(", ".join(f"{k}={v}" for k, v in sorted(contagem.items())))
    for c, st, d in ruins:
        print(f"[{st}] {c.id}: {d}")
    for p in estrutura:
        print(f"[estrutura] {p}")
    return 1 if (ruins or estrutura) else 0


if __name__ == "__main__":
    sys.exit(main())
