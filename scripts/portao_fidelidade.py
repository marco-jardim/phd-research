#!/usr/bin/env python
"""Portão de fidelidade para revisões só de texto da tese.

Compara cada parágrafo editado com a versão da base (por padrão a tag
``pre-revisao-2``) e sinaliza mudanças em tokens que carregam sentido:
números, negadores, quantificadores e marcadores de força da afirmação,
chaves de citação, alvos de ``\\ref``/``\\label``, siglas e alvos de ``\\path``.
Um revisor independente só precisa olhar os parágrafos sinalizados.

Há duas verificações, escolhidas por ``--modo`` (padrão ``auto``):

* ``paragrafo``: o alinhamento parágrafo a parágrafo descrito acima (parágrafos
  modificados, adicionados e removidos).
* ``capitulo``: compara, por arquivo ``.tex`` alterado, o multiconjunto de números
  da base com o do alvo (mesma extração de números dos parágrafos), sem alinhar
  parágrafos. Serve a revisões que fundem, movem e cortam parágrafos: remoções e
  fusões de parágrafo não geram pendência por parágrafo, só valem estas regras:

  - número novo (ausente na base) ou com contagem maior que na base: pendência,
    salvo se o valor for texto (``texto_pt``) de uma entrada N## do registro de
    invariantes (``tests/data/registro_invariantes.yaml``) que liste o arquivo nas
    ocorrências, ou se houver aceite por arquivo na seção ``aceites_numeros`` de
    ``docs/fidelidade_aceites.yaml`` (``{arquivo, numero, autora_aprovou: true,
    motivo}``, com o número entre aspas);
  - número removido (contagem menor): só registrado no relatório, salvo se for valor
    de invariante exigido (``exige``) pelo registro nesse arquivo e deixar de
    aparecer nele; nesse caso é pendência (o mesmo aceite por arquivo a libera);
  - número movido (mesma contagem no arquivo): não gera nada;
  - números de rótulo ("passo 11", "Tabela 3", "seção 2.1", capítulo, figura, equação,
    estudo, objetivo, meta...) não entram na contagem: mesma regra de isenção ``rotulo``
    de ``scripts/cobertura_numeros.py`` (importada, não duplicada);
  - um número conta como coberto por invariante quando é o ``texto_pt`` de qualquer valor
    exigido por alguma ocorrência do registro que cite o arquivo, onde quer que o valor
    esteja definido (valor definido em N01 e só referenciado por nome em N09 cobre N09).

* ``ambos``: as duas verificações.
* ``auto`` (padrão): decide por arquivo. Arquivo listado em
  ``tests/data/arquivos_revisados.yaml`` (revisado pelo regime aprovado pela autora) passa
  pela comparação ``capitulo``, porque nele o sentido é protegido pelas âncoras do registro
  de invariantes (``vigencia: pos_revisao``) e pela revisão de fidelidade independente, e a
  comparação por parágrafo só gera ruído de fusão, movimento e remoção. Arquivo não listado
  segue em ``ambos``. Só vale o arquivo inteiro; chaves ``arquivo#secao`` (resumo e
  abstract de ``tese.tex``) não mudam o modo, pois o portão não trata seções.

Uso:
    python scripts/portao_fidelidade.py [--base pre-revisao-2]
        [--alvo WORKTREE|<ref>] [--arquivos text/latex/x.tex ...]
        [--modo auto|capitulo|paragrafo|ambos] [--registro caminho.yaml]
        [--saida .local/fidelidade/relatorio.md] [--json saida.json]

Código de saída: 0 se não há alertas pendentes; 1 se há; 2 em erro de uso
ou de git. Somente biblioteca padrão (PyYAML é usado se estiver instalado; o
registro de invariantes exige PyYAML).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_PADRAO = "pre-revisao-2"
ALVO_WORKTREE = "WORKTREE"
LIMIAR_JACCARD = 0.35
MAX_TRECHO = 300
ACEITES_PADRAO = "docs/fidelidade_aceites.yaml"
SAIDA_PADRAO = ".local/fidelidade/relatorio.md"
REGISTRO_PADRAO = "tests/data/registro_invariantes.yaml"
SECAO_ACEITES_NUMEROS = "aceites_numeros"
MODOS = ("auto", "capitulo", "paragrafo", "ambos")
MODO_PADRAO = "auto"
REVISADOS_PADRAO = "tests/data/arquivos_revisados.yaml"
ARQUIVO_EXTRA = "text/slides/defesa.tex"
EXCLUIDOS = {"tese_rev_diff.tex"}
MARCA_GERADA = "Gerada por scripts/"

NEGADORES = {"não", "nem", "sem", "nunca", "nenhum", "nenhuma", "jamais", "tampouco"}
# Formas flexionadas são reduzidas a um lema para não gerar alerta por concordância.
QUANTIFICADORES = {
    "todos": "todos",
    "todas": "todos",
    "sempre": "sempre",
    "apenas": "apenas",
    "somente": "somente",
    "exclusivamente": "exclusivamente",
    "pode": "pode",
    "podem": "pode",
    "sugere": "sugere",
    "sugerem": "sugere",
    "indica": "indica",
    "indicam": "indica",
    "demonstra": "demonstra",
    "demonstram": "demonstra",
    "comprova": "comprova",
    "comprovam": "comprova",
}
# Siglas escritas em caixa mista (além das de 2 ou mais maiúsculas).
SIGLAS_MISTAS = {"sinan", "sinasc", "datasus"}
# Qualificadores de quantidade com mais de uma palavra ou com sentido de limite.
QUALIFICADORES = (
    "no máximo",
    "pelo menos",
    "mais de",
    "menos de",
    "cerca de",
    "aproximadamente",
    "até",
    "ao menos",
)
# Comparativos; o plural é reduzido ao lema.
COMPARATIVOS = {
    "maior": "maior",
    "maiores": "maior",
    "menor": "menor",
    "menores": "menor",
    "acima": "acima",
    "abaixo": "abaixo",
    "superior": "superior",
    "superiores": "superior",
    "inferior": "inferior",
    "inferiores": "inferior",
}
# Modos de operação e métricas, cuja troca altera a afirmação.
METRICAS = {
    "vigilância": "vigilância",
    "confirmação": "confirmação",
    "sensibilidade": "sensibilidade",
    "vpp": "VPP",
    "especificidade": "especificidade",
    "f1": "F1",
    "vpn": "VPN",
}

CATEGORIAS = (
    "numeros",
    "ordem_numeros",
    "ordem_paragrafo",
    "negadores",
    "quantificadores",
    "qualificadores",
    "comparativos",
    "metricas",
    "citacoes",
    "referencias",
    "siglas",
    "caminhos",
)
# Categorias que tornam um parágrafo removido/adicionado um alerta.
CATEGORIAS_REMOVIDO = ("numeros", "citacoes", "negadores")
CATEGORIAS_ADICIONADO = ("numeros", "citacoes", "negadores")

RE_CITE = re.compile(r"\\[A-Za-z]*cite[A-Za-z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]*)\}")
RE_REF = re.compile(r"\\(ref|eqref|autoref|pageref|cref|Cref|vref|nameref|label)\*?\s*\{([^{}]*)\}")
RE_PATH = re.compile(r"\\(?:path|url)\s*\{([^{}]*)\}")
RE_COMANDO = re.compile(r"\\[A-Za-z@]+\*?")
RE_PALAVRA = re.compile(r"[^\W_]+")
# Sinal negativo: "$-$", "−" (U+2212) ou hífen que não segue letra, dígito, "-" ou "\".
RE_NUMERO = re.compile(
    r"(?:(?P<sinal>\$\s*[-\u2212]\s*\$|\u2212|(?<![\w\\-])-)(?=\d))?"
    r"(?<!\d)(?P<num>\d+(?:[.,]\d+)*)(?P<pct>(?:\s|~|\\,|\\;)*\\?%)?"
)
RE_SEP_MILHAR = re.compile("(?<=\\d)(?:\\\\,|\u00a0|\u202f|\u2009)(?=\\d{3}(?!\\d))")
RE_FRASE = re.compile(r"(?<=[.!?;])\s+")
RE_QUALIFICADOR = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        r"[\s~]+".join(re.escape(w) for w in q.split())
        for q in sorted(QUALIFICADORES, key=len, reverse=True)
    )
    + r")(?!\w)"
)
SEP_ORDEM = "|"


# ---------------------------------------------------------------------------
# Segmentação e normalização
# ---------------------------------------------------------------------------


def normalizar(par: str) -> str:
    return unicodedata.normalize("NFC", " ".join(par.split()))


def hash_paragrafo(par: str) -> str:
    """sha256 do parágrafo normalizado, 16 primeiros hexadecimais."""
    return hashlib.sha256(normalizar(par).encode("utf-8")).hexdigest()[:16]


def segmentar(texto: str) -> list[str]:
    """Parágrafos separados por linha em branco, sem comentários, com LaTeX preservado."""
    linhas = []
    for linha in texto.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if linha.lstrip().startswith("%"):
            continue
        linhas.append(re.sub(r"(?<!\\)%.*$", "", linha))
    blocos = re.split(r"\n\s*\n", "\n".join(linhas))
    pars = [normalizar(b) for b in blocos]
    return [p for p in pars if p]


def normalizar_numero(bruto: str) -> str:
    """Forma canônica de um número pt-BR ou en: 39,3 = 39.3; 84.368 = 84368."""
    seps = re.findall(r"[.,]", bruto)
    if not seps:
        return bruto
    if "." in seps and "," in seps:
        dec = bruto[max(bruto.rfind("."), bruto.rfind(","))]
        mil = "," if dec == "." else "."
        inteiro, frac = bruto.rsplit(dec, 1)
        return f"{inteiro.replace(mil, '')}.{frac}"
    sep = seps[0]
    partes = bruto.split(sep)
    if len(partes) > 2:
        if all(len(p) == 3 for p in partes[1:]) and 1 <= len(partes[0]) <= 3:
            return "".join(partes)
        return bruto  # lista ou formato ambíguo: compara como veio
    ini, fim = partes
    if sep == "." and len(fim) == 3 and 1 <= len(ini) <= 3 and not ini.startswith("0"):
        return ini + fim
    return f"{ini}.{fim}"


# ---------------------------------------------------------------------------
# Extração de tokens
# ---------------------------------------------------------------------------


_RE_ROTULO_CACHE: list = []


def _re_rotulo() -> re.Pattern:
    """Regra de rótulo (passo, capítulo, seção, figura, tabela...) de ``cobertura_numeros``.

    Importada sob demanda: ``cobertura_numeros`` importa este módulo.
    """
    if not _RE_ROTULO_CACHE:
        ultimo: ImportError | None = None
        for nome in ("scripts.cobertura_numeros", "cobertura_numeros"):
            try:
                _RE_ROTULO_CACHE.append(importlib.import_module(nome)._RE_ROTULO)
                break
            except ImportError as exc:
                ultimo = exc
        else:
            raise ErroInvariantes(f"regra de rótulos indisponível (cobertura_numeros): {ultimo}")
    return _RE_ROTULO_CACHE[0]


def _e_rotulo(texto: str, inicio: int) -> bool:
    """O número que começa em ``texto[inicio]`` é rótulo ("passo 11", "Tabela 3", "seção 2.1")?"""
    antes = re.sub(r"\s+", " ", texto[max(0, inicio - 120) : inicio].replace("~", " "))
    return bool(_re_rotulo().search(antes))


def extrair_tokens(par: str, isentar_rotulos: bool = False) -> dict[str, Counter]:
    """Multiconjuntos de itens com sentido, por categoria.

    Com ``isentar_rotulos``, os números de rótulo (passo, capítulo, seção, figura, tabela,
    equação, estudo, objetivo, meta...) não entram em ``numeros`` nem na ordem dos números.
    """
    tok = {cat: Counter() for cat in CATEGORIAS}
    trabalho = par

    for m in RE_CITE.finditer(trabalho):
        for chave in m.group(1).split(","):
            if chave.strip():
                tok["citacoes"][chave.strip()] += 1
    trabalho = RE_CITE.sub(" ", trabalho)

    for m in RE_REF.finditer(trabalho):
        tok["referencias"][f"{m.group(1)}:{m.group(2).strip()}"] += 1
    trabalho = RE_REF.sub(" ", trabalho)

    for m in RE_PATH.finditer(trabalho):
        tok["caminhos"][m.group(1).strip()] += 1
    trabalho = RE_PATH.sub(" ", trabalho)

    numerico = RE_SEP_MILHAR.sub("", trabalho.replace("{,}", ","))
    sequencia_paragrafo = []
    posicao = 0
    for frase in RE_FRASE.split(numerico):
        inicio_frase = numerico.find(frase, posicao)
        posicao = inicio_frase + len(frase)
        sequencia = []
        for m in RE_NUMERO.finditer(frase):
            if isentar_rotulos and _e_rotulo(numerico, inicio_frase + m.start("num")):
                continue
            item = normalizar_numero(m.group("num"))
            if m.group("sinal"):
                item = "-" + item
            if m.group("pct"):
                item += "%"
            tok["numeros"][item] += 1
            sequencia.append(item)
        sequencia_paragrafo.extend(sequencia)
        if len(sequencia) >= 2:
            tok["ordem_numeros"][SEP_ORDEM.join(sequencia)] += 1
    if len(sequencia_paragrafo) >= 2:
        tok["ordem_paragrafo"][SEP_ORDEM.join(sequencia_paragrafo)] += 1

    sem_comandos = RE_COMANDO.sub(" ", trabalho)
    minusculo = unicodedata.normalize("NFC", sem_comandos).lower()
    for m in RE_QUALIFICADOR.finditer(minusculo):
        tok["qualificadores"][" ".join(m.group(0).replace("~", " ").split())] += 1
    for palavra in RE_PALAVRA.findall(sem_comandos):
        baixa = unicodedata.normalize("NFC", palavra).lower()
        if baixa in NEGADORES:
            tok["negadores"][baixa] += 1
        if baixa in QUANTIFICADORES:
            tok["quantificadores"][QUANTIFICADORES[baixa]] += 1
        if baixa in COMPARATIVOS:
            tok["comparativos"][COMPARATIVOS[baixa]] += 1
        if baixa in METRICAS:
            tok["metricas"][METRICAS[baixa]] += 1
        if baixa in SIGLAS_MISTAS:
            tok["siglas"][palavra.upper()] += 1
        elif sum(1 for c in palavra if c.isupper()) >= 2:
            tok["siglas"][palavra] += 1
    return tok


def _pares_permutados(rem: list[str], add: list[str]) -> list[tuple[str, str]]:
    """Sequências removidas e adicionadas com os mesmos números em outra ordem."""
    livres = list(add)
    pares = []
    for r in rem:
        chave = sorted(r.split(SEP_ORDEM))
        for k, a in enumerate(livres):
            if sorted(a.split(SEP_ORDEM)) == chave:
                pares.append((r, livres.pop(k)))
                break
    return pares


def comparar(tok_a: dict[str, Counter], tok_b: dict[str, Counter]) -> dict[str, dict[str, list[str]]]:
    dif: dict[str, dict[str, list[str]]] = {}
    for cat in CATEGORIAS:
        rem = tok_a[cat] - tok_b[cat]
        add = tok_b[cat] - tok_a[cat]
        if cat in ("ordem_numeros", "ordem_paragrafo"):
            # Só interessa a troca de ordem: mesmos números, outra sequência.
            # Números diferentes já aparecem na categoria "numeros".
            # A ordem do parágrafo pega trocas entre orações (a ordem por frase não
            # as vê) e fica de fora quando a ordem dentro da frase já alertou.
            if cat == "ordem_paragrafo" and "ordem_numeros" in dif:
                continue
            pares = _pares_permutados(sorted(rem.elements()), sorted(add.elements()))
            if pares:
                dif[cat] = {
                    "removidos": sorted(a for a, _ in pares),
                    "adicionados": sorted(b for _, b in pares),
                }
            continue
        if rem or add:
            dif[cat] = {
                "removidos": sorted(rem.elements()),
                "adicionados": sorted(add.elements()),
            }
    return dif


def ids_itens(dif: dict[str, dict[str, list[str]]]) -> list[str]:
    ids = set()
    for cat, d in dif.items():
        ids.update(f"{cat}:-{x}" for x in d["removidos"])
        ids.update(f"{cat}:+{x}" for x in d["adicionados"])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Alinhamento
# ---------------------------------------------------------------------------


def _conjunto(par: str) -> frozenset[str]:
    return frozenset(p.lower() for p in RE_PALAVRA.findall(par))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class Alinhamento:
    pares: list[tuple[int, int]]  # modificados: (índice base, índice alvo)
    removidos: list[int]
    adicionados: list[int]
    iguais: int = 0
    movidos: int = 0


def alinhar(base: list[str], alvo: list[str], limiar: float = LIMIAR_JACCARD) -> Alinhamento:
    usados_a: set[int] = set()
    usados_b: set[int] = set()
    iguais = 0
    sm = difflib.SequenceMatcher(None, base, alvo, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                usados_a.add(i1 + k)
                usados_b.add(j1 + k)
            iguais += i2 - i1

    # Parágrafos idênticos que mudaram de lugar.
    livres_b: dict[str, list[int]] = {}
    for j, txt in enumerate(alvo):
        if j not in usados_b:
            livres_b.setdefault(txt, []).append(j)
    movidos = 0
    for i, txt in enumerate(base):
        if i in usados_a:
            continue
        cand = livres_b.get(txt)
        if cand:
            j = cand.pop(0)
            usados_a.add(i)
            usados_b.add(j)
            movidos += 1

    # Pares modificados por similaridade, do mais parecido ao menos.
    conj_a = {i: _conjunto(base[i]) for i in range(len(base)) if i not in usados_a}
    conj_b = {j: _conjunto(alvo[j]) for j in range(len(alvo)) if j not in usados_b}
    candidatos = []
    for i, ca in conj_a.items():
        for j, cb in conj_b.items():
            s = jaccard(ca, cb)
            if s >= limiar:
                candidatos.append((-s, abs(i - j), i, j))
    candidatos.sort()
    pares = []
    for _, _, i, j in candidatos:
        if i in usados_a or j in usados_b:
            continue
        usados_a.add(i)
        usados_b.add(j)
        pares.append((i, j))
    pares.sort(key=lambda p: p[1])
    return Alinhamento(
        pares=pares,
        removidos=[i for i in range(len(base)) if i not in usados_a],
        adicionados=[j for j in range(len(alvo)) if j not in usados_b],
        iguais=iguais,
        movidos=movidos,
    )


# ---------------------------------------------------------------------------
# Aceites
# ---------------------------------------------------------------------------


def _escalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
        return s[1:-1]
    s = re.sub(r"\s+#.*$", "", s).strip()
    baixa = s.lower()
    if baixa in ("true", "yes"):
        return True
    if baixa in ("false", "no"):
        return False
    if baixa in ("", "~", "null"):
        return None
    return s


def _lista_inline(s: str) -> list:
    interior = s.strip()[1:-1]
    return [_escalar(x) for x in interior.split(",") if x.strip()]


RE_SECAO = re.compile(r"^([A-Za-z_]\w*)\s*:")


def _secoes_toplevel(texto: str) -> dict[str, list[str]]:
    """Linhas de cada seção de nível zero (chave sem indentação), cabeçalho incluso."""
    secoes: dict[str, list[str]] = {}
    atual: str | None = None
    for linha in texto.splitlines():
        m = RE_SECAO.match(linha)
        if m:
            atual = m.group(1)
            secoes[atual] = [linha]
        elif atual is not None:
            secoes[atual].append(linha)
    return secoes


def parse_aceites_simples(texto: str, secao: str = "aceites") -> list[dict]:
    """Analisador mínimo para o formato plano de ``fidelidade_aceites.yaml``.

    Se o arquivo tem mais de uma seção de nível zero (``aceites`` e
    ``aceites_numeros``), analisa só a seção pedida.
    """
    chaves = {m.group(1) for ln in texto.splitlines() if (m := RE_SECAO.match(ln))}
    if chaves and chaves != {"aceites"}:
        linhas = _secoes_toplevel(texto).get(secao, [])
        texto = "\n".join(["aceites:", *linhas[1:]])
    entradas: list[dict] = []
    atual: dict | None = None
    chave_lista: str | None = None
    indent_entrada: int | None = None
    for bruta in texto.splitlines():
        linha = bruta.rstrip()
        corpo = linha.strip()
        if not corpo or corpo.startswith("#"):
            continue
        indent = len(linha) - len(linha.lstrip())
        if re.match(r"^aceites\s*:\s*(\[\s*\])?\s*(#.*)?$", corpo):
            continue
        if corpo.startswith("- "):
            if indent_entrada is None:
                indent_entrada = indent
            if indent == indent_entrada:
                atual = {}
                entradas.append(atual)
                chave_lista = None
                corpo = corpo[2:].strip()
            elif atual is not None and chave_lista is not None:
                atual[chave_lista].append(_escalar(corpo[2:]))
                continue
            else:
                continue
        if atual is None:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*:\s*(.*)$", corpo)
        if not m:
            continue
        chave, valor = m.group(1), m.group(2).strip()
        if valor == "" or valor.startswith("#"):
            atual[chave] = []
            chave_lista = chave
        elif valor.startswith("["):
            atual[chave] = _lista_inline(re.sub(r"\s+#.*$", "", valor))
            chave_lista = None
        else:
            atual[chave] = _escalar(valor)
            chave_lista = None
    return entradas


class ErroAceites(ValueError):
    """Arquivo de aceites malformado (por exemplo, entrada sem itens)."""


RE_HASH_ACEITE = re.compile(r"[0-9a-f]{16}")


def ler_entradas_aceites(caminho: Path) -> list:
    """Entradas do YAML de aceites como vieram, sem normalização."""
    if not caminho.exists():
        return []
    texto = caminho.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        dados = yaml.safe_load(texto)
        if isinstance(dados, dict):
            dados = dados.get("aceites")
        return list(dados or [])
    except ImportError:
        return parse_aceites_simples(texto)


def validar_aceite(e: object) -> list[str]:
    """Erros de esquema de uma entrada bruta de aceite (lista vazia se válida)."""
    if not isinstance(e, dict):
        return ["a entrada não é um mapeamento"]
    erros = []
    if not isinstance(e.get("arquivo"), str) or not e["arquivo"].strip():
        erros.append("arquivo ausente ou vazio")
    h = e.get("hash_paragrafo_novo")
    if not isinstance(h, str) or not RE_HASH_ACEITE.fullmatch(h):
        erros.append("hash_paragrafo_novo deve ser texto com 16 hexadecimais minúsculos")
    itens = e.get("itens")
    if (
        not isinstance(itens, list)
        or not itens
        or not all(isinstance(x, str) and x.strip() for x in itens)
    ):
        erros.append("itens deve ser uma lista não vazia de textos")
    j = e.get("justificativa")
    if not isinstance(j, str) or not j.strip():
        erros.append("justificativa ausente ou vazia")
    if not isinstance(e.get("autora_aprovou"), bool):
        erros.append("autora_aprovou deve ser booleano")
    return erros


def carregar_aceites(caminho: Path) -> list[dict]:
    entradas = ler_entradas_aceites(caminho)
    saida = []
    for e in entradas:
        if not isinstance(e, dict):
            continue
        itens = e.get("itens") or []
        if isinstance(itens, str):
            itens = [itens]
        if not itens:
            raise ErroAceites(
                f"aceite sem itens: arquivo={e.get('arquivo')!s} "
                f"hash_paragrafo_novo={e.get('hash_paragrafo_novo')!s}; "
                "liste em `itens` cada alerta aceito (itens vazios não cobrem tudo)"
            )
        saida.append(
            {
                "arquivo": str(e.get("arquivo", "")).replace("\\", "/"),
                "hash_paragrafo_novo": str(e.get("hash_paragrafo_novo", "")).strip(),
                "itens": [str(x) for x in itens],
                "justificativa": str(e.get("justificativa") or ""),
                "autora_aprovou": e.get("autora_aprovou") is True,
            }
        )
    return saida


def chave_numero(texto: str) -> str | None:
    """Forma canônica do único número de ``texto`` (a mesma da extração); ``None`` se não for um só."""
    chaves = list(extrair_tokens(texto)["numeros"])
    return chaves[0] if len(chaves) == 1 else None


def _frouxa(chave: str) -> str:
    """Chave sem o ``%`` final, para casar ``39,3`` do registro com ``39,3\\%`` do texto."""
    return chave[:-1] if chave.endswith("%") else chave


def ler_entradas_aceites_numeros(caminho: Path) -> list:
    """Entradas da seção ``aceites_numeros`` do YAML de aceites, como vieram."""
    if not caminho.exists():
        return []
    texto = caminho.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_aceites_simples(texto, SECAO_ACEITES_NUMEROS)
    dados = yaml.safe_load(texto)
    if isinstance(dados, dict):
        return list(dados.get(SECAO_ACEITES_NUMEROS) or [])
    return []


def carregar_aceites_numeros(caminho: Path) -> list[dict]:
    """Aceites por arquivo e número: ``{arquivo, numero, autora_aprovou, motivo}``."""
    saida = []
    for e in ler_entradas_aceites_numeros(caminho):
        if not isinstance(e, dict):
            raise ErroAceites(f"aceite por arquivo que não é um mapeamento: {e!r}")
        arquivo = str(e.get("arquivo") or "").replace("\\", "/").strip()
        numero = e.get("numero")
        chave = None if numero is None or isinstance(numero, bool) else chave_numero(str(numero))
        if not arquivo or chave is None:
            raise ErroAceites(
                f"aceite por arquivo inválido: arquivo={e.get('arquivo')!s} numero={numero!s}; "
                "informe `arquivo` e `numero` (um só número, entre aspas)"
            )
        saida.append(
            {
                "arquivo": arquivo,
                "numero": chave,
                "autora_aprovou": e.get("autora_aprovou") is True,
                "motivo": str(e.get("motivo") or ""),
            }
        )
    return saida


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------


@dataclass
class Achado:
    arquivo: str
    tipo: str  # modificado | removido | adicionado
    hash: str
    posicao_base: int | None
    posicao_alvo: int | None
    diferencas: dict
    itens: list
    alerta: bool
    status: str  # pendente | aceito | info
    motivo: str
    antigo: str
    novo: str
    nota_aceite: str = ""


def trecho(texto: str, limite: int = MAX_TRECHO) -> str:
    t = " ".join(texto.split())
    return t if len(t) <= limite else t[: limite - 1] + "…"


def analisar_textos(
    arquivo: str,
    texto_base: str,
    texto_alvo: str,
    aceites: list[dict] | None = None,
    usados: set | None = None,
) -> tuple[list[Achado], dict]:
    """Compara dois textos; devolve os achados e contagens do alinhamento."""
    base = segmentar(texto_base)
    alvo = segmentar(texto_alvo)
    al = alinhar(base, alvo)
    achados: list[Achado] = []

    for i, j in al.pares:
        dif = comparar(extrair_tokens(base[i]), extrair_tokens(alvo[j]))
        achados.append(
            Achado(
                arquivo=arquivo,
                tipo="modificado",
                hash=hash_paragrafo(alvo[j]),
                posicao_base=i + 1,
                posicao_alvo=j + 1,
                diferencas=dif,
                itens=ids_itens(dif),
                alerta=bool(dif),
                status="pendente" if dif else "info",
                motivo="tokens com sentido mudaram" if dif else "",
                antigo=trecho(base[i]),
                novo=trecho(alvo[j]),
            )
        )
    for j in al.adicionados:
        tok = extrair_tokens(alvo[j])
        dif = {
            cat: {"removidos": [], "adicionados": sorted(tok[cat].elements())}
            for cat in CATEGORIAS_ADICIONADO
            if tok[cat]
        }
        achados.append(
            Achado(
                arquivo=arquivo,
                tipo="adicionado",
                hash=hash_paragrafo(alvo[j]),
                posicao_base=None,
                posicao_alvo=j + 1,
                diferencas=dif,
                itens=ids_itens(dif),
                alerta=bool(dif),
                status="pendente" if dif else "info",
                motivo="novo conteúdo factual" if dif else "",
                antigo="",
                novo=trecho(alvo[j]),
            )
        )
    for i in al.removidos:
        tok = extrair_tokens(base[i])
        dif = {
            cat: {"removidos": sorted(tok[cat].elements()), "adicionados": []}
            for cat in CATEGORIAS_REMOVIDO
            if tok[cat]
        }
        achados.append(
            Achado(
                arquivo=arquivo,
                tipo="removido",
                hash=hash_paragrafo(base[i]),
                posicao_base=i + 1,
                posicao_alvo=None,
                diferencas=dif,
                itens=ids_itens(dif),
                alerta=bool(dif),
                status="pendente" if dif else "info",
                motivo="parágrafo removido com números, citações ou negadores" if dif else "",
                antigo=trecho(base[i]),
                novo="",
            )
        )
    aplicar_aceites(achados, aceites or [], usados if usados is not None else set())
    contagens = {
        "iguais": al.iguais,
        "movidos": al.movidos,
        "modificados": len(al.pares),
        "adicionados": len(al.adicionados),
        "removidos": len(al.removidos),
    }
    return achados, contagens


def aplicar_aceites(achados: list[Achado], aceites: list[dict], usados: set) -> None:
    indice = {(a["arquivo"], a["hash_paragrafo_novo"]): a for a in aceites}
    for ach in achados:
        if not ach.alerta:
            continue
        ent = indice.get((ach.arquivo, ach.hash))
        if ent is None:
            continue
        usados.add((ent["arquivo"], ent["hash_paragrafo_novo"]))
        if not ent["autora_aprovou"]:
            ach.nota_aceite = "aceite sem aprovação da autora (autora_aprovou: false)"
            continue
        cobertos = set(ent["itens"])
        if not cobertos:
            ach.nota_aceite = "aceite sem itens não cobre nenhum alerta"
            continue
        if not set(ach.itens) <= cobertos:
            faltam = sorted(set(ach.itens) - cobertos)
            ach.nota_aceite = "aceite não cobre: " + ", ".join(faltam)
            continue
        ach.status = "aceito"
        ach.nota_aceite = ent["justificativa"]


# ---------------------------------------------------------------------------
# Números por capítulo (arquivo)
# ---------------------------------------------------------------------------

RE_NUMERO_PURO = re.compile(r"^[-\u2212]?\d[\d.,]*\s*(?:\\?%)?$")


class ErroInvariantes(ValueError):
    """Registro de invariantes ilegível ou módulo do registro indisponível."""


@dataclass
class Invariantes:
    """Números do registro (entradas N##) por arquivo, em chaves sem ``%`` final."""

    cobertura: dict = field(default_factory=dict)  # arquivo -> {número: [ids]}: números aceitos como novos
    exigidos: dict = field(default_factory=dict)  # arquivo -> {número: [ids]}: números que o arquivo deve ter


@dataclass
class AchadoNumero:
    arquivo: str
    numero: str
    tipo: str  # novo | aumentado | removido | removido_invariante | (movido não gera achado)
    contagem_base: int
    contagem_alvo: int
    alerta: bool
    status: str  # pendente | aceito | coberto | info
    nota: str = ""
    invariantes: list = field(default_factory=list)


def _importar_registro():
    ultimo: ImportError | None = None
    for nome in ("scripts.registro_invariantes", "registro_invariantes"):
        try:
            return importlib.import_module(nome)
        except ImportError as exc:
            ultimo = exc
    raise ErroInvariantes(f"registro de invariantes indisponível: {ultimo}")


def carregar_invariantes(caminho: Path) -> Invariantes:
    """Lê as entradas N## do registro; arquivo ausente equivale a registro vazio.

    A cobertura de um arquivo reúne (1) os ``texto_pt`` de todos os valores de cada
    entrada que o cita nas ocorrências e (2) os ``texto_pt`` dos valores exigidos
    (``exige``, ``exige_todos``, ``exige_algum``) por qualquer ocorrência que cite o
    arquivo, onde quer que o valor esteja definido (um valor definido em N01 e só
    referenciado por nome em N09 cobre o arquivo de N09). Os exigidos reúnem só os
    números puros listados em ``exige`` (por nome de valor ou literal) nas ocorrências
    do arquivo.
    """
    inv = Invariantes()
    if not caminho.is_file():
        return inv
    ri = _importar_registro()
    try:
        reg = ri.carregar_registro(caminho)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ri.ErroRegistro, ri.yaml.YAMLError) as exc:
        raise ErroInvariantes(f"registro de invariantes ilegível ({caminho}): {exc}") from exc
    textos = {
        v["nome"]: v.get("texto_pt")
        for e in reg.entradas
        for v in e.get("valores") or []
        if isinstance(v, dict) and "nome" in v
    }
    conhecidos = {str(t) for t in textos.values() if t is not None}
    for e in reg.entradas:
        ident = str(e.get("id", ""))
        if not ident.startswith("N"):
            continue
        textos_entrada = [
            str(v["texto_pt"]) for v in e.get("valores") or [] if isinstance(v, dict) and v.get("texto_pt") is not None
        ]
        for oc in e.get("ocorrencias") or []:
            arquivo = str(oc.get("arquivo", "")).replace("\\", "/")
            if not arquivo:
                continue
            cob = inv.cobertura.setdefault(arquivo, {})
            for t in textos_entrada:
                for chave in extrair_tokens(t)["numeros"]:
                    cob.setdefault(_frouxa(chave), []).append(ident)
            for campo in ("exige", "exige_todos", "exige_algum"):
                for token in oc.get(campo) or []:
                    for alt in (token if isinstance(token, list) else [token]):
                        alt = str(alt)
                        # nome de valor (definido em qualquer entrada) ou literal igual a um texto_pt
                        texto = textos.get(alt) if alt in textos else (alt if alt in conhecidos else None)
                        if texto is None:
                            continue
                        for chave in extrair_tokens(str(texto))["numeros"]:
                            cob.setdefault(_frouxa(chave), []).append(ident)
            exi = inv.exigidos.setdefault(arquivo, {})
            for token in oc.get("exige") or []:
                if isinstance(token, list):
                    continue  # sublista = alternativas (OU): nenhuma é exigida sozinha
                texto = textos.get(token) if token in textos else token
                if texto is None or not RE_NUMERO_PURO.match(str(texto).strip()):
                    continue
                chave = chave_numero(str(texto))
                if chave is not None:
                    exi.setdefault(_frouxa(chave), []).append(ident)
    for tabela in (inv.cobertura, inv.exigidos):
        for por_numero in tabela.values():
            for chave, ids in por_numero.items():
                por_numero[chave] = sorted(set(ids))
    return inv


def contar_numeros(texto: str) -> Counter:
    """Multiconjunto de números do arquivo, pela mesma extração dos parágrafos.

    Os números de rótulo ("passo 11", "Tabela 3", "seção 2.1") ficam de fora: são
    isentos pela regra ``rotulo`` de ``scripts/cobertura_numeros.py``.
    """
    total: Counter = Counter()
    for par in segmentar(texto):
        total.update(extrair_tokens(par, isentar_rotulos=True)["numeros"])
    return total


def _agregar_frouxo(contagem: Counter) -> Counter:
    total: Counter = Counter()
    for chave, n in contagem.items():
        total[_frouxa(chave)] += n
    return total


def analisar_numeros_arquivo(
    arquivo: str,
    texto_base: str,
    texto_alvo: str,
    invariantes: Invariantes | None = None,
    aceites_numeros: list[dict] | None = None,
    usados: set | None = None,
) -> list[AchadoNumero]:
    """Compara os multiconjuntos de números de um arquivo na base e no alvo.

    Número novo ou com contagem maior: pendente, salvo cobertura por invariante
    (``coberto``) ou aceite por arquivo (``aceito``). Contagem menor: ``info``, salvo
    quando o número é exigido pelo registro neste arquivo e deixa de aparecer nele
    (``removido_invariante``, pendente, liberado só por aceite). Mesma contagem
    (número movido): nada.
    """
    inv = invariantes or Invariantes()
    cobertura = inv.cobertura.get(arquivo, {})
    exigidos = inv.exigidos.get(arquivo, {})
    aceites: dict[tuple[str, str], dict] = {}
    for a in aceites_numeros or []:
        chave_a = (a["arquivo"], _frouxa(a["numero"]))
        if chave_a not in aceites or a["autora_aprovou"]:
            aceites[chave_a] = a

    base = contar_numeros(texto_base)
    alvo = contar_numeros(texto_alvo)
    alvo_frouxo = _agregar_frouxo(alvo)
    achados: list[AchadoNumero] = []

    def pendencia(ach: AchadoNumero) -> None:
        ent = aceites.get((arquivo, _frouxa(ach.numero)))
        if ent is None:
            return
        if usados is not None:
            usados.add((ent["arquivo"], _frouxa(ent["numero"])))
        if ent["autora_aprovou"]:
            ach.status = "aceito"
            ach.alerta = False
            ach.nota = ent["motivo"]
        else:
            ach.nota = "aceite sem aprovação da autora (autora_aprovou: false)"

    for numero in sorted(set(base) | set(alvo)):
        antes, depois = base[numero], alvo[numero]
        if depois == antes:
            continue  # inclusive número movido de parágrafo
        frouxo = _frouxa(numero)
        if depois > antes:
            ach = AchadoNumero(
                arquivo, numero, "novo" if antes == 0 else "aumentado", antes, depois, True, "pendente"
            )
            ids = cobertura.get(frouxo)
            if ids:
                ach.alerta = False
                ach.status = "coberto"
                ach.invariantes = list(ids)
                ach.nota = "valor de invariante do registro: " + ", ".join(ids)
            else:
                pendencia(ach)
        elif frouxo in exigidos and alvo_frouxo[frouxo] == 0:
            ach = AchadoNumero(
                arquivo, numero, "removido_invariante", antes, depois, True, "pendente",
                invariantes=list(exigidos[frouxo]),
            )  # fmt: skip
            ach.nota = "valor de invariante exigido neste arquivo: " + ", ".join(exigidos[frouxo])
            pendencia(ach)
        else:
            ach = AchadoNumero(arquivo, numero, "removido", antes, depois, False, "info")
        achados.append(ach)
    return achados


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


class ErroGit(RuntimeError):
    pass


def _git(raiz: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=str(raiz),
        capture_output=True,
        creationflags=flags,
    )
    if check and proc.returncode != 0:
        raise ErroGit(proc.stderr.decode("utf-8", errors="replace").strip() or "git falhou")
    return proc


def raiz_git(cwd: Path) -> Path:
    proc = _git(cwd, "rev-parse", "--show-toplevel")
    return Path(proc.stdout.decode("utf-8").strip())


def ref_valida(raiz: Path, ref: str) -> bool:
    return _git(raiz, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False).returncode == 0


def ler_ref(raiz: Path, ref: str, arquivo: str) -> str | None:
    proc = _git(raiz, "show", f"{ref}:{arquivo}", check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def ler_disco(raiz: Path, arquivo: str) -> str | None:
    caminho = raiz / arquivo
    if not caminho.is_file():
        return None
    return caminho.read_bytes().decode("utf-8", errors="replace")


def _lista_z(proc: subprocess.CompletedProcess) -> list[str]:
    return [p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p]


def eh_arquivo_alvo(arquivo: str) -> bool:
    if Path(arquivo).name in EXCLUIDOS:
        return False
    if arquivo == ARQUIVO_EXTRA:
        return True
    return arquivo.startswith("text/latex/") and arquivo.endswith(".tex")


def arquivos_alterados(raiz: Path, base: str, alvo: str) -> list[str]:
    if alvo.upper() == ALVO_WORKTREE:
        achados = _lista_z(_git(raiz, "diff", "--name-only", "--no-renames", "-z", base))
        achados += _lista_z(_git(raiz, "ls-files", "--others", "--exclude-standard", "-z"))
    else:
        achados = _lista_z(_git(raiz, "diff", "--name-only", "--no-renames", "-z", base, alvo))
    return sorted({a for a in achados if eh_arquivo_alvo(a)})


def eh_tabela_gerada(arquivo: str, conteudo: str | None) -> bool:
    if not arquivo.startswith("text/latex/tables/") or not conteudo:
        return False
    return MARCA_GERADA in "\n".join(conteudo.splitlines()[:10])


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


@dataclass
class Resultado:
    base: str
    alvo: str
    achados: list = field(default_factory=list)
    gerados: list = field(default_factory=list)
    contagens: dict = field(default_factory=dict)
    orfaos: list = field(default_factory=list)
    numeros: list = field(default_factory=list)  # AchadoNumero, comparação por arquivo
    orfaos_numeros: list = field(default_factory=list)
    modo: str = MODO_PADRAO
    modos_arquivo: dict = field(default_factory=dict)  # arquivo -> modo efetivo (capitulo|paragrafo|ambos)

    @property
    def pendentes(self) -> list:
        return [a for a in self.achados if a.alerta and a.status == "pendente"]

    @property
    def numeros_pendentes(self) -> list:
        return [n for n in self.numeros if n.alerta and n.status == "pendente"]

    @property
    def n_aceitos(self) -> int:
        """Alertas aceitos, de parágrafo e de número por arquivo."""
        return len(self.aceitos) + sum(1 for n in self.numeros if n.status == "aceito")

    @property
    def gerados_pendentes(self) -> list:
        """Tabelas geradas que mudaram em relação à base e não foram aceitas."""
        return [g for g in self.gerados if g.get("status") == "pendente"]

    @property
    def n_pendentes(self) -> int:
        """Total que conta para o código de saída."""
        return len(self.pendentes) + len(self.gerados_pendentes) + len(self.numeros_pendentes)

    @property
    def aceitos(self) -> list:
        return [a for a in self.achados if a.status == "aceito"]


def _arquivo_revisado(arquivo: str, revisados: frozenset[str] | None) -> bool:
    """O arquivo inteiro consta da lista de revisados (chaves ``arq#secao`` não valem aqui)?"""
    return arquivo.replace("\\", "/") in (revisados or frozenset())


def _sem_eol(texto: str) -> str:
    return texto.replace("\r\n", "\n").replace("\r", "\n")


def aceitar_tabela(ent: dict, aceites: list[dict], usados: set) -> None:
    """Tabela gerada alterada só deixa de ser pendente com aceite que cite ``tabela_gerada:<estado>``."""
    for a in aceites:
        if (a["arquivo"], a["hash_paragrafo_novo"]) != (ent["arquivo"], ent["hash"]):
            continue
        usados.add((a["arquivo"], a["hash_paragrafo_novo"]))
        if not a["autora_aprovou"]:
            ent["nota_aceite"] = "aceite sem aprovação da autora (autora_aprovou: false)"
        elif f"tabela_gerada:{ent['estado']}" not in a["itens"]:
            ent["nota_aceite"] = f"aceite não cobre: tabela_gerada:{ent['estado']}"
        else:
            ent["status"] = "aceito"
            ent["nota_aceite"] = a["justificativa"]
        return


def carregar_revisados(raiz: Path) -> frozenset[str]:
    """Chaves de ``tests/data/arquivos_revisados.yaml`` do repositório (vazio se ausente)."""
    caminho = raiz / REVISADOS_PADRAO
    if not caminho.is_file():
        return frozenset()
    ri = _importar_registro()
    try:
        return ri.chaves_revisadas(caminho)
    except (OSError, ValueError, TypeError, ri.ErroRegistro, ri.yaml.YAMLError) as exc:
        raise ErroInvariantes(f"lista de arquivos revisados ilegível ({caminho}): {exc}") from exc


def analisar_repositorio(
    raiz: Path,
    base: str = BASE_PADRAO,
    alvo: str = ALVO_WORKTREE,
    arquivos: list[str] | None = None,
    aceites: list[dict] | None = None,
    modo: str = MODO_PADRAO,
    invariantes: Invariantes | None = None,
    aceites_numeros: list[dict] | None = None,
    registro: Path | None = None,
    revisados: frozenset[str] | None = None,
) -> Resultado:
    """Compara base e alvo.

    ``modo="auto"``: arquivo listado em ``arquivos_revisados.yaml`` (``revisados``) passa
    pela comparação ``capitulo``; os demais, por ``ambos``.
    """
    if modo not in MODOS:
        raise ValueError(f"modo inválido: {modo} (use {', '.join(MODOS)})")
    if modo == "auto" and revisados is None:
        revisados = carregar_revisados(raiz)
    if modo != "paragrafo" and invariantes is None:
        invariantes = carregar_invariantes(
            _resolver(raiz, str(registro)) if registro else raiz / REGISTRO_PADRAO
        )
    if not ref_valida(raiz, base):
        raise ErroGit(f"base inexistente: {base}")
    if alvo.upper() != ALVO_WORKTREE and not ref_valida(raiz, alvo):
        raise ErroGit(f"alvo inexistente: {alvo}")
    if arquivos:
        lista = sorted(
            {a.replace("\\", "/") for a in arquivos if Path(a).name not in EXCLUIDOS}
        )
    else:
        lista = arquivos_alterados(raiz, base, alvo)

    res = Resultado(base=base, alvo=alvo, modo=modo)
    usados: set = set()
    usados_numeros: set = set()
    aceites = aceites or []
    aceites_numeros = aceites_numeros or []
    for arq in lista:
        antigo = ler_ref(raiz, base, arq)
        if alvo.upper() == ALVO_WORKTREE:
            novo = ler_disco(raiz, arq)
        else:
            novo = ler_ref(raiz, alvo, arq)
        if antigo is None and novo is None:
            continue
        if antigo == novo:
            continue
        if eh_tabela_gerada(arq, novo if novo is not None else antigo):
            if antigo is not None and novo is not None and _sem_eol(antigo) == _sem_eol(novo):
                continue  # regenerada com conteúdo idêntico (só mudou o fim de linha)
            estado = "removida" if novo is None else ("nova" if antigo is None else "alterada")
            ent = {
                "arquivo": arq,
                "estado": estado,
                "hash": hash_paragrafo(novo if novo is not None else antigo or ""),
                "status": "pendente",
                "nota_aceite": "",
            }
            aceitar_tabela(ent, aceites, usados)
            res.gerados.append(ent)
            continue
        if modo == "auto":
            efetivo = "capitulo" if _arquivo_revisado(arq, revisados) else "ambos"
        else:
            efetivo = modo
        res.modos_arquivo[arq] = efetivo
        if efetivo != "capitulo":
            achados, cont = analisar_textos(arq, antigo or "", novo or "", aceites, usados)
            res.achados.extend(achados)
            res.contagens[arq] = cont
        if efetivo != "paragrafo":
            res.numeros.extend(
                analisar_numeros_arquivo(
                    arq, antigo or "", novo or "", invariantes, aceites_numeros, usados_numeros
                )
            )
    if modo != "capitulo":
        # aceite de parágrafo de arquivo comparado só por capítulo não tem o que casar
        so_capitulo = {a for a, m in res.modos_arquivo.items() if m == "capitulo"}
        res.orfaos = [
            {"arquivo": a["arquivo"], "hash_paragrafo_novo": a["hash_paragrafo_novo"]}
            for a in aceites
            if (a["arquivo"], a["hash_paragrafo_novo"]) not in usados and a["arquivo"] not in so_capitulo
        ]
    if modo != "paragrafo":
        res.orfaos_numeros = [
            {"arquivo": a["arquivo"], "numero": a["numero"]}
            for a in aceites_numeros
            if (a["arquivo"], _frouxa(a["numero"])) not in usados_numeros
        ]
    return res


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

NOMES_CATEGORIA = {
    "numeros": "números",
    "ordem_numeros": "ordem dos números na frase",
    "ordem_paragrafo": "ordem dos números no parágrafo",
    "negadores": "negadores",
    "quantificadores": "quantificadores/hedges",
    "qualificadores": "qualificadores numéricos",
    "comparativos": "comparativos",
    "metricas": "métricas e modos",
    "citacoes": "citações",
    "referencias": "referências cruzadas",
    "siglas": "siglas",
    "caminhos": "caminhos",
}


def _fmt_itens(lista: list[str]) -> str:
    cont = Counter(lista)
    return ", ".join(f"`{k}`" + (f" (×{v})" if v > 1 else "") for k, v in sorted(cont.items()))


def renderizar_markdown(res: Resultado) -> str:
    out = ["# Portão de fidelidade", ""]
    out.append(f"Base: `{res.base}`. Alvo: `{res.alvo}`. Modo: `{res.modo}`.")
    out.append("")
    if res.modo == "auto":
        cap = sorted(a for a, m in res.modos_arquivo.items() if m == "capitulo")
        out.append(
            "Arquivos revisados (comparação por capítulo): "
            + (", ".join(f"`{a}`" for a in cap) if cap else "nenhum")
            + "."
        )
        out.append("")
    veredito = "FALHA" if res.n_pendentes else "OK"
    out.append(
        f"**Resultado: {veredito}** ({res.n_pendentes} alertas pendentes, "
        f"{res.n_aceitos} aceitos)."
    )
    out.append("")
    por_arquivo: dict[str, list[Achado]] = {}
    for a in res.achados:
        por_arquivo.setdefault(a.arquivo, []).append(a)
    for arq in sorted(por_arquivo):
        achados = por_arquivo[arq]
        cont = res.contagens.get(arq, {})
        out.append(f"## {arq}")
        out.append("")
        out.append(
            "Parágrafos: {iguais} idênticos, {movidos} movidos, {modificados} modificados, "
            "{adicionados} adicionados, {removidos} removidos.".format(
                **{k: cont.get(k, 0) for k in ("iguais", "movidos", "modificados", "adicionados", "removidos")}
            )
        )
        out.append("")
        alertas = [a for a in achados if a.alerta]
        if not alertas:
            out.append("Sem alertas.")
            out.append("")
            continue
        for n, a in enumerate(alertas, 1):
            pos = []
            if a.posicao_base:
                pos.append(f"base §{a.posicao_base}")
            if a.posicao_alvo:
                pos.append(f"alvo §{a.posicao_alvo}")
            rotulo = "ACEITO" if a.status == "aceito" else "PENDENTE"
            out.append(f"### {n}. {a.tipo} ({', '.join(pos)}) [{rotulo}]")
            out.append("")
            out.append(f"Hash: `{a.hash}`. {a.motivo}.")
            if a.nota_aceite:
                out.append(f"Aceite: {a.nota_aceite}")
            out.append("")
            for cat in CATEGORIAS:
                d = a.diferencas.get(cat)
                if not d:
                    continue
                partes = []
                if d["removidos"]:
                    partes.append("removido: " + _fmt_itens(d["removidos"]))
                if d["adicionados"]:
                    partes.append("adicionado: " + _fmt_itens(d["adicionados"]))
                out.append(f"- {NOMES_CATEGORIA[cat]}: " + "; ".join(partes))
            out.append("")
            if a.antigo:
                out.append(f"> Antes: {a.antigo}")
            if a.novo:
                out.append(f"> Depois: {a.novo}")
            out.append("")
    if res.modo != "paragrafo":
        out.append("## Números por capítulo")
        out.append("")
        out.append(
            "Comparação do multiconjunto de números de cada arquivo alterado com a base. "
            "Número novo ou mais frequente é pendência, salvo valor de invariante do registro "
            "ou aceite por arquivo; número removido só é registrado, salvo valor de invariante "
            "exigido no arquivo; número movido não aparece."
        )
        out.append("")
        por_arq_num: dict[str, list[AchadoNumero]] = {}
        for n in res.numeros:
            por_arq_num.setdefault(n.arquivo, []).append(n)
        if not por_arq_num:
            out.append("Sem diferenças de números entre os arquivos alterados.")
            out.append("")
        rotulos = {"pendente": "PENDENTE", "aceito": "ACEITO", "coberto": "COBERTO", "info": "REGISTRADO"}
        for arq in sorted(por_arq_num):
            out.append(f"### {arq}")
            out.append("")
            for n in por_arq_num[arq]:
                linha = (
                    f"- `{n.numero}` {n.tipo.replace('_', ' ')} "
                    f"(base ×{n.contagem_base}, alvo ×{n.contagem_alvo}) [{rotulos.get(n.status, n.status)}]"
                )
                if n.nota:
                    linha += f". {n.nota}"
                out.append(linha)
            out.append("")
        if res.orfaos_numeros:
            out.append("Aceites por arquivo sem correspondência (número sem diferença atual ou aceite obsoleto):")
            out.append("")
            for o in res.orfaos_numeros:
                out.append(f"- `{o['arquivo']}` número `{o['numero']}`")
            out.append("")
    if res.gerados:
        out.append("## Tabelas geradas por script: regenerar e conferir")
        out.append("")
        out.append(
            "Tabela alterada em relação à base é pendência até ser regenerada com conteúdo "
            "idêntico ou aceita (item `tabela_gerada:<estado>`)."
        )
        out.append("")
        for g in res.gerados:
            rotulo = "ACEITA" if g.get("status") == "aceito" else "PENDENTE"
            linha = f"- `{g['arquivo']}` ({g['estado']}) [{rotulo}] hash `{g.get('hash', '')}`"
            if g.get("nota_aceite"):
                linha += f". Aceite: {g['nota_aceite']}"
            out.append(linha)
        out.append("")
    if res.orfaos:
        out.append("## Aceites sem correspondência")
        out.append("")
        out.append("Entradas de `docs/fidelidade_aceites.yaml` que não casam com nenhum alerta atual (texto reeditado ou aceite obsoleto):")
        out.append("")
        for o in res.orfaos:
            out.append(f"- `{o['arquivo']}` hash `{o['hash_paragrafo_novo']}`")
        out.append("")
    return "\n".join(out)


def resultado_json(res: Resultado) -> dict:
    return {
        "base": res.base,
        "alvo": res.alvo,
        "pendentes": res.n_pendentes,
        "pendentes_tabelas": len(res.gerados_pendentes),
        "aceitos": res.n_aceitos,
        "modo": res.modo,
        "modos_arquivo": res.modos_arquivo,
        "pendentes_numeros": len(res.numeros_pendentes),
        "achados": [asdict(a) for a in res.achados],
        "numeros": [asdict(n) for n in res.numeros],
        "gerados": res.gerados,
        "contagens": res.contagens,
        "orfaos": res.orfaos,
        "orfaos_numeros": res.orfaos_numeros,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolver(raiz: Path, caminho: str) -> Path:
    p = Path(caminho)
    return p if p.is_absolute() else raiz / p


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Portão de fidelidade para revisões só de texto.")
    ap.add_argument("--base", default=BASE_PADRAO, help="ref da versão de base (padrão: pre-revisao-2)")
    ap.add_argument("--alvo", default=ALVO_WORKTREE, help="WORKTREE (padrão) ou uma ref git")
    ap.add_argument("--arquivos", nargs="+", help="arquivos a comparar (padrão: .tex alterados)")
    ap.add_argument("--saida", default=SAIDA_PADRAO, help="relatório Markdown")
    ap.add_argument("--json", dest="json_saida", help="relatório JSON opcional")
    ap.add_argument("--aceites", default=ACEITES_PADRAO, help="YAML de alertas reconhecidos")
    ap.add_argument("--raiz", help="raiz do repositório (padrão: toplevel do cwd)")
    ap.add_argument(
        "--modo",
        choices=MODOS,
        default=MODO_PADRAO,
        help="auto (padrão: capitulo nos arquivos de tests/data/arquivos_revisados.yaml, ambos nos demais), "
        "capitulo (números por arquivo), paragrafo (alinhamento por parágrafo) ou ambos",
    )
    ap.add_argument("--registro", default=REGISTRO_PADRAO, help="registro de invariantes (YAML)")
    args = ap.parse_args(argv)

    try:
        raiz = Path(args.raiz).resolve() if args.raiz else raiz_git(Path.cwd())
        caminho_aceites = _resolver(raiz, args.aceites)
        aceites = carregar_aceites(caminho_aceites)
        aceites_numeros = carregar_aceites_numeros(caminho_aceites)
        res = analisar_repositorio(
            raiz,
            args.base,
            args.alvo,
            args.arquivos,
            aceites,
            modo=args.modo,
            aceites_numeros=aceites_numeros,
            registro=Path(args.registro),
        )
    except (ErroGit, ErroAceites, ErroInvariantes, OSError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    saida = _resolver(raiz, args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(renderizar_markdown(res), encoding="utf-8")
    if args.json_saida:
        js = _resolver(raiz, args.json_saida)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(resultado_json(res), ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Portão de fidelidade: {res.n_pendentes} alertas pendentes "
        f"({len(res.gerados_pendentes)} de tabelas geradas, "
        f"{len(res.numeros_pendentes)} de números por arquivo), "
        f"{res.n_aceitos} aceitos, {len(res.gerados)} tabelas geradas alteradas. "
        f"Relatório: {saida}"
    )
    return 1 if res.n_pendentes else 0


if __name__ == "__main__":
    sys.exit(main())
