#!/usr/bin/env python3
"""Blindagem de números na reescrita de trechos da tese por LLM.

Fluxo:
  1. ``mascarar``  troca cada número do trecho por um marcador ⟦k⟧ e grava o trecho
     mascarado e o mapa em ``.local/reescrita/``.
  2. Quem reescreve edita o ``.mascarado.tex``: só pode mover ou remover marcadores.
  3. ``verificar`` / ``restaurar`` conferem a edição e, sem violação, devolvem os valores
     e substituem as linhas A-B do .tex (``restaurar``).

Validações (sai com código 1 e lista cada problema):
  (a) todo marcador presente existe no mapa;
  (b) nenhum marcador aparece mais de uma vez;
  (c) não sobrou dígito fora de marcador, salvo em estruturas LaTeX isentas;
  (d) o trecho original no .tex mantém o mesmo sha256.
Números removidos viram aviso. ``--permitir-novos`` libera números novos de forma explícita.
``--permitir-ids "O3,F4"`` libera só ocorrências livres desses identificadores exatos (letras
seguidas de dígitos, com fronteira de palavra: O3 não libera O35 nem BO3); o dígito de outros
identificadores e números soltos continuam bloqueados, e cada liberação sai como ``LIBERADO-ID``.

``mascarar`` recusa (código 2) qualquer ``--arquivo`` sob ``text/latex/tables/``: tabelas ficam
fora da reescrita e mudam só pelo gerador (ver ``tests/data/tabelas_tese.yaml``).

Estruturas isentas (não mascaradas e não checadas como "dígito livre"): argumentos de
\\ref, \\label, \\cite*, \\input, \\includegraphics, \\hspace, \\vspace e afins; opções entre
colchetes e argumentos de \\begin{...} (tabular, minipage etc.); comandos de tabela
(\\multicolumn, \\cmidrule, \\\\[2pt]); larguras e comprimentos (0.5\\textwidth, 2cm);
comentários LaTeX (% até o fim da linha). Além da isenção, o texto editado não pode
conter estrutura isenta com dígitos que não exista no trecho original.

Por que não reaproveitar ``portao_fidelidade.extrair_tokens``: ela devolve valores
normalizados (39,3 = 39.3; separadores de milhar removidos; ``{,}`` trocado por vírgula)
sobre um texto já transformado, sem posições no texto original, e não conhece o sinal
``$-5{,}648$`` como unidade. Para mascarar é preciso a posição e o literal exato de cada
ocorrência, então a extração aqui é própria (``RE_NUMERO``), com os mesmos critérios de
sinal e separador de milhar do portão.

Uso:
  python scripts/reescrita_marcadores.py mascarar --arquivo text/latex/cap5.tex --linhas 120-140 --id cap5_res
  python scripts/reescrita_marcadores.py verificar --id cap5_res
  python scripts/reescrita_marcadores.py restaurar --id cap5_res --entrada caminho/editado.tex
  python scripts/reescrita_marcadores.py restaurar --id cap5_res --permitir-novos "0,95;12"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RAIZ_PADRAO = Path(__file__).resolve().parents[1]
SAIDA_REL = Path(".local") / "reescrita"
PASTA_TABELAS_REL = Path("text") / "latex" / "tables"
ABRE = "\u27e6"
FECHA = "\u27e7"
RE_MARCADOR = re.compile(ABRE + r"(\d+)" + FECHA)
RE_ID = re.compile(r"[\w.\-]+")
CONTEXTO = 30

# Número com sinal opcional, separador decimal (",", ".", "{,}") e milhar com espaço fino.
# O "%" não faz parte da unidade. Mesmo critério de sinal do portão de fidelidade.
RE_NUMERO = re.compile(
    r"(?:(?P<sinal>\$\s*[-\u2212]\s*\$|\u2212|(?<![\w\\\-])-)(?=\d))?"
    r"(?<!\d)(?P<num>\d+(?:(?:\{[.,]\}|[.,])\d+"
    r"|(?:\\,|[\u00a0\u202f\u2009])(?=\d{3}(?!\d))\d+)*)"
)

# Larguras e comprimentos soltos: 0.5\textwidth, 2cm, -1mm (unidade colada ao número).
RE_COMPRIMENTO = re.compile(
    r"(?<![\w.,\\])[-\u2212]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    r"(?:\s*\\(?:textwidth|linewidth|columnwidth|textheight|paperwidth|paperheight|hsize|vsize"
    r"|baselineskip|parindent|tabcolsep|arraycolsep)(?![A-Za-z])"
    r"|(?:cm|mm|pt|em|ex|in|bp|sp|pc|dd|cc|mu)(?![A-Za-z]))"
)

RE_TOKEN = re.compile(
    r"\\\\(?P<quebra>\*?\[[^\]\n]*\])?"
    r"|\\(?P<cmd>[A-Za-z@]+)\*?"
    r"|\\[^A-Za-z@\\]"
    r"|(?P<com>%[^\n]*)"
)

# Comando -> quantidade de argumentos entre chaves isentos (opções [..] antes de cada um).
CMD_ARGS = {
    "ref": 1, "eqref": 1, "autoref": 1, "pageref": 1, "cref": 1, "Cref": 1, "vref": 1,
    "nameref": 1, "label": 1, "input": 1, "include": 1, "includeonly": 1,
    "includegraphics": 1, "includepdf": 1, "bibliography": 1, "bibliographystyle": 1,
    "usepackage": 1, "url": 1, "path": 1, "href": 1, "hyperref": 0, "hspace": 1, "vspace": 1,
    "rule": 2, "scalebox": 1, "resizebox": 2, "fontsize": 2, "linespread": 1, "setstretch": 1,
    "setlength": 2, "addtolength": 2, "setcounter": 2, "addtocounter": 2, "multicolumn": 2,
    "multirow": 2, "cmidrule": 1, "cline": 1, "rowcolors": 3, "definecolor": 3,
    "textcolor": 1, "color": 1, "colorbox": 1, "rowcolor": 1, "cellcolor": 1,
    "columncolor": 1, "parbox": 1, "end": 1, "toprule": 0, "midrule": 0, "bottomrule": 0,
    "hline": 0, "geometry": 1,
}
# Argumentos entre chaves que seguem \begin{env}{...}.
ENV_ARGS = {
    "tabular": 1, "tabular*": 2, "tabularx": 2, "tabulary": 2, "longtable": 1, "array": 1,
    "minipage": 1, "adjustbox": 1, "multicols": 1, "multicols*": 1, "wrapfigure": 2,
}


class ErroUso(Exception):
    """Erro de uso ou de ambiente (arquivo ausente, linhas inválidas); sai com código 2."""


# ---------------------------------------------------------------------------
# Estruturas LaTeX isentas
# ---------------------------------------------------------------------------


def _opcoes(t: str, p: int, parens: bool = False, ws: bool = True) -> int:
    """Consome opções ``[..]`` (e ``(..)`` de \\cmidrule) a partir de ``p``."""
    sp = r"[ \t]*" if ws else ""
    alt = r"\[[^\]\n]*\]" + (r"|\([^)\n]*\)" if parens else "")
    padrao = re.compile(sp + "(?:" + alt + ")")
    while True:
        m = padrao.match(t, p)
        if not m:
            return p
        p = m.end()


def _chave(t: str, p: int) -> int | None:
    """Fim do grupo ``{..}`` balanceado que começa em ``p`` (após espaços), ou None."""
    j = re.compile(r"[ \t]*\n?[ \t]*").match(t, p).end()
    if j >= len(t) or t[j] != "{":
        return None
    nivel = 0
    i = j
    while i < len(t):
        c = t[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                return i + 1
        i += 1
    return None


def _consumir(t: str, pos: int, n: int, parens: bool = False, cauda: bool = False) -> int:
    """Consome ``n`` grupos entre chaves, cada um precedido de opções; devolve o fim."""
    cur = pos
    for _ in range(n):
        g = _chave(t, _opcoes(t, cur, parens))
        if g is None:
            break
        cur = g
    if n == 0 or cauda:
        cur = _opcoes(t, cur, parens, ws=False)
    return cur


def _fundir(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    saida: list[tuple[int, int]] = []
    for ini, fim in sorted(spans):
        if saida and ini < saida[-1][1]:
            saida[-1] = (saida[-1][0], max(saida[-1][1], fim))
        else:
            saida.append((ini, fim))
    return saida


def espacos_isentos(t: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Devolve (estruturas, comentarios) como listas de intervalos [ini, fim)."""
    estruturas: list[tuple[int, int]] = []
    comentarios: list[tuple[int, int]] = []
    pos = 0
    while True:
        m = RE_TOKEN.search(t, pos)
        if not m:
            break
        pos = m.end()
        if m.group("com"):
            comentarios.append((m.start(), m.end()))
            continue
        if m.group("quebra"):
            estruturas.append((m.start(), m.end()))
            continue
        nome = m.group("cmd")
        if not nome:
            continue
        if nome == "begin":
            fim_nome = _consumir(t, m.end(), 1)
            env = t[m.end():fim_nome].strip()[1:-1].strip()
            fim = _consumir(t, fim_nome, ENV_ARGS.get(env, 0), cauda=True)
        elif nome in CMD_ARGS:
            fim = _consumir(t, m.end(), CMD_ARGS[nome], parens=(nome == "cmidrule"))
        elif "cite" in nome:
            fim = _consumir(t, m.end(), 1)
        else:
            continue
        estruturas.append((m.start(), fim))
        pos = max(pos, fim)
    for m in RE_COMPRIMENTO.finditer(t):
        if not any(ini < m.end() and m.start() < fim for ini, fim in comentarios):
            estruturas.append((m.start(), m.end()))
    return _fundir(estruturas), comentarios


def localizar_numeros(t: str) -> tuple[list[re.Match], list[tuple[int, int]], list[tuple[int, int]]]:
    """Números fora de estruturas isentas e de comentários, com os intervalos isentos."""
    estruturas, comentarios = espacos_isentos(t)
    isentos = estruturas + comentarios
    achados = [
        m for m in RE_NUMERO.finditer(t)
        if not any(ini < m.end() and m.start() < fim for ini, fim in isentos)
    ]
    return achados, estruturas, comentarios


def _textos_isentos_com_digito(t_original: str, estruturas: list[tuple[int, int]]) -> Counter:
    c: Counter = Counter()
    for ini, fim in estruturas:
        s = " ".join(t_original[ini:fim].split())
        if re.search(r"\d", s):
            c[s] += 1
    return c


# ---------------------------------------------------------------------------
# Mascarar
# ---------------------------------------------------------------------------


def marcador(k: int) -> str:
    return f"{ABRE}{k}{FECHA}"


def mascarar_texto(texto: str, linha_inicial: int = 1) -> tuple[str, list[dict]]:
    """Troca cada número por ⟦k⟧. Devolve o texto mascarado e a lista k -> literal."""
    achados, _, _ = localizar_numeros(texto)
    partes: list[str] = []
    itens: list[dict] = []
    cursor = 0
    for k, m in enumerate(achados, start=1):
        partes.append(texto[cursor:m.start()])
        partes.append(marcador(k))
        cursor = m.end()
        antes = " ".join(texto[max(0, m.start() - CONTEXTO):m.start()].split())
        depois = " ".join(texto[m.end():m.end() + CONTEXTO].split())
        itens.append({
            "k": k,
            "texto": m.group(0),
            "linha": linha_inicial + texto.count("\n", 0, m.start()),
            "contexto": f"{antes} [{m.group(0)}] {depois}".strip(),
        })
    partes.append(texto[cursor:])
    return "".join(partes), itens


# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------


def dividir_linhas(texto: str) -> list[str]:
    """Linhas com o terminador original (só ``\\n`` separa; ``\\r\\n`` fica na linha)."""
    return re.findall(r"[^\n]*\n|[^\n]+", texto)


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def resolver(raiz: Path, caminho: str) -> Path:
    p = Path(caminho)
    return p if p.is_absolute() else raiz / p


def parse_linhas(valor: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", valor)
    if not m:
        raise ErroUso(f"--linhas inválido: {valor!r} (use A-B)")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    if a < 1 or b < a:
        raise ErroUso(f"--linhas inválido: {valor!r} (precisa 1 <= A <= B)")
    return a, b


def ler_utf8(caminho: Path) -> str:
    try:
        return caminho.read_bytes().decode("utf-8")
    except FileNotFoundError as e:
        raise ErroUso(f"arquivo não encontrado: {caminho}") from e
    except UnicodeDecodeError as e:
        raise ErroUso(f"arquivo não está em UTF-8: {caminho} ({e})") from e


def caminhos_saida(raiz: Path, id_: str) -> tuple[Path, Path]:
    if not RE_ID.fullmatch(id_):
        raise ErroUso(f"--id inválido: {id_!r} (use letras, dígitos, _, - e .)")
    pasta = raiz / SAIDA_REL
    return pasta / f"{id_}.mascarado.tex", pasta / f"{id_}.mapa.json"


def recusar_tabela(raiz: Path, caminho: Path) -> None:
    """Tabelas da tese (``text/latex/tables/``) mudam só pelo gerador, nunca por reescrita."""
    pasta = os.path.normcase(str((raiz / PASTA_TABELAS_REL).resolve()))
    alvo = os.path.normcase(str(caminho.resolve()))
    if alvo == pasta or alvo.startswith(pasta + os.sep):
        raise ErroUso(
            f"{arquivo_rel(raiz, caminho)} é tabela da tese (text/latex/tables/): tabelas mudam só "
            "pelo gerador, nunca por reescrita de texto. Para alterar uma tabela, altere o gerador "
            "e regenere; tabelas sem gerador mudam só com aprovação da autora "
            "(tests/data/tabelas_tese.yaml)."
        )


def arquivo_rel(raiz: Path, caminho: Path) -> str:
    try:
        return caminho.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        return str(caminho)


def cmd_mascarar(raiz: Path, arquivo: str, linhas: str, id_: str | None) -> int:
    a, b = parse_linhas(linhas)
    caminho = resolver(raiz, arquivo)
    recusar_tabela(raiz, caminho)
    todas = dividir_linhas(ler_utf8(caminho))
    if b > len(todas):
        raise ErroUso(f"o arquivo tem {len(todas)} linhas; pedido {a}-{b}")
    trecho = "".join(todas[a - 1:b])
    id_ = id_ or f"{caminho.stem}_L{a}-{b}"
    f_mask, f_mapa = caminhos_saida(raiz, id_)
    mascarado, itens = mascarar_texto(trecho, a)
    sobra = validar_texto(mascarado, {"linhas": [a, b], "marcadores": itens, "isentos": {}}, [])
    if any(e.startswith("dígito livre") for e in sobra.erros):
        raise ErroUso("inconsistência interna: o trecho mascarado ainda tem dígitos livres:\n"
                      + "\n".join(sobra.erros))
    try:
        rel = caminho.resolve().relative_to(raiz.resolve()).as_posix()
    except ValueError:
        rel = str(caminho.resolve())
    estruturas, _ = espacos_isentos(trecho)
    mapa = {
        "versao": 1,
        "arquivo": rel,
        "linhas": [a, b],
        "sha256": sha256_texto(trecho),
        "eol": "CRLF" if "\r\n" in trecho else "LF",
        "isentos": dict(_textos_isentos_com_digito(trecho, estruturas)),
        "marcadores": itens,
    }
    f_mask.parent.mkdir(parents=True, exist_ok=True)
    f_mask.write_bytes(mascarado.encode("utf-8"))
    f_mapa.write_text(json.dumps(mapa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Mascarado {rel} linhas {a}-{b}: {len(itens)} números em marcadores.")
    print(f"  trecho: {f_mask}")
    print(f"  mapa:   {f_mapa}")
    return 0


# ---------------------------------------------------------------------------
# Validar
# ---------------------------------------------------------------------------


@dataclass
class Resultado:
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    mantidos: list[int] = field(default_factory=list)
    removidos: list[int] = field(default_factory=list)
    movidos: list[int] = field(default_factory=list)
    liberados: list[str] = field(default_factory=list)
    liberados_id: list[str] = field(default_factory=list)


RE_ID_LIVRE = re.compile(r"[A-Za-z]+\d+")


def parse_ids(valores: list[str] | None) -> list[str]:
    """Identificadores exatos liberados (``O3``, ``F4``, ``K1``): letras seguidas de dígitos.

    A vírgula (ou ';') separa; repetir a opção vale. Qualquer item fora do formato é erro de uso.
    """
    saida: list[str] = []
    for v in valores or []:
        for item in re.split(r"[,;]", v):
            item = item.strip()
            if not item:
                continue
            if not RE_ID_LIVRE.fullmatch(item):
                raise ErroUso(
                    f"--permitir-ids inválido: {item!r} (use letras seguidas de dígitos, ex.: O3,F4)"
                )
            if item not in saida:
                saida.append(item)
    return saida


def parse_permitir(valores: list[str] | None) -> list[str]:
    """Lista de números liberados. Repetir a opção vale.

    ';' sempre separa. Em cada bloco, se ele inteiro for um número (0,95; 61.696; -5{,}648),
    é um único valor; senão a vírgula separa. Um valor liberado vale para todas as
    ocorrências dele no texto editado.
    """
    saida: list[str] = []
    for v in valores or []:
        for bloco in v.split(";"):
            bloco = bloco.strip()
            if not bloco:
                continue
            if RE_NUMERO.fullmatch(bloco):
                saida.append(bloco)
            else:
                saida.extend(p.strip() for p in bloco.split(",") if p.strip())
    return saida


def _movidos(ordem: list[int]) -> list[int]:
    """Marcadores fora da maior subsequência crescente (os que mudaram de ordem)."""
    n = len(ordem)
    if n == 0:
        return []
    tam = [1] * n
    ant = [-1] * n
    for i in range(n):
        for j in range(i):
            if ordem[j] < ordem[i] and tam[j] + 1 > tam[i]:
                tam[i] = tam[j] + 1
                ant[i] = j
    fim = max(range(n), key=lambda i: tam[i])
    ficam = set()
    while fim != -1:
        ficam.add(fim)
        fim = ant[fim]
    return [ordem[i] for i in range(n) if i not in ficam]


def validar_texto(
    texto: str, mapa: dict, permitir: list[str], ids: list[str] | None = None
) -> Resultado:
    """Validações (a), (b) e (c) do texto editado contra o mapa.

    ``ids`` libera ocorrências livres de identificadores exatos (``O3``), com fronteira de
    palavra dos dois lados; o dígito de qualquer outro identificador continua bloqueado.
    """
    res = Resultado()
    a = mapa["linhas"][0]
    literais = {m["k"]: m for m in mapa["marcadores"]}

    def pos_linha(p: int) -> str:
        rel = texto.count("\n", 0, p) + 1
        return f"linha {a + rel - 1} do .tex (linha {rel} do trecho)"

    presentes: list[tuple[int, int]] = []
    for m in RE_MARCADOR.finditer(texto):
        if m.group(1) != str(int(m.group(1))):
            res.erros.append(f"marcador malformado {m.group(0)} na {pos_linha(m.start())}")
            continue
        presentes.append((int(m.group(1)), m.start()))
    branco = RE_MARCADOR.sub(lambda m: "\x00" * len(m.group(0)), texto)
    for m in re.finditer(f"[{ABRE}{FECHA}]", branco):
        res.erros.append(f"marcador malformado (delimitador solto) na {pos_linha(m.start())}")

    contagem = Counter(k for k, _ in presentes)
    for k, p in presentes:
        if k not in literais:
            res.erros.append(f"marcador desconhecido {marcador(k)} na {pos_linha(p)}")
    for k, n in sorted(contagem.items()):
        if n > 1 and k in literais:
            linhas = ", ".join(pos_linha(p).split(" do ")[0] for kk, p in presentes if kk == k)
            res.erros.append(f"marcador duplicado {marcador(k)} ({n} vezes; {linhas})")

    achados, estruturas, _ = localizar_numeros(branco)
    spans_id: list[tuple[int, int, str]] = []
    if ids:
        alternativa = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
        for mi in re.finditer(rf"(?<!\w)(?:{alternativa})(?!\w)", branco):
            spans_id.append((mi.start(), mi.end(), mi.group(0)))
    for m in achados:
        lit = m.group(0)
        id_cobre = next((s for s in spans_id if s[0] <= m.start() and m.end() <= s[1]), None)
        if id_cobre is not None:
            res.liberados_id.append(f"{id_cobre[2]} ({pos_linha(m.start())})")
            continue
        if lit in permitir or m.group("num") in permitir:
            res.liberados.append(f"{lit} ({pos_linha(m.start())})")
            continue
        res.erros.append(
            f"dígito livre {lit!r} na {pos_linha(m.start())} "
            "(para liberar use --permitir-novos; ';' separa valores; "
            "--permitir-ids para identificadores como O3)"
        )

    originais = Counter(mapa.get("isentos", {}))
    for s, n in _textos_isentos_com_digito(texto, estruturas).items():
        if n > originais.get(s, 0):
            if any(v in s for v in permitir):
                res.liberados.append(f"estrutura isenta {s!r}")
            else:
                res.erros.append(
                    f"estrutura LaTeX isenta com dígitos nova ou alterada: {s!r} "
                    "(não existe no trecho original)"
                )

    vistos = [k for k, _ in presentes if k in literais]
    res.mantidos = sorted(set(vistos))
    res.removidos = sorted(set(literais) - set(vistos))
    res.movidos = sorted(set(_movidos(list(dict.fromkeys(vistos)))))
    for k in res.removidos:
        it = literais[k]
        res.avisos.append(
            f"número removido {marcador(k)} = {it['texto']!r} (linha {it['linha']}: {it['contexto']})"
        )
    return res


def ajustar_eol(editado: str, trecho: str, mascarado: str) -> str:
    """Leva o texto editado ao padrão de quebra de linha do trecho original."""
    if editado == mascarado:
        return editado
    eol = "\r\n" if "\r\n" in trecho else "\n"
    t = editado.replace("\r\n", "\n")
    if eol == "\r\n":
        t = t.replace("\n", "\r\n")
    if trecho.endswith("\n") and not t.endswith("\n"):
        t += eol
    elif not trecho.endswith("\n") and t.endswith("\n"):
        t = t[: -len(eol)]
    return t


def carregar_mapa(raiz: Path, id_: str) -> tuple[dict, Path]:
    f_mask, f_mapa = caminhos_saida(raiz, id_)
    if not f_mapa.exists():
        raise ErroUso(f"mapa não encontrado: {f_mapa} (rode 'mascarar' antes)")
    return json.loads(f_mapa.read_text(encoding="utf-8")), f_mask


def executar(
    raiz: Path, id_: str, entrada: str | None, permitir: list[str], escrever: bool,
    ids: list[str] | None = None,
) -> int:
    mapa, f_mask = carregar_mapa(raiz, id_)
    f_entrada = resolver(raiz, entrada) if entrada else f_mask
    editado = ler_utf8(f_entrada)

    caminho = resolver(raiz, mapa["arquivo"])
    original = ler_utf8(caminho)
    todas = dividir_linhas(original)
    a, b = mapa["linhas"]
    res = validar_texto(editado, mapa, permitir, ids)

    trecho: str | None = None
    if b > len(todas):
        res.erros.append(
            f"o trecho original mudou (sha256 não confere: o arquivo tem {len(todas)} linhas "
            f"e o mapa pede {a}-{b}; sha256 do mapa {mapa['sha256'][:16]}...)"
        )
    else:
        trecho = "".join(todas[a - 1:b])
        if sha256_texto(trecho) != mapa["sha256"]:
            res.erros.append(
                f"sha256 do trecho original diverge: mapa {mapa['sha256'][:16]}..., "
                f"arquivo agora {sha256_texto(trecho)[:16]}... (o .tex mudou desde 'mascarar')"
            )

    for av in res.avisos:
        print(f"AVISO: {av}")
    for lib in res.liberados:
        print(f"LIBERADO: número novo autorizado {lib}")
    for lib in res.liberados_id:
        print(f"LIBERADO-ID: identificador autorizado {lib}")
    if res.erros:
        for e in res.erros:
            print(f"ERRO: {e}")
        print(f"{len(res.erros)} violação(ões); nada foi escrito.")
        return 1

    assert trecho is not None
    mascarado, _ = mascarar_texto(trecho, a)
    if editado.startswith("\ufeff") and not trecho.startswith("\ufeff"):
        editado = editado[1:]
    editado = ajustar_eol(editado, trecho, mascarado)
    literais = {m["k"]: m["texto"] for m in mapa["marcadores"]}
    restaurado = RE_MARCADOR.sub(lambda m: literais[int(m.group(1))], editado)

    if escrever:
        novo = "".join(todas[:a - 1]) + restaurado + "".join(todas[b:])
        caminho.write_bytes(novo.encode("utf-8"))
    verbo = "Restaurado" if escrever else "Verificado (nada escrito)"
    print(f"{verbo}: {mapa['arquivo']} linhas {a}-{b}.")
    print(
        f"Marcadores: {len(res.mantidos)} mantidos, {len(res.removidos)} removidos, "
        f"{len(res.movidos)} movidos"
        + (f" ({', '.join(marcador(k) for k in res.movidos)})" if res.movidos else "")
        + "."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--raiz", default=str(RAIZ_PADRAO), help="raiz do repositório")
    sub = ap.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("mascarar", parents=[comum], help="troca números por marcadores")
    p.add_argument("--arquivo", required=True)
    p.add_argument("--linhas", required=True, help="A-B (1-based, inclusivo)")
    p.add_argument("--id", default=None)

    for nome, ajuda in (("restaurar", "valida e devolve os números ao .tex"),
                        ("verificar", "valida sem escrever")):
        q = sub.add_parser(nome, parents=[comum], help=ajuda)
        q.add_argument("--id", required=True)
        q.add_argument("--entrada", default=None, help="texto editado (padrão: .mascarado.tex)")
        q.add_argument("--permitir-novos", action="append", default=None,
                       help="números novos autorizados, por valor (vale para todas as "
                            "ocorrências dele): 'a;b'; a vírgula separa só quando o bloco "
                            "não é um número ('12, 7'); '0,95' é um único número; repetível")
        q.add_argument("--permitir-ids", action="append", default=None,
                       help="identificadores exatos autorizados no texto editado, separados por "
                            "vírgula (letras + dígitos, ex.: 'O3,F4,K1'); só casa o "
                            "identificador inteiro (O3 não libera O35 nem BO3); repetível")
    return ap


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        codificacao = (getattr(fluxo, "encoding", None) or "").lower().replace("-", "")
        if hasattr(fluxo, "reconfigure") and codificacao != "utf8":
            fluxo.reconfigure(encoding="utf-8")
    args = construir_parser().parse_args(argv)
    raiz = Path(args.raiz)
    try:
        if args.comando == "mascarar":
            return cmd_mascarar(raiz, args.arquivo, args.linhas, args.id)
        permitir = parse_permitir(args.permitir_novos)
        ids = parse_ids(args.permitir_ids)
        return executar(raiz, args.id, args.entrada, permitir,
                        escrever=args.comando == "restaurar", ids=ids)
    except ErroUso as e:
        print(f"ERRO DE USO: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
