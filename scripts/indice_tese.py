"""Índice de navegação da tese, gerado por script (nunca editado à mão).

Antes de mexer num trecho, o editor consulta o índice para ver tudo o que o trecho
toca: rótulos e referências cruzadas, citações, tabelas, figuras, artefatos
``\\path{data/...}``, entradas do registro de invariantes, veredito da atestação
bibliográfica, pendências abertas e contagens do linter de prosa.

Estrutura (na ordem do documento): capítulo, seção, subseção, com ``arquivo:linha``
inicial e final. Capítulos são ``\\begin{chapter}{...}`` ou ``\\chapter{...}``; o que vem
depois de ``\\appendix`` recebe letra. Trechos sem capítulo (página de rosto, ``Resumo`` e
``Abstract`` de ``tese.tex``) viram nós próprios (``bloco`` ou ``avulso``).

Leitores reaproveitados (sem alterá-los):

* ``scripts/linter_prosa.py``: árvore de ``\\input`` (``carregar_arvore``), texto sem
  comentários (``Arquivo.limpo``), prosa neutralizada, segmentação em parágrafos;
* ``scripts/citacoes_contexto.py``: contextos de citação com ``id`` estável, que ligam o
  ``ledger.csv`` e as pendências da atestação às seções.

Contagens por nó (``palavras``, ``paragrafos``) são do texto direto do nó, sem os filhos;
os totais do capítulo somam a subárvore. A contagem de palavras segue a R09 do linter,
exceto que nomes de ambientes (``\\begin{itemize}``) não contam como palavra. O linter
neutraliza o Abstract em inglês, então o nó ``Abstract`` sai com 0 palavras (as citações e
as entradas do registro dele aparecem normalmente).

Registro de invariantes e linter são por arquivo (não por seção): cada nó mostra o que
vale para o arquivo em que foi escrito; o resumo do capítulo soma os arquivos que ele
leu primeiro (tabelas incluídas). Atestação é por seção, pelo ``id`` da citação; se o
``id`` mudou (frase editada), cai-se na chave dentro do arquivo.

Uso::

    python scripts/indice_tese.py [--saida .local/indice/INDICE_TESE.md] \\
                                  [--json .local/indice/indice_tese.json]

Só biblioteca padrão e PyYAML. A saída é determinística (sem data nem hora).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import citacoes_contexto as cc  # noqa: E402
from scripts import linter_prosa as lp  # noqa: E402

TESE_PADRAO = ROOT / "text" / "latex" / "tese.tex"
REGISTRO_PADRAO = ROOT / "tests" / "data" / "registro_invariantes.yaml"
LEDGER_PADRAO = ROOT / "docs" / "atestado" / "ledger.csv"
PENDENCIAS_PADRAO = ROOT / "docs" / "atestado" / "pendencias.yaml"
BASELINE_PADRAO = ROOT / "tests" / "data" / "linter_prosa_baseline.json"
SAIDA_MD_PADRAO = ROOT / ".local" / "indice" / "INDICE_TESE.md"
SAIDA_JSON_PADRAO = ROOT / ".local" / "indice" / "indice_tese.json"

VERSAO = 1

NIVEIS = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
NOMES_BLOCO = {"resumo": "Resumo", "abstract": "Abstract"}
SUFIXOS_FIGURA = (".pgf", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg", ".tikz")

RE_TITULO = re.compile(r"\\(chapter|section|subsection|subsubsection)(\*?)(?![A-Za-z])")
RE_BEGIN_CHAPTER = re.compile(r"\\begin\{chapter\}")
RE_APPENDIX = re.compile(r"\\appendix(?![A-Za-z])")
RE_LABEL = re.compile(r"\\label\s*\{([^{}]*)\}")
RE_REF = re.compile(r"\\(ref|autoref|eqref|cref|Cref|pageref|nameref|vref)\*?\s*\{([^{}]*)\}")
RE_HYPERREF = re.compile(r"\\hyperref\s*\[([^\]]*)\]")
RE_PATH = re.compile(r"\\path\s*\{([^{}]*)\}")
RE_GRAFICO = re.compile(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
RE_AMBIENTE = re.compile(r"\\(?:begin|end)\s*\{[^{}]*\}")
RE_SEM_CONTEUDO = re.compile(
    r"\\(?:begin|end)\s*\{[^{}]*\}"
    r"|\\(?:bibliography|bibliographystyle|addcontentsline|input|include|label|vspace|hspace"
    r"|setcounter|pagenumbering)\s*(?:\{[^{}]*\})+"
)

# Marcadores de Resumo e Abstract: os mesmos de scripts/registro_invariantes.py.
BLOCOS_FRONTAIS = {
    "resumo": (r"\{\\LARGE\\bf Resumo\}", r"\\textbf\{Palavras-chave", "\\begin{resumo}", "\\end{resumo}"),
    "abstract": (r"\{\\LARGE\\bf Abstract\}", r"\\textbf\{Keywords", "\\begin{abstract}", "\\end{abstract}"),
}


# ---------------------------------------------------------------------------
# Cabeçalhos (capítulo, seção, subseção) de um arquivo
# ---------------------------------------------------------------------------
@dataclass
class Cabecalho:
    pos: int
    nivel: int  # -1 = marcador (\appendix)
    titulo: str
    tipo: str  # capitulo | secao | bloco | avulso | marcador
    estrela: bool = False
    secao: str = ""  # resumo | abstract, nos blocos


def _ler_grupo(s: str, i: int) -> tuple[str, int] | None:
    """Conteúdo do grupo ``{...}`` que começa a partir de ``i`` (pulando ``[...]``)."""
    n = len(s)
    j = i
    while j < n and s[j] in " \t\r\n":
        j += 1
    while j < n and s[j] == "[":
        fim = s.find("]", j)
        if fim < 0:
            return None
        j = fim + 1
        while j < n and s[j] in " \t\r\n":
            j += 1
    if j >= n or s[j] != "{":
        return None
    prof = 0
    k = j
    while k < n:
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            prof += 1
        elif c == "}":
            prof -= 1
            if prof == 0:
                return s[j + 1 : k], k + 1
        k += 1
    return None


def _limpar_titulo(t: str) -> str:
    t = re.sub(r"\\label\s*\{[^{}]*\}", "", t)
    t = re.sub(r"\\[A-Za-z]+\*?", "", t)
    t = re.sub(r"\\(.)", r"\1", t)
    t = t.replace("{", "").replace("}", "").replace("~", " ")
    t = " ".join(t.split())
    return t if len(t) <= 100 else t[:99] + "…"


def _blocos_frontais(bruto: str) -> list[tuple[str, int, int]]:
    """Blocos Resumo/Abstract do texto bruto: ``(secao, linha_ini, linha_fim)`` (1-based)."""
    sem = re.sub(r"(?<!\\)%.*", "", bruto)
    saida: list[tuple[str, int, int]] = []
    for secao, (ini, fim, abre, fecha) in BLOCOS_FRONTAIS.items():
        m = re.search(ini, sem)
        if m:
            f = re.search(fim, sem[m.end() :])
            if f:
                fim_pos = m.end() + f.end()
                nl = sem.find("\n", fim_pos)
                fim_pos = nl if nl != -1 else len(sem)
                saida.append((secao, sem.count("\n", 0, m.start()) + 1, sem.count("\n", 0, fim_pos) + 1))
            continue
        i = sem.find(abre)
        j = sem.find(fecha)
        if i != -1 and j > i:
            saida.append((secao, sem.count("\n", 0, i) + 1, sem.count("\n", 0, j) + 1))
    return saida


def extrair_cabecalhos(arq: lp.Arquivo, bruto: str | None) -> list[Cabecalho]:
    """Cabeçalhos de ``arq`` em ordem de posição (posições de ``arq.limpo``)."""
    s = arq.limpo
    cabs: list[Cabecalho] = []
    for m in RE_TITULO.finditer(s):
        g = _ler_grupo(s, m.end())
        if g is None:
            continue
        tipo = "capitulo" if m.group(1) == "chapter" else "secao"
        cabs.append(Cabecalho(m.start(), NIVEIS[m.group(1)], _limpar_titulo(g[0]), tipo, bool(m.group(2))))
    for m in RE_BEGIN_CHAPTER.finditer(s):
        g = _ler_grupo(s, m.end())
        if g is None:
            continue
        cabs.append(Cabecalho(m.start(), 0, _limpar_titulo(g[0]), "capitulo"))
    for m in RE_APPENDIX.finditer(s):
        cabs.append(Cabecalho(m.start(), -1, "", "marcador"))
    if bruto is not None and not arq.definicao:
        inicios = [0] + [i + 1 for i, c in enumerate(s) if c == "\n"]
        for secao, li, lf in _blocos_frontais(bruto):
            if li - 1 >= len(inicios):
                continue
            cabs.append(Cabecalho(inicios[li - 1], 0, NOMES_BLOCO[secao], "bloco", secao=secao))
            fim = inicios[lf] if lf < len(inicios) else len(s)
            cabs.append(Cabecalho(fim, 0, f"[{arq.rel}]", "avulso"))
    # No mesmo ponto, o fim de bloco (avulso) vem antes do cabeçalho seguinte.
    cabs.sort(key=lambda c: (c.pos, c.nivel, c.tipo != "avulso"))
    return cabs


# ---------------------------------------------------------------------------
# Nós do índice
# ---------------------------------------------------------------------------
@dataclass
class No:
    id: int
    nivel: int
    tipo: str  # capitulo | secao | bloco | avulso
    titulo: str
    arquivo: str
    pos: int
    linha_ini: int
    pai: int | None = None
    capitulo: int = 0
    apendice: bool = False
    secao_bloco: str = ""
    numero: str = ""
    linha_fim: int = 0
    filhos: list[int] = field(default_factory=list)
    pecas: list[tuple[str, int, int]] = field(default_factory=list)
    arquivos: set[str] = field(default_factory=set)
    palavras: int = 0
    paragrafos: int = 0
    paragrafos_longos: int = 0
    rotulos: list[dict] = field(default_factory=list)
    referencias: list[dict] = field(default_factory=list)
    citacoes: Counter = field(default_factory=Counter)
    figuras_tabelas: list[dict] = field(default_factory=list)
    artefatos: list[dict] = field(default_factory=list)
    atestacao: Counter = field(default_factory=Counter)
    pendencias: list[dict] = field(default_factory=list)

    @property
    def rotulo(self) -> str:
        if self.tipo in ("bloco", "avulso"):
            return self.titulo
        base = f"{self.numero} {self.titulo}".strip()
        if self.nivel == 0 and self.numero:
            base = f"{'Apêndice' if self.apendice else 'Cap.'} {base}"
        return base


def _satelite(rel: str) -> bool:
    """Arquivos de ``tables/`` pertencem à seção que os inclui."""
    return "tables" in PurePosixPath(rel).parts


def _classificar_include(alvo: str) -> str | None:
    a = alvo.strip().replace("\\", "/")
    if a.lower().endswith(SUFIXOS_FIGURA):
        return "figura"
    if "tables/" in a or a.startswith("tables"):
        return "tabela"
    return None


def _palavras(trecho: str) -> int:
    return lp.contar_palavras(RE_AMBIENTE.sub(" ", trecho))


class _Construtor:
    def __init__(self, arvore: lp.Arvore, base: Path) -> None:
        self.arvore = arvore
        self.base = base
        self.por_rel = {a.rel: a for a in arvore.arquivos.values()}
        self.nos: list[No] = []
        self.pilha: list[No] = []
        self.atual: No | None = None
        self.apendice = False
        self.n_cap = 0
        self.n_ap = 0
        self.cont = [0, 0, 0, 0]
        self.dono: dict[str, int] = {}
        self._cabs_cache: dict[str, list[Cabecalho]] = {}
        self._paras_cache: dict[str, tuple[list[int], list[tuple[int, int]]]] = {}
        self.incs: dict[tuple[str, int], str] = {}
        self._pecas: dict[str, list[tuple[int, int, int]]] | None = None
        for arq in arvore.arquivos.values():
            for ini, _fim, alvo in arq.includes:
                self.incs[(arq.rel, ini)] = alvo

    # -- caches ------------------------------------------------------------
    def cabs(self, arq: lp.Arquivo) -> list[Cabecalho]:
        if arq.rel not in self._cabs_cache:
            bruto = None
            caminho = self.base / arq.rel
            if caminho.is_file():
                bruto = caminho.read_text(encoding="utf-8-sig")
            self._cabs_cache[arq.rel] = extrair_cabecalhos(arq, bruto)
        return self._cabs_cache[arq.rel]

    def _paras(self, arq: lp.Arquivo) -> tuple[list[int], list[tuple[int, int]]]:
        if arq.rel not in self._paras_cache:
            ps = lp.paragrafos(arq)
            self._paras_cache[arq.rel] = ([a for a, _ in ps], ps)
        return self._paras_cache[arq.rel]

    # -- nós ---------------------------------------------------------------
    def _numerar(self, nivel: int, cab: Cabecalho, capitulo: No | None) -> str:
        if nivel == 0:
            self.cont = [0, 0, 0, 0]
        if cab.tipo in ("bloco", "avulso") or cab.estrela:
            return ""
        if nivel == 0:
            if self.apendice:
                self.n_ap += 1
                return chr(ord("A") + self.n_ap - 1)
            self.n_cap += 1
            return str(self.n_cap)
        self.cont[nivel] += 1
        for i in range(nivel + 1, 4):
            self.cont[i] = 0
        partes = ".".join(str(self.cont[i]) for i in range(1, nivel + 1))
        prefixo = capitulo.numero if capitulo is not None else ""
        return f"{prefixo}.{partes}" if prefixo else partes

    def novo(self, cab: Cabecalho, arq: lp.Arquivo) -> No:
        nivel = max(cab.nivel, 0)
        while self.pilha and self.pilha[-1].nivel >= nivel:
            self.pilha.pop()
        pai = self.pilha[-1] if self.pilha else None
        no = No(
            id=len(self.nos),
            nivel=nivel,
            tipo=cab.tipo,
            titulo=cab.titulo,
            arquivo=arq.rel,
            pos=cab.pos,
            linha_ini=arq.linha(cab.pos),
            pai=pai.id if pai else None,
            apendice=self.apendice,
            secao_bloco=cab.secao,
        )
        no.capitulo = no.id if pai is None else pai.capitulo
        capitulo = None if pai is None else self.nos[pai.capitulo]
        no.numero = self._numerar(nivel, cab, capitulo)
        if pai is not None:
            pai.filhos.append(no.id)
        self.nos.append(no)
        self.pilha.append(no)
        return no

    # -- peças ---------------------------------------------------------------
    def _cabe(self, no: No, arq: lp.Arquivo) -> bool:
        """O trecho de ``arq`` continua o nó atual? Sim se for do mesmo arquivo ou satélite.

        Satélite: arquivo de ``tables/`` ou subarquivo sem nenhum cabeçalho (que não é a raiz).
        """
        if no.arquivo == arq.rel or _satelite(arq.rel):
            return True
        raiz = self.arvore.visitas[0][0].rel if self.arvore.visitas else ""
        return arq.rel != raiz and not self.cabs(arq)

    def pedaco(self, arq: lp.Arquivo, a: int, b: int) -> None:
        if b <= a:
            return
        trecho = arq.limpo[a:b]
        no = self.atual
        if no is None or not self._cabe(no, arq):
            if _palavras(RE_SEM_CONTEUDO.sub(" ", trecho)) == 0:
                return
            pos = a + (len(trecho) - len(trecho.lstrip()))
            no = self.atual = self.novo(Cabecalho(pos, 0, f"[{arq.rel}]", "avulso"), arq)
        self._adicionar(no, arq, a, b)

    def _adicionar(self, no: No, arq: lp.Arquivo, a: int, b: int) -> None:
        no.arquivos.add(arq.rel)
        self.dono.setdefault(arq.rel, no.capitulo)
        no.pecas.append((arq.rel, arq.linha(a), arq.linha(b - 1)))
        starts, intervalos = self._paras(arq)
        i = bisect.bisect_left(starts, a)
        while i < len(intervalos) and intervalos[i][0] < b:
            pa, pb = intervalos[i]
            n = _palavras(arq.prosa[pa : min(pb, b)])
            if n:
                no.paragrafos += 1
                no.palavras += n
                if n > lp.PALAVRAS_MAX_PARAGRAFO:
                    no.paragrafos_longos += 1
            i += 1
        s = arq.limpo
        for m in RE_LABEL.finditer(s, a, b):
            rotulo = m.group(1).strip()
            if rotulo:
                no.rotulos.append({"rotulo": rotulo, "arquivo": arq.rel, "linha": arq.linha(m.start())})
        for m in RE_REF.finditer(s, a, b):
            for alvo in m.group(2).split(","):
                if alvo.strip():
                    no.referencias.append(
                        {"alvo": alvo.strip(), "comando": m.group(1), "arquivo": arq.rel, "linha": arq.linha(m.start())}
                    )
        for m in RE_HYPERREF.finditer(s, a, b):
            if m.group(1).strip():
                no.referencias.append(
                    {"alvo": m.group(1).strip(), "comando": "hyperref", "arquivo": arq.rel, "linha": arq.linha(m.start())}
                )
        for m in RE_GRAFICO.finditer(s, a, b):
            no.figuras_tabelas.append(
                {"tipo": "figura", "alvo": m.group(1).strip(), "arquivo": arq.rel, "linha": arq.linha(m.start())}
            )
        for m in RE_PATH.finditer(s, a, b):
            caminho = m.group(1).strip()
            if caminho.startswith("data/"):
                no.artefatos.append({"caminho": caminho, "arquivo": arq.rel, "linha": arq.linha(m.start())})

    def _include(self, arq: lp.Arquivo, ini: int, alvo: str) -> None:
        tipo = _classificar_include(alvo)
        if tipo and self.atual is not None:
            self.atual.figuras_tabelas.append(
                {"tipo": tipo, "alvo": alvo.strip(), "arquivo": arq.rel, "linha": arq.linha(ini)}
            )

    # -- execução ----------------------------------------------------------
    def executar(self) -> list[No]:
        for arq, a, b in self.arvore.visitas:
            cursor = a
            for c in self.cabs(arq):
                if not (a <= c.pos < b):
                    continue
                self.pedaco(arq, cursor, c.pos)
                if c.nivel < 0:
                    self.apendice = True
                elif c.tipo == "avulso":
                    # Fim de Resumo/Abstract: o que vier depois só vira nó se tiver texto.
                    self.atual = None
                else:
                    self.atual = self.novo(c, arq)
                cursor = c.pos
            self.pedaco(arq, cursor, b)
            alvo = self.incs.get((arq.rel, b))
            if alvo is not None:
                self._include(arq, b, alvo)
        self._fechar_linhas()
        return self.nos

    def _fechar_linhas(self) -> None:
        for no in self.nos:
            arq = self.por_rel[no.arquivo]
            fim = arq.limpo.rstrip().count("\n") + 1
            for c in self.cabs(arq):
                if c.pos > no.pos and 0 <= c.nivel <= no.nivel:
                    fim = arq.linha(c.pos) - 1
                    break
            no.linha_fim = max(no.linha_ini, fim)

    def no_em(self, rel: str, linha: int) -> No | None:
        """Nó que contém ``rel:linha``; em empate de linha, o que começa mais tarde."""
        if self._pecas is None:
            self._pecas = {}
            for no in self.nos:
                for r, la, lb in no.pecas:
                    self._pecas.setdefault(r, []).append((la, no.id, lb))
        melhor: tuple[int, int] | None = None
        for la, nid, lb in self._pecas.get(rel, ()):
            if la <= linha <= lb and (melhor is None or (la, nid) > melhor):
                melhor = (la, nid)
        return self.nos[melhor[1]] if melhor else None


# ---------------------------------------------------------------------------
# Leitores das fontes externas
# ---------------------------------------------------------------------------
def _yaml(caminho: Path | None):
    if caminho is None or not Path(caminho).is_file():
        return None
    import yaml

    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def ler_registro(caminho: Path | None) -> list[dict]:
    dados = _yaml(caminho)
    if not dados:
        return []
    return [e for e in dados if isinstance(e, dict) and e.get("id")]


def ler_ledger(caminho: Path | None) -> list[dict]:
    if caminho is None or not Path(caminho).is_file():
        return []
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def ler_pendencias(caminho: Path | None) -> list[dict]:
    dados = _yaml(caminho)
    if not dados:
        return []
    return [p for p in dados.get("pendencias", []) if isinstance(p, dict)]


def ler_baseline(caminho: Path | None) -> dict[str, dict[str, int]]:
    """Contagens ``regra -> arquivo -> n`` do baseline do linter.

    Formato v2: ``{"violacoes": {fp: {"regra", "arquivo", "trecho"}}}`` (uma entrada por
    violação); formato v1: ``{"contagens": {regra: {arquivo: n}}}``.
    """
    if caminho is None or not Path(caminho).is_file():
        return {}
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    contagens: dict[str, dict[str, int]] = {}
    if "violacoes" in dados:
        for entrada in dados["violacoes"].values():
            regra, arquivo = entrada.get("regra"), entrada.get("arquivo")
            if regra and arquivo:
                por_arq = contagens.setdefault(regra, {})
                por_arq[arquivo] = por_arq.get(arquivo, 0) + 1
        return contagens
    for regra, por_arq in dados.get("contagens", {}).items():
        contagens[regra] = {arq: int(n) for arq, n in por_arq.items()}
    return contagens


def _relativo(caminho: Path, base: Path) -> str:
    try:
        return caminho.relative_to(base).as_posix()
    except ValueError:
        return caminho.as_posix()


def _chave_registro(eid: str) -> tuple[bool, str]:
    return (not eid.startswith("N"), eid)


# ---------------------------------------------------------------------------
# Montagem do índice
# ---------------------------------------------------------------------------
def _atribuir_citacoes(
    cons: _Construtor, registros: list[dict]
) -> tuple[dict[str, No], dict[tuple[str, str], No], list[dict]]:
    """Conta as citações por nó; devolve ``id -> nó``, ``(arquivo, chave) -> nó`` e as sem nó."""
    por_id: dict[str, No] = {}
    por_chave: dict[tuple[str, str], No] = {}
    sem_no: list[dict] = []
    for r in registros:
        locais = [(r["file"], r["line"])]
        for o in r.get("outras_ocorrencias", []):
            arq, _, lin = o.rpartition(":")
            if lin.isdigit():
                locais.append((arq, int(lin)))
        for i, (arq, lin) in enumerate(locais):
            no = cons.no_em(arq, lin)
            if no is None:
                sem_no.append({"chave": r["key"], "arquivo": arq, "linha": lin})
                continue
            no.citacoes[r["key"]] += 1
            por_chave.setdefault((arq, r["key"]), no)
            if i == 0:
                por_id[r["id"]] = no
    return por_id, por_chave, sem_no


def construir_indice(
    tese: Path,
    base: Path | None = None,
    registro: Path | None = None,
    ledger: Path | None = None,
    pendencias: Path | None = None,
    baseline: Path | None = None,
) -> dict:
    """Constrói o índice completo (estrutura serializável em JSON)."""
    tese = Path(tese).resolve()
    base = (Path(base) if base is not None else ROOT).resolve()
    arvore = lp.carregar_arvore(tese, base)
    cons = _Construtor(arvore, base)
    nos = cons.executar()
    documento = cc.build_document(tese, base)
    registros_cit = cc.extract_citations(documento)
    por_id, por_chave, cit_sem_no = _atribuir_citacoes(cons, registros_cit)

    # --- atestação -----------------------------------------------------------
    ledger_linhas = ler_ledger(ledger)
    atestacao_sem_secao = 0
    for linha in ledger_linhas:
        no = por_id.get(linha.get("id", "")) or por_chave.get(
            (linha.get("arquivo", ""), linha.get("chave", ""))
        )
        if no is None:
            atestacao_sem_secao += 1
            continue
        no.atestacao[linha.get("veredito_final", "") or "(vazio)"] += 1
    pend_abertas: list[dict] = []
    pend_sem_secao: list[dict] = []
    for p in ler_pendencias(pendencias):
        if (p.get("decisao") or "pendente") != "pendente":
            continue
        no = por_id.get(str(p.get("id", ""))) or por_chave.get((p.get("arquivo", ""), p.get("chave", "")))
        item = {"id": str(p.get("id", "")), "chave": p.get("chave", ""), "arquivo": p.get("arquivo", "")}
        if no is None:
            pend_sem_secao.append(item)
        else:
            no.pendencias.append({"id": item["id"], "chave": item["chave"]})
            pend_abertas.append({**item, "no": no.id})

    # --- registro ------------------------------------------------------------
    entradas = ler_registro(registro)
    reg_arquivo: dict[str, set[str]] = {}
    reg_secao: dict[tuple[str, str], set[str]] = {}
    reg_tabela: dict[str, dict] = {}
    for e in entradas:
        eid = str(e["id"])
        ocs: set[str] = set()
        for oc in e.get("ocorrencias") or []:
            arq, sec = oc.get("arquivo", ""), oc.get("secao")
            if not arq:
                continue
            if sec:
                reg_secao.setdefault((arq, sec), set()).add(eid)
                ocs.add(f"{arq} [{sec}]")
            else:
                reg_arquivo.setdefault(arq, set()).add(eid)
                ocs.add(arq)
        reg_tabela[eid] = {
            "descricao": str(e.get("descricao", "")),
            "tipo": str(e.get("tipo", "")),
            "ocorrencias": sorted(ocs),
            "capitulos": [],
        }

    def registro_do_no(no: No) -> set[str]:
        if no.tipo == "bloco":
            return set(reg_secao.get((no.arquivo, no.secao_bloco), ()))
        return set(reg_arquivo.get(no.arquivo, ()))

    # --- linter --------------------------------------------------------------
    contagens = ler_baseline(baseline)
    linter_arq: dict[str, dict[str, int]] = {}
    for regra, por_arq in contagens.items():
        for arq, n in por_arq.items():
            linter_arq.setdefault(arq, {})[regra] = int(n)
    regras = sorted(contagens)

    # --- totais por nó (subárvore) ---------------------------------------------
    tot = {
        n.id: {"palavras": n.palavras, "paragrafos": n.paragrafos, "longos": n.paragrafos_longos,
               "citacoes": Counter(n.citacoes), "atestacao": Counter(n.atestacao), "pendencias": len(n.pendencias)}
        for n in nos
    }
    for n in reversed(nos):
        if n.pai is not None:
            p, t = tot[n.pai], tot[n.id]
            p["palavras"] += t["palavras"]
            p["paragrafos"] += t["paragrafos"]
            p["longos"] += t["longos"]
            p["citacoes"].update(t["citacoes"])
            p["atestacao"].update(t["atestacao"])
            p["pendencias"] += t["pendencias"]

    # --- rótulos e referências (índice reverso) --------------------------------
    rotulos: dict[str, dict] = {}
    for n in nos:
        for r in n.rotulos:
            rotulos.setdefault(r["rotulo"], {"definido": [], "referenciado_em": []})["definido"].append(
                {"arquivo": r["arquivo"], "linha": r["linha"], "no": n.id}
            )
    refs_sem_rotulo: dict[str, list[dict]] = {}
    for n in nos:
        for r in n.referencias:
            ref = {"arquivo": r["arquivo"], "linha": r["linha"], "comando": r["comando"], "no": n.id}
            if r["alvo"] in rotulos:
                rotulos[r["alvo"]]["referenciado_em"].append(ref)
            else:
                refs_sem_rotulo.setdefault(r["alvo"], []).append(ref)
    rotulos_ord = {k: rotulos[k] for k in sorted(rotulos)}
    sem_referencia = [k for k, v in rotulos_ord.items() if not v["referenciado_em"]]
    refs_sem_rotulo_ord = {k: refs_sem_rotulo[k] for k in sorted(refs_sem_rotulo)}

    # --- nós serializados ----------------------------------------------------
    lista_nos: list[dict] = []
    for n in nos:
        t = tot[n.id]
        lista_nos.append(
            {
                "id": n.id,
                "nivel": n.nivel,
                "tipo": n.tipo,
                "numero": n.numero,
                "titulo": n.titulo,
                "rotulo": n.rotulo,
                "apendice": n.apendice,
                "arquivo": n.arquivo,
                "linha_ini": n.linha_ini,
                "linha_fim": n.linha_fim,
                "pai": n.pai,
                "capitulo": n.capitulo,
                "filhos": list(n.filhos),
                "arquivos": sorted(n.arquivos),
                "palavras": n.palavras,
                "paragrafos": n.paragrafos,
                "paragrafos_longos": n.paragrafos_longos,
                "palavras_total": t["palavras"],
                "paragrafos_longos_total": t["longos"],
                "rotulos": [r["rotulo"] for r in n.rotulos],
                "referencias": [
                    {"alvo": r["alvo"], "comando": r["comando"], "linha": r["linha"], "arquivo": r["arquivo"]}
                    for r in n.referencias
                ],
                "citacoes": {k: n.citacoes[k] for k in sorted(n.citacoes)},
                "figuras_tabelas": list(n.figuras_tabelas),
                "artefatos": list(n.artefatos),
                "registro": sorted(registro_do_no(n), key=_chave_registro),
                "atestacao": {k: n.atestacao[k] for k in sorted(n.atestacao)},
                "pendencias": list(n.pendencias),
            }
        )

    # --- capítulos -----------------------------------------------------------
    capitulos: list[dict] = []
    for n in nos:
        if n.nivel != 0:
            continue
        donos = sorted(a for a, c in cons.dono.items() if c == n.id)
        reg: set[str] = set()
        for a in donos:
            reg |= reg_arquivo.get(a, set())
        reg |= registro_do_no(n) if n.tipo == "bloco" else set()
        lint: Counter = Counter()
        for a in donos:
            lint.update(linter_arq.get(a, {}))
        t = tot[n.id]
        for eid in reg:
            if eid in reg_tabela:
                reg_tabela[eid]["capitulos"].append(n.rotulo)
        capitulos.append(
            {
                "no": n.id,
                "rotulo": n.rotulo,
                "arquivo": n.arquivo,
                "linha_ini": n.linha_ini,
                "linha_fim": n.linha_fim,
                "arquivos": donos,
                "palavras": t["palavras"],
                "paragrafos": t["paragrafos"],
                "paragrafos_longos": t["longos"],
                "citacoes": sum(t["citacoes"].values()),
                "chaves_distintas": len(t["citacoes"]),
                "atestacao": {k: t["atestacao"][k] for k in sorted(t["atestacao"])},
                "pendencias": t["pendencias"],
                "linter": {k: lint[k] for k in sorted(lint)},
                "linter_total": sum(lint.values()),
                "registro": sorted(reg, key=_chave_registro),
            }
        )
    for v in reg_tabela.values():
        v["capitulos"] = sorted(set(v["capitulos"]))
    total = {
        "palavras": sum(c["palavras"] for c in capitulos),
        "paragrafos": sum(c["paragrafos"] for c in capitulos),
        "paragrafos_longos": sum(c["paragrafos_longos"] for c in capitulos),
        "citacoes": sum(c["citacoes"] for c in capitulos),
        "pendencias": sum(c["pendencias"] for c in capitulos),
        "linter_total": sum(c["linter_total"] for c in capitulos),
        "registro": len({e for c in capitulos for e in c["registro"]}),
    }

    # --- arquivos ----------------------------------------------------------
    arquivos: dict[str, dict] = {}
    for rel in sorted(cons.dono):
        arquivos[rel] = {
            "capitulo": nos[cons.dono[rel]].rotulo,
            "linter": {k: v for k, v in sorted(linter_arq.get(rel, {}).items())},
            "registro": sorted(reg_arquivo.get(rel, set()), key=_chave_registro),
        }
    fora = {a: dict(sorted(v.items())) for a, v in sorted(linter_arq.items()) if a not in arquivos}

    return {
        "versao": VERSAO,
        "tese": _relativo(tese, base),
        "resumo": {"capitulos": capitulos, "total": total},
        "nos": lista_nos,
        "rotulos": rotulos_ord,
        "rotulos_sem_referencia": sem_referencia,
        "referencias_sem_rotulo": refs_sem_rotulo_ord,
        "registro": {k: reg_tabela[k] for k in sorted(reg_tabela, key=_chave_registro)},
        "atestacao": {
            "pendencias_abertas": sorted(pend_abertas, key=lambda p: (p["no"], p["chave"], p["id"])),
            "pendencias_sem_secao": sorted(pend_sem_secao, key=lambda p: (p["arquivo"], p["chave"], p["id"])),
            "ledger_sem_secao": atestacao_sem_secao,
        },
        "linter": {"regras": regras, "arquivos": arquivos, "fora_da_arvore": fora},
        "citacoes_sem_secao": sorted(cit_sem_no, key=lambda c: (c["arquivo"], c["linha"], c["chave"])),
        "avisos": sorted(set(arvore.avisos) | set(documento.warnings)),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _loc(arquivo: str, a: int, b: int | None = None) -> str:
    return f"`{arquivo}:{a}-{b}`" if b is not None and b != a else f"`{arquivo}:{a}`"


def _contagens(d: dict, sep: str = ", ") -> str:
    return sep.join(f"{k}={v}" for k, v in d.items())


def _citacoes_txt(d: dict) -> str:
    return ", ".join(f"{k}" if v == 1 else f"{k} ×{v}" for k, v in d.items())


def _referencias_txt(refs: list[dict]) -> str:
    c = Counter(r["alvo"] for r in refs)
    return ", ".join(f"{k}" if c[k] == 1 else f"{k} ×{c[k]}" for k in sorted(c))


def _cel(texto: str) -> str:
    return texto.replace("|", "\\|").replace("\n", " ")


def _linhas_no(n: dict, titulo: str | None = None) -> list[str]:
    """Bullet do nó e, abaixo, só as categorias não vazias (capítulo e seção ficam no mesmo recuo)."""
    ind = "  " * max(n["nivel"] - 1, 0)
    cab = f"{ind}- **{titulo or n['rotulo']}** {_loc(n['arquivo'], n['linha_ini'], n['linha_fim'])}"
    cab += f" (palavras {n['palavras']}; parágrafos {n['paragrafos']}; > 200 palavras: {n['paragrafos_longos']})"
    saida = [cab]
    sub = ind + "  - "
    if n["rotulos"]:
        saida.append(f"{sub}rótulos: {', '.join(n['rotulos'])}")
    if n["referencias"]:
        saida.append(f"{sub}referências: {_referencias_txt(n['referencias'])}")
    if n["citacoes"]:
        saida.append(f"{sub}citações: {_citacoes_txt(n['citacoes'])}")
    if n["figuras_tabelas"]:
        itens = [f"{f['tipo']} `{f['alvo']}`" for f in n["figuras_tabelas"]]
        saida.append(f"{sub}figuras e tabelas: {'; '.join(itens)}")
    if n["artefatos"]:
        saida.append(f"{sub}artefatos: {', '.join('`' + a['caminho'] + '`' for a in n['artefatos'])}")
    if n["atestacao"]:
        saida.append(f"{sub}atestação: {_contagens(n['atestacao'])}")
    if n["pendencias"]:
        itens = [f"{p['chave']} ({p['id']})" for p in n["pendencias"]]
        saida.append(f"{sub}pendências abertas: {'; '.join(itens)}")
    return saida


def formatar_markdown(indice: dict) -> str:
    nos = indice["nos"]
    rotulo_de = {n["id"]: n["rotulo"] for n in nos}
    caps = indice["resumo"]["capitulos"]
    regras = indice["linter"]["regras"]
    L: list[str] = []
    L.append("# Índice da tese")
    L.append("")
    L.append(
        f"Gerado por `scripts/indice_tese.py` a partir de `{indice['tese']}`. Não editar à mão. "
        "Palavras e parágrafos: texto direto de cada nó; os totais do capítulo somam as seções. "
        "Registro e linter valem por arquivo."
    )
    L.append("")
    L.append("## Resumo por capítulo")
    L.append("")
    L.append("| Capítulo | Arquivo | Palavras | Parágrafos > 200 | Citações | Pendências | Linter | Registro |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in caps:
        L.append(
            f"| {_cel(c['rotulo'])} | {_loc(c['arquivo'], c['linha_ini'], c['linha_fim'])} | {c['palavras']} | "
            f"{c['paragrafos_longos']} | {c['citacoes']} | {c['pendencias']} | {c['linter_total']} | "
            f"{len(c['registro'])} |"
        )
    t = indice["resumo"]["total"]
    L.append(
        f"| **Total** | | {t['palavras']} | {t['paragrafos_longos']} | {t['citacoes']} | {t['pendencias']} | "
        f"{t['linter_total']} | {t['registro']} |"
    )
    L.append("")
    L.append("## Estrutura")
    L.append("")
    cap_por_no = {c["no"]: c for c in caps}
    for n in nos:
        if n["nivel"] == 0:
            c = cap_por_no[n["id"]]
            L.append(f"### {n['rotulo']} {_loc(n['arquivo'], n['linha_ini'], n['linha_fim'])}")
            L.append("")
            L.append(
                f"- Totais: palavras {c['palavras']}; parágrafos {c['paragrafos']} "
                f"(> 200 palavras: {c['paragrafos_longos']}); citações {c['citacoes']} "
                f"({c['chaves_distintas']} chaves); pendências abertas {c['pendencias']}"
            )
            if c["arquivos"]:
                L.append(f"- Arquivos lidos: {', '.join('`' + a + '`' for a in c['arquivos'])}")
            if c["registro"]:
                L.append(f"- Registro de invariantes (por arquivo): {', '.join(c['registro'])}")
            if c["atestacao"]:
                L.append(f"- Atestação: {_contagens(c['atestacao'])}")
            if c["linter"]:
                L.append(f"- Linter: {_contagens(c['linter'])}")
            L.append("")
        L.extend(_linhas_no(n, "Texto direto do capítulo" if n["nivel"] == 0 else None))
    L.append("")
    L.append("## Rótulos e referências (índice reverso)")
    L.append("")

    def onde(itens: list[dict]) -> str:
        return "; ".join(f"`{i['arquivo']}:{i['linha']}` ({rotulo_de[i['no']]})" for i in itens)

    L.append("| Rótulo | Definido em | Referenciado em |")
    L.append("|---|---|---|")
    for k, v in indice["rotulos"].items():
        L.append(f"| `{k}` | {onde(v['definido'])} | {onde(v['referenciado_em']) or '-'} |")
    L.append("")
    if indice["rotulos_sem_referencia"]:
        L.append(f"Rótulos sem referência ({len(indice['rotulos_sem_referencia'])}): "
                 + ", ".join(f"`{k}`" for k in indice["rotulos_sem_referencia"]))
        L.append("")
    if indice["referencias_sem_rotulo"]:
        L.append("Referências sem rótulo definido:")
        L.append("")
        for k, v in indice["referencias_sem_rotulo"].items():
            L.append(f"- `{k}`: {onde(v)}")
        L.append("")
    L.append("## Registro de invariantes")
    L.append("")
    L.append("| Entrada | Descrição | Ocorrências no registro | Capítulos que as contêm |")
    L.append("|---|---|---|---|")
    for k, v in indice["registro"].items():
        L.append(
            f"| {k} | {_cel(v['descricao'])} | {', '.join('`' + o + '`' for o in v['ocorrencias']) or '-'} | "
            f"{'; '.join(v['capitulos']) or '-'} |"
        )
    L.append("")
    L.append("## Atestação bibliográfica")
    L.append("")
    vereditos = sorted({k for c in caps for k in c["atestacao"]})
    L.append("| Capítulo | " + " | ".join(vereditos + ["Pendências abertas"]) + " |")
    L.append("|---|" + "---:|" * (len(vereditos) + 1))
    for c in caps:
        L.append(
            f"| {_cel(c['rotulo'])} | " + " | ".join(str(c["atestacao"].get(v, 0)) for v in vereditos)
            + f" | {c['pendencias']} |"
        )
    L.append("")
    at = indice["atestacao"]
    L.append("Pendências abertas (id, chave, seção):")
    L.append("")
    if not at["pendencias_abertas"] and not at["pendencias_sem_secao"]:
        L.append("- nenhuma")
    for p in at["pendencias_abertas"]:
        L.append(f"- `{p['id']}` {p['chave']}: {rotulo_de[p['no']]} (`{p['arquivo']}`)")
    for p in at["pendencias_sem_secao"]:
        L.append(f"- `{p['id']}` {p['chave']}: sem seção localizada (`{p['arquivo']}`)")
    if at["ledger_sem_secao"]:
        L.append("")
        L.append(f"Linhas do ledger sem seção localizada: {at['ledger_sem_secao']}")
    L.append("")
    L.append("## Linter de prosa (por arquivo)")
    L.append("")
    L.append("| Arquivo | Capítulo | " + " | ".join(regras) + " | Total |")
    L.append("|---|---|" + "---:|" * (len(regras) + 1))
    for arq, v in indice["linter"]["arquivos"].items():
        cont = v["linter"]
        L.append(
            f"| `{arq}` | {_cel(v['capitulo'])} | " + " | ".join(str(cont.get(r, 0)) for r in regras)
            + f" | {sum(cont.values())} |"
        )
    if indice["avisos"]:
        L.append("")
        L.append("## Avisos")
        L.append("")
        for a in indice["avisos"]:
            L.append(f"- {a}")
    return "\n".join(L).rstrip() + "\n"


def _escrever(caminho: Path, texto: str) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="\n") as f:
        f.write(texto)


def serializar_json(indice: dict) -> str:
    return json.dumps(indice, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Índice de navegação da tese")
    ap.add_argument("--tese", type=Path, default=TESE_PADRAO)
    ap.add_argument("--base", type=Path, default=ROOT, help="raiz dos caminhos relativos")
    ap.add_argument("--saida", type=Path, default=SAIDA_MD_PADRAO)
    ap.add_argument("--json", type=Path, default=SAIDA_JSON_PADRAO, dest="saida_json")
    ap.add_argument("--registro", type=Path, default=REGISTRO_PADRAO)
    ap.add_argument("--ledger", type=Path, default=LEDGER_PADRAO)
    ap.add_argument("--pendencias", type=Path, default=PENDENCIAS_PADRAO)
    ap.add_argument("--baseline", type=Path, default=BASELINE_PADRAO)
    args = ap.parse_args(argv)
    indice = construir_indice(
        args.tese, args.base, args.registro, args.ledger, args.pendencias, args.baseline
    )
    _escrever(args.saida, formatar_markdown(indice))
    _escrever(args.saida_json, serializar_json(indice))
    t = indice["resumo"]["total"]
    print(
        f"{len(indice['resumo']['capitulos'])} capítulos, {len(indice['nos'])} nós, "
        f"{t['palavras']} palavras, {t['citacoes']} citações, {len(indice['rotulos'])} rótulos"
    )
    print(f"markdown: {args.saida}")
    print(f"json: {args.saida_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
