"""Seleção de trechos por relevância para evidências mais longas que ``--max-chars`` (sem rede, sem API).

Em vez de cortar o texto completo no início, o texto é dividido em trechos de cerca de 1.500 caracteres
(sobreposição de 200), cada trecho é pontuado por BM25 contra uma consulta montada a partir de cada
contexto da tese que cita a fonte, e os melhores trechos de todos os contextos da chave entram, em
rodízio, até ``max_chars``. O que vai ao modelo é: os 3.000 primeiros caracteres (título, resumo,
folha de rosto) mais os trechos escolhidos, em ordem de documento, cada um com marcador
``[trecho 12, p. 184]`` (``p.`` é a página do PDF; sem páginas, ``[trecho 12]``).

Consulta de cada contexto (pesos em ``W_*``):

(a) fichas (tokens) da sentença e do parágrafo da tese;
(b) fichas "invariantes" extraídas por regra: números (formato brasileiro e inglês casam, ver
    ``tokenize``), siglas, nomes próprios e termos técnicos em inglês já presentes no texto em português
    (SMOTE, Random Forest, blocking, Fellegi-Sunter);
(c) tradução da sentença para o inglês e 5 a 12 palavras-chave em inglês, geradas uma vez pelo
    subcomando ``consultas`` (Claude Haiku, Message Batches) e guardadas por id de contexto em
    ``<out>/evidencias/_consultas_en.json``. Sem a consulta em inglês vale só (a) + (b).

O índice de trechos de cada chave fica em ``<out>/evidencias/<chave>.chunks.json`` (chaveado pelo hash do
texto e dos parâmetros). A seleção é determinística: mesma entrada, mesmo texto enviado, mesma
impressão digital; se contextos, consultas ou fonte mudam, a seleção e a impressão digital mudam.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .common import key_filename, read_json, strip_accents, write_json_atomic, write_text_atomic
from .numeros import extract_numbers

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
FRONT_CHARS = 3000
FRONT_MARKER = "[início do documento]"
INDEX_VERSION = 1
BM25_K1 = 1.5
BM25_B = 0.75

W_PARAGRAPH = 0.5
W_SENTENCE = 1.0
W_INVARIANT_PARAGRAPH = 1.5
W_INVARIANT_SENTENCE = 2.0
W_EN_TEXT = 1.5
W_EN_KEYWORDS = 2.5

_PT_STOP = """a o as os um uma uns umas de do da dos das em no na nos nas por para pelo pela pelos pelas com sem sob
sobre entre ate apos ante e ou mas nem que se como quando onde qual quais quem cujo cuja ao aos a as ser sao foi
foram era eram sera ter tem tinha tinham ha havia estar esta estao estava este esta estes estas esse essa esses
essas isto isso aquilo aquele aquela seu sua seus suas meu minha nosso nossa ele ela eles elas nao mais menos muito
muitos muita muitas pouco tambem ja so apenas ainda cada outro outra outros outras mesmo mesma entao assim porque
pois logo deve devem pode podem sendo sido vez vezes partir segundo conforme acerca todo toda todos todas
cit ref""".split()
_EN_STOP = """the a an of in on at to for from by with without about into over under between and or but nor not no
is are was were be been being am do does did done has have had having it its this that these those as than then
there their they them he she his her we our you your which who whom whose what when where why how all any each
both few more most other some such only own same so too very can could should would may might will shall also
i e g et al fig table""".split()
STOPWORDS = frozenset(_PT_STOP) | frozenset(_EN_STOP)

# Termos técnicos em inglês que aparecem no texto em português da tese.
EN_TERMS = frozenset(
    """blocking linkage matching threshold boosting bagging overfitting underfitting recall precision score
    oversampling undersampling embedding transformer prompt prompting fine tuning dedup deduplication
    benchmark baseline pipeline cutoff random forest cross validation gradient learning machine deep
    probabilistic deterministic clustering classifier feature features""".split()
)

_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
_MARKUP = re.compile(r"\[(?:cit|ref)\]")
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[a-z]*\d*|[A-Z]+\d+|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b")
_CAPITALIZED = re.compile(r"\b[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]{2,}")
_WORD = re.compile(r"[^\W\d_]+")


# ---------------------------------------------------------------------------
# Fichas
# ---------------------------------------------------------------------------
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Fichas comparáveis entre português e inglês.

    Palavras: sem acento, minúsculas, sem stopwords (pt e en), plural simples removido. Números com ao
    menos dois dígitos viram ``n<dígitos>`` (separadores descartados): ``39,3`` e ``39.3`` dão ``n393``;
    ``84.368`` e ``84,368`` dão ``n84368``."""
    out: list[str] = []
    for match in _TOKEN.finditer(_MARKUP.sub(" ", text)):
        token = match.group(0)
        if token[0].isdigit():
            digits = re.sub(r"\D", "", token)
            if len(digits) >= 2:
                out.append("n" + digits)
            continue
        word = strip_accents(token).casefold()
        if len(word) < 2 or word in STOPWORDS:
            continue
        out.append(_stem(word))
    return out


def _looks_english(word: str) -> bool:
    """Palavra sem acento que lembra inglês (k, w, y, th, ck ou sufixo -ing), inexistentes no português."""
    return len(word) >= 4 and (
        any(c in word for c in "kwy") or "th" in word or "ck" in word or (len(word) > 5 and word.endswith("ing"))
    )


def invariant_tokens(text: str) -> set[str]:
    """Fichas que sobrevivem à tradução: números, siglas, nomes próprios e termos técnicos em inglês."""
    clean = _MARKUP.sub(" ", text)
    found: set[str] = set()
    for _text, integer, decimals in extract_numbers(clean):
        found.add("n" + integer + decimals)
    for match in _ACRONYM.finditer(clean):
        found.update(tokenize(match.group(0)))
    for match in _CAPITALIZED.finditer(clean):
        before = clean[: match.start()].rstrip()
        if before and before[-1] not in ".!?:;":
            found.update(tokenize(match.group(0)))
    for word in _WORD.findall(clean):
        plain = strip_accents(word).casefold()
        if plain in STOPWORDS:
            continue
        if plain in EN_TERMS or _looks_english(plain):
            found.update(tokenize(plain))
    return found


def build_query(ctx: dict, english: dict | None = None) -> dict[str, float]:
    """Consulta de um contexto, como mapa ficha para peso: (a) sentença e parágrafo, (b) invariantes,
    (c) tradução e palavras-chave em inglês, se houver. O peso é o maior entre as origens."""
    query: dict[str, float] = {}

    def put(tokens, weight: float) -> None:
        for token in tokens:
            if weight > query.get(token, 0.0):
                query[token] = weight

    sentence = ctx.get("sentence_text", "") or ""
    paragraph = ctx.get("paragraph_text", "") or ""
    put(tokenize(paragraph), W_PARAGRAPH)
    put(tokenize(sentence), W_SENTENCE)
    put(invariant_tokens(paragraph), W_INVARIANT_PARAGRAPH)
    put(invariant_tokens(sentence), W_INVARIANT_SENTENCE)
    if english:
        put(tokenize(str(english.get("traducao") or "")), W_EN_TEXT)
        for keyword in english.get("palavras_chave") or []:
            put(tokenize(str(keyword)), W_EN_KEYWORDS)
    return query


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def bm25_scores(docs: list[Counter], query: dict[str, float]) -> list[float]:
    """BM25 (Okapi) de cada documento contra a consulta ponderada (peso multiplica o termo)."""
    n = len(docs)
    if n == 0:
        return []
    lengths = [sum(doc.values()) for doc in docs]
    average = (sum(lengths) / n) or 1.0
    scores = [0.0] * n
    for term, weight in query.items():
        df = sum(1 for doc in docs if term in doc)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(docs):
            tf = doc.get(term, 0)
            if tf:
                norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[i] / average)
                scores[i] += weight * idf * tf * (BM25_K1 + 1) / norm
    return scores


def rank_by_bm25(docs: list[Counter], query: dict[str, float]) -> list[int]:
    """Índices dos documentos com pontuação positiva, do melhor ao pior (empate: menor índice)."""
    scores = bm25_scores(docs, query)
    return [i for i in sorted(range(len(scores)), key=lambda j: (-scores[j], j)) if scores[i] > 0]


# ---------------------------------------------------------------------------
# Trechos (chunking) e índice em cache
# ---------------------------------------------------------------------------
def page_at(position: int, page_starts: list[int] | None) -> int | None:
    if not page_starts:
        return None
    return max(1, bisect.bisect_right(page_starts, position))


def chunk_spans(
    text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Intervalos ``(início, fim)`` de trechos de cerca de ``size`` caracteres com ``overlap`` de
    sobreposição; fim e início procuram um limite de palavra para não cortar palavras."""
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    slack = max(1, size // 10)
    while start < n:
        end = min(start + size, n)
        if end < n:
            for j in range(end, max(start + 1, end - slack), -1):
                if text[j - 1].isspace():
                    end = j
                    break
        spans.append((start, end))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        for j in range(nxt, min(end, nxt + slack)):
            if text[j - 1].isspace():
                nxt = j
                break
        start = min(nxt, end)
    return spans


def build_chunks(
    text: str,
    page_starts: list[int] | None = None,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Trechos numerados a partir de 1: ``{"i", "ini", "fim", "pagina", "pagina_fim"}``.

    A página é a do PDF em que o trecho começa (e termina); None quando a fonte não tem páginas."""
    chunks = []
    for i, (start, end) in enumerate(chunk_spans(text, size, overlap), start=1):
        chunks.append({
            "i": i, "ini": start, "fim": end,
            "pagina": page_at(start, page_starts), "pagina_fim": page_at(max(start, end - 1), page_starts),
        })
    return chunks


def chunks_hash(text: str, page_starts: list[int] | None, size: int, overlap: int) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(json.dumps([INDEX_VERSION, size, overlap, page_starts or []]).encode())
    return digest.hexdigest()[:16]


def chunks_path(out_dir: Path, key: str) -> Path:
    return out_dir / "evidencias" / f"{key_filename(key)}.chunks.json"


def load_chunks(
    text: str,
    key: str,
    out_dir: Path | None,
    page_starts: list[int] | None = None,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Índice de trechos da chave: do cache se o hash do texto e dos parâmetros confere, senão refeito."""
    digest = chunks_hash(text, page_starts, size, overlap)
    path = chunks_path(out_dir, key) if out_dir is not None else None
    if path is not None:
        stored = read_json(path)
        if isinstance(stored, dict) and stored.get("hash") == digest and stored.get("versao") == INDEX_VERSION:
            return [
                {"i": i, "ini": a, "fim": b, "pagina": p, "pagina_fim": q}
                for i, a, b, p, q in stored["trechos"]
            ]
    chunks = build_chunks(text, page_starts, size, overlap)
    if path is not None:
        payload = {
            "versao": INDEX_VERSION, "chave": key, "hash": digest, "tamanho_texto": len(text),
            "tamanho": size, "sobreposicao": overlap,
            "trechos": [[c["i"], c["ini"], c["fim"], c["pagina"], c["pagina_fim"]] for c in chunks],
        }
        write_text_atomic(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return chunks


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------
def front_end_of(text: str, front_chars: int = FRONT_CHARS) -> int:
    """Fim do início do documento (sempre enviado), recuado até um espaço para não cortar palavra."""
    if len(text) <= front_chars:
        return len(text)
    for j in range(front_chars, max(0, front_chars - 100), -1):
        if text[j - 1].isspace():
            return j
    return front_chars


def _marker(first: dict, last: dict) -> str:
    ids = f"{first['i']}" if first["i"] == last["i"] else f"{first['i']}-{last['i']}"
    label = f"trecho {ids}" if first["i"] == last["i"] else f"trechos {ids}"
    page, page_end = first.get("pagina"), last.get("pagina_fim")
    if page is None:
        return f"[{label}]"
    pages = f"{page}" if page == page_end or page_end is None else f"{page}-{page_end}"
    return f"[{label}, p. {pages}]"


def build_runs(chunks: list[dict], chosen: set[int], front_end: int) -> list[dict]:
    """Agrupa trechos consecutivos escolhidos (posições em ``chunks``) em blocos contínuos, sem repetir
    a sobreposição nem o início do documento. Cada bloco: ``{"marcador", "ini", "fim", "trechos"}``."""
    runs: list[dict] = []
    group: list[int] = []
    for pos in sorted(chosen) + [None]:
        if group and (pos is None or pos != group[-1] + 1):
            first, last = chunks[group[0]], chunks[group[-1]]
            runs.append({
                "marcador": _marker(first, last), "ini": max(first["ini"], front_end), "fim": last["fim"],
                "trechos": [chunks[g] for g in group],
            })
            group = []
        if pos is not None:
            group.append(pos)
    return runs


def rendered_length(runs: list[dict], front_end: int) -> int:
    """Tamanho do texto de evidência (início do documento + blocos), igual ao de ``render``."""
    total = len(FRONT_MARKER) + 1 + front_end
    for run in runs:
        total += 2 + len(run["marcador"]) + 1 + (run["fim"] - run["ini"])
    return total


def render(text: str, runs: list[dict], front_end: int) -> str:
    parts = [f"{FRONT_MARKER}\n{text[:front_end]}"]
    parts += [f"{run['marcador']}\n{text[run['ini']:run['fim']]}" for run in runs]
    return "\n\n".join(parts)


def select_chunk_positions(
    text: str, chunks: list[dict], queries: list[dict[str, float]], max_chars: int, front_end: int
) -> set[int]:
    """Posições dos trechos escolhidos: união dos melhores trechos de cada consulta, em rodízio, enquanto
    o texto renderizado (início + blocos) couber em ``max_chars``. Trechos dentro do início do documento
    não entram (já estão lá)."""
    candidates = [pos for pos, chunk in enumerate(chunks) if chunk["fim"] > front_end]
    if not candidates or not queries:
        return set()
    docs = [Counter(tokenize(text[chunks[pos]["ini"] : chunks[pos]["fim"]])) for pos in candidates]
    rankings = [[candidates[i] for i in rank_by_bm25(docs, query)] for query in queries]
    chosen: set[int] = set()
    pointers = [0] * len(rankings)
    active = [i for i, ranking in enumerate(rankings) if ranking]
    while active:
        still: list[int] = []
        for qi in active:
            ranking = rankings[qi]
            while pointers[qi] < len(ranking):
                pos = ranking[pointers[qi]]
                pointers[qi] += 1
                if pos in chosen:
                    continue
                trial = chosen | {pos}
                if rendered_length(build_runs(chunks, trial, front_end), front_end) <= max_chars:
                    chosen = trial
                    break
            if pointers[qi] < len(ranking):
                still.append(qi)
        active = still
    return chosen


@dataclass
class Selecao:
    texto: str  # início do documento + blocos marcados, como enviado
    trechos: list[dict]  # [{"trecho": 12, "pagina": 184}], em ordem de documento
    total_trechos: int
    consultas: int  # contextos que entraram na seleção
    consultas_en: int  # desses, os que tinham consulta em inglês


def select_evidence_text(
    text: str,
    key: str,
    contexts: list[dict],
    english: dict[str, dict] | None,
    max_chars: int,
    out_dir: Path | None = None,
    page_starts: list[int] | None = None,
) -> Selecao | None:
    """Texto selecionado para a chave ou None se a seleção não é possível (sem contextos, ou
    ``max_chars`` menor que o início do documento mais um trecho)."""
    if not contexts or max_chars < FRONT_CHARS + CHUNK_CHARS:
        return None
    english = english or {}
    ordered = sorted(contexts, key=lambda c: (c.get("file", ""), c.get("line", 0), c["id"]))
    queries = [build_query(c, english.get(c["id"])) for c in ordered]
    chunks = load_chunks(text, key, out_dir, page_starts)
    front_end = front_end_of(text)
    chosen = select_chunk_positions(text, chunks, queries, max_chars, front_end)
    runs = build_runs(chunks, chosen, front_end)
    sent = [
        {"trecho": chunk["i"], "pagina": chunk["pagina"]}
        for run in runs
        for chunk in run["trechos"]
    ]
    return Selecao(
        render(text, runs, front_end), sent, len(chunks), len(ordered),
        sum(1 for c in ordered if english.get(c["id"])),
    )


# ---------------------------------------------------------------------------
# Consultas em inglês (cache por id de contexto)
# ---------------------------------------------------------------------------
def english_queries_path(out_dir: Path) -> Path:
    return out_dir / "evidencias" / "_consultas_en.json"


def load_english_queries(out_dir: Path | None) -> dict[str, dict]:
    if out_dir is None:
        return {}
    data = read_json(english_queries_path(out_dir), default={})
    return data if isinstance(data, dict) else {}


def save_english_queries(out_dir: Path, queries: dict[str, dict]) -> None:
    write_json_atomic(english_queries_path(out_dir), queries)
