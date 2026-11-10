"""Etapa 2: recuperação de evidências em fontes abertas, com checagem de identidade.

Antes da rede, usa-se o texto completo fornecido pelo autor em ``.local/fontes/<chave>.<ext>``
(``find_local_fulltext``); havendo texto local, as fontes de texto completo da rede são puladas
(o resumo do OpenAlex continua sendo buscado).

Ordem do texto completo (somente acesso aberto): Europe PMC (PMCID), SciELO (DOI 10.1590),
Unpaywall (melhor local aberto, HTML ou PDF) e arXiv. Antes de aceitar qualquer resumo ou
texto, o título da fonte tem de ser equivalente ao do .bib (similaridade >= 0,85) ou o DOI
tem de coincidir exatamente. Nunca se consulta biblioteca-sombra (ver ``http.py``).
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

from scripts.citacoes_contexto import BibEntry, parse_bibtex, split_authors

from .common import (
    clean_latex,
    key_filename,
    normalize_doi,
    normalize_title,
    read_json,
    strip_accents,
    title_similarity,
    utc_now_iso,
    write_json_atomic,
    write_text_atomic,
)
from .http import HttpClient, HttpError

OPENALEX_WORKS = "https://api.openalex.org/works"
EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf"

IDENTITY_MIN_SIM = 0.85
SEARCH_MIN_SIM = 0.90
MIN_FULLTEXT_CHARS = 3000
HEAD_CHARS = 15000

LITERATURA_TYPES = {"article", "inproceedings", "book", "incollection"}
LEVELS = ("texto_completo", "resumo", "consensus", "nenhum")


# ---------------------------------------------------------------------------
# Referências do .bib e classificação
# ---------------------------------------------------------------------------
@dataclass
class BibRef:
    key: str
    entry_type: str
    title: str
    author_raw: str
    first_surname: str
    year: int | None
    doi: str
    url: str


def first_author_surname(author_raw: str) -> str:
    names = split_authors(author_raw)
    if not names:
        return ""
    first = names[0].strip()
    if first.startswith("{") and first.endswith("}"):
        return clean_latex(first)
    if "," in first:
        return clean_latex(first.split(",", 1)[0])
    tokens = clean_latex(first).split()
    return tokens[-1] if tokens else ""


def bib_ref(entry: BibEntry) -> BibRef:
    fields = entry.fields
    year_raw = fields.get("year") or fields.get("date", "")[:4]
    match = re.search(r"\d{4}", year_raw or "")
    author_raw = fields.get("author") or fields.get("editor") or fields.get("organization") or ""
    return BibRef(
        key=entry.key,
        entry_type=entry.entry_type,
        title=clean_latex(fields.get("title", "")),
        author_raw=author_raw,
        first_surname=first_author_surname(author_raw),
        year=int(match.group(0)) if match else None,
        doi=normalize_doi(fields.get("doi", "")),
        url=clean_latex(fields.get("url", "")),
    )


def load_bib_refs(path: Path) -> dict[str, BibRef]:
    entries, _warnings = parse_bibtex(path.read_text(encoding="utf-8-sig"))
    return {key: bib_ref(entry) for key, entry in entries.items()}


def classify_key(ref: BibRef | None, proprios: set[str] | frozenset[str] = frozenset()) -> str:
    """``proprio`` (trabalho do autor), ``literatura`` ou ``cinzenta``."""
    if ref is None:
        return "cinzenta"
    if ref.key in proprios or "jardim" in strip_accents(ref.author_raw).casefold():
        return "proprio"
    if ref.doi or ref.entry_type in LITERATURA_TYPES:
        return "literatura"
    return "cinzenta"


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Texto do resumo a partir do ``abstract_inverted_index`` do OpenAlex."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, places in inverted_index.items():
        for place in places:
            positions[place] = word
    return " ".join(positions[p] for p in sorted(positions))


@dataclass
class WorkView:
    title: str
    doi: str
    year: int | None
    first_author: str
    abstract: str
    pmcid: str
    arxiv_id: str
    landing_url: str


_PMCID_RE = re.compile(r"PMC\d+", re.IGNORECASE)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:\.pdf)?(?:v\d+)?(?:[?#]|$)", re.IGNORECASE)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)$", re.IGNORECASE)


def work_view(work: dict) -> WorkView:
    ids = work.get("ids") or {}
    pmc = _PMCID_RE.search(str(ids.get("pmcid") or ""))
    authorships = work.get("authorships") or []
    first_author = ""
    if authorships:
        first_author = ((authorships[0].get("author") or {}).get("display_name")) or ""
    arxiv_id = ""
    doi = normalize_doi(work.get("doi"))
    from_doi = _ARXIV_DOI_RE.match(doi)
    if from_doi:
        arxiv_id = from_doi.group(1)
    locations = list(work.get("locations") or [])
    for loc in locations:
        match = _ARXIV_RE.search(str((loc or {}).get("landing_page_url") or ""))
        if match and not arxiv_id:
            arxiv_id = match.group(1)
    primary = work.get("primary_location") or {}
    return WorkView(
        title=work.get("title") or work.get("display_name") or "",
        doi=doi,
        year=work.get("publication_year"),
        first_author=first_author,
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        pmcid=pmc.group(0).upper() if pmc else "",
        arxiv_id=arxiv_id,
        landing_url=str(primary.get("landing_page_url") or work.get("doi") or ""),
    )


def surname_matches(surname: str, display_name: str) -> bool:
    if not surname or not display_name:
        return False
    a = normalize_title(surname)
    b = normalize_title(display_name)
    return bool(a) and re.search(rf"\b{re.escape(a)}\b", b) is not None


def accept_search_candidate(view: WorkView, ref: BibRef) -> tuple[bool, float]:
    """Aceita o resultado de busca por título: similaridade >= 0,90, ano ±1 e sobrenome."""
    sim = title_similarity(ref.title, view.title)
    if sim < SEARCH_MIN_SIM:
        return False, sim
    if ref.year is None or view.year is None or abs(ref.year - int(view.year)) > 1:
        return False, sim
    if not surname_matches(ref.first_surname, view.first_author):
        return False, sim
    return True, sim


def pick_search_result(works: list[dict], ref: BibRef) -> WorkView | None:
    best: tuple[float, WorkView] | None = None
    for work in works:
        view = work_view(work)
        ok, sim = accept_search_candidate(view, ref)
        if ok and (best is None or sim > best[0]):
            best = (sim, view)
    return best[1] if best else None


def openalex_by_doi(http: HttpClient, doi: str) -> dict | None:
    url = f"{OPENALEX_WORKS}/https://doi.org/{urllib.parse.quote(doi, safe='/:()')}"
    return http.get_json(url, _mailto_params(http))


def openalex_search(http: HttpClient, title: str) -> list[dict]:
    params = {"search": title, "per-page": "5", **_mailto_params(http)}
    data = http.get_json(OPENALEX_WORKS, params)
    return list((data or {}).get("results") or [])


def _mailto_params(http: HttpClient) -> dict[str, str]:
    return {"mailto": http.mailto} if http.mailto else {}


def check_identity(
    ref: BibRef, source_title: str, source_doi: str = ""
) -> tuple[bool, str]:
    """DOI exato ou título equivalente (>= 0,85). Devolve (ok, método)."""
    if ref.doi and source_doi and ref.doi == normalize_doi(source_doi):
        return True, "doi_exato"
    sim = title_similarity(ref.title, source_title)
    if sim >= IDENTITY_MIN_SIM:
        return True, f"titulo_sim_{sim:.2f}"
    return False, f"titulo_divergente_sim_{sim:.2f}"


def content_identity(ref: BibRef, text: str) -> tuple[bool, str]:
    """Texto sem metadados: o título ou o DOI do .bib deve aparecer no início."""
    head = text[:HEAD_CHARS]
    title = normalize_title(ref.title)
    if title and title in normalize_title(head):
        return True, "titulo_no_texto"
    if ref.doi and ref.doi in head.lower():
        return True, "doi_no_texto"
    return False, "identidade_nao_encontrada_no_texto"


# ---------------------------------------------------------------------------
# Texto completo: Europe PMC (JATS), SciELO/HTML, PDF, arXiv
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    fonte: str
    url: str
    text: str
    title: str = ""
    doi: str = ""
    abstract: str = ""
    identity_by_content: bool = False


def parse_jats(xml_bytes: bytes) -> dict | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    title_el = root.find(".//front//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    doi = ""
    for node in root.iter("article-id"):
        if node.get("pub-id-type") == "doi":
            doi = (node.text or "").strip()
            break
    body = root.find(".//body")
    parts: list[str] = []
    if body is not None:
        for el in body.iter():
            if el.tag in ("title", "p"):
                text = re.sub(r"\s+", " ", "".join(el.itertext())).strip()
                if text:
                    parts.append(text)
    return {"title": title, "doi": doi, "text": "\n\n".join(parts)}


def epmc_pmcid_by_doi(http: HttpClient, doi: str) -> str:
    params = {"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"}
    data = http.get_json(f"{EPMC_REST}/search", params)
    for item in ((data or {}).get("resultList") or {}).get("result") or []:
        if item.get("pmcid") and str(item.get("isOpenAccess", "")).upper() == "Y":
            return str(item["pmcid"]).upper()
    return ""


def fetch_epmc(http: HttpClient, pmcid: str) -> Candidate | None:
    resp = http.get(f"{EPMC_REST}/{pmcid}/fullTextXML", accept="application/xml")
    if resp is None:
        return None
    parsed = parse_jats(resp.body)
    if parsed is None:
        return None
    return Candidate(
        "europepmc", f"https://europepmc.org/article/PMC/{pmcid.removeprefix('PMC')}",
        parsed["text"], parsed["title"], parsed["doi"],
    )


_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "head"}
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "table"}


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self._in_title = False
        self._skip = 0
        self._article_depth = 0
        self.everything: list[str] = []
        self.article: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta":
            name = (values.get("name") or values.get("property") or "").lower()
            if name and values.get("content"):
                self.meta.setdefault(name, values["content"] or "")
            return
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "article":
            self._article_depth += 1
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "article":
            self._article_depth = max(0, self._article_depth - 1)
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._skip:
            return
        self._emit(data)

    def _emit(self, chunk: str) -> None:
        self.everything.append(chunk)
        if self._article_depth:
            self.article.append(chunk)


def _squeeze(chunks: list[str]) -> str:
    text = "".join(chunks)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_html(html: str) -> dict:
    parser = _HtmlText()
    parser.feed(html)
    parser.close()
    article = _squeeze(parser.article)
    text = article if len(article) >= 2000 else _squeeze(parser.everything)
    title = parser.meta.get("citation_title") or parser.meta.get("dc.title") or parser.title.strip()
    doi = parser.meta.get("citation_doi") or parser.meta.get("dc.identifier") or ""
    return {"title": title.strip(), "doi": doi.strip(), "text": text}


_SCIELO_PID_RE = re.compile(r"^s\d{4}-\d{3}[\dx]\d{13}$", re.IGNORECASE)


def scielo_urls(doi: str) -> list[str]:
    """Endereços a tentar: resolução do DOI e, se o sufixo for um PID legado, a página ``sci_arttext``."""
    doi = normalize_doi(doi)
    urls = [f"https://doi.org/{urllib.parse.quote(doi, safe='/:()')}"]
    suffix = doi.partition("/")[2]
    if _SCIELO_PID_RE.match(suffix):
        pid = suffix.upper()
        urls.append(f"https://www.scielo.br/scielo.php?script=sci_arttext&pid={pid}&lng=pt")
    return urls


def fetch_scielo(http: HttpClient, doi: str, errors: list[str] | None = None) -> Candidate | None:
    """Texto do artigo SciELO pelo DOI 10.1590; cada falha vira uma linha em ``errors``."""
    notes = errors if errors is not None else []
    if not doi.startswith("10.1590/"):
        return None
    best: Candidate | None = None
    for url in scielo_urls(doi):
        host = urllib.parse.urlsplit(url).hostname
        try:
            resp = http.get(url, accept="text/html")
        except HttpError as exc:
            notes.append(f"scielo: {host}: {exc}")
            continue
        if resp is None:
            notes.append(f"scielo: {host}: HTTP {getattr(http, 'last_status', None) or '?'} sem página")
            continue
        parsed = parse_html(resp.text())
        cand = Candidate("scielo", resp.url, parsed["text"], parsed["title"], parsed["doi"])
        if len(cand.text) >= MIN_FULLTEXT_CHARS:
            return cand
        if best is None or len(cand.text) > len(best.text):
            best = cand
    return best


def pdf_pages(data: bytes) -> list[str] | None:
    """Texto de cada página do PDF via PyMuPDF; None se ``fitz`` não estiver instalado ou o PDF não abrir."""
    try:
        import fitz
    except ImportError:
        return None
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return [page.get_text() for page in pdf]
    except (RuntimeError, ValueError):
        return None


def pdf_to_text(data: bytes) -> str | None:
    """Texto do PDF via PyMuPDF (páginas unidas por quebra de linha); None se não for possível extrair."""
    pages = pdf_pages(data)
    return None if pages is None else "\n".join(pages)


def page_starts_for(pages: list[str], text: str) -> list[int] | None:
    """Deslocamentos em ``text`` onde começa cada página (``starts[k]`` é o início da página ``k + 1``).

    ``text`` tem de ser as páginas unidas por quebra de linha, com ou sem ``strip()``; outro texto
    devolve None (os deslocamentos não valeriam)."""
    joined = "\n".join(pages)
    if text == joined:
        lead = 0
    elif text == joined.strip():
        lead = len(joined) - len(joined.lstrip())
    else:
        return None
    starts: list[int] = []
    pos = 0
    for page in pages:
        starts.append(max(0, pos - lead))
        pos += len(page) + 1
    return starts


def local_pdf_page_starts(path: Path, text: str) -> list[int] | None:
    """Início de cada página do PDF local no texto já extraído; None sem PyMuPDF, ilegível ou divergente."""
    try:
        pages = pdf_pages(path.read_bytes())
    except OSError:
        return None
    return page_starts_for(pages, text) if pages else None


def fetch_unpaywall(http: HttpClient, doi: str, email: str) -> list[Candidate]:
    data = http.get_json(f"{UNPAYWALL}/{urllib.parse.quote(doi, safe='/:()')}", {"email": email})
    best = (data or {}).get("best_oa_location") or {}
    record_title = (data or {}).get("title") or ""
    record_doi = (data or {}).get("doi") or ""
    out: list[Candidate] = []
    pdf_url = best.get("url_for_pdf")
    if pdf_url:
        resp = http.get(pdf_url, accept="application/pdf")
        if resp is not None and resp.body[:5] == b"%PDF-":
            text = pdf_to_text(resp.body)
            if text:
                out.append(Candidate("unpaywall_pdf", resp.url, text, record_title, record_doi,
                                     identity_by_content=True))
    html_url = best.get("url_for_landing_page") or best.get("url")
    if html_url and html_url != pdf_url:
        resp = http.get(html_url, accept="text/html")
        if resp is not None and "pdf" not in resp.content_type.lower():
            parsed = parse_html(resp.text())
            out.append(Candidate("unpaywall_html", resp.url, parsed["text"],
                                 parsed["title"] or record_title, parsed["doi"] or record_doi,
                                 identity_by_content=True))
    return out


_ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv(http: HttpClient, arxiv_id: str) -> Candidate | None:
    resp = http.get(ARXIV_API, {"id_list": arxiv_id, "max_results": "1"})
    if resp is None:
        return None
    try:
        root = ET.fromstring(resp.body)
    except ET.ParseError:
        return None
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None
    title = re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}title") or "").strip()
    summary = re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}summary") or "").strip()
    text = ""
    pdf = http.get(f"{ARXIV_PDF}/{arxiv_id}", accept="application/pdf")
    if pdf is not None and pdf.body[:5] == b"%PDF-":
        text = pdf_to_text(pdf.body) or ""
    return Candidate("arxiv", f"https://arxiv.org/abs/{arxiv_id}", text, title,
                     abstract=summary, identity_by_content=True)


# ---------------------------------------------------------------------------
# Fontes locais fornecidas pelo autor (.local/fontes/<chave>.<ext>)
# ---------------------------------------------------------------------------
LOCAL_IDENTITY_MIN_SIM = 0.6
LOCAL_IDENTITY_HEAD = 3000
LOCAL_EXTENSIONS = ("txt", "pdf", "xml", "html")
FLAG_PREPRINT = "fonte_local_preprint"
FLAG_IDENTIDADE_LOCAL = "identidade_local_nao_confirmada"


@dataclass
class LocalText:
    path: Path
    text: str
    preprint: bool = False


def _normalized_len(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_tags(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _local_xml_text(path: Path) -> str:
    raw = path.read_bytes()
    parsed = parse_jats(raw)
    if parsed and _normalized_len(parsed["text"]) >= MIN_FULLTEXT_CHARS:
        return parsed["text"]
    return _strip_tags(raw.decode("utf-8", errors="replace"))


def _local_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return _read_text_file(path)
    if ext == ".pdf":
        return pdf_to_text(path.read_bytes()) or ""
    if ext == ".xml":
        return _local_xml_text(path)
    return parse_html(_read_text_file(path))["text"]


def find_local_fulltext(directory: Path | None, key: str) -> LocalText | None:
    """Texto local da chave. Só vale o nome exato ``<chave>.<ext>`` (txt, pdf, xml, html, nessa ordem de
    preferência, mínimo de 3.000 caracteres após normalizar espaços); ``<chave>_preprint.<ext>`` entra
    apenas se nenhum arquivo principal servir. Tentativas (``_tentativa*``, ``_oa`` etc.) são ignoradas."""
    if directory is None or not directory.is_dir():
        return None
    stem = key_filename(key)
    for suffix, preprint in (("", False), ("_preprint", True)):
        for ext in LOCAL_EXTENSIONS:
            path = directory / f"{stem}{suffix}.{ext}"
            if not path.is_file():
                continue
            try:
                text = _local_file_text(path).strip()
            except OSError:
                continue
            if _normalized_len(text) >= MIN_FULLTEXT_CHARS:
                return LocalText(path, text, preprint)
    return None


def local_title_score(title: str, text: str) -> float:
    """Melhor janela do início do texto (primeiros 3.000 caracteres) frente ao título do .bib:
    maior entre a fração das palavras do título presentes na janela (token-set) e a similaridade
    de sequência (partial) da janela."""
    wanted = normalize_title(title).split()
    head = normalize_title(text[:LOCAL_IDENTITY_HEAD])
    words = head.split()
    if not wanted or not words:
        return 0.0
    needed = set(wanted)
    size = len(wanted) + 4
    best = 0.0
    for start in range(0, max(1, len(words) - size + 1)):
        window = words[start:start + size]
        best = max(best, len(needed & set(window)) / len(needed))
        if best >= 1.0:
            return 1.0
    target = " ".join(wanted)
    span = int(len(target) * 1.3) + 1
    step = max(1, len(target) // 4)
    for start in range(0, max(1, len(head) - span + 1), step):
        piece = head[start:start + span]
        ratio = SequenceMatcher(None, target, piece, autojunk=False).find_longest_match(
            0, len(target), 0, len(piece)).size / len(target)
        best = max(best, ratio)
    return best


def local_identity(ref: BibRef, text: str) -> tuple[bool, str]:
    """Título (similaridade >= 0,6 na melhor janela inicial) ou DOI no texto; o arquivo foi nomeado pelo autor."""
    head = text[:HEAD_CHARS].lower()
    if ref.doi and ref.doi in head:
        return True, "doi_no_texto"
    score = local_title_score(ref.title, text)
    if score >= LOCAL_IDENTITY_MIN_SIM:
        return True, f"titulo_local_sim_{score:.2f}"
    return False, f"titulo_local_sim_{score:.2f}"


# ---------------------------------------------------------------------------
# Coleta por chave
# ---------------------------------------------------------------------------
@dataclass
class Collector:
    http: HttpClient
    unpaywall_email: str = ""
    errors: list[str] = field(default_factory=list)
    fontes_locais: Path | None = None

    def safe(self, label: str, call: Callable[[], object]):
        """Executa uma consulta; falhas de rede ficam registradas e não interrompem a coleta."""
        try:
            return call()
        except HttpError as exc:
            self.errors.append(f"{label}: {exc}")
            return None


def resolve_openalex(col: Collector, ref: BibRef) -> tuple[WorkView | None, str]:
    """Obra do OpenAlex por DOI; sem DOI (ou DOI desconhecido), busca por título."""
    if ref.doi:
        work = col.safe("openalex_doi", lambda: openalex_by_doi(col.http, ref.doi))
        if work:
            return work_view(work), "openalex_doi"
    if ref.title:
        works = col.safe("openalex_busca", lambda: openalex_search(col.http, ref.title)) or []
        view = pick_search_result(works, ref)
        if view is not None:
            return view, "openalex_busca_titulo"
        if works:
            col.errors.append("openalex_busca: nenhum resultado passou nos limiares (sim>=0,90, ano±1, sobrenome)")
    return None, ""


def candidate_identity(ref: BibRef, cand: Candidate) -> tuple[bool, str]:
    """Identidade do candidato: metadados da fonte e, quando a fonte é um texto solto, o conteúdo."""
    if cand.title or cand.doi:
        ok, method = check_identity(ref, cand.title, cand.doi)
        if ok and (not cand.identity_by_content or not cand.text):
            return True, method
        if ok:
            ok_content, method_content = content_identity(ref, cand.text)
            return ok_content, f"{method}+{method_content}"
        if cand.identity_by_content and cand.text:
            return content_identity(ref, cand.text)
        return False, method
    if cand.text:
        return content_identity(ref, cand.text)
    return False, "sem_metadados"


def collect_evidence(col: Collector, ref: BibRef, classe: str) -> dict:
    """Resumo e texto completo de uma chave (sem consensus), sempre com identidade verificada."""
    record: dict = {
        "key": ref.key,
        "classe": classe,
        "doi": ref.doi,
        "titulo_bib": ref.title,
        "titulo_fonte": "",
        "identidade_verificada": False,
        "metodo": "",
        "abstract": "",
        "fulltext": "",
        "fulltext_fonte": "",
        "url": ref.url,
        "obtido_em": utc_now_iso(),
        "erros": col.errors,
    }
    methods: list[str] = []
    view, how = resolve_openalex(col, ref)
    doi = ref.doi
    pmcid = arxiv_id = ""
    if view is not None:
        ok, method = check_identity(ref, view.title, view.doi)
        if ok:
            record["titulo_fonte"] = view.title
            record["identidade_verificada"] = True
            record["url"] = view.landing_url or record["url"]
            doi = doi or view.doi
            pmcid, arxiv_id = view.pmcid, view.arxiv_id
            if view.abstract:
                record["abstract"] = view.abstract
                methods.append(f"abstract:{how}:{method}")
        else:
            col.errors.append(f"openalex: identidade rejeitada ({method}); resumo descartado")
            record["titulo_fonte"] = view.title
    record["doi"] = doi

    local = find_local_fulltext(col.fontes_locais, ref.key)
    chosen = None
    if local is not None:
        ok, method = local_identity(ref, local.text)
        flags = [FLAG_PREPRINT] if local.preprint else []
        if not ok:
            flags.append(FLAG_IDENTIDADE_LOCAL)
        record.update(
            fulltext=local.text, fulltext_fonte=f"local:{local.path.name}",
            fulltext_tamanho=len(local.text), identidade_verificada=ok,
            flags=flags,
        )
        methods.append(f"fulltext:local:{method}")
        if local.path.suffix.lower() == ".pdf":
            starts = local_pdf_page_starts(local.path, local.text)
            if starts:
                record["fulltext_paginas"] = starts
    else:
        chosen = _first_fulltext(col, ref, doi, pmcid, arxiv_id)
    if chosen is not None:
        cand, method = chosen
        record.update(identidade_verificada=True, url=cand.url,
                      titulo_fonte=cand.title or record["titulo_fonte"])
        if cand.text:
            record.update(fulltext=cand.text, fulltext_fonte=cand.fonte)
            methods.append(f"fulltext:{cand.fonte}:{method}")
        if cand.abstract and not record["abstract"]:
            record["abstract"] = cand.abstract
            methods.append(f"abstract:{cand.fonte}:{method}")
    record["metodo"] = "; ".join(methods) if methods else "sem_evidencia_verificada"
    return record


def _first_fulltext(
    col: Collector, ref: BibRef, doi: str, pmcid: str, arxiv_id: str
) -> tuple[Candidate, str] | None:
    def accepted(cand: Candidate | None) -> tuple[Candidate, str] | None:
        if cand is None:
            return None
        if len(cand.text) < MIN_FULLTEXT_CHARS:
            if cand.abstract and not cand.text:
                ok, method = candidate_identity(ref, cand)
                return (cand, method) if ok else None
            col.errors.append(f"{cand.fonte}: texto curto demais ({len(cand.text)} caracteres)")
            return None
        ok, method = candidate_identity(ref, cand)
        if not ok:
            col.errors.append(f"{cand.fonte}: identidade rejeitada ({method}); texto descartado")
            return None
        return cand, method

    if not pmcid and doi:
        pmcid = col.safe("europepmc_busca", lambda: epmc_pmcid_by_doi(col.http, doi)) or ""
    if pmcid:
        hit = accepted(col.safe("europepmc", lambda: fetch_epmc(col.http, pmcid)))
        if hit:
            return hit
    if doi.startswith("10.1590/"):
        hit = accepted(col.safe("scielo", lambda: fetch_scielo(col.http, doi, col.errors)))
        if hit:
            return hit
    if doi:
        if col.unpaywall_email:
            cands = col.safe("unpaywall", lambda: fetch_unpaywall(col.http, doi, col.unpaywall_email)) or []
            for cand in cands:
                hit = accepted(cand)
                if hit:
                    return hit
        else:
            col.errors.append("unpaywall: sem e-mail (UNPAYWALL_EMAIL ou CROSSREF_MAILTO)")
    if not arxiv_id and doi:
        match = _ARXIV_DOI_RE.match(doi)
        arxiv_id = match.group(1) if match else ""
    if arxiv_id:
        return accepted(col.safe("arxiv", lambda: fetch_arxiv(col.http, arxiv_id)))
    return None


# ---------------------------------------------------------------------------
# Consensus, cache e níveis
# ---------------------------------------------------------------------------
def load_consensus(path: Path, ref: BibRef) -> tuple[list[dict], int]:
    """Citações do Consensus da própria obra (identidade checada); devolve (mantidas, descartadas)."""
    data = read_json(path, default=None)
    if not isinstance(data, list):
        return [], 0
    kept: list[dict] = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict) or not str(item.get("quote") or "").strip():
            dropped += 1
            continue
        ok, _method = check_identity(ref, str(item.get("paper_title") or ""), str(item.get("doi") or ""))
        if ok:
            kept.append({k: item.get(k, "") for k in ("quote", "paper_title", "doi", "source")})
        else:
            dropped += 1
    return kept, dropped


def evidence_level(record: dict) -> str:
    if record.get("fulltext"):
        return "texto_completo"
    if record.get("abstract"):
        return "resumo"
    if record.get("consensus"):
        return "consensus"
    return "nenhum"


def cache_path(out_dir: Path, key: str) -> Path:
    return out_dir / "evidencias" / f"{key_filename(key)}.json"


def _backfill_pages(col: Collector, record: dict) -> None:
    """Registros antigos de PDF local sem ``fulltext_paginas``: recalcula a partir do arquivo local."""
    source = str(record.get("fulltext_fonte") or "")
    if "fulltext_paginas" in record or not source.startswith("local:") or col.fontes_locais is None:
        return
    path = col.fontes_locais / source[len("local:"):]
    if path.suffix.lower() != ".pdf" or not path.is_file():
        return
    starts = local_pdf_page_starts(path, record.get("fulltext") or "")
    if starts:
        record["fulltext_paginas"] = starts


def get_evidence(
    col: Collector, ref: BibRef, classe: str, out_dir: Path, *, refresh: bool = False
) -> dict:
    """Evidência da chave, do cache (a menos que ``refresh``) ou da rede; consensus relido sempre."""
    path = cache_path(out_dir, ref.key)
    record = None if refresh else read_json(path)
    if record is not None and not str(record.get("fulltext_fonte") or "").startswith("local:"):
        if find_local_fulltext(col.fontes_locais, ref.key) is not None:
            record = None  # há texto local novo: o cache de rede não vale
    if record is None:
        col.errors = []
        record = collect_evidence(col, ref, classe)
    _backfill_pages(col, record)
    record["classe"] = classe
    consensus, dropped = load_consensus(out_dir / "consensus" / f"{key_filename(ref.key)}.json", ref)
    record["consensus"] = consensus
    record["consensus_descartadas"] = dropped
    write_json_atomic(path, record)
    return record


def source_label(record: dict) -> str:
    if record.get("fulltext"):
        return record.get("fulltext_fonte") or "?"
    if record.get("abstract"):
        return "openalex"
    if record.get("consensus"):
        return "consensus"
    return "-"


def write_resumo(path: Path, rows: list[dict], capitulo: str) -> None:
    """``evidencias_resumo.md``: nível, fonte e identidade por chave."""
    lines = [
        f"# Evidências recuperadas: {capitulo}",
        "",
        "| Chave | Classe | Nível | Fonte | Identidade | Método | Observações |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    counts = {level: 0 for level in LEVELS}
    for row in rows:
        if row["classe"] == "proprio":
            lines.append(f"| `{row['key']}` | proprio | artefato | repositório | n/a | {row['artefato']} | sem recuperação |")
            continue
        level = evidence_level(row)
        counts[level] += 1
        identity = "sim" if row.get("identidade_verificada") else "não"
        notes = "; ".join([*(row.get("flags") or []), *(row.get("erros") or [])])[:300].replace("|", "/")
        method = (row.get("metodo") or "").replace("|", "/")
        lines.append(f"| `{row['key']}` | {row['classe']} | {level} | {source_label(row)} | {identity} | {method} | {notes} |")
    lines += ["", "## Totais por nível", ""]
    lines += [f"- {level}: {n}" for level, n in counts.items()]
    write_text_atomic(path, "\n".join(lines) + "\n")
