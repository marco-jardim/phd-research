"""Testes da atestação bibliográfica (scripts/atestar_citacoes.py e scripts/atestado/).

Toda rede e toda chamada à API são simuladas; nada depende de .env, data/ ou da tese real.
"""

import json
import os
import re
import unicodedata
import urllib.request
from types import SimpleNamespace

import pytest

from scripts import atestar_citacoes as cli
from scripts.atestado import adjudicar as adj
from scripts.atestado import avaliar as av
from scripts.atestado import common
from scripts.atestado import consultas as cons
from scripts.atestado import evidencias as ev
from scripts.atestado import http as net
from scripts.atestado import numeros
from scripts.atestado import prompts
from scripts.atestado import proprios as pr
from scripts.atestado import relatorio as rel
from scripts.atestado import trechos as tr
from scripts.atestado import verificar as vf
from scripts.citacoes_contexto import BibEntry

ABSTRACT = "The coverage reached 92.4% in 2015 across all regions."
QUOTE = "coverage reached 92.4% in 2015 across all regions"  # 8 palavras, 49 caracteres
TITLE = "Probabilistic linkage in cancer registries"
DOI = "10.1234/abc.1"
EMAIL = "test@example.org"


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def make_ref(key="k", etype="article", author="Silva, Ana and Souza, B.", doi="", title=TITLE, year="2018"):
    fields = {"author": author, "title": title, "year": year}
    if doi:
        fields["doi"] = doi
    return ev.bib_ref(BibEntry(key, etype, fields))


class FakeHttp(net.HttpClient):
    """HttpClient real (com a guarda de hosts), mas com o transporte substituído por rotas."""

    def __init__(self, routes, retries=0, sleeps=None):
        super().__init__(EMAIL, min_interval=0, retries=retries,
                         sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
        self.routes = routes
        self.requested = []

    def _fetch(self, url, headers):
        self.requested.append(url)
        for prefix, status, body, ctype in self.routes:
            if url.startswith(prefix):
                if isinstance(body, (dict, list)):
                    body = json.dumps(body).encode()
                elif isinstance(body, str):
                    body = body.encode()
                return status, body, ctype, url
        return 404, b"", "", url


def openalex_doi_prefix(doi=DOI):
    return f"https://api.openalex.org/works/https://doi.org/{doi}"


def work(title=TITLE, doi=DOI, year=2018, author="Ana Silva", abstract=None, pmcid=None):
    data = {
        "title": title,
        "doi": f"https://doi.org/{doi}",
        "publication_year": year,
        "authorships": [{"author": {"display_name": author}}],
        "ids": {"pmcid": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"} if pmcid else {},
    }
    if abstract:
        words = abstract.split()
        data["abstract_inverted_index"] = {w: [i] for i, w in enumerate(words)}
    return data


def jats(title, doi, n=60):
    paras = "".join(f"<p>{'Texto do artigo %d sobre vinculação de registros. ' % i * 3}</p>" for i in range(n))
    return (
        '<?xml version="1.0"?><article><front><article-meta><title-group>'
        f"<article-title>{title}</article-title></title-group>"
        f'<article-id pub-id-type="doi">{doi}</article-id></article-meta></front>'
        f"<body><sec><title>Introdução</title>{paras}</sec></body></article>"
    ).encode()


def html_page(title, doi, n=60):
    paras = "".join(f"<p>{'Parágrafo %d do artigo completo da revista. ' % i * 3}</p>" for i in range(n))
    return (
        f'<html><head><title>x</title><meta name="citation_title" content="{title}">'
        f'<meta name="citation_doi" content="{doi}"></head>'
        f"<body><nav>menu do site</nav><article>{paras}</article></body></html>"
    )


def collector(http):
    return ev.Collector(http, unpaywall_email=EMAIL)


def ctx(i, key, sentence="A cobertura chegou a 92% [cit].", chapter="Introdução"):
    return {
        "id": f"id{i:02d}", "key": key, "file": "text/latex/cap1.tex", "line": 10 + i,
        "chapter": chapter, "section_path": [], "sentence_latex": sentence,
        "sentence_text": sentence, "paragraph_text": "Parágrafo. " + sentence, "co_cited": [],
    }


# ---------------------------------------------------------------------------
# Classificação de chaves e .bib
# ---------------------------------------------------------------------------
def test_classify_key():
    assert ev.classify_key(make_ref(author="Jardim, Marco")) == "proprio"
    assert ev.classify_key(make_ref(key="meu", author="Silva, A."), {"meu"}) == "proprio"
    assert ev.classify_key(make_ref(etype="article")) == "literatura"
    assert ev.classify_key(make_ref(etype="book")) == "literatura"
    assert ev.classify_key(make_ref(etype="inproceedings")) == "literatura"
    assert ev.classify_key(make_ref(etype="incollection")) == "literatura"
    assert ev.classify_key(make_ref(etype="misc", doi="10.1/x")) == "literatura"
    assert ev.classify_key(make_ref(etype="misc")) == "cinzenta"
    assert ev.classify_key(make_ref(etype="techreport")) == "cinzenta"
    assert ev.classify_key(make_ref(etype="phdthesis")) == "cinzenta"
    assert ev.classify_key(make_ref(etype="mastersthesis")) == "cinzenta"
    assert ev.classify_key(None) == "cinzenta"


def test_load_bib_refs(tmp_path):
    bib = tmp_path / "r.bib"
    bib.write_text(
        "@article{a, author={Silva, Ana and Souza, B.}, title={Linkage de {R}egistros}, year={2018},"
        " doi={https://doi.org/10.1/ABC}}\n"
        "@misc{b, author={{World Health Organization}}, title={Relat\\'orio}, year={2020}}\n"
        "@book{c, author={Ana Maria Lima}, title={Livro}, year={2001}}\n",
        encoding="utf-8",
    )
    refs = ev.load_bib_refs(bib)
    assert refs["a"].first_surname == "Silva" and refs["a"].doi == "10.1/abc" and refs["a"].year == 2018
    assert refs["a"].title == "Linkage de Registros"
    assert refs["b"].first_surname == "World Health Organization"
    assert refs["c"].first_surname == "Lima"


# ---------------------------------------------------------------------------
# OpenAlex: resumo e busca por título
# ---------------------------------------------------------------------------
def test_reconstruct_abstract_from_inverted_index():
    assert ev.reconstruct_abstract({"Hello": [0, 3], "world": [1], "again": [2]}) == "Hello world again Hello"
    assert ev.reconstruct_abstract(None) == ""
    assert ev.reconstruct_abstract({}) == ""


def test_search_acceptance_thresholds():
    ref = make_ref(title="x" * 100)
    near = ev.work_view(work(title="x" * 100 + "y" * 10))  # similaridade ~0,95
    far = ev.work_view(work(title="x" * 100 + "y" * 25))  # similaridade ~0,89
    assert ev.accept_search_candidate(near, ref)[0] is True
    assert ev.accept_search_candidate(far, ref)[0] is False
    assert common.title_similarity("x" * 100, "x" * 125) < ev.SEARCH_MIN_SIM <= common.title_similarity(
        "x" * 100, "x" * 110
    )


def test_search_acceptance_year_and_author():
    ref = make_ref()
    assert ev.accept_search_candidate(ev.work_view(work(year=2019)), ref)[0] is True
    assert ev.accept_search_candidate(ev.work_view(work(year=2017)), ref)[0] is True
    assert ev.accept_search_candidate(ev.work_view(work(year=2020)), ref)[0] is False
    assert ev.accept_search_candidate(ev.work_view(work(year=None)), ref)[0] is False
    assert ev.accept_search_candidate(ev.work_view(work(author="Maria Oliveira")), ref)[0] is False
    assert ev.accept_search_candidate(ev.work_view(work(title="Hospital records for a different topic")), ref)[0] is False


def test_pick_search_result_prefers_best_accepted():
    ref = make_ref()
    works = [work(title="Outro assunto totalmente diferente"), work(title=TITLE + "."), work(title=TITLE)]
    picked = ev.pick_search_result(works, ref)
    assert picked is not None and picked.title in (TITLE, TITLE + ".")
    assert ev.pick_search_result([work(author="Maria Oliveira")], ref) is None


def test_resolve_by_title_search_applies_thresholds():
    ref = make_ref()
    good = FakeHttp([("https://api.openalex.org/works?search=", 200, {"results": [work()]}, "application/json")])
    view, how = ev.resolve_openalex(collector(good), ref)
    assert view is not None and how == "openalex_busca_titulo"
    bad = FakeHttp([("https://api.openalex.org/works?search=", 200,
                     {"results": [work(author="Maria Oliveira")]}, "application/json")])
    col = collector(bad)
    assert ev.resolve_openalex(col, ref) == (None, "")
    assert any("limiares" in e for e in col.errors)


# ---------------------------------------------------------------------------
# Identidade, texto completo e cache
# ---------------------------------------------------------------------------
def test_check_identity():
    ref = make_ref(doi=DOI)
    assert ev.check_identity(ref, "qualquer", DOI.upper()) == (True, "doi_exato")
    ok, method = ev.check_identity(ref, TITLE.upper() + ".", "")
    assert ok and method.startswith("titulo_sim_")
    assert ev.check_identity(ref, "Cats and dogs in veterinary clinics", "10.9/zzz")[0] is False


def test_wrong_paper_abstract_is_rejected():
    ref = make_ref(doi=DOI)
    wrong = work(title="Cats and dogs in veterinary clinics", doi="10.9/zzz",
                 abstract="Cats are mammals that purr loudly.")
    http = FakeHttp([(openalex_doi_prefix(), 200, wrong, "application/json")])
    col = collector(http)
    record = ev.collect_evidence(col, ref, "literatura")
    assert record["abstract"] == "" and record["fulltext"] == ""
    assert record["identidade_verificada"] is False
    assert record["metodo"] == "sem_evidencia_verificada"
    assert any("identidade rejeitada" in e for e in record["erros"])
    assert ev.evidence_level(record) == "nenhum"


def test_right_abstract_is_accepted_by_doi():
    ref = make_ref(doi=DOI)
    http = FakeHttp([(openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json")])
    record = ev.collect_evidence(collector(http), ref, "literatura")
    assert record["abstract"] == ABSTRACT
    assert record["identidade_verificada"] is True
    assert "doi_exato" in record["metodo"]
    assert ev.evidence_level(record) == "resumo"


def test_europepmc_fulltext_accepted_and_wrong_paper_rejected():
    ref = make_ref(doi=DOI)
    oa = (openalex_doi_prefix(), 200, work(abstract=ABSTRACT, pmcid="PMC12345"), "application/json")
    epmc = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12345/fullTextXML"
    right = FakeHttp([oa, (epmc, 200, jats(TITLE, DOI), "application/xml")])
    record = ev.collect_evidence(collector(right), ref, "literatura")
    assert record["fulltext_fonte"] == "europepmc" and len(record["fulltext"]) > ev.MIN_FULLTEXT_CHARS
    assert ev.evidence_level(record) == "texto_completo"

    wrong = FakeHttp([oa, (epmc, 200, jats("Cats and dogs in veterinary clinics", "10.9/zzz"), "application/xml")])
    record = ev.collect_evidence(collector(wrong), ref, "literatura")
    assert record["fulltext"] == "" and record["fulltext_fonte"] == ""
    assert any("europepmc: identidade rejeitada" in e for e in record["erros"])
    assert record["abstract"] == ABSTRACT


def test_scielo_fulltext_by_doi_prefix():
    doi = "10.1590/s0102-311x2020000100001"
    ref = make_ref(doi=doi)
    routes = [
        (openalex_doi_prefix(doi), 200, work(doi=doi, abstract=ABSTRACT), "application/json"),
        (f"https://doi.org/{doi}", 200, html_page(TITLE, doi), "text/html; charset=utf-8"),
    ]
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert record["fulltext_fonte"] == "scielo"
    assert "menu do site" not in record["fulltext"]
    assert "Parágrafo 3 do artigo" in record["fulltext"]


SCIELO_DOI = "10.1590/s0102-311x2000000200014"
SCIELO_PID = "S0102-311X2000000200014"


@pytest.mark.parametrize("meta_doi", [
    SCIELO_DOI.upper(), SCIELO_DOI, "10.1590/" + SCIELO_PID, f"https://doi.org/{SCIELO_DOI.upper()}",
])
def test_scielo_identity_by_citation_doi_meta_in_any_case(meta_doi):
    ref = make_ref(doi=SCIELO_DOI)
    routes = [
        (openalex_doi_prefix(SCIELO_DOI), 200, work(doi=SCIELO_DOI, abstract=ABSTRACT), "application/json"),
        (f"https://doi.org/{SCIELO_DOI}", 200, html_page("Título impresso diferente na página", meta_doi),
         "text/html; charset=utf-8"),
    ]
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert record["fulltext_fonte"] == "scielo"
    assert "fulltext:scielo:doi_exato" in record["metodo"]
    assert ev.evidence_level(record) == "texto_completo"


def test_scielo_falls_back_to_legacy_pid_page_when_doi_org_fails():
    ref = make_ref(doi=SCIELO_DOI)
    legacy = f"https://www.scielo.br/scielo.php?script=sci_arttext&pid={SCIELO_PID}"
    routes = [
        (openalex_doi_prefix(SCIELO_DOI), 200, work(doi=SCIELO_DOI, abstract=ABSTRACT), "application/json"),
        (f"https://doi.org/{SCIELO_DOI}", 403, b"", "text/html"),
        (legacy, 200, html_page(TITLE, SCIELO_DOI.upper()), "text/html; charset=utf-8"),
    ]
    http = FakeHttp(routes)
    record = ev.collect_evidence(collector(http), ref, "literatura")
    assert record["fulltext_fonte"] == "scielo"
    assert any(u.startswith(legacy) for u in http.requested)  # PID em maiúsculas
    assert any("scielo: doi.org: HTTP 403" in e for e in record["erros"])


def test_scielo_failure_is_visible_in_observations():
    ref = make_ref(doi=SCIELO_DOI)
    routes = [
        (openalex_doi_prefix(SCIELO_DOI), 200, work(doi=SCIELO_DOI, abstract=ABSTRACT), "application/json"),
        (f"https://doi.org/{SCIELO_DOI}", 403, b"", "text/html"),
        ("https://www.scielo.br/scielo.php", 404, b"", "text/html"),
    ]
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert ev.evidence_level(record) == "resumo"
    assert any("scielo: doi.org: HTTP 403 sem página" in e for e in record["erros"])
    assert any("scielo: www.scielo.br: HTTP 404 sem página" in e for e in record["erros"])


def test_scielo_wrong_article_is_rejected_with_reason():
    ref = make_ref(doi=SCIELO_DOI)
    routes = [
        (openalex_doi_prefix(SCIELO_DOI), 200, work(doi=SCIELO_DOI, abstract=ABSTRACT), "application/json"),
        (f"https://doi.org/{SCIELO_DOI}", 200,
         html_page("Cats and dogs in veterinary clinics", "10.1590/s0000-0000000000000000"), "text/html"),
    ]
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert record["fulltext"] == ""
    assert any("scielo: identidade rejeitada" in e for e in record["erros"])


def test_refresh_refetches_record_cached_without_fulltext(tmp_path):
    ref = make_ref(doi=SCIELO_DOI)
    oa = (openalex_doi_prefix(SCIELO_DOI), 200, work(doi=SCIELO_DOI, abstract=ABSTRACT), "application/json")
    falha = FakeHttp([oa, (f"https://doi.org/{SCIELO_DOI}", 403, b"", "text/html")])
    primeiro = ev.get_evidence(collector(falha), ref, "literatura", tmp_path)
    assert ev.evidence_level(primeiro) == "resumo"

    ok = FakeHttp([oa, (f"https://doi.org/{SCIELO_DOI}", 200, html_page(TITLE, SCIELO_DOI), "text/html")])
    em_cache = ev.get_evidence(collector(ok), ref, "literatura", tmp_path)
    assert ev.evidence_level(em_cache) == "resumo" and ok.requested == []
    refeito = ev.get_evidence(collector(ok), ref, "literatura", tmp_path, refresh=True)
    assert ev.evidence_level(refeito) == "texto_completo" and refeito["fulltext_fonte"] == "scielo"


def test_unpaywall_pdf_requires_identity_in_text(monkeypatch):
    ref = make_ref(doi=DOI)
    unpaywall = {"title": TITLE, "doi": DOI, "best_oa_location": {"url_for_pdf": "https://repo.example.org/p.pdf"}}
    routes = [
        (openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json"),
        (f"https://api.unpaywall.org/v2/{DOI}", 200, unpaywall, "application/json"),
        ("https://repo.example.org/p.pdf", 200, b"%PDF-1.7 conteudo", "application/pdf"),
    ]
    monkeypatch.setattr(ev, "pdf_to_text", lambda data: TITLE + "\n" + "palavra " * 1000)
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert record["fulltext_fonte"] == "unpaywall_pdf"

    monkeypatch.setattr(ev, "pdf_to_text", lambda data: "Cats and dogs.\n" + "palavra " * 1000)
    record = ev.collect_evidence(collector(FakeHttp(routes)), ref, "literatura")
    assert record["fulltext"] == ""
    assert any("unpaywall_pdf: identidade rejeitada" in e for e in record["erros"])


def test_unpaywall_needs_email():
    ref = make_ref(doi=DOI)
    http = FakeHttp([(openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json")])
    record = ev.collect_evidence(ev.Collector(http, unpaywall_email=""), ref, "literatura")
    assert any("unpaywall: sem e-mail" in e for e in record["erros"])
    assert not any("api.unpaywall.org" in u for u in http.requested)


def test_consensus_is_loaded_with_identity_check_and_cache_is_reused(tmp_path):
    ref = make_ref(doi=DOI)
    consensus_dir = tmp_path / "consensus"
    consensus_dir.mkdir()
    (consensus_dir / "k.json").write_text(json.dumps([
        {"quote": "Linkage improved recall.", "paper_title": TITLE, "doi": DOI, "source": "consensus"},
        {"quote": "Cats purr.", "paper_title": "Cats and dogs in veterinary clinics", "doi": "10.9/z", "source": "consensus"},
    ]), encoding="utf-8")
    http = FakeHttp([])
    record = ev.get_evidence(collector(http), ref, "literatura", tmp_path)
    assert len(record["consensus"]) == 1 and record["consensus_descartadas"] == 1
    assert ev.evidence_level(record) == "consensus"
    n_requests = len(http.requested)
    assert n_requests > 0
    ev.get_evidence(collector(http), ref, "literatura", tmp_path)
    assert len(http.requested) == n_requests  # cache reutilizado
    ev.get_evidence(collector(http), ref, "literatura", tmp_path, refresh=True)
    assert len(http.requested) > n_requests


# ---------------------------------------------------------------------------
# Bibliotecas-sombra nunca são requisitadas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://annas-archive.org/md5/abc", "https://sci-hub.se/10.1/x", "https://www.scihub.ru/x",
    "http://libgen.is/book", "https://z-lib.org/x", "https://library.lol/main/1",
])
def test_forbidden_hosts_raise_before_any_request(url):
    http = FakeHttp([])
    with pytest.raises(net.ForbiddenHostError):
        http.get(url)
    assert http.requested == []


def test_shadow_library_urls_from_unpaywall_are_never_requested():
    ref = make_ref(doi=DOI)
    unpaywall = {"title": TITLE, "doi": DOI, "best_oa_location": {
        "url_for_pdf": "https://sci-hub.se/10.1234/abc.1",
        "url_for_landing_page": "https://annas-archive.org/md5/abc"}}
    http = FakeHttp([
        (openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json"),
        (f"https://api.unpaywall.org/v2/{DOI}", 200, unpaywall, "application/json"),
    ])
    record = ev.collect_evidence(collector(http), ref, "literatura")
    assert http.requested
    assert not any(net.is_forbidden_host(u) for u in http.requested)
    assert any("host proibido" in e for e in record["erros"])
    assert record["fulltext"] == ""


def test_redirect_and_final_url_to_shadow_library_are_blocked():
    handler = net._GuardedRedirect()
    request = urllib.request.Request("https://doi.org/10.1/x")
    with pytest.raises(net.ForbiddenHostError):
        handler.redirect_request(request, None, 302, "Found", {}, "https://sci-hub.se/x")

    class Redirected(FakeHttp):
        def _fetch(self, url, headers):
            return 200, b"ok", "text/html", "https://annas-archive.org/final"

    with pytest.raises(net.ForbiddenHostError):
        Redirected([]).get("https://doi.org/10.1/x")


def test_http_retry_with_backoff_and_user_agent():
    sleeps = []
    answers = iter([503, 503, 200])

    class Flaky(FakeHttp):
        def _fetch(self, url, headers):
            status = next(answers)
            return status, b'{"ok": true}', "application/json", url

    http = Flaky([], retries=2, sleeps=sleeps)
    assert http.get_json("https://api.openalex.org/works") == {"ok": True}
    assert sleeps == [2.0, 4.0]
    assert f"mailto:{EMAIL}" in http.user_agent
    with pytest.raises(net.HttpError):
        FakeHttp([("https://x.example.org/", 500, b"", "")]).get("https://x.example.org/")
    assert FakeHttp([("https://x.example.org/", 404, b"", "")]).get("https://x.example.org/") is None


# ---------------------------------------------------------------------------
# Normalização e verificação mecânica de trechos
# ---------------------------------------------------------------------------
def test_normalize_text_basics():
    assert common.normalize_text("  A\u00a0 B\n\nC ") == "a b c"
    assert common.normalize_text("lin\u00adkage") == "linkage"
    assert common.normalize_text("\u2018x\u2019 \u201cy\u201d \u2013 z \u2014 w") == "'x' \"y\" - z - w"
    assert common.normalize_text("ﬁnal") == "final"


@pytest.mark.parametrize("quote,evidence", [
    ("coverage reached 92.4% in 2015, as 'previously' reported - see below",
     "The coverage reached 92.4% in 2015, as \u2018previously\u2019 reported \u2014 see below."),
    (unicodedata.normalize("NFD", "vigilância epidemiológica"), "A Vigilância Epidemiológica é"),
    ("VIGILÂNCIA EPIDEMIOLÓGICA", "a vigilância epidemiológica é"),
    ("the record linkage method", "uses the record link-\nage method here"),
    ("a b c", "x a  b\n\n c y"),
    ("well-known", "a well-\nknown result"),
    ("final result", "the ﬁnal result"),
])
def test_verify_quote_accepts_equivalent_text(quote, evidence):
    assert vf.verify_quote(quote, evidence)


def test_verify_quote_rejects_missing_text():
    assert not vf.verify_quote("completely different words", ABSTRACT)
    assert not vf.verify_quote("", ABSTRACT)
    assert not vf.verify_quote("coverage reached 99% in 2015", ABSTRACT)


def item(verdict="SUSTENTA", quote=QUOTE, problem=None, ident="id01", level="resumo"):
    return {"id": ident, "veredito": verdict, "trecho_fonte": quote, "justificativa": "j",
            "problema": problem, "nivel_evidencia": level}


def test_verdict_downgraded_when_quote_not_found():
    ok = vf.make_assessment(item(), ABSTRACT)
    assert ok["veredito"] == "SUSTENTA" and ok["trecho_verificado"] is True and ok["flags"] == []

    bad = vf.make_assessment(item(quote="invented passage not in the source"), ABSTRACT)
    assert bad["veredito"] == "NAO_VERIFICAVEL" and bad["veredito_modelo"] == "SUSTENTA"
    assert bad["trecho_verificado"] is False and "trecho_nao_encontrado" in bad["flags"]

    missing = vf.make_assessment(item(quote=""), ABSTRACT)
    assert missing["veredito"] == "NAO_VERIFICAVEL" and "trecho_ausente" in missing["flags"]

    no_quote_needed = vf.make_assessment(item("NAO_SUSTENTA", quote=""), ABSTRACT)
    assert no_quote_needed["veredito"] == "NAO_SUSTENTA" and no_quote_needed["trecho_verificado"] is None

    invalid = vf.make_assessment(item("TALVEZ"), ABSTRACT)
    assert invalid["veredito"] == "NAO_VERIFICAVEL" and "veredito_invalido" in invalid["flags"]

    assert vf.make_assessment(item(problem="inventado"), ABSTRACT)["problema"] == "outro"
    assert "sem_resposta_modelo" in vf.make_assessment(None, ABSTRACT)["flags"]


def test_quote_length_rule():
    assert vf.quote_long_enough("one two three four five six")  # 6 palavras
    assert vf.quote_long_enough("x" * 40)  # 40 caracteres
    assert not vf.quote_long_enough("one two three four five")
    assert not vf.quote_long_enough("x" * 39)
    assert not vf.quote_long_enough("")


@pytest.mark.parametrize("verdict", ["SUSTENTA", "SUSTENTA_PARCIALMENTE", "CONTRADIZ"])
def test_short_quote_downgrades_supporting_verdicts(verdict):
    short = "coverage reached 92.4% in 2015"  # encontrado na fonte, mas só 5 palavras e 30 caracteres
    assert vf.verify_quote(short, ABSTRACT)
    result = vf.make_assessment(item(verdict, quote=short), ABSTRACT)
    assert result["veredito"] == "NAO_VERIFICAVEL" and result["veredito_modelo"] == verdict
    assert result["trecho_verificado"] is False
    assert "trecho_curto" in result["flags"] and "trecho_nao_encontrado" not in result["flags"]
    assert vf.needs_second_pass(result)


def test_short_quote_is_accepted_by_chars_or_words_and_ignored_for_nao_sustenta():
    by_words = vf.make_assessment(item(quote="The coverage reached 92.4% in 2015"), ABSTRACT)  # 6 palavras
    assert by_words["veredito"] == "SUSTENTA" and by_words["trecho_verificado"] is True
    six_words = vf.make_assessment(item(quote="coverage reached 92.4% in 2015 across"), ABSTRACT)  # 37 chars
    assert six_words["trecho_verificado"] is True
    word = "pneumonoultramicroscopicsilicovolcanoconiosis"  # 1 palavra, 45 caracteres
    by_chars = vf.make_assessment(item(quote=word), f"Case of {word} reported.")
    assert by_chars["veredito"] == "SUSTENTA" and by_chars["trecho_verificado"] is True
    nao = vf.make_assessment(item("NAO_SUSTENTA", quote="coverage reached 92.4%"), ABSTRACT)
    assert nao["veredito"] == "NAO_SUSTENTA" and nao["trecho_verificado"] is True and nao["flags"] == []


def test_second_pass_trigger():
    def a(verdict, quote=QUOTE, problem=None):
        return vf.make_assessment(item(verdict, quote, problem), ABSTRACT)

    assert vf.needs_second_pass(a("NAO_SUSTENTA", ""))
    assert vf.needs_second_pass(a("CONTRADIZ"))
    assert vf.needs_second_pass(a("SUSTENTA_PARCIALMENTE", problem="afirma_mais_que_a_fonte"))
    assert not vf.needs_second_pass(a("SUSTENTA_PARCIALMENTE"))
    assert not vf.needs_second_pass(a("SUSTENTA"))
    assert not vf.needs_second_pass(a("NAO_VERIFICAVEL", ""))
    assert vf.needs_second_pass(a("SUSTENTA", quote="invented passage"))  # trecho não verificado
    assert vf.needs_second_pass(vf.make_assessment(None, ABSTRACT))


def test_combine_passes_and_divergente():
    def a(verdict, quote=QUOTE):
        return vf.make_assessment(item(verdict, quote), ABSTRACT)

    first = a("NAO_SUSTENTA", "")
    assert vf.combine_passes(first, a("NAO_SUSTENTA", "")) == ("NAO_SUSTENTA", [])
    assert vf.combine_passes(first, a("SUSTENTA"))[0] == vf.DIVERGENTE
    verdict, flags = vf.combine_passes(first, None)
    assert verdict == "NAO_SUSTENTA" and "segunda_passada_pendente" in flags
    sustenta = a("SUSTENTA")
    assert vf.combine_passes(sustenta, a("NAO_SUSTENTA", ""))[0] == "SUSTENTA"  # não houve gatilho
    unverified = a("SUSTENTA", "invented passage")
    assert vf.combine_passes(unverified, a("SUSTENTA"))[0] == vf.DIVERGENTE
    assert vf.combine_passes(unverified, a("SUSTENTA", "invented passage"))[0] == "NAO_VERIFICAVEL"


# ---------------------------------------------------------------------------
# Prompts e construção das requisições
# ---------------------------------------------------------------------------
def test_system_rubric_is_long_enough_to_cache():
    assert len(prompts.SYSTEM_RUBRIC) // 4 >= 1100
    for word in ("SUSTENTA_PARCIALMENTE", "trecho_fonte", "nivel_evidencia", "afirma_mais_que_a_fonte"):
        assert word in prompts.SYSTEM_RUBRIC


def test_evidence_block_and_paragraph_trim():
    record = {"key": "k", "titulo_bib": "T", "doi": DOI, "fulltext": "palavra " * 100, "fulltext_fonte": "scielo",
              "abstract": "resumo", "consensus": [{"quote": "q1"}]}
    text, level = prompts.build_evidence_block(record, 100)
    assert level == "texto_completo" and "TRUNCADO" in text and 'nivel="texto_completo"' in text
    assert '1. "q1"' in text
    assert prompts.build_evidence_block({"key": "k"}, 100) == ("", "nenhum")
    text, level = prompts.build_evidence_block({"key": "k", "abstract": ABSTRACT}, 100)
    assert level == "resumo" and ABSTRACT in text

    sentence = "Esta é a sentença [cit]."
    paragraph = "x" * 2000 + sentence + "y" * 2000
    trimmed = prompts.trim_paragraph(paragraph, sentence)
    assert len(trimmed) <= 1200 and sentence in trimmed
    assert prompts.trim_paragraph("curto", "c") == "curto"


def test_parse_items():
    assert av.parse_items('```json\n[{"id": "a"}]\n```') == [{"id": "a"}]
    with pytest.raises(ValueError):
        av.parse_items("sem json")
    with pytest.raises(ValueError):
        av.parse_items('{"id": "a"}')


def test_request_construction_cache_control_and_identical_prefixes():
    settings = av.Settings(max_contextos=2)
    groups = {"kA": [ctx(1, "kA"), ctx(2, "kA"), ctx(3, "kA")], "kB": [ctx(4, "kB")]}
    evidence = {
        "kA": '<evidencia chave="kA" nivel="resumo">\nA\n</evidencia>',
        "kB": '<evidencia chave="kB" nivel="resumo">\nB\n</evidencia>',
    }
    plan = av.plan_requests(groups, evidence, "Introdução", settings, 1)
    assert len(plan) == 3
    ids = [r.custom_id for r in plan]
    assert len(set(ids)) == 3 and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids)
    for req in plan:
        system = req.params["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"} and system[0]["text"] == prompts.SYSTEM_RUBRIC
        assert "temperature" not in req.params and req.params["model"] == common.DEFAULT_MODEL
    a_reqs = [r for r in plan if r.key == "kA"]
    b_req = next(r for r in plan if r.key == "kB")

    def prefix(req):
        return json.dumps([req.params["system"], req.params["messages"][0]["content"][0]], sort_keys=True)

    assert len(a_reqs) == 2 and prefix(a_reqs[0]) == prefix(a_reqs[1])  # prefixo idêntico entre requisições
    assert a_reqs[0].params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert a_reqs[0].params["messages"][0]["content"][1] != a_reqs[1].params["messages"][0]["content"][1]
    assert "cache_control" not in b_req.params["messages"][0]["content"][0]  # chave com uma só requisição
    assert json.dumps(a_reqs[0].params["system"]) == json.dumps(b_req.params["system"])

    with_temperature = av.plan_requests({"kB": groups["kB"]}, evidence, "Introdução",
                                        av.Settings(send_temperature=True), 1)
    assert with_temperature[0].params["temperature"] == 0.0

    second = av.plan_requests({"kB": groups["kB"]}, evidence, "Introdução", settings, 2, cached_before={"kB"})[0]
    assert "cache_control" in second.params["messages"][0]["content"][0]
    text_b = second.params["messages"][0]["content"][1]["text"]
    assert "Revisão independente" in text_b and prompts.INSTRUCAO_PASSADA_1 not in text_b
    assert second.params["system"] == b_req.params["system"]
    assert second.params["messages"][0]["content"][0]["text"] == b_req.params["messages"][0]["content"][0]["text"]
    assert second.custom_id != b_req.custom_id


# ---------------------------------------------------------------------------
# Execução com cliente simulado: checkpoint, segunda passada, lote
# ---------------------------------------------------------------------------
class FakeApiError(Exception):
    status_code = 400


def fake_message(items):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(items))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                              cache_creation_input_tokens=10, cache_read_input_tokens=5),
        stop_reason="end_turn",
    )


def respond(policy, params):
    content = params["messages"][0]["content"]
    key = re.search(r'chave="([^"]+)"', content[0]["text"]).group(1)
    passada = 2 if "Revisão independente" in content[1]["text"] else 1
    ids = re.findall(r"^id: (\S+)$", content[1]["text"], re.M)
    return fake_message([policy(passada, key, cid) for cid in ids]), key, passada


def good(cid):
    return {"id": cid, "veredito": "SUSTENTA", "trecho_fonte": QUOTE, "justificativa": "ok",
            "problema": None, "nivel_evidencia": "resumo"}


def critical(cid):
    return {"id": cid, "veredito": "NAO_SUSTENTA", "trecho_fonte": "", "justificativa": "não consta",
            "problema": "fora_de_escopo", "nivel_evidencia": "resumo"}


def always_good(passada, key, cid):
    return good(cid)


class FakeBatches:
    def __init__(self, policy):
        self.policy = policy
        self.created = []
        self.store = {}
        self.polls = {}

    def create(self, requests):
        batch_id = f"msgbatch_{len(self.created) + 1}"
        self.created.append(batch_id)
        self.store[batch_id] = requests
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        status = "in_progress" if self.polls[batch_id] == 1 else "ended"
        return SimpleNamespace(id=batch_id, processing_status=status)

    def results(self, batch_id):
        for request in self.store[batch_id]:
            message, _key, _passada = respond(self.policy, request["params"])
            yield SimpleNamespace(custom_id=request["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=message))


class FakeMessages:
    def __init__(self, policy, fail_keys=()):
        self.policy = policy
        self.fail_keys = set(fail_keys)
        self.calls = []
        self.batches = FakeBatches(policy)

    def create(self, **params):
        message, key, passada = respond(self.policy, params)
        if key in self.fail_keys:
            raise FakeApiError()
        self.calls.append((key, passada))
        return message


def fake_client(policy=always_good, fail_keys=()):
    return SimpleNamespace(messages=FakeMessages(policy, fail_keys))


def write_evidence(out_dir, key, abstract=ABSTRACT):
    path = out_dir / "evidencias" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "titulo_bib": TITLE, "abstract": abstract, "fulltext": "", "consensus": [],
              "identidade_verificada": True}
    path.write_text(json.dumps(record), encoding="utf-8")


def run(tmp_path, contexts, client, settings, proprios=frozenset()):
    refs = {c["key"]: make_ref(c["key"], doi=DOI) for c in contexts}
    return av.run_avaliacao(
        capitulo="Introdução", contexts=contexts, refs=refs, proprios=set(proprios), out_dir=tmp_path,
        pilot_dir=tmp_path / "piloto" / "introducao", settings=settings, client=client, sleep=lambda s: None,
    )


def test_checkpoint_resume_skips_done_keys(tmp_path):
    for key in ("kA", "kB"):
        write_evidence(tmp_path, key)
    contexts = [ctx(1, "kA"), ctx(2, "kB")]
    settings = av.Settings(modo="sync", concorrencia=1)

    first = run(tmp_path, contexts, fake_client(fail_keys={"kB"}), settings)
    assert first["incompleto"] is True and first["metadados"]["status"] == "incompleto"
    assert {r["key"]: r["veredito_final"] for r in first["rows"]} == {"kA": "SUSTENTA", "kB": "PENDENTE"}
    assert first["metadados"]["falhas"] == 1

    client = fake_client()
    second = run(tmp_path, contexts, client, settings)
    assert client.messages.calls == [("kB", 1)]  # kA veio do checkpoint
    assert second["incompleto"] is False
    assert {r["key"]: r["veredito_final"] for r in second["rows"]} == {"kA": "SUSTENTA", "kB": "SUSTENTA"}
    usage = second["metadados"]["uso"]
    assert usage["sync"] == {"input_tokens": 200, "output_tokens": 100,
                             "cache_creation_input_tokens": 20, "cache_read_input_tokens": 10}
    assert usage["batch"]["input_tokens"] == 0

    third_client = fake_client()
    run(tmp_path, contexts, third_client, settings)
    assert third_client.messages.calls == []


def test_second_pass_divergente_and_agreement(tmp_path):
    for key in ("kA", "kB", "kC"):
        write_evidence(tmp_path, key)
    contexts = [ctx(1, "kA"), ctx(2, "kB"), ctx(3, "kC")]

    def policy(passada, key, cid):
        if key == "kA":
            return critical(cid) if passada == 1 else good(cid)  # discordam
        if key == "kB":
            return critical(cid)  # concordam em NAO_SUSTENTA
        return good(cid)  # sem gatilho, sem segunda passada

    client = fake_client(policy)
    result = run(tmp_path, contexts, client, av.Settings(modo="sync", concorrencia=1))
    final = {r["key"]: r["veredito_final"] for r in result["rows"]}
    assert final == {"kA": "DIVERGENTE", "kB": "NAO_SUSTENTA", "kC": "SUSTENTA"}
    assert sorted(client.messages.calls) == [("kA", 1), ("kA", 2), ("kB", 1), ("kB", 2), ("kC", 1)]
    assert result["metadados"]["requisicoes"] == {"passada1": 3, "passada2": 2}
    row_a = next(r for r in result["rows"] if r["key"] == "kA")
    assert row_a["passada1"]["veredito"] == "NAO_SUSTENTA" and row_a["passada2"]["veredito"] == "SUSTENTA"
    saved = common.read_jsonl(tmp_path / "piloto" / "introducao" / "avaliacoes.jsonl")
    assert len(saved) == 3


def test_unverified_quote_triggers_second_pass(tmp_path):
    write_evidence(tmp_path, "kA")

    def policy(passada, key, cid):
        item_ = good(cid)
        if passada == 1:
            item_["trecho_fonte"] = "passagem inventada que não existe na fonte"
        return item_

    result = run(tmp_path, [ctx(1, "kA")], fake_client(policy), av.Settings(modo="sync", concorrencia=1))
    row = result["rows"][0]
    assert row["veredito_final"] == "DIVERGENTE"
    assert row["passada1"]["veredito"] == "NAO_VERIFICAVEL" and row["passada1"]["trecho_verificado"] is False
    assert "trecho_nao_encontrado" in row["flags"]


def test_no_evidence_means_no_api_call(tmp_path):
    client = fake_client()
    result = run(tmp_path, [ctx(1, "kSem")], client, av.Settings(modo="sync"))
    row = result["rows"][0]
    assert row["veredito_final"] == "NAO_VERIFICAVEL" and row["nivel_evidencia"] == "nenhum"
    assert row["flags"] == ["sem_evidencia"]
    assert client.messages.calls == []


def test_own_work_keys_are_not_sent_to_the_model(tmp_path):
    write_evidence(tmp_path, "kA")
    client = fake_client()
    contexts = [ctx(1, "kA"), ctx(2, "meu")]
    result = run(tmp_path, contexts, client, av.Settings(modo="sync"), proprios={"meu"})
    assert [r["key"] for r in result["rows"]] == ["kA"]
    assert client.messages.calls == [("kA", 1)]


def test_batch_mode_persists_id_polls_and_resumes_without_resubmitting(tmp_path):
    for key in ("kA", "kB"):
        write_evidence(tmp_path, key)
    contexts = [ctx(1, "kA"), ctx(2, "kB")]

    def policy(passada, key, cid):
        if key == "kA":
            return critical(cid) if passada == 1 else good(cid)
        return good(cid)

    client = fake_client(policy)
    settings = av.Settings(modo="batch", poll_inicial_s=0.0)
    first = run(tmp_path, contexts, client, settings)
    batches = client.messages.batches
    assert batches.created == ["msgbatch_1", "msgbatch_2"]  # passada 1 e passada 2
    assert first["metadados"]["lotes"] == ["msgbatch_1", "msgbatch_2"]
    assert batches.polls["msgbatch_1"] == 2  # uma consulta em andamento, outra encerrada
    assert {r["key"]: r["veredito_final"] for r in first["rows"]} == {"kA": "DIVERGENTE", "kB": "SUSTENTA"}
    usage = first["metadados"]["uso"]
    assert usage["batch"]["input_tokens"] == 300 and usage["sync"]["input_tokens"] == 0
    pdir = tmp_path / "piloto" / "introducao"
    request_files = sorted((pdir / "lotes").glob("lote_p1_*.jsonl"))
    assert len(request_files) == 1
    lines = [json.loads(line) for line in request_files[0].read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and all({"custom_id", "params"} <= set(line) for line in lines)
    state = common.read_json(pdir / "checkpoint.json")
    assert [b["id"] for b in state["lotes"]] == ["msgbatch_1", "msgbatch_2"] and all(b["baixado"] for b in state["lotes"])

    again = run(tmp_path, contexts, client, settings)
    assert batches.created == ["msgbatch_1", "msgbatch_2"]  # nada reenviado
    assert again["rows"] == first["rows"]


def test_batch_with_unrecorded_submission_refuses_to_resubmit(tmp_path):
    write_evidence(tmp_path, "kA")
    pdir = tmp_path / "piloto" / "introducao"
    checkpoint = av.Checkpoint(pdir / "checkpoint.json", {"capitulo": "Introdução", "modelo": "m"})
    checkpoint.batches.append({"id": None, "passada": 1, "status": "enviando", "baixado": False, "custom_ids": ["x"]})
    checkpoint.save()
    with pytest.raises(RuntimeError, match="sem id registrado"):
        run(tmp_path, [ctx(1, "kA")], fake_client(), av.Settings(modo="batch", poll_inicial_s=0.0))


def test_sync_retries_retryable_errors():
    sleeps = []
    attempts = []

    class Overloaded(Exception):
        status_code = 529

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Overloaded()
        return "ok"

    assert av.with_retry(flaky, sleeps.append) == "ok" and sleeps == [3.0, 6.0]
    with pytest.raises(av.ApiCallError):
        av.with_retry(lambda: (_ for _ in ()).throw(FakeApiError()), sleeps.append)
    with pytest.raises(KeyError):  # erros de programação não são engolidos
        av.with_retry(lambda: {}["x"], sleeps.append)


# ---------------------------------------------------------------------------
# Obras próprias
# ---------------------------------------------------------------------------
def test_extract_path_targets():
    latex = r"Ver \path{data/a\_b.csv} e \path{scripts/x.py}, de novo \path{scripts/x.py}."
    assert pr.extract_path_targets(latex) == ["data/a_b.csv", "scripts/x.py"]
    assert pr.extract_path_targets("sem caminho") == []


def test_own_work_artifact_check(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "foo.csv").write_text("x", encoding="utf-8")
    contexts = [
        {**ctx(1, "meu"), "sentence_latex": r"Ver \path{data/foo.csv}."},
        {**ctx(2, "meu"), "sentence_latex": r"Ver \path{data/foo.csv} e \path{data/missing.csv}."},
        {**ctx(3, "meu"), "sentence_latex": "Sem caminho."},
        {**ctx(4, "meu"), "sentence_latex": r"Ver \path{scripts/versionado.py}."},
        {**ctx(5, "meu"), "sentence_latex": r"Ver \path{data/*.csv}."},
    ]
    rows = pr.check_artifacts(contexts, tmp_path, tracked={"scripts/versionado.py"})
    assert [r["veredito"] for r in rows] == [
        pr.ARTEFATO_OK, pr.ARTEFATO_AUSENTE, pr.SEM_ARTEFATO, pr.ARTEFATO_OK, pr.ARTEFATO_OK,
    ]
    assert rows[1]["ausentes"] == ["data/missing.csv"]


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def row(i, key, verdict, level="resumo", quote="", verified=None, flags=(), problem=None):
    return {"id": f"id{i:02d}", "key": key, "file": "text/latex/cap1.tex", "line": 10 + i,
            "veredito_final": verdict, "nivel_evidencia": level, "trecho_fonte": quote,
            "trecho_verificado": verified, "problema": problem, "justificativa": f"just {i}",
            "flags": list(flags)}


def test_report_counts_lists_and_cost():
    long_quote = " ".join(f"w{n}" for n in range(1, 41))
    rows = [
        row(1, "kA", "SUSTENTA", "texto_completo", long_quote, True),
        row(2, "kA", "SUSTENTA", "texto_completo", QUOTE, True),
        row(3, "kB", "NAO_SUSTENTA", "resumo", "", None, problem="fora_de_escopo"),
        row(4, "kC", "DIVERGENTE", "resumo", "", None),
        row(5, "kD", "NAO_VERIFICAVEL", "nenhum", flags=["sem_evidencia"]),
    ]
    meta = {"modelo": "claude-sonnet-5-5", "data_utc": "2026-11-07T00:00:00+00:00", "status": "completo",
            "parametros": {"modo": "batch"},
            "uso": {"batch": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                    "sync": {"input_tokens": 1_000_000, "output_tokens": 0,
                             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}
    artefatos = [{"id": "x", "key": "meu", "file": "a.tex", "line": 3, "veredito": "ARTEFATO_AUSENTE",
                  "alvos": ["data/x.csv"], "ausentes": ["data/x.csv"]}]
    classificacao = {"kE": {"classe": "cinzenta", "nivel": "nenhum"}, "kA": {"classe": "literatura"}}
    precos = {"input": 3, "output": 15, "cache_write": 3.75, "cache_read": 0.3, "batch_desconto": 0.5}
    text = rel.build_report(rows, meta, artefatos, classificacao, "Introdução", precos)

    assert "| SUSTENTA | 2 | 40,0 |" in text
    assert "| NAO_SUSTENTA | 1 | 20,0 |" in text and "| DIVERGENTE | 1 | 20,0 |" in text
    assert "| texto_completo | 2 |" in text and "| resumo | 2 |" in text and "| nenhum | 1 |" in text
    assert "text/latex/cap1.tex:11" in text
    assert "w25…" in text and "w26" not in text  # trecho limitado a 25 palavras
    assert "## DIVERGENTE (revisão humana)" in text and "- text/latex/cap1.tex:14 `kC`" in text
    assert "### sem evidência recuperada (1)" in text
    assert "ARTEFATO_AUSENTE: 1" in text and "ausentes: data/x.csv" in text
    assert "- `kE`: nível de evidência nenhum" in text and "`kA`: nível" not in text
    assert "| batch | 1.000.000 | 1.000.000 | 0 | 0 |" in text
    assert "batch 9,00; sync 3,00; total 12,00" in text


def test_number_formatting():
    assert rel.fmt_int(1234567) == "1.234.567"
    assert rel.fmt_dec(12.34) == "12,3"
    assert rel.fmt_dec(1234.5, 2) == "1.234,50"


# ---------------------------------------------------------------------------
# Utilitários e CLI
# ---------------------------------------------------------------------------
def test_atomic_write_retries_on_permission_error(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = []

    def locked(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("bloqueado")
        return real_replace(src, dst)

    monkeypatch.setattr(common.os, "replace", locked)
    sleeps = []
    common.write_json_atomic(tmp_path / "a.json", {"x": 1}, sleep=sleeps.append)
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"x": 1}
    assert len(calls) == 3 and sleeps == [0.1, 0.2]


def test_load_env_var_reads_env_file_without_exposing_it(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comentário\nATESTADO_TESTE_KEY="valor-secreto"\n', encoding="utf-8")
    monkeypatch.delenv("ATESTADO_TESTE_KEY", raising=False)
    assert common.load_env_var("ATESTADO_TESTE_KEY", env) == "valor-secreto"
    assert common.load_env_var("OUTRA_ATESTADO_TESTE", env) == ""
    monkeypatch.setenv("ATESTADO_TESTE_KEY", "do-ambiente")
    assert common.load_env_var("ATESTADO_TESTE_KEY", env) == "do-ambiente"


def test_slug_and_select_contexts():
    assert common.slug("Introdução") == "introducao"
    records = [ctx(1, "kA"), ctx(2, "kB"), ctx(3, "kA", chapter="Introdução Geral"), ctx(4, "kA", chapter="Métodos")]
    assert [r["id"] for r in cli.select_contexts(records, "Introdução", [])] == ["id01", "id02"]
    assert [r["id"] for r in cli.select_contexts(records, "Introdução", ["kB"])] == ["id02"]


BIB = (
    "@article{kA, author={Silva, Ana}, title={%s}, year={2018}, doi={%s}}\n"
    "@misc{meu, author={Jardim, Marco}, title={Pacote publicado}, year={2026}}\n" % (TITLE, DOI)
)


def write_inputs(tmp_path, contexts):
    (tmp_path / "citacoes.jsonl").write_text("\n".join(json.dumps(c) for c in contexts) + "\n", encoding="utf-8")
    bib = tmp_path / "ref.bib"
    bib.write_text(BIB, encoding="utf-8")
    return bib


def test_cli_evidencias_end_to_end(tmp_path, monkeypatch, capsys):
    contexts = [ctx(1, "kA"), {**ctx(2, "meu"), "sentence_latex": r"Ver \path{pacote/arquivo.py}."}]
    bib = write_inputs(tmp_path, contexts)
    (tmp_path / "root" / "pacote").mkdir(parents=True)
    (tmp_path / "root" / "pacote" / "arquivo.py").write_text("x", encoding="utf-8")
    http = FakeHttp([(openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json")])
    monkeypatch.setattr(cli, "HttpClient", lambda mailto, min_interval: http)
    monkeypatch.setattr(cli, "ROOT", tmp_path / "root")
    monkeypatch.setattr(cli.pr, "git_tracked_files", lambda root: set())
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: EMAIL)

    code = cli.main(["evidencias", "--capitulo", "Introdução", "--out", str(tmp_path), "--bib", str(bib)])
    assert code == 0
    pdir = tmp_path / "piloto" / "introducao"
    resumo = (pdir / "evidencias_resumo.md").read_text(encoding="utf-8")
    assert "| `kA` | literatura | resumo | openalex | sim |" in resumo
    assert "| `meu` | proprio | artefato |" in resumo and "ARTEFATO_OK=1" in resumo
    assert common.read_json(pdir / "classificacao.json")["kA"]["nivel"] == "resumo"
    assert (tmp_path / "evidencias" / "kA.json").is_file()
    assert common.read_json(pdir / "artefatos.json")[0]["veredito"] == pr.ARTEFATO_OK
    assert not any(net.is_forbidden_host(u) for u in http.requested)


def test_cli_avaliar_and_relatorio_never_expose_api_key(tmp_path, monkeypatch, capsys):
    secret = "sk-ant-segredo-de-teste-123"
    bib = write_inputs(tmp_path, [ctx(1, "kA")])
    write_evidence(tmp_path, "kA")
    client = fake_client()
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: secret)
    monkeypatch.setattr(cli.avaliar_mod, "make_client", lambda api_key: client)
    base = ["--capitulo", "Introdução", "--out", str(tmp_path)]

    assert cli.main(["avaliar", *base, "--bib", str(bib), "--modo", "sync", "--concorrencia", "1"]) == 0
    assert client.messages.calls == [("kA", 1)]
    assert cli.main(["relatorio", *base]) == 0
    pdir = tmp_path / "piloto" / "introducao"
    assert "claude-sonnet-5-5" in (pdir / "metadados.json").read_text(encoding="utf-8")
    assert "| SUSTENTA | 1 | 100,0 |" in (pdir / "relatorio.md").read_text(encoding="utf-8")
    shown = capsys.readouterr()
    assert secret not in shown.out and secret not in shown.err
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8", errors="replace")


def test_cli_avaliar_without_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    bib = write_inputs(tmp_path, [ctx(1, "kA")])
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: "")
    code = cli.main(["avaliar", "--capitulo", "Introdução", "--out", str(tmp_path), "--bib", str(bib)])
    assert code == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Fontes locais fornecidas pelo autor
# ---------------------------------------------------------------------------
def long_text(prefix="", n=700):
    return prefix + " " + "Conteúdo qualquer sem relação com o assunto. " * n


@pytest.fixture
def fontes(tmp_path):
    pasta = tmp_path / "fontes"
    pasta.mkdir()
    return pasta


def test_local_prefers_txt_over_pdf_xml_html(fontes, monkeypatch):
    monkeypatch.setattr(ev, "pdf_to_text", lambda data: long_text("do pdf"))
    (fontes / "k.txt").write_text(long_text("do txt"), encoding="utf-8")
    (fontes / "k.pdf").write_bytes(b"%PDF-fake")
    (fontes / "k.xml").write_bytes(jats(TITLE, DOI))
    (fontes / "k.html").write_text(html_page(TITLE, DOI), encoding="utf-8")
    assert ev.find_local_fulltext(fontes, "k").path.name == "k.txt"
    (fontes / "k.txt").unlink()
    assert ev.find_local_fulltext(fontes, "k").path.name == "k.pdf"
    (fontes / "k.pdf").unlink()
    found = ev.find_local_fulltext(fontes, "k")
    assert found.path.name == "k.xml" and "Texto do artigo 1" in found.text and "<p>" not in found.text
    (fontes / "k.xml").unlink()
    found = ev.find_local_fulltext(fontes, "k")
    assert found.path.name == "k.html" and "menu do site" not in found.text and "<p>" not in found.text
    assert not found.preprint


def test_local_small_txt_falls_through_to_pdf(fontes, monkeypatch):
    monkeypatch.setattr(ev, "pdf_to_text", lambda data: long_text("do pdf"))
    (fontes / "k.txt").write_text("curto demais " * 20, encoding="utf-8")
    (fontes / "k.pdf").write_bytes(b"%PDF-fake")
    found = ev.find_local_fulltext(fontes, "k")
    assert found.path.name == "k.pdf" and found.text.startswith("do pdf")


def test_local_pdf_without_fitz_falls_to_xml(fontes, monkeypatch):
    monkeypatch.setattr(ev, "pdf_to_text", lambda data: None)
    (fontes / "k.pdf").write_bytes(b"%PDF-fake")
    (fontes / "k.xml").write_bytes(jats(TITLE, DOI))
    assert ev.find_local_fulltext(fontes, "k").path.name == "k.xml"


def test_local_whitespace_does_not_count_towards_minimum(fontes):
    (fontes / "k.txt").write_text("a" + " \n\t " * 5000, encoding="utf-8")
    assert ev.find_local_fulltext(fontes, "k") is None


def test_local_ignores_attempts_and_helper_files(fontes):
    for name in ("k_tentativa2.html", "k_tentativa2_html.txt", "k_oa.xml", "k_busca_complementar.json",
                 "k_README.md", "k_equacoes_conferidas.txt", "candidato_k.txt", "outra.txt"):
        (fontes / name).write_text(long_text(), encoding="utf-8")
    assert ev.find_local_fulltext(fontes, "k") is None
    assert ev.find_local_fulltext(fontes / "inexistente", "k") is None
    assert ev.find_local_fulltext(None, "k") is None


def test_local_preprint_is_only_a_fallback_and_flagged(fontes):
    (fontes / "k_preprint.txt").write_text(long_text("preprint"), encoding="utf-8")
    found = ev.find_local_fulltext(fontes, "k")
    assert found.path.name == "k_preprint.txt" and found.preprint
    (fontes / "k.txt").write_text(long_text("principal"), encoding="utf-8")
    found = ev.find_local_fulltext(fontes, "k")
    assert found.path.name == "k.txt" and not found.preprint

    (fontes / "k.txt").unlink()
    (fontes / "k_preprint.txt").write_text(TITLE + "\n" + long_text(), encoding="utf-8")
    col = ev.Collector(FakeHttp([]), unpaywall_email=EMAIL, fontes_locais=fontes)
    record = ev.collect_evidence(col, make_ref(key="k"), "literatura")
    assert record["fulltext_fonte"] == "local:k_preprint.txt"
    assert ev.FLAG_PREPRINT in record["flags"] and ev.FLAG_IDENTIDADE_LOCAL not in record["flags"]


def test_local_identity_confirmed_by_title_or_doi(fontes):
    ref = make_ref(key="k", doi=DOI)
    ok, _ = ev.local_identity(ref, f"Probabilistic Linkage in Cancer Registries\nAutores\n" + long_text())
    assert ok
    ok, method = ev.local_identity(ref, "Outro título totalmente diferente. " + f"doi: {DOI}. " + long_text())
    assert ok and method == "doi_no_texto"
    ok, _ = ev.local_identity(make_ref(key="k"), "Cancer registries: probabilistic linkage evaluated.\n" + long_text())
    assert ok


def test_local_identity_flag_when_title_is_absent(fontes):
    (fontes / "k.txt").write_text(long_text("Receita de bolo de cenoura."), encoding="utf-8")
    col = ev.Collector(FakeHttp([]), unpaywall_email=EMAIL, fontes_locais=fontes)
    record = ev.collect_evidence(col, make_ref(key="k"), "literatura")
    assert record["fulltext"] and record["identidade_verificada"] is False
    assert record["flags"] == [ev.FLAG_IDENTIDADE_LOCAL]
    assert record["fulltext_fonte"] == "local:k.txt" and record["fulltext_tamanho"] == len(record["fulltext"])
    assert ev.evidence_level(record) == "texto_completo"
    md = fontes.parent / "resumo.md"
    ev.write_resumo(md, [record], "Introdução")
    assert ev.FLAG_IDENTIDADE_LOCAL in md.read_text(encoding="utf-8")
    assert "local:k.txt" in md.read_text(encoding="utf-8")


def test_local_cinzenta_key_uses_local_file(fontes, tmp_path):
    ref = make_ref(key="Oliveira2026tb", etype="mastersthesis", title="Óbito por tuberculose pulmonar")
    assert ev.classify_key(ref) == "cinzenta"
    (fontes / "Oliveira2026tb.txt").write_text(
        "Óbito por tuberculose pulmonar\n" + long_text(), encoding="utf-8")
    col = ev.Collector(FakeHttp([]), unpaywall_email=EMAIL, fontes_locais=fontes)
    record = ev.get_evidence(col, ref, "cinzenta", tmp_path / "saida")
    assert record["classe"] == "cinzenta" and record["fulltext_fonte"] == "local:Oliveira2026tb.txt"
    assert record["identidade_verificada"] is True and record["flags"] == []


def test_local_text_skips_network_fulltext_but_keeps_abstract(fontes):
    (fontes / "k.txt").write_text(TITLE + "\n" + long_text(), encoding="utf-8")
    http = FakeHttp([
        (openalex_doi_prefix(), 200, work(abstract=ABSTRACT, pmcid="PMC123"), "application/json"),
        ("https://www.ebi.ac.uk/europepmc", 200, jats(TITLE, DOI), "application/xml"),
    ])
    col = ev.Collector(http, unpaywall_email=EMAIL, fontes_locais=fontes)
    record = ev.collect_evidence(col, make_ref(key="k", doi=DOI), "literatura")
    assert record["fulltext_fonte"] == "local:k.txt" and record["abstract"] == ABSTRACT
    assert record["identidade_verificada"] is True
    assert all(url.startswith("https://api.openalex.org/") for url in http.requested)


def test_without_local_dir_network_is_used_as_before(fontes):
    http = FakeHttp([
        (openalex_doi_prefix(), 200, work(abstract=ABSTRACT, pmcid="PMC123"), "application/json"),
        ("https://www.ebi.ac.uk/europepmc", 200, jats(TITLE, DOI), "application/xml"),
    ])
    col = ev.Collector(http, unpaywall_email=EMAIL, fontes_locais=fontes)
    record = ev.collect_evidence(col, make_ref(key="k", doi=DOI), "literatura")
    assert record["fulltext_fonte"] == "europepmc" and "flags" not in record


def test_local_text_replaces_cached_network_record(fontes, tmp_path):
    saida = tmp_path / "saida"
    ref = make_ref(key="k", doi=DOI)
    oa = (openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json")
    col = ev.Collector(FakeHttp([oa]), unpaywall_email=EMAIL, fontes_locais=fontes)
    assert ev.get_evidence(col, ref, "literatura", saida)["fulltext_fonte"] == ""
    (fontes / "k.txt").write_text(TITLE + "\n" + long_text(), encoding="utf-8")
    refeito = ev.get_evidence(col, ref, "literatura", saida)
    assert refeito["fulltext_fonte"] == "local:k.txt"


def test_cli_fontes_locais_option(tmp_path, monkeypatch):
    bib = write_inputs(tmp_path, [ctx(1, "kA")])
    pasta = tmp_path / "meus_textos"
    pasta.mkdir()
    (pasta / "kA.txt").write_text(TITLE + "\n" + long_text(), encoding="utf-8")
    http = FakeHttp([(openalex_doi_prefix(), 200, work(abstract=ABSTRACT), "application/json")])
    monkeypatch.setattr(cli, "HttpClient", lambda mailto, min_interval: http)
    monkeypatch.setattr(cli.pr, "git_tracked_files", lambda root: set())
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: EMAIL)
    code = cli.main(["evidencias", "--capitulo", "Introdução", "--out", str(tmp_path), "--bib", str(bib),
                     "--fontes-locais", str(pasta)])
    assert code == 0
    resumo = (tmp_path / "piloto" / "introducao" / "evidencias_resumo.md").read_text(encoding="utf-8")
    assert "| `kA` | literatura | texto_completo | local:kA.txt | sim |" in resumo
    assert cli.build_parser().parse_args(["evidencias", "--capitulo", "x"]).fontes_locais == ".local/fontes"


# ---------------------------------------------------------------------------
# Regra de conflito numérico no prompt e conferência mecânica de números
# ---------------------------------------------------------------------------
def test_rubric_has_numeric_conflict_and_qualifier_rules():
    rubric = prompts.SYSTEM_RUBRIC
    assert len(rubric) // 4 >= 1100
    assert 'problema="numero_divergente"' in rubric and "39,3% no resumo e 39,4%" in rubric
    assert "não pode ser SUSTENTA sem que o conflito seja registrado" in rubric
    assert "valor que discorda do valor da tese" in rubric
    assert 'problema="afirma_mais_que_a_fonte"' in rubric and "causa básica ou associada" in rubric
    assert "—" not in rubric and "–" not in rubric


def test_rubric_system_block_is_byte_identical_across_requests():
    settings = av.Settings(max_contextos=2)
    contexts = [ctx(1, "kA", chapter="Introdução"), ctx(2, "kA", chapter="Métodos"), ctx(3, "kB", chapter="Anexo")]
    evidence = {k: f'<evidencia chave="{k}" nivel="resumo">\n{k}\n</evidencia>' for k in ("kA", "kB")}
    groups = av.group_by_key(contexts)
    plans = [
        *av.plan_requests(groups, evidence, "Introdução", settings, 1),
        *av.plan_requests(groups, evidence, common.TESE_LABEL, settings, 1, global_scope=True),
        *av.plan_requests(groups, evidence, common.TESE_LABEL, settings, 2, {"kA"}, global_scope=True),
    ]
    blocks = {json.dumps(r.params["system"], sort_keys=True) for r in plans}
    assert len(blocks) == 1
    assert plans[0].params["system"][0]["text"] == prompts.SYSTEM_RUBRIC


def test_extract_numbers_pt_br_formats():
    sentence = "A prevalência foi de 39,3% em 84.368 casos (21,4%), 2015 e 3 grupos de COVID-19 [cit]."
    found = [text for text, _int, _dec in numeros.extract_numbers(sentence)]
    assert found == ["39,3%", "84.368", "21,4%", "2015"]  # 1 dígito solto e COVID-19 ignorados
    assert numeros.extract_numbers("1.234.567,89 e 12,5")[0][1:] == ("1234567", "89")
    assert numeros.extract_numbers("sem números [cit].") == []


def test_numbers_match_between_pt_and_en_formats():
    evidence = numeros.prepare_evidence("Prevalence was 39.3% (n = 84,368); 21.4 percent in 2015.")
    sentence = "A prevalência foi 39,3%, com 84.368 casos e 21,4% em 2015 [cit]."
    assert numeros.numeros_nao_encontrados(sentence, evidence) == []
    # inglês e português são equivalentes também para o espaço fino/inseparável como milhar
    spaced = numeros.prepare_evidence("n = 84\u00a0368 patients")
    assert numeros.numeros_nao_encontrados("Foram 84.368 casos [cit].", spaced) == []
    # sem agrupamento
    assert numeros.numeros_nao_encontrados("Foram 84.368 casos [cit].", numeros.prepare_evidence("n=84368")) == []


def test_numbers_not_found_are_reported_without_false_matches():
    evidence = numeros.prepare_evidence("Prevalence was 39.3% and coverage reached 92.4%; total 139.3 and 1,250.")
    assert numeros.numeros_nao_encontrados("Foi de 39,4% [cit].", evidence) == ["39,4%"]
    assert numeros.numeros_nao_encontrados("Foi de 92% [cit].", evidence) == ["92%"]  # arredondamento não casa
    assert numeros.numeros_nao_encontrados("Foi de 13,9 [cit].", evidence) == ["13,9"]
    assert numeros.numeros_nao_encontrados("Foi de 39,3 [cit].", numeros.prepare_evidence("valor 139.3")) == ["39,3"]
    assert numeros.numeros_nao_encontrados("Foram 1.250 [cit].", evidence) == []
    assert numeros.numeros_nao_encontrados("Foi de 39,4% [cit].", "") == []  # sem evidência, nada a sinalar


# ---------------------------------------------------------------------------
# Escopo global (tese inteira)
# ---------------------------------------------------------------------------
def test_select_contexts_and_pilot_dir_for_whole_thesis(tmp_path):
    records = [ctx(1, "kA"), ctx(2, "kB", chapter="Métodos"), ctx(3, "kA", chapter="Apêndice A"), ctx(4, "kA", chapter="")]
    assert [r["id"] for r in cli.select_contexts(records, None, [])] == ["id01", "id02", "id03", "id04"]
    assert [r["id"] for r in cli.select_contexts(records, None, ["kA"])] == ["id01", "id03", "id04"]
    assert cli.pilot_dir(tmp_path, None) == tmp_path / "piloto" / "tese"
    assert cli.pilot_dir(tmp_path, "Introdução") == tmp_path / "piloto" / "introducao"


def test_cli_scope_options_are_mutually_exclusive_and_temperature_is_opt_in(capsys):
    parser = cli.build_parser()
    for command in ("evidencias", "avaliar", "relatorio"):
        assert parser.parse_args([command, "--escopo", "tese"]).capitulo is None
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--capitulo", "Introdução", "--escopo", "tese"])
        with pytest.raises(SystemExit):
            parser.parse_args([command])
    with pytest.raises(SystemExit):
        parser.parse_args(["avaliar", "--escopo", "global"])
    capsys.readouterr()
    assert parser.parse_args(["avaliar", "--escopo", "tese"]).temperature is None
    assert parser.parse_args(["avaliar", "--escopo", "tese", "--temperature", "0.2"]).temperature == 0.2
    assert av.Settings().send_temperature is False


def test_global_scope_plans_one_request_per_key_across_chapters():
    contexts = [
        ctx(1, "kA", chapter="Introdução"), ctx(2, "kA", chapter="Métodos"), ctx(3, "kA", chapter="Apêndice A"),
        ctx(4, "kB", chapter="Introdução"), ctx(5, "kB", chapter="Resultados"),
    ]
    evidence = {k: f'<evidencia chave="{k}" nivel="resumo">\n{k}\n</evidencia>' for k in ("kA", "kB")}
    plan = av.plan_requests(av.group_by_key(contexts), evidence, common.TESE_LABEL, av.Settings(), 1, global_scope=True)
    assert [(r.key, r.ids) for r in plan] == [("kA", ["id01", "id02", "id03"]), ("kB", ["id04", "id05"])]
    text_a = plan[0].params["messages"][0]["content"][1]["text"]
    assert "Escopo: tese completa" in text_a and "Capítulo da tese:" not in text_a
    for chapter in ("Introdução", "Métodos", "Apêndice A"):
        assert f"Capítulo: {chapter}" in text_a
    assert all("cache_control" not in r.params["messages"][0]["content"][0] for r in plan)  # uma requisição por chave

    by_chapter = av.plan_requests(av.group_by_key(contexts), evidence, "Introdução", av.Settings(), 1)
    assert "Capítulo da tese: Introdução" in by_chapter[0].params["messages"][0]["content"][1]["text"]
    assert "Capítulo: Métodos" not in by_chapter[0].params["messages"][0]["content"][1]["text"]


def test_global_scope_split_requests_share_cached_evidence_block():
    contexts = [ctx(i, "kA", chapter=f"Cap {i}") for i in range(1, 6)]
    evidence = {"kA": '<evidencia chave="kA" nivel="resumo">\nA\n</evidencia>'}
    plan = av.plan_requests(av.group_by_key(contexts), evidence, common.TESE_LABEL,
                            av.Settings(max_contextos=2), 1, global_scope=True)
    assert [r.ids for r in plan] == [["id01", "id02"], ["id03", "id04"], ["id05"]]
    first_blocks = [r.params["messages"][0]["content"][0] for r in plan]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in first_blocks)
    assert len({json.dumps(b, sort_keys=True) for b in first_blocks}) == 1


def evidence_fp(out_dir, key):
    record = common.read_json(out_dir / "evidencias" / f"{key}.json")
    return av.evidence_fingerprint(prompts.build_evidence_block(record, av.Settings().max_chars)[0])


def write_prior(out_dir, name, rows, checkpoint=None):
    pdir = out_dir / "piloto" / name
    common.write_jsonl_atomic(pdir / "avaliacoes.jsonl", rows)
    if checkpoint is not None:
        common.write_json_atomic(pdir / "checkpoint.json", checkpoint)


def run_global(out_dir, contexts, client, settings=None, reuse=True):
    refs = {c["key"]: make_ref(c["key"], doi=DOI) for c in contexts}
    return av.run_avaliacao(
        capitulo=common.TESE_LABEL, escopo="tese", contexts=contexts, refs=refs, proprios=set(), out_dir=out_dir,
        pilot_dir=out_dir / "piloto" / "tese", settings=settings or av.Settings(modo="sync", concorrencia=1),
        client=client, sleep=lambda s: None, reuse_root=(out_dir / "piloto") if reuse else None,
    )


def spy_client():
    client = fake_client()
    seen = []
    original = client.messages.create

    def create(**params):
        seen.append(params)
        return original(**params)

    client.messages.create = create
    return client, seen


def sent_ids(params):
    return re.findall(r"^id: (\S+)$", params["messages"][0]["content"][1]["text"], re.M)


GLOBAL_CONTEXTS = [
    ctx(1, "kA", chapter="Introdução"),
    ctx(2, "kA", sentence="A cobertura chegou a 92,4% em 2015 [cit].", chapter="Métodos"),
    ctx(3, "kB", chapter="Métodos"),
]


def test_global_run_reuses_only_final_verdicts_with_matching_fingerprint(tmp_path):
    for key in ("kA", "kB"):
        write_evidence(tmp_path, key)
    prior = [
        {**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")},
        {**row(3, "kB", "PENDENTE", flags=["pendente"]), "evidencia_fingerprint": evidence_fp(tmp_path, "kB")},
    ]
    write_prior(tmp_path, "introducao", prior)
    client, seen = spy_client()
    result = run_global(tmp_path, GLOBAL_CONTEXTS, client)

    final = {r["id"]: r for r in result["rows"]}
    assert final["id01"]["reaproveitado_de"] == "introducao"
    assert "reaproveitado_de=introducao" in final["id01"]["flags"]
    assert final["id01"]["chapter"] == "Introdução" and final["id01"]["veredito_final"] == "SUSTENTA"
    assert "reaproveitado_de" not in final["id02"] and "reaproveitado_de" not in final["id03"]  # PENDENTE não vale
    assert sorted(sent_ids(p) for p in seen) == [["id02"], ["id03"]]
    assert all("temperature" not in p for p in seen)
    meta = result["metadados"]
    assert meta["escopo"] == "tese" and meta["contextos_reaproveitados"] == 1
    assert meta["reaproveitados_de"] == {"introducao": 1}
    assert meta["parametros"]["temperature"] is None and meta["parametros"]["temperature_enviada"] is False
    # sinal mecânico: 92% não aparece na evidência (só 92.4%); 92,4% aparece
    assert final["id01"]["numeros_nao_encontrados"] == ["92%"]
    assert final["id02"]["numeros_nao_encontrados"] == []
    assert final["id03"]["veredito_final"] == "SUSTENTA"  # o sinal não altera o veredito

    again_client, again_seen = spy_client()
    again = run_global(tmp_path, GLOBAL_CONTEXTS, again_client)
    assert again_seen == [] and again["rows"] == result["rows"]


def test_global_run_sends_one_request_per_key_when_nothing_is_reusable(tmp_path):
    for key in ("kA", "kB"):
        write_evidence(tmp_path, key)
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "introducao", prior)
    write_evidence(tmp_path, "kA", abstract=ABSTRACT + " An extra sentence changes the evidence.")  # evidência mudou
    client, seen = spy_client()
    result = run_global(tmp_path, GLOBAL_CONTEXTS, client)
    assert sorted(sent_ids(p) for p in seen) == [["id01", "id02"], ["id03"]]  # kA atravessa dois capítulos
    assert result["metadados"]["contextos_reaproveitados"] == 0
    assert all("reaproveitado_de" not in r for r in result["rows"])


def test_global_run_without_reuse_root_regrades_everything(tmp_path):
    for key in ("kA", "kB"):
        write_evidence(tmp_path, key)
    write_prior(tmp_path, "introducao",
                [{**row(1, "kA", "SUSTENTA"), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}])
    client, seen = spy_client()
    run_global(tmp_path, GLOBAL_CONTEXTS, client, reuse=False)
    assert sorted(sent_ids(p) for p in seen) == [["id01", "id02"], ["id03"]]


def test_global_split_requests_share_cache_control_in_a_real_run(tmp_path):
    write_evidence(tmp_path, "kA")
    contexts = [ctx(i, "kA", chapter=f"Cap {i}") for i in range(1, 4)]
    client, seen = spy_client()
    run_global(tmp_path, contexts, client, av.Settings(modo="sync", concorrencia=1, max_contextos=2))
    assert len(seen) == 2
    blocks = [p["messages"][0]["content"][0] for p in seen]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in blocks) and blocks[0] == blocks[1]


@pytest.mark.parametrize("evidence_changed", [False, True])
def test_reuse_of_legacy_rows_without_fingerprint_uses_old_checkpoint(tmp_path, evidence_changed):
    write_evidence(tmp_path, "kA")
    record = common.read_json(tmp_path / "evidencias" / "kA.json")
    old_text = prompts.build_evidence_block(record, av.Settings().max_chars)[0]
    cid = av.make_custom_id(1, "kA", ["id01"], common.DEFAULT_MODEL, old_text)
    checkpoint = {"versao": 1, "modelo": common.DEFAULT_MODEL,
                  "requisicoes": {cid: {"key": "kA", "passada": 1, "ids": ["id01"]}}}
    write_prior(tmp_path, "introducao", [row(1, "kA", "SUSTENTA", "resumo", QUOTE, True)], checkpoint)
    if evidence_changed:
        write_evidence(tmp_path, "kA", abstract=ABSTRACT + " Another sentence.")
    client, seen = spy_client()
    result = run_global(tmp_path, [ctx(1, "kA"), ctx(2, "kA", chapter="Métodos")], client)
    final = {r["id"]: r for r in result["rows"]}
    if evidence_changed:
        assert "reaproveitado_de" not in final["id01"] and sorted(sent_ids(p) for p in seen) == [["id01", "id02"]]
    else:
        assert final["id01"]["reaproveitado_de"] == "introducao" and [sent_ids(p) for p in seen] == [["id02"]]


def new_ctx(i, legacy, key="kA", **kw):
    return {**ctx(i, key, **kw), "id": f"new{i:02d}", "id_legado": legacy}


def test_reuse_matches_prior_row_by_id_legado_and_writes_new_id(tmp_path):
    write_evidence(tmp_path, "kA")
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "introducao", prior)  # id antigo: id01
    client, seen = spy_client()
    result = run_global(tmp_path, [new_ctx(1, "id01"), new_ctx(2, "id02", chapter="Métodos")], client)
    final = {r["id"]: r for r in result["rows"]}
    assert set(final) == {"new01", "new02"}
    assert final["new01"]["reaproveitado_de"] == "introducao" and final["new01"]["id_legado"] == "id01"
    assert final["new01"]["veredito_final"] == "SUSTENTA"
    assert "reaproveitado_de" not in final["new02"]
    assert [sent_ids(p) for p in seen] == [["new02"]]
    assert result["metadados"]["contextos_reaproveitados"] == 1

    again_client, again_seen = spy_client()  # a pasta tese já tem o id novo; nada é reenviado
    again = run_global(tmp_path, [new_ctx(1, "id01"), new_ctx(2, "id02", chapter="Métodos")], again_client)
    assert again_seen == [] and {r["id"] for r in again["rows"]} == {"new01", "new02"}


def test_rows_with_legacy_id_in_the_own_tese_folder_are_reused_without_api_calls(tmp_path):
    write_evidence(tmp_path, "kA")
    fp = evidence_fp(tmp_path, "kA")
    adjudicacao = {"modelo": "claude-opus-5-5", "veredito": "SUSTENTA", "em": "2026-11-06T09:00:00+00:00"}
    prior = [
        {**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": fp, "adjudicacao": adjudicacao},
        {**row(2, "kA", "NAO_SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": fp},
    ]
    write_prior(tmp_path, "tese", prior)  # ids antigos (fórmula anterior), sem reaproveitado_de, na própria pasta
    contexts = [new_ctx(1, "id01"), new_ctx(2, "id02", chapter="Métodos")]
    client, seen = spy_client()
    result = run_global(tmp_path, contexts, client)

    assert seen == []  # nenhum envio à API
    final = {r["id"]: r for r in result["rows"]}
    assert set(final) == {"new01", "new02"}
    assert final["new01"]["adjudicacao"] == adjudicacao and final["new01"]["veredito_final"] == "SUSTENTA"
    assert final["new01"]["id_legado"] == "id01" and final["new01"]["reaproveitado_de"] == "tese"
    assert final["new02"]["veredito_final"] == "NAO_SUSTENTA"
    assert result["metadados"]["contextos_reaproveitados"] == 2
    checkpoint = common.read_json(tmp_path / "piloto" / "tese" / "checkpoint.json", default={}) or {}
    assert checkpoint.get("requisicoes", {}) == {}

    again_client, again_seen = spy_client()
    again = run_global(tmp_path, contexts, again_client)
    assert again_seen == [] and again["rows"] == result["rows"]


def test_own_tese_folder_takes_precedence_over_chapter_folders(tmp_path):
    write_evidence(tmp_path, "kA")
    fp = evidence_fp(tmp_path, "kA")
    adjudicacao = {"modelo": "claude-opus-5-5", "veredito": "NAO_SUSTENTA", "em": "2026-11-06T09:00:00+00:00"}
    write_prior(tmp_path, "introducao", [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": fp}])
    write_prior(tmp_path, "tese", [{**row(1, "kA", "NAO_SUSTENTA", "resumo", QUOTE, True),
                                    "evidencia_fingerprint": fp, "adjudicacao": adjudicacao}])
    client, seen = spy_client()
    result = run_global(tmp_path, [new_ctx(1, "id01")], client)
    (final,) = result["rows"]
    assert seen == [] and final["adjudicacao"] == adjudicacao and final["veredito_final"] == "NAO_SUSTENTA"
    assert final["reaproveitado_de"] == "tese"


def test_own_folder_rows_with_the_same_new_id_and_no_origin_are_not_copied(tmp_path):
    write_evidence(tmp_path, "kA")
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "tese", prior)  # mesmo id novo: a linha vem do checkpoint, não da cópia
    found = av.find_reusable([ctx(1, "kA")], {"kA": prompts.build_evidence_block(
        common.read_json(tmp_path / "evidencias" / "kA.json"), av.Settings().max_chars)[0]},
        tmp_path / "piloto", tmp_path / "piloto" / "tese")
    assert found == {}


def test_reuse_by_id_legado_requires_unchanged_fingerprint(tmp_path):
    write_evidence(tmp_path, "kA")
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "introducao", prior)
    write_evidence(tmp_path, "kA", abstract=ABSTRACT + " An extra sentence changes the evidence.")
    client, seen = spy_client()
    result = run_global(tmp_path, [new_ctx(1, "id01")], client)
    assert [sent_ids(p) for p in seen] == [["new01"]]
    assert all("reaproveitado_de" not in r for r in result["rows"])


def test_reuse_by_id_legado_of_rows_without_fingerprint_uses_old_checkpoint(tmp_path):
    write_evidence(tmp_path, "kA")
    record = common.read_json(tmp_path / "evidencias" / "kA.json")
    old_text = prompts.build_evidence_block(record, av.Settings().max_chars)[0]
    cid = av.make_custom_id(1, "kA", ["id01"], common.DEFAULT_MODEL, old_text)  # o checkpoint antigo guarda o id antigo
    checkpoint = {"versao": 1, "modelo": common.DEFAULT_MODEL,
                  "requisicoes": {cid: {"key": "kA", "passada": 1, "ids": ["id01"]}}}
    write_prior(tmp_path, "introducao", [row(1, "kA", "SUSTENTA", "resumo", QUOTE, True)], checkpoint)
    client, seen = spy_client()
    result = run_global(tmp_path, [new_ctx(1, "id01"), new_ctx(2, "id02", chapter="Métodos")], client)
    final = {r["id"]: r for r in result["rows"]}
    assert final["new01"]["reaproveitado_de"] == "introducao" and [sent_ids(p) for p in seen] == [["new02"]]


def test_ambiguous_id_legado_is_reused_only_for_the_identical_sentence(tmp_path):
    write_evidence(tmp_path, "kA")
    first, second = "A [cit] e B [ref] afirmam 92%.", "A [ref] e B [cit] afirmam 92%."
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "sentence_text": first,
              "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "introducao", prior)
    contexts = [new_ctx(1, "id01", sentence=first), new_ctx(2, "id01", sentence=second)]
    client, seen = spy_client()
    result = run_global(tmp_path, contexts, client)
    final = {r["id"]: r for r in result["rows"]}
    assert final["new01"]["reaproveitado_de"] == "introducao"
    assert "reaproveitado_de" not in final["new02"]
    assert [sent_ids(p) for p in seen] == [["new02"]]


def test_ambiguous_id_legado_without_old_sentence_text_is_regraded(tmp_path):
    write_evidence(tmp_path, "kA")
    prior = [{**row(1, "kA", "SUSTENTA", "resumo", QUOTE, True), "evidencia_fingerprint": evidence_fp(tmp_path, "kA")}]
    write_prior(tmp_path, "introducao", prior)
    contexts = [new_ctx(1, "id01", sentence="A [cit] e B [ref]."), new_ctx(2, "id01", sentence="A [ref] e B [cit].")]
    client, seen = spy_client()
    result = run_global(tmp_path, contexts, client)
    assert sorted(sent_ids(p) for p in seen) == [["new01", "new02"]]
    assert result["metadados"]["contextos_reaproveitados"] == 0


def test_cli_global_scope_end_to_end_report_has_chapter_and_key_tables(tmp_path, monkeypatch):
    contexts = [ctx(1, "kA", chapter="Introdução"), ctx(2, "kA", chapter="Métodos")]
    bib = write_inputs(tmp_path, contexts)
    write_evidence(tmp_path, "kA")
    client, seen = spy_client()
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: "sk-ant-teste")
    monkeypatch.setattr(cli.avaliar_mod, "make_client", lambda api_key: client)
    base = ["--escopo", "tese", "--out", str(tmp_path)]

    assert cli.main(["avaliar", *base, "--bib", str(bib), "--modo", "sync", "--concorrencia", "1"]) == 0
    assert [sent_ids(p) for p in seen] == [["id01", "id02"]] and "temperature" not in seen[0]
    assert cli.main(["relatorio", *base]) == 0
    pdir = tmp_path / "piloto" / "tese"
    for name in ("checkpoint.json", "avaliacoes.jsonl", "metadados.json", "relatorio.md"):
        assert (pdir / name).is_file()
    report = (pdir / "relatorio.md").read_text(encoding="utf-8")
    assert "Relatório da atestação bibliográfica: Tese completa" in report
    assert "### Por capítulo" in report and "| Introdução | 1 | 1 |" in report and "| Métodos | 1 | 1 |" in report
    assert "## Por chave" in report and "| `kA` | 2 | SUSTENTA 2 | resumo |" in report


def test_report_chapter_and_key_tables_sorted_problems_first():
    def with_chapter(r, chapter, numbers=()):
        return {**r, "chapter": chapter, "numeros_nao_encontrados": list(numbers)}

    rows = [
        with_chapter(row(1, "kA", "SUSTENTA"), "Introdução"),
        with_chapter(row(2, "kA", "SUSTENTA"), "Métodos"),
        with_chapter(row(3, "kB", "NAO_SUSTENTA"), "Métodos"),
        with_chapter(row(4, "kC", "SUSTENTA"), "Introdução", ["39,3"]),
        with_chapter(row(5, "kD", "CONTRADIZ"), "Apêndice"),
    ]
    rows[0]["reaproveitado_de"] = "introducao"
    meta = {"modelo": "m", "data_utc": "x", "status": "completo", "parametros": {"modo": "sync"}, "uso": {}}
    text = rel.build_report(rows, meta, [], {}, "Tese completa", None, escopo="tese")

    assert "| Capítulo | Contextos | SUSTENTA | NAO_SUSTENTA | CONTRADIZ |" in text
    assert "| Introdução | 2 | 2 | 0 | 0 |" in text and "| Métodos | 2 | 1 | 1 | 0 |" in text
    assert "| Apêndice | 1 | 0 | 0 | 1 |" in text
    order = [text.index(f"| `{k}` | 1 |") if k != "kA" else text.index("| `kA` | 2 |") for k in ("kD", "kB", "kA", "kC")]
    assert order == sorted(order)
    assert "| `kD` | 1 | CONTRADIZ 1 | resumo |" in text
    assert "Vereditos reaproveitados de execuções anteriores (evidência inalterada): 1 (introducao 1)" in text
    assert "## Números da sentença não encontrados na evidência" in text and "kC` (SUSTENTA): 39,3" in text

    chapter_text = rel.build_report(rows, meta, [], {}, "Introdução", None)
    assert "### Por capítulo" not in chapter_text and "## Por chave" not in chapter_text
    assert "## Adjudicação" not in chapter_text


# ---------------------------------------------------------------------------
# Adjudicação dos contextos DIVERGENTE (API simulada)
# ---------------------------------------------------------------------------
class RecordingMessages(FakeMessages):
    def __init__(self, policy, fail_keys=()):
        super().__init__(policy, fail_keys)
        self.params = []

    def create(self, **params):
        self.params.append(params)
        return super().create(**params)


def recording_client(policy):
    return SimpleNamespace(messages=RecordingMessages(policy))


def grade_policy(passada, key, cid):
    if key == "kC" or passada == 2:
        return good(cid)
    return critical(cid)  # passada 1 crítica e passada 2 favorável: DIVERGENTE


ADJ_ANSWERS = {
    "id01": ("SUSTENTA", QUOTE, None, "alta"),  # coincide com a Avaliação 2
    "id02": ("NAO_SUSTENTA", "", "fora_de_escopo", "MEDIA"),  # coincide com a Avaliação 1
    "id03": ("CONTRADIZ", QUOTE, "numero_divergente", "xx"),  # nenhuma das duas
}


def adj_policy(passada, key, cid):
    verdict, quote, problem, confidence = ADJ_ANSWERS[cid]
    return {"id": cid, "veredito": verdict, "trecho_fonte": quote, "justificativa": "árbitro",
            "problema": problem, "confianca": confidence}


def invented_quote_policy(passada, key, cid):
    return {**adj_policy(passada, key, cid), "veredito": "SUSTENTA", "trecho_fonte": "passagem que não existe"}


ADJ_CONTEXTS = [ctx(1, "kA"), ctx(2, "kA"), ctx(3, "kB"), ctx(4, "kC")]


def graded_run(tmp_path):
    """Avalia ADJ_CONTEXTS: id01 a id03 ficam DIVERGENTE e id04 (kC) SUSTENTA."""
    for key in ("kA", "kB", "kC"):
        write_evidence(tmp_path, key)
    client = recording_client(grade_policy)
    result = run(tmp_path, ADJ_CONTEXTS, client, av.Settings(modo="sync", concorrencia=1))
    assert {r["id"]: r["veredito_final"] for r in result["rows"]} == {
        "id01": "DIVERGENTE", "id02": "DIVERGENTE", "id03": "DIVERGENTE", "id04": "SUSTENTA"}
    return client, tmp_path / "piloto" / "introducao"


def adjudicate(tmp_path, client, **kwargs):
    settings = av.Settings(**{"model": common.DEFAULT_ADJ_MODEL, "max_tokens": 8192, "modo": "sync",
                              "concorrencia": 1, "poll_inicial_s": 0.0, **kwargs})
    return adj.run_adjudicacao(
        capitulo="Introdução", contexts=ADJ_CONTEXTS, out_dir=tmp_path,
        pilot_dir=tmp_path / "piloto" / "introducao", settings=settings, client=client, sleep=lambda s: None,
    )


def request_key(params):
    return re.search(r'chave="([^"]+)"', params["messages"][0]["content"][0]["text"]).group(1)


def request_ids(params):
    return re.findall(r"^id: (\S+)$", params["messages"][0]["content"][1]["text"], re.M)


def test_adjudication_rubric_is_long_enough_and_reuses_grader_rules():
    assert len(prompts.ADJ_RUBRIC) // 4 >= 1100
    for text in (prompts._ADJ_NUMERICO, prompts._ADJ_TRECHO, prompts._ADJ_CRITERIOS):
        assert text in prompts.ADJ_RUBRIC and text in prompts.SYSTEM_RUBRIC
    for word in ("Conflito numérico e qualificadores", "afirma_mais_que_a_fonte", "Avaliação 1", "Avaliação 2",
                 '"confianca"', "alta", "80 palavras", "60 palavras"):
        assert word in prompts.ADJ_RUBRIC
    assert "—" not in prompts.ADJ_RUBRIC and "–" not in prompts.ADJ_RUBRIC


def test_adjudication_sends_only_divergente_one_request_per_key(tmp_path):
    grading, _pdir = graded_run(tmp_path)
    client = recording_client(adj_policy)
    adjudicate(tmp_path, client)

    sent = client.messages.params
    assert sorted((request_key(p), request_ids(p)) for p in sent) == [("kA", ["id01", "id02"]), ("kB", ["id03"])]
    for params in sent:
        assert params["model"] == "claude-opus-5-5" and "temperature" not in params
        assert "id04" not in params["messages"][0]["content"][1]["text"]  # SUSTENTA não é reenviado
    # bloco de sistema idêntico entre requisições, com cache efêmero
    systems = {json.dumps(p["system"], sort_keys=True) for p in sent}
    assert len(systems) == 1
    assert sent[0]["system"] == [{"type": "text", "text": prompts.ADJ_RUBRIC, "cache_control": {"type": "ephemeral"}}]
    # bloco A: o mesmo texto de evidência das passadas de avaliação
    graded_evidence = {request_key(p): p["messages"][0]["content"][0]["text"] for p in grading.messages.params}
    for params in sent:
        assert params["messages"][0]["content"][0]["text"] == graded_evidence[request_key(params)]


def test_adjudication_block_b_has_context_and_both_prior_assessments(tmp_path):
    graded_run(tmp_path)
    client = recording_client(adj_policy)
    adjudicate(tmp_path, client)
    params = next(p for p in client.messages.params if request_key(p) == "kA")
    block_b = params["messages"][0]["content"][1]["text"]
    first = block_b.index("id: id01")
    one, two = block_b.index("#### Avaliação 1", first), block_b.index("#### Avaliação 2", first)
    assert first < one < two < block_b.index("id: id02")
    assert "Sentença: A cobertura chegou a 92% [cit]." in block_b and "Parágrafo: Parágrafo. " in block_b
    part1, part2 = block_b[one:two], block_b[two : block_b.index("id: id02")]
    assert "Veredito: NAO_SUSTENTA" in part1 and "Justificativa: não consta" in part1
    assert "Problema: fora_de_escopo" in part1 and "Trecho da fonte: (vazio)" in part1
    assert "Veredito: SUSTENTA" in part2 and f"Trecho da fonte: {QUOTE}" in part2
    assert "Justificativa: ok" in part2 and "Problema: nenhum" in part2


def test_adjudication_paragraph_is_limited_to_1200_chars(tmp_path):
    graded_run(tmp_path)
    long_context = {**ADJ_CONTEXTS[2], "paragraph_text": "x" * 2000 + ADJ_CONTEXTS[2]["sentence_text"] + "y" * 2000}
    rows = common.read_jsonl(tmp_path / "piloto" / "introducao" / "avaliacoes.jsonl")
    row_kb = next(r for r in rows if r["id"] == "id03")
    text = prompts.build_adjudication_block("Introdução", "kB", [(long_context, row_kb)])
    paragraph = next(line for line in text.splitlines() if line.startswith("Parágrafo: "))
    assert len(paragraph) - len("Parágrafo: ") <= 1200 and long_context["sentence_text"] in paragraph


def test_adjudication_updates_rows_in_place_and_keeps_previous_verdicts(tmp_path):
    _grading, pdir = graded_run(tmp_path)
    before = common.read_jsonl(pdir / "avaliacoes.jsonl")
    checkpoint_before = common.read_json(pdir / "checkpoint.json")
    client = recording_client(adj_policy)
    result = adjudicate(tmp_path, client)

    after = common.read_jsonl(pdir / "avaliacoes.jsonl")
    assert [r["id"] for r in after] == [r["id"] for r in before]
    old = {r["id"]: r for r in before}
    new = {r["id"]: r for r in after}
    assert new["id04"] == old["id04"]  # contexto que não era DIVERGENTE não muda
    assert {k: new[k]["veredito_final"] for k in ("id01", "id02", "id03")} == {
        "id01": "SUSTENTA", "id02": "NAO_SUSTENTA", "id03": "CONTRADIZ"}
    for ident in ("id01", "id02", "id03"):
        assert new[ident]["passada1"] == old[ident]["passada1"] and new[ident]["passada2"] == old[ident]["passada2"]
        assert new[ident]["veredito_pre_adjudicacao"] == "DIVERGENTE" and "adjudicado" in new[ident]["flags"]
    record = new["id01"]["adjudicacao"]
    assert set(record) >= {"modelo", "veredito", "trecho_fonte", "trecho_verificado", "justificativa",
                           "problema", "confianca", "batch_id", "em"}
    assert record["modelo"] == "claude-opus-5-5" and record["veredito"] == "SUSTENTA"
    assert record["trecho_fonte"] == QUOTE and record["trecho_verificado"] is True
    assert record["confianca"] == "alta" and record["batch_id"] is None and record["em"]
    assert new["id02"]["adjudicacao"]["confianca"] == "media" and new["id03"]["adjudicacao"]["confianca"] is None
    assert [rel.adjudication_agreement(new[k]) for k in ("id01", "id02", "id03")] == [
        "avaliacao_2", "avaliacao_1", "nenhuma"]
    assert result["incompleto"] is False and result["adjudicados"] == 3

    meta = common.read_json(pdir / "metadados.json")
    assert meta["uso"]["adjudicacao"]["sync"]["input_tokens"] == 200  # duas requisições
    assert meta["uso"]["adjudicacao"]["sync"]["output_tokens"] == 100
    assert meta["uso"]["sync"] == av.usage_totals(av.Checkpoint(pdir / "checkpoint.json", {}))["sync"]
    assert meta["adjudicacao"]["status"] == "completo" and meta["adjudicacao"]["modelo"] == "claude-opus-5-5"

    state = common.read_json(pdir / "checkpoint.json")
    assert state["requisicoes"] == checkpoint_before["requisicoes"]  # avaliação intacta
    assert len(state["adjudicacao"]["requisicoes"]) == 2 and state["adjudicacao"]["lotes"] == []

    again = recording_client(adj_policy)
    assert adjudicate(tmp_path, again)["adjudicados"] == 0 and again.messages.params == []


def test_adjudication_downgrades_quote_not_found(tmp_path):
    _grading, pdir = graded_run(tmp_path)
    adjudicate(tmp_path, recording_client(invented_quote_policy))
    rows = {r["id"]: r for r in common.read_jsonl(pdir / "avaliacoes.jsonl")}
    for ident in ("id01", "id02", "id03"):
        row_ = rows[ident]
        assert row_["veredito_final"] == "NAO_VERIFICAVEL" and row_["adjudicacao"]["veredito"] == "NAO_VERIFICAVEL"
        assert row_["adjudicacao"]["veredito_modelo"] == "SUSTENTA"
        assert row_["adjudicacao"]["trecho_verificado"] is False
        assert "trecho_nao_encontrado" in row_["flags"] and "trecho_nao_encontrado" in row_["adjudicacao"]["flags"]
        assert rel.nv_reason(row_) == "trecho citado pelo modelo não encontrado na evidência"


def test_adjudication_batch_resumes_by_polling_recorded_batch(tmp_path):
    _grading, pdir = graded_run(tmp_path)
    client = SimpleNamespace(messages=FakeMessages(adj_policy))
    batches = client.messages.batches

    first = adjudicate(tmp_path, client, modo="batch", espera_max_s=0.0)
    assert batches.created == ["msgbatch_1"] and first["incompleto"] is True
    section = common.read_json(pdir / "checkpoint.json")["adjudicacao"]
    assert section["lotes"][0]["id"] == "msgbatch_1" and section["lotes"][0]["baixado"] is False
    assert section["requisicoes"] == {}
    assert {r["veredito_final"] for r in common.read_jsonl(pdir / "avaliacoes.jsonl")} == {"DIVERGENTE", "SUSTENTA"}
    for request in batches.store["msgbatch_1"]:
        assert "temperature" not in request["params"]
        assert request["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["custom_id"])

    second = adjudicate(tmp_path, client, modo="batch")
    assert batches.created == ["msgbatch_1"]  # nada reenviado; o lote registrado foi consultado
    assert second["incompleto"] is False and second["adjudicados"] == 3
    rows = {r["id"]: r for r in common.read_jsonl(pdir / "avaliacoes.jsonl")}
    assert rows["id01"]["adjudicacao"]["batch_id"] == "msgbatch_1"
    assert common.read_json(pdir / "metadados.json")["uso"]["adjudicacao"]["batch"]["input_tokens"] == 200
    assert all(b["baixado"] for b in common.read_json(pdir / "checkpoint.json")["adjudicacao"]["lotes"])


class ErroredFirstBatch(FakeBatches):
    def results(self, batch_id):
        if batch_id != "msgbatch_1":
            yield from super().results(batch_id)
            return
        for request in self.store[batch_id]:
            yield SimpleNamespace(custom_id=request["custom_id"],
                                  result=SimpleNamespace(type="errored", error=SimpleNamespace(type="overloaded_error")))


def test_adjudication_does_not_resubmit_failed_batch_without_refazer_falhas(tmp_path):
    _grading, pdir = graded_run(tmp_path)
    client = SimpleNamespace(messages=FakeMessages(adj_policy))
    client.messages.batches = ErroredFirstBatch(adj_policy)
    batches = client.messages.batches

    first = adjudicate(tmp_path, client, modo="batch")
    assert batches.created == ["msgbatch_1"] and first["incompleto"] is True and first["adjudicados"] == 0
    assert common.read_json(pdir / "metadados.json")["adjudicacao"]["falhas"] == 2

    adjudicate(tmp_path, client, modo="batch")
    assert batches.created == ["msgbatch_1"]  # id registrado e terminado com erro: sem reenvio

    third = adjudicate(tmp_path, client, modo="batch", refazer_falhas=True)
    assert batches.created == ["msgbatch_1", "msgbatch_2"] and third["incompleto"] is False
    rows = {r["id"]: r for r in common.read_jsonl(pdir / "avaliacoes.jsonl")}
    assert rows["id01"]["adjudicacao"]["batch_id"] == "msgbatch_2"


def test_adjudication_skips_key_when_evidence_changed(tmp_path):
    _grading, pdir = graded_run(tmp_path)
    write_evidence(tmp_path, "kB", abstract="Outro resumo da fonte kB, diferente do avaliado.")
    client = recording_client(adj_policy)
    result = adjudicate(tmp_path, client)
    assert [request_key(p) for p in client.messages.params] == ["kA"]
    assert result["incompleto"] is True
    rows = {r["id"]: r["veredito_final"] for r in common.read_jsonl(pdir / "avaliacoes.jsonl")}
    assert rows["id03"] == "DIVERGENTE" and rows["id01"] == "SUSTENTA"


def test_report_has_adjudication_section_with_counts_and_critical_list():
    def adjudicated(i, key, final, adjudicated_verdict, p1, p2, confidence="alta"):
        base = row(i, key, final)
        return {**base, "passada1": {"veredito": p1}, "passada2": {"veredito": p2},
                "adjudicacao": {"modelo": "claude-opus-5-5", "veredito": adjudicated_verdict, "confianca": confidence,
                                "justificativa": f"árbitro {i}"}}

    rows = [
        adjudicated(1, "kA", "SUSTENTA", "SUSTENTA", "NAO_SUSTENTA", "SUSTENTA"),
        adjudicated(2, "kB", "NAO_SUSTENTA", "NAO_SUSTENTA", "NAO_SUSTENTA", "SUSTENTA", "media"),
        adjudicated(3, "kC", "CONTRADIZ", "CONTRADIZ", "NAO_SUSTENTA", "SUSTENTA", "baixa"),
        row(4, "kD", "SUSTENTA"),
    ]
    tokens = {"batch": {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0}}
    meta = {"modelo": "claude-sonnet-5-5", "data_utc": "x", "status": "completo", "parametros": {"modo": "batch"},
            "uso": {"batch": {"input_tokens": 1_000_000, "output_tokens": 0, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0}, "adjudicacao": tokens},
            "adjudicacao": {"status": "completo", "contextos_ainda_divergentes": 0}}
    precos = {"input": 3, "output": 15, "cache_write": 3.75, "cache_read": 0.3, "batch_desconto": 0.5}
    text = rel.build_report(rows, meta, [], {}, "Tese completa", precos)

    section = text[text.index("## Adjudicação") : text.index("## Números da sentença")]
    assert "Contextos DIVERGENTE adjudicados: 3" in section
    assert "| SUSTENTA | 1 |" in section and "| NAO_SUSTENTA | 1 |" in section and "| CONTRADIZ | 1 |" in section
    assert "| Avaliação 1 | 1 |" in section and "| Avaliação 2 | 1 |" in section and "| Nenhuma das duas | 1 |" in section
    critical = section[section.index("### NAO_SUSTENTA e CONTRADIZ após adjudicação") :]
    assert "`kB` (NAO_SUSTENTA; coincide com avaliacao_1; confiança media): árbitro 2" in critical
    assert "`kC` (CONTRADIZ; coincide com nenhuma" in critical and "`kA`" not in critical
    assert "| batch | 1.000 | 500 | 0 | 0 |" in section
    assert "batch 1,50; sync 0,00; total 1,50" in text  # a adjudicação não entra no custo da avaliação


def test_cli_adjudicar_end_to_end_default_model_and_report(tmp_path, monkeypatch, capsys):
    bib = write_inputs(tmp_path, [ctx(1, "kA")])
    write_evidence(tmp_path, "kA")
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: "sk-ant-teste")
    grading = recording_client(grade_policy)
    monkeypatch.setattr(cli.avaliar_mod, "make_client", lambda api_key: grading)
    base = ["--capitulo", "Introdução", "--out", str(tmp_path)]
    assert cli.main(["avaliar", *base, "--bib", str(bib), "--modo", "sync", "--concorrencia", "1"]) == 0
    pdir = tmp_path / "piloto" / "introducao"
    assert common.read_jsonl(pdir / "avaliacoes.jsonl")[0]["veredito_final"] == "DIVERGENTE"

    parsed = cli.build_parser().parse_args(["adjudicar", "--escopo", "tese"])
    assert parsed.modelo == "claude-opus-5-5" and parsed.modo == "batch" and parsed.max_chars == 120000

    judge = recording_client(adj_policy)
    monkeypatch.setattr(cli.avaliar_mod, "make_client", lambda api_key: judge)
    assert cli.main(["adjudicar", *base, "--modo", "sync", "--concorrencia", "1"]) == 0
    assert [p["model"] for p in judge.messages.params] == ["claude-opus-5-5"]
    assert "temperature" not in judge.messages.params[0]
    assert common.read_jsonl(pdir / "avaliacoes.jsonl")[0]["veredito_final"] == "SUSTENTA"

    assert cli.main(["relatorio", *base]) == 0
    assert "## Adjudicação" in (pdir / "relatorio.md").read_text(encoding="utf-8")
    assert cli.main(["adjudicar", *base, "--modo", "sync"]) == 0  # nada mais DIVERGENTE
    shown = capsys.readouterr()
    assert "sk-ant-teste" not in shown.out + shown.err and "nenhum contexto DIVERGENTE" in shown.out
    assert len(judge.messages.params) == 1


def test_cli_adjudicar_without_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path, [ctx(1, "kA")])
    pdir = tmp_path / "piloto" / "introducao"
    common.write_jsonl_atomic(pdir / "avaliacoes.jsonl", [row(1, "kA", "DIVERGENTE")])
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: "")
    code = cli.main(["adjudicar", "--capitulo", "Introdução", "--out", str(tmp_path)])
    assert code == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Seleção de trechos para evidências longas (BM25, páginas, consultas em inglês)
# ---------------------------------------------------------------------------
FILL = "Unrelated filler sentence about geology and cooking recipes. "
SPECIAL_A = "Indexing yields candidate pairs for comparison."
SPECIAL_B = "A recall threshold was tuned on validation data."
CTX_A = ctx(1, "kL", sentence="O indexador reduz comparações [cit].")
CTX_B = ctx(2, "kL", sentence="O limiar altera a cobertura [cit].")
EN_A = {"traducao": "The indexer reduces comparisons", "palavras_chave": ["candidate pairs", "indexing"]}
EN_B = {"traducao": "The cutoff changes coverage", "palavras_chave": ["recall threshold", "tuned"]}


def long_book(specials=None, n_pages=60):
    """Texto de ``n_pages`` páginas (mais de 200.000 caracteres) e o início de cada página.

    Uma página com frase especial a traz no meio (depois de 1.647 caracteres de enchimento)."""
    specials = specials or {}
    pages = []
    for number in range(1, n_pages + 1):
        special = specials.get(number)
        pages.append(FILL * 27 + special + " " + FILL * 27 if special else FILL * 55)
    text = "\n".join(pages)
    return text, ev.page_starts_for(pages, text)


def write_long_record(out_dir, key, text, starts):
    path = out_dir / "evidencias" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "titulo_bib": TITLE, "abstract": "", "fulltext": text, "fulltext_fonte": f"local:{key}.pdf",
              "fulltext_paginas": starts, "consensus": [], "identidade_verificada": True}
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def test_page_starts_for_matches_joined_text_with_or_without_strip():
    pages = ["  primeira página", "segunda", "terceira  "]
    joined = "\n".join(pages)
    assert ev.page_starts_for(pages, joined) == [0, len(pages[0]) + 1, len(pages[0]) + len(pages[1]) + 2]
    stripped = ev.page_starts_for(pages, joined.strip())
    assert stripped[0] == 0 and joined.strip()[stripped[1] : stripped[1] + 7] == "segunda"
    assert ev.page_starts_for(pages, "outro texto") is None


def test_local_pdf_page_starts_come_from_the_pdf_pages(tmp_path):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for line in ("Primeira pagina", "Segunda pagina", "Terceira pagina"):
        document.new_page().insert_text((72, 72), line)
    path = tmp_path / "k.pdf"
    document.save(str(path))
    document.close()
    text = ev.pdf_to_text(path.read_bytes()).strip()
    starts = ev.local_pdf_page_starts(path, text)
    assert len(starts) == 3 and starts[0] == 0
    assert text[starts[1] :].startswith("Segunda") and text[starts[2] :].startswith("Terceira")
    assert ev.local_pdf_page_starts(path, "texto diferente") is None
    assert ev.local_pdf_page_starts(tmp_path / "ausente.pdf", text) is None


def test_chunking_has_overlap_pages_and_covers_the_whole_text():
    text, starts = long_book(n_pages=12)
    chunks = tr.build_chunks(text, starts)
    assert chunks[0]["ini"] == 0 and chunks[-1]["fim"] == len(text) and chunks[0]["i"] == 1
    assert [c["i"] for c in chunks] == list(range(1, len(chunks) + 1))
    for chunk, following in zip(chunks, chunks[1:]):
        assert chunk["fim"] - chunk["ini"] <= tr.CHUNK_CHARS
        assert 40 <= chunk["fim"] - following["ini"] <= tr.CHUNK_OVERLAP  # sobreposição de até 200
    for chunk in chunks:
        assert chunk["pagina"] == sum(1 for s in starts if s <= chunk["ini"])
        assert chunk["pagina_fim"] == sum(1 for s in starts if s <= chunk["fim"] - 1)
    assert chunks[0]["pagina"] == 1 and chunks[-1]["pagina_fim"] == 12
    assert all(c["pagina"] is None for c in tr.build_chunks(text))  # .txt/.html: sem página


def test_chunk_index_is_cached_by_text_hash(tmp_path, monkeypatch):
    text, starts = long_book(n_pages=6)
    first = tr.load_chunks(text, "k.a", tmp_path, starts)
    assert (tmp_path / "evidencias" / "k.a.chunks.json").is_file()
    calls = []
    original = tr.build_chunks
    monkeypatch.setattr(tr, "build_chunks", lambda *a, **k: calls.append(1) or original(*a, **k))
    assert tr.load_chunks(text, "k.a", tmp_path, starts) == first and calls == []  # veio do cache
    changed = tr.load_chunks(text + " mais um trecho no fim.", "k.a", tmp_path, starts)
    assert calls == [1] and changed[-1]["fim"] == len(text) + len(" mais um trecho no fim.")


def test_tokenize_matches_numbers_across_pt_and_en_formats():
    assert tr.tokenize("39,3% e 84.368 casos em 2015") == ["n393", "n84368", "caso", "n2015"]
    assert tr.tokenize("39.3 percent, 84,368 cases in 2015") == ["n393", "percent", "n84368", "case", "n2015"]
    assert tr.tokenize("Os pares [cit] e [ref] 7") == ["pare"]  # marcadores, stopwords e dígito solto saem


def test_invariant_tokens_numbers_acronyms_proper_nouns_and_english_terms():
    text = "Aplicamos SMOTE e Random Forest ao Sinan, com 1.234 pares e 93,4% [cit]; o blocking segue Fellegi-Sunter."
    found = tr.invariant_tokens(text)
    assert {"smote", "random", "forest", "sinan", "n1234", "n934", "blocking", "fellegi", "sunter"} <= found
    assert "aplicamos" not in found  # início de sentença não conta como nome próprio


def test_bm25_ranks_relevant_passage_first_for_portuguese_sentence_and_english_query():
    passages = [
        "Geology notes about rocks and minerals in the northern basin.",
        "The Fellegi-Sunter model computes match weights; blocking reduces candidate pairs and recall was 93.4 percent.",
        "A cooking recipe with flour, eggs and sugar for a small cake.",
        "Candidate pairs are generated by indexing keys before comparison.",
    ]
    context = ctx(1, "kA", sentence="O blocking reduz os pares candidatos no modelo de Fellegi-Sunter [cit].")
    english = {"traducao": "Blocking reduces the candidate pairs in the Fellegi-Sunter model",
               "palavras_chave": ["blocking", "candidate pairs", "match weights"]}
    docs = [tr.Counter(tr.tokenize(p)) for p in passages]
    ranking = tr.rank_by_bm25(docs, tr.build_query(context, english))
    assert ranking[0] == 1 and 3 in ranking and 0 not in ranking and 2 not in ranking
    # só em português (invariantes) a passagem certa continua à frente
    assert tr.rank_by_bm25(docs, tr.build_query(context, None))[0] == 1


def test_selection_offline_fallback_without_english_query_then_english_finds_the_passage():
    text, starts = long_book({20: SPECIAL_A})
    context = ctx(1, "kL", sentence="O método reduz os pares candidatos à comparação [cit].")
    offline = tr.select_evidence_text(text, "kL", [context], None, 120000, None, starts)
    assert offline is not None and offline.trechos == [] and offline.consultas_en == 0
    assert SPECIAL_A not in offline.texto

    english = {context["id"]: {"traducao": "The method reduces the candidate pairs to compare",
                               "palavras_chave": ["candidate pairs", "indexing"]}}
    online = tr.select_evidence_text(text, "kL", [context], english, 120000, None, starts)
    assert online.consultas == 1 and online.consultas_en == 1
    assert online.trechos and {t["pagina"] for t in online.trechos} == {20}
    assert SPECIAL_A in online.texto
    assert re.search(r"\[trechos? \d+(?:-\d+)?, p\. 20(?:-\d+)?\]", online.texto)


def test_front_matter_is_always_included_even_when_nothing_matches():
    text, starts = long_book({1: "Handbook of Record Linkage. FRONTMARK Abstract."})
    context = ctx(1, "kL", sentence="O gradiente descendente converge [cit].")
    sel = tr.select_evidence_text(text, "kL", [context], {}, 120000, None, starts)
    assert sel.texto.startswith(tr.FRONT_MARKER) and "FRONTMARK" in sel.texto
    assert sel.trechos == [] and sel.total_trechos > 100
    assert len(sel.texto) <= tr.FRONT_CHARS + len(tr.FRONT_MARKER) + 1
    assert sel.texto == tr.render(text, [], tr.front_end_of(text))


def test_union_round_robin_respects_max_chars_and_document_order():
    specials = {**{n: SPECIAL_A for n in range(10, 20)}, **{n: SPECIAL_B for n in range(40, 50)}}
    text, starts = long_book(specials)
    english = {CTX_A["id"]: EN_A, CTX_B["id"]: EN_B}
    max_chars = 11000
    sel = tr.select_evidence_text(text, "kL", [CTX_B, CTX_A], english, max_chars, None, starts)
    assert len(sel.texto) <= max_chars and len(sel.texto) > 8000  # enche o orçamento
    ids = [t["trecho"] for t in sel.trechos]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    shown = [int(m) for m in re.findall(r"\[trechos? (\d+)", sel.texto)]
    assert shown == sorted(shown)  # ordem de documento no texto enviado
    pages = [t["pagina"] for t in sel.trechos]
    assert any(10 <= p <= 19 for p in pages) and any(40 <= p <= 49 for p in pages)  # os dois contextos entram
    # a ordem dos contextos não muda a seleção
    same = tr.select_evidence_text(text, "kL", [CTX_A, CTX_B], english, max_chars, None, starts)
    assert same.texto == sel.texto
    # sem páginas, o marcador é só [trecho N]
    plain = tr.select_evidence_text(text, "kL", [CTX_A], {CTX_A["id"]: EN_A}, max_chars, None, None)
    assert plain.trechos and all(t["pagina"] is None for t in plain.trechos)
    assert re.search(r"\[trechos? \d+(?:-\d+)?\]", plain.texto) and "p." not in plain.texto


def test_evidence_block_selection_changes_fingerprint_and_short_text_is_untouched(tmp_path):
    text, starts = long_book({20: SPECIAL_A, 40: SPECIAL_B})
    record = write_long_record(tmp_path, "kL", text, starts)

    def block(contexts, english):
        return prompts.build_evidence_block(record, 120000, contexts, tmp_path, english)[0]

    a, again, b = block([CTX_A], {CTX_A["id"]: EN_A}), block([CTX_A], {CTX_A["id"]: EN_A}), block(
        [CTX_A], {CTX_A["id"]: EN_B})
    assert "SELEÇÃO DE TRECHOS" in a and "TRUNCADO" not in a and text[:2900] in a
    assert SPECIAL_A in a and SPECIAL_B not in a and SPECIAL_B in b and SPECIAL_A not in b
    assert av.evidence_fingerprint(a) == av.evidence_fingerprint(again) != av.evidence_fingerprint(b)
    assert (tmp_path / "evidencias" / "kL.chunks.json").is_file()
    # sem contextos, ou com orçamento pequeno demais, volta ao corte no início
    assert "TRUNCADO" in prompts.build_evidence_block(record, 120000)[0]
    assert "TRUNCADO" in prompts.build_evidence_block(record, 2000, [CTX_A], tmp_path, {})[0]
    # texto que cabe em max_chars: o bloco não depende dos contextos (veredito anterior continua valendo)
    short = {**record, "fulltext": "Texto curto. " * 50, "fulltext_paginas": None}
    assert prompts.build_evidence_block(short, 120000, [CTX_A], tmp_path, {CTX_A["id"]: EN_A}) == \
        prompts.build_evidence_block(short, 120000, [CTX_B], tmp_path, {})


def test_run_avaliacao_grades_long_source_on_the_selection_and_records_passages(tmp_path):
    text, starts = long_book({30: ABSTRACT})
    write_long_record(tmp_path, "kL", text, starts)
    context = ctx(1, "kL", sentence="A cobertura chegou a 92,4% em 2015 [cit].")
    client, seen = spy_client()
    result = run(tmp_path, [context], client, av.Settings(modo="sync", concorrencia=1))
    graded = result["rows"][0]
    assert graded["veredito_final"] == "SUSTENTA" and graded["passada1"]["trecho_verificado"] is True
    assert graded["trechos_enviados"] and {t["pagina"] for t in graded["trechos_enviados"]} == {30}
    sent_text = seen[0]["messages"][0]["content"][0]["text"]
    assert "SELEÇÃO DE TRECHOS" in sent_text and QUOTE in sent_text and len(sent_text) < len(text)
    assert graded["evidencia_fingerprint"] == av.evidence_fingerprint(sent_text)
    assert graded["numeros_nao_encontrados"] == []  # conferido contra o texto enviado
    saved = common.read_jsonl(tmp_path / "piloto" / "introducao" / "avaliacoes.jsonl")
    assert saved[0]["trechos_enviados"] == graded["trechos_enviados"]


def test_rubric_has_selection_sentence_and_system_block_stays_byte_stable():
    sentence = "a ausência de apoio nos trechos selecionados não prova ausência na fonte"
    assert sentence in prompts.SYSTEM_RUBRIC and sentence in prompts.ADJ_RUBRIC
    assert "use NAO_VERIFICAVEL em vez de NAO_SUSTENTA" in prompts.SYSTEM_RUBRIC
    assert "—" not in prompts.SYSTEM_RUBRIC and "–" not in prompts.SYSTEM_RUBRIC
    text, starts = long_book({20: SPECIAL_A})
    record = {"key": "kL", "titulo_bib": TITLE, "fulltext": text, "fulltext_paginas": starts, "fulltext_fonte": "x"}
    selected = prompts.build_evidence_block(record, 120000, [CTX_A], None, {CTX_A["id"]: EN_A})[0]
    evidence = {"kL": selected, "kB": '<evidencia chave="kB" nivel="resumo">\nB\n</evidencia>'}
    contexts = [CTX_A, ctx(5, "kB")]
    plans = [
        *av.plan_requests(av.group_by_key(contexts), evidence, "Introdução", av.Settings(), 1),
        *av.plan_requests(av.group_by_key(contexts), evidence, "Introdução", av.Settings(), 2, {"kL"}),
    ]
    assert len({json.dumps(r.params["system"], sort_keys=True) for r in plans}) == 1
    assert plans[0].params["system"][0]["text"] == prompts.SYSTEM_RUBRIC


# --- consultas em inglês (Haiku, Message Batches) ---------------------------------------------------------------
class FakeQueryBatches:
    def __init__(self):
        self.created = []
        self.store = {}
        self.polls = {}

    def create(self, requests):
        batch_id = f"msgbatch_q{len(self.created) + 1}"
        self.created.append(batch_id)
        self.store[batch_id] = requests
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        return SimpleNamespace(id=batch_id, processing_status="ended" if self.polls[batch_id] >= 2 else "in_progress")

    def results(self, batch_id):
        for request in self.store[batch_id]:
            text = request["params"]["messages"][0]["content"][0]["text"]
            ident = re.search(r"^id: (\S+)$", text, re.M).group(1)
            message = fake_message([{"id": ident, "traducao": f"English version of {ident}",
                                     "palavras_chave": ["linkage", "blocking", "recall", "pairs", "threshold"]}])
            yield SimpleNamespace(custom_id=request["custom_id"],
                                  result=SimpleNamespace(type="succeeded", message=message))


def query_client():
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeQueryBatches(), create=None))


def query_settings(**kwargs):
    return av.Settings(**{"model": common.DEFAULT_QUERY_MODEL, "max_tokens": 600, "modo": "batch",
                          "poll_inicial_s": 0.0, **kwargs})


def test_consultas_request_uses_haiku_without_temperature_and_caches_the_instruction_block():
    assert common.DEFAULT_QUERY_MODEL == "claude-haiku-5-5"
    for settings in (query_settings(), query_settings(send_temperature=True, temperature=0.3)):
        plan = cons.plan_consultas([CTX_B, CTX_A], settings)
        assert [r.ids for r in plan] == [["id01"], ["id02"]]  # ordem estável, uma requisição por contexto
        for req in plan:
            assert req.params["model"] == "claude-haiku-5-5" and "temperature" not in req.params
            assert req.params["system"] == [
                {"type": "text", "text": cons.CONSULTAS_SYSTEM, "cache_control": {"type": "ephemeral"}}]
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", req.custom_id)
            user = req.params["messages"][0]["content"][0]["text"]
            assert f"id: {req.ids[0]}" in user and "Sentença: " in user
    assert "—" not in cons.CONSULTAS_SYSTEM and "–" not in cons.CONSULTAS_SYSTEM
    assert "5 a 12" in cons.CONSULTAS_SYSTEM


def test_consultas_batch_checkpoint_resume_without_resubmission(tmp_path):
    text, starts = long_book({20: SPECIAL_A})
    write_long_record(tmp_path, "kL", text, starts)
    write_evidence(tmp_path, "kS")  # fonte curta: não precisa de consulta
    contexts = [CTX_A, CTX_B, ctx(3, "kS")]
    client = query_client()
    batches = client.messages.batches
    pdir = tmp_path / "piloto" / "tese"

    def go(**kwargs):
        return cons.run_consultas(contexts=contexts, out_dir=tmp_path, pilot_dir=pdir, client=client,
                                  settings=query_settings(max_chars=120000, **kwargs), sleep=lambda s: None)

    first = go(espera_max_s=0.0)
    assert first["pedidas"] == 2 and first["incompleto"] is True and first["novas"] == 0
    assert batches.created == ["msgbatch_q1"] and tr.load_english_queries(tmp_path) == {}
    for request in batches.store["msgbatch_q1"]:
        assert request["params"]["model"] == "claude-haiku-5-5" and "temperature" not in request["params"]
        assert request["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    state = common.read_json(pdir / "consultas_checkpoint.json")
    assert state["lotes"][0]["id"] == "msgbatch_q1" and state["lotes"][0]["baixado"] is False

    second = go()
    assert batches.created == ["msgbatch_q1"]  # lote registrado: consultado, não reenviado
    assert second["novas"] == 2 and second["incompleto"] is False
    queries = tr.load_english_queries(tmp_path)
    assert set(queries) == {"id01", "id02"}  # kS (texto curto) fica de fora
    assert queries["id01"]["traducao"] == "English version of id01" and queries["id01"]["modelo"] == "claude-haiku-5-5"
    assert len(queries["id01"]["palavras_chave"]) == 5

    third = go()
    assert third["pedidas"] == 0 and batches.created == ["msgbatch_q1"]  # nada a pedir: já está em cache


def test_consultas_cleaning_of_answers_and_cache_file(tmp_path):
    assert cons.clean_query({"traducao": " Blocking ", "palavras_chave": [" a ", 3, ""]}) == {
        "traducao": "Blocking", "palavras_chave": ["a"]}
    assert cons.clean_query({"traducao": "", "palavras_chave": ["a"]}) is None
    assert cons.clean_query({"traducao": "x", "palavras_chave": []}) is None
    assert cons.clean_query({"traducao": "x"}) is None
    tr.save_english_queries(tmp_path, {CTX_A["id"]: EN_A})
    assert tr.load_english_queries(tmp_path) == {CTX_A["id"]: EN_A}
    assert tr.load_english_queries(tmp_path / "nada") == {}


def test_cli_consultas_options_and_missing_key(tmp_path, monkeypatch, capsys):
    parsed = cli.build_parser().parse_args(["consultas", "--escopo", "tese"])
    assert parsed.modelo == "claude-haiku-5-5" and parsed.modo == "batch" and parsed.func is cli.cmd_consultas
    assert parsed.capitulo is None and parsed.max_chars == 120000 and not hasattr(parsed, "temperature")
    assert cli.build_parser().parse_args(["consultas", "--capitulo", "Introdução"]).capitulo == "Introdução"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["consultas"])
    capsys.readouterr()
    write_inputs(tmp_path, [ctx(1, "kL")])
    monkeypatch.setattr(cli, "load_env_var", lambda name, env_file=None: "")
    assert cli.main(["consultas", "--escopo", "tese", "--out", str(tmp_path)]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
