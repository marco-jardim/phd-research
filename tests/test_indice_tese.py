"""Testes de scripts/indice_tese.py (índice de navegação da tese)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import citacoes_contexto as cc
from scripts import indice_tese as it

REPO = Path(__file__).resolve().parents[1]
TESE_REAL = REPO / "text" / "latex" / "tese.tex"

ROOT_TEX = """\\documentclass{book}
\\begin{document}
\\input{cap1.tex}
\\appendix
\\input{apx.tex}
\\end{document}
"""

CAP1_TEX = """\\begin{chapter}{Introdução}\\label{cap:intro}
Primeiro parágrafo cita \\cite{Silva2020a}. Veja a Seção~\\ref{sec:met}.

\\section{Métodos}\\label{sec:met}
Texto de métodos usa \\path{data/x.csv} e cita \\cite{Silva2020a,Souza2019b}. Ver \\autoref{cap:intro} e \\autoref{tab:a}.

\\input{tables/tab_a.tex}

\\subsection{Detalhe}
Texto do detalhe \\cite{Souza2019b}.
\\end{chapter}
"""

TAB_TEX = """\\begin{table}\\caption{Tabela A}\\label{tab:a}
\\begin{tabular}{l}
x \\\\
\\end{tabular}
\\end{table}
"""

APX_TEX = """\\chapter{Material suplementar}\\label{ap:sup}
Texto do apêndice cita \\cite{Silva2020a} e usa \\eqref{eq:x}.
"""

REGISTRO_YAML = """\
- id: C01
  descricao: "Afirmação | com barra"
  tipo: afirmacao
  ocorrencias:
    - {arquivo: apx.tex, exige_todos: [a]}
    - {arquivo: cap1.tex, exige_todos: [a]}
- id: N01
  descricao: "Total"
  tipo: numero
  ocorrencias:
    - {arquivo: cap1.tex, exige: [x]}
    - {arquivo: tese.tex, secao: resumo, exige: [x]}
    - {arquivo: slides/defesa.tex, exige: [x]}
"""

# Baseline do linter, formato v1 (contagens) e v2 (uma entrada por violação); mesmas contagens.
BASELINE_V1 = {
    "versao": 1,
    "contagens": {
        "R02": {"cap1.tex": 3, "tables/tab_a.tex": 1, "outro.tex": 2},
        "R09": {"cap1.tex": 1},
    },
}


def _baseline_v2() -> dict:
    pares = [("R02", "cap1.tex")] * 3 + [("R02", "tables/tab_a.tex")] + [("R02", "outro.tex")] * 2
    pares += [("R09", "cap1.tex")]
    return {
        "versao": 2,
        "gerado_em": "2026-11-08T00:00:00",
        "violacoes": {
            f"fp{i:02d}": {"regra": regra, "arquivo": arq, "trecho": f"trecho {i}"}
            for i, (regra, arq) in enumerate(pares)
        },
    }


BASELINE_JSON = _baseline_v2()  # formato atual, usado nos testes de saída
CONTAGENS_ESPERADAS = {"R02": {"cap1.tex": 3, "tables/tab_a.tex": 1, "outro.tex": 2}, "R09": {"cap1.tex": 1}}


def _escrever(caminho: Path, texto: str) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8", newline="\n")
    return caminho


@pytest.fixture()
def projeto(tmp_path: Path) -> Path:
    _escrever(tmp_path / "tese.tex", ROOT_TEX)
    _escrever(tmp_path / "cap1.tex", CAP1_TEX)
    _escrever(tmp_path / "tables" / "tab_a.tex", TAB_TEX)
    _escrever(tmp_path / "apx.tex", APX_TEX)
    return tmp_path


def _indice(base: Path, **kw) -> dict:
    return it.construir_indice(base / "tese.tex", base, **kw)


def _por_titulo(indice: dict) -> dict[str, dict]:
    return {n["titulo"]: n for n in indice["nos"]}


# ---------------------------------------------------------------------------
# Árvore
# ---------------------------------------------------------------------------
def test_arvore_com_input_aninhado_e_begin_chapter(projeto: Path) -> None:
    indice = _indice(projeto)
    nos = indice["nos"]
    assert [n["titulo"] for n in nos] == ["Introdução", "Métodos", "Detalhe", "Material suplementar"]
    assert [n["nivel"] for n in nos] == [0, 1, 2, 0]
    assert [n["numero"] for n in nos] == ["1", "1.1", "1.1.1", "A"]
    assert [n["apendice"] for n in nos] == [False, False, False, True]
    assert nos[2]["pai"] == nos[1]["id"] and nos[1]["pai"] == nos[0]["id"] and nos[3]["pai"] is None
    assert nos[0]["filhos"] == [nos[1]["id"]]
    assert nos[3]["rotulo"] == "Apêndice A Material suplementar"
    assert nos[0]["rotulo"] == "Cap. 1 Introdução"


def test_arvore_intervalos_de_linhas(projeto: Path) -> None:
    por = _por_titulo(_indice(projeto))
    assert (por["Introdução"]["linha_ini"], por["Introdução"]["linha_fim"]) == (1, 11)
    assert (por["Métodos"]["linha_ini"], por["Métodos"]["linha_fim"]) == (4, 11)
    assert (por["Detalhe"]["linha_ini"], por["Detalhe"]["linha_fim"]) == (9, 11)
    assert (por["Material suplementar"]["linha_ini"], por["Material suplementar"]["linha_fim"]) == (1, 2)
    assert por["Métodos"]["arquivo"] == "cap1.tex"
    assert por["Material suplementar"]["arquivo"] == "apx.tex"


def test_palavras_e_paragrafos_longos(projeto: Path) -> None:
    longo = " ".join(["palavra"] * 201)
    _escrever(projeto / "cap1.tex", CAP1_TEX.replace("Texto do detalhe", longo + " Texto do detalhe"))
    por = _por_titulo(_indice(projeto))
    assert por["Detalhe"]["paragrafos_longos"] == 1
    assert por["Detalhe"]["palavras"] > 201
    assert por["Introdução"]["paragrafos_longos"] == 0
    assert por["Introdução"]["palavras"] > 0
    assert por["Introdução"]["palavras_total"] >= por["Métodos"]["palavras_total"] > por["Detalhe"]["palavras"] - 1


def test_resumo_e_abstract_viram_blocos(tmp_path: Path) -> None:
    raiz = (
        "\\documentclass{book}\n\\begin{document}\n"
        "{\\LARGE\\bf Resumo}\\\\\nTotal de 247 vínculos no extrato.\n\n"
        "\\textbf{Palavras-chave:} a. b.\n"
        "{\\LARGE\\bf Abstract}\\\\\nTotal of 247 links.\n\n"
        "\\textbf{Keywords:} a. b.\n"
        "\\input{cap1.tex}\n\\end{document}\n"
    )
    _escrever(tmp_path / "tese.tex", raiz)
    _escrever(tmp_path / "cap1.tex", CAP1_TEX)
    _escrever(tmp_path / "tables" / "tab_a.tex", TAB_TEX)
    reg = _escrever(
        tmp_path / "registro.yaml",
        "- id: N01\n  descricao: x\n  tipo: numero\n  ocorrencias:\n"
        "    - {arquivo: tese.tex, secao: resumo, exige: [a]}\n"
        "    - {arquivo: tese.tex, secao: abstract, exige: [a]}\n",
    )
    indice = _indice(tmp_path, registro=reg)
    nos = indice["nos"]
    assert [(n["tipo"], n["titulo"]) for n in nos[:2]] == [("bloco", "Resumo"), ("bloco", "Abstract")]
    assert (nos[0]["linha_ini"], nos[0]["linha_fim"]) == (3, 6)
    assert (nos[1]["linha_ini"], nos[1]["linha_fim"]) == (7, 10)
    assert nos[0]["registro"] == ["N01"] and nos[1]["registro"] == ["N01"]
    assert nos[2]["titulo"] == "Introdução" and nos[2]["registro"] == []


# ---------------------------------------------------------------------------
# Rótulos, referências, citações, artefatos
# ---------------------------------------------------------------------------
def test_indice_reverso_de_rotulos_e_referencias(projeto: Path) -> None:
    indice = _indice(projeto)
    por = _por_titulo(indice)
    rot = indice["rotulos"]
    assert list(rot) == ["ap:sup", "cap:intro", "sec:met", "tab:a"]
    # tab:a é definido no arquivo de tabela, que pertence à seção Métodos.
    assert [d["no"] for d in rot["tab:a"]["definido"]] == [por["Métodos"]["id"]]
    assert rot["tab:a"]["definido"][0]["arquivo"] == "tables/tab_a.tex"
    assert [r["no"] for r in rot["tab:a"]["referenciado_em"]] == [por["Métodos"]["id"]]
    assert rot["sec:met"]["referenciado_em"][0]["no"] == por["Introdução"]["id"]
    assert rot["sec:met"]["referenciado_em"][0]["comando"] == "ref"
    assert rot["cap:intro"]["referenciado_em"][0]["comando"] == "autoref"
    assert indice["rotulos_sem_referencia"] == ["ap:sup"]
    assert list(indice["referencias_sem_rotulo"]) == ["eq:x"]
    assert indice["referencias_sem_rotulo"]["eq:x"][0]["comando"] == "eqref"
    assert por["Métodos"]["rotulos"] == ["sec:met", "tab:a"]
    assert sorted(r["alvo"] for r in por["Métodos"]["referencias"]) == ["cap:intro", "tab:a"]


def test_citacoes_por_secao(projeto: Path) -> None:
    por = _por_titulo(_indice(projeto))
    assert por["Introdução"]["citacoes"] == {"Silva2020a": 1}
    assert por["Métodos"]["citacoes"] == {"Silva2020a": 1, "Souza2019b": 1}
    assert por["Detalhe"]["citacoes"] == {"Souza2019b": 1}
    assert por["Material suplementar"]["citacoes"] == {"Silva2020a": 1}


def test_figuras_tabelas_e_artefatos(projeto: Path) -> None:
    por = _por_titulo(_indice(projeto))
    ft = por["Métodos"]["figuras_tabelas"]
    assert [(f["tipo"], f["alvo"]) for f in ft] == [("tabela", "tables/tab_a.tex")]
    assert [a["caminho"] for a in por["Métodos"]["artefatos"]] == ["data/x.csv"]
    assert por["Introdução"]["artefatos"] == []


def test_includegraphics_e_figura_pgf(projeto: Path) -> None:
    _escrever(
        projeto / "cap1.tex",
        CAP1_TEX.replace(
            "\\subsection{Detalhe}",
            "\\includegraphics[width=5cm]{figures/f1.pdf}\n\\input{figures/f2.pgf}\n\n\\subsection{Detalhe}",
        ),
    )
    por = _por_titulo(_indice(projeto))
    alvos = sorted((f["tipo"], f["alvo"]) for f in por["Métodos"]["figuras_tabelas"])
    assert alvos == [
        ("figura", "figures/f1.pdf"),
        ("figura", "figures/f2.pgf"),
        ("tabela", "tables/tab_a.tex"),
    ]


# ---------------------------------------------------------------------------
# Junções com registro, atestação e linter
# ---------------------------------------------------------------------------
def test_juncao_com_registro(projeto: Path) -> None:
    reg = _escrever(projeto / "registro.yaml", REGISTRO_YAML)
    indice = _indice(projeto, registro=reg)
    por = _por_titulo(indice)
    # N antes de C; vale por arquivo, então a subseção herda as entradas do cap1.tex.
    assert por["Introdução"]["registro"] == ["N01", "C01"]
    assert por["Detalhe"]["registro"] == ["N01", "C01"]
    assert por["Material suplementar"]["registro"] == ["C01"]
    assert list(indice["registro"]) == ["N01", "C01"]
    assert indice["registro"]["N01"]["ocorrencias"] == ["cap1.tex", "slides/defesa.tex", "tese.tex [resumo]"]
    assert indice["registro"]["N01"]["capitulos"] == ["Cap. 1 Introdução"]
    assert indice["registro"]["C01"]["capitulos"] == ["Apêndice A Material suplementar", "Cap. 1 Introdução"]
    caps = {c["rotulo"]: c for c in indice["resumo"]["capitulos"]}
    assert caps["Cap. 1 Introdução"]["registro"] == ["N01", "C01"]
    assert indice["resumo"]["total"]["registro"] == 2


def test_juncao_com_ledger_e_pendencias(projeto: Path) -> None:
    recs = cc.extract_citations(cc.build_document(projeto / "tese.tex", projeto))
    id_met = next(r["id"] for r in recs if r["key"] == "Souza2019b" and r["line"] == 5)
    cab = "id;chave;arquivo;capitulo;veredito_final;nivel_evidencia;adjudicado;modelo;data\n"
    ledger = _escrever(
        projeto / "ledger.csv",
        cab
        + f"{id_met};Souza2019b;cap1.tex;Introdução;SUSTENTA;texto_completo;nao;m;2026-11-08\n"
        + "ffffffffffffffff;Silva2020a;apx.tex;Apêndice;NAO_SUSTENTA;resumo;nao;m;2026-11-08\n"
        + "0000000000000000;Inexistente;cap1.tex;Introdução;SUSTENTA;resumo;nao;m;2026-11-08\n",
    )
    pend = _escrever(
        projeto / "pendencias.yaml",
        "pendencias:\n"
        f"- {{id: '{id_met}', chave: Souza2019b, arquivo: cap1.tex, decisao: pendente}}\n"
        "- {id: 'aaaa', chave: Silva2020a, arquivo: cap1.tex, decisao: aceita}\n"
        "- {id: 'bbbb', chave: Nada, arquivo: cap1.tex, decisao: pendente}\n",
    )
    indice = _indice(projeto, ledger=ledger, pendencias=pend)
    por = _por_titulo(indice)
    assert por["Métodos"]["atestacao"] == {"SUSTENTA": 1}
    assert por["Material suplementar"]["atestacao"] == {"NAO_SUSTENTA": 1}  # via arquivo + chave
    assert indice["atestacao"]["ledger_sem_secao"] == 1
    assert [p["chave"] for p in por["Métodos"]["pendencias"]] == ["Souza2019b"]
    assert [p["no"] for p in indice["atestacao"]["pendencias_abertas"]] == [por["Métodos"]["id"]]
    assert [p["chave"] for p in indice["atestacao"]["pendencias_sem_secao"]] == ["Nada"]
    caps = {c["rotulo"]: c for c in indice["resumo"]["capitulos"]}
    assert caps["Cap. 1 Introdução"]["atestacao"] == {"SUSTENTA": 1}
    assert caps["Cap. 1 Introdução"]["pendencias"] == 1
    assert caps["Apêndice A Material suplementar"]["pendencias"] == 0
    assert indice["resumo"]["total"]["pendencias"] == 1


@pytest.mark.parametrize("formato", ["v1", "v2"])
def test_ler_baseline_aceita_v1_e_v2(tmp_path: Path, formato: str) -> None:
    dados = BASELINE_V1 if formato == "v1" else _baseline_v2()
    caminho = _escrever(tmp_path / "baseline.json", json.dumps(dados))
    assert it.ler_baseline(caminho) == CONTAGENS_ESPERADAS
    assert it.ler_baseline(tmp_path / "nao_existe.json") == {}


@pytest.mark.parametrize("formato", ["v1", "v2"])
def test_juncao_com_linter(projeto: Path, formato: str) -> None:
    dados = BASELINE_V1 if formato == "v1" else _baseline_v2()
    base = _escrever(projeto / "baseline.json", json.dumps(dados))
    indice = _indice(projeto, baseline=base)
    assert indice["linter"]["regras"] == ["R02", "R09"]
    arq = indice["linter"]["arquivos"]
    assert arq["cap1.tex"]["linter"] == {"R02": 3, "R09": 1}
    assert arq["tables/tab_a.tex"]["capitulo"] == "Cap. 1 Introdução"
    assert indice["linter"]["fora_da_arvore"] == {"outro.tex": {"R02": 2}}
    caps = {c["rotulo"]: c for c in indice["resumo"]["capitulos"]}
    cap1 = caps["Cap. 1 Introdução"]
    assert cap1["linter"] == {"R02": 4, "R09": 1} and cap1["linter_total"] == 5
    assert caps["Apêndice A Material suplementar"]["linter_total"] == 0
    assert indice["resumo"]["total"]["linter_total"] == 5


def test_resumo_por_capitulo(projeto: Path) -> None:
    indice = _indice(projeto)
    caps = indice["resumo"]["capitulos"]
    assert [c["rotulo"] for c in caps] == ["Cap. 1 Introdução", "Apêndice A Material suplementar"]
    assert caps[0]["citacoes"] == 4 and caps[0]["chaves_distintas"] == 2  # 1 + 2 + 1 nas três seções
    assert caps[1]["citacoes"] == 1
    assert caps[0]["arquivo"] == "cap1.tex" and (caps[0]["linha_ini"], caps[0]["linha_fim"]) == (1, 11)
    assert caps[0]["arquivos"] == ["cap1.tex", "tables/tab_a.tex"]
    assert indice["resumo"]["total"]["citacoes"] == 5
    assert indice["resumo"]["total"]["palavras"] == sum(c["palavras"] for c in caps)


# ---------------------------------------------------------------------------
# Saídas
# ---------------------------------------------------------------------------
def test_markdown_e_json(projeto: Path) -> None:
    reg = _escrever(projeto / "registro.yaml", REGISTRO_YAML)
    base = _escrever(projeto / "baseline.json", json.dumps(BASELINE_JSON))
    indice = _indice(projeto, registro=reg, baseline=base)
    md = it.formatar_markdown(indice)
    assert md.startswith("# Índice da tese")
    for trecho in (
        "## Resumo por capítulo",
        "## Estrutura",
        "## Rótulos e referências (índice reverso)",
        "## Registro de invariantes",
        "## Atestação bibliográfica",
        "## Linter de prosa (por arquivo)",
        "### Cap. 1 Introdução `cap1.tex:1-11`",
        "- **1.1 Métodos** `cap1.tex:4-11`",
        "  - **1.1.1 Detalhe** `cap1.tex:9-11`",
        "tab:a",
        "`data/x.csv`",
        "Afirmação \\| com barra",
    ):
        assert trecho in md, trecho
    assert "\r" not in md
    assert it.serializar_json(indice).endswith("}\n")
    assert json.loads(it.serializar_json(indice)) == indice


def test_saida_deterministica(projeto: Path) -> None:
    reg = _escrever(projeto / "registro.yaml", REGISTRO_YAML)
    base = _escrever(projeto / "baseline.json", json.dumps(BASELINE_JSON))
    a = _indice(projeto, registro=reg, baseline=base)
    b = _indice(projeto, registro=reg, baseline=base)
    assert it.serializar_json(a) == it.serializar_json(b)
    assert it.formatar_markdown(a) == it.formatar_markdown(b)


def test_cli_grava_markdown_e_json(projeto: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md = projeto / "saida" / "INDICE.md"
    js = projeto / "saida" / "indice.json"
    ausente = projeto / "nao_existe"
    rc = it.main(
        [
            "--tese", str(projeto / "tese.tex"),
            "--base", str(projeto),
            "--saida", str(md),
            "--json", str(js),
            "--registro", str(ausente),
            "--ledger", str(ausente),
            "--pendencias", str(ausente),
            "--baseline", str(ausente),
        ]
    )
    assert rc == 0
    assert md.read_text(encoding="utf-8").startswith("# Índice da tese")
    dados = json.loads(js.read_text(encoding="utf-8"))
    assert dados["versao"] == it.VERSAO and len(dados["nos"]) == 4
    assert "2 capítulos" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tese real
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not TESE_REAL.is_file(), reason="text/latex/tese.tex ausente")
def test_indice_da_tese_real_cobre_os_capitulos_incluidos() -> None:
    indice = it.construir_indice(
        TESE_REAL,
        REPO,
        it.REGISTRO_PADRAO,
        it.LEDGER_PADRAO,
        it.PENDENCIAS_PADRAO,
        it.BASELINE_PADRAO,
    )
    arquivos_cap = {c["arquivo"] for c in indice["resumo"]["capitulos"]}
    tex = TESE_REAL.read_text(encoding="utf-8-sig")
    esperados = [
        m
        for m in re.findall(r"^[^%\n]*\\input\{([^}]+)\}", tex, re.M)
        if re.match(r"(capitulo|estudo|conclusao|apendice)", Path(m).name)
    ]
    assert esperados, "nenhum capítulo incluído em tese.tex"
    for nome in esperados:
        nome = nome if nome.endswith(".tex") else nome + ".tex"
        assert f"text/latex/{nome}" in arquivos_cap, nome
    assert indice["resumo"]["total"]["palavras"] > 0
    assert it.formatar_markdown(indice).startswith("# Índice da tese")
    json.loads(it.serializar_json(indice))
