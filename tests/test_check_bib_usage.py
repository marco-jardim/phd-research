"""Contrato de ``scripts/check_bib_usage.py``: chaves ausentes do .bib falham; órfãs avisam."""

import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_bib_usage.py"


def _projeto(tmp_path, tese, bib, extras=None):
    latex = tmp_path / "text" / "latex"
    latex.mkdir(parents=True)
    (latex / "referencias.bib").write_text(bib, encoding="utf-8")
    (latex / "tese.tex").write_text(tese, encoding="utf-8")
    for nome, texto in (extras or {}).items():
        (latex / nome).write_text(texto, encoding="utf-8")
    return tmp_path


def _rodar(raiz, *args, cwd=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--raiz", str(raiz), *args],
        cwd=cwd or raiz,
        capture_output=True,
        text=True,
    )


BIB = (
    "@article{Example2026, title={Example}, year={2026}}\n"
    "@article{Orfa2026, title={Orfa}, year={2026}}\n"
)


def test_citation_wrapper_arguments_are_not_keys(tmp_path):
    raiz = _projeto(
        tmp_path,
        r"\providecommand{\citeyearpar}[1]{(\citeyear{#1})}"
        r"\cite{Example2026}\citeonline{Missing2026}\bibliography{referencias}",
        BIB,
    )
    result = _rodar(raiz)
    assert "Total cited keys:   2" in result.stdout
    assert "Phantom (cited in .tex but NOT in .bib): 1" in result.stdout
    assert "  - Missing2026" in result.stdout
    assert "  - #1" not in result.stdout


def test_chave_ausente_do_bib_falha(tmp_path):
    raiz = _projeto(
        tmp_path, r"\cite{Example2026,Missing2026}\bibliography{referencias}", BIB
    )
    assert _rodar(raiz).returncode == 1


def test_orfa_e_aviso_salvo_estrito(tmp_path):
    raiz = _projeto(tmp_path, r"\cite{Example2026}\bibliography{referencias}", BIB)
    normal = _rodar(raiz)
    assert normal.returncode == 0
    assert "Orphan (in .bib but NOT cited in any .tex): 1" in normal.stdout
    assert "  - Orfa2026" in normal.stdout
    assert "aviso:" in normal.stderr
    assert _rodar(raiz, "--estrito").returncode == 1


def test_tudo_consistente_sai_com_zero(tmp_path):
    raiz = _projeto(
        tmp_path, r"\cite{Example2026}\citeonline{Orfa2026}\bibliography{referencias}", BIB
    )
    assert _rodar(raiz, "--estrito").returncode == 0


def test_caminhos_partem_da_raiz_e_nao_do_cwd(tmp_path):
    raiz = _projeto(tmp_path / "proj", r"\cite{Missing2026}\bibliography{referencias}", BIB)
    outro = tmp_path / "outro"
    outro.mkdir()
    assert _rodar(raiz, cwd=outro).returncode == 1


def test_so_a_arvore_de_tese_tex_conta(tmp_path):
    raiz = _projeto(
        tmp_path,
        r"\begin{document}\input{cap}\end{document}\bibliography{referencias}",
        BIB,
        extras={
            "cap.tex": r"\cite{Example2026}\citeonline{Orfa2026}",
            "tese_rev_diff.tex": r"\cite{FantasmaDoDiff2026}",
            "solto.tex": r"\cite{OutroFantasma2026}",
        },
    )
    result = _rodar(raiz, "--estrito")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FantasmaDoDiff2026" not in result.stdout
    assert "OutroFantasma2026" not in result.stdout


def test_input_ausente_e_erro_de_uso(tmp_path):
    raiz = _projeto(
        tmp_path,
        r"\begin{document}\input{cap_inexistente}\cite{Example2026}\end{document}\bibliography{referencias}",
        BIB,
    )
    result = _rodar(raiz)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "arquivo incluído não encontrado" in result.stderr
    assert "cap_inexistente" in result.stderr


def test_include_ausente_tambem_e_erro_de_uso(tmp_path):
    raiz = _projeto(
        tmp_path,
        r"\begin{document}\include{outro_inexistente}\end{document}\bibliography{referencias}",
        BIB,
    )
    assert _rodar(raiz).returncode == 2


def test_workflow_bib_reage_a_mudanca_nos_scripts():
    texto = (SCRIPT.parents[1] / ".github" / "workflows" / "latex.yml").read_text(encoding="utf-8")
    for evento in ("push", "pull_request"):
        bloco = re.split(r"\n  (?=\S)", texto.split(f"\n  {evento}:", 1)[1], maxsplit=1)[0]
        # Sem filtro de paths, o workflow já reage a qualquer mudança, inclusive nos scripts.
        sem_filtro = "paths:" not in bloco
        for caminho in ("scripts/check_bib_usage.py", "scripts/citacoes_contexto.py"):
            assert sem_filtro or caminho in bloco, (evento, caminho)


def test_citacao_comentada_nao_conta(tmp_path):
    raiz = _projeto(
        tmp_path,
        "\\cite{Example2026}\n% \\cite{Missing2026}\n\\bibliography{referencias}",
        BIB,
    )
    assert _rodar(raiz).returncode == 0


def test_usa_o_bib_declarado_em_tese_tex(tmp_path):
    raiz = _projeto(
        tmp_path,
        r"\cite{Example2026}\cite{Outra2026}\bibliography{outras}",
        BIB,
        extras={"outras.bib": "@article{Outra2026, title={Outra}, year={2026}}\n"},
    )
    result = _rodar(raiz)
    assert result.returncode == 1
    assert "  - Example2026" in result.stdout.split("Phantom")[1]


def test_sem_bibliography_e_erro_de_uso(tmp_path):
    raiz = _projeto(tmp_path, r"\cite{Example2026}", BIB)
    assert _rodar(raiz).returncode == 2
