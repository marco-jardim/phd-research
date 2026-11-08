"""Cobertura de números: classifica cada número do texto compilado da tese em
coberto (registrado), isento (data, rótulo, norma, estrutura LaTeX) ou descoberto."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cobertura_numeros as cn  # noqa: E402
import linter_prosa as lp  # noqa: E402
import registro_invariantes as ri  # noqa: E402


def _numeros(tmp_path, tex, cobertos=()):
    caminho = tmp_path / "t.tex"
    caminho.write_text(tex, encoding="utf-8")
    arq = lp.preparar_arquivo(caminho, "t.tex", False)
    return cn.classificar_arquivo(arq, set(cobertos))


def _por_numero(nums):
    return {n.numero: n for n in nums}


def _classes(tmp_path, tex, cobertos=()):
    return {n.numero: n.classe for n in _numeros(tmp_path, tex, cobertos)}


# --- classes -------------------------------------------------------------------

def test_coberto_quando_texto_pt_esta_registrado_para_o_arquivo(tmp_path):
    cls = _classes(tmp_path, "Foram 247 vínculos e 211 pares verdadeiros.\n", {"247"})
    assert cls == {"247": "coberto", "211": "descoberto"}


def test_coberto_compara_pelo_valor_e_nao_pelo_formato_do_milhar(tmp_path):
    cls = _classes(tmp_path, "Há 61.696 pares, dos quais 87,0\\% e 0,87.\n",
                   {"61.696", "87,0", "0,870"})
    assert cls == {"61.696": "coberto", "87,0%": "coberto", "0,87": "descoberto"}


def test_coberto_ignora_sinal_e_separador_tex(tmp_path):
    cls = _classes(tmp_path, "O ganho foi $-$5{,}65 e houve 1\\,234 casos.\n",
                   {"-5,65", "1.234"})
    assert cls == {"5,65": "coberto", "1234": "coberto"}


def test_coberto_com_texto_pt_nao_numerico(tmp_path):
    cls = _classes(tmp_path, "A razão foi de 1:249 entre as classes.\n", {"1:249"})
    assert cls["249"] == "coberto" and cls["1"] == "coberto"


@pytest.mark.parametrize("tex", [
    "Em 2019, o estudo começou.",
    "Publicado em junho de 2024.",
    "Entre 2010 e 2020, houve mudanças.",
    "Como mostram Silva et al. (2018), isso vale.",
    "O período 2015 a 2017 foi analisado.",
    "A coleta ocorreu em 03/2021.",
    "Registrado em 12/03/2019.",
])
def test_isento_ano_e_data(tmp_path, tex):
    classes = {n.classe for n in _numeros(tmp_path, tex + "\n")}
    assert classes == {"isento"}, tex


def test_ano_fora_do_contexto_ou_do_intervalo_nao_e_isento(tmp_path):
    assert _classes(tmp_path, "Foram 2019 pares.\n") == {"2019": "descoberto"}
    assert _classes(tmp_path, "Foram 2.019 casos em 2020.\n") == {
        "2.019": "descoberto", "2020": "isento"}
    assert _classes(tmp_path, "Em 1850 e em 2150 nada.\n") == {
        "1850": "descoberto", "2150": "descoberto"}


@pytest.mark.parametrize("tex", [
    "Ver Capítulo 3.",
    "Descrito na Seção 2.1.3.",
    "Ver Figura 4 e Tabela 7.",
    "Conforme a Equação 5.",
    "No passo 2 da blocagem.",
    "As Figuras 3 e 4 mostram.",
    "Veja a Tabela A.1.",
    "Nos Capítulos 2, 3 e 4.",
])
def test_isento_rotulos_numerados(tmp_path, tex):
    assert {n.classe for n in _numeros(tmp_path, tex + "\n")} == {"isento"}, tex


def test_rotulo_nao_isenta_o_numero_seguinte_do_texto(tmp_path):
    cls = _classes(tmp_path, "Na Tabela 7 há 211 pares.\n")
    assert cls == {"7": "isento", "211": "descoberto"}


@pytest.mark.parametrize("tex", [
    "Conforme a ABNT NBR 6023 e a NBR 14724.",
    "Segundo a Resolução CNS nº 466/2012.",
    "A Lei nº 13.709 trata de dados.",
    "Classificação CID-10 usada.",
    "Norma ISO 9001 citada.",
])
def test_isento_citacao_de_norma(tmp_path, tex):
    assert {n.classe for n in _numeros(tmp_path, tex + "\n")} == {"isento"}, tex


@pytest.mark.parametrize("tex", [
    "O modelo $k = 96$ foi usado.",
    r"O arquivo \texttt{run_2024_v3} foi lido.",
    r"Ver \cite{silva2019,costa88}.",
    r"\vspace{3cm} Fim.",
    r"\includegraphics[width=0.8\textwidth]{fig3.pdf} Fim.",
    r"\hspace{1.5em} Fim.",
    r"\setlength{\parindent}{12pt} Fim.",
    r"\rule{3cm}{0.4pt} Fim.",
    r"\renewcommand{\arraystretch}{1.2} Fim.",
    r"Ver \ref{tab:resultados2} e \label{fig:a1}.",
    r"O quadro tem 0.8\textwidth de largura.",
])
def test_isento_numeros_em_estrutura_latex(tmp_path, tex):
    assert {n.classe for n in _numeros(tmp_path, tex + "\n")} <= {"isento"}, tex


def test_isento_identificadores_com_letra_colada(tmp_path):
    cls = _classes(tmp_path, "O O3 e o H1 usaram K2.5 e COVID-19 em 247 casos.\n")
    assert cls["3"] == cls["1"] == cls["2.5"] == cls["19"] == "isento"
    assert cls["247"] == "descoberto"


def test_comentario_nao_conta(tmp_path):
    assert _classes(tmp_path, "% nota 999\nFoi 5 vezes. % 77\n") == {"5": "descoberto"}


def test_numero_em_titulo_ou_legenda_conta(tmp_path):
    tex = "\\section{Resultados com 12 grupos}\n\\caption{Há 88 pares.}\n"
    assert _classes(tmp_path, tex) == {"12": "descoberto", "88": "descoberto"}


# --- linha, frase e tabelas ----------------------------------------------------

def test_linha_e_frase(tmp_path):
    tex = "Primeira linha sem número.\nSegunda: foram 247 vínculos.\nOutra frase com 36.\n"
    por = _por_numero(_numeros(tmp_path, tex))
    assert por["247"].linha == 2 and por["247"].arquivo == "t.tex"
    assert por["247"].frase == "Segunda: foram 247 vínculos."
    assert por["36"].linha == 3 and por["36"].frase == "Outra frase com 36."


def test_frase_nao_quebra_em_abreviacao_nem_decimal(tmp_path):
    por = _por_numero(_numeros(
        tmp_path, "Ver Fig. 3 mostra 87,5\\% e 1.234 casos. Fim.\n"))
    assert por["87,5%"].frase == "Ver Fig. 3 mostra 87,5\\% e 1.234 casos."


def test_frase_do_csv_vai_ate_200_caracteres(tmp_path):
    longa = ("palavra " * 60) + "com 247 vínculos."
    n = _numeros(tmp_path, longa + "\n")[0]
    assert len(n.frase) > 200
    assert len(n.frase_csv()) == 200


def test_tabela_usa_a_linha_como_frase(tmp_path):
    tex = ("\\begin{tabular}{lrr}\n\\hline\n"
           "Modo & 211 & 0{,}87 \\\\\nOutro & 36 & 0{,}5 \\\\\n"
           "\\end{tabular}\n")
    por = _por_numero(_numeros(tmp_path, tex, {"211"}))
    assert por["211"].classe == "coberto"
    assert por["211"].frase == "Modo & 211 & 0{,}87 \\\\"
    assert por["0,5"].frase == "Outro & 36 & 0{,}5 \\\\"
    assert por["0,5"].classe == "descoberto"


# --- percentuais ----------------------------------------------------------------

def _n(classe):
    return cn.Numero("a.tex", 1, "1", classe, "", "f")


def test_percentuais():
    nums = [_n("coberto")] * 3 + [_n("isento")] + [_n("descoberto")]
    r = cn.resumir(nums)
    assert (r["total"], r["coberto"], r["isento"], r["descoberto"]) == (5, 3, 1, 1)
    assert r["pct_geral"] == pytest.approx(60.0)
    assert r["pct_nao_isentos"] == pytest.approx(75.0)


def test_percentual_sem_numeros_nao_isentos_e_cem():
    r = cn.resumir([_n("isento")])
    assert r["pct_nao_isentos"] == 100.0
    assert cn.resumir([])["pct_geral"] == 100.0


# --- registro -------------------------------------------------------------------

def test_textos_por_arquivo_do_registro():
    reg = ri.Registro([
        {"id": "N1", "tipo": "numero", "ocorrencias": [
            {"arquivo": "text/latex/a.tex", "exige": ["v1", "pares", "36"]},
            {"arquivo": "text/latex/b.tex", "exige_todos": [["v2", "v3"]]}],
         "valores": [{"nome": "v1", "fonte": {"arquivo": "x.csv"}, "texto_pt": "247"},
                     {"nome": "v2", "fonte": {"arquivo": "x.csv"}, "texto_pt": "1,5"},
                     {"nome": "v3", "fonte": {"arquivo": "x.csv"}, "texto_pt": "36"}]},
    ])
    por = cn.textos_por_arquivo(reg)
    assert por["text/latex/a.tex"] == {"247", "36"}  # "pares" não é valor
    assert por["text/latex/b.tex"] == {"1,5", "36"}


def test_arquivos_do_texto_compilado():
    ok = cn.eh_arquivo_compilado
    assert ok("text/latex/capitulo1.tex")
    assert ok("text/latex/tables/tab_x.tex")
    assert not ok("text/latex/tese_rev_diff.tex")
    assert not ok("text/latex/capitulo1_diff_v2.tex")
    assert not ok("text/slides/defesa.tex")
    assert not ok("text/slides/roteiro_defesa.md")


# --- árvore e CLI ---------------------------------------------------------------

REGISTRO_YAML = """\
- id: N01
  descricao: teste
  tipo: numero
  valores:
    - {nome: v247, fonte: {arquivo: x.csv}, texto_pt: "247"}
  ocorrencias:
    - {arquivo: cap.tex, exige: [v247]}
"""


def _montar(tmp_path):
    (tmp_path / "tese.tex").write_text(
        "\\documentclass{abntex2}\n\\begin{document}\n\\input{cap}\n\\input{outro}\n"
        "\\input{tese_rev_diff}\n\\end{document}\n", encoding="utf-8")
    (tmp_path / "cap.tex").write_text("Foram 247 vínculos e 36 pares.\n", encoding="utf-8")
    (tmp_path / "outro.tex").write_text("Ver Capítulo 3 sobre 247 casos.\n", encoding="utf-8")
    (tmp_path / "tese_rev_diff.tex").write_text("Sobra 999 aqui.\n", encoding="utf-8")
    (tmp_path / "registro.yaml").write_text(REGISTRO_YAML, encoding="utf-8")
    return tmp_path


def test_analisar_percorre_a_arvore_sem_o_diff(tmp_path):
    base = _montar(tmp_path)
    reg = ri.carregar_registro(base / "registro.yaml")
    nums = cn.analisar(base / "tese.tex", reg, base)
    por = {(n.arquivo, n.numero): n.classe for n in nums}
    assert por[("cap.tex", "247")] == "coberto"
    assert por[("cap.tex", "36")] == "descoberto"
    # 247 está registrado só para cap.tex, não para outro.tex
    assert por[("outro.tex", "247")] == "descoberto"
    assert por[("outro.tex", "3")] == "isento"
    assert all(n.arquivo != "tese_rev_diff.tex" for n in nums)


def test_cli_grava_csv_resumo_e_codigo_de_saida(tmp_path, capsys):
    base = _montar(tmp_path)
    saida = base / "out" / "numeros.csv"
    args = ["--tese", str(base / "tese.tex"), "--registro", str(base / "registro.yaml"),
            "--raiz", str(base), "--saida", str(saida)]
    assert cn.main(args) == 0
    texto = capsys.readouterr().out
    assert "cap.tex" in texto and "descobertos" in texto.lower()
    with open(saida, encoding="utf-8", newline="") as f:
        linhas = list(csv.reader(f))
    assert linhas[0] == ["arquivo", "linha", "numero", "classe", "frase"]
    assert ["cap.tex", "1", "247", "coberto", "Foram 247 vínculos e 36 pares."] in linhas
    # coberto 1 de 3 não isentos (247 em cap; 36; 247 em outro) = 33,3%
    assert cn.main([*args, "--min-cobertura", "30"]) == 0
    assert cn.main([*args, "--min-cobertura", "40"]) == 1


@pytest.mark.skipif(not lp.TESE_PADRAO.exists(), reason="tese.tex ausente")
def test_arvore_real_nao_inclui_diff_nem_slides():
    nums = cn.analisar(lp.TESE_PADRAO, ri.carregar_registro(), lp.REPO)
    assert nums
    arquivos = {n.arquivo for n in nums}
    assert not any("_diff" in a or a.startswith("text/slides") for a in arquivos)
    assert {n.classe for n in nums} <= {"coberto", "isento", "descoberto"}
