"""Portão de fidelidade: comparação de parágrafos entre base e alvo."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import portao_fidelidade as pf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "portao_fidelidade.py"
FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, creationflags=FLAGS
    )


def escrever(repo, rel, texto):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texto, encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "teste")
    git(tmp_path, "config", "user.email", "teste@example.org")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.autocrlf", "false")
    return tmp_path


def base_commit(repo, arquivos):
    for rel, texto in arquivos.items():
        escrever(repo, rel, texto)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "tag", "pre-revisao-2")


def analisar(antigo, novo, aceites=None):
    achados, _ = pf.analisar_textos("text/latex/x.tex", antigo, novo, aceites)
    return achados


def alertas(achados):
    return [a for a in achados if a.alerta]


# ---------------------------------------------------------------------------
# Normalização e segmentação
# ---------------------------------------------------------------------------


def test_normalizar_numero_pt_en():
    n = pf.normalizar_numero
    assert n("39,3") == n("39.3") == "39.3"
    assert n("84.368") == "84368"
    assert n("1.234,5") == "1234.5"
    assert n("1,234.5") == "1234.5"
    assert n("1.234.567") == "1234567"
    assert n("0.368") == "0.368"
    assert n("42") == "42"


def test_numeros_latex_e_porcentagem():
    tok = pf.extrair_tokens(r"Taxa de 39{,}3\% e 39,3 \% com 84\,368 pares e 12.5.")
    assert tok["numeros"]["39.3%"] == 2
    assert tok["numeros"]["84368"] == 1
    assert tok["numeros"]["12.5"] == 1
    assert "39.3" not in tok["numeros"]


def test_numeros_pt_en_equivalentes_nao_alertam():
    antigo = r"A taxa foi de 39,3\% em 84.368 pares."
    novo = r"A taxa foi de 39.3\% em 84368 pares."
    assert alertas(analisar(antigo, novo)) == []


def test_comentarios_nao_entram_na_segmentacao():
    texto = "Primeiro % comentario 99\n% linha inteira 77\ncontinua.\n\nSegundo 50\\% igual.\n"
    pars = pf.segmentar(texto)
    assert pars == ["Primeiro continua.", r"Segundo 50\% igual."]


# ---------------------------------------------------------------------------
# Detecção de mudanças
# ---------------------------------------------------------------------------


def test_remocao_de_negador():
    ach = alertas(analisar("O método não foi validado em campo.", "O método foi validado em campo."))
    assert len(ach) == 1
    assert ach[0].tipo == "modificado"
    assert ach[0].diferencas["negadores"]["removidos"] == ["não"]
    assert "negadores:-não" in ach[0].itens


def test_troca_de_chave_de_citacao():
    antigo = r"Conforme \citeonline{silva2020}, a taxa é alta."
    novo = r"Conforme \citeonline{souza2019}, a taxa é alta."
    ach = alertas(analisar(antigo, novo))
    assert len(ach) == 1
    assert ach[0].diferencas["citacoes"] == {"removidos": ["silva2020"], "adicionados": ["souza2019"]}
    # o ano dentro da chave não vira número solto
    assert "numeros" not in ach[0].diferencas


def test_citacao_com_varias_chaves_e_opcional():
    tok = pf.extrair_tokens(r"Ver \cite[p.~3]{a1, b2} e \citeyearpar{c3}.")
    assert sorted(tok["citacoes"]) == ["a1", "b2", "c3"]


def test_mudanca_de_ref():
    antigo = r"Ver Tabela~\ref{tab:a} para detalhes."
    novo = r"Ver Tabela~\ref{tab:b} para detalhes."
    ach = alertas(analisar(antigo, novo))
    assert len(ach) == 1
    assert ach[0].diferencas["referencias"] == {"removidos": ["ref:tab:a"], "adicionados": ["ref:tab:b"]}


def test_mudanca_de_sigla():
    antigo = "Os dados vêm do SIM e do SIH."
    novo = "Os dados vêm do SIM e do Sinan."
    ach = alertas(analisar(antigo, novo))
    assert len(ach) == 1
    assert ach[0].diferencas["siglas"] == {"removidos": ["SIH"], "adicionados": ["SINAN"]}


def test_quantificador_e_hedge():
    ach = alertas(analisar("O resultado sugere ganho em todos os casos.", "O resultado demonstra ganho em todos os casos."))
    assert len(ach) == 1
    assert ach[0].diferencas["quantificadores"] == {
        "removidos": ["sugere"],
        "adicionados": ["demonstra"],
    }
    # concordância não é mudança de sentido
    assert alertas(analisar("Isso pode ocorrer.", "Isso podem ocorrer.")) == []


def test_mudanca_de_caminho():
    ach = alertas(analisar(r"Veja \path{data/a.csv} agora.", r"Veja \path{data/b.csv} agora."))
    assert ach[0].diferencas["caminhos"] == {"removidos": ["data/a.csv"], "adicionados": ["data/b.csv"]}


def test_numero_alterado_e_trecho_limitado():
    longo = "Palavra " * 80
    antigo = longo + r"taxa de 39,3\% ao final."
    novo = longo + r"taxa de 93,3\% ao final."
    ach = alertas(analisar(antigo, novo))
    assert ach[0].diferencas["numeros"] == {"removidos": ["39.3%"], "adicionados": ["93.3%"]}
    assert len(ach[0].antigo) <= 300 and len(ach[0].novo) <= 300


def test_paragrafo_movido_nao_e_sinalizado():
    a = "Primeiro parágrafo com 12 casos e nenhum erro."
    b = r"Segundo parágrafo cita \cite{x1} sem ressalvas."
    c = "Terceiro parágrafo trata de outro assunto, 7 vezes."
    achados, cont = pf.analisar_textos(
        "x.tex", f"{a}\n\n{b}\n\n{c}\n", f"{c}\n\n{a}\n\n{b}\n"
    )
    assert alertas(achados) == []
    assert cont["movidos"] + cont["iguais"] == 3


def test_reescrita_com_mesmos_tokens_nao_e_sinalizada():
    antigo = r"A taxa foi de 39,3\% \cite{silva2020} e não houve perda no SIM."
    novo = r"Não houve perda no SIM, e a taxa foi de 39,3\% \cite{silva2020}."
    # "não" continua uma vez; só a ordem e a caixa mudaram
    assert alertas(analisar(antigo, novo)) == []


def test_paragrafo_removido_com_numero_e_alerta():
    p1 = "Introdução geral do tema sem dados."
    p2 = "Observamos 84.368 pares distintos na amostra final."
    ach = alertas(analisar(f"{p1}\n\n{p2}\n", f"{p1}\n"))
    assert len(ach) == 1
    assert ach[0].tipo == "removido"
    assert ach[0].diferencas["numeros"]["removidos"] == ["84368"]


def test_paragrafo_removido_sem_itens_nao_e_alerta():
    p1 = "Introdução geral do tema."
    p2 = "Um comentário puramente narrativo sobre o contexto."
    achados = analisar(f"{p1}\n\n{p2}\n", f"{p1}\n")
    assert alertas(achados) == []
    assert [a.tipo for a in achados] == ["removido"]


def test_paragrafo_adicionado_com_citacao_e_alerta():
    p1 = "Introdução geral do tema."
    p2 = r"Um achado novo foi relatado por \cite{novo2024}."
    ach = alertas(analisar(f"{p1}\n", f"{p1}\n\n{p2}\n"))
    assert len(ach) == 1
    assert ach[0].tipo == "adicionado"
    assert ach[0].motivo == "novo conteúdo factual"
    assert ach[0].diferencas["citacoes"]["adicionados"] == ["novo2024"]


# ---------------------------------------------------------------------------
# Aceites
# ---------------------------------------------------------------------------


def aceite_para(achado, **extra):
    base = {
        "arquivo": achado.arquivo,
        "hash_paragrafo_novo": achado.hash,
        "itens": [],
        "justificativa": "conferido",
        "autora_aprovou": True,
    }
    base.update(extra)
    return base


def test_hash_do_paragrafo_normalizado():
    assert pf.hash_paragrafo("a  b\n c") == pf.hash_paragrafo("a b c")
    assert len(pf.hash_paragrafo("a")) == 16


def test_aceite_suprime_falha():
    antigo = "O método não foi validado em campo."
    novo = "O método foi validado em campo."
    ach = alertas(analisar(antigo, novo))[0]
    assert ach.status == "pendente"
    aceitos = analisar(antigo, novo, [aceite_para(ach)])
    assert alertas(aceitos)[0].status == "aceito"
    assert alertas(aceitos)[0].nota_aceite == "conferido"


def test_aceite_sem_aprovacao_ou_incompleto_nao_suprime():
    antigo = "O método não foi validado em campo."
    novo = "O método foi validado em campo."
    ach = alertas(analisar(antigo, novo))[0]
    sem_aprov = analisar(antigo, novo, [aceite_para(ach, autora_aprovou=False)])
    assert alertas(sem_aprov)[0].status == "pendente"
    outro_item = analisar(antigo, novo, [aceite_para(ach, itens=["numeros:-1"])])
    assert alertas(outro_item)[0].status == "pendente"
    item_certo = analisar(antigo, novo, [aceite_para(ach, itens=["negadores:-não"])])
    assert alertas(item_certo)[0].status == "aceito"


def test_parser_simples_do_yaml_de_aceites():
    texto = (
        "# comentário\n"
        "aceites:\n"
        "  - arquivo: text/latex/x.tex\n"
        '    hash_paragrafo_novo: "0123456789abcdef"\n'
        "    itens:\n"
        "      - numeros:-39.3\n"
        "      - citacoes:+silva2020\n"
        '    justificativa: "Número corrigido."\n'
        "    autora_aprovou: true\n"
        "  - arquivo: text/latex/y.tex\n"
        "    hash_paragrafo_novo: fedcba9876543210\n"
        "    itens: [negadores:-não, siglas:+SIM]\n"
        "    autora_aprovou: false\n"
    )
    ent = pf.parse_aceites_simples(texto)
    assert len(ent) == 2
    assert ent[0]["hash_paragrafo_novo"] == "0123456789abcdef"
    assert ent[0]["itens"] == ["numeros:-39.3", "citacoes:+silva2020"]
    assert ent[0]["autora_aprovou"] is True
    assert ent[1]["itens"] == ["negadores:-não", "siglas:+SIM"]
    assert ent[1]["autora_aprovou"] is False
    assert pf.parse_aceites_simples("aceites: []\n") == []


def test_template_de_aceites_do_repositorio_esta_vazio():
    assert pf.carregar_aceites(ROOT / "docs" / "fidelidade_aceites.yaml") == []


def test_carregar_aceites_normaliza(tmp_path):
    arq = tmp_path / "a.yaml"
    arq.write_text(
        "aceites:\n"
        "  - arquivo: text/latex/x.tex\n"
        '    hash_paragrafo_novo: "0123456789abcdef"\n'
        "    justificativa: ok\n"
        "    autora_aprovou: true\n",
        encoding="utf-8",
    )
    ent = pf.carregar_aceites(arq)
    assert ent == [
        {
            "arquivo": "text/latex/x.tex",
            "hash_paragrafo_novo": "0123456789abcdef",
            "itens": [],
            "justificativa": "ok",
            "autora_aprovou": True,
        }
    ]


# ---------------------------------------------------------------------------
# Repositório git e CLI
# ---------------------------------------------------------------------------

TEX_BASE = "Introdução sem números.\n\nO método não foi validado em campo.\n"
TEX_NOVO = "Introdução sem números.\n\nO método foi validado em campo.\n"
TABELA = "% Gerada por scripts/foo.py; não editar manualmente.\n\\begin{tabular}{c}\n1\n\\end{tabular}\n"


def test_tabela_gerada_listada_a_parte(repo):
    base_commit(
        repo,
        {"text/latex/cap.tex": TEX_BASE, "text/latex/tables/tab_a.tex": TABELA},
    )
    escrever(repo, "text/latex/tables/tab_a.tex", TABELA.replace("1\n", "2\n"))
    escrever(repo, "text/latex/tese_rev_diff.tex", "ignorado 99\n")
    res = pf.analisar_repositorio(repo)
    assert res.gerados == [{"arquivo": "text/latex/tables/tab_a.tex", "estado": "alterada"}]
    assert res.achados == []
    assert "regenerar e conferir" in pf.renderizar_markdown(res)


def test_arquivos_padrao_incluem_novos_e_slides(repo):
    base_commit(repo, {"text/latex/cap.tex": TEX_BASE, "README.md": "x\n"})
    escrever(repo, "text/latex/cap.tex", TEX_NOVO)
    escrever(repo, "text/latex/novo.tex", "Capítulo novo com 12 casos.\n")
    escrever(repo, "text/slides/defesa.tex", "Slide com 5 itens.\n")
    escrever(repo, "README.md", "mudou\n")
    res = pf.analisar_repositorio(repo)
    assert sorted({a.arquivo for a in res.achados}) == [
        "text/latex/cap.tex",
        "text/latex/novo.tex",
        "text/slides/defesa.tex",
    ]


def test_alvo_como_ref(repo):
    base_commit(repo, {"text/latex/cap.tex": TEX_BASE})
    escrever(repo, "text/latex/cap.tex", TEX_NOVO)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "revisao")
    escrever(repo, "text/latex/cap.tex", TEX_BASE)  # worktree volta ao original
    res = pf.analisar_repositorio(repo, alvo="HEAD")
    assert len(res.pendentes) == 1
    assert pf.analisar_repositorio(repo).pendentes == []


def test_cli_codigos_de_saida(repo):
    base_commit(repo, {"text/latex/cap.tex": TEX_BASE})
    saida = repo / ".local" / "fidelidade" / "relatorio.md"

    assert pf.main(["--raiz", str(repo)]) == 0  # sem mudanças

    escrever(repo, "text/latex/cap.tex", TEX_NOVO)
    jason = repo / "rel.json"
    assert pf.main(["--raiz", str(repo), "--json", str(jason)]) == 1
    relatorio = saida.read_text(encoding="utf-8")
    assert "FALHA" in relatorio and "text/latex/cap.tex" in relatorio and "PENDENTE" in relatorio
    assert '"pendentes": 1' in jason.read_text(encoding="utf-8")

    achado = alertas(analisar(TEX_BASE, TEX_NOVO))[0]
    aceites = repo / "aceites.yaml"
    aceites.write_text(
        "aceites:\n"
        "  - arquivo: text/latex/cap.tex\n"
        f'    hash_paragrafo_novo: "{achado.hash}"\n'
        '    justificativa: "negador retirado de propósito"\n'
        "    autora_aprovou: true\n",
        encoding="utf-8",
    )
    assert pf.main(["--raiz", str(repo), "--aceites", str(aceites)]) == 0
    assert "ACEITO" in saida.read_text(encoding="utf-8")

    assert pf.main(["--raiz", str(repo), "--base", "tag-inexistente"]) == 2


def test_cli_como_subprocesso(repo):
    base_commit(repo, {"text/latex/cap.tex": TEX_BASE})
    escrever(repo, "text/latex/cap.tex", TEX_NOVO)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--raiz", str(repo), "--arquivos", "text/latex/cap.tex"],
        capture_output=True,
        creationflags=FLAGS,
    )
    assert proc.returncode == 1
    assert (repo / ".local" / "fidelidade" / "relatorio.md").exists()
