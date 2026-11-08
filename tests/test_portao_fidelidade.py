"""Portão de fidelidade: comparação de parágrafos entre base e alvo."""

import re
import shutil
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


def test_paragrafo_adicionado_com_negador_e_alerta():
    p1 = "Introdução geral do tema."
    p2 = "Esse efeito não foi observado em nenhum grupo."
    ach = alertas(analisar(f"{p1}\n", f"{p1}\n\n{p2}\n"))
    assert len(ach) == 1
    assert ach[0].tipo == "adicionado"
    assert ach[0].diferencas["negadores"]["adicionados"] == ["nenhum", "não"]
    assert "negadores:+não" in ach[0].itens


# ---------------------------------------------------------------------------
# Qualificadores, comparativos, métricas, sinal e ordem dos números
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,antes,depois",
    [
        ("no máximo", "pelo menos", "no máximo", "pelo menos"),
        ("mais de", "menos de", "mais de", "menos de"),
        ("cerca de", "ao menos", "cerca de", "ao menos"),
        ("aproximadamente", "até", "aproximadamente", "até"),
    ],
)
def test_qualificador_numerico_trocado(a, b, antes, depois):
    ach = alertas(analisar(f"Foram revisados {a} 50 casos no estudo.", f"Foram revisados {b} 50 casos no estudo."))
    assert len(ach) == 1
    assert ach[0].diferencas == {"qualificadores": {"removidos": [antes], "adicionados": [depois]}}


def test_qualificador_removido_e_espaco_latex():
    ach = alertas(analisar("Foram revisados mais de 50 casos.", "Foram revisados 50 casos."))
    assert ach[0].diferencas["qualificadores"] == {"removidos": ["mais de"], "adicionados": []}
    assert alertas(analisar("Foram revisados mais de 50 casos.", "Foram revisados mais~de 50 casos.")) == []
    # "até" dentro de outra palavra não conta
    assert pf.extrair_tokens("O atendimento ocorreu.")["qualificadores"] == {}


@pytest.mark.parametrize(
    "a,b",
    [("maior", "menor"), ("acima", "abaixo"), ("superior", "inferior")],
)
def test_comparativo_trocado(a, b):
    ach = alertas(analisar(f"O desempenho foi {a} que o do método anterior.", f"O desempenho foi {b} que o do método anterior."))
    assert len(ach) == 1
    assert ach[0].diferencas["comparativos"] == {"removidos": [a], "adicionados": [b]}


def test_comparativo_plural_nao_alerta():
    assert alertas(analisar("Houve valores maiores no grupo.", "Houve valores maior no grupo.")) == []


def test_metricas_e_modos_trocados():
    ach = alertas(
        analisar(
            "A sensibilidade foi alta no modo vigilância.",
            "A especificidade foi alta no modo confirmação.",
        )
    )
    assert len(ach) == 1
    d = ach[0].diferencas["metricas"]
    assert d["removidos"] == ["sensibilidade", "vigilância"]
    assert d["adicionados"] == ["confirmação", "especificidade"]
    vpp = alertas(analisar("O VPP foi alto em todos os casos.", "O VPN foi alto em todos os casos."))
    assert vpp[0].diferencas["metricas"] == {"removidos": ["VPP"], "adicionados": ["VPN"]}
    f1 = pf.extrair_tokens("O F1 e a sensibilidade subiram.")
    assert f1["metricas"] == {"F1": 1, "sensibilidade": 1}


@pytest.mark.parametrize(
    "escrita",
    ["-5,6485", "\u22125,6485", "$-$5,6485", "$-5{,}6485$", "($-$5{,}6485)"],
)
def test_sinal_negativo_e_preservado(escrita):
    tok = pf.extrair_tokens(f"A diferença foi de {escrita} pontos percentuais.")
    assert tok["numeros"] == {"-5.6485": 1}


def test_perda_do_sinal_negativo_alerta():
    ach = alertas(
        analisar(
            "A diferença foi de \u22125,6485 pontos percentuais.",
            "A diferença foi de 5,6485 pontos percentuais.",
        )
    )
    assert len(ach) == 1
    assert ach[0].diferencas["numeros"] == {"removidos": ["-5.6485"], "adicionados": ["5.6485"]}
    # grafias equivalentes do sinal não alertam
    assert alertas(analisar("Foi de -5,6485 pontos.", "Foi de $-$5{,}6485 pontos.")) == []
    assert alertas(analisar("Foi de -5,6485 pontos.", "Foi de \u22125,6485 pontos.")) == []


def test_hifen_entre_numeros_e_palavras_nao_e_sinal():
    tok = pf.extrair_tokens("No período 2020-2021 e 10--15 casos, pré-1999 e covid-19.")
    assert tok["numeros"] == {"2020": 1, "2021": 1, "10": 1, "15": 1, "1999": 1, "19": 1}


def test_troca_de_ordem_de_numeros_na_frase_alerta():
    antigo = "O grupo A teve 12 casos e o grupo B teve 30 casos."
    novo = "O grupo A teve 30 casos e o grupo B teve 12 casos."
    ach = alertas(analisar(antigo, novo))
    assert len(ach) == 1
    # o multiconjunto de números é igual: só a ordem acusa a troca
    assert set(ach[0].diferencas) == {"ordem_numeros"}
    assert ach[0].diferencas["ordem_numeros"] == {"removidos": ["12|30"], "adicionados": ["30|12"]}
    assert ach[0].itens == ["ordem_numeros:+30|12", "ordem_numeros:-12|30"]


def test_ordem_de_numeros_sem_troca_nao_alerta():
    antigo = "O grupo A teve 12 casos e o grupo B teve 30 casos. Houve 7 perdas."
    # mesma ordem em cada frase, frases em outra ordem
    novo = "Houve 7 perdas. O grupo A teve 12 casos e o grupo B teve 30 casos."
    assert alertas(analisar(antigo, novo)) == []
    # números em frases distintas trocados de frase não são "ordem dentro da frase"
    assert pf.extrair_tokens("A teve 12 casos. B teve 30 casos.")["ordem_numeros"] == {}


def test_numero_diferente_nao_gera_alerta_de_ordem():
    ach = alertas(
        analisar(
            "O grupo A teve 12 casos e o grupo B teve 30 casos.",
            "O grupo A teve 12 casos e o grupo B teve 31 casos.",
        )
    )
    assert set(ach[0].diferencas) == {"numeros"}


# ---------------------------------------------------------------------------
# Aceites
# ---------------------------------------------------------------------------


def aceite_para(achado, **extra):
    base = {
        "arquivo": achado.arquivo,
        "hash_paragrafo_novo": achado.hash,
        "itens": list(achado.itens),
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


ACEITES_REPO = ROOT / "docs" / "fidelidade_aceites.yaml"


def test_aceites_do_repositorio_seguem_o_esquema():
    entradas = pf.ler_entradas_aceites(ACEITES_REPO)
    for e in entradas:
        assert pf.validar_aceite(e) == [], e
        assert set(e) <= {"arquivo", "hash_paragrafo_novo", "itens", "justificativa", "autora_aprovou"}
    # hash sempre entre aspas no arquivo (hash só com dígitos viraria número no YAML)
    if ACEITES_REPO.exists():
        linhas = [
            ln.split(":", 1)[1].strip()
            for ln in ACEITES_REPO.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("hash_paragrafo_novo:")
        ]
        assert len(linhas) == len(entradas)
        assert all(re.fullmatch(r"""(["'])[0-9a-f]{16}\1\s*(#.*)?""", v) for v in linhas), linhas
    # o carregador aceita tudo o que o esquema aceita
    assert len(pf.carregar_aceites(ACEITES_REPO)) == len(entradas)


def aceite_valido(**extra):
    base = {
        "arquivo": "text/latex/x.tex",
        "hash_paragrafo_novo": "0123456789abcdef",
        "itens": ["numeros:-39.3"],
        "justificativa": "ok",
        "autora_aprovou": True,
    }
    base.update(extra)
    return base


def test_validar_aceite_aceita_entrada_completa():
    assert pf.validar_aceite(aceite_valido()) == []


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("arquivo", ""),
        ("arquivo", None),
        ("hash_paragrafo_novo", 1234567890123456),  # hash só com dígitos sem aspas
        ("hash_paragrafo_novo", 1.2345e20),
        ("hash_paragrafo_novo", "0123456789ABCDEF"),
        ("hash_paragrafo_novo", "0123456789abcde"),
        ("itens", []),
        ("itens", None),
        ("itens", "numeros:-1"),
        ("itens", [""]),
        ("justificativa", ""),
        ("justificativa", None),
        ("autora_aprovou", "true"),
        ("autora_aprovou", None),
    ],
)
def test_validar_aceite_rejeita_campos_invalidos(campo, valor):
    assert pf.validar_aceite(aceite_valido(**{campo: valor}))


def test_validar_aceite_rejeita_campo_ausente_e_nao_mapeamento():
    for campo in ("arquivo", "hash_paragrafo_novo", "itens", "justificativa", "autora_aprovou"):
        e = aceite_valido()
        del e[campo]
        assert pf.validar_aceite(e), campo
    assert pf.validar_aceite("texto")


def test_carregar_aceites_normaliza(tmp_path):
    arq = tmp_path / "a.yaml"
    arq.write_text(
        "aceites:\n"
        "  - arquivo: text/latex/x.tex\n"
        '    hash_paragrafo_novo: "0123456789abcdef"\n'
        "    itens:\n"
        "      - numeros:-39.3\n"
        "    justificativa: ok\n"
        "    autora_aprovou: true\n",
        encoding="utf-8",
    )
    ent = pf.carregar_aceites(arq)
    assert ent == [
        {
            "arquivo": "text/latex/x.tex",
            "hash_paragrafo_novo": "0123456789abcdef",
            "itens": ["numeros:-39.3"],
            "justificativa": "ok",
            "autora_aprovou": True,
        }
    ]


def test_carregar_aceites_rejeita_itens_vazios(tmp_path):
    for itens in ("", "    itens: []\n"):
        arq = tmp_path / "b.yaml"
        arq.write_text(
            "aceites:\n"
            "  - arquivo: text/latex/x.tex\n"
            '    hash_paragrafo_novo: "0123456789abcdef"\n'
            + itens
            + "    justificativa: ok\n"
            "    autora_aprovou: true\n",
            encoding="utf-8",
        )
        with pytest.raises(pf.ErroAceites, match="sem itens"):
            pf.carregar_aceites(arq)
    assert pf.main(["--raiz", str(tmp_path), "--aceites", str(arq)]) == 2


def test_aceite_com_itens_vazios_nao_cobre_nada():
    antigo = "O método não foi validado em campo."
    novo = "O método foi validado em campo."
    ach = alertas(analisar(antigo, novo))[0]
    res = analisar(antigo, novo, [aceite_para(ach, itens=[])])
    assert alertas(res)[0].status == "pendente"
    assert "sem itens" in alertas(res)[0].nota_aceite


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
    assert [(g["arquivo"], g["estado"], g["status"]) for g in res.gerados] == [
        ("text/latex/tables/tab_a.tex", "alterada", "pendente")
    ]
    assert res.achados == []
    assert "regenerar e conferir" in pf.renderizar_markdown(res)


def test_tabela_gerada_alterada_e_pendente_e_falha_o_portao(repo):
    base_commit(repo, {"text/latex/tables/tab_a.tex": TABELA})
    escrever(repo, "text/latex/tables/tab_a.tex", TABELA.replace("1\n", "2\n"))
    res = pf.analisar_repositorio(repo)
    assert res.pendentes == []
    assert res.n_pendentes == 1
    md = pf.renderizar_markdown(res)
    assert "FALHA" in md and "PENDENTE" in md
    assert pf.main(["--raiz", str(repo)]) == 1


def test_tabela_gerada_regenerada_identica_nao_e_pendente(repo):
    base_commit(repo, {"text/latex/tables/tab_a.tex": TABELA})
    escrever(repo, "text/latex/tables/tab_a.tex", TABELA.replace("1\n", "2\n"))
    escrever(repo, "text/latex/tables/tab_a.tex", TABELA)  # regenerada: volta idêntica
    res = pf.analisar_repositorio(repo)
    assert res.gerados == []
    assert res.n_pendentes == 0
    assert pf.main(["--raiz", str(repo)]) == 0


def test_tabela_gerada_so_com_fim_de_linha_diferente_nao_e_pendente(repo):
    base_commit(repo, {"text/latex/tables/tab_a.tex": TABELA})
    p = repo / "text/latex/tables/tab_a.tex"
    p.write_bytes(TABELA.replace("\n", "\r\n").encode("utf-8"))
    res = pf.analisar_repositorio(repo)
    assert res.gerados == [] and res.n_pendentes == 0


def test_tabela_gerada_nova_ou_removida_e_pendente(repo):
    base_commit(repo, {"text/latex/tables/tab_a.tex": TABELA})
    (repo / "text/latex/tables/tab_a.tex").unlink()
    escrever(repo, "text/latex/tables/tab_b.tex", TABELA)
    res = pf.analisar_repositorio(repo)
    assert sorted((g["arquivo"][-10:], g["estado"]) for g in res.gerados) == [
        ("/tab_a.tex", "removida"),
        ("/tab_b.tex", "nova"),
    ]
    assert res.n_pendentes == 2


def test_tabela_gerada_aceita_com_item_e_aprovacao(repo):
    base_commit(repo, {"text/latex/tables/tab_a.tex": TABELA})
    escrever(repo, "text/latex/tables/tab_a.tex", TABELA.replace("1\n", "2\n"))
    h = pf.analisar_repositorio(repo).gerados[0]["hash"]
    arq = "text/latex/tables/tab_a.tex"

    def aceite(**extra):
        return aceite_valido(
            arquivo=arq, hash_paragrafo_novo=h, itens=["tabela_gerada:alterada"], **extra
        )

    ok = pf.analisar_repositorio(repo, aceites=[aceite()])
    assert ok.gerados[0]["status"] == "aceito" and ok.n_pendentes == 0
    assert ok.orfaos == []
    sem = pf.analisar_repositorio(repo, aceites=[aceite(autora_aprovou=False)])
    assert sem.n_pendentes == 1
    outro = pf.analisar_repositorio(
        repo, aceites=[aceite_valido(arquivo=arq, hash_paragrafo_novo=h, itens=["numeros:-1"])]
    )
    assert outro.n_pendentes == 1


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
        "    itens:\n"
        "      - negadores:-não\n"
        '    justificativa: "negador retirado de propósito"\n'
        "    autora_aprovou: true\n",
        encoding="utf-8",
    )
    assert pf.main(["--raiz", str(repo), "--aceites", str(aceites)]) == 0
    assert "ACEITO" in saida.read_text(encoding="utf-8")

    assert pf.main(["--raiz", str(repo), "--base", "tag-inexistente"]) == 2


def test_portao_no_repositorio_real(tmp_path):
    """O portão roda contra o repositório de verdade: base pre-revisao-2, alvo HEAD."""
    if shutil.which("git") is None:
        pytest.skip("git indisponível")
    try:
        raiz = pf.raiz_git(ROOT)
        base_ok = pf.ref_valida(raiz, "pre-revisao-2") and pf.ref_valida(raiz, "HEAD")
    except (pf.ErroGit, OSError):
        pytest.skip("repositório git indisponível")
    if not base_ok:
        pytest.skip("tag pre-revisao-2 ausente (clone raso ou sem tags)")
    saida = tmp_path / "relatorio.md"
    codigo = pf.main(
        [
            "--raiz", str(raiz),
            "--base", "pre-revisao-2",
            "--alvo", "HEAD",
            "--saida", str(saida),
        ]
    )  # fmt: skip
    relatorio = saida.read_text(encoding="utf-8") if saida.exists() else "(sem relatório)"
    assert codigo == 0, relatorio[:6000]


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
