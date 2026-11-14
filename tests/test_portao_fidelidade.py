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
    assert alertas(analisar(antigo, antigo)) == []
    # mesma ordem em cada frase, frases em outra ordem: a ordem por frase não alerta
    # (a troca de frases inteiras só aparece como ordem do parágrafo, ver teste abaixo)
    novo = "Houve 7 perdas. O grupo A teve 12 casos e o grupo B teve 30 casos."
    ach = alertas(analisar(antigo, novo))
    assert all(set(a.diferencas) == {"ordem_paragrafo"} for a in ach)
    # números em frases distintas trocados de frase não são "ordem dentro da frase"
    assert pf.extrair_tokens("A teve 12 casos. B teve 30 casos.")["ordem_numeros"] == {}


@pytest.mark.parametrize("sep", ["; ", ". "])
def test_troca_de_numeros_entre_oracoes_alerta(sep):
    antigo = f"A revisão aceitou 215 dos 247 vínculos{sep}na confirmação, 219 dos 247."
    novo = f"A revisão aceitou 219 dos 247 vínculos{sep}na confirmação, 215 dos 247."
    ach = alertas(analisar(antigo, novo))
    assert len(ach) == 1
    # multiconjuntos iguais, inclusive por frase: só a ordem do parágrafo acusa a troca
    assert set(ach[0].diferencas) == {"ordem_paragrafo"}
    assert ach[0].diferencas["ordem_paragrafo"] == {
        "removidos": ["215|247|219|247"],
        "adicionados": ["219|247|215|247"],
    }
    assert ach[0].itens == ["ordem_paragrafo:+219|247|215|247", "ordem_paragrafo:-215|247|219|247"]
    # exige aceite como as demais categorias de sentido
    assert ach[0].status == "pendente"
    aceito = alertas(analisar(antigo, novo, [aceite_para(ach[0])]))[0]
    assert aceito.status == "aceito"
    outro = alertas(analisar(antigo, novo, [aceite_para(ach[0], itens=["numeros:-215"])]))[0]
    assert outro.status == "pendente"


def test_ordem_do_paragrafo_sem_mudanca_nao_alerta():
    texto = "A revisão aceitou 215 dos 247 vínculos; na confirmação, 219 dos 247."
    assert alertas(analisar(texto, texto + " Nada mais.")) == []
    assert alertas(analisar(texto, texto.replace("A revisão", "A equipe de revisão"))) == []
    assert pf.extrair_tokens(texto)["ordem_paragrafo"] == {"215|247|219|247": 1}
    # um único número não define ordem
    assert pf.extrair_tokens("Houve 7 perdas.")["ordem_paragrafo"] == {}


def test_insercao_ou_remocao_de_numero_continua_em_numeros():
    antigo = "A revisão aceitou 215 dos 247 vínculos; na confirmação, 219 dos 247."
    inserido = "A revisão aceitou 215 dos 247 vínculos; na confirmação, 219 dos 247 e 12 perdas."
    ach = alertas(analisar(antigo, inserido))
    assert len(ach) == 1
    assert set(ach[0].diferencas) == {"numeros"}
    assert ach[0].diferencas["numeros"] == {"removidos": [], "adicionados": ["12"]}
    ach = alertas(analisar(inserido, antigo))
    assert set(ach[0].diferencas) == {"numeros"}
    assert ach[0].diferencas["numeros"] == {"removidos": ["12"], "adicionados": []}


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


@pytest.mark.deck_qualificacao
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


def test_portao_no_repositorio_real_por_capitulo(tmp_path):
    """Mesma comparação, só por arquivo, com o registro de invariantes real."""
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
            "--modo", "capitulo",
            "--saida", str(saida),
        ]
    )  # fmt: skip
    relatorio = saida.read_text(encoding="utf-8") if saida.exists() else "(sem relatório)"
    assert codigo == 0, relatorio[:6000]
    assert "Números por capítulo" in relatorio


def test_registro_real_cobre_numero_central_do_capitulo_de_dados():
    inv = pf.carregar_invariantes(ROOT / pf.REGISTRO_PADRAO)
    assert "N01" in inv.cobertura["text/latex/capitulo_dados.tex"]["61696"]
    assert "N01" in inv.exigidos["text/latex/capitulo_dados.tex"]["61696"]


# ---------------------------------------------------------------------------
# Números por capítulo (arquivo)
# ---------------------------------------------------------------------------

ARQ = "text/latex/x.tex"
REGISTRO_YAML = """\
- id: N05
  descricao: "teste"
  tipo: numero
  valores:
    - {nome: n_total, fonte: {arquivo: data/x.csv, filtro: {m: a}, coluna: v}, texto_pt: "61.696"}
    - {nome: n_vinc, fonte: {arquivo: data/x.csv, filtro: {m: b}, coluna: v}, texto_pt: "247"}
  ocorrencias:
    - {arquivo: "@ARQ@", exige: [n_total, n_vinc]}
- id: N06
  descricao: "outro arquivo"
  tipo: numero
  valores:
    - {nome: n_outro, fonte: {arquivo: data/x.csv, filtro: {m: c}, coluna: v}, texto_pt: "99"}
  ocorrencias:
    - {arquivo: text/latex/y.tex, exige: [n_outro]}
"""


def numeros(antigo, novo, arquivo=ARQ, **kw):
    return pf.analisar_numeros_arquivo(arquivo, antigo, novo, **kw)


def pendentes_numeros(achados):
    return [a for a in achados if a.alerta and a.status == "pendente"]


@pytest.fixture
def invariantes(tmp_path):
    reg = tmp_path / "registro.yaml"
    reg.write_text(REGISTRO_YAML.replace("@ARQ@", ARQ), encoding="utf-8")
    return pf.carregar_invariantes(reg)


def test_numero_novo_bloqueia():
    ach = numeros("Havia 10 casos.", "Havia 10 casos e 77 perdas.")
    assert [(a.numero, a.tipo, a.status) for a in ach] == [("77", "novo", "pendente")]
    assert pendentes_numeros(ach) == ach


def test_numero_novo_coberto_por_invariante_passa(invariantes):
    ach = numeros("Havia 10 casos.", "Havia 10 casos em 61.696 pares.", invariantes=invariantes)
    assert [(a.numero, a.status, a.invariantes) for a in ach] == [("61696", "coberto", ["N05"])]
    assert pendentes_numeros(ach) == []
    # invariante de outro arquivo não cobre este
    outro = numeros(
        "Havia 10 casos.", "Havia 10 casos em 61.696 pares.", arquivo="text/latex/z.tex", invariantes=invariantes
    )
    assert len(pendentes_numeros(outro)) == 1
    outro = numeros("Havia 10 casos.", "Havia 10 casos e 99 perdas.", invariantes=invariantes)
    assert len(pendentes_numeros(outro)) == 1  # o 99 é invariante de y.tex, não de x.tex


def test_contagem_aumentada_bloqueia():
    ach = numeros("Foram 12 casos.", "Foram 12 casos. Mais 12 depois.")
    assert [(a.numero, a.tipo, a.contagem_base, a.contagem_alvo, a.status) for a in ach] == [
        ("12", "aumentado", 1, 2, "pendente")
    ]


def test_remocao_e_registrada_sem_bloquear():
    ach = numeros("Foram 12 casos e 7 perdas.", "Foram 12 casos.")
    assert [(a.numero, a.tipo, a.status, a.alerta) for a in ach] == [("7", "removido", "info", False)]
    assert pendentes_numeros(ach) == []
    # contagem menor que ainda deixa o número no arquivo também só é registrada
    menor = numeros("Foram 12 casos. Outros 12.", "Foram 12 casos.")
    assert [(a.tipo, a.alerta) for a in menor] == [("removido", False)]


def test_remocao_de_valor_de_invariante_bloqueia(invariantes):
    ach = numeros("Foram 247 vínculos e 7 perdas.", "Foram 7 perdas.", invariantes=invariantes)
    pend = pendentes_numeros(ach)
    assert [(a.numero, a.tipo, a.invariantes) for a in pend] == [("247", "removido_invariante", ["N05"])]
    # exigido só em x.tex: em y.tex a remoção é apenas registrada
    y = numeros("Foram 247 vínculos e 7 perdas.", "Foram 7 perdas.", arquivo="text/latex/y.tex", invariantes=invariantes)
    assert pendentes_numeros(y) == []
    # o valor continua no arquivo (outra ocorrência): não bloqueia
    resta = numeros("Foram 247 vínculos. De novo 247.", "Foram 247 vínculos.", invariantes=invariantes)
    assert pendentes_numeros(resta) == []


def test_numero_movido_nao_gera_nada():
    assert numeros("A teve 12 casos.\n\nB teve 30 casos.\n", "B teve 30 casos.\n\nA teve 12 casos.\n") == []
    assert numeros("Dados: 12 e 30.\n\nOutro.\n", "Dados.\n\nOutro com 12 e 30.\n") == []


def test_extracao_de_numeros_do_capitulo_usa_a_do_paragrafo():
    total = pf.contar_numeros("Taxa de 39,3\\% em 84.368 pares.\n\nOutra de 39.3\\% % comentário 77\n")
    assert total == {"39.3%": 2, "84368": 1}


def test_aceite_por_arquivo_unitario():
    ace = [{"arquivo": ARQ, "numero": "77", "autora_aprovou": True, "motivo": "conferido"}]
    usados = set()
    ach = numeros("Havia 10.", "Havia 10 e 77.", aceites_numeros=ace, usados=usados)
    assert [(a.status, a.alerta, a.nota) for a in ach] == [("aceito", False, "conferido")]
    assert usados == {(ARQ, "77")}
    nao = numeros("Havia 10.", "Havia 10 e 77.", aceites_numeros=[dict(ace[0], autora_aprovou=False)])
    assert [(a.status, a.alerta) for a in nao] == [("pendente", True)]
    outro = numeros("Havia 10.", "Havia 10 e 77.", aceites_numeros=[dict(ace[0], arquivo="text/latex/z.tex")])
    assert len(pendentes_numeros(outro)) == 1


def test_parser_simples_separa_secoes_de_aceites():
    texto = (
        "aceites:\n"
        "  - arquivo: a.tex\n"
        '    hash_paragrafo_novo: "0123456789abcdef"\n'
        "    itens: [numeros:-1]\n"
        "    autora_aprovou: true\n"
        "aceites_numeros:\n"
        "  - arquivo: b.tex\n"
        '    numero: "77"\n'
        "    autora_aprovou: true\n"
        "    motivo: ok\n"
    )
    par = pf.parse_aceites_simples(texto)
    assert [e["arquivo"] for e in par] == ["a.tex"]
    num = pf.parse_aceites_simples(texto, pf.SECAO_ACEITES_NUMEROS)
    assert num == [{"arquivo": "b.tex", "numero": "77", "autora_aprovou": True, "motivo": "ok"}]
    assert pf.parse_aceites_simples("aceites: []\n\naceites_numeros: []\n") == []


def test_carregar_aceites_numeros_normaliza_e_valida(tmp_path):
    arq = tmp_path / "a.yaml"
    arq.write_text(
        "aceites: []\n\naceites_numeros:\n"
        "  - arquivo: text\\latex\\x.tex\n"
        '    numero: "39,3%"\n'
        "    autora_aprovou: true\n"
        "    motivo: ok\n",
        encoding="utf-8",
    )
    assert pf.carregar_aceites_numeros(arq) == [
        {"arquivo": "text/latex/x.tex", "numero": "39.3%", "autora_aprovou": True, "motivo": "ok"}
    ]
    assert pf.carregar_aceites(arq) == []
    arq.write_text("aceites_numeros:\n  - arquivo: x.tex\n    autora_aprovou: true\n", encoding="utf-8")
    with pytest.raises(pf.ErroAceites):
        pf.carregar_aceites_numeros(arq)
    assert pf.main(["--raiz", str(tmp_path), "--aceites", str(arq)]) == 2


def test_fusao_de_paragrafos_com_mesmos_numeros_passa_em_capitulo(repo):
    base_commit(repo, {"text/latex/cap.tex": "A teve 12 casos.\n\nB teve 30 casos.\n\nC fecha.\n"})
    escrever(repo, "text/latex/cap.tex", "A teve 12 casos e B teve 30 casos.\n\nC fecha.\n")
    cap = pf.analisar_repositorio(repo, modo="capitulo")
    assert cap.n_pendentes == 0 and cap.achados == [] and cap.numeros == []
    # o modo por parágrafo continua acusando a fusão (e não olha números por arquivo)
    par = pf.analisar_repositorio(repo, modo="paragrafo")
    assert par.n_pendentes > 0 and par.numeros == []
    assert pf.analisar_repositorio(repo).n_pendentes > 0  # padrão: ambos
    assert pf.main(["--raiz", str(repo), "--modo", "capitulo"]) == 0
    assert pf.main(["--raiz", str(repo), "--modo", "paragrafo"]) == 1
    assert pf.main(["--raiz", str(repo)]) == 1


def test_capitulo_bloqueia_numero_novo_no_repositorio(repo):
    base_commit(repo, {"text/latex/cap.tex": "Havia 10 casos.\n"})
    escrever(repo, "text/latex/cap.tex", "Havia 10 casos e 77 perdas.\n")
    res = pf.analisar_repositorio(repo, modo="capitulo")
    assert res.n_pendentes == 1 and res.achados == []
    md = pf.renderizar_markdown(res)
    assert "FALHA" in md and "Números por capítulo" in md and "`77`" in md and "PENDENTE" in md
    assert pf.main(["--raiz", str(repo), "--modo", "capitulo"]) == 1


def test_capitulo_registra_remocao_no_relatorio(repo):
    base_commit(repo, {"text/latex/cap.tex": "Havia 10 casos e 7 perdas.\n"})
    escrever(repo, "text/latex/cap.tex", "Havia 10 casos.\n")
    res = pf.analisar_repositorio(repo, modo="capitulo")
    assert res.n_pendentes == 0
    md = pf.renderizar_markdown(res)
    assert "OK" in md and "REGISTRADO" in md and "`7` removido" in md


def test_capitulo_usa_registro_do_repositorio(repo):
    base_commit(repo, {"text/latex/cap.tex": "Vínculos 247 e perdas 7.\n"})
    escrever(repo, "tests/data/registro_invariantes.yaml", REGISTRO_YAML.replace("@ARQ@", "text/latex/cap.tex"))
    escrever(repo, "text/latex/cap.tex", "Perdas 7 em 61.696 pares.\n")
    res = pf.analisar_repositorio(repo, modo="capitulo")
    # 61.696 é coberto por N05; 247 sumiu e N05 o exige em cap.tex
    assert [(n.numero, n.tipo, n.status) for n in res.numeros] == [
        ("247", "removido_invariante", "pendente"),
        ("61696", "novo", "coberto"),
    ]
    assert res.n_pendentes == 1
    ace = [{"arquivo": "text/latex/cap.tex", "numero": "247", "autora_aprovou": True, "motivo": "cortado de propósito"}]
    assert pf.analisar_repositorio(repo, modo="capitulo", aceites_numeros=ace).n_pendentes == 0


def test_aceite_por_arquivo_libera(repo):
    base_commit(repo, {"text/latex/cap.tex": "Havia 10 casos.\n"})
    escrever(repo, "text/latex/cap.tex", "Havia 10 casos e 77 perdas.\n")
    ace = {"arquivo": "text/latex/cap.tex", "numero": "77", "autora_aprovou": True, "motivo": "conferido"}
    res = pf.analisar_repositorio(repo, modo="capitulo", aceites_numeros=[ace])
    assert res.n_pendentes == 0 and res.n_aceitos == 1 and res.orfaos_numeros == []
    assert "ACEITO" in pf.renderizar_markdown(res)
    sem = pf.analisar_repositorio(repo, modo="capitulo", aceites_numeros=[dict(ace, autora_aprovou=False)])
    assert sem.n_pendentes == 1
    outro = pf.analisar_repositorio(
        repo, modo="capitulo", aceites_numeros=[dict(ace, arquivo="text/latex/outro.tex")]
    )
    assert outro.n_pendentes == 1 and len(outro.orfaos_numeros) == 1

    aceites = repo / "aceites.yaml"
    aceites.write_text(
        "aceites: []\n\naceites_numeros:\n"
        "  - arquivo: text/latex/cap.tex\n"
        '    numero: "77"\n'
        "    autora_aprovou: true\n"
        '    motivo: "conferido"\n',
        encoding="utf-8",
    )
    assert pf.main(["--raiz", str(repo), "--modo", "capitulo", "--aceites", str(aceites)]) == 0
    assert pf.main(["--raiz", str(repo), "--modo", "capitulo"]) == 1


# ---------------------------------------------------------------------------
# Modo auto, valor referenciado por nome e rótulos isentos
# ---------------------------------------------------------------------------

FUNDIDO_BASE = "A teve 12 casos.\n\nB teve 30 casos.\n\nC fecha.\n"
FUNDIDO_NOVO = "A teve 12 casos e B teve 30 casos.\n\nC fecha.\n"
REVISADOS_YAML = "# revisados\n- text/latex/cap.tex\n"


def _repo_fundido(repo, alterar=("cap", "outro")):
    base_commit(
        repo, {"text/latex/cap.tex": FUNDIDO_BASE, "text/latex/outro.tex": FUNDIDO_BASE}
    )
    escrever(repo, "tests/data/arquivos_revisados.yaml", REVISADOS_YAML)
    for nome in alterar:
        escrever(repo, f"text/latex/{nome}.tex", FUNDIDO_NOVO)


def test_modo_padrao_e_auto():
    assert pf.MODO_PADRAO == "auto" and "auto" in pf.MODOS
    assert {"capitulo", "paragrafo", "ambos"} <= set(pf.MODOS)


def test_auto_aplica_capitulo_a_revisado_e_ambos_a_nao_revisado(repo):
    _repo_fundido(repo)
    res = pf.analisar_repositorio(repo)  # padrão: auto
    assert res.modo == "auto"
    assert res.modos_arquivo == {"text/latex/cap.tex": "capitulo", "text/latex/outro.tex": "ambos"}
    # o revisado não passa pelo alinhamento por parágrafo; o não revisado, sim
    assert {a.arquivo for a in res.achados} == {"text/latex/outro.tex"}
    assert set(res.contagens) == {"text/latex/outro.tex"}
    assert {p.arquivo for p in res.pendentes} == {"text/latex/outro.tex"}
    assert res.n_pendentes > 0
    md = pf.renderizar_markdown(res)
    assert "Modo: `auto`" in md and "`text/latex/cap.tex`" in md.split("## ")[0]
    # o não revisado também passa pela comparação por capítulo (aqui sem diferença de números)
    assert res.numeros == []


def test_auto_so_com_revisado_fundido_passa(repo):
    _repo_fundido(repo, alterar=("cap",))
    assert pf.analisar_repositorio(repo).n_pendentes == 0
    assert pf.main(["--raiz", str(repo)]) == 0
    # um número novo no revisado continua bloqueando, por capítulo
    escrever(repo, "text/latex/cap.tex", FUNDIDO_NOVO + "\nMais 77 perdas.\n")
    res = pf.analisar_repositorio(repo)
    assert res.achados == [] and [(n.numero, n.status) for n in res.numeros] == [("77", "pendente")]
    assert pf.main(["--raiz", str(repo)]) == 1


def test_auto_sem_lista_de_revisados_equivale_a_ambos(repo):
    base_commit(repo, {"text/latex/cap.tex": FUNDIDO_BASE})
    escrever(repo, "text/latex/cap.tex", FUNDIDO_NOVO)
    res = pf.analisar_repositorio(repo)
    assert res.modos_arquivo == {"text/latex/cap.tex": "ambos"}
    assert res.n_pendentes == pf.analisar_repositorio(repo, modo="ambos").n_pendentes > 0


def test_chaves_de_secao_nao_mudam_o_modo_do_arquivo(repo):
    _repo_fundido(repo, alterar=("outro",))
    escrever(repo, "tests/data/arquivos_revisados.yaml", "- text/latex/outro.tex#resumo\n")
    res = pf.analisar_repositorio(repo)
    assert res.modos_arquivo == {"text/latex/outro.tex": "ambos"}


def test_modos_explicitos_ignoram_a_lista_de_revisados(repo):
    _repo_fundido(repo)
    par = pf.analisar_repositorio(repo, modo="paragrafo")
    assert {a.arquivo for a in par.achados} == {"text/latex/cap.tex", "text/latex/outro.tex"}
    assert par.numeros == [] and par.n_pendentes > 0
    assert set(par.modos_arquivo.values()) == {"paragrafo"}
    amb = pf.analisar_repositorio(repo, modo="ambos")
    assert {a.arquivo for a in amb.achados} == {"text/latex/cap.tex", "text/latex/outro.tex"}
    cap = pf.analisar_repositorio(repo, modo="capitulo")
    assert cap.achados == [] and cap.n_pendentes == 0
    assert pf.main(["--raiz", str(repo), "--modo", "paragrafo"]) == 1
    assert pf.main(["--raiz", str(repo), "--modo", "capitulo"]) == 0


def test_aceite_de_paragrafo_em_arquivo_so_capitulo_nao_e_orfao(repo):
    _repo_fundido(repo, alterar=("cap",))
    ace = [aceite_valido(arquivo="text/latex/cap.tex")]
    assert pf.analisar_repositorio(repo, aceites=ace).orfaos == []
    assert len(pf.analisar_repositorio(repo, modo="ambos", aceites=ace).orfaos) == 1


REGISTRO_POR_NOME = """\
- id: N01
  descricao: "define o valor"
  tipo: numero
  valores:
    - {nome: n_total, fonte: {arquivo: data/x.csv, filtro: {m: a}, coluna: v}, texto_pt: "61.696"}
  ocorrencias:
    - {arquivo: text/latex/y.tex, exige: [n_total]}
- id: N09
  descricao: "só referencia o valor de N01 por nome"
  tipo: numero
  valores:
    - {nome: abaixo, fonte: {arquivo: data/x.csv, filtro: {m: b}, coluna: v}, texto_pt: "61.485"}
  ocorrencias:
    - {arquivo: "@ARQ@", exige: [abaixo, n_total], mesma_frase: ["grey area"]}
"""


def test_valor_referenciado_por_nome_em_outra_entrada_conta_como_coberto(tmp_path):
    reg = tmp_path / "registro.yaml"
    reg.write_text(REGISTRO_POR_NOME.replace("@ARQ@", ARQ), encoding="utf-8")
    inv = pf.carregar_invariantes(reg)
    assert inv.cobertura[ARQ]["61485"] == ["N09"]
    assert inv.cobertura[ARQ]["61696"] == ["N09"]  # definido em N01, exigido por N09
    assert "61696" not in inv.cobertura.get("text/latex/z.tex", {})
    ach = numeros("Havia 10 casos.", "Havia 10 casos em 61.696 pares.", invariantes=inv)
    assert [(a.numero, a.status, a.invariantes) for a in ach] == [("61696", "coberto", ["N09"])]
    assert pendentes_numeros(ach) == []
    # arquivo que não cita a ocorrência não é coberto
    outro = numeros(
        "Havia 10 casos.", "Havia 10 casos em 61.696 pares.", arquivo="text/latex/z.tex", invariantes=inv
    )
    assert len(pendentes_numeros(outro)) == 1


def test_registro_real_cobre_valor_por_nome_do_capitulo3():
    inv = pf.carregar_invariantes(ROOT / pf.REGISTRO_PADRAO)
    assert "N09" in inv.cobertura["text/latex/capitulo3.tex"]["61696"]


@pytest.mark.parametrize(
    "rotulo",
    [
      "no passo 11 da blocagem",
      "no Passo~11",
      "nos passos 2 e 11",
      "na Tabela~3",
      "no Capítulo 4",
      "na Seção 2.1",
      "na Figura 5",
      "na Equação 7",
      "no Estudo 2",
      "no objetivo 3",
      "na meta 4",
    ],
)
def test_rotulos_sao_isentos_na_comparacao_por_capitulo(rotulo):
    assert numeros("Havia 10 casos.", f"Havia 10 casos {rotulo}.") == []
    # no alinhamento por parágrafo o número continua contando
    assert sum(pf.extrair_tokens(f"Havia 10 casos {rotulo}.")["numeros"].values()) >= 2


def test_numero_depois_do_rotulo_e_numero_antes_da_palavra_continuam_contando():
    ach = numeros("Havia 10 casos.", "Havia 10 casos no passo 11 com 77 perdas.")
    assert [(a.numero, a.tipo) for a in ach] == [("77", "novo")]
    ach = numeros("Havia 10 casos.", "Havia 10 casos em 11 passos.")
    assert [(a.numero, a.tipo) for a in ach] == [("11", "novo")]
    # rótulo na base e no alvo: nada a comparar
    assert numeros("Ver o passo 11 e depois 5 casos.", "Ver 5 casos no passo 11.") == []


@pytest.mark.parametrize(
    "identificador",
    ["a ordenação O3", "a fusão F4", "a hipótese H1", "o Kimi K2.5", "a COVID-19", "o modelo F3 e O1"],
)
def test_identificadores_sao_isentos_na_comparacao_por_capitulo(identificador):
    assert numeros("Havia 10 casos.", f"Havia 10 casos com {identificador}.") == []
    # no alinhamento por parágrafo o dígito do identificador continua contando
    assert sum(pf.extrair_tokens(f"Havia 10 casos com {identificador}.")["numeros"].values()) >= 2


def test_identificador_novo_nao_conta_mas_numero_solto_continua():
    base = "A recomendação usa O1 e F4 em 7 casos."
    alvo = "A recomendação usa O3, F3 e F4 em 7 casos."
    assert numeros(base, alvo) == []
    ach = numeros(base, "A recomendação usa O3, F3 e F4 em 7 casos e 3 revisões.")
    assert [(a.numero, a.tipo) for a in ach] == [("3", "novo")]
    ach = numeros(base, "A recomendação usa O3 e F4 em 7 casos e 4 revisões.")
    assert [(a.numero, a.tipo) for a in ach] == [("4", "novo")]


def test_rotulo_de_classe_par_igual_a_2_continua_contando():
    # "PAR~=~2" não é identificador (o dígito vem depois de "=~"): fica na contagem
    assert pf.contar_numeros("Os pares com PAR~=~2 somam 5.")["2"] == 1


def test_modo_paragrafo_nao_isenta_identificadores():
    dif = pf.comparar(pf.extrair_tokens("Usa O1 e F4."), pf.extrair_tokens("Usa O3 e F4."))
    assert dif["numeros"] == {"removidos": ["1"], "adicionados": ["3"]}


# Nos textos de capítulo o "%" vem escapado ("95\%"): "%" sem escape abre comentário LaTeX
# e ``segmentar`` descarta o resto da linha.
@pytest.mark.parametrize(
    "nivel",
    ["IC 95\\%", "IC~95\\%", "IC de 95\\%", "IC 95", "intervalo de confiança de 95\\%",
     "intervalo de confiança de~95\\%"],
)
def test_nivel_de_confianca_do_ic_e_isento_na_comparacao_por_capitulo(nivel):
    # só o nível é isento: o número que vem depois continua contando
    ach = numeros("Havia 10 casos.", f"Havia 10 casos ({nivel}) em 3 réplicas.")
    assert [(a.numero, a.tipo, a.status) for a in ach] == [("3", "novo", "pendente")]
    # no modo parágrafo o "95" continua contando
    assert sum(pf.extrair_tokens(f"Havia 10 casos ({nivel}).")["numeros"].values()) >= 2


def test_limites_do_ic_e_95_solto_continuam_contando():
    ach = numeros("Havia 10 casos.", "Havia 10 casos (IC 95\\%: 6,4 a 34,7).")
    assert sorted(a.numero for a in ach) == ["34.7", "6.4"]
    ach = numeros("Havia 10 casos.", "Havia 10 casos em 95\\% das réplicas.")
    assert [a.numero for a in ach] == ["95%"]
    ach = numeros("Havia 10 casos.", "Havia 10 casos com cobertura de 95\\%.")
    assert [a.numero for a in ach] == ["95%"]


def test_modo_paragrafo_nao_isenta_nivel_de_confianca():
    dif = pf.comparar(pf.extrair_tokens("Taxa de 5."), pf.extrair_tokens("Taxa de 5 (IC 95\\%)."))
    assert dif["numeros"] == {"removidos": [], "adicionados": ["95%"]}


def test_modo_invalido(repo):
    base_commit(repo, {"text/latex/cap.tex": TEX_BASE})
    with pytest.raises(ValueError):
        pf.analisar_repositorio(repo, modo="nenhum")
    with pytest.raises(SystemExit) as exc:
        pf.main(["--raiz", str(repo), "--modo", "nenhum"])
    assert exc.value.code == 2


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
