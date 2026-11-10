"""Tabela tab_balanceamento_xgb: gerador determinístico, números iguais ao CSV, rótulo e legenda.

Não lê o extrato: só data/balanceamento_xgb.csv e data/balanceamento_xgb_metadata.json.
"""
import copy
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from scripts import tabela_balanceamento_xgb as gen
from scripts.tabelas_persona_editorial import proteger_caminhos_legenda

ROOT = Path(__file__).resolve().parents[1]
TABELA = ROOT / "text/latex/tables/editorial/tab_balanceamento_xgb.tex"
APENDICE = ROOT / "text/latex/apendice_sensibilidade.tex"
HISTORICA = ROOT / "text/latex/tables/editorial/tab_imbalance_sensitivity.tex"
FEATURE_ENGINEERING = ROOT / "scripts/feature_engineering.py"
SCRIPT_ANALISE = ROOT / "scripts/balanceamento_xgb.py"

TITULOS = {"rf": "Random Forest", "xgb": "XGBoost"}


@pytest.fixture(scope="module")
def dados():
    return gen.carregar()


@pytest.fixture(scope="module")
def texto(dados):
    return gen.render(*dados)


def esperado(valor):
    """Três casas, meia unidade para cima, vírgula decimal; -0,000 vira 0,000 (cálculo independente do gerador)."""
    q = Decimal(str(float(valor))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    s = f"{abs(q):.3f}".replace(".", ",")
    return "$-$" + s if q < 0 else s


METRICAS = ("VPP", "Sensibilidade", "F1", "AUC-PR")


def rotulos_historicos():
    """Rótulos da primeira coluna da tabela histórica, na ordem (lidos do arquivo, não copiados)."""
    rotulos = []
    for linha in HISTORICA.read_text(encoding="utf-8").splitlines():
        if " & " in linha and linha.endswith(r"\\") and not linha.startswith("Estrat"):
            rotulos.append(linha.split(" & ")[0].strip())
    return rotulos


def linhas_de_dados(texto):
    """{(estimador, painel 'A'|'B', rótulo da estratégia): células} a partir do corpo do tabular."""
    estimador, painel, saida = None, None, {}
    for linha in texto.splitlines():
        if linha.startswith(r"\multicolumn"):
            for codigo, titulo in TITULOS.items():
                if r"\textit{" + titulo + "}" in linha:
                    estimador, painel = codigo, None
            if "Painel A" in linha:
                painel = "A"
            elif "Painel B" in linha:
                painel = "B"
            continue
        if estimador and painel and " & " in linha and linha.endswith(r"\\") and not linha.startswith("Estrat"):
            celulas = [c.strip() for c in linha[:-2].split(" & ")]
            assert (estimador, painel, celulas[0]) not in saida
            saida[(estimador, painel, celulas[0])] = celulas[1:]
    return saida


def test_gerador_deterministico(dados, tmp_path):
    assert gen.render(*dados) == gen.render(*dados)
    a = gen.generate(tmp_path / "a").read_bytes()
    b = gen.generate(tmp_path / "b").read_bytes()
    assert a == b
    assert b"\r" not in a


def test_cada_numero_corresponde_ao_csv(dados, texto):
    df, meta = dados
    linhas = linhas_de_dados(texto)
    estrategias = meta["strategies"]
    # painel A: 9 estratégias por estimador; painel B: 8 (sem a referência)
    assert len(linhas) == len(TITULOS) * (len(estrategias) + len(estrategias) - 1) == 34
    for codigo in TITULOS:
        for chave in estrategias:
            rotulo = gen.ROTULOS[chave]
            celulas = linhas[(codigo, "A", rotulo)]
            assert len(celulas) == 4
            for celula, metrica in zip(celulas, METRICAS):
                m = df[(df.tipo == "desempenho") & (df.estimador == codigo)
                       & (df.estrategia == chave) & (df.metrica == metrica)]
                assert len(m) == 1
                assert celula == esperado(m.media.iloc[0]), (codigo, chave, metrica)
            if chave == "No balancing":
                assert (codigo, "B", rotulo) not in linhas
                continue
            celulas = linhas[(codigo, "B", rotulo)]
            assert len(celulas) == 4
            for celula, metrica in zip(celulas, METRICAS):
                d = df[(df.tipo == "dif_sem_balanceamento") & (df.estimador == codigo)
                       & (df.estrategia == chave) & (df.metrica == metrica)]
                assert len(d) == 1
                r = d.iloc[0]
                assert celula == (f"{esperado(r.diferenca_media)} "
                                  f"[{esperado(r.ic95_inf)}; {esperado(r.ic95_sup)}]"), (codigo, chave, metrica)


def test_valores_literais_do_csv(texto):
    """Conferência manual contra linhas lidas do CSV (arredondamento e sinal)."""
    linhas = linhas_de_dados(texto)
    # rf, sem balanceamento: VPP 0,96041..., sensibilidade 0,84685..., F1 0,89904..., AUC-PR 0,96303...
    assert linhas[("rf", "A", "Sem balanceamento")] == ["0,960", "0,847", "0,899", "0,963"]
    # xgb, sem balanceamento: F1 0,92930...
    assert linhas[("xgb", "A", "Sem balanceamento")][2] == "0,929"
    # rf, pesos de classe: dVPP -0,06062 [-0,09506; -0,02619]; dSens 0,01864 [-0,01766; 0,05495]
    assert linhas[("rf", "B", "Pesos de classe balanceados")][0] == "$-$0,061 [$-$0,095; $-$0,026]"
    assert linhas[("rf", "B", "Pesos de classe balanceados")][1] == "0,019 [$-$0,018; 0,055]"
    # rf, pesos de classe: dF1 -0,01796 [-0,04098; 0,00507]; dAUC-PR -0,01830 [-0,03264; -0,00395]
    assert linhas[("rf", "B", "Pesos de classe balanceados")][2] == "$-$0,018 [$-$0,041; 0,005]"
    assert linhas[("rf", "B", "Pesos de classe balanceados")][3] == "$-$0,018 [$-$0,033; $-$0,004]"
    # rf, SMOTE 0,3: dSens 0,04620 [0,01010; 0,08231]
    assert linhas[("rf", "B", "SMOTE 0,3")][1] == "0,046 [0,010; 0,082]"
    # xgb, pesos de classe: dVPP -0,08685 [-0,12266; -0,05103] (valores citados no Estudo 2)
    assert linhas[("xgb", "B", "Pesos de classe balanceados")][0] == "$-$0,087 [$-$0,123; $-$0,051]"
    # xgb, ADASYN 0,3: dAUC-PR 0,00344 [-0,00781; 0,01470]
    assert linhas[("xgb", "B", "ADASYN 0,3")][3] == "0,003 [$-$0,008; 0,015]"
    # xgb, SMOTETomek 0,3: dF1 -2,3e-05 [-0,01717; 0,01712] vira 0,000, sem sinal
    assert linhas[("xgb", "B", "SMOTETomek 0,3")][2] == "0,000 [$-$0,017; 0,017]"


def test_rotulos_iguais_aos_da_tabela_historica(dados, texto):
    """Os rótulos das estratégias são os de editorial/tab_imbalance_sensitivity.tex, na mesma ordem."""
    _, meta = dados
    historicos = rotulos_historicos()
    assert len(historicos) == 9
    assert [gen.ROTULOS[chave] for chave in meta["strategies"]] == historicos
    linhas = linhas_de_dados(texto)
    for codigo in TITULOS:
        assert [r for (e, p, r) in linhas if e == codigo and p == "A"] == historicos
        assert [r for (e, p, r) in linhas if e == codigo and p == "B"] == historicos[1:]
    assert "Ponderação" not in texto


def test_arredondamento_meia_unidade_para_cima():
    assert gen.numero(0.0005) == "0,001"
    assert gen.numero(0.0015) == "0,002"
    assert gen.numero(-0.0005) == "$-$0,001"
    assert gen.numero(-0.00001) == "0,000"
    assert gen.numero(4.1073252965652675e-05) == "0,000"


def test_rotulo_unico_e_estrutura(texto):
    assert texto.count(r"\label{tab:balanceamento-xgb}") == 1
    assert texto.count(r"\begin{table}") == texto.count(r"\end{table}") == 1
    assert texto.count(r"\begin{tabular}") == texto.count(r"\end{tabular}") == 1
    assert r"\textit{Random Forest}" in texto and r"\textit{XGBoost}" in texto


def test_sem_path_desprotegido_na_legenda_e_com_path_na_nota(texto):
    assert proteger_caminhos_legenda(texto) == texto
    legenda = re.search(r"\\caption\{(.*?)\}\n\\label", texto, re.S).group(1)
    assert "\\path" not in legenda and "\\url" not in legenda
    assert r"\path{data/balanceamento_xgb.csv}" in texto


def test_estilo_sem_travessao_e_terminologia(texto):
    assert not re.search(r"[\u2014\u2013]|--", texto)
    assert "precisão" not in texto.lower() and "precis" not in texto.lower()
    assert "clerical" not in texto.lower()


def test_nota_traz_ressalvas_conferidas_nos_metadados(dados, texto):
    _, meta = dados
    n = meta["features"]["n"]
    assert f"sobre os mesmos {n} atributos do Estudo 2, não sobre os 30 atributos de O3" in texto
    assert "configuração do estimador de O3" in texto and "fábrica" not in texto
    assert r"\path{gzcmd/ml_oof.py}" in texto
    assert r"O \texttt{scale\_pos\_weight} vale 1 sem pesos de classe; com pesos de classe, é a razão entre " \
           "negativos e positivos do treino da partição (após a reamostragem, na combinação com SMOTE)" in texto
    assert (f"Os {n} atributos incluem os indicadores de encerramento por óbito, ausentes de O3 "
            r"(\texttt{obito\_tb} e \texttt{obito\_\allowbreak sinan})") in texto
    assert "Os IC não têm correção para comparações múltiplas e são descritivos" in texto
    assert r"\cite{Nadeau2003inference}" in texto
    assert r"teste $t$ corrigido para validação cruzada repetida" in texto
    assert "estratificadas por linha, sem agrupar por REFREC" in texto
    assert r"da Tabela~\ref{tab:imbalance-sensitivity}" in texto
    # sem a redundância "5x5 repetida" e sem a frase que remetia ao CSV
    assert "repetida, limiar" not in texto and "constam do CSV" not in texto
    assert r"validação cruzada $5 \times 5$" in texto
    assert "n.a." not in texto


def test_nota_dos_indicadores_de_obito_confere_com_o_codigo():
    """A nota afirma que os 58 atributos incluem os indicadores de óbito; o código da análise os usa."""
    fe = FEATURE_ENGINEERING.read_text(encoding="utf-8")
    assert 'X["obito_sinan"] = df["C_SITUENCE"].isin([3, 4])' in fe
    assert 'X["obito_tb"] = (df["C_SITUENCE"] == 3)' in fe
    analise = SCRIPT_ANALISE.read_text(encoding="utf-8")
    assert "X, y = engineer_features(df)" in analise
    # O3 não os usa: numeric_features de gzcmd/ml_oof.py não os cita
    ml_oof = (ROOT / "gzcmd/ml_oof.py").read_text(encoding="utf-8")
    assert "obito" not in ml_oof


def test_gerador_recusa_metadados_que_nao_sustentam_a_nota(dados):
    df, meta = dados
    ruim = copy.deepcopy(meta)
    ruim["historical_reproduction"]["passou"] = False
    with pytest.raises(ValueError):
        gen.render(df, ruim)
    ruim = copy.deepcopy(meta)
    ruim["features"]["note"] = "sem referência aos atributos de O3"
    with pytest.raises(ValueError):
        gen.render(df, ruim)
    ruim = copy.deepcopy(meta)
    ruim["strategies"]["Estratégia nova"] = ruim["strategies"]["SMOTE 0.3"]
    with pytest.raises(ValueError):
        gen.render(df, ruim)


def test_tab_balanceamento_xgb_versionada_igual_a_regenerada(dados):
    """Manifesto: editorial/tab_balanceamento_xgb.tex só muda pelo gerador."""
    assert TABELA.is_file(), "execute: python scripts/tabela_balanceamento_xgb.py"
    versionada = TABELA.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert versionada == gen.render(*dados)


def test_apendice_inclui_e_apresenta_a_tabela():
    tex = APENDICE.read_text(encoding="utf-8")
    assert tex.count(r"\input{tables/editorial/tab_balanceamento_xgb") == 1
    assert r"Tabela~\ref{tab:balanceamento-xgb}" in tex
