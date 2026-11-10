"""\\path e \\url em \\caption: \\protect só dentro da legenda, idempotente."""

from scripts.tabelas_persona_editorial import proteger_caminhos_legenda, transform


def test_caminho_na_legenda_e_protegido():
    text = r"\caption{Fonte: \path{data/x.csv}.}"
    assert proteger_caminhos_legenda(text) == r"\caption{Fonte: \protect\path{data/x.csv}.}"


def test_url_na_legenda_e_protegida():
    text = r"\caption{Ver \url{https://exemplo.org/a_b}.}"
    assert proteger_caminhos_legenda(text) == r"\caption{Ver \protect\url{https://exemplo.org/a_b}.}"


def test_ja_protegido_nao_e_duplicado():
    text = r"\caption{Fonte: \protect\path{data/x.csv} e \protect\url{http://a.b}.}"
    assert proteger_caminhos_legenda(text) == text


def test_idempotente():
    text = r"\caption{Fonte: \path{a} e \url{b}, ainda \protect\path{c}.}"
    uma_vez = proteger_caminhos_legenda(text)
    assert uma_vez == r"\caption{Fonte: \protect\path{a} e \protect\url{b}, ainda \protect\path{c}.}"
    assert proteger_caminhos_legenda(uma_vez) == uma_vez


def test_fora_da_legenda_nao_muda():
    text = (
        "Prosa com \\path{data/a.csv} e \\url{http://a.b}.\n"
        "\\caption{Legenda sem caminho.}\n"
        "\\par\\footnotesize Nota: \\path{data/b.csv}.\n"
        "Mais prosa com \\path{data/c.csv}."
    )
    assert proteger_caminhos_legenda(text) == text


def test_so_o_caminho_dentro_da_legenda_e_protegido():
    text = (
        "\\caption{Legenda \\path{a}.}\n"
        "\\par\\footnotesize Nota \\path{b}.\n"
        "Prosa \\url{c}."
    )
    esperado = (
        "\\caption{Legenda \\protect\\path{a}.}\n"
        "\\par\\footnotesize Nota \\path{b}.\n"
        "Prosa \\url{c}."
    )
    assert proteger_caminhos_legenda(text) == esperado


def test_legenda_com_chaves_aninhadas():
    text = r"\caption{Nota \textbf{em {negrito \path{a}}} fim \path{b}.} Depois \path{c}."
    esperado = r"\caption{Nota \textbf{em {negrito \protect\path{a}}} fim \protect\path{b}.} Depois \path{c}."
    assert proteger_caminhos_legenda(text) == esperado


def test_legenda_com_argumento_opcional():
    text = r"\caption[Curta \path{a}]{Longa \path{b}.} Fora \path{c}."
    esperado = r"\caption[Curta \protect\path{a}]{Longa \protect\path{b}.} Fora \path{c}."
    assert proteger_caminhos_legenda(text) == esperado


def test_legenda_com_argumento_opcional_sem_caminho_curto():
    text = r"\caption[Curta]{Longa \path{b}.}"
    assert proteger_caminhos_legenda(text) == r"\caption[Curta]{Longa \protect\path{b}.}"


def test_chave_escapada_na_legenda_nao_desequilibra():
    text = r"\caption{Conjunto \{a, \path{b}.} Depois \path{c}."
    esperado = r"\caption{Conjunto \{a, \protect\path{b}.} Depois \path{c}."
    assert proteger_caminhos_legenda(text) == esperado


def test_varias_legendas():
    text = "\\caption{A \\path{a}}\n\\path{x}\n\\caption[c]{B \\url{b}}\n\\url{y}"
    esperado = "\\caption{A \\protect\\path{a}}\n\\path{x}\n\\caption[c]{B \\protect\\url{b}}\n\\url{y}"
    assert proteger_caminhos_legenda(text) == esperado


def test_texto_sem_legenda_inalterado():
    text = "\\begin{tabular}{ll}\na & \\path{x} \\\\\n\\end{tabular}\n"
    assert proteger_caminhos_legenda(text) == text
    assert proteger_caminhos_legenda("") == ""


def test_transform_aplica_a_protecao():
    text = "\\caption{Fonte: \\path{data/x.csv}.}\n\\par\\footnotesize Nota \\path{data/y.csv}.\n"
    saida = transform("tab_inexistente_no_mapa", text)
    assert r"\caption{Fonte: \protect\path{data/x.csv}.}" in saida
    assert r"Nota \path{data/y.csv}." in saida
    assert transform("tab_inexistente_no_mapa", saida) == saida
