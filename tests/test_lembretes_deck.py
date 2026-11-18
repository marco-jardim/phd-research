"""Testes do gerador de lembretes de siglas e termos do deck de defesa (scripts/lembretes_deck.py).

Cobre a detecção por palavra inteira e os casos ambíguos (F1 fila e F₁ métrica, C e CM, passo, meta,
vigilância e confirmação como modos, desempate, PAR, linkage em nome de pacote), a geração
determinística (idempotência), o `--verificar` e a prova de que cada verificação acusa o defeito.
O deck real é conferido em tests/test_defesa_html.py (bloco H11).
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import lembretes_deck as lem  # noqa: E402


def chaves_de(fragmento_html: str, *chaves: str) -> list[str]:
    return lem.termos_do_texto(lem.texto_visivel(fragmento_html), list(chaves))


# ---------------------------------------------------------------- casos ambíguos

def test_f1_fila_e_f1_metrica_sao_entradas_distintas():
    ambas = "<p>F1 intercala as filas; o <var>F</var><sub>1</sub> no limiar de 0,5</p>"
    assert chaves_de(ambas, "F1 (fila)", "F1 (métrica)") == ["F1 (fila)", "F1 (métrica)"]
    assert chaves_de("<p>o <var>F</var><sub>1</sub> médio</p>", "F1 (fila)", "F1 (métrica)") == ["F1 (métrica)"]
    assert chaves_de("<p>F1, intercala O3 e O1</p>", "F1 (fila)", "F1 (métrica)") == ["F1 (fila)"]


@pytest.mark.parametrize("trecho,esperado", [
    ("<p>filas C e CM, exploratórias</p>", ["C (fila)", "CM (fila)"]),
    ("<li>C: O3 com atributos de contexto</li>", ["C (fila)"]),
    ("<tr><th>C</th><td>20</td></tr>", ["C (fila)"]),
    ("<p>custos C<sub>H</sub> = 1 e C<sub>LLM</sub> = 0,005</p>", []),
    ("<p>critério C1 e a letra C do protocolo</p>", []),
])
def test_c_e_cm_so_como_filas(trecho, esperado):
    assert chaves_de(trecho, "C (fila)", "CM (fila)") == esperado


@pytest.mark.parametrize("trecho,achou", [
    ("<p>no passo 11 e nos passos 6 e 9</p>", True),
    ("<p>corte definido por passo de blocagem</p>", True),
    ("<p>a cada passo, qual faixa revisar</p>", False),
    ("<p>os próximos passos e a etapa seguinte</p>", False),
])
def test_passo_de_blocagem_contra_passo_no_sentido_geral(trecho, achou):
    assert bool(chaves_de(trecho, "passo")) is achou


@pytest.mark.parametrize("trecho,achou", [
    ("<p>na meta de 23 pares verdadeiros</p>", True),
    ("<th>Meta 9</th>", True),
    ("<p>as metas de 9 e 36</p>", True),
    ("<p>a meta do estudo e a metade dos casos</p>", False),
])
def test_meta_de_recuperacao(trecho, achou):
    assert bool(chaves_de(trecho, "meta")) is achou


def test_vigilancia_modo_contra_vigilancia_do_obito_e_confirmacao():
    assert chaves_de("<p>a vigilância do óbito por tuberculose</p>", "vigilância do óbito", "vigilância (modo)") \
        == ["vigilância do óbito"]
    assert chaves_de("<p>na vigilância, ASR aceitou 4; na confirmação, 9</p>",
                     "vigilância do óbito", "vigilância (modo)", "confirmação (modo)") \
        == ["vigilância (modo)", "confirmação (modo)"]


def test_desempate_pelo_llm_contra_desempate_aleatorio():
    assert chaves_de("<p>desempate pelo LLM sobre O1</p>", "desempate pelo LLM", "desempate aleatório") \
        == ["desempate pelo LLM"]
    assert chaves_de("<p>O1, desempate aleatório (média)</p>", "desempate pelo LLM", "desempate aleatório") \
        == ["desempate aleatório"]
    assert chaves_de("<p>30 sementes de desempate de O1</p>", "desempate pelo LLM", "desempate aleatório") \
        == ["desempate aleatório"]


def test_par_maiusculo_nao_confunde_com_par_comum_e_sinan_com_sinan_tb():
    assert chaves_de("<p>o campo PAR da referência</p>", "PAR", "par verdadeiro e não par") == ["PAR"]
    assert chaves_de("<p>um par verdadeiro e um não par</p>", "PAR", "par verdadeiro e não par") \
        == ["par verdadeiro e não par"]
    assert chaves_de("<p>casos do Sinan para tuberculose</p>", "Sinan", "Sinan-TB") == ["Sinan"]
    assert chaves_de("<p>no Sinan\u2011TB</p>", "Sinan", "Sinan-TB") == ["Sinan-TB"]
    assert chaves_de("<p>no Sinan-TB</p>", "Sinan", "Sinan-TB") == ["Sinan-TB"]


def test_linkage_no_nome_do_pacote_e_gz_cmd_com_hifen_nao_separavel():
    assert chaves_de("<p>pacote <code>gzcmd-record-linkage</code></p>", "linkage") == []
    assert chaves_de('<p>o <em lang="en">linkage</em> probabilístico</p>', "linkage") == ["linkage"]
    assert chaves_de("<p>pacote GZ\u2011CMD e GZ-CMD</p>", "GZ-CMD") == ["GZ-CMD"]


def test_match_nao_aparece_dentro_de_nonmatch_e_auc_nao_dentro_de_auc_pr():
    assert chaves_de("<p><code>NONMATCH</code> e <code>MATCH</code></p>", "MATCH", "NONMATCH") == ["NONMATCH", "MATCH"]
    assert chaves_de("<p><code>NONMATCH</code></p>", "MATCH", "NONMATCH") == ["NONMATCH"]
    assert chaves_de("<p>a AUC-PR e a AUC</p>", "AUC", "AUC-PR") == ["AUC-PR", "AUC"] or \
        chaves_de("<p>a AUC-PR e a AUC</p>", "AUC", "AUC-PR") == ["AUC", "AUC-PR"]
    assert chaves_de("<p>a AUC-PR</p>", "AUC", "AUC-PR") == ["AUC-PR"]


def test_texto_visivel_exclui_fonte_legenda_sr_voltar_lembrete_e_atributos():
    bloco = ('<section class="slide" id="S09" aria-label="O1 no rótulo"><header><h2 class="titulo">Título</h2>'
             '<a class="voltar" href="#S01" aria-label="O2">← voltar</a></header><div class="corpo">'
             '<table><caption class="sr">O3 só para leitor</caption><tr><td>texto</td></tr></table>'
             '<svg><title>O4</title><text>F3 visível</text></svg></div><footer class="pe">'
             '<!-- LEMBRETE -->\n<aside class="lembrete"><span class="it"><strong>F4</strong>: x</span></aside>\n<!-- /LEMBRETE -->\n'
             '<p class="fonte">Fonte: tese, O1.</p></footer></section>')
    t = lem.texto_visivel(bloco)
    assert "F3 visível" in t and "texto" in t and "Título" in t
    for fora in ("O1", "O2", "O3", "O4", "F4", "Fonte"):
        assert fora not in t, fora


# ---------------------------------------------------------------- geração e verificação

def _entrada(i, chave, nivel, exibir, defin):
    return (f'<div class="gl">\n<dt id="g-{i}" data-chave="{chave}" data-nivel="{nivel}" data-fonte="glossario.tex:1">{exibir}</dt>\n'
            f'<dd class="def">{defin}</dd>\n<dd class="ocor">Quadros: <!-- OCOR:{chave} --><!-- /OCOR:{chave} --></dd>\n</div>')


ENTRADAS_SINTETICAS = [("escore", "base", "escore", "resumo da concordância entre os registros"),
                       ("F3", "sigla", "F3", "fusão de O3, O1 e fila com MACD"),
                       ("O1", "sigla", "O1", "ordenação pelo escore do OpenRecLink"),
                       ("O3", "sigla", "O3", "ordenação pela probabilidade calibrada"),
                       ("curva de esforço", "termo", "curva de esforço", "pares recuperados por revisões feitas")]


def html_sintetico(corpo_s01="Usa O3 e depois O1, o escore e F3.", corpo_s02="Só o O1 e a curva de esforço.",
                   entradas=ENTRADAS_SINTETICAS) -> str:
    glossario = ('<div id="glossario" hidden><dl id="lista-glossario">\n'
                 + "\n".join(_entrada(i, *e) for i, e in enumerate(entradas)) + "\n</dl></div>")
    return ('<html><body><main id="palco">\n'
            '<section class="slide" id="S01" data-tipo="principal">\n<header><h2 class="titulo">Capa</h2></header>\n'
            f'<div class="corpo"><p>{corpo_s01}</p></div>\n<footer class="pe">\n'
            '<p class="fonte">Fonte: x.</p></footer>\n</section>\n'
            '<section class="slide" id="S02" data-tipo="principal">\n<header><h2 class="titulo">Segundo</h2></header>\n'
            f'<div class="corpo"><p>{corpo_s02}</p></div>\n<footer class="pe">\n'
            '<p class="fonte">Fonte: y.</p></footer>\n</section>\n</main>\n' + glossario + '\n</body></html>\n')


def itens(h, quadro):
    fim = h.index('id="S02"') if quadro == "S01" else h.index("</main>")
    ini = h.index(f'id="{quadro}"')
    return re.findall(r'<span class="it"><strong>(.*?)</strong>', h[ini:fim])


def test_aplicar_grava_lembrete_so_com_sigla_e_termo_na_ordem_de_aparicao_e_e_idempotente():
    h = lem.aplicar(html_sintetico())
    assert lem.aplicar(h) == h
    assert lem.verificar_lembretes(h) == []
    assert itens(h, "S01") == ["O3", "O1", "F3"], itens(h, "S01")          # "escore" (base) fica de fora
    assert itens(h, "S02") == ["O1", "curva de esforço"]
    assert "escore" not in "".join(itens(h, "S01"))


def test_data_termos_traz_os_tres_niveis_na_ordem_de_aparicao():
    h = lem.aplicar(html_sintetico())
    assert 'data-termos="O3|O1|escore|F3"' in h
    assert 'data-termos="O1|curva de esforço"' in h


def test_termo_definido_no_quadro_por_dfn_sai_do_lembrete_mas_fica_em_data_termos():
    h = lem.aplicar(html_sintetico(corpo_s02='<dfn data-termo="O1">O1</dfn>: fila pelo escore; e a curva de esforço.'))
    assert itens(h, "S02") == ["curva de esforço"]
    assert 'data-termos="O1|escore|curva de esforço"' in h
    assert lem.verificar_lembretes(h) == []
    assert lem.definidos_no_quadro(h[h.index('id="S02"'):]) == ["O1"]


def test_dfn_com_varias_chaves():
    h = lem.aplicar(html_sintetico(corpo_s02='<dfn data-termo="O1|curva de esforço">O1 e a curva de esforço</dfn>.'))
    assert itens(h, "S02") == []
    assert "<!-- LEMBRETE -->\n<!-- /LEMBRETE -->" in h[h.index('id="S02"'):]


def test_verificar_acusa_termo_usado_sem_lembrete():
    erros = lem.verificar_lembretes(html_sintetico())
    assert any("S01" in e and "usado sem item" in e for e in erros) or any("S01" in e and "0 lembretes" in e for e in erros)


def test_verificar_acusa_item_divergente_do_glossario():
    h = lem.aplicar(html_sintetico())
    adulterado = h.replace("ordenação pelo escore do OpenRecLink</span>", "ordenação por outro escore</span>", 1)
    assert adulterado != h
    assert any("diverge da definição do glossário" in e for e in lem.verificar_lembretes(adulterado))


def test_verificar_acusa_termo_base_no_lembrete():
    h = lem.aplicar(html_sintetico())
    com_base = h.replace('<aside class="lembrete" aria-label="%s"><strong class="rot">%s</strong> ' % (lem.ARIA, lem.ROTULO),
                         '<aside class="lembrete" aria-label="%s"><strong class="rot">%s</strong> <span class="it"><strong>escore'
                         '</strong>: resumo da concordância entre os registros</span> ' % (lem.ARIA, lem.ROTULO), 1)
    assert com_base != h
    assert any("nível base no lembrete" in e for e in lem.verificar_lembretes(com_base))


def test_verificar_acusa_termo_definido_no_quadro_repetido_no_lembrete():
    base = html_sintetico(corpo_s02='<dfn data-termo="O1">O1</dfn> e a curva de esforço.')
    h = lem.aplicar(base)
    repetido = h.replace('<strong class="rot">%s</strong> ' % lem.ROTULO,
                         '<strong class="rot">%s</strong> <span class="it"><strong>O1</strong>: ordenação pelo escore do '
                         'OpenRecLink</span> ' % lem.ROTULO)
    assert any("repete termo definido no próprio quadro" in e for e in lem.verificar_lembretes(repetido))


def test_verificar_acusa_item_sem_uso_fora_do_glossario_ordem_trocada_e_duplicado():
    h = lem.aplicar(html_sintetico())
    fora = h.replace("<strong>F3</strong>", "<strong>ZZ</strong>", 1)
    assert any("fora do glossário" in e for e in lem.verificar_lembretes(fora))
    s01 = h[h.index('id="S01"'):h.index('id="S02"')]
    item_o3 = re.search(r'<span class="it"><strong>O3</strong>.*?</span>', s01).group(0)
    item_o1 = re.search(r'<span class="it"><strong>O1</strong>.*?</span>', s01).group(0)
    trocado = h.replace(item_o3, "@@A@@", 1).replace(item_o1, item_o3, 1).replace("@@A@@", item_o1, 1)
    assert any("fora da ordem de aparição" in e for e in lem.verificar_lembretes(trocado))
    dup = re.search(r'<aside class="lembrete".*?</aside>', h, re.S).group(0)
    assert any("2 lembretes" in e for e in lem.verificar_lembretes(h.replace(dup, dup + dup, 1)))
    sem_termo = h.replace("Só o O1 e a curva de esforço.", "Nada a lembrar.")
    assert any("S02" in e and "sem termo da lista" in e for e in lem.verificar_lembretes(sem_termo))
    sobra = h.replace("Usa O3 e depois O1, o escore e F3.", "Usa O3 e depois O1 e o escore.")
    assert any("S01" in e and "sem uso no quadro" in e for e in lem.verificar_lembretes(sobra))


def test_verificar_acusa_data_termos_errado():
    h = lem.aplicar(html_sintetico())
    assert any("data-termos difere" in e for e in lem.verificar_lembretes(h.replace('data-termos="O1|curva de esforço"', 'data-termos="O1"')))


def test_divergencias_acusa_circular_nivel_invalido_dfn_fora_do_glossario_e_padrao_ausente():
    h = lem.aplicar(html_sintetico())
    assert any("padrão de detecção sem entrada no glossário" in e for e in lem.divergencias(h))
    circular = h.replace("ordenação pelo escore do OpenRecLink</dd>", "O1 é o O1</dd>")
    assert any("definição circular em O1" in e for e in lem.divergencias(circular))
    assert any("nível inválido" in e for e in lem.divergencias(h.replace('data-nivel="termo"', 'data-nivel="outro"')))
    sem_padrao = h.replace('data-chave="O1"', 'data-chave="ZZ"').replace("OCOR:O1", "OCOR:ZZ")
    assert any("entrada sem padrão de detecção: ZZ" in e for e in lem.divergencias(sem_padrao))
    with pytest.raises(ValueError):
        lem.ler_glossario("<html></html>")


def test_circularidade_da_definicao():
    def entrada(exibir, defin):
        return lem.Entrada("k", "k", "sigla", exibir, defin, "glossario.tex:1")
    assert lem.definicao_circular(entrada("corte (ponto de corte)", "ponto de corte do escore")) is False
    assert lem.definicao_circular(entrada("passo", "passo de blocagem")) is True
    assert lem.definicao_circular(entrada("vínculo", "vínculo aceito pela regra")) is True
    assert lem.definicao_circular(entrada("passo", "etapa de blocagem")) is False
    assert lem.definicao_circular(entrada("O1", "ordenação pelo escore")) is False


def test_injetar_e_verificar_pela_cli(tmp_path):
    arquivo = tmp_path / "deck.html"
    arquivo.write_bytes(html_sintetico().encode("utf-8"))
    assert lem.verificar_lembretes(lem.ler(arquivo))
    assert lem.injetar(arquivo) is True
    assert lem.verificar_lembretes(lem.ler(arquivo)) == []
    assert lem.injetar(arquivo) is False


def test_deck_real_em_sincronia_pela_cli(capsys):
    assert lem.main(["--verificar", str(lem.DECK)]) == 0
    assert "em sincronia" in capsys.readouterr().out


def test_todo_padrao_compila_e_nao_casa_com_texto_vazio():
    for chave, padrao in lem.PADROES.items():
        assert re.compile(padrao).search("") is None, chave


# ---------------------------------------------------------------- vocabulário mínimo por bloco

ENTRADAS_VOCABULARIO = [("O1", "sigla", "O1", "ordenação pelo escore"),
                        ("linkage", "base", '<em lang="en">linkage</em>', "método"),
                        ("curva de esforço", "termo", "curva de esforço", "pares por revisões")]


def test_blocos_do_roteiro_expande_a_tabela_de_tempo_e_ignora_total_e_quadros():
    md = ("| Bloco | Quadros | Segundos | Minutos |\n| :- | :- | -: | -: |\n| Abertura | S01 a S03 | 90 | 1,5 |\n"
          "| Fechamento | S04 a S05 | 60 | 1,0 |\n| Total | S01 a S05 | 150 | 2,5 |\n| S01 | Capa | 20 | 43 |\n")
    assert lem.blocos_do_roteiro(md) == {"Abertura": ["S01", "S02", "S03"], "Fechamento": ["S04", "S05"]}


def test_vocabulario_apresentados_trazidos_e_italico():
    h = lem.aplicar(html_sintetico(corpo_s01="Usa o O1 e o linkage.", corpo_s02="Só o O1, o linkage e a curva de esforço.",
                                   entradas=ENTRADAS_VOCABULARIO))
    md = lem.vocabulario_markdown(h, {"A": ["S01"], "B": ["S02"]})
    assert md == ("- **A (S01).** Apresentados no bloco: O1 [S01]; *linkage* [S01]. Trazidos de blocos anteriores: nenhum.\n"
                  "- **B (S02).** Apresentados no bloco: curva de esforço [S02]. "
                  "Trazidos de blocos anteriores: O1 [S01]; *linkage* [S01].")


def test_vocabulario_so_conta_quadros_principais_e_apresenta_no_primeiro_quadro_do_termo():
    h = html_sintetico(corpo_s01="Usa o O1.", corpo_s02="Só o O1 e o linkage.", entradas=ENTRADAS_VOCABULARIO)
    reserva = ('<section class="slide" id="R01" data-tipo="reserva">\n<header><h2 class="titulo">R</h2></header>\n'
               '<div class="corpo"><p>Só a curva de esforço.</p></div>\n<footer class="pe">\n'
               '<p class="fonte">Fonte: z.</p></footer>\n</section>\n')
    h = lem.aplicar(h.replace("</main>", reserva + "</main>"))
    md = lem.vocabulario_markdown(h, {"A": ["S01", "S02"]})
    assert "curva de esforço" not in md                       # só aparece em reserva
    assert md == "- **A (S01 a S02).** Apresentados no bloco: O1 [S01]; *linkage* [S02]. Trazidos de blocos anteriores: nenhum."


def test_vocabulario_inclui_termo_de_dfn_fora_de_data_termos():
    h = lem.aplicar(html_sintetico(corpo_s01="Usa o O1.", corpo_s02="Só o O1.", entradas=ENTRADAS_VOCABULARIO))
    h = re.sub(r'(id="S02"[^>]*?) data-termos="[^"]*"', r'\1 data-termos="O1"', h, count=1)
    h = h.replace("<p>Só o O1.</p>", '<p>Só o O1 e <dfn data-termo="curva de esforço">a curva</dfn>.</p>', 1)
    md = lem.vocabulario_markdown(h, {"B": ["S02"]})
    assert "curva de esforço [S02]" in md


def test_vocabulario_pela_cli(tmp_path, capsys):
    h = lem.aplicar(html_sintetico(corpo_s01="Usa o O1.", corpo_s02="Só o O1.", entradas=ENTRADAS_VOCABULARIO))
    html_arq, md_arq = tmp_path / "deck.html", tmp_path / "roteiro.md"
    html_arq.write_bytes(h.encode("utf-8"))
    md_arq.write_bytes("| Bloco | Quadros | Segundos | Minutos |\n| A | S01 a S02 | 10 | 0,2 |\n".encode("utf-8"))
    assert lem.main(["--vocabulario", str(html_arq), "--roteiro", str(md_arq)]) == 0
    esperado = lem.vocabulario_markdown(h, {"A": ["S01", "S02"]})
    assert capsys.readouterr().out.strip() == esperado


# ---------------------------------------------------------------- orçamento, data-define e identificadores

def entradas_do_orcamento():
    def e(chave, nivel, defin):
        return lem.Entrada(chave, chave, nivel, chave, defin, "glossario.tex:1")
    return [e("A", "sigla", "x" * 60), e("B", "termo", "y" * 60), e("C", "sigla", "z" * 60), e("D", "termo", "w" * 60),
            e("E", "sigla", "v" * 60)]


def test_prioridade_siglas_antes_de_termos_e_selecao_por_prefixo():
    g = entradas_do_orcamento()                              # A sigla, B termo, C sigla, D termo, E sigla
    nivel = {e.chave: e.nivel for e in g}
    todos = ["A", "B", "C", "D", "E"]
    assert lem.por_prioridade(todos, nivel) == ["A", "C", "E", "B", "D"]     # siglas primeiro, ordem estável em cada nível
    assert lem.por_prioridade(["B", "A", "D", "C"], nivel) == ["A", "C", "B", "D"]
    prio = lem.por_prioridade(todos, nivel)
    assert lem.selecionar_itens(prio, None) == (prio, [])
    mantidos, retirados = lem.selecionar_itens(prio, 3)
    assert mantidos == ["A", "C", "E"] and retirados == ["B", "D"]           # primeiro saem os termos, do fim para o começo
    mantidos, retirados = lem.selecionar_itens(prio, 2)
    assert mantidos == ["A", "C"] and retirados == ["E", "B", "D"]   # depois as siglas, do fim para o começo
    assert lem.caracteres(["abc", "de"]) == len("abc · de")


def test_ajuste_padrao_e_tabela_por_quadro(monkeypatch):
    assert lem.ajuste_do_quadro("ZZ99") == ("", None)                         # sem entrada: letra normal e lista inteira
    monkeypatch.setitem(lem.AJUSTES, "R10", ("mini", 3))
    assert lem.ajuste_do_quadro("R10") == ("mini", 3)
    assert lem.DEGRAUS == ("", "compacto", "mini")                            # não há degrau abaixo de mini
    assert "micro" not in lem.DEGRAUS


def html_com_orcamento(entradas, corpo, classe="", quadro="R10"):
    glossario = ('<div id="glossario" hidden><dl id="lista-glossario">\n'
                 + "\n".join(_entrada(i, e.chave, e.nivel, e.exibir, e.defin) for i, e in enumerate(entradas)) + "\n</dl></div>")
    return ('<html><body><main id="palco">\n'
            f'<section class="slide {classe}" id="{quadro}" data-tipo="reserva">\n<header><h2 class="titulo">T</h2></header>\n'
            f'<div class="corpo"><p>{corpo}</p></div>\n<footer class="pe">\n<p class="fonte">Fonte: x.</p></footer>\n'
            '</section>\n</main>\n' + glossario + '\n</body></html>\n')


PADROES_SINTETICOS = {"A": r"\bAA\b", "B": r"\bBB\b", "C": r"\bCC\b", "D": r"\bDD\b", "E": r"\bEE\b"}


def test_sem_corte_mostra_todos_os_termos_na_letra_da_tabela(monkeypatch):
    for k, v in PADROES_SINTETICOS.items():
        monkeypatch.setitem(lem.PADROES, k, v)
    ent = entradas_do_orcamento()
    longo = "AA BB CC DD EE"
    h = lem.aplicar(html_com_orcamento(ent, longo))
    assert itens(h, "R10") == ["A", "C", "E", "B", "D"]            # siglas primeiro, depois os termos
    assert "demais termos" not in h[h.index('id="R10"'):h.index("</main>")]
    assert '<aside class="lembrete"' in h and lem.verificar_lembretes(h) == []
    assert 'data-termos="A|B|C|D|E"' in h                          # o painel segue na ordem de aparição
    monkeypatch.setitem(lem.AJUSTES, "R10", ("compacto", None))
    h = lem.aplicar(html_com_orcamento(ent, longo))
    assert '<aside class="lembrete compacto"' in h and lem.verificar_lembretes(h) == []
    monkeypatch.setitem(lem.AJUSTES, "R10", ("mini", None))
    h = lem.aplicar(html_com_orcamento(ent, longo))
    assert '<aside class="lembrete mini"' in h and lem.verificar_lembretes(h) == []
    # o degrau fora da tabela ou incoerente com ela é acusado
    assert any("incoerente com a tabela" in e for e in lem.verificar_lembretes(h.replace("lembrete mini", "lembrete")))
    monkeypatch.setitem(lem.AJUSTES, "R10", ("micro", None))
    assert any("degrau de letra 'micro' fora de" in e for e in lem.verificar_lembretes(h))


def test_corte_da_tabela_mantem_prefixo_de_prioridade_e_poe_o_item_fixo(monkeypatch):
    for k, v in PADROES_SINTETICOS.items():
        monkeypatch.setitem(lem.PADROES, k, v)
    monkeypatch.setitem(lem.AJUSTES, "R10", ("compacto", 2))
    ent = entradas_do_orcamento()
    h = lem.aplicar(html_com_orcamento(ent, "AA BB CC DD EE"))
    assert itens(h, "R10") == ["A", "C", "demais termos"]           # os 2 primeiros por prioridade, mais o item fixo
    assert lem.verificar_lembretes(h) == []
    assert 'data-termos="A|B|C|D|E"' in h                            # o painel continua completo
    curto = lem.aplicar(html_com_orcamento(ent, "AA BB"))           # só A e B: o corte de 2 deixaria a lista inteira
    assert any("corte de 2 itens fora de" in e for e in lem.verificar_lembretes(curto))
    # item fixo ausente quando deveria existir, e presente quando não deveria
    sem_fixo = h.replace('<span class="it"><strong>demais termos</strong>: tecla g</span>', "")
    assert any("'demais termos' ausente" in e for e in lem.verificar_lembretes(sem_fixo))
    monkeypatch.setitem(lem.AJUSTES, "R10", ("", None))
    inteiro = lem.aplicar(html_com_orcamento(ent, "AA BB CC DD EE"))
    com_fixo_indevido = inteiro.replace("</aside>", '<span class="it"><strong>demais termos</strong>: tecla g</span></aside>')
    assert any("'demais termos' indevido" in e for e in lem.verificar_lembretes(com_fixo_indevido))
    # item de nível termo antes de sigla no lembrete é acusado
    assert any("fora da ordem" in e for e in lem.verificar_lembretes(
        inteiro.replace("<strong>A</strong>", "<strong>@@</strong>").replace("<strong>B</strong>", "<strong>A</strong>").replace("<strong>@@</strong>", "<strong>B</strong>")))


def test_data_define_pula_o_termo_no_lembrete_mas_ele_fica_em_data_termos():
    h = html_sintetico(corpo_s02='Só o O1 e a <figure class="figura" data-define="O1"><svg><text>O1: escore</text></svg></figure>.')
    h = lem.aplicar(h)
    assert itens(h, "S02") == []
    assert 'data-termos="O1|escore"' in h or 'data-termos="O1"' in h
    assert lem.definidos_no_quadro(h[h.index('id="S02"'):]) == ["O1"]
    assert lem.verificar_lembretes(h) == []


def test_regra_dos_identificadores_internos_nas_definicoes():
    def e(chave, defin):
        return lem.Entrada(chave, chave, "termo", chave, defin, "glossario.tex:1")
    assert lem.identificadores_na_definicao(e("bootstrap", "reamostragem por óbito (REFREC)")) == ["REFREC"]
    assert lem.identificadores_na_definicao(e("referência operacional", "classificação do campo PAR")) == ["PAR"]
    assert lem.identificadores_na_definicao(e("bootstrap", "reamostragem por óbito")) == []
    assert lem.identificadores_na_definicao(e("REFREC", "identificador do registro do SIM")) == []
    assert lem.identificadores_na_definicao(e("pair_id", "pseudônimo do par")) == []
    assert lem.identificadores_na_definicao(e("par", "partido; PARTE do par")) == []        # só palavra inteira


# ---------------------------------------------------------------- regra da calibração (sem navegador)

def test_recomendacao_mais_itens_e_em_empate_a_letra_maior():
    from scripts import verificar_deck_navegador as nav
    rec = nav.recomendar
    assert rec({"normal": 5, "compacto": 5, "mini": 5}, 5) == ("", None)           # lista inteira em tudo: letra normal
    assert rec({"normal": 2, "compacto": 6, "mini": 6}, 6) == ("compacto", None)   # inteira no maior degrau que couber
    assert rec({"normal": 2, "compacto": 3, "mini": 4}, 5) == ("mini", 4)          # S15: mais itens, mesmo a letra menor
    assert rec({"normal": 3, "compacto": 7, "mini": 8}, 10) == ("mini", 8)         # S21
    assert rec({"normal": 6, "compacto": 7, "mini": 12}, 13) == ("mini", 12)       # S23
    assert rec({"normal": 4, "compacto": 4, "mini": 4}, 9) == ("", 4)              # empate no número de itens: letra maior
    assert rec({"normal": 3, "compacto": 5, "mini": 5}, 9) == ("compacto", 5)
    assert rec({"normal": 0, "compacto": 0, "mini": 0}, 9) is None                 # nem um item cabe
