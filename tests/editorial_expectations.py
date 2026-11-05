"""Transformações numéricas autorizadas nas notas editoriais do arranjo."""


def arrangement_expected_notes(text):
    """Atualiza só as duas frases históricas substituídas na S9, sem cortar notas."""
    replacements = [
        (r'IC de Wilson de 95\% descritivo, sem correção pela dependência entre potenciais pares ou pelos vieses do padrão-ouro. ', ''),
        (r'$S_H=(V_A+0{,}95V_H)/n$ é cenário de erro humano de 5\%, não resultado humano observado.',
         r'$S_H=(V_A+0{,}95V_H)/n$ exibe o cenário de sensibilidade humana 0,95, não resultado humano observado. Os três cenários, 0,95, 0,99 e 1,00, constam da Tabela~\ref{tab:cap8-sensibilidade-humana} e de \path{data/cenarios_sensibilidade_humana.csv}; a especificidade humana é perfeita por hipótese.'),
    ]
    for old,new in replacements:
        assert text.count(old)==1,old
        text = text.replace(old,new)
    return text
