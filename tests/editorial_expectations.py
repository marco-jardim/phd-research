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
    # W13a acrescenta definições, sem alterar as estimativas das células.
    # O texto esperado é independente do transformador de produção.
    glossary = (
        r'\par\footnotesize P: arranjo principal, que preserva o corte do OpenRecLink e aplica a política com recalibração isotônica abaixo dele. '
        r'PAR~=~2: vínculos recuperados na revisão clerical abaixo do corte. '
        r'B1: mantém a base inteira e a probabilidade original, substituindo apenas as proporções de erro do revisor. '
        r'A3: restringe também o treinamento do classificador à região abaixo do corte. '
        r'P-PAR2-grey: estima a omissão do revisor somente nos vínculos PAR~=~2 e agrupa as faixas da área cinza da nota. '
        r'P-Platt: análise de sensibilidade de P com \textit{Platt scaling} abaixo do corte. '
        r'Braço \textit{a posteriori}: cascata publicada aplicada abaixo do corte, sem recalibração. '
        r'Viés de incorporação: uso, na construção da referência, das mesmas evidências avaliadas pelo método.'
    )
    return text.replace(r'\end{longtable}', r'\end{longtable}' + '\n' + glossary + '\n')
