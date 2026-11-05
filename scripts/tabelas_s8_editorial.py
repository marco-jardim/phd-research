"""Apresentação S8: agregados versionados e cópias editoriais, nunca históricas."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tabelas_arranjo_editorial import replace_once


def number(value, decimals=0):
    if pd.isna(value):
        return 'n.a.'
    return f'{value:,.{decimals}f}'.replace(',', 'X').replace('.', '{,}').replace('X', '.')


def same_load(data):
    lines = [r'\begin{table}[!htbp]', r'\centering\small',
             r'\caption{Ordenações abaixo do corte na mesma carga humana de P. Recuperação de PAR~=~2 sob revisor perfeito; análise exploratória, \textit{post hoc}, não registrada.}',
             r'\label{tab:cap8-mesma-carga}', r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Modo & Ordem & Revisões & PAR=2 & Com erro 5\% & $k$ para igualar P \\', r'\midrule']
    for mode, label in [('vigilancia', 'Vigilância'), ('confirmacao', 'Confirmação')]:
        for _, row in data.loc[data.modo.eq(mode) & data.carga.eq('K1_humana')].iterrows():
            # Todas as recuperações destas ordenações vêm do revisor humano.
            recovered05 = row.par2_topk * .95
            lines.append(f'{label} & {row.ordenacao.split("_")[0]} & {number(row.k)} & '
                         f'{number(row.par2_topk)} & {number(recovered05, 2)} & {number(row.k_min_equivalente_P)} ' + r'\\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\parbox{\linewidth}{\footnotesize O1: SCORE; O2: nota do CdR; O3: \texttt{p\_cal\_ml}; O4: probabilidade recalibrada. A curva completa está em \path{data/comparador_mesma_carga_curva.csv}; os pontos e as projeções vêm de \path{data/comparador_mesma_carga.csv}.}', r'\end{table}', ''])


def effort(data):
    lines = [r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Esforço hipotético com 20 minutos por par. Economia frente ao cenário de 2.156 revisões; horas são ordem de grandeza, não medida.}',
             r'\label{tab:cap8-esforco}', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lllrrrrrr}', r'\toprule',
             r'Modo & Braço & Carga & Rev. & PAR=2 & Horas & h poupadas & h/perdido & h/FP \\', r'\midrule']
    for _, row in data.iterrows():
        mode = 'Vig.' if row.modo == 'vigilancia' else 'Conf.'
        cells = [mode, row.braco, row.carga.split('_')[0], number(row.revisoes), number(row.par2_recuperados),
                 number(row.horas, 2), number(row.horas_poupadas, 2),
                 number(row.horas_por_vinculo_perdido, 2), number(row.horas_por_fp, 2)]
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\parbox{\linewidth}{\footnotesize K1: mesma carga humana; K2: total humano e LLM de P convertido em orçamento humano para O3. n.a.: denominador FP zero. Recuperação pressupõe revisor perfeito; cenários com erro humano e razões de horas poupadas por erro constam de \path{data/esforco_arranjo.csv}. Análise exploratória, \textit{post hoc}, não registrada.}', r'\end{table}', ''])


def generate(root=ROOT, output=None):
    output = output or root / 'text/latex/tables/editorial'
    output.mkdir(parents=True, exist_ok=True)
    historical = root / 'text/latex/tables'
    for name in ('tab_cap8_ancoras_beta', 'tab_cap8_ancoras_decisao', 'tab_cap8_robustez_arranjo'):
        text = (historical / (name + '.tex')).read_text(encoding='utf-8')
        if name.endswith('robustez_arranjo'):
            text = replace_once(text, r'N ($k/20$) & P ($k/20$) & Inversa (N; P)',
                                r'Numérico ($k/20$) & Partição ($k/20$) & Inversa (num.; part.)')
        else:
            # Uma substituição verificada por linha, incluindo as duas linhas com 247.
            lines = []
            for line in text.splitlines():
                if 'rótulos' in line:
                    line = replace_once(line, 'rótulos', r'\textit{labels}')
                if 'Platt (nota)' in line:
                    line = replace_once(line, 'Platt (nota)', r'\textit{Platt scaling} (nota, supervisionado)')
                if 'Platt por âncoras (ML)' in line:
                    line = replace_once(line, 'Platt por âncoras (ML)', 'Sigmoide por momentos nas âncoras (ML)')
                if 'Supervisionada (Platt,' in line:
                    line = replace_once(line, 'Supervisionada (Platt,', r'Supervisionada (\textit{Platt scaling},')
                lines.append(line)
            text = '\n'.join(lines) + '\n'
        (output / (name + '.tex')).write_text(text, encoding='utf-8', newline='\n')
    (output / 'tab_cap8_mesma_carga.tex').write_text(same_load(pd.read_csv(root / 'data/comparador_mesma_carga.csv')), encoding='utf-8', newline='\n')
    (output / 'tab_cap8_esforco.tex').write_text(effort(pd.read_csv(root / 'data/esforco_arranjo.csv')), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    generate()
