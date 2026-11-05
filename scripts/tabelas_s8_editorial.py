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


def same_load(data, projections=None, matches=None, scenarios=None, metrics=None):
    projections = projections if projections is not None else pd.read_csv(ROOT / 'data/comparador_mesma_carga_erro05.csv')
    keys = ['modo', 'ordenacao', 'carga', 'k']
    if projections.duplicated(keys).any():
        raise ValueError('Projeção de erro humano ambígua')
    data = data.merge(projections[keys + ['par2_topk_erro05']], on=keys, how='left', validate='one_to_one')
    if data.par2_topk_erro05.isna().any():
        raise ValueError('Projeção de erro humano ausente no CSV')
    matches = matches if matches is not None else pd.read_csv(ROOT / 'data/comparador_p_match_humano.csv')
    scenarios = scenarios if scenarios is not None else pd.read_csv(ROOT / 'data/cenarios_sensibilidade_humana.csv')
    metrics = metrics if metrics is not None else pd.read_csv(ROOT / 'data/arranjo_corte_metricas.csv')
    lines = [r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Ordenações abaixo do corte na mesma carga humana de P. Recuperação de PAR~=~2 sob revisor perfeito; análise exploratória, \textit{post hoc}, não registrada.}',
             r'\label{tab:cap8-mesma-carga}', r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Modo & Ordem & Revisões & PAR=2 & $S_{0,95}$ (\%) & Pares LLM (chamadas) & FP & $k$ para igualar P \\', r'\midrule']
    for mode, label in [('vigilancia', 'Vigilância'), ('confirmacao', 'Confirmação')]:
        for _, row in data.loc[data.modo.eq(mode) & data.carga.eq('K1_humana')].iterrows():
            lines.append(f'{label} & {row.ordenacao.split("_")[0]} & {number(row.k)} & '
                         f'{number(row.par2_topk)} & {number(row.sens_humano_05 * 100, 2)} & 0 (0) & '
                         f'{number(row.comparador_fp_revisor_perfeito)} & {number(row.k_min_equivalente_P)} ' + r'\\')
        policy = metrics.loc[metrics.modo.eq(mode) & metrics.braco.eq('P') & metrics.faixa.eq('TODAS')].iloc[0]
        match = matches.loc[matches.modo.eq(mode)].iloc[0]
        for strategy, title, reviews, fp in (
            ('P', 'P', policy.pares_humano, policy.fp),
            ('P_MATCH_ao_humano', r'P, \texttt{MATCH} ao humano', match.P_revisoes_humanas_total, match.P_FP_revisor_perfeito),
        ):
            projection = scenarios.loc[scenarios.modo.eq(mode) & scenarios.estrategia.eq(strategy) & scenarios.s.eq(.95)].iloc[0]
            lines.append(f'{label} & {title} & {number(reviews)} & {number(match.P_par2_recuperados)} & '
                         f'{number(projection.sensibilidade_total * 100, 2)} & '
                         f'{number(policy.pares_llm)} ({number(policy.chamadas_llm)}) & {number(fp)} & n.a. ' + r'\\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}}',
        r'\par\smallskip\parbox{\linewidth}{\footnotesize O1: escore do OpenRecLink; O2: nota do CdR; O3: \texttt{p\_cal\_ml}; O4: probabilidade recalibrada. PAR=2 recuperados pressupõe revisor perfeito dos dois lados; encaminhamento não é acerto observado. $S_{0,95}$ utiliza sensibilidade humana hipotética de 0,95, com especificidade perfeita. As outras sensibilidades estão na Tabela~\ref{tab:cap8-sensibilidade-humana}. Fontes: \path{data/comparador_mesma_carga.csv}, \path{data/comparador_p_match_humano.csv}, \path{data/cenarios_sensibilidade_humana.csv} e \path{data/arranjo_corte_metricas.csv}.}', r'\end{table}', ''])


def effort(data):
    lines = [r'\begin{table}[!htbp]', r'\centering\scriptsize',
             r'\caption{Esforço hipotético com 20 minutos por par. Economia frente ao cenário de 2.156 revisões; horas são ordem de grandeza, não medida.}',
             r'\label{tab:cap8-esforco}', r'\setlength{\tabcolsep}{3pt}',
             r'\resizebox{\linewidth}{!}{\begin{tabular}{lllrrrrrrr}', r'\toprule',
             r'Modo & Braço & Cenário & Humano & Pares LLM & PAR=2 & Horas & h poupadas & \shortstack{Horas poupadas\\por vínculo perdido} & h/FP \\', r'\midrule']
    for _, row in data.iterrows():
        mode = 'Vig.' if row.modo == 'vigilancia' else 'Conf.'
        cells = [mode, row.braco, row.carga.split('_')[0], number(row.revisoes), number(row.pares_llm), number(row.par2_recuperados),
                 number(row.horas), number(row.horas_poupadas),
                 number(row.horas_poupadas_por_vinculo_perdido), number(row.horas_por_fp)]
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}}',
        r'\par\smallskip\parbox{\linewidth}{\footnotesize K1: mesma carga humana. K2: orçamento misto hipotético; em P, humano e LLM permanecem separados; em O3, o número total de avaliações é reservado ao humano, sem equivalência temporal entre LLM e humano. Horas arredondadas a inteiros. n.a.: FP zero. Recuperação pressupõe revisor perfeito; razões não arredondadas e fontes temporais em \path{data/esforco_arranjo.csv}. Análise exploratória; cenários de sensibilidade na Tabela~\ref{tab:cap8-sensibilidade-humana}.}', r'\end{table}', ''])


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
    (output / 'tab_cap8_mesma_carga.tex').write_text(same_load(pd.read_csv(root / 'data/comparador_mesma_carga.csv'), pd.read_csv(root / 'data/comparador_mesma_carga_erro05.csv'), pd.read_csv(root / 'data/comparador_p_match_humano.csv'), pd.read_csv(root / 'data/cenarios_sensibilidade_humana.csv'), pd.read_csv(root / 'data/arranjo_corte_metricas.csv')), encoding='utf-8', newline='\n')
    (output / 'tab_cap8_esforco.tex').write_text(effort(pd.read_csv(root / 'data/esforco_arranjo.csv')), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    generate()
