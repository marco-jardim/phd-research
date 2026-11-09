"""Tabelas editoriais de análises existentes; nenhum treino ou nova classificação."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tabelas_s8_editorial import number
from scripts.tabelas_persona_editorial import transform


def table(caption, label, columns, header, rows, note):
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering\scriptsize', r'\caption{'+caption+'}',
        r'\label{'+label+'}', r'\resizebox{\linewidth}{!}{\begin{tabular}{'+columns+'}', r'\toprule',
        header + r' \\', r'\midrule', *(' & '.join(row)+r' \\' for row in rows), r'\bottomrule',
        r'\end{tabular}}', r'\par\smallskip\parbox{\linewidth}{\footnotesize '+note+'}', r'\end{table}', ''])


def subsampling(data):
    selected = data.loc[data.population.eq('below') & data.variant.eq('isotonic')]
    metrics = ['par2_top_96', 'brier', 'ece', 'sens96_passo_11', 'sens96_sexo_M']
    rows = []
    first = selected.loc[selected.ratio.eq('1:1')].set_index('metric')
    rows.append(['Original', number(first.loc[metrics[0], 'reference']),
                 *[number(first.loc[m, 'reference'], 6) for m in metrics[1:3]],
                 *[number(first.loc[m, 'reference'] * 100, 1) for m in metrics[3:]]])
    for ratio in ('1:1', '1:5', '1:20', '1:50'):
        part = selected.loc[selected.ratio.eq(ratio)].set_index('metric')
        top = part.loc['par2_top_96']
        rows.append([ratio, number(top['median'], 1)+' ['+number(top['minimum'])+'; '+number(top['maximum'])+']',
            *[number(part.loc[m, 'median'], 6) for m in metrics[1:3]],
            *[number(part.loc[m, 'median'] * 100, 1) for m in metrics[3:]]])
    return table('Subamostragem no treinamento: análise exploratória, posterior à avaliação principal. Ordenação isotônica abaixo do corte.',
        'tab:cap8-subamostragem', 'lrrrrr', r'Pos.:neg. & PAR=2 no topo 96 & Brier & ECE & \shortstack{Fração dos PAR=2 do passo 11\\no topo 96 (\%)} & Sens. sexo M (\%)', rows,
        r'Medianas entre sementes; colchetes indicam mínimo e máximo, não IC. A referência foi executada uma vez. A avaliação mantém a prevalência original. Métricas das outras variantes e grupos: \path{data/subamostragem_treino_resumo.csv}.')


def llm_filter(data):
    rows = []
    for _, row in data.iterrows():
        rows.append(['Vig.' if row.modo == 'vigilancia' else 'Conf.', number(row.k), number(row.H),
                     number(row.par2_recuperados), number(row.par2_descartados), number(row.o3_par2_mesma_carga),
                     number(row.vpn_nonmatch * 100, 1)+' ['+number(row.vpn_ic95_inf * 100, 1)+'; '+number(row.vpn_ic95_sup * 100, 1)+']'])
    return table('Filtro LLM de NONMATCH antes do revisor humano: análise exploratória. $k$ é a fila submetida ao LLM, não a carga humana $H$.',
        'tab:cap8-filtro-llm', 'lrrrrrl', r'Modo & $k$ & $H$ & PAR=2 recup. & PAR=2 descart. & O3, mesmo $H$ & VPN (\%; IC 95\%)', rows,
        r'Recuperação pressupõe revisor perfeito após o filtro. MATCH e UNSURE vão ao humano. Somente orçamentos com cobertura completa; topo 440 incompleto, não interpolado. Fontes: \path{data/filtro_llm.csv} e \path{data/filtro_llm_cobertura.csv}.')


def human_sensitivity(data, curve=None):
    # Camada de apresentação: não altera os cenários históricos ou metadados.
    # A fila O3 é comum; as cargas pertinentes à confirmação são 17/29/58.
    curve = curve if curve is not None else pd.read_csv(ROOT / 'data/comparador_mesma_carga_curva.csv')
    data = data.copy()
    templates = data.loc[data.modo.eq('confirmacao') & data.estrategia.eq('O3') & data.k.eq(96)]
    extra = []
    for k in (17, 29, 58):
        count = curve.loc[curve.ordenacao.eq('O3_p_cal_ml') & curve.k.eq(k), 'par2_topk']
        if len(count) != 1:
            raise ValueError('Ponto O3 ausente ou ambíguo')
        for _, original in templates.iterrows():
            row = original.copy()
            row['k'] = row['revisoes_humanas'] = k
            row['vinculos_ao_humano'] = int(count.iloc[0])
            row['sensibilidade_total'] = (row['vinculos_aceitos_sem_humano'] + row['s'] * row['vinculos_ao_humano']) / row['vinculos_total']
            extra.append(row)
    if len(extra) != 9:
        raise ValueError('Três cenários humanos de confirmação são necessários')
    data = pd.concat([data, pd.DataFrame(extra)], ignore_index=True)
    selected = data.loc[data.estrategia.isin(['humano_integral', 'P', 'P_MATCH_ao_humano']) |
        (data.estrategia.eq('O3') & ((data.modo.eq('vigilancia') & data.k.isin([96, 125, 440])) |
                                     (data.modo.eq('confirmacao') & data.k.isin([17, 29, 58])))) |
        (data.estrategia.eq('filtro_LLM') & data.k.isin([96, 125]))]
    labels = {'humano_integral': 'Integral', 'P': 'ASR', 'P_MATCH_ao_humano': r'ASR, MATCH ao humano',
              'O3': 'O3', 'filtro_LLM': 'Filtro'}
    rows = []
    for (mode, strategy, k), group in selected.groupby(['modo', 'estrategia', 'k'], sort=False):
        values = group.set_index('s')
        rows.append(['Vig.' if mode == 'vigilancia' else 'Conf.', labels[strategy], number(k),
            number(group.iloc[0].revisoes_humanas), *[number(values.loc[s, 'sensibilidade_total'] * 100, 2) for s in (.95, .99, 1.)]])
    return table('Sensibilidade total em três cenários de sensibilidade humana $s$, não medida, com especificidade humana perfeita.',
        'tab:cap8-sensibilidade-humana', 'llrrrrr', r'Modo & Estratégia & $k$ & Humano & $s=0{,}95$ & $s=0{,}99$ & $s=1{,}00$', rows,
        r'Percentuais sobre os vínculos da referência. $k$ do filtro é fila LLM, não humano. Não se modelam FP humanos. O3 na confirmação usa as cargas 17, 29 e 58, com contagens de \path{data/comparador_mesma_carga_curva.csv} e a fórmula $(211+sV_H)/247$, cálculo da tese na camada editorial. Demais cenários, amplitude e comparações de ordem em \path{data/cenarios_sensibilidade_humana.csv}, \path{data/cenarios_sensibilidade_humana_amplitude.csv} e \path{data/cenarios_sensibilidade_humana_ordem.csv}.')


def generate(root=ROOT, output=None):
    output = output or root / 'text/latex/tables/editorial'
    if output.resolve() == (root / 'text/latex/tables').resolve():
        raise ValueError('As tabelas históricas são imutáveis')
    output.mkdir(parents=True, exist_ok=True)
    for name, renderer, source in (
        ('tab_cap8_subamostragem', subsampling, 'subamostragem_treino_resumo'),
        ('tab_cap8_filtro_llm', llm_filter, 'filtro_llm'),
        ('tab_cap8_sensibilidade_humana', human_sensitivity, 'cenarios_sensibilidade_humana'),
    ):
        (output / (name+'.tex')).write_text(transform(name, renderer(pd.read_csv(root / ('data/'+source+'.csv')))), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    generate()
