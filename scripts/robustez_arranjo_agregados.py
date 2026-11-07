"""Reconstitui a etapa agregada da fase C, sem treino, API ou dados privados.

Não escreve nos artefatos históricos. Serve à verificação independente da
apresentação e dos cálculos registrados em scripts/arranjo_robustez.py.
"""
import pandas as pd

from gzcmd.arranjo_robustez import dominates, clopper_pearson, classify


def regenerate(table, primary, stability):
    numeric = list(table.select_dtypes(include='number').columns.difference(['fold_seed', 'xgb_seed', 'n_jobs']))
    records = []
    for (replica, block, mode, scope), part in table.groupby(['replica', 'bloco', 'modo', 'faixa'], sort=False):
        indexed = part.set_index('braco')
        p, q = indexed.loc['P'], indexed.loc['P-Platt']
        records.append(dict(replica=replica, bloco=block, modo=mode, faixa=scope,
                            **{key: p[key] - q[key] for key in numeric},
                            platt_domina=dominates(q, p), inversa=dominates(p, q), valida=True))
    differences = pd.DataFrame(records)
    summary, conclusions = [], {}
    for mode in ('vigilancia', 'confirmacao'):
        counts, inverse = [], []
        for block in ('N', 'P'):
            part = differences.loc[differences.modo.eq(mode) & differences.bloco.eq(block)
                                   & differences.faixa.eq('TODAS')]
            k, inv = int(part.platt_domina.sum()), int(part.inversa.sum())
            counts.append(k)
            inverse.append(inv)
            low, high = clopper_pearson(k)
            summary.append(dict(tipo='dominancia', modo=mode, bloco=block, k=k, n=20,
                                proporcao=k / 20, ic95_inf=low, ic95_sup=high,
                                inversa=inv, fp_positivo=int(part.fp.gt(0).sum())))
        conclusions[mode] = dict(classificacao=classify(counts, inverse),
                                 dominancia_k20=counts, inversa_k20=inverse)
    for (block, mode, arm), part in primary.groupby(['bloco', 'modo', 'braco']):
        other = primary.loc[primary.bloco.eq(block) & primary.modo.eq(mode)].pivot(
            index='replica', columns='braco', values='aceitos')
        summary.append(dict(tipo='PAR2', bloco=block, modo=mode, braco=arm,
                            platt_aceita_pelo_menos_P=float(other['P-Platt'].ge(other.P).sum() / 20),
                            perde_mais_aposteriori=float(part.perdidos.gt(9 if mode == 'vigilancia' else 17).sum() / 20)))
    for kind, frame, groups in (
        ('desfecho', table, ['bloco', 'modo', 'braco', 'faixa']),
        ('diferenca_P_menos_Platt', differences, ['bloco', 'modo', 'faixa']),
        ('PAR2_distribuicao', primary, ['bloco', 'modo', 'braco']),
        ('estabilidade', stability, ['bloco', 'modo', 'braco']),
    ):
        columns = frame.select_dtypes(include='number').columns.difference(['fold_seed', 'xgb_seed', 'n_jobs'])
        for keys, part in frame.groupby(groups):
            for column in columns:
                values = part[column]
                summary.append(dict(tipo=kind, **dict(zip(groups, keys)), metrica=column,
                                    mediana=values.median(), minimo=values.min(), maximo=values.max(),
                                    iqr=values.quantile(.75) - values.quantile(.25),
                                    positivo=values.gt(0).sum() / 20, zero=values.eq(0).sum() / 20,
                                    negativo=values.lt(0).sum() / 20))
    return differences, pd.DataFrame(summary), conclusions
