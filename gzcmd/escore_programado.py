"""M7: escore sem gradiente, sintetizado por Haiku; não é probabilidade calibrada.

Versão escolhida apenas no treino do fold 1; não incorporada ao motor.
"""

def escore_programado(features):
    """Escore de vinculação semântica, em [0, 1], entre registro SIM e notificação Sinan-TB.
    Soma evidências ponderadas de concordância entre campos. Não identifica pessoas e não afirma identidade.
    Valor zero pode significar ausência ou discordância e contribui zero, sem veto.
    Migração de município/UF e incerteza na data de nascimento apenas reduzem ou mantêm a evidência; não eliminam o par.
    Não altera o dicionário de entrada."""
    n_pi = min(max(float(features.get('NOME prim frag igual', 0)), 0.0), 1.0)
    n_ul = min(max(float(features.get('NOME ult frag igual', 0)), 0.0), 1.0)
    n_ig = min(max(float(features.get('NOME qtd frag iguais', 0)), 0.0), 1.0)
    n_ra = min(max(float(features.get('NOME qtd frag raros', 0)), 0.0), 1.0)
    n_co = min(max(float(features.get('NOME qtd frag comuns', 0)), -1.0), 0.0)
    n_mp = min(max(float(features.get('NOME qtd frag muito parec', 0)), 0.0), 1.0)
    n_ab = min(max(float(features.get('NOME qtd frag abrev', 0)), 0.0), 1.0)
    nome = 0.25 * n_pi + 0.15 * n_ul + 0.20 * n_ig + 0.15 * n_ra + 0.15 * n_mp + 0.10 * n_ab + 0.15 * n_co
    nome = min(max(nome, 0.0), 1.0)

    m_pi = min(max(float(features.get('NOMEMAE prim frag igual', 0)), 0.0), 1.0)
    m_ul = min(max(float(features.get('NOMEMAE ult frag igual', 0)), 0.0), 1.0)
    m_ig = min(max(float(features.get('NOMEMAE qtd frag iguais', 0)), 0.0), 1.0)
    m_ra = min(max(float(features.get('NOMEMAE qtd frag raros', 0)), 0.0), 1.0)
    m_co = min(max(float(features.get('NOMEMAE qtd frag comuns', 0)), -1.0), 0.0)
    m_mp = min(max(float(features.get('NOMEMAE qtd frag muito parec', 0)), 0.0), 1.0)
    m_ab = min(max(float(features.get('NOMEMAE qtd frag abrev', 0)), 0.0), 1.0)
    nomemae = 0.25 * m_pi + 0.15 * m_ul + 0.20 * m_ig + 0.15 * m_ra + 0.15 * m_mp + 0.10 * m_ab + 0.15 * m_co
    nomemae = min(max(nomemae, 0.0), 1.0)

    d_ig = min(max(float(features.get('DTNASC dt iguais', 0)), 0.0), 1.0)
    d_ap = min(max(float(features.get('DTNASC dt ap 1digi', 0)), 0.0), 1.0)
    d_di = min(max(float(features.get('DTNASC dt inv dia', 0)), 0.0), 1.0)
    d_me = min(max(float(features.get('DTNASC dt inv mes', 0)), 0.0), 1.0)
    d_an = min(max(float(features.get('DTNASC dt inv ano', 0)), 0.0), 1.0)
    dt = max(d_ig, 0.6 * d_ap, 0.5 * max(d_di, d_me), 0.3 * d_an)
    dt = min(max(dt, 0.0), 1.0)

    u_ig = min(max(float(features.get('CODMUNRES uf igual', 0)), 0.0), 1.0)
    u_pr = min(max(float(features.get('CODMUNRES uf prox', 0)), 0.0), 0.5)
    l_ig = min(max(float(features.get('CODMUNRES local igual', 0)), 0.0), 1.0)
    l_pr = min(max(float(features.get('CODMUNRES local prox', 0)), 0.0), 0.5)
    mun = max(l_ig, 2.0 * l_pr, 0.4 * u_ig, 0.8 * u_pr)
    mun = min(max(mun, 0.0), 1.0)

    v_ig = min(max(float(features.get('ENDERECO via igual', 0)), 0.0), 1.0)
    v_pr = min(max(float(features.get('ENDERECO via prox', 0)), 0.0), 0.8)
    e_ig = min(max(float(features.get('ENDERECO numero igual', 0)), 0.0), 1.0)
    c_pr = min(max(float(features.get('ENDERECO compl prox', 0)), 0.0), 0.5)
    t_pr = min(max(float(features.get('ENDERECO texto prox', 0)), 0.0), 0.8)
    jac = min(max(float(features.get('ENDERECO tokens jacc', 0)), 0.0), 0.5)
    end = 0.30 * v_ig + 0.20 * v_pr + 0.20 * e_ig + 0.10 * c_pr + 0.10 * t_pr + 0.10 * jac
    end = min(max(end, 0.0), 1.0)

    score = 0.35 * nome + 0.20 * nomemae + 0.25 * dt + 0.10 * mun + 0.10 * end
    return float(min(max(score, 0.0), 1.0))

