"""Análise quantitativa de viés: sensibilidade à referência, não erro estimado."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.loader import LoadConfig, load_comparador_csv
from gzcmd.ml_oof import pair_ids
from scripts.comparador_mesma_carga import (
    align_original, cumulative_counts, load_raw, matches_hash, numeric,
)
from scripts.fusao_filas import queue, rrf, validate_order

SEED = 20261107
REPETITIONS = 1000
FP_GRID = (0., .25, .5, .75, 1.)
PAR_GRID = (0., .05, .1, .2)
TARGETS = (25, 64, 100)
STRONG_FIELDS = tuple(
    f"{name} {suffix}" for name in ("NOME", "NOMEMAE")
    for suffix in ("qtd frag iguais", "prim frag igual", "ult frag igual")
) + ("DTNASC dt iguais",)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def expected_metrics(base, low, high, a_low, a_high, b):
    """Uniform error within PAR=2; all reclassified FP were accepted by P."""
    added = low * a_low + high * a_high
    accepted_below = float(base.aceitos) - 211
    total = 247 - 36 * b + added
    accepted = float(base.aceitos) - accepted_below * b + added
    human = float(base.vinc_humano) * (1 - b)
    return {
        "vpp_llm_abaixo": (accepted_below * (1 - b) + added)
        / (accepted_below + float(base.fp)),
        "aceitos_vp": accepted,
        "fp": float(base.fp) - added + accepted_below * b,
        "perdidos": float(base.perdidos) * (1 - b),
        "vinculos": total,
        "vinculos_abaixo": 36 * (1 - b) + added,
        "vinculos_humano": human,
        "sensibilidade_auto": accepted / total,
        "sensibilidade_humano_perfeito": (accepted + human) / total,
    }


def target_counts(n):
    # Nearest integer, halves up: 36 -> 9/23/36 (64% rounded in Study 4).
    return np.maximum(1, np.floor(n * np.array(TARGETS) / 100 + .5).astype(int))


def minimum_target_counts(n):
    """Strict fraction attained, complementary to Study 4's rounded targets."""
    return np.maximum(1, np.ceil(n * np.array(TARGETS) / 100).astype(int))


def select_changes(group, fraction, permutation, uniform):
    """Fixed-size simple sample, with unbiased stochastic rounding of n*f."""
    expected = len(group) * fraction
    n = math.floor(expected) + int(uniform < expected - math.floor(expected))
    return group[permutation[:n]]


def simulate(ranks, positives, groups, fractions, repetitions=REPETITIONS):
    rng = np.random.default_rng(SEED)
    values = {f"{name}_revisoes_{t}": [] for name in ranks for t in TARGETS}
    values.update({f"{pair}_revisoes_{t}": [] for pair in
                   ("F4-O1", "F4-O3", "O3-O1") for t in TARGETS})
    values.update({f"O3-O1_margem25_{t}": [] for t in TARGETS})
    values.update({f"{name}_revisoes_minimo_{t}": [] for name in ranks for t in TARGETS})
    values.update({f"{pair}_revisoes_minimo_{t}": [] for pair in
                   ("F4-O1", "F4-O3", "O3-O1") for t in TARGETS})
    values["total_positivos_sorteado"] = []
    for _ in range(repetitions):
        changes = [select_changes(g, f, rng.permutation(len(g)), rng.random())
                   for g, f in zip(groups, fractions, strict=True)]
        remaining = np.setdiff1d(positives, changes[2], assume_unique=True)
        selected = np.concatenate((remaining, changes[0], changes[1]))
        targets = target_counts(len(selected))
        sorted_ranks = {name: np.sort(rank[selected]) for name, rank in ranks.items()}
        effort = {name: ordered[targets - 1] for name, ordered in sorted_ranks.items()}
        minimum = {name: ordered[minimum_target_counts(len(selected)) - 1]
                   for name, ordered in sorted_ranks.items()}
        values["total_positivos_sorteado"].append(len(selected))
        for name, reviews in effort.items():
            for t, value in zip(TARGETS, reviews, strict=True):
                values[f"{name}_revisoes_{t}"].append(int(value))
                values[f"{name}_revisoes_minimo_{t}"].append(int(minimum[name][TARGETS.index(t)]))
        for pair in ("F4-O1", "F4-O3", "O3-O1"):
            left, right = pair.split("-")
            for j, t in enumerate(TARGETS):
                values[f"{pair}_revisoes_{t}"].append(int(effort[left][j] - effort[right][j]))
                values[f"{pair}_revisoes_minimo_{t}"].append(int(minimum[left][j] - minimum[right][j]))
        for j, t in enumerate(TARGETS):
            values[f"O3-O1_margem25_{t}"].append(float(effort["O3"][j] - .75 * effort["O1"][j]))
    return {key: np.quantile(value, [.5, .025, .975], method="linear")
            for key, value in values.items()}


def load_inputs(data_root, policy_dir):
    from scripts import arranjo_corte_s2_suplementar as supplement

    source = data_root / "data/COMPARADORSEMIDENT.csv"
    raw = load_raw(source)  # Identity-preserving project comparator loader.
    engineered = load_comparador_csv(source, cfg=LoadConfig(macd_enabled=False))
    if len(engineered) != len(raw):
        raise ValueError("Loader row alignment changed")
    for field in STRONG_FIELDS:
        if not np.allclose(numeric(raw[field]), numeric(engineered[field]), equal_nan=False):
            raise ValueError("Loader subscore alignment changed")
    below = below_cutoff(raw)
    ids = pair_ids(raw).to_numpy()
    original = align_original(raw, data_root / "data/pcal_ml_oof.csv", source)
    lexical = np.sort(ids[below])
    orders = {name: np.searchsorted(lexical, queue(ids, score, below)) for name, score in (
        ("O1", numeric(raw.SCORE).to_numpy()),
        ("O3", original.p_cal_ml.to_numpy()),
    )}
    orders["F4"] = rrf([orders["O3"], orders["O1"]])
    ranks = {}
    for name, order in orders.items():
        validate_order(order, len(lexical))
        ranks[name] = np.empty(len(order), dtype=int)
        ranks[name][order] = np.arange(1, len(order) + 1)
    frame = raw.copy()
    frame.index = ids
    frame = frame.loc[lexical].copy()
    par = numeric(frame.PAR).to_numpy()
    positives = np.flatnonzero(par == 2)
    if len(frame) != 61485 or len(positives) != 36:
        raise ValueError("Universe does not reproduce 61485/36")
    for name, score in (("O1", numeric(raw.SCORE).to_numpy()),
                        ("O3", original.p_cal_ml.to_numpy())):
        counts = cumulative_counts(ids[below], score[below], numeric(raw.PAR).to_numpy()[below] == 2)
        if not np.array_equal(counts, (par[orders[name]] == 2).cumsum()):
            raise ValueError("Comparator queue mismatch")
    reference = pd.read_csv(ROOT / "data/fusao_filas.csv")
    for name in orders:
        rows = reference.loc[reference.fila.eq(name)].set_index("metrica")
        observed = np.sort(ranks[name][positives])[[8, 22, 35]]
        expected = rows.loc[["revisoes_9", "revisoes_23", "revisoes_36"], "media"].to_numpy()
        if not np.array_equal(observed, expected):
            raise ValueError("Fusion baseline mismatch")
    thresholds = frame.loc[par == 2].groupby("PASSO").SCORE.apply(lambda s: numeric(s).min())
    above_step = numeric(frame.SCORE).to_numpy() >= frame.PASSO.map(thresholds).to_numpy()
    above_reference = raw.loc[~below & numeric(raw.PAR).eq(1)].copy()
    conflict = np.zeros(len(frame), dtype=bool)
    for field in ("COMPREC", "REFREC"):
        conflict |= frame[field].str.strip().isin(above_reference[field].str.strip()).to_numpy()
    cardinality = json.loads((ROOT / "data/fp_cardinalidade.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "data/arranjo_corte_metricas.csv")
    region_ref = pd.read_csv(ROOT / "data/fp_arranjo_por_regiao.csv")
    steps_ref = pd.read_csv(ROOT / "data/par_por_passo.csv")
    policy_meta = json.loads((ROOT / "data/robustez_arranjo_metadata.json").read_text(encoding="utf-8"))
    policies, fingerprints = {}, {"data/COMPARADORSEMIDENT.csv": sha256(source)}
    for mode in ("vigilancia", "confirmacao"):
        path = policy_dir / f"policy_{mode}_P.csv"
        logical = f".local/arranjo_corte/{path.name}"
        if not matches_hash(path, policy_meta["registered_sha256"][logical]):
            raise ValueError("Archived policy fingerprint mismatch")
        policy = pd.read_csv(path).set_index("pair_id")
        if not policy.index.is_unique or set(policy.index) != set(lexical):
            raise ValueError("Policy identity mismatch")
        policy = policy.loc[lexical]
        # Historical supplements can live alongside archived policies rather
        # than in the portable worktree. Change only in-memory resolver roots.
        old_base, old_out = supplement.BASE_OUT, supplement.OUT
        try:
            roots = (data_root, policy_dir.parents[1])
            resolved = []
            for folder in ("anthropic_sem_mo_v4_arranjo_corte", "anthropic_sem_mo_v4_arranjo_corte_supl"):
                resolved.append(next((root / "data/sprint3b" / folder for root in roots
                                      if (root / "data/sprint3b" / folder / f"exp2_llm_sync_{mode}.json").is_file()),
                                     data_root / "data/sprint3b" / folder))
            supplement.BASE_OUT, supplement.OUT = resolved
            known, sources = supplement.responses(data_root, mode, include_supplement=True)
        finally:
            supplement.BASE_OUT, supplement.OUT = old_base, old_out
        decisions = pd.Series(policy.index.map(known), index=policy.index)
        llm = policy.action.eq("LLM_REVIEW")
        if decisions.loc[llm].isna().any():
            raise ValueError("Archived LLM response missing")
        accepted = (policy.action.eq("MATCH") | (llm & decisions.eq("MATCH"))).to_numpy()
        fp = accepted & (par == 0)
        base = metrics.loc[metrics.modo.eq(mode) & metrics.braco.eq("P") & metrics.faixa.eq("TODAS")].iloc[0]
        if int(fp.sum()) != base.fp or int(((par == 2) & accepted).sum()) != base.aceitos - 211:
            raise ValueError("Historical P metrics mismatch")
        if int((fp & ~conflict).sum()) != cardinality["modos"][mode]["nenhum_ja_aceito"]:
            raise ValueError("Cardinality mismatch")
        for region, mask in (("menor", ~above_step), ("maior_ou_igual", above_step)):
            expected = region_ref.loc[region_ref.modo.eq(mode) & region_ref.limiar.eq("por_passo")
                                      & region_ref.passo.eq("TOTAL") & region_ref.regiao_SCORE.eq(region), "FP"].item()
            if int((fp & mask).sum()) != expected:
                raise ValueError("Step-threshold FP mismatch")
        for step in (6, 9, 11):
            for label in (0, 2):
                observed = int(((numeric(frame.PASSO).to_numpy() == step) & (par == label) & accepted).sum())
                expected = steps_ref.loc[steps_ref.modo.eq(mode) & steps_ref.passo.eq(step)
                                        & steps_ref.par.eq(label) & steps_ref.decisao.eq("MATCH"), "n"].sum()
                if observed != expected:
                    raise ValueError("PAR by step mismatch")
        policies[mode] = {"base": base, "fp": fp,
                          "groups": (np.flatnonzero(fp & ~above_step), np.flatnonzero(fp & above_step), positives)}
        fingerprints[logical] = sha256(path)
        for response_path in sources:
            fingerprints["responses/" + response_path.parent.name + "/" + response_path.name] = sha256(response_path)
    maxima = {field: float(numeric(raw[field]).max()) for field in STRONG_FIELDS}
    strong = np.ones(len(frame), dtype=bool)
    for field, maximum in maxima.items():
        strong &= numeric(frame[field]).to_numpy() == maximum
    sample = np.random.default_rng(SEED).choice(np.flatnonzero(par == 0), 1000, replace=False)
    cohorts = [(mode, "FP_P", work["fp"]) for mode, work in policies.items()]
    cohorts += [("referencia", "PAR2", par == 2),
                ("referencia", "amostra_PAR0", np.isin(np.arange(len(frame)), sample))]
    identifier_rows = []
    for mode, cohort, mask in cohorts:
        for region, region_mask in (("menor", ~above_step), ("maior_ou_igual", above_step),
                                    ("TOTAL", np.ones(len(frame), dtype=bool))):
            use = mask & region_mask
            identifier_rows.append({"modo": mode, "grupo": cohort, "regiao_limiar_passo": region,
                                    "n": int(use.sum()), "concordancia_maxima_simultanea": int((use & strong).sum()),
                                    "sem_conflito_cardinalidade": int((use & ~conflict).sum()),
                                    "ambos": int((use & strong & ~conflict).sum())})
    return ranks, positives, policies, pd.DataFrame(identifier_rows), maxima, fingerprints


def turning_points(table):
    """Conditional first grid crossing, never interpolation or causal inference."""
    definitions = {
        "C1_VPP_menor_50": ("vpp_llm_abaixo", .5, "lt"),
        "C2_perdidos_13_27": ("perdidos", None, "eq"),
        "C3_O3_margem25_meta64": ("O3-O1_margem25_64", 0., "le"),
        "C4_F4_O1_meta25": ("F4-O1_revisoes_25", 0., "lt"),
        "C4_F4_O1_meta100": ("F4-O1_revisoes_100", 0., "lt"),
    }
    rows = []
    for mode in ("vigilancia", "confirmacao"):
        subset = table.loc[table.modo.eq(mode)]
        for conclusion, (metric, threshold, op) in definitions.items():
            data = subset.loc[subset.metrica.eq(metric)].copy()
            if data.empty:
                continue
            limit = (13 if mode == "vigilancia" else 27) if threshold is None else threshold
            data["holds"] = {"lt": data.mediana < limit, "le": data.mediana <= limit,
                             "eq": np.isclose(data.mediana, limit)}[op]
            for axis in ("fp_abaixo", "fp_acima", "par2_nao_vinculo"):
                fixed = [c for c in ("fp_abaixo", "fp_acima", "par2_nao_vinculo") if c != axis]
                for keys, group in data.groupby(fixed, sort=True):
                    group = group.sort_values(axis)
                    start = group.iloc[0]
                    changed = group.loc[group.holds.ne(start.holds)]
                    row = {"modo": mode, "conclusao": conclusion, "eixo": axis,
                           **dict(zip(fixed, keys, strict=True)), axis: "variavel",
                           "conclusao_no_inicio": bool(start.holds),
                           "fracao_virada": float(changed.iloc[0][axis]) if len(changed) else None,
                           "status": "primeira_mudanca_na_grade" if len(changed) else "sem_virada_na_grade",
                           "fracao_anterior": float(group.loc[group[axis] < changed.iloc[0][axis], axis].max()) if len(changed) else None}
                    rows.append(row)
            # Joint FP fraction with the two regions varying together, conditional on b.
            for b, group in data.loc[data.fp_abaixo.eq(data.fp_acima)].groupby("par2_nao_vinculo"):
                group = group.sort_values("fp_abaixo")
                changed = group.loc[group.holds.ne(group.iloc[0].holds)]
                rows.append({"modo": mode, "conclusao": conclusion, "eixo": "fp_uniforme",
                             "fp_abaixo": "variavel", "fp_acima": "variavel", "par2_nao_vinculo": b,
                             "conclusao_no_inicio": bool(group.iloc[0].holds),
                             "fracao_virada": float(changed.iloc[0].fp_abaixo) if len(changed) else None,
                             "status": "primeira_mudanca_na_grade" if len(changed) else "sem_virada_na_grade",
                             "fracao_anterior": float(group.loc[group.fp_abaixo < changed.iloc[0].fp_abaixo, "fp_abaixo"].max()) if len(changed) else None})
    return pd.DataFrame(rows)


def write_report(table, turns, identifiers):
    lines = ["# Sensibilidade das conclusões ao erro da referência operacional PAR", "",
             "Análise quantitativa de viés, exploratória post hoc, CPU e sem LLM. "
             "Não estima a frequência real de erro nem corrige a referência como se fosse verdade conhecida.", "",
             "## Convenções", "",
             "Frações dos FP abaixo/acima do limiar **por passo** variam independentemente "
             "em 0/25/50/75/100%; remoção uniforme entre os 36 PAR=2: 0/5/10/20%. "
             "Todos esses FP estão abaixo do corte operacional ORL; não confundir os dois limiares. "
             "Confirmação: zero FP abaixo do limiar por passo, quatro acima. "
             "Aceitos significa VP automáticos; sensibilidade automática não conta encaminhamentos. "
             "Também se publica sensibilidade com revisor perfeito.", "",
             "Ordenações O1/O3 e RRF(60) F4 permanecem fixas; não há retreino sob os novos rótulos. "
             "1.000 sorteios por cenário, seed 20261107. Número de trocas: n×fração com "
             "arredondamento estocástico não enviesado; subconjuntos uniformes sem reposição. "
             "Mediana e P2,5–P97,5 são dispersão de cenários, não IC inferencial. "
             "Diferenças são pareadas por sorteio. Metas 25/64/100%: inteiro mais próximo "
             "(metades para cima), reproduzindo 9/23/36 na referência (23/36=63,89%, "
             "rotulado 64% no estudo); denominador recalculado em cada sorteio. "
             "Para evitar que arredondamento seja confundido com atingir literalmente a fração, "
             "as métricas adicionais `revisoes_minimo_25/64/100` usam teto(n×fração): "
             "na referência exigem 9/24/36. As viradas C3/C4 mantêm a convenção nominal "
             "do estudo; 9 e 36 são metas de referência, não contagens absolutas fixas "
             "após alteração do total de vínculos.", "",
             "## Viradas condicionais (outros eixos em zero)", "",
             "| Modo | Conclusão operacional | Eixo | Válida no início? | Primeira mudança na grade |",
             "|---|---|---|---|---|" ]
    for row in turns.itertuples():
        fixed_columns = [c for c in ("fp_abaixo", "fp_acima", "par2_nao_vinculo")
                         if c != row.eixo and not (row.eixo == "fp_uniforme" and c.startswith("fp_"))]
        if any(str(getattr(row, c)) not in ("0.0", "0") for c in fixed_columns):
            continue
        crossing = "não observada" if pd.isna(row.fracao_virada) else f"{100 * row.fracao_virada:g}%"
        lines.append(f"| {row.modo} | {row.conclusao} | {row.eixo} | {'sim' if row.conclusao_no_inicio else 'não'} | {crossing} |")
    lines += ["", "Todas as superfícies condicionais, inclusive combinações, estão em "
              "vies_referencia_viradas.csv. Virada é mudança do indicador baseado na mediana; "
              "o intervalo completo deve ser consultado no CSV de métricas. Não se interpola entre pontos.",
              "", "## Leitura curta", ""]
    for mode in ("vigilancia", "confirmacao"):
        base = table.loc[table.modo.eq(mode) & table.fp_abaixo.eq(0) & table.fp_acima.eq(0)
                         & table.par2_nao_vinculo.eq(0)].set_index("metrica")
        lines.append(f"- {mode}: VPP inicial {100 * base.loc['vpp_llm_abaixo', 'valor_esperado']:.1f}%; "
                     f"perdidos iniciais {base.loc['perdidos', 'valor_esperado']:g}. "
                     "Reclassificar FP aumenta VP, não recupera vínculos perdidos; retirar PAR=2 "
                     "reduz os perdidos esperados proporcionalmente (hipótese de erro uniforme).")
    lines += ["- C1 é apenas o diagnóstico VPP<50%, não prova de segurança de aceitação autônoma. "
              "Na confirmação ele já é falso no cenário inicial (8/12); não se generaliza a "
              "recomendação da vigilância com esse limiar arbitrário.",
              "- C2 literal (13/27) muda com qualquer fração positiva de PAR=2 removida; "
              "na grade a primeira é 5%. Isso não elimina perdas: a 20% restam 10,4/21,6 esperadas.",
              "- C3 exige redução de pelo menos 25% em O3 frente a O1 na meta nominal de 64%; "
              "C4 exige diferença F4−O1 estritamente negativa, nas metas móveis correspondentes "
              "a 9 e 36 na referência. Não são testes de significância.", "",
              "## Identificadores fortes (somente agregados)", "",
              "Máximo empírico no extrato completo de cada um dos sete subescores: "
              "qtd/primeiro/último fragmento iguais para NOME e NOMEMAE, e DTNASC dt iguais. "
              "Concordância forte requer os sete máximos simultaneamente, sem imputação. "
              "Máximos registrados no metadado. Não equivale à igualdade dos nomes originais "
              "nem à confirmação de vínculo. Ausência de conflito significa que nem COMPREC "
              "nem REFREC constam dos 211 PAR=1 acima do corte (mesmo critério de fp_cardinalidade). "
              "Não avalia conflitos entre candidatos abaixo do corte.", "",
              "| Grupo | n | Máximos simultâneos | Sem conflito | Ambos |",
              "|---|---:|---:|---:|---:|"]
    for row in identifiers.loc[identifiers.regiao_limiar_passo.eq("TOTAL")].itertuples():
        lines.append(f"| {row.modo}/{row.grupo} | {row.n} | {row.concordancia_maxima_simultanea} | "
                     f"{row.sem_conflito_cardinalidade} | {row.ambos} |")
    lines += ["", "Nenhum dos FP, PAR=2 ou dos 1.000 PAR=0 amostrados atinge os sete máximos "
              "simultâneos (todos os máximos são 1). Esse critério estrito não sustenta que "
              "estes FP sejam vínculos fortes ignorados; tampouco comprova que sejam não vínculos. "
              "Ausência de conflito de cardinalidade não confirma vínculo. Viés de verificação dos PAR=2 "
              "pode favorecer O1; a análise não elimina esse viés nem modela erros nos 211 PAR=1.", "",
              "## Reprodução", "",
              "`python scripts/vies_referencia.py --data-root <dados> --policy-dir <politicas>`", "",
              "Testes portáveis em tests/test_vies_referencia.py; entradas privadas opcionais "
              "via tests/local_data.py e PHD_RESEARCH_POLICY_DIR. Metadado registra grade "
              "antes da leitura, hashes, versões e regras. Nenhum identificador ou valor por par é publicado."]
    # Quantitative reading of the full combination grid, not just one-axis paths.
    position = lines.index("## Identificadores fortes (somente agregados)")
    extra = ["## Robustez nas combinações da vigilância", "",
             "| Indicador pareado (negativo favorece a conclusão) | Maior mediana entre os 100 cenários | Maior P97,5 |",
             "|---|---:|---:|"]
    vigilance = table.loc[table.modo.eq("vigilancia")]
    for metric in ("O3-O1_margem25_64", "F4-O1_revisoes_25", "F4-O1_revisoes_100"):
        selected = vigilance.loc[vigilance.metrica.eq(metric)]
        extra.append(f"| {metric} | {selected.mediana.max():.2f} | {selected.p975.max():.2f} |")
    margin = vigilance.loc[vigilance.metrica.eq("O3-O1_margem25_64")]
    extra += ["", f"C3 mantém o sinal pela mediana nos {int((margin.mediana <= 0).sum())}/100 cenários, "
              f"mas P97,5 cruza zero em {int((margin.p975 > 0).sum())}/100: "
              "não se deve traduzir robustez da mediana em garantia em todos os sorteios.", "",
              "C1, cálculo contínuo sob reclassificação uniforme dos 21 FP: VPP=50% em "
              "40,48% dos FP se nenhum PAR=2 for removido (primeiro ponto da grade: 50%). "
              "Se apenas os dez FP abaixo do limiar por passo mudam, são necessários 85%; "
              "se apenas os onze acima mudam, 77,27% (ambos: primeiro ponto 100%). "
              "Com remoção de 20% dos PAR=2, o limiar uniforme é 44,29%. "
              "Essas são soluções algébricas do VPP esperado, não estimativas da prevalência de erro.", ""]
    lines[position:position] = extra
    (ROOT / "docs/plans/RELATORIO_VIES_REFERENCIA_2026-11-07.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def generate(data_root, policy_dir):
    metadata_path = ROOT / "data/vies_referencia_metadata.json"
    metadata = {"analysis": "analise_quantitativa_de_vies_sensibilidade_nao_estimativa_erro_real",
                "registered": False, "cpu_only": True, "llm_calls": 0, "model_fits": 0,
                "fixed_a_priori_for_this_run_utc": datetime.now(UTC).isoformat(),
                "fp_grid": FP_GRID, "par2_grid": PAR_GRID, "seed": SEED, "repetitions": REPETITIONS,
                "grid": "vigilancia: 5x5x4; confirmacao: 1x5x4 (grupo FP abaixo vazio)",
                "rng": "numpy.default_rng PCG64; restart seed each scenario, common random numbers",
                "rounding_changes": "floor(n*f)+Bernoulli(frac(n*f)); uniform subsets without replacement",
                "targets_percent": TARGETS, "targets_rounding": "nearest integer, halves up, min 1; baseline 9/23/36",
                "strict_fraction_metrics": "revisoes_minimo: ceil(n*f), baseline 9/24/36; paired differences also provided",
                "quantiles": "linear median/P2.5/P97.5 of 1000 paired metric draws; not confidence intervals",
                "assumptions": "Uniform error within each group; 211 PAR1 fixed; policies and rankings fixed; no refit. "
                "FP groups disjoint from PAR2; baseline P accepted FP reclassified as links enter all queues. "
                "PAR2 removal applies uniformly to accepted, referred and lost positives.",
                "conclusions": {"C1": "VPP<0.5, diagnostic only; confirmation already >=0.5",
                                "C2": "lost equals 13/27 literally; any b>0 changes expectation",
                                "C3": "median paired O3-0.75*O1 <=0 at target64, surveillance",
                                "C4": "median paired F4-O1 <0 at targets25/100 (baseline9/36)"}}
    write_json(metadata_path, metadata)  # Freeze grid before reading outcomes.
    ranks, positives, policies, identifiers, maxima, fingerprints = load_inputs(data_root, policy_dir)
    rows = []
    for mode, work in policies.items():
        low, high, _ = work["groups"]
        for a_low in (FP_GRID if mode == "vigilancia" else (0.,)):
            for a_high in FP_GRID:
                for b in PAR_GRID:
                    scenario = {"modo": mode, "fp_abaixo": a_low, "fp_acima": a_high, "par2_nao_vinculo": b}
                    for metric, value in expected_metrics(work["base"], len(low), len(high), a_low, a_high, b).items():
                        rows.append({**scenario, "metrica": metric, "valor_esperado": value,
                                     "mediana": value, "p025": value, "p975": value, "tipo": "valor_esperado"})
                    simulation = simulate(ranks, positives, work["groups"], (a_low, a_high, b))
                    for metric, (median, p025, p975) in simulation.items():
                        rows.append({**scenario, "metrica": metric, "valor_esperado": None,
                                     "mediana": median, "p025": p025, "p975": p975, "tipo": "simulacao"})
    table = pd.DataFrame(rows)
    turns = turning_points(table)
    for frame, name in ((table, "vies_referencia.csv"), (turns, "vies_referencia_viradas.csv"),
                        (identifiers, "vies_referencia_identificadores.csv")):
        with (ROOT / "data" / name).open("w", encoding="utf-8", newline="\n") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
    metadata.update({"baseline_reproduction": {"passed": True, "below_cutoff": 61485, "par2": 36,
                                               "checks": ["comparator queue", "O1/O3/F4 reference efforts", "P counts", "FP step regions", "PAR by step", "cardinality", "archived policy hashes"]},
                     "strong_identifier_maxima": maxima,
                     "strong_identifier_criterion": "all seven empirical full-extract maxima simultaneously; no imputation",
                     "cardinality_criterion": "no COMPREC or REFREC shared with 211 accepted PAR1 above cutoff",
                     "par0_sample": {"n": 1000, "seed": SEED, "sampling": "uniform without replacement, lexical hash order"},
                     "input_sha256": fingerprints,
                     "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                     "C1_uniform_analytic_threshold": {mode: (0.5 * (float(work["base"].aceitos) - 211 + float(work["base"].fp))
                                                                               - (float(work["base"].aceitos) - 211)) / float(work["base"].fp)
                                                        for mode, work in policies.items()}})
    for logical in ("data/arranjo_corte_metricas.csv", "data/fp_arranjo_por_regiao.csv", "data/par_por_passo.csv",
                    "data/fp_cardinalidade.json", "data/fusao_filas.csv", "data/robustez_arranjo_metadata.json"):
        metadata["input_sha256"][logical] = sha256(ROOT / logical)
    for logical in ("data/pcal_ml_oof.csv", "data/pcal_ml_oof.json"):
        metadata["input_sha256"][logical] = sha256(data_root / logical)
    metadata["code_sha256_lf"] = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for name in ("scripts/vies_referencia.py", "scripts/fusao_filas.py", "scripts/comparador_mesma_carga.py", "gzcmd/loader.py")}
    metadata["output_sha256"] = {name: sha256(ROOT / "data" / name) for name in
                                 ("vies_referencia.csv", "vies_referencia_viradas.csv", "vies_referencia_identificadores.csv")}
    write_json(metadata_path, metadata)
    write_report(table, turns, identifiers)
    print(json.dumps({"scenarios": int(table[["modo", "fp_abaixo", "fp_acima", "par2_nao_vinculo"]].drop_duplicates().shape[0]),
                      "metric_rows": len(table), "turn_rows": len(turns), "baseline": metadata["baseline_reproduction"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--policy-dir", type=Path, default=ROOT / ".local/arranjo_corte")
    args = parser.parse_args()
    generate(args.data_root, args.policy_dir)
