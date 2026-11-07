"""Exploratory post-hoc person-ID propagation; CPU, historical inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.loader import LoadConfig, load_comparador_csv
from gzcmd.ml_oof import pair_ids
from scripts.fp_cardinalidade import counts
from scripts.vies_referencia import load_inputs, load_raw, numeric

PERSON_FIELDS = ("C_IDPESSOA", "C_NOVOIDPESSOA")


def clean(series):
    """Literal identifiers: trim only; never numeric conversion or imputation."""
    return series.astype("string").fillna("").str.strip()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def propagation(raw, above, fields=PERSON_FIELDS):
    """One hop; union is OR of same-column equalities, not a cross-ID merge."""
    seeds = raw.loc[above]
    result = {"R2_REFREC": raw.REFREC.isin(set(seeds.REFREC)).to_numpy()}
    union = np.zeros(len(raw), dtype=bool)
    for field in fields:
        lookup = seeds.loc[seeds[field].ne("")].groupby(["REFREC", field]).COMPREC.agg(set)
        mapping = lookup.to_dict()
        mask = np.array([
            bool(person) and bool(set(mapping.get((ref, person), ())) - {notification})
            for ref, person, notification in zip(raw.REFREC, raw[field], raw.COMPREC, strict=True)
        ], dtype=bool)
        result["R1_" + field] = mask
        union |= mask
    result["R1_UNIAO"] = union
    return {name: mask & ~np.asarray(above) for name, mask in result.items()}


def identifier_diagnostics(raw, above, fields=PERSON_FIELDS):
    # COMPREC is reused across blocking steps. Scope its local row number;
    # sparse new-ID values can be absent in other candidate rows of that record.
    keys = ["PASSO", "COMPREC"]
    notifications = raw[keys].drop_duplicates()
    accepted = raw.loc[above, ["REFREC", "COMPREC", *fields]].drop_duplicates()
    output = {}
    for field in fields:
        available = raw.loc[raw[field].ne(""), [*keys, field]].drop_duplicates()
        if available.duplicated(keys).any():
            raise ValueError("Person fields vary within a notification")
        sizes = available.groupby(field).size()
        multi = accepted.groupby("REFREC").filter(lambda g: g.COMPREC.nunique() > 1)
        ref_groups = list(multi.groupby("REFREC"))
        complete = [g for _, g in ref_groups if g[field].ne("").all()]
        consistent = sum(g[field].nunique() == 1 for g in complete)
        persons = accepted.loc[accepted[field].ne("")].groupby(field)
        person_groups = [g for _, g in persons]
        repeated = [g for g in person_groups if g.COMPREC.nunique() > 1]
        output[field] = {
            "pares_total": len(raw), "pares_nao_vazios": int(raw[field].ne("").sum()),
            "completude_pares_pct": 100 * raw[field].ne("").mean(),
            "notificacoes_total": len(notifications), "notificacoes_nao_vazias": len(available),
            "completude_notificacoes_pct": 100 * len(available) / len(notifications),
            "pessoas_distintas": len(sizes),
            "notificacoes_por_pessoa_media": float(sizes.mean()) if len(sizes) else None,
            "notificacoes_por_pessoa_max": int(sizes.max()) if len(sizes) else None,
            "histograma_notificacoes_por_pessoa": {
                str(int(n)): int(total) for n, total in sizes.value_counts().sort_index().items()},
            "refrec_multiplos_comprec": len(ref_groups),
            "refrec_multiplos_id_completo": len(complete),
            "refrec_multiplos_id_unico": consistent,
            "proporcao_refrec_multiplos_id_unico": ratio(consistent, len(ref_groups)),
            "proporcao_refrec_multiplos_id_unico_completos": ratio(consistent, len(complete)),
            "pessoas_com_vinculo_aceito": len(person_groups),
            "pessoas_aceitas_refrec_unico": sum(g.REFREC.nunique() == 1 for g in person_groups),
            "proporcao_pessoas_aceitas_refrec_unico": ratio(
                sum(g.REFREC.nunique() == 1 for g in person_groups), len(person_groups)),
            "pessoas_multiplos_comprec_aceitos": len(repeated),
            "pessoas_multiplos_comprec_refrec_unico": sum(g.REFREC.nunique() == 1 for g in repeated),
            "proporcao_pessoas_multiplos_comprec_refrec_unico": ratio(
                sum(g.REFREC.nunique() == 1 for g in repeated), len(repeated)),
        }
        if "C_IDLINHA" in raw:
            physical = raw.loc[raw[field].ne(""), ["C_IDLINHA", field]].drop_duplicates()
            ambiguous = physical.groupby("C_IDLINHA")[field].nunique().gt(1)
            unambiguous = physical.loc[~physical.C_IDLINHA.isin(ambiguous.index[ambiguous])]
            physical_sizes = unambiguous.groupby(field).C_IDLINHA.nunique()
            output[field]["por_C_IDLINHA"] = {
                "notificacoes_total": int(raw.C_IDLINHA.nunique()),
                "notificacoes_nao_vazias": int(physical.C_IDLINHA.nunique()),
                "completude_pct": 100 * physical.C_IDLINHA.nunique() / raw.C_IDLINHA.nunique(),
                "notificacoes_id_ambiguo_excluidas_cardinalidade": int(ambiguous.sum()),
                "pessoas_distintas_sem_ambiguidade": len(physical_sizes),
                "notificacoes_por_pessoa_media": float(physical_sizes.mean()) if len(physical_sizes) else None,
                "histograma_notificacoes_por_pessoa": {
                    str(int(n)): int(total) for n, total in physical_sizes.value_counts().sort_index().items()},
            }
    return output


def summarize(raw, below, masks, fp_masks):
    rows = []
    labels, steps = numeric(raw.PAR).to_numpy(), numeric(raw.PASSO).to_numpy()
    for rule, propagated in masks.items():
        for step in ("TOTAL", 6, 9, 11):
            region = below if step == "TOTAL" else below & (steps == step)
            selected = propagated & region
            n = int(selected.sum())
            positive = int((selected & (labels == 2)).sum())
            rows.append({"regra": rule, "passo": str(step), "pares_abaixo": int(region.sum()),
                         "par2_disponiveis": int((region & (labels == 2)).sum()),
                         "propagados": n, "par2": positive,
                         "par0": int((selected & (labels == 0)).sum()),
                         "precisao_PAR": ratio(positive, n),
                         "recuperacao_PAR2": ratio(positive, int((region & (labels == 2)).sum())),
                         **{f"fp_P_{mode}_propagados": int((selected & fp).sum())
                            for mode, fp in fp_masks.items()},
                         **{f"fp_P_{mode}_total": int((region & fp).sum())
                            for mode, fp in fp_masks.items()}})
    return pd.DataFrame(rows)


def analyze(data_root, policy_dir):
    source = data_root / "data/COMPARADORSEMIDENT.csv"
    raw = load_raw(source)  # Identity-preserving project loader, as in vies_referencia.
    header = pd.read_csv(source, sep=";", nrows=0).columns.tolist()
    fields = tuple(name for name in raw if name.startswith("C_") and "IDPESSOA" in name)
    if set(fields) != set(PERSON_FIELDS):
        raise ValueError("Unexpected person-ID schema; review before analysis")
    engineered = load_comparador_csv(source, cfg=LoadConfig(macd_enabled=False))
    if len(engineered) != len(raw):
        raise ValueError("Project loader row alignment changed")
    for field in ("COMPREC", "REFREC"):
        if not numeric(raw[field]).equals(numeric(engineered[field])):
            raise ValueError("Project loader identity alignment changed")
    for field in ("COMPREC", "REFREC", *fields):
        raw[field] = clean(raw[field])
    if raw[["COMPREC", "REFREC"]].eq("").any().any():
        raise ValueError("Missing record identifier")
    below = below_cutoff(raw)
    above = ~below & numeric(raw.PAR).eq(1).to_numpy()
    ids = pair_ids(raw).to_numpy()
    if len(set(ids)) != len(raw):
        raise ValueError("Duplicate pair identities")
    # Reuse the historical, hash-validated P reconstruction rather than copying it.
    _, _, policies, _, _, fingerprints = load_inputs(data_root, policy_dir)
    lexical = np.sort(ids[below])
    fp_masks = {}
    cardinality = json.loads((ROOT / "data/fp_cardinalidade.json").read_text(encoding="utf-8"))
    for mode, expected in (("vigilancia", 21), ("confirmacao", 4)):
        fp_masks[mode] = pd.Series(policies[mode]["fp"], index=lexical).reindex(ids, fill_value=False).to_numpy()
        if int(fp_masks[mode].sum()) != expected:
            raise ValueError("P false positives do not reproduce 21/4")
        observed = counts(raw.loc[fp_masks[mode]], raw.loc[above])
        if observed != cardinality["modos"][mode]:
            raise ValueError("Imported cardinality diagnostic mismatch")
    masks = propagation(raw, above, fields)
    table = summarize(raw, below, masks, fp_masks)
    metadata = {
        "analysis": "A_exploratoria_post_hoc_propagacao_pessoa_Sinan", "registered": False,
        "cpu_only": True, "llm_calls": 0, "model_fits": 0,
        "person_headers": [name for name in header if name.split(",")[0] in fields],
        "columns_used": ["COMPREC", "REFREC", "PAR", "PASSO", "SCORE", "C_IDLINHA", "R_IDLINHA", *fields],
        "normalization": "strip; preserve literal string/leading zeroes; empty/NA missing; no imputation; zero is nonempty",
        "cardinality_universe": "unique (PASSO, COMPREC) local notification records in candidate extract; not unique people or the entire Sinan population",
        "identifier_scope_audit": {
            "COMPREC_distintos": int(raw.COMPREC.nunique()),
            "PASSO_COMPREC_distintos": len(raw[["PASSO", "COMPREC"]].drop_duplicates()),
            "C_IDLINHA_distintos": int(raw.C_IDLINHA.nunique()),
            "COMPREC_com_multiplos_C_IDLINHA": int(raw.groupby("COMPREC").C_IDLINHA.nunique().gt(1).sum()),
            "REFREC_com_multiplos_R_IDLINHA": int(raw.groupby("REFREC").R_IDLINHA.nunique().gt(1).sum()),
            "warning": "COMPREC reused across steps; cardinality scoped by PASSO. R1/R2 keep literal REFREC/COMPREC as requested; R1 requires different literal COMPREC. Not external ID validation."},
        "reliability_universe": "PAR=1 above ORL cutoff; distinct REFREC/COMPREC; missing IDs cannot count as agreement",
        "rules": {"R1": "one-hop from PAR1 above cutoff; same REFREC, same nonempty ID in same column, distinct COMPREC",
                  "R1_UNIAO": "OR of R1 columns; no cross-column equality, transitive closure or fallback filling",
                  "R2": "same REFREC as any PAR1 above cutoff, no person condition",
                  "precision": "PAR2 / all propagated below cutoff; operational reference, not adjudicated truth; null if none",
                  "blocking": "destination pair PASSO; seeds from all steps"},
        "baseline": {"above_PAR1": int(above.sum()), "below": int(below.sum()),
                     "below_PAR2": int((below & numeric(raw.PAR).eq(2)).sum()),
                     "P_FP": {mode: int(fp.sum()) for mode, fp in fp_masks.items()}},
        "identifier_diagnostics": identifier_diagnostics(raw, above, fields),
        "input_sha256": fingerprints,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256_lf": {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                           for name in ("scripts/propagacao_pessoa.py", "scripts/vies_referencia.py",
                                        "scripts/fp_cardinalidade.py", "scripts/comparador_mesma_carga.py",
                                        "scripts/arranjo_corte_s2_suplementar.py", "gzcmd/loader.py",
                                        "gzcmd/arranjo_corte.py", "gzcmd/ml_oof.py")},
    }
    metadata["input_sha256"]["data/COMPARADORSEMIDENT.csv"] = sha256(source)
    for name in ("data/fp_cardinalidade.json", "data/robustez_arranjo_metadata.json",
                 "data/arranjo_corte_metricas.csv", "data/fp_arranjo_por_regiao.csv",
                 "data/par_por_passo.csv", "data/fusao_filas.csv"):
        metadata["input_sha256"][name] = sha256(ROOT / name)
    return table, metadata


def report(table, metadata):
    totals = table.loc[table.passo.eq("TOTAL")].set_index("regra")
    union, contrast = totals.loc["R1_UNIAO"], totals.loc["R2_REFREC"]
    precision = "indefinida (nenhum propagado)" if pd.isna(union.precisao_PAR) else f"{100 * union.precisao_PAR:.2f}%"
    old = metadata["identifier_diagnostics"]["C_IDPESSOA"]
    lines = ["# A — Propagação pelo identificador de pessoa do Sinan", "", "## Leitura em cinco linhas", "",
             f"1. R1 união recupera {int(union.par2)}/36 PAR=2 em {int(union.propagados)} pares: isso não autoriza dispensar revisão clínica/documental.",
             f"2. Precisão contra PAR: {precision}; PAR=0 propagados: {int(union.par0)}. Referência operacional não é verdade adjudicada.",
             f"3. Sinal nos FP de P: {int(union.fp_P_vigilancia_propagados)}/21 (vigilância), {int(union.fp_P_confirmacao_propagados)}/4 (confirmação); concordância por ID não prova vínculo não examinado.",
             f"4. R2 sem pessoa propaga {int(contrast.propagados)} pares, {int(contrast.par2)} PAR=2 e {int(contrast.par0)} PAR=0: compartilhar REFREC não basta.",
             f"5. C_IDPESSOA é completo; {old['refrec_multiplos_id_unico']}/{old['refrec_multiplos_comprec']} REFREC múltiplos compartilham ID (72,22%) e {old['pessoas_multiplos_comprec_refrec_unico']}/{old['pessoas_multiplos_comprec_aceitos']} pessoas com múltiplos aceitos compartilham REFREC; C_NOVOIDPESSOA tem só oito valores presentes. Confiabilidade externa não demonstrada.",
             "", "## Método e limites", "",
             "Exploratória post hoc, CPU, sem ajuste de modelos, execução de política ou chamadas LLM. "
             "Somente vínculos PAR=1 acima dos cortes ORL são sementes; propagação em um salto, para outro COMPREC. "
             "União significa OR entre igualdades dentro da mesma coluna, sem cruzar identificadores antigos/novos, "
             "sem fusão transitiva e sem preencher ausências. Passo refere-se ao candidato de destino (6/9/11); "
             "sementes podem vir de qualquer passo. R2 exige somente REFREC aceito. Precisão indefinida se não há propagação.", "",
             "FP de P reconstruídos importando `vies_referencia.load_inputs`, com políticas arquivadas/hash e respostas "
             "históricas (não novas inferências); conferência adicional importando `fp_cardinalidade.counts`. "
             "Um FP propagado seria apenas sinal compatível com vínculo não examinado, não reclassificação automática.", "",
             "## Cabeçalhos e identificadores (somente agregados)", "",
             *[f"- `{name}`" for name in metadata["person_headers"]], "",
             "Registro local de notificação = (PASSO, COMPREC) único no extrato de candidatos. Completude por par sofre repetição "
             "de registros; a medida por registro e a cardinalidade deduplicam (PASSO, COMPREC). IDs literais, "
             "sem conversão numérica; vazios não concordam. Não se estima validade externa dos identificadores.", "",
             "| Coluna | Registros locais | Não vazios (%) | Pessoas | Média registros/pessoa | Máximo |",
             "|---|---:|---:|---:|---:|---:|"]
    for field, d in metadata["identifier_diagnostics"].items():
        lines.append(f"| {field} | {d['notificacoes_total']} | {d['completude_notificacoes_pct']:.2f} | "
                     f"{d['pessoas_distintas']} | {d['notificacoes_por_pessoa_media']} | {d['notificacoes_por_pessoa_max']} |")
    scope = metadata["identifier_scope_audit"]
    lines += ["", f"**Limite de identidade:** {scope['COMPREC_distintos']} COMPREC literais, "
              f"{scope['PASSO_COMPREC_distintos']} registros (PASSO, COMPREC), "
              f"{scope['C_IDLINHA_distintos']} C_IDLINHA distintos; "
              f"{scope['COMPREC_com_multiplos_C_IDLINHA']} COMPREC apontam para mais de um C_IDLINHA entre passos. "
              "Logo, COMPREC sozinho não permite contar notificações únicas no Sinan. A cardinalidade publicada "
              "conta registros locais por passo, não notificações físicas únicas. R1 mantém a regra solicitada "
              "de COMPREC literal diferente; isso pode excluir números reutilizados entre passos. "
              "Completude por registro considera qualquer valor não vazio consistente da coluna naquele registro, "
              "sem imputá-lo nos pares durante a propagação.",
              "O metadado também publica completude e cardinalidade por C_IDLINHA (linha de origem); "
              "notificações com mais de um ID não vazio são contadas como ambíguas e excluídas somente "
              "dessa cardinalidade, sem escolher um valor arbitrário.",
              "", "### Concordância interna entre PAR=1 acima do corte", "",
              "| Coluna | REFREC com >1 COMPREC | ID completo | Mesmo ID | Pessoas com >1 COMPREC aceito | Mesmo REFREC |",
              "|---|---:|---:|---:|---:|---:|"]
    for field, d in metadata["identifier_diagnostics"].items():
        lines.append(f"| {field} | {d['refrec_multiplos_comprec']} | {d['refrec_multiplos_id_completo']} | "
                     f"{d['refrec_multiplos_id_unico']} | {d['pessoas_multiplos_comprec_aceitos']} | "
                     f"{d['pessoas_multiplos_comprec_refrec_unico']} |")
    lines += ["", "Proporções, completude por par e histograma de notificações por pessoa estão no metadado. "
              "Denominador zero resulta em null, não 100% de confiabilidade. O diagnóstico inverso considera "
              "somente COMPREC com vínculo aceito; não trata notificações sem semente como discordantes.", "",
              "## Propagação por passo", "",
              "| Regra | Passo | Propagados | PAR=2 | PAR=0 | Precisão PAR | FP vig. | FP conf. |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table.itertuples():
        p = "indefinida" if pd.isna(row.precisao_PAR) else f"{100 * row.precisao_PAR:.2f}%"
        lines.append(f"| {row.regra} | {row.passo} | {row.propagados} | {row.par2} | {row.par0} | "
                     f"{p} | {row.fp_P_vigilancia_propagados} | {row.fp_P_confirmacao_propagados} |")
    lines += ["", "## Reprodução", "",
              "`python scripts/propagacao_pessoa.py --data-root <dados> --policy-dir <politicas>`", "",
              "`python -m pytest tests/test_propagacao_pessoa.py tests/test_s9c.py -q`", "",
              "Entradas privadas opcionais via `tests/local_data.py` e `PHD_RESEARCH_POLICY_DIR`; "
              "saídas públicas exclusivamente agregadas. Hashes SHA-256 de código (LF), extrato, "
              "políticas e respostas, versões e convenções em `data/propagacao_pessoa_metadata.json`."]
    return "\n".join(lines) + "\n"


def generate(data_root, policy_dir, output_root=ROOT):
    table, metadata = analyze(data_root, policy_dir)
    csv_path = output_root / "data/propagacao_pessoa.csv"
    with csv_path.open("w", encoding="utf-8", newline="\n") as stream:
        table.to_csv(stream, index=False, lineterminator="\n")
    metadata["output_sha256"] = {"data/propagacao_pessoa.csv": sha256(csv_path)}
    (output_root / "data/propagacao_pessoa_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output_root / "docs/plans/RELATORIO_A_PROPAGACAO_2026-11-07.md").write_text(
        report(table, metadata), encoding="utf-8", newline="\n")
    print(table.loc[table.passo.eq("TOTAL")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--policy-dir", type=Path, default=ROOT / ".local/arranjo_corte")
    args = parser.parse_args()
    generate(args.data_root, args.policy_dir)
