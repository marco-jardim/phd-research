"""Análise agregada do corte OpenRecLink seguido da cascata REFREC existente.

Não recalibra a política e nunca grava identificadores ou dados por par.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import tempfile
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODES = ("vigilancia", "confirmacao")
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from gzcmd.llm_usage_cost import PRICES_USD_PER_MILLION, selected_cost
from scripts.arranjo_paths import portable, relative_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.strip().str.replace(",", ".", regex=False), errors="raise")


def load_extract(path: Path) -> pd.DataFrame:
    columns = {"PAR", "PASSO", "COMPREC", "REFREC", "SCORE", "nota final"}
    frame = pd.read_csv(path, sep=";", dtype=str,
                        usecols=lambda name: name.split(",")[0] in columns)
    frame.columns = [name.split(",")[0] for name in frame.columns]
    require(set(frame.columns) == columns, "Campos necessários ausentes no extrato")
    require(not frame.isna().any().any(), "Campos necessários ausentes em algum potencial par")
    for name in ("PAR", "SCORE", "nota final"):
        frame[name] = numeric(frame[name])
    frame["pair_id"] = [
        hashlib.sha256("|".join(value.strip() for value in fields).encode()).hexdigest()[:16]
        for fields in frame[["COMPREC", "REFREC", "PASSO"]].itertuples(index=False, name=None)
    ]
    require(not frame.pair_id.duplicated().any(), "pair_id duplicado no extrato")
    require(frame.PAR.isin([0, 1, 2]).all(), "PAR fora do domínio esperado")
    require(frame.PAR.value_counts().to_dict() == {0: 61449, 1: 211, 2: 36},
            "Totais PAR não reproduzem os valores de referência")
    return frame.drop(columns=["COMPREC", "REFREC"])


def wilson(successes: int, total: int) -> tuple[float, float]:
    require(total > 0, "Denominador vazio para Wilson")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def summarize(frame: pd.DataFrame, mode: str, arrangement: str,
              scope: str, costs: dict[str, float]) -> dict:
    positive = frame.PAR.isin([1, 2])
    above = frame.acima_corte
    cascade = frame.final.eq("MATCH")
    human = frame.final.isin(["UNSURE", "DEFERRED"])
    llm = frame.action.eq("LLM_REVIEW")
    rejected = frame.final.eq("NONMATCH")
    empty = pd.Series(False, index=frame.index)
    if arrangement == "corte_sozinho":
        accepted, human, llm, rejected = above, empty, empty, ~above
    elif arrangement == "corte_mais_cascata":
        accepted = above | (~above & cascade)
        human, llm, rejected = ~above & human, ~above & llm, ~above & rejected
    elif arrangement == "corte_mais_humano_integral":
        accepted, human, llm, rejected = above, ~above, empty, empty
    elif arrangement == "cascata_sozinha":
        accepted = cascade
    else:
        raise ValueError("Arranjo desconhecido")
    require((accepted.astype(int) + human.astype(int) + rejected.astype(int)).eq(1).all(),
            "Decisões finais não formam partição")
    region = pd.Series(True, index=frame.index) if scope == "TODAS" else frame["nota final"].ge(5) & frame["nota final"].lt(9)
    total = int((region & positive).sum())
    automatic = int((region & positive & accepted).sum())
    human_positive = int((region & positive & human).sum())
    lost = int((region & positive & rejected).sum())
    require(automatic + human_positive + lost == total, "Vínculos não formam partição")
    low, high = wilson(automatic, total)
    return {
        "modo": mode, "arranjo": arrangement, "faixa": scope,
        "potenciais_pares": int(region.sum()), "vinculos": total,
        "vinculos_automaticos": automatic, "sensibilidade_automatica": automatic / total,
        "wilson95_inferior": low, "wilson95_superior": high,
        "falsos_positivos_automaticos": int((region & ~positive & accepted).sum()),
        "falsos_positivos_abaixo_corte": int((region & ~positive & accepted & ~above).sum()),
        "vinculos_rejeitados_automaticamente": lost,
        "vinculos_nao_aceitos_automaticamente": total - automatic,
        "pares_ao_humano": int((region & human).sum()), "vinculos_ao_humano": human_positive,
        "pares_ao_llm": int((region & llm).sum()),
        "custo_llm_usd": sum(costs[pid] for pid in frame.loc[region & llm, "pair_id"]),
        "sensibilidade_com_humano_erro_05": (automatic + .95 * human_positive) / total,
        "sensibilidade_com_humano_perfeito": (automatic + human_positive) / total,
    }


def execute(out: Path, data_root: Path = ROOT) -> pd.DataFrame:
    data = data_root / "data"
    extract_path = data / "COMPARADORSEMIDENT.csv"
    metrics_path = data / "funil_gzcmd_final_metricas.csv"
    final_path = data / "funil_gzcmd_final.csv"
    metadata_path = data / "funil_gzcmd_final_metadata.json"
    frame = load_extract(extract_path)
    cutoffs = frame.loc[frame.PAR.eq(1)].groupby("PASSO").SCORE.min()
    require(frame.PASSO.isin(cutoffs.index).all(), "Passo sem corte identificável entre os PAR=1")
    frame["acima_corte"] = frame.SCORE.ge(frame.PASSO.map(cutoffs))
    require(frame.acima_corte.equals(frame.PAR.eq(1)), "Corte não separa exatamente PAR=1")
    require(not (frame.acima_corte & frame.PAR.eq(0)).any(), "Não pares acima do corte")
    metrics = pd.read_csv(metrics_path)
    final = pd.read_csv(final_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["pricing_usd_per_million_tokens"] == PRICES_USD_PER_MILLION,
            "Tarifa do funil diverge da função de custo")
    sources = [extract_path, metrics_path, final_path, metadata_path, Path(__file__),
               ROOT / "gzcmd/llm_usage_cost.py", ROOT / "gzcmd/final_funnel.py",
               ROOT / "gzcmd/llm_dossier.py"]
    rows, audit, journal_sources, provenance_warnings = [], {}, [], []
    expected = {"vigilancia": (194, 34, 111, 347), "confirmacao": (132, 0, 127, 44)}
    for mode in MODES:
        funnel_path = data / f"funil_gzcmd_{mode}.csv"
        sources.append(funnel_path)
        funnel = pd.read_csv(funnel_path, dtype={"pair_id": str},
                             usecols=["pair_id", "nota", "y", "action", "final"])
        require(not funnel.pair_id.duplicated().any(), "pair_id duplicado no funil")
        require(set(frame.pair_id) == set(funnel.pair_id), "Universos distintos no extrato e funil")
        merged = frame.merge(funnel, on="pair_id", validate="one_to_one")
        require(merged.y.eq(merged.PAR.isin([1, 2]).astype(int)).all(), "Gabaritos distintos")
        require((merged.nota - merged["nota final"]).abs().le(1e-10).all(), "Notas distintas")
        require(merged.final.isin(["MATCH", "NONMATCH", "UNSURE", "DEFERRED"]).all(),
                "Decisão ausente ou desconhecida")
        reference = metrics.loc[metrics.modo.eq(mode) & metrics.versao.eq("v3.1")]
        require(len(reference) == 1, "Métrica final v3.1 ausente ou duplicada")
        reference = reference.iloc[0]
        positive, accepted = merged.PAR.isin([1, 2]), merged.final.eq("MATCH")
        human = merged.final.isin(["UNSURE", "DEFERRED"])
        llm = merged.action.eq("LLM_REVIEW")
        observed = (int((positive & accepted).sum()), int((~positive & accepted).sum()),
                    int(human.sum()), int(llm.sum()))
        require(observed == expected[mode], f"PARADA: cascata sozinha não reproduz referência em {mode}")
        for key, count in zip(("VP_cadeia", "FP_cadeia", "pares_ao_humano", "pares_ao_llm"), observed):
            require(int(reference[key]) == count, f"{key} diverge das métricas finais")
        for key, count in {
            "FN_cadeia": int((positive & ~accepted).sum()),
            "VN_cadeia": int((~positive & ~accepted).sum()),
            "FN_nonmatch_observado": int((positive & merged.final.eq("NONMATCH")).sum()),
            "vinculos_ao_humano": int((positive & human).sum()),
        }.items():
            require(int(reference[key]) == count, f"{key} diverge das métricas finais")
        require(math.isclose(observed[0] / 247, reference.sens_cadeia_observada, abs_tol=1e-12),
                "Sensibilidade diverge das métricas finais")
        source_group = [entry for entry in metadata["sources"]
                        if entry["modo"] == mode and entry["versao"] == "v3.1"]
        require(len(source_group) == 1, "Metadados de custo ausentes ou duplicados")
        recorded_sources = source_group[0]["sources"]
        paths = [Path(entry["source"]) for entry in recorded_sources]
        costs, cost_sources = selected_cost(paths, set(merged.loc[llm, "pair_id"]))
        for recorded, actual in zip(recorded_sources, cost_sources):
            if recorded["sha256"] != actual["sha256"]:
                provenance_warnings.append({
                    "modo": mode, "source": actual["source"],
                    "aviso": "JSON do run mudou desde o funil. Journals dos selecionados e custo são conferidos separadamente.",
                    "sha256_no_funil": recorded["sha256"], "sha256_atual": actual["sha256"],
                })
            for key in ("selected_journal_sha256", "pares_selecionados"):
                require(recorded[key] == actual[key], f"Origem/journal do custo diverge: {key}")
        require(math.isclose(sum(costs.values()), reference.custo_llm_usd, abs_tol=1e-10),
                "Custo calculado não reproduz funil final")
        sources.extend(paths)
        journal_sources.append({"modo": mode, "sources": cost_sources})
        for scope in ("TODAS", "[5,9)"):
            for arrangement in ("corte_sozinho", "cascata_sozinha", "corte_mais_cascata", "corte_mais_humano_integral"):
                rows.append(summarize(merged, mode, arrangement, scope, costs))
            band = final.loc[final.modo.eq(mode) & final.versao.eq("v3.1") & final.faixa.eq(scope)]
            if scope == "[5,9)":
                for stage, observed_count in (("HUMANO_TOTAL", int((human & merged.nota.ge(5) & merged.nota.lt(9)).sum())),
                                              ("LLM", int((llm & merged.nota.ge(5) & merged.nota.lt(9)).sum()))):
                    require(int(band.loc[band.etapa.eq(stage), "n"].sum()) == observed_count,
                            "Agregado por faixa diverge do funil final")
        alone = next(row for row in rows if row["modo"] == mode and row["faixa"] == "TODAS"
                     and row["arranjo"] == "cascata_sozinha")
        require(math.isclose(alone["sensibilidade_com_humano_erro_05"],
                             reference.sens_com_humano_erro_05, abs_tol=1e-12),
                "Cenário de erro humano não reproduz a tese")
        audit[mode] = {"referencia_reproduzida": True, "VP_FP_humano_LLM": list(observed)}
    table = pd.DataFrame(rows)
    notes = [
        "PAR=1 é definido pelo corte OpenRecLink: precisão de 100% por construção, com viés de verificação. Falsos positivos da cascata são medidos contra esse rótulo.",
        "Política REFREC v3.1 calibrada sobre toda a base e mantida sem recalibração: somente filtragem abaixo do corte. Um arranjo real reotimizaria os custos.",
        "Automático significa final=MATCH, inclusive MATCH do LLM. Humano significa final em UNSURE/DEFERRED. Rejeição automática significa NONMATCH, não todos os não aceitos.",
        "LLM e humano não são categorias disjuntas: UNSURE do LLM também segue ao humano. Custos estimados com tokens registrados e tarifa histórica, não fatura, sem novas chamadas.",
        "Sensibilidade com humano = (vínculos automáticos + 0,95 vezes vínculos ao humano) / vínculos. Cenário hipotético, não resultado humano observado.",
        "Referência operacional observada: 211 + 36 = 247. Com erro humano hipotético de 5%, não é 100%. Revisão integral abaixo do corte é uma referência hipotética de carga: limite inferior efetivamente revisado não documentado.",
        "Área cinza definida exclusivamente pela nota final do CdR em [5,9), não pelo SCORE do OpenRecLink. Wilson 95% descreve proporção condicional a este gabarito e não corrige viés de verificação.",
        "Nenhuma identificação ou linha de potencial par é gravada. Análise determinística sem amostragem ou ajuste de modelos, portanto sem necessidade de seed.",
    ]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "arranjo.csv", index=False)
    meta = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "script": relative_path(__file__, ROOT), "particionamento": "REFREC",
        "python": sys.version, "pandas": pd.__version__,
        "cortes_SCORE_por_PASSO": {str(key): float(value) for key, value in cutoffs.items()},
        "nao_pares_acima_corte": 0, "nao_pares_total": 61449,
        "conferencias": audit, "notas_metodologicas": notes,
        "fontes": [{"caminho": relative_path(path, ROOT, data_root), "sha256": sha256(path)} for path in dict.fromkeys(sources)],
        "custos_journals": portable(journal_sources, ROOT, data_root), "tarifa_usd_por_milhao_tokens": PRICES_USD_PER_MILLION,
        "avisos_proveniencia": portable(provenance_warnings, ROOT, data_root),
        "saida": "arranjo.csv",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CONFERÊNCIA REFREC:", json.dumps(audit, ensure_ascii=False))
    print("CORTES SCORE POR PASSO:", meta["cortes_SCORE_por_PASSO"])
    print("NÃO PARES ACIMA DO CORTE: 0 de 61449")
    print("AVISOS DE PROVENIÊNCIA:", json.dumps(provenance_warnings, ensure_ascii=False))
    print(table.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print("NOTAS METODOLÓGICAS:")
    for note in notes:
        print(note)
    print("ARQUIVOS:", out / "arranjo.csv", out / "meta.json")
    return table


def self_test() -> None:
    low, high = wilson(50, 100)
    require(math.isclose(low, .4038315303659956, abs_tol=1e-12), "Wilson inferior incorreto")
    require(math.isclose(high, .5961684696340044, abs_tol=1e-12), "Wilson superior incorreto")
    synthetic = pd.DataFrame({
        "PAR": [1, 1, 2, 2, 2, 0, 0], "acima_corte": [True, True, False, False, False, False, False],
        "final": ["NONMATCH", "DEFERRED", "MATCH", "UNSURE", "NONMATCH", "MATCH", "NONMATCH"],
        "action": ["NONMATCH", "DEFERRED", "LLM_REVIEW", "LLM_REVIEW", "NONMATCH", "MATCH", "NONMATCH"],
        "nota final": [6, 9, 7, 8, 4, 6, 5], "pair_id": list("abcdefg"),
    })
    costs = {"c": .002, "d": .003}
    row = summarize(synthetic, "teste", "corte_mais_cascata", "TODAS", costs)
    require((row["vinculos_automaticos"], row["pares_ao_humano"], row["pares_ao_llm"],
             row["vinculos_rejeitados_automaticamente"], row["falsos_positivos_automaticos"]) == (3, 1, 2, 1, 1),
            "Contagens sequenciais incorretas")
    require(math.isclose(row["sensibilidade_com_humano_erro_05"], .79), "Erro humano incorreto")
    for arrangement in ("corte_sozinho", "cascata_sozinha", "corte_mais_cascata", "corte_mais_humano_integral"):
        for scope in ("TODAS", "[5,9)"):
            summarize(synthetic, "teste", arrangement, scope, costs)
    print("SELF-TEST OK: Wilson, partições em quatro arranjos e duas faixas, filtro sequencial, erro humano e custos.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "arranjo")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    execute(args.out)


if __name__ == "__main__":
    main()
