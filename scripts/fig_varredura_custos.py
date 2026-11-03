"""Sweep human-equivalent costs offline and plot the two GZ-CMD modes.

python scripts/fig_varredura_custos.py --rates-yaml gzcmd/error_rates_by_band_estimadas.yaml
60 scenarios/mode; reference-reviewer expectations, not simulated responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.bands import BandAssigner
from gzcmd.config import load_config
from gzcmd.cost_evaluation import expected_funnel_metrics
from gzcmd.guardrails import apply_guardrails
from gzcmd.ml_oof import fit_xgb_oof, pair_ids
from gzcmd.loader import parse_date_yyyymmdd, resolve_prefixed_column
from gzcmd.np_evaluation import NP_RATES, compare_confirmation, evaluate_birth_variant, evaluate_np_point
from gzcmd.policy_engine import Budget, Costs, HumanBudget, HumanError, PolicyEngineV3
from gzcmd.reliability import load_band_rates, reference_rates_model

LLM_COSTS = (.005, .05, .25)
HUMAN_ERRORS = (0.0, .02, .05, .10)
HUMAN_BUDGETS = (25, 50, 100, 200, None)
MODES = ("vigilancia", "confirmacao")


def decimal_pt(value, _position=None):
    return f"{value:,.6f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")


def load_candidates(input_csv: Path, oof_csv: Path, cfg) -> pd.DataFrame:
    oof = fit_xgb_oof(input_csv, output_path=oof_csv, reuse=True).set_index("pair_id")
    # Pair metadata/labels and birth dates only for an in-memory year-conflict
    # indicator; never persist raw dates, years, names or addresses.
    raw = pd.read_csv(input_csv, sep=";", dtype=str, low_memory=False,
                      usecols=lambda c: c.split(",")[0] in {"PAR", "COMPREC", "REFREC", "PASSO", "R_DTNASC", "C_DTNASC"})
    labels = pd.Series(pd.to_numeric(raw.PAR, errors="coerce").isin([1, 2]).astype(int).to_numpy(), index=pair_ids(raw))
    if set(oof.index) != set(labels.index):
        raise ValueError("OOF does not match the gold-standard extract")
    frame = pd.DataFrame({"nota final": oof.nota, "p_cal": oof.p_cal_ml, "TARGET": labels})
    birth = [resolve_prefixed_column(raw, name) for name in ("R_DTNASC", "C_DTNASC")]
    if all(col is not None for col in birth):
        year_a, year_b = [parse_date_yyyymmdd(raw[col]).dt.year for col in birth]
        frame["diff_ano"] = pd.Series((year_a - year_b).abs().to_numpy(), index=pair_ids(raw))
    else:
        frame["diff_ano"] = np.nan
    frame["band"] = BandAssigner.from_config(cfg).assign(frame["nota final"])
    if frame.band.isna().any():
        raise ValueError("Unassigned score band")
    frame["guardrail"] = apply_guardrails(frame, policy_version="v3.1").guardrail
    return frame


def sweep(frame: pd.DataFrame, cfg, rates_path: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    rates_model = reference_rates_model(rates_path)
    for mode in MODES:
        mode_cfg = cfg.decision_policy.modes[mode]
        rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode, source=rates_path)
        rows = []
        for llm_cost in LLM_COSTS:
            for error in HUMAN_ERRORS:
                for cap in HUMAN_BUDGETS:
                    human_error = HumanError(error, error)
                    engine = PolicyEngineV3(
                        Costs(mode_cfg.false_positive_cost, mode_cfg.false_negative_cost, llm_cost, 1.0),
                        rates, Budget(mode_cfg.llm_max_calls_per_window), policy_version="v3.1",
                        human_error=human_error, human_budget=HumanBudget(cap),
                    )
                    out = engine.triage(frame)
                    metrics = expected_funnel_metrics(out, rates, human_error)
                    assert metrics["lote_llm"] <= mode_cfg.llm_max_calls_per_window
                    assert cap is None or metrics["humano_direto"] <= cap
                    rows.append({"modo": mode, "C_llm": llm_cost, "C_humano": 1.0,
                                 "C_fp": mode_cfg.false_positive_cost, "C_fn": mode_cfg.false_negative_cost,
                                 "e_fp_h": error, "e_fn_h": error, "orcamento_humano": np.inf if cap is None else cap,
                                 "orcamento_llm": mode_cfg.llm_max_calls_per_window,
                                 "rates_model": rates_model, **metrics})
        tables[mode] = pd.DataFrame(rows)
    return tables


def draw_figure(tables: dict[str, pd.DataFrame], output_base: Path) -> None:
    # fig_cascata_reducao.py uses Agg, default sans-serif, 8.5–11 pt text,
    # subtle grids and hidden top/right spines. Keep that family without a title.
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42}
    colors = ("#222222", "#3b82a0", "#b25c25", "#6a7955")
    x = np.array([25, 50, 100, 200, 400])  # final position is a categorical ∞ proxy
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.2), sharex=True)
        fig.subplots_adjust(left=.11, right=.98, bottom=.14, top=.93, hspace=.34, wspace=.30)
        for col, mode in enumerate(MODES):
            table = tables[mode]
            name = "Vigilância" if mode == "vigilancia" else "Confirmação"
            for color, error in zip(colors, HUMAN_ERRORS):
                main = table[np.isclose(table.C_llm, .005) & np.isclose(table.e_fp_h, error)].sort_values("orcamento_humano")
                axes[0, col].plot(x, main.sens_esperada_com_humano * 100, color=color, marker="o", ms=3, lw=1.4)
                axes[0, col].plot(x, main.sens_esperada_automatica * 100, color=color, ls="--", lw=1.1)
                axes[1, col].plot(x, main.lote_llm, color=color, marker="o", ms=3, lw=1.3)
                for cost, linestyle in ((.05, "-."), (.25, ":")):
                    secondary = table[np.isclose(table.C_llm, cost) & np.isclose(table.e_fp_h, error)].sort_values("orcamento_humano")
                    axes[2, col].plot(x, secondary.sens_esperada_com_humano * 100, color=color, ls=linestyle, marker="o", ms=2.5, lw=1.2)
            axes[0, col].set_ylabel(f"{name}\nSensibilidade esperada (%)\nC$_{{LLM}}$ = 0,005")
            axes[0, col].set_ylim(50, 100)
            axes[1, col].set_ylabel(f"{name}\nLote LLM (pares)\nC$_{{LLM}}$ = 0,005")
            axes[1, col].set_ylim(bottom=0)
            axes[2, col].set_ylabel(f"{name}\nSensibilidade com humano (%)")
            axes[2, col].set_ylim(80, 100)
            axes[2, col].set_xlabel("Orçamento humano direto (pares; log)")
            for ax in axes[:, col]:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_pt))
                ax.set_xscale("log")
                ax.set_xticks(x, ["25", "50", "100", "200", "∞"])
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", lw=.35, alpha=.4)
        error_handles = [Line2D([0], [0], color=color, lw=1.5, label=f"Erro humano: {error:.0%}") for color, error in zip(colors, HUMAN_ERRORS)]
        styles = [Line2D([0], [0], color=".3", lw=1.3, label="Principal: automática + humano"),
                  Line2D([0], [0], color=".3", ls="--", label="Principal: somente automática"),
                  Line2D([0], [0], color=".3", ls="-.", label="Secundário: C$_{LLM}$ = 0,05"),
                  Line2D([0], [0], color=".3", ls=":", label="Secundário: C$_{LLM}$ = 0,25")]
        fig.legend(handles=error_handles + styles, loc="lower center", bbox_to_anchor=(.54, .005), ncol=4, frameon=False, fontsize=8)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = output_base.with_suffix("." + extension)
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"FIGURA {path}: {path.stat().st_size} bytes")
        plt.close(fig)


def run_np_analysis(frame, cfg, rates_path: Path, output_base: Path | None = None) -> pd.DataFrame:
    rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode="confirmacao", source=rates_path)
    rows = []
    for alpha in NP_RATES:
        for error in HUMAN_ERRORS:
            for cap in HUMAN_BUDGETS:
                row, _ = evaluate_np_point(frame, cfg, rates, alpha=alpha, error=error, cap=cap)
                rows.append(row)
    table = pd.DataFrame(rows)
    table["rates_model"] = reference_rates_model(rates_path)
    comparison = compare_confirmation(frame, cfg, rates_path)
    comparison["rates_model"] = reference_rates_model(rates_path)
    for col in comparison.columns:
        if col not in table:
            table[col] = np.nan
    table.to_csv(ROOT / "data/varredura_np_confirmacao.csv", index=False)
    configured, _ = evaluate_np_point(frame, cfg, rates,
                                      alpha=cfg.decision_policy.modes["confirmacao"].np_policy.max_fp_rate_auto)
    configured.update({"rates_model": reference_rates_model(rates_path),
                       "rates_sha256": hashlib.sha256(rates_path.read_bytes()).hexdigest()})
    pd.DataFrame([configured]).to_csv(ROOT / "data/funil_gzcmd_np_status.csv", index=False)
    comparison.to_csv(ROOT / "data/comparacao_np_confirmacao.csv", index=False)
    variant = evaluate_birth_variant(frame)
    variant.to_csv(ROOT / "data/variante_guarda_confirmacao.csv", index=False)
    plot_np_figure(table, comparison, output_base or ROOT / "text/latex/figures/fig_cap8_np_confirmacao")
    report = ["# Confirmação por restrição versus custo", "",
              f"M3 referência={reference_rates_model(rates_path)}; mesmos OOF, C_llm=0,005, C_humano=1, C_fp/C_fn=100/20 e erro humano-base 5%.",
              "A restrição é fração esperada de FP entre MATCH automáticos, incluindo guardas: não é P(MATCH|não vínculo).",
              "Guardas são obrigatórias e consomem o orçamento de FP. Inviável significa que não há política satisfazendo a restrição, não um resultado com FP zero.",
              "Nesses pontos, decisões e métricas são NA; não há fallback automático que viole a restrição. A figura sombreia a região inviável.",
              "NONMATCH usa p<=max_fn_rate_auto (0,02), condição pontual conservadora que também limita sua média de FN esperados.",
              "Na área restante, LLM só se reduzir a perda humana e couber no orçamento; senão HUMANO. Se a capacidade humana for insuficiente, o cenário também é inviável.",
              "## Custo vs NP 0,001 e 0,002 (+0,01 como referência viável)", "```", comparison.to_string(index=False), "```", "",
              "## Variante de guarda de nascimento", "```", variant.to_string(index=False), "```",
              "Conflito forte: anos conhecidos diferentes (diff_ano>0); ausência de um lado não é conflito.",
              "A variante p>=0,95 é subconjunto da guarda atual p>=0,9: não pode recuperar os 17 rejeitados. Não foi adotada.",
              "## Grade", f"Pontos={len(table)}; viáveis={int(table.viavel.sum())}; inviáveis={int((~table.viavel).sum())}.",
              "Os valores são expectativas condicionais à calibração e às taxas M3, não desempenho LLM observado.",
              f"Taxas: {rates_path}; SHA256={hashlib.sha256(rates_path.read_bytes()).hexdigest()}."]
    (ROOT / "data/np_confirmacao_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nCOMPARACAO CUSTO vs NP\n" + comparison.to_string(index=False))
    print("\nVARIANTE GUARDA\n" + variant.to_string(index=False))
    print(f"\nNP GRID: {len(table)} pontos; viaveis={int(table.viavel.sum())}; inviaveis={int((~table.viavel).sum())}")
    return table


def plot_np_figure(table, comparison, output_base: Path) -> None:
    colors = ("#222222", "#3b82a0", "#b25c25", "#6a7955")
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.0), sharex=True)
        fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.17, hspace=.23)
        for color, error in zip(colors, HUMAN_ERRORS):
            subset = table[np.isclose(table.e_fn_h, error) & np.isinf(table.orcamento_humano)].sort_values("max_fp_rate_auto")
            valid = subset[subset.viavel]
            x = valid.max_fp_rate_auto
            axes[0].plot(x, valid.sens_esperada_automatica * 100, color=color, ls="--", marker="o", ms=5)
            axes[0].plot(x, valid.sens_esperada_com_humano * 100, color=color, marker="s", ms=4)
            axes[1].plot(x, valid.lote_llm, color=color, marker="o", ms=5)
            axes[1].plot(x, valid.humano_direto, color=color, ls="--", marker="s", ms=4)
        minimum = float(table.taxa_minima_atingivel.dropna().min())
        for ax in axes:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_pt))
            ax.set_xscale("log")
            ax.axvspan(min(NP_RATES) * .85, minimum, color=".9", zorder=0)
            ax.axvline(minimum, color=".45", ls=":", lw=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", lw=.35, alpha=.4)
            ax.set_xlim(min(NP_RATES) * .85, max(NP_RATES) * 1.2)
            ax.set_xticks(NP_RATES, ["0,05%", "0,1%", "0,2%", "0,5%", "1%"])
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Sensibilidade esperada (%)\nConfirmação")
        axes[0].text(.025, .78, "Região inviável com guardas fixas\nMétricas não definidas", transform=axes[0].transAxes,
                     va="top", fontsize=9, color=".35")
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel("Pares encaminhados (n)")
        axes[1].set_xlabel("Limite esperado de falsos vínculos entre MATCH automáticos (log)")
        cost = comparison[comparison.policy_kind.eq("cost")].iloc[0]
        axes[0].axhline(cost.sens_esperada_com_humano * 100, color=".4", ls=":", lw=1)
        errors = [Line2D([0], [0], color=c, lw=1.5, label=f"Erro humano: {e:.0%}") for c, e in zip(colors, HUMAN_ERRORS)]
        fig.legend(handles=errors, loc="upper center", ncol=4, frameon=False, fontsize=8)
        methods = [Line2D([0], [0], color=".3", ls="--", marker="o", label="Sens.: automática"),
                   Line2D([0], [0], color=".3", marker="s", label="Sens.: com humano"),
                   Line2D([0], [0], color=".3", marker="o", label="Carga: LLM"),
                   Line2D([0], [0], color=".3", ls="--", marker="s", label="Carga: humano direto"),
                   Line2D([0], [0], color=".4", ls=":", label="Custo base: sens. com humano")]
        fig.legend(handles=methods, loc="lower center", ncol=3, frameon=False, fontsize=8)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            path = output_base.with_suffix("." + extension)
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"FIGURA NP {path}: {path.stat().st_size} bytes")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "gzcmd/gzcmd_v3_config.yaml")
    parser.add_argument("--rates-yaml", type=Path, default=None)
    parser.add_argument("--policy-kind", choices=["cost", "np"], default=None)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--oof-csv", type=Path, default=ROOT / "data/pcal_ml_oof.csv")
    parser.add_argument("--figure-base", type=Path, default=ROOT / "text/latex/figures/fig_cap8_varredura_custos")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rates_path = args.rates_yaml or Path(cfg.llm_review.estimated_rates_path)
    rates_model = reference_rates_model(rates_path)
    if rates_model == "claude-haiku-5-5":
        from gzcmd.artifacts import preserve_m3v3_artifacts
        archive = preserve_m3v3_artifacts(ROOT)
        print("ARQUIVOS M3V3 PRESERVADOS", len(archive["files"]))
    rates_hash = hashlib.sha256(rates_path.read_bytes()).hexdigest()
    candidates = load_candidates(args.input_csv, args.oof_csv, cfg)
    kind = args.policy_kind or cfg.decision_policy.modes["confirmacao"].policy_kind
    if kind == "np":
        run_np_analysis(candidates, cfg, rates_path)
        if hashlib.sha256(rates_path.read_bytes()).hexdigest() != rates_hash:
            raise ValueError("Rates changed during NP sweep; regenerate on a stable snapshot")
        (ROOT / "data/varredura_np_metadata.json").write_text(json.dumps({
            "rates_yaml": str(rates_path), "rates_sha256": rates_hash, "rates_model": rates_model,
            "oof_sha256": hashlib.sha256(args.oof_csv.read_bytes()).hexdigest(),
            "grid": {"max_fp_rate_auto": NP_RATES, "human_error": HUMAN_ERRORS, "human_budget": HUMAN_BUDGETS},
        }, indent=2), encoding="utf-8")
        return
    tables = sweep(candidates, cfg, rates_path)
    if hashlib.sha256(rates_path.read_bytes()).hexdigest() != rates_hash:
        raise ValueError("Rates changed during the sweep; rerun on a stable snapshot")
    comparison = []
    for mode, table in tables.items():
        table.to_csv(ROOT / f"data/varredura_custos_{mode}.csv", index=False)
        for scenario, cost, error in (("antes", .25, 0.0), ("depois", .005, .05)):
            row = table[np.isclose(table.C_llm, cost) & np.isclose(table.e_fp_h, error) & np.isinf(table.orcamento_humano)].iloc[0].to_dict()
            comparison.append({"cenario": scenario, **row})
    result = pd.DataFrame(comparison)
    result.to_csv(ROOT / "data/varredura_custos_antes_depois.csv", index=False)
    metadata = {"rates_yaml": str(rates_path), "rates_sha256": rates_hash, "rates_model": rates_model,
                "oof_sha256": hashlib.sha256(args.oof_csv.read_bytes()).hexdigest(),
                "grid": {"C_llm": LLM_COSTS, "human_error": HUMAN_ERRORS, "human_budget": HUMAN_BUDGETS},
                "infinity_plot_proxy": 400, "direct_human_budget_excludes_llm_unsure": True}
    (ROOT / "data/varredura_custos_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    draw_figure(tables, args.figure_base)
    columns = ["modo", "cenario", "C_llm", "e_fn_h", "lote_llm", "humano_direto", "humano_total_esperado",
               "AUTO_MATCH", "AUTO_NONMATCH", "vinculos_perdidos_auto_nonmatch", "FP_automaticos",
               "sens_esperada_automatica", "sens_esperada_com_humano", "perda_esperada_total"]
    print(f"\nANTES/DEPOIS (M3 {rates_model}, orçamento humano ilimitado)\n" + result[columns].to_string(index=False))
    print("\nVARREDURA:", {mode: len(table) for mode, table in tables.items()})
    report = ["# Varredura de custos — revisões humanas equivalentes", "",
              f"Taxas M3: {rates_path}; SHA256={rates_hash}.",
              f"60 pontos por modo; p_cal_ml OOF fixo, seed 42. Expectativas do bloco {rates_model}, sem respostas LLM simuladas ou chamadas reais.",
              "C_llm=0,005 deriva do arredondamento de 5s/1200s=1/240; humano base=1.",
              "Erro humano é cenário, sem concordância inter-humana medida. A perda humana inclui o erro esperado.",
              "Orçamento limita humano direto; UNSURE LLM adiciona carga humana esperada, discriminada separadamente.",
              "O ponto ∞ é categórico na figura (posição gráfica 400), não um limite finito.",
              "A mudança simultânea de custo LLM e erro humano não garante perda menor: barato não significa infalível.",
              "## Antes/depois: C_llm 0,25/e_h=0 → 0,005/e_h=0,05", "```",
              result[columns].to_string(index=False), "```", "",
              "Dados completos: varredura_custos_vigilancia.csv e varredura_custos_confirmacao.csv.",
              "Figura: text/latex/figures/fig_cap8_varredura_custos.pdf (+ PNG). Nenhum .tex modificado."]
    (ROOT / "data/varredura_custos_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
