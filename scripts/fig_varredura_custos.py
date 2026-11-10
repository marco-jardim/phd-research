"""Sweep human-equivalent costs offline and plot the two GZ-CMD modes.

python scripts/fig_varredura_custos.py --rates-yaml gzcmd/error_rates_by_band_estimadas.yaml
60 scenarios/mode; reference-reviewer expectations, not simulated responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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
    oof = fit_xgb_oof(input_csv, output_path=oof_csv, reuse=True,
                      split_by=cfg.evaluation.split_by).set_index("pair_id")
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
            axes[0, col].set_ylim(40, 100)
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


def _round_half_up(value, places: int, scale: int = 1) -> Decimal:
    """Round-half-up on the shortest decimal repr of the float (no binary artefacts)."""
    return (Decimal(repr(float(value))) * scale).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def pct_pt(value, places: int = 1) -> str:
    """Fraction as percentage text with decimal comma, rounded half up (0.5385 -> '53,8')."""
    return f"{_round_half_up(value, places, 100):.{places}f}".replace(".", ",")


def num_pt(value, places: int = 2) -> str:
    return f"{_round_half_up(value, places):.{places}f}".replace(".", ",")


def limit_label(alpha) -> str:
    text = pct_pt(alpha, 2)
    if "," in text:
        text = text.rstrip("0").rstrip(",")
    return text + "%"


NP_STATE_GUARD, NP_STATE_BUDGET, NP_STATE_OK = "guardas", "orcamento", "viavel"


def _np_state(row) -> str:
    if bool(row.viavel):
        return NP_STATE_OK
    return NP_STATE_GUARD if row.motivo == "guardas_fp" else NP_STATE_BUDGET


def np_figure_data(table: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    """Everything annotated in the NP figure, read from the two CSV tables (no recomputation)."""
    budgets = sorted(table.orcamento_humano.unique())
    errors = sorted(table.e_fn_h.unique())
    columns = []
    for alpha in sorted(table.max_fp_rate_auto.unique()):
        sub = table[np.isclose(table.max_fp_rate_auto, alpha)]
        if sub.motivo.eq("guardas_fp").all():
            columns.append({"alpha": float(alpha), "error": None, "label": limit_label(alpha),
                            "states": [NP_STATE_GUARD] * len(budgets), "needed": None})
            continue
        for error in errors:
            cell = sub[np.isclose(sub.e_fn_h, error)]
            states = [_np_state(cell[np.isclose(cell.orcamento_humano, b)].iloc[0]) for b in budgets]
            unlimited = cell[np.isinf(cell.orcamento_humano)].iloc[0]
            needed = unlimited.humano_direto if unlimited.viavel else unlimited.humanos_necessarios
            columns.append({"alpha": float(alpha), "error": float(error), "label": f"{error:.0%}",
                            "states": states, "needed": None if pd.isna(needed) else int(round(needed))})
    guard = table.dropna(subset=["taxa_minima_atingivel"]).iloc[0]
    alpha_ref = float(table[table.viavel].max_fp_rate_auto.min())
    dumbbell = []
    for error in errors:
        row = table[np.isclose(table.max_fp_rate_auto, alpha_ref) & np.isclose(table.e_fn_h, error)
                    & np.isinf(table.orcamento_humano)].iloc[0]
        if not bool(row.viavel):
            continue
        llm, direct = int(round(row.lote_llm)), int(round(row.humano_direto))
        dumbbell.append({"error": float(error), "label": f"{error:.0%}",
                         "auto": float(row.sens_esperada_automatica), "human": float(row.sens_esperada_com_humano),
                         "auto_label": pct_pt(row.sens_esperada_automatica), "human_label": pct_pt(row.sens_esperada_com_humano),
                         "llm": llm, "direct": direct, "total": llm + direct})
    cost = comparison[comparison.policy_kind.eq("cost")].iloc[0]
    cost_llm, cost_direct = int(round(cost.lote_llm)), int(round(cost.humano_direto))
    return {"budgets": budgets, "columns": columns, "alpha_ref": alpha_ref, "dumbbell": dumbbell,
            "minimum": float(table.taxa_minima_atingivel.dropna().min()),
            "minimum_label": pct_pt(table.taxa_minima_atingivel.dropna().min(), 2),
            "n_min_vinculos": int(round(guard.n_match_na_taxa_minima)),
            "n_guardas": int(round(guard.n_guardas_match)),
            "fp_guardas_label": num_pt(guard.fp_esperado_guardas, 2),
            "cost": {"error": float(cost.e_fn_h), "sens": float(cost.sens_esperada_com_humano),
                     "sens_label": pct_pt(cost.sens_esperada_com_humano), "llm": cost_llm, "direct": cost_direct,
                     "total": cost_llm + cost_direct}}


def build_np_figure(table: pd.DataFrame, comparison: pd.DataFrame):
    """Three panels: (a) viability matrix limit x human budget, (b) sensitivity dumbbells, (c) review load."""
    data = np_figure_data(table, comparison)
    fills = {NP_STATE_GUARD: "#d9d9d9", NP_STATE_BUDGET: "#f2d3b8", NP_STATE_OK: "#b9d6e3"}
    orange, blue, ink = "#b25c25", "#3b82a0", "#222222"
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 8.5,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "hatch.linewidth": .6}
    budgets, columns = data["budgets"], data["columns"]
    budget_labels = ["ilimitado" if np.isinf(b) else str(int(b)) for b in budgets]
    finite = [b for b in budgets if np.isfinite(b)]
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.4, 6.3))
        gs = fig.add_gridspec(6, 2, height_ratios=[2.6, .5, .45, 2.8, 1.05, .75], hspace=0, wspace=.34,
                              left=.10, right=.93, top=.88, bottom=.02)

        # (a) viability matrix: limit (columns; the viable limit is split by human-error hypothesis) x human budget
        ax = fig.add_subplot(gs[0, :])
        for j, col in enumerate(columns):
            for i, state in enumerate(col["states"]):
                if state == NP_STATE_BUDGET:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fills[state], edgecolor="#cf9a72", hatch="////", lw=0))
                else:
                    ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fills[state], lw=0))
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="white", lw=1.5))
                if col["needed"] is not None:
                    ax.text(j + .5, i + .5, str(col["needed"]), ha="center", va="center", fontsize=7.5, color=ink,
                            bbox={"boxstyle": "round,pad=.15", "fc": "white", "ec": "none", "alpha": .75})
        guard_cols = [j for j, col in enumerate(columns) if col["error"] is None]
        open_cols = [j for j, col in enumerate(columns) if col["error"] is not None]
        if guard_cols:
            ax.text((guard_cols[0] + guard_cols[-1] + 1) / 2, len(budgets) / 2,
                    f"Inviável pelas guardas\n{data['n_guardas']} vínculos das guardas somam {data['fp_guardas_label']}\n"
                    f"falso positivo esperado\nMínimo atingível: {data['minimum_label']}% ({data['n_min_vinculos']} vínculos)",
                    ha="center", va="center", fontsize=7.5, color=".2", linespacing=1.35)
        if guard_cols and open_cols:
            ax.axvline(guard_cols[-1] + 1, color=ink, lw=1.8)
        for j, col in enumerate(columns):
            ax.annotate(col["label"], (j + .5, 0), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, annotation_clip=False)
        for cols, text in ((guard_cols, "Limite de falsos positivos esperados"),
                           (open_cols, f"Limite de {limit_label(data['alpha_ref'])}, com erro humano hipotético de")):
            if cols:
                ax.annotate(text, ((cols[0] + cols[-1] + 1) / 2, 0), xytext=(0, 16), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=".25", annotation_clip=False)
        ax.set_xlim(0, len(columns))
        ax.set_ylim(len(budgets), 0)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(budgets)) + .5, budget_labels)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel("Orçamento humano direto (pares)")
        ax.spines[:].set_visible(False)
        ax.set_title("(a) Viabilidade segundo o limite e o orçamento humano", loc="left", fontsize=9,
                     fontweight="bold", pad=30)
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        legend_ax.legend(handles=[Patch(facecolor=fills[NP_STATE_GUARD], label="Inviável pelas guardas"),
                                  Patch(facecolor=fills[NP_STATE_BUDGET], edgecolor="#cf9a72", hatch="////",
                                        label="Inviável pelo orçamento humano"),
                                  Patch(facecolor=fills[NP_STATE_OK], label="Viável"),
                                  Line2D([], [], linestyle="none", label="Número: pares humanos diretos exigidos")],
                         loc="center", ncol=2, frameon=False, fontsize=7.5, labelspacing=.35, columnspacing=2.5,
                         handlelength=1.8)

        dumbbell, cost = data["dumbbell"], data["cost"]
        xs = np.arange(len(dumbbell))
        error_labels = [d["label"] for d in dumbbell]
        xlabel = f"Erro humano hipotético\n(limite de {limit_label(data['alpha_ref'])}; orçamento ilimitado)"

        # (b) dumbbell: automatic sensitivity -> sensitivity after human review, with the cost-policy reference
        axb = fig.add_subplot(gs[3, 0])
        auto = np.array([d["auto"] for d in dumbbell]) * 100
        human = np.array([d["human"] for d in dumbbell]) * 100
        axb.axhline(cost["sens"] * 100, color=".4", ls=":", lw=1.2, zorder=1)
        axb.vlines(xs, auto, human, color=".6", lw=2.2, zorder=2)
        axb.plot(xs, auto, "o", mfc="white", mec=ink, mew=1.3, ms=6.5, ls="none", zorder=3)
        axb.plot(xs, human, "s", color=ink, ms=6, ls="none", zorder=3)
        for x, d in zip(xs, dumbbell):
            axb.annotate(d["auto_label"], (x, d["auto"] * 100), xytext=(0, -6), textcoords="offset points",
                         ha="center", va="top", fontsize=7.5)
            axb.annotate(d["human_label"], (x, d["human"] * 100), xytext=(0, 6), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5)
        axb.set_xlim(-.5, len(xs) - .5)
        axb.set_ylim(40, 100)
        axb.set_xticks(xs, error_labels)
        axb.set_xlabel(xlabel)
        axb.set_ylabel("Sensibilidade esperada (%)")
        axb.set_title("(b) Sensibilidade esperada", loc="left", fontsize=9, fontweight="bold")

        # (c) stacked review load, cost policy as a separate bar, human budgets as dashed lines
        axc = fig.add_subplot(gs[3, 1])
        width, cost_x = .62, len(xs) - 1 + 1.35
        direct = np.array([d["direct"] for d in dumbbell])
        llm = np.array([d["llm"] for d in dumbbell])
        axc.bar(xs, direct, width, color=orange, zorder=2)
        axc.bar(xs, llm, width, bottom=direct, color=blue, zorder=2)
        axc.bar([cost_x], [cost["direct"]], width, color=orange, zorder=2)
        axc.bar([cost_x], [cost["llm"]], width, bottom=[cost["direct"]], color=blue, zorder=2)
        for budget in finite:
            axc.axhline(budget, color=".25", ls="--", lw=.7, zorder=3)
            axc.annotate(str(int(budget)), (1, budget), xycoords=axc.get_yaxis_transform(), xytext=(3, 0),
                         textcoords="offset points", ha="left", va="center", fontsize=6.5, color=".25",
                         annotation_clip=False)
        bars = [(x, d["direct"], d["llm"]) for x, d in zip(xs, dumbbell)] + [(cost_x, cost["direct"], cost["llm"])]
        for x, n_direct, n_llm in bars:
            for bottom, height in ((0, n_direct), (n_direct, n_llm)):
                if height <= 0:
                    continue
                if height >= 28:
                    axc.text(x, bottom + height / 2, str(height), ha="center", va="center", fontsize=7, color="white",
                             zorder=4)
                else:
                    axc.annotate(str(height), (x + width / 2, bottom + height / 2), xytext=(3, 0),
                                 textcoords="offset points", ha="left", va="center", fontsize=7, zorder=4)
            axc.annotate(str(n_direct + n_llm), (x, n_direct + n_llm), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        axc.axvline((xs[-1] + cost_x) / 2, color=".6", lw=.8, ls=":")
        axc.set_xlim(-.5, cost_x + .6)
        axc.set_ylim(0, max(d["total"] for d in dumbbell + [cost]) * 1.12)
        axc.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_pt))
        axc.set_xticks(list(xs) + [cost_x], error_labels + [f"Política\npor custo\n(erro {cost['error']:.0%})"])
        axc.get_xticklabels()[-1].set_fontsize(7)
        axc.set_xlabel(xlabel)
        axc.set_ylabel("Pares encaminhados (n)")
        axc.set_title("(c) Pares encaminhados", loc="left", fontsize=9, fontweight="bold")
        for panel in (axb, axc):
            panel.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_pt))
            panel.spines[["top", "right"]].set_visible(False)
            panel.grid(axis="y", lw=.35, alpha=.4, zorder=0)
            panel.set_axisbelow(True)
        budget_text = ", ".join(str(int(b)) for b in finite[:-1]) + f" e {int(finite[-1])}" if len(finite) > 1 else ""
        for slot, handles in ((0, [Line2D([], [], marker="o", mfc="white", mec=ink, mew=1.3, ls="none", label="Automática"),
                                   Line2D([], [], marker="s", color=ink, ls="none", label="Com revisão humana"),
                                   Line2D([], [], color=".4", ls=":", lw=1.2,
                                          label=f"Política por custo, com revisão\nhumana (erro {cost['error']:.0%}): {cost['sens_label']}%")]),
                              (1, [Patch(facecolor=orange, label="Revisor humano direto"),
                                   Patch(facecolor=blue, label="LLM"),
                                   Line2D([], [], color=".25", ls="--", lw=.8,
                                          label=f"Orçamento humano direto\n({budget_text} pares)")])):
            lax = fig.add_subplot(gs[5, slot])
            lax.axis("off")
            lax.legend(handles=handles, loc="upper center", frameon=False, fontsize=7, labelspacing=.3, handlelength=1.6)
    return fig, data


def plot_np_figure(table, comparison, output_base: Path) -> None:
    fig, _ = build_np_figure(table, comparison)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    # Fixed metadata (no creation timestamp / software string) so regenerating gives identical files.
    metadata = {"pdf": {"CreationDate": None}, "png": {"Software": None}}
    for extension in ("pdf", "png"):
        path = output_base.with_suffix("." + extension)
        fig.savefig(path, bbox_inches="tight", dpi=300, metadata=metadata[extension])
        print(f"FIGURA NP {path}: {path.stat().st_size} bytes")
    plt.close(fig)


def figure_np_only(table_csv: Path | None = None, comparison_csv: Path | None = None,
                   output_base: Path | None = None) -> None:
    """Redraw the NP figure from the stored CSVs, without rerunning the sweep or rewriting data/."""
    plot_np_figure(pd.read_csv(table_csv or ROOT / "data/varredura_np_confirmacao.csv"),
                   pd.read_csv(comparison_csv or ROOT / "data/comparacao_np_confirmacao.csv"),
                   output_base or ROOT / "text/latex/figures/fig_cap8_np_confirmacao")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "gzcmd/gzcmd_v3_config.yaml")
    parser.add_argument("--rates-yaml", type=Path, default=None)
    parser.add_argument("--policy-kind", choices=["cost", "np"], default=None)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--oof-csv", type=Path, default=ROOT / "data/pcal_ml_oof.csv")
    parser.add_argument("--figure-base", type=Path, default=ROOT / "text/latex/figures/fig_cap8_varredura_custos")
    parser.add_argument("--so-figura-np", action="store_true",
                        help="redesenha só a figura NP a partir de data/varredura_np_confirmacao.csv e "
                             "data/comparacao_np_confirmacao.csv; não roda a varredura nem reescreve data/")
    args = parser.parse_args()
    if args.so_figura_np:
        figure_np_only()
        return
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
