"""Avalia o Exp2 com Claude Sonnet 5.5 (sem model_outputs, prompt v4) contra o Haiku v4.

Saídas agregadas (sem linhas por par): data/exp2_sonnet_v4_resumo.csv,
data/taxas_erro_sonnet_v4.csv, data/exp2_sonnet_v4_discordancia_banda.csv e
data/exp2_sonnet_v4_relatorio.md. Seed não se aplica: nenhuma amostragem.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from estimar_taxas_erro_revisor import assign_bands, load_gold, summarize, wilson  # noqa: E402

SONNET = ROOT / "data/sprint3b/anthropic_sonnet_sem_mo_v4"
HAIKU = ROOT / "data/sprint3b/anthropic_sem_mo_v4"
MODES = ("vigilancia", "confirmacao")


def load(path: Path) -> tuple[dict, pd.DataFrame]:
    run = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"pair_id": r["pair_id"], "decision": r["decision"], "protocol": r["protocol"],
             "latency": r.get("total_latency_s"), "tokens": r.get("total_tokens")} for r in run["results"]]
    return run, pd.DataFrame(rows).set_index("pair_id")


def counts(frame: pd.DataFrame) -> dict:
    d, y = frame.decision, frame.y
    vp = int((d.eq("MATCH") & y.eq(1)).sum())
    fp = int((d.eq("MATCH") & y.eq(0)).sum())
    fn = int((d.eq("NONMATCH") & y.eq(1)).sum())
    vn = int((d.eq("NONMATCH") & y.eq(0)).sum())
    un = int(d.eq("UNSURE").sum())
    out = {"n": len(frame), "VP": vp, "FP": fp, "FN": fn, "VN": vn, "UNSURE": un,
           "frac_unsure": un / len(frame) if len(frame) else None}
    for name, k, n in (("sens", vp, vp + fn), ("espec", vn, vn + fp)):
        p, lo, hi = wilson(k, n)
        out.update({f"{name}_sem_unsure": p, f"{name}_ic95_low": lo, f"{name}_ic95_high": hi})
    out["acerto_estrito"] = (vp + vn) / len(frame) if len(frame) else None
    return out


def correct(frame: pd.DataFrame, col: str) -> pd.Series:
    return (frame[col].eq("MATCH") & frame.y.eq(1)) | (frame[col].eq("NONMATCH") & frame.y.eq(0))


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    return f"{x * 100:.1f}%".replace(".", ",") if pct else f"{x:.4f}".replace(".", ",")


def main() -> None:
    config = yaml.safe_load((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))
    definitions = config["bands"]["definitions"]
    fixed = config["llm_review"]["reliability"]["error_rates_by_band"]
    gold = load_gold(ROOT / "data/COMPARADORSEMIDENT.csv").set_index("pair_id")
    gold["band"] = assign_bands(gold.nota_final, definitions)
    summary, rates, discord, md = [], [], [], []
    for mode in MODES:
        s_run, s = load(SONNET / f"exp2_llm_sync_{mode}.json")
        h_run, h = load(HAIKU / f"exp2_llm_sync_{mode}.json")
        cohort_ids = list(s_run["selected_pair_ids"])
        s = s.join(gold[["y", "nota_final", "band"]])
        h_all = h.join(gold[["y", "nota_final", "band"]])
        common = s.index.intersection(h.index)
        both = s.loc[common, ["decision", "y", "nota_final", "band", "latency", "tokens", "protocol"]].join(
            h.loc[common, ["decision"]].rename(columns={"decision": "dec_haiku"}))
        h_common = h_all.loc[common]
        sc = s.join(pd.DataFrame(index=s.index))
        arms = (("sonnet-5.5", sc, s_run), ("haiku-5.5 (mesmos pares)", h_common, h_run),
                ("haiku-5.5 (coorte completa)", h_all, h_run))
        for arm, frame, run in arms:
            row = {"mode": mode, "arm": arm, "coorte": len(cohort_ids), **counts(frame)}
            if arm == "sonnet-5.5":
                lat = frame.latency.dropna()
                row.update({"completo": bool(s_run.get("complete")),
                            "latencia_mediana_s": statistics.median(lat) if len(lat) else None,
                            "tokens_por_par_mediana": statistics.median(frame.tokens.dropna()),
                            "frac_arbitro": float(frame.protocol.ne("consensus").mean()),
                            "chamadas": run["transport"]["calls"],
                            "tokens_entrada": run["transport"]["input_tokens"],
                            "tokens_saida": run["transport"]["output_tokens"],
                            "custo_estimado_usd": run["cost_estimate_usd"],
                            "erros_api": run["n_errors"]})
            summary.append(row)
        # McNemar sobre acerto estrito (UNSURE = não acerto), pares em comum
        both = both.rename(columns={"decision": "dec_sonnet"})
        sc_ok, hk_ok = correct(both, "dec_sonnet"), correct(both, "dec_haiku")
        b, c = int((sc_ok & ~hk_ok).sum()), int((~sc_ok & hk_ok).sum())
        p = float(binomtest(b, b + c, .5).pvalue) if b + c else 1.0
        summary[-2].update({"mcnemar_sonnet_acerta_haiku_erra": b, "mcnemar_haiku_acerta_sonnet_erra": c,
                            "mcnemar_p_exato": p, "n_comum": len(both)})
        summary[-3].update({"mcnemar_sonnet_acerta_haiku_erra": b, "mcnemar_haiku_acerta_sonnet_erra": c,
                            "mcnemar_p_exato": p, "n_comum": len(both)})
        for band in definitions:
            part = both[both.band.eq(band)]
            sok, hok = correct(part, "dec_sonnet"), correct(part, "dec_haiku")
            discord.append({"mode": mode, "band": band, "n": len(part),
                            "sonnet_acerta_haiku_erra": int((sok & ~hok).sum()),
                            "haiku_acerta_sonnet_erra": int((~sok & hok).sum()),
                            "ambos_acertam": int((sok & hok).sum()), "ambos_erram": int((~sok & ~hok).sum()),
                            "n_coorte_banda": int(gold.loc[cohort_ids].band.eq(band).sum())})
            r = summarize(s[s.band.eq(band)].rename(columns={"decision": "decision"}), mode, "claude-sonnet-5-5",
                          band, not bool(s_run.get("complete")), fixed)
            rates.append(r)
    pd.DataFrame(summary).to_csv(ROOT / "data/exp2_sonnet_v4_resumo.csv", index=False)
    pd.DataFrame(rates).to_csv(ROOT / "data/taxas_erro_sonnet_v4.csv", index=False)
    pd.DataFrame(discord).to_csv(ROOT / "data/exp2_sonnet_v4_discordancia_banda.csv", index=False)

    md += ["# Exp2: Claude Sonnet 5.5, dossiê sem model_outputs, prompt v4", "",
           "Modelo claude-sonnet-5-5 (API Anthropic), mesmo protocolo Agent-A + Agent-B + árbitro, mesmo dossiê "
           "sem `model_outputs`, mesmo prompt v4 e mesmas coortes históricas do Haiku v4. Configuração idêntica: "
           "sem temperature, `max_tokens` 2500, sem extended thinking, 4 workers. Executor: "
           "`scripts/run_exp2_sonnet_sem_mo.py` (reutiliza `run_exp2_sem_model_outputs.py`, só troca a tabela de preço). "
           "Avaliação: `scripts/avaliar_sonnet_v4.py`.", ""]
    for mode in MODES:
        rows = [r for r in summary if r["mode"] == mode]
        son = rows[0]
        md += [f"## {mode}", "",
               f"Respostas do Sonnet: {son['n']} de {son['coorte']} "
               f"({'completo' if son['completo'] else 'parcial, interrompido pelo teto de custo'}).", "",
               "| Braço | N | VP | FP | FN | VN | UNSURE | % UNSURE | Sens. sem U (IC95) | Esp. sem U (IC95) | Acerto estrito |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
        for r in rows:
            md.append(f"| {r['arm']} | {r['n']} | {r['VP']} | {r['FP']} | {r['FN']} | {r['VN']} | {r['UNSURE']} | "
                      f"{fmt(r['frac_unsure'])} | {fmt(r['sens_sem_unsure'])} [{fmt(r['sens_ic95_low'])}; {fmt(r['sens_ic95_high'])}] | "
                      f"{fmt(r['espec_sem_unsure'])} [{fmt(r['espec_ic95_low'])}; {fmt(r['espec_ic95_high'])}] | "
                      f"{fmt(r['acerto_estrito'])} |")
        md += ["", f"McNemar exato (acerto estrito, UNSURE conta como não acerto), {son['n_comum']} pares em comum: "
               f"Sonnet acerta e Haiku erra = {son['mcnemar_sonnet_acerta_haiku_erra']}; "
               f"Haiku acerta e Sonnet erra = {son['mcnemar_haiku_acerta_sonnet_erra']}; "
               f"p {'< 0,0001' if son['mcnemar_p_exato'] < 1e-4 else '= ' + fmt(son['mcnemar_p_exato'], False)}.", "",
               "Discordância por banda de nota:", "",
               "| Banda | N | Sonnet certo/Haiku errado | Haiku certo/Sonnet errado | Ambos certos | Ambos errados | N da coorte na banda |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for d in [d for d in discord if d["mode"] == mode]:
            md.append(f"| {d['band']} | {d['n']} | {d['sonnet_acerta_haiku_erra']} | {d['haiku_acerta_sonnet_erra']} | "
                      f"{d['ambos_acertam']} | {d['ambos_erram']} | {d['n_coorte_banda']} |")
        md += ["", "Taxas de erro do Sonnet por banda (IC95 de Wilson; e_FP = FP/não vínculos, e_FN = FN/vínculos, "
               "u = UNSURE/N):", "",
               "| Banda | N | vínculos | e_FP | e_FN | u |", "|---|---:|---:|---|---|---|"]
        for r in [r for r in rates if r["mode"] == mode]:
            def cell(m):
                return (f"{fmt(r[m])} [{fmt(r[m + '_ci95_low'])}; {fmt(r[m + '_ci95_high'])}]"
                        if r[m] is not None and not pd.isna(r[m]) else "NA")
            md.append(f"| {r['band']} | {r['n']} | {r['n_vinculos']} | {cell('e_fp')} (FP {r['fp']}) | "
                      f"{cell('e_fn')} (FN {r['fn']}) | {cell('u')} (U {r['n_unsure']}) |")
        md += ["", f"Latência mediana por par: {br(son['latencia_mediana_s'], 1)} s; tokens por par (mediana): "
               f"{br(son['tokens_por_par_mediana'], 0)}; fração decidida por árbitro: {fmt(son['frac_arbitro'])}; "
               f"chamadas: {br(son['chamadas'], 0)}; tokens de entrada: {br(son['tokens_entrada'], 0)}; saída: {br(son['tokens_saida'], 0)}; "
               f"erros de API: {son['erros_api']}; custo estimado: US$ {br(son['custo_estimado_usd'])}.", ""]
    md += cost_and_cascade_sections(summary)
    (ROOT / "data/exp2_sonnet_v4_relatorio.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def br(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def cost_and_cascade_sections(summary) -> list[str]:
    dirs = [("coorte histórica", SONNET), ("lote v3.1 REFREC", ROOT / "data/sprint3b/anthropic_sonnet_sem_mo_v4_lote31_refrec"),
            ("complemento do lote v3.1 (pares do lote em que o Haiku usou a execução por linha)",
             ROOT / "data/sprint3b/anthropic_sonnet_sem_mo_v4_lote31_complemento")]
    cost_rows, total = [], 0.0
    for label, directory in dirs:
        for mode in MODES:
            run = json.loads((directory / f"exp2_llm_sync_{mode}.json").read_text(encoding="utf-8"))
            total += run["cost_estimate_usd"]
            tr = run["transport"]
            cost_rows.append(f"| {label} | {mode} | {run['n_reviewed']} | {br(tr['calls'], 0)} | {br(tr['input_tokens'], 0)} | "
                             f"{br(tr['output_tokens'], 0)} | {br(run['cost_estimate_usd'])} |")
    out = ["## Custo", "",
           "Estimativa tarifária (não é fatura): Claude Sonnet 5.5 a US$ 2 por milhão de tokens de entrada e US$ 10 por "
           "milhão de saída (https://platform.claude.com/docs/en/about-claude/pricing, consultado em 2026-11-03). "
           "O Sonnet 5.5 custa 20 vezes o Haiku 5.5 por token e gerou mais tokens de saída por chamada.", "",
           "| Execução | Modo | Pares | Chamadas | Tokens de entrada | Tokens de saída | US$ |", "|---|---|---:|---:|---:|---:|---:|",
           *cost_rows, f"| **Total** | | | | | | **{br(total)}** |", "",
           "O primeiro teto (US$ 40) interrompeu a vigilância em 651 pares; com autorização, foi retomada do checkpoint "
           "sem repetir pares e concluída (teto acumulado de US$ 80, não atingido). Houve um erro HTTP 529 recuperado por retry; "
           "nenhum par falhou.", ""]
    cascade = pd.read_csv(ROOT / "data/funil_sonnet_v4_cascata.csv")
    cover = json.loads((ROOT / "data/funil_sonnet_v4_cobertura.json").read_text(encoding="utf-8"))
    out += ["## Cascata v3.1 com Sonnet como revisor", "",
            "`scripts/funil_gzcmd_v2_sonnet.py` reexecuta o pipeline v3 e v3.1 sobre o extrato inteiro (61.696 pares, 247 vínculos) "
            "e troca apenas as respostas do revisor LLM: Sonnet 5.5 v4 contra Haiku 5.5 v4, ambos sem `model_outputs`. "
            "A seleção do lote LLM usa as taxas de erro por banda do Haiku, sem reestimar para o Sonnet (por isso o lote é o mesmo). "
            "Execução offline, sem chamadas LLM. Saídas: `data/funil_sonnet_v4_cascata.csv` e `funil_sonnet_v4_cobertura.json`.", "",
            "Cobertura do lote v3.1 pelas respostas do Sonnet: " + "; ".join(
                f"{c['modo']}: {c['com_resposta_sonnet']} de {c['lote_v3.1']} "
                f"(coorte histórica {c['lote_antigo']}, REFREC e complemento cobrem o restante)" for c in cover) + ". "
            "A lista de pendentes REFREC cobria só os pares sem resposta do Haiku na execução REFREC; os pares do lote v3.1 que o "
            "Haiku tinha respondido na execução por linha (100 na vigilância, 3 na confirmação) não tinham resposta do Sonnet "
            "e foram executados à parte (complemento).", "",
            "Definições: VP e FP são decisões MATCH aceitas automaticamente (motor ou LLM); perdidos são vínculos finalizados como "
            "NONMATCH; humano soma UNSURE do LLM e encaminhamento direto do motor; sens. automática = VP/vínculos; "
            "sens. com humano = (VP + 0,95 × vínculos ao humano)/vínculos, hipótese de erro humano de 5%, sem decisões humanas "
            "observadas. IC95 de Wilson para a sensibilidade automática.", ""]
    for version in ("v3.1", "v3"):
        out += [f"### Versão {version}", "",
                "| Modo | Revisor | VP | FP | Perdidos | Humano (pares / vínculos) | Sens. automática (IC95) | Sens. com humano | LLM: MATCH / NONMATCH / UNSURE |",
                "|---|---|---:|---:|---:|---|---|---:|---|"]
        for _, r in cascade[cascade.versao.eq(version)].iterrows():
            out.append(f"| {r.modo} | {r.revisor.replace(' v4 sem MO', '')} | {r.VP} | {r.FP} | {r.perdidos_FN_nonmatch} | "
                       f"{r.humano_pares} / {r.humano_vinculos} | {fmt(r.sens_automatica)} [{fmt(r.sens_auto_ic95_low)}; {fmt(r.sens_auto_ic95_high)}] | "
                       f"{fmt(r.sens_com_humano_erro_05)} | {r.llm_match} / {r.llm_nonmatch} / {r.llm_unsure} |")
        out.append("")
    out += ["## Ressalvas", "",
            "- Execução única por revisor, sem ajuste de multiplicidade; UNSURE conta como não acerto no McNemar.",
            "- O Sonnet usa mais UNSURE que o Haiku: isso reduz perdidos e eleva a sensibilidade, mas desloca pares para o humano.",
            "- Os conjuntos de vínculos do lote LLM são pequenos (28 na vigilância e 26 na confirmação no lote v3.1); diferenças de poucos pares não são conclusivas.",
            "- A seleção do lote usa taxas estimadas com o Haiku; um lote selecionado com taxas do Sonnet poderia diferir.",
            "- Extrato anonimizado; nenhum dado por par nos artefatos."]
    return out


if __name__ == "__main__":
    main()


