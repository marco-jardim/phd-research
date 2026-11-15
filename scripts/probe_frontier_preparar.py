"""Congela a amostra local do probe; não faz chamadas de inferência."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_exp2_sem_model_outputs as runner

OUT = ROOT / ".local/probe_frontier"
EXTERNAL = Path(os.environ.get("PROBE_FRONTIER_EXTERNAL") or Path.home() / ".local/share/opencode/worktree/d66565/pq-arranjo")
SEED = 20261107
MODELS = ["claude-opus-5-5", "claude-fable-5-1", "claude-haiku-5-5"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histories(mode, pattern, prompt_sha):
    result, provenance, fingerprints = {}, [], {}
    for base in (ROOT, EXTERNAL):
        for folder in sorted((base / "data/sprint3b").glob(pattern)):
            # Exclude later interventions; use exactly the requested historical v4 cohorts.
            suffix = folder.name.split("sem_mo_v4", 1)[1]
            if suffix and not suffix.startswith(("_arranjo_corte", "_supl", "_lote31")):
                continue
            path = folder / f"exp2_llm_sync_{mode}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["mode"] != mode or data["prompt_sha256"] != prompt_sha:
                raise ValueError("Historical prompt/mode mismatch")
            provenance.append({"path": str(path), "sha256": sha(path)})
            for row in data["results"]:
                pid = row["pair_id"]
                if pid in result and result[pid] != row["decision"]:
                    raise ValueError("Conflicting historical decisions; select source explicitly")
                result[pid] = row["decision"]
                fingerprint = data.get("dossier_sha256", {}).get(pid)
                if fingerprint:
                    if pid in fingerprints and fingerprints[pid] != fingerprint:
                        raise ValueError("Conflicting dossier fingerprints")
                    fingerprints[pid] = fingerprint
    return result, provenance, fingerprints


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "PREREG.md").exists():
        raise ValueError("Preregistration already exists; refusing to overwrite")
    metadata = runner.configure_prompt("v4")
    raw = pd.read_csv(ROOT / "data/COMPARADORSEMIDENT.csv", sep=";", dtype=str)
    col = lambda n: next(c for c in raw.columns if c == n or c.startswith(n + ","))
    ids = [hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
           for a, b, c in zip(*(raw[col(k)].str.strip() for k in ("COMPREC", "REFREC", "PASSO")))]
    labels = pd.Series(pd.to_numeric(raw.PAR, errors="coerce").fillna(0).astype(int).to_numpy(), index=ids)
    if not labels.index.is_unique:
        raise ValueError("Duplicate pair IDs")
    metrics = pd.read_csv(ROOT / "data/arranjo_corte_metricas.csv")
    rows, audit, baseline, hashes = [], {}, {}, {}
    rng = np.random.default_rng(SEED)
    for mode, expected in (("vigilancia", 344), ("confirmacao", 41)):
        path = EXTERNAL / f".local/arranjo_corte/policy_{mode}_P.csv"
        frame = pd.read_csv(path, dtype=str)
        cohort = sorted(frame.loc[frame.action.eq("LLM_REVIEW"), "pair_id"])
        metric = metrics.loc[(metrics.modo == mode) & (metrics.braco == "P") & (metrics.faixa == "TODAS")]
        if len(cohort) != expected or len(set(cohort)) != expected or int(metric.iloc[0].pares_llm) != expected:
            raise ValueError("P cohort count mismatch")
        haiku, hs, hf = histories(mode, "anthropic_sem_mo_v4*", metadata["prompt_sha256"])
        sonnet, ss, _ = histories(mode, "anthropic_sonnet_sem_mo_v4*", metadata["prompt_sha256"])
        if set(cohort) - haiku.keys() or set(cohort) - hf.keys():
            raise ValueError("Missing historical Haiku decisions/dossier fingerprints")
        baseline[mode] = {p: {"PAR": int(labels[p]), "haiku": haiku[p], "sonnet": sonnet.get(p)} for p in cohort}
        hashes[mode] = {p: hf[p] for p in cohort}
        strata = {}
        if mode == "vigilancia":
            strata["S1"] = [p for p in cohort if haiku[p] == "MATCH"]
            strata["S2"] = [p for p in cohort if labels[p] == 2 and p not in strata["S1"]]
            strata["S3"] = [p for p in cohort if labels[p] == 0 and p in sonnet and haiku[p] != sonnet[p]
                             and p not in strata["S1"]]
            if len(strata["S1"]) != 25 or sum(labels[p] == 0 for p in strata["S1"]) != 21:
                raise ValueError("S1 does not reproduce 25 MATCH / 21 FP")
            certain = set().union(*map(set, strata.values()))
            remaining = sorted(set(cohort) - certain)
            n4 = min(len(remaining), max(10, 150 - len(certain)))
            strata["S4"] = sorted(rng.choice(remaining, size=n4, replace=False).tolist())
            fraction = n4 / len(remaining)
        else:
            strata["C"] = cohort
            fraction = 1.0
        for label, selected in strata.items():
            for pid in selected:
                rows.append({"pair_id": pid, "modo": mode, "estrato": label,
                             "peso": 1 / fraction if label == "S4" else 1})
        audit[mode] = {"population": len(cohort), "selected": sum(map(len, strata.values())),
                       "strata": {k: len(v) for k, v in strata.items()}, "s4_fraction": fraction,
                       "sonnet_coverage": sum(p in sonnet for p in cohort),
                       "policy_path": str(path), "policy_sha256": sha(path), "sources": hs + ss}
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "estratos.csv", index=False)
    pilot = frame.loc[(frame.modo == "vigilancia") & (frame.estrato == "S4"), "pair_id"].tolist()[:10]
    runner.save(OUT / "baseline.json", baseline)
    runner.save(OUT / "dossier_hashes.json", hashes)
    runner.save(OUT / "amostragem.json", {"seed": SEED, "modes": audit, "pilot_ids": pilot,
                                         "estratos_sha256": sha(OUT / "estratos.csv"), **metadata})
    runner.save(OUT / "custo.json", {"spent_usd": 0, "reservations": {}, "calls": {}})
    text = f"""# Pré-registro do probe exploratório de revisores clericais

Registrado em {runner.utcnow()}, antes de inferência. Seed: {SEED}.
Modelos: Claude Opus 5.5 (`claude-opus-5-5`), Claude Fable 5.1 (`claude-fable-5-1`),
Claude Haiku 5.5 (`claude-haiku-5-5`); comparador histórico Claude Sonnet 5.5 (`claude-sonnet-5-5`).
Disponibilidade será verificada na API. Nenhuma substituição de modelo.

## População e seleção
Políticas P já calculadas, não os arquivos pendentes (estes são P união B1 menos cache).
Contagens confrontadas com `data/arranjo_corte_metricas.csv`: 344 vigilância e 41 confirmação.
S1: MATCH do Haiku, 25 (21 PAR=0, quatro PAR em {{1,2}}).
S2: PAR=2 fora de S1. S3: PAR=0 com Haiku diferente de Sonnet, fora de S1.
S4: demais, amostragem aleatória simples sem reposição, peso N4/n4; demais pesos 1.
O sorteio opera sobre IDs ordenados, modos sequenciais com default_rng({SEED}).
Piloto: primeiros dez IDs ordenados da amostra S4, iguais nos dois modelos; não integram
a passada Batch principal (nova execução independente evita mistura de transporte).

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## Protocolo e execução
Prompt v4: `gzcmd/gzcmd_v4_llm_prompt.md`, SHA256 {metadata['prompt_sha256']}.
Mesmo `prepare`/`DossierView` do runner sem model_outputs, comparação de fingerprints
com os dossiês históricos antes da inferência. Nenhum dossiê ou resposta livre em data/.
Mesmo `_build_messages` por papel, `_build_arbiter_messages` e validação do runner:
A+B independentes, consenso com maior confiança; árbitro somente no desacordo.
max_tokens=2500; sem temperature, sem thinking; nenhum cache_control adicionado.
Uma passada completa para Opus e Fable; réplica de S1-S3 e de toda a confirmação
para ambos. Uma nova passada completa do Haiku nos mesmos pares, comparada ao histórico.
Batch etapa 1 A+B; etapa 2 árbitros; polling 60 segundos, checkpoint de IDs e uso,
submissões ambíguas bloqueiam retomada automática. Não persistir evidência livre;
na retomada, rebaixar resultados do lote no servidor para reconstituir opiniões.
Chamadas diretas sequenciais no piloto, retry exponencial em 429/529, no máximo dois workers.
Se Batch não funcionar em cerca de 45 minutos, registrar fallback direto (workers <=2),
sem abandonar nem duplicar lotes já submetidos e com nova projeção de custo.

## Custos e regras de parada
Preço oficial consultado em 2026-11-06: https://platform.claude.com/docs/en/about-claude/pricing.
USD/MTok entrada/saída: Opus 4/20, Fable 10/50, Haiku 0,10/0,50; Batch metade.
Piloto direto de dez S4 por modelo mede usage, chamadas/par e stop_reason.
Projetar plano inteiro usando média observada por modelo mais margem de 20%; referência
do Haiku 7.000 entrada e 4.000 saída/par até uso observado disponível.
Se >85 USD: primeiro retirar réplica Fable, depois reduzir S4 Fable à metade
(subamostra aleatória com seed fixa, peso ajustado); se ainda >85, não prosseguir.
Toda alteração constará de adendo datado antes das chamadas principais.
Teto absoluto 90 USD; reservar custo máximo por lote com tokens de entrada contados
e saída máxima de 2.500 por chamada, incluindo todas as reservas ainda em aberto.
Nenhuma submissão cujo custo máximo somado ao gasto/reservas ultrapasse 90 USD.
Cada usage é contabilizado antes do parse; timeout de resultado desconhecido mantém reserva
e bloqueia nova tentativa automática. Uso de cache, se inesperado, é cobrado à tarifa oficial.

## Métricas e interpretação
Por modelo, modo, estrato e passada: n, MATCH, NONMATCH, UNSURE, VP, FP relativos a PAR,
PAR=2 recuperados (MATCH), descartados (NONMATCH), PAR=2 UNSURE, VPP e IC de Wilson
para amostras não ponderadas; VPP em S1; totais e VPP ponderados de Horvitz-Thompson.
Não aplicar Wilson ingênuo a pseudocontagens ponderadas. Proporções de NONMATCH/UNSURE,
abstenção=UNSURE, concordância simples e kappa com Haiku/Sonnet e entre réplicas,
e número de respostas truncadas (stop_reason=max_tokens, inclusive falhas de parse).
PAR=0 não prova ausência de vínculo: discordância não demonstra erro do modelo ou do gabarito.
Probe exploratório, sem substituir resultados principais, sem inferência causal nem ajuste
de multiplicidade. Comparar diferença entre modelos à variabilidade entre réplicas.
"""
    (OUT / "PREREG.md").write_text(text, encoding="utf-8")
    print(json.dumps({m: {k: v for k, v in a.items() if k in {"population", "selected", "strata", "s4_fraction", "sonnet_coverage"}}
                      for m, a in audit.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    prepare()
