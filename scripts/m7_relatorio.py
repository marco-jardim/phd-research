"""Render the M7 report exclusively from saved experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m7_common import ROOT, OUT, save


def main():
    e1 = json.loads((OUT / "e1_summary.json").read_text(encoding="utf-8"))
    e2 = json.loads((OUT / "e2_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "e1_audit.json").read_text(encoding="utf-8"))["response"]
    calls = json.loads((OUT / "calls.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(OUT / "e2_validation.csv")
    transport = pd.read_csv(OUT / "e2_transport.csv")
    rules = pd.read_csv(OUT / "e1_rules.csv")
    pricing = {"source": "https://platform.claude.com/docs/en/about-claude/pricing",
               "accessed": "2026-11-02", "model": "claude-haiku-5-5",
               "input_usd_per_million_up_to_100k": .1, "output_usd_per_million_up_to_100k": .5,
               "input_usd_per_million_over_100k": .5, "output_usd_per_million_over_100k": 2.5,
               "note": "Public global API list price, no caching/tools/taxes/negotiated discounts; estimate, not invoice."}
    paid = []
    for call in calls:
        nt, no = call.get("input_tokens", 0), call.get("output_tokens", 0)
        paid.append({**call, "estimated_usd": (nt * (.1 if nt <= 100000 else .5)
                    + no * (.5 if nt <= 100000 else 2.5)) / 1e6})
    cost = {"pricing": pricing, "attempted_calls": len(calls), "completed_calls": sum(c["status"] == "completed" for c in calls),
            "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
            "estimated_usd": sum(c["estimated_usd"] for c in paid), "calls": paid}
    save("cost.json", cost)
    grey = validation[validation.scope == "gray"].set_index("model")
    base = grey.loc["programado"]
    high_rule = rules[(rules.rule == "crossfit_high") & (rules.scope == "high") & (rules.split == "grouped_oof_all")].iloc[0]
    gray_rule = rules[(rules.rule == "crossfit_gray") & (rules.scope == "gray") & (rules.split == "grouped_oof_all")].iloc[0]
    def table(df):
        return "```text\n" + df.to_string(index=False, float_format=lambda v: f"{v:.8f}") + "\n```"
    text = ["# M7 — programação no espaço de pesos: auditoria e escore sem gradiente", "",
        "## Veredito", "",
        "**Não incorporar como substituto do XGBoost nem como guarda automática.** O papel sustentado por este ensaio é auditoria offline/geração de hipóteses interpretáveis. Uma feature extra ou escore de transporte exigiria experimento incremental aninhado e validação externa, não executados aqui.",
        f"Área cinza: AUC programado={base.auc:.8f}; nota={grey.loc['nota'].auc:.8f}; XGBoost OOF={grey.loc['xgb_oof_isotonica'].auc:.8f}; logística L2={grey.loc['logistica_l2'].auc:.8f}. Brier ancorado do programado={base.brier:.8f}, mediana calibrada dos vínculos={base.median_positive_calibrated:.8f}.",
        "Não há evidência de que legibilidade de regras torne o escore calibrado ou uma guarda segura. Não confundir zero FP numa amostra pequena com garantia de segurança.", "",
        "## Reprodução, dados e privacidade", "", "```powershell",
        "python scripts/m7_auditoria_xgb.py", "python scripts/m7_escore_programado.py", "python scripts/m7_relatorio.py",
        "python -m pytest tests/test_m7_escore_programado.py -q", "```", "",
        "Python Miniconda; seed 42; extrato lido com dtype=str e conversão explícita por to_numeric, decimal vírgula convertido. Gabarito offline PAR∈{1,2}; identificadores/linhas de pares nunca são enviados à API. Allowlist M1 de 30 comparações/nota, sem identificadores, sexo ou datas brutas. Cache M1 validado por hashes da fonte/código/CSV e versões; folds persistidos (0 a 4), alinhados por pair_id e conferidos contra StratifiedKFold. Nenhum cache/configuração/módulo existente foi alterado.",
        "O dicionário é derivado do cabeçalho e de faixas observadas, não de um features.py inexistente neste pacote. A semântica exata de algumas proximidades/inversões não é identificável pelo cabeçalho: as descrições são hipóteses nominais, não documentação independente do comparador. Zero após imputação não distingue ausência de discordância.",
        "Todas as saídas M7 são parâmetros, código, hashes, métricas e agregados. Não há PII, pair_ids ou linhas de pares nos prompts ou artefatos M7. A chave é carregada do .env pelo transporte já existente; nunca impressa/persistida. Chamadas sequenciais com uma tentativa e ledger gravado antes do envio; checkpoints impedem repetição automática.", "",
        "## E1 — leitura como auditoria", "",
        "Fábrica _classifier de gzcmd/ml_oof.py usada pelo runner ml_xgb: 300 árvores, profundidade 6, learning_rate=0,05, subsample/colsample=0,8, scale_pos_weight=#neg/#pos, seed=42. Ajuste em fold !=1 e teste fold=1 (numeração zero-based). Dump completo com estatísticas em e1_tree_dump.json; seleção das 40 árvores com maior soma de split gains, respeitando limite de 155.000 caracteres sem cortar árvore. Somar ganhos não identifica efeito causal nem importância global independente.",
        f"Árvores efetivamente enviadas: {e1['trees_in_prompt']}. SHAP exato via booster.predict(pred_contribs=True), em margem log-odds; médias assinadas e absolutas por feature. SHAP dos {e1['gray_positive_n']} vínculos cinzentos e dos erros v3.1 de ambos os modos salvo em e1_shap.json; somente subconjuntos de treino enviados à API. Erros do motor: {json.dumps(e1['error_counts'], ensure_ascii=False)}.",
        "Prompt resumido: dicionário + dump priorizado + importâncias gain/total_gain/weight/cover + SHAP agregado de treino; pedir especificação, modos de falha ranqueados e regras booleanas (aceitação MATCH) para ≥9 e cinza. Sem linhas de pares, rótulos individuais ou exemplos identificáveis.",
        "### Especificação inferida pelo LLM", "", str(audit["specification"]), "",
        "### Hipóteses de falha (ordem do LLM, não contagens observadas)", "", "```json",
        json.dumps(audit["failure_modes"], ensure_ascii=False, indent=2), "```", "",
        "### Regras propostas", "", f"Alta: `{audit['high_rule']}`", "", f"Cinza: `{audit['gray_rule']}`", "",
        f"Síntese da avaliação independente agrupada: regra alta seleciona {int(high_rule.selected)}/95, TP={int(high_rule.tp)}, FP={int(high_rule.fp)}, cobertura={high_rule.coverage:.8f}, contra 78/95 e zero FP da guarda atual. Regra cinza seleciona {int(gray_rule.selected)}, TP={int(gray_rule.tp)}, FP={int(gray_rule.fp)}, precisão={gray_rule.precision:.8f}, recall={gray_rule.recall:.8f}. As expressões por grupo estão no JSON de auditoria; a expressão inicial acima não é reutilizada nessa avaliação.", "",
        "Cada regra é interpretada por AST restrita sobre features; não há eval de código arbitrário. As regras são avaliadas em cada escopo indicado (não são aplicadas fora dele). Precisão=TP/(TP+FP); cobertura=selecionados/N da banda; recall=TP/vínculos da banda. Colunas de erros contam erros existentes que a regra aceitaria; aceitar um FN pode resgatar vínculo, aceitar um FP piora segurança.",
        table(rules), "",
        "**OOF E1 e limites:** pooled_exploratory aplica uma regra pós-auditoria ao próprio treino e não é validação externa. A auditoria inicial não é reutilizada nas regras crossfit: suas máscaras de erros históricos usam cache M1 OOF congelado e podem ter dependência indireta dos rótulos externos. Para eliminar essa dependência e cobrir todos os cinco folds dentro de três chamadas E1, duas chamadas novas/sem histórico treinam excluindo [0,1,4] e [2,3]. Só recebem parâmetros do modelo local, SHAP de vínculos cinzentos do treino e proxies de erro calculados no treino (não máscaras do cache global). grouped_oof_all compõe essas regras independentes; fold_0…fold_4 discriminam os resultados. Nenhum rótulo do grupo avaliado entra em parâmetros, escolha de subconjuntos SHAP ou prompt. Há OOF para todos os pares sobre as mesmas partições persistidas, mas treino agrupado 40%/60%, NÃO cinco auditorias com 80% de treino. Regras/partições em e1_crossfit_audit.json. Essa diferença de tamanho de treino torna a comparação conservadora e não exatamente pareada ao treinamento 5-fold original. A guarda comparadora usa p_cal_ml OOF original em todos os pares. Referência histórica: 78/95, zero FP. Nenhuma regra nova é incorporada.", "",
        "## E2 — escrita a partir da especificação", "",
        "Duas chamadas independentes dos rótulos: (1) gerar função pura a partir do dicionário e da descrição SIM × Sinan-TB; (2) revisão semântica do primeiro código, pedindo controle de dupla contagem, raridade, ausência e migração. Nenhum parâmetro XGBoost, SHAP, métrica/rótulo de teste ou par individual é fornecido ao gerador. Refinamento é semântico, não orientado por resultados. Código limitado por AST a uma função sem I/O/imports/loops/globals, com operações escalares e features.get; todas as predições verificadas como finitas em [0,1]. Zero rótulos embutidos no código.",
        "Em cada fold externo, seleciona-se a melhor das duas funções pela AUC cinza apenas do treino. Como funções fixas não são ajustadas aos pares, essa validação de seleção não precisa refazer treino do escore. A calibração usa somente as taxas das âncoras nota≥9 e nota<5 do treino externo e a implementação fit_masked_anchor_platt/rates/predict de validar_ancoras_ml.py (mesmo kernel do pacote, Jeffreys apenas em taxas de fronteira). Logística L2 C=1, padronização ajustada só no treino, 5 margens OOF internas para ajustar o Platt por âncoras. Nota usa o mesmo Platt ancorado no treino. XGBoost usa o cache original 5×5 isotônico, não é recalibrado com informação do teste.",
        "AUC/AP são sobre o escore bruto para comparar ordenação; Brier e mediana calibrada são após Platt ancorado (XGBoost mantém isotônica). Para o programado, a entrada do Platt é logit(clip(escore,10^-6,1−10^-6)), pois o escore é limitado a [0,1] enquanto o adaptador M5 foi projetado para margens. programado_platt_direto mostra a sensibilidade a usar o escore diretamente: o limite α≤10 pode ficar ativo e tornar a comparação dependente da unidade do escore. Ambos são ajustados só no treino, sem escolher a transformação por resultados de teste. Transformações/calibradores por fold podem mudar a ordenação pooled; as duas medianas são explicitadas. e2_training_bands.csv contém AUC/AP agregadas por banda somente no treino de cada fold; essas métricas não foram enviadas ao LLM.",
        table(validation), "",
        f"Módulo gzcmd/escore_programado.py: versão {e2['saved_version']} escolhida exclusivamente no treino do fold 1; AUCs de seleção={e2['saved_selection_train_fold1_auc']}. O módulo exporta escore bruto, não calibrador/probabilidade operacional. A avaliação principal é do procedimento de seleção por fold; não é uma alegação de que uma versão global foi escolhida por desempenho no teste.", "",
        "## Transporte entre estratos PASSO", "",
        f"Corte fixado pela ordem dos valores únicos PASSO: ≤{e2['transport_passo_cut']} versus >{e2['transport_passo_cut']}. Sem rótulos para escolher corte. Cada direção reserva 25% do estrato de origem como validação (seed 42, estratificado); ajusta XGBoost e escolhe/calibra o programado apenas nos 75% restantes. Compara AUC no holdout de origem versus todo estrato destino, que nunca participa do ajuste. Queda positiva indica piora; negativa indica aumento. Métricas total/cinza, N e positivos explicitam diferenças de composição.",
        table(transport), "",
        "PASSO representa estrato de blocagem/processo, não ano/local independente: não demonstra transporte temporal/geográfico. AUC também pode mudar por case-mix e raridade; estratos sem duas classes têm AUC ausente, não zero. O programado não tem pesos supervisionados para transportar; seleção é ajustada na origem. Platt só é ajustado se há ≥30 pares em cada âncora da origem, conforme M5; insuficiência é registrada, sem substituir por taxas do destino. AUC usa escores brutos e independe desse ajuste. O XGBoost de transporte é o classificador bruto com mesma fábrica; não há ajuste no destino.", "",
        "## Chamadas e custo", "",
        f"{cost['attempted_calls']} tentativas; {cost['completed_calls']} concluídas. {cost['input_tokens']} tokens de entrada + {cost['output_tokens']} de saída. Custo estimado USD {cost['estimated_usd']:.8f} (não fatura). Todas as chamadas Claude Haiku 5.5 via AnthropicReviewer de run_exp2_anthropic.py, max_tokens=12000 e sem parâmetro temperature no Messages API. Nenhum Fireworks/OpenCode Go.",
        "Preço público consultado em 02/11/2026: entrada USD 0,10/MTok e saída USD 0,50/MTok até 100.000 tokens; acima disso USD 0,50/2,50. Sem cache, ferramentas ou prêmio regional. Fonte: https://platform.claude.com/docs/en/about-claude/pricing. Ledger e cálculo por chamada em calls.json/cost.json.", "",
        "## Limites e decisão operacional", "",
        "Separação por par, não por pessoa; registros compartilhados podem produzir dependência. Sem inferência causal de importância/SHAP, sem prova de estabilidade fora do extrato. Comparar score programado com logística separa interpretabilidade de eventual sinal linear; transportar entre PASSO é teste de sensibilidade interno. Regra falha não deve virar veto a nascimento ausente ou migração. Melhor uso imediato: especificação/auditoria, documentando hipóteses para estudo prospectivo; guarda/feature extra/escala de transporte ficam fora do motor até validação incremental e externa.", "",
        "## Artefatos e verificação", "",
        "scripts/m7_{common,auditoria_xgb,escore_programado,relatorio}.py; gzcmd/escore_programado.py; tests/test_m7_escore_programado.py; data/m7/e1_* (dump/SHAP/prompt/resposta/regras), e2_* (prompts/respostas/validação/bandas/ajustes/transporte), calls.json e cost.json. Este relatório copia números diretamente desses arquivos; não inventa métricas/uso. Verificação de testes deve ser reportada pelo executor com a saída do comando, não presumida por este renderizador."]
    verification_path = OUT / "verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        text.extend(["", "### Verificação executada", "", "```text",
                     verification["test_command"], verification["test_output"].strip(), "```",
                     f"Exit code: {verification['test_exit_code']}. Verificação de privacidade dos artefatos: {verification['privacy_check']}. Nenhum commit realizado."])
    (ROOT / "gzcmd/M7_weight_space.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(cost, ensure_ascii=False, indent=2))
    print("Report written: gzcmd/M7_weight_space.md")


if __name__ == "__main__":
    main()
