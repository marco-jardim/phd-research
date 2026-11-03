"""M8 parte B: duas chamadas agregadas via AnthropicReviewer; checkpoint e teto 10.

Reexecutar reutiliza respostas. Nenhum identificador, par, valor demográfico ou
segredo entra nos prompts/artefatos. --offline reavalia respostas sem rede.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from m8_auditoria_blocagem import load_extract, write_json  # noqa: E402

MODEL = "claude-haiku-5-5"
SYSTEM = """Você audita blocagem epidemiológica ORL. Recebe SOMENTE agregados.
Não invente observações nem trate proxy como Soundex real. Responda JSON válido,
em português. Configuração não disponível: toda inferência é 'a validar'.
Não confunda associação com identificação causal da chave; prevalência geográfica,
seleção/exportação e remoção sequencial podem explicar concordância alta.
Não identifique pessoas nem solicite linhas de pares. Não use nota/rótulo na chave.
"""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def chapter_context() -> dict:
    """Contexto real, com as especificações por passo deliberadamente retidas.

    A inferência fica cega à lista de chaves publicada; ela será confrontada no doc.
    Nunca se exportam parágrafos sobre profissionais ou metadados de pacientes.
    """
    return read_json(ROOT / "data/m8/contexto_capitulo.json")


def call_once(folder: Path, name: str, payload: dict, instruction: str, offline: bool) -> dict:
    response_path = folder / f"{name}.json"
    prompt = instruction + "\n" + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if response_path.exists():
        saved = read_json(folder / f"{name}_prompt.json")
        if saved != {"system": SYSTEM, "user": prompt}:
            raise ValueError("Checkpoint incompatível com agregados/prompt; use nova pasta auditada, preservando o teto global de chamadas")
        return read_json(response_path)
    if offline:
        raise ValueError(f"Resposta offline ausente: {name}")
    from run_exp2_anthropic import AnthropicReviewer, _extract_json

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY ausente; nenhuma chamada efetuada")
    write_json(folder / f"{name}_prompt.json", {"system": SYSTEM, "user": prompt})
    ledger_path = folder / "chamadas.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    if len(ledger) >= 10:
        raise ValueError("Teto global de 10 tentativas já atingido nesta pasta")
    record = {"etapa": name, "modelo": MODEL, "inicio_utc": datetime.now(timezone.utc).isoformat(),
              "status": "iniciada", "max_tokens": 16000,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    ledger.append(record)
    write_json(ledger_path, ledger)  # reserva ANTES da rede; sem retentativa escondida
    reviewer = AnthropicReviewer(api_key=key, base_url="https://api.anthropic.com", model=MODEL,
                                 fallback_model=MODEL, max_tokens=16000, max_retries=1, timeout_s=240)
    try:
        import time

        started = time.monotonic()
        # Mesmo cliente/extração do runner; transporte direto preserva usage mesmo
        # quando uma resposta é truncada (o wrapper original o descartaria).
        response = reviewer._get_client().messages.create(
            model=MODEL, system=SYSTEM, messages=[{"role": "user", "content": prompt}],
            max_tokens=16000)
        latency = time.monotonic() - started
        n_in, n_out = int(response.usage.input_tokens), int(response.usage.output_tokens)
        record.update(input_tokens=n_in, output_tokens=n_out, segundos=latency,
                      stop_reason=response.stop_reason)
        write_json(ledger_path, ledger)
        if response.stop_reason == "max_tokens":
            raise ValueError("Resposta truncada")
        text = "".join(getattr(b, "text", "") for b in response.content if b.type == "text")
        parsed = _extract_json(text)
    except Exception:
        record["status"] = "falhou_sem_retentativa"
        write_json(ledger_path, ledger)
        raise RuntimeError("Chamada Anthropic falhou; consultar status, sem registrar conteúdo sensível") from None
    record.update(status="concluída", segundos=latency, input_tokens=n_in, output_tokens=n_out)
    write_json(ledger_path, ledger)
    write_json(response_path, parsed)
    return parsed


def proxy_mask(df: pd.DataFrame, rule: dict, allowed: set[str]) -> pd.Series:
    """DNF declarativa: OR entre grupos, AND entre condições; sem eval/exec."""
    groups = rule.get("proxy_dnf")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 8:
        raise ValueError("proxy_dnf exige 1–8 grupos")
    result = pd.Series(False, index=df.index)
    for group in groups:
        if not isinstance(group, list) or not 1 <= len(group) <= 8:
            raise ValueError("Grupo exige 1–8 condições")
        mask = pd.Series(True, index=df.index)
        for condition in group:
            field, op, threshold = condition["campo"], condition["op"], condition["valor"]
            if field not in allowed or op not in {">=", "==", "<="}:
                raise ValueError("Campo ou operador fora da allowlist")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
                raise ValueError("Limiar precisa ser finito")
            x = df[field]
            compared = {">=": x.ge, "==": x.eq, "<=": x.le}[op](threshold)
            mask &= x.notna() & compared
        result |= mask
    return result


def evaluate(input_path: Path, folder: Path, proposals: dict) -> pd.DataFrame:
    df = load_extract(input_path)
    allowed = set(read_json(folder / "dicionario_campos.json"))
    known = df[df["y"]].drop_duplicates(["COMPREC", "REFREC"])
    rows = []
    rules = proposals.get("chaves", [])
    if not 1 <= len(rules) <= 5:
        raise ValueError("São exigidas de 1 a 5 propostas")
    for rule in rules:
        mask = proxy_mask(df, rule, allowed)
        matched = proxy_mask(known, rule, allowed)
        rows.append({"id": rule["id"], "chave": rule["chave"],
                     "justificativa": rule["justificativa"], "campos": json.dumps(rule["campos"], ensure_ascii=False),
                     "erro_recuperado": rule["erro_recuperado"],
                     "custo_llm_ordem_grandeza": rule["custo_ordem_grandeza"],
                     "vinculos_concordantes": int(matched.sum()), "vinculos_conhecidos": len(known),
                     "recall_proxy": float(matched.mean()), "linhas_concordantes": int(mask.sum()),
                     "linhas_extrato": len(df), "volume_relativo_proxy": float(mask.mean()),
                     "pares_distintos_concordantes": len(df[mask].drop_duplicates(["COMPREC", "REFREC"])),
                     "ordem_grandeza_no_extrato": 10 ** int(np.floor(np.log10(mask.sum()))) if mask.any() else 0,
                     "proxy_dnf": json.dumps(rule["proxy_dnf"], ensure_ascii=False),
                     "limitacao": "Somente pares já gerados; não estima recall absoluto, novos vínculos ou custo nas bases completas. Soundex/ano não verificáveis pelos escores."})
    result = pd.DataFrame(rows)
    result.to_csv(folder / "chaves_propostas_avaliacao.csv", index=False)
    return result


def md_table(frame: pd.DataFrame) -> str:
    """Markdown sem dependência opcional tabulate; números vêm dos artefatos."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.6f}" if np.isfinite(value) else "não estimável"
        return str(value).replace("|", "/").replace("\n", "<br>")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def render_report(folder: Path, inferred: dict, proposals: dict, results: pd.DataFrame) -> None:
    pre = read_json(folder / "precondicao.json")
    cost = read_json(folder / "custo.json")
    yields = pd.read_csv(folder / "blocagem_rendimento_por_passo.csv")
    agreements = pd.read_csv(folder / "concordancia_por_passo.csv")
    profiles = pd.read_csv(folder / "perfil_vinculos.csv")
    fields = ["NOME qtd frag iguais", "NOMEMAE qtd frag iguais", "DTNASC dt iguais", "CODMUNRES local igual", "SEXO igualdade"]
    focus = profiles[profiles.campo.isin(fields)]
    inferred_table = pd.DataFrame([{"passo": p["passo"], "inferência LLM": p["chave_inferida"], "status": "a validar"}
                                   for p in sorted(inferred["passos"], key=lambda p: int(p["passo"]))])
    selected = agreements[agreements.campo.isin(["NOME prim frag igual", "NOME ult frag igual", "NOMEMAE prim frag igual", "NOMEMAE ult frag igual", "CODMUNRES uf igual", "CODMUNRES local igual", "SEXO igualdade"])]
    agreement_table = selected.pivot(index="campo", columns="passo", values="fracao_exata_proxy").reset_index()
    matrix = pd.read_csv(folder / "blocagem_sobreposicao.csv")
    candidate_matrix = matrix.pivot(index="passo_a", columns="passo_b", values="pares_em_comum").reset_index()
    match_matrix = matrix.pivot(index="passo_a", columns="passo_b", values="vinculos_em_comum").reset_index()
    text = f"""# M8 — leitura da blocagem do OpenRecLink a partir de seus produtos

## 1. Pré-condição: resultado antes de qualquer estimativa

**{pre['linhas']:,} linhas; {pre['pares_distintos']:,} pares distintos; {pre['pares_multiplos_passos']} pares presentes em mais de um PASSO.**
**{pre['linhas_vinculo']} linhas de vínculo; {pre['vinculos_distintos']} vínculos distintos; {pre['vinculos_multiplos_passos']} vínculos recapturados.**
O extrato não é perfeitamente desduplicado por par: há 12 repetições entre passos,
todas de não vínculos. Entretanto, os vínculos não têm recapturas. Além disso,
capitulo5.tex:40,48 documenta exclusão progressiva dos registros Sinan já pareados.
Assim, **captura-recaptura é inviável**, não apenas imprecisa. Não se deve substituir
os zeros por uma pseudocontagem e apresentar um total artificial.

## 2. Motivação, método e privacidade

A motivação é a leitura de um sistema a partir dos produtos que deixa observar,
inspirada em Takehi et al. (2026), *Weight-Space Programming*, post de blog indicado
no enunciado. Trata-se de analogia metodológica, não replicação nem validação desse
trabalho; a parte de edição de pesos não se aplica ao ORL. Não há alteração da
configuração, do software de origem, dos dados nominais ou de pesos de modelos.

Leitura `dtype=str`, delimitador ponto-e-vírgula e `to_numeric` explícito, convertendo
decimal vírgula. Vínculo: PAR em {{1,2}}. Identidade de par: (COMPREC, REFREC);
`pair_id = sha256("COMPREC|REFREC|PASSO")[:16]` é calculado apenas em memória, nunca
exportado. Matrizes usam conjuntos, não contagens repetidas. Rendimento distingue
linhas e pares distintos; quantis são por linha, frações positivas por par distinto.
Campos vêm do cabeçalho e de `gzcmd/loader.py` (a função de engenharia disponível;
`gzcmd/features.py` não existe neste checkout). Usa-se allowlist dos 29 subescores,
mais igualdade local de sexo (somente códigos válidos 1/2/M/F). Não se exportam
sexo individual, nomes, datas, municípios, identificadores, pair_id nem linhas de
pares; apenas agregados, descrições, pseudocódigo e proveniência. Ausência não
equivale a concordância: fica fora do numerador, com denominador total do passo.
Indicadores de composição, abreviação e inversão não são tratados como igualdade.
Limiar alto operacional >=0,8 não normaliza os tetos distintos dos subescores.

**Duas escalas diferentes:** aqui as faixas solicitadas [5,9) e >=9 usam **nota final
do comparador**. O capítulo 5:21 e a tabela tab_score_bands usam **SCORE do ORL**
para sua área cinza. Portanto não se espera reproduzir os 151/247 vínculos cinzentos
do capítulo com a nota final. Esta diferença é registrada, não silenciosamente
resolvida trocando a variável. Quantis e frações deste relatório são da nota final.

## 3. Rendimento por passo

{md_table(yields)}

São rendimentos **do que foi exportado**, condicionais à seleção dos passos e à
remoção sequencial. 'Só neste passo' significa exclusivo **neste extrato**, não que
nenhuma outra chave recuperaria o vínculo. Tardios significa PASSO > 6 (9 e 11),
não uma sequência de captura independente. A sequência numérica é a descrita no
capítulo; a execução completa dos 30 passos não foi observada.

## 4. Sobreposição

Pares candidatos (diagonal = número de pares distintos):

{md_table(candidate_matrix)}

Vínculos (diagonal = número de vínculos distintos):

{md_table(match_matrix)}

Entre 6 e 11 há 9 pares comuns; entre 9 e 11, 3; entre 6 e 9, zero. Todos os
vínculos exportados são exclusivos. Isso **não demonstra completude da blocagem**.

## 5. Captura-recaptura: não estimar neste extrato

`blocagem_captura_recaptura.csv` registra status inviável; total, não capturados e
IC95% ficam ausentes, não zero. Os 247 observados coincidem com capitulo5.tex:43–48
(80+127+40) e capitulo6.tex:9. Essas seções não oferecem número independente de
vínculos 'esperados' para comparação: 247 é o total observado, não a população
verdadeira. Não inferimos sensibilidade absoluta ou número de vínculos perdidos.

Para futuros históricos completos, o script oferece Chapman para duas listas:
N̂=((nA+1)(nB+1)/(m+1))-1;
Var(N̂)=(nA+1)(nB+1)(nA-m)(nB-m)/((m+1)²(m+2)); IC normal aproximado 95%,
truncado inferiormente à união observada. Sem sobreposição positiva a função
recusa a estimativa. Para 3+ listas há Poisson GLM via statsmodels sobre todos os
2^k−1 perfis, incluindo frequências zero: intercepto + efeitos principais +
interações duplas (interações de primeira ordem entre listas); a célula ausente é
exp(intercepto), com IC95% delta na escala log. Com 3 listas o modelo é saturado
nos 7 perfis observados: não há graus de liberdade residuais para testar ajuste.
O teste sintético confirma a fórmula, não sua adequação a este extrato.

Hipóteses: população fechada, captura homogênea e classificação correta; para
Chapman, independência entre listas; para GLM, ausência de interações de ordem
superior às incluídas e identificabilidade. Dependência positiva de chaves tende
a subestimar o total sob independência (leitura de limite inferior, não garantia
matemática universal). Exclusão progressiva induz dependência estrutural/negativa
e truncamento por desenho, que **não** se corrige chamando a estimativa de limite
inferior. A hipótese de listas comparáveis foi violada; não há IC válido aqui.

## 6. Perfil dos vínculos tardios e de baixa concordância nominal

Fração abaixo de 0,8 e fração zero são discordância/baixa similaridade **proxy**,
não prova de erro nem separação entre ausência e divergência textual. Nome baixo
foi pré-definido como `NOME qtd frag iguais < 0,5` (1 vínculo; grupo muito pequeno,
sem inferência populacional). Os tardios exclusivos somam 167 vínculos.

{md_table(focus)}

No passo 11, 37/40 não têm concordância exata de data, 33/40 têm concordância da
mãe <0,8 e 13/40 têm concordância do nome <0,8. Nos tardios combinados, 128/167 têm
zero em fragmentos iguais da mãe, 53/167 não têm data exata e 26/167 têm nome <0,8.
No passo 6, data exata é zero em todos os 80 vínculos: investigar o comparador e a
exportação antes de interpretar isso como qualidade intrínseca das datas.

## 7. Chaves inferidas — a validar

A primeira chamada recebe apenas a tabela agregada de todos os candidatos por
passo, dicionário e descrição geral do ORL extraída do capítulo 5:21,40,73. As
chaves enumeradas nas linhas 43–45 foram **retidas do prompt**, para não fabricar
uma 'inferência' que apenas copia a documentação. Soundex, comparação APROX e
distância de edição são documentados; Jaro-Winkler específico não foi confirmado
pelas passagens consultadas e não é atribuído como fato. Ano não é observado
diretamente nos subescores; inversão de ano não significa igualdade de ano.

Frações de concordância exata proxy em todos os pares de cada passo:

{md_table(agreement_table)}

{md_table(inferred_table)}

**Confronto posterior com a documentação, não configuração validada:**

| Passo | Chave descrita no capítulo 5:43–45 | Conferência na configuração ORL |
| --- | --- | --- |
| 6 | Soundex primeiro+último bloco do nome e primeiro+último bloco da mãe | a validar |
| 9 | Soundex primeiro+último bloco do nome + ano de nascimento + sexo | a validar |
| 11 | Soundex primeiro+último bloco do nome + sexo + município | a validar; linha 70 menciona só primeiro fragmento |

O LLM não recuperou componentes fonéticos nem ano. UF≈100% é principalmente
compatível com homogeneidade geográfica, e não confirma que UF seja chave. A
inferência UF+município inclui redundância. **Discordância textual não exclui
igualdade fonética**, ao contrário do que alguns trechos brutos do LLM sugerem.
Sua frase de que não há número de candidatos por passo também é incorreta: `n`
estava no agregado. Preservam-se respostas brutas como material auditável, não
como configuração reconstruída. As inferências completas e pseudocódigo estão
em `chaves_inferidas.json`, todas com status 'a validar'.

## 8. Novas chaves e avaliação local

Segunda tarefa LLM: recebe agregados do perfil, inviabilidade da recaptura,
concordâncias, contexto geral e inferências. Propostas estruturadas em DNF são
validadas por allowlist; sem `eval`, código gerado ou execução de pseudocódigo.
São **chaves alternativas a testar**, não melhorias demonstradas. Para retirar
dependência do primeiro nome, prioriza-se discussão de K1/K4 (data exata + sexo e
geografia), K2 (último fragmento + sexo + município), K3 (últimos fragmentos do
nome e da mãe + município). O custo declarado pelo LLM é hipótese, separado do
volume efetivamente medido:

{md_table(results[['id', 'chave', 'vinculos_concordantes', 'vinculos_conhecidos', 'recall_proxy', 'linhas_concordantes', 'linhas_extrato', 'volume_relativo_proxy', 'ordem_grandeza_no_extrato']])}

Recall proxy = vínculos conhecidos concordantes / 247; volume relativo = linhas
concordantes / 61.696. Os 12 pares repetidos ficam no denominador de linhas pedido;
CSV também registra pares distintos concordantes. Não se estima incremento: todos
os 247 vínculos já foram gerados por alguma chave. Os proxies de igualdade literal
não simulam uma chave Soundex; não se multiplicam estes volumes para obter pares
nas bases originais. Regras, justificativas, tipo de erro, custo sugerido, limitações
e pseudocódigo estão em `chaves_propostas.json` e no CSV de avaliação.

**Auditoria crítica das propostas:** K1: 116 linhas (ordem 10²), compatível com a
faixa sugerida 10²–10³; K2: 58.451 (10⁴), mas cobre 94,74% do extrato, custo proxy
muito alto; K3: 21.855 (10⁴), não os 10³ sugeridos; K4: 128 (10²), não os 10³–10⁴
sugeridos. Em K4 o LLM inverteu a métrica: 0,317365 é a fração **sem** data exata
nos tardios, portanto a concordância é 0,682635, não 0,317. Suas alegações de
quantos casos com erro nominal seriam recuperados não foram medidas por aquelas
interseções e não são resultados da auditoria. K2 pode já ser parte de uma chave
existente; 'nova' significa combinação candidata, não novidade comprovada diante
dos outros 27 passos não exportados. A seleção de propostas sobre o próprio
gabarito torna o recall descritivo/in-sample, não validação externa.

## 9. Chamadas e custo

{cost['tentativas']} tentativas, {cost['concluidas']} concluídas; uma resposta de propostas truncada em 8.000
tokens foi descartada pelo wrapper original. Não houve retry automático; a nova
tentativa usou o mesmo prompt com max_tokens=16.000. As chamadas iniciais usaram
8.000; todas >=4.000, somente Claude Haiku 5.5, via cliente de
`scripts/run_exp2_anthropic.py`, sem parâmetro temperature, Fireworks ou OpenCode Go.
Ledger persistido antes de cada request, teto de 10, checkpoints reusados offline.

Usage preservado das concluídas: {cost['input_tokens']} tokens de entrada e
{cost['output_tokens']} de saída. Estimativa adicional da truncada: {cost['input_tokens_estimados_falhas']}
de entrada (prompt com SHA256 idêntico à tentativa concluída) e
{cost['output_tokens_estimados_falhas']} de saída (teto atingido); esses valores são
**reconstruídos, não usage preservado**. Subtotal pelas chamadas com usage:
USD {cost['subtotal_usage_usd']:.8f}; total incluindo reconstrução:
**USD {cost['estimativa_usd']:.8f}**, estimativa e não fatura.
Tarifa pública consultada em 02/11/2026: entrada USD 0,10/MTok e saída 0,50/MTok
para prompts até 100.000 tokens (todos abaixo do limite), sem cache/prêmios.
Fonte: https://platform.claude.com/docs/en/about-claude/pricing.
Contagens e premissas estão em `chamadas.json` e `custo.json`.

## 10. Reproduzir e próximos passos

```powershell
# Python Miniconda; pandas, numpy, scipy; statsmodels para funções/teste GLM.
python scripts/m8_auditoria_blocagem.py
# anthropic + python-dotenv e ANTHROPIC_API_KEY no .env, nunca impresso/gravado:
python scripts/m8_chaves_llm.py --input-usd-mtok 0.10 --output-usd-mtok 0.50
# Respostas e relatório existentes, sem rede:
python scripts/m8_chaves_llm.py --offline --input-usd-mtok 0.10 --output-usd-mtok 0.50
python -m pytest tests -q
```

A parte A não importa o cliente e não acessa rede. `statsmodels` não estava
instalado no Python disponível e foi instalado para verificar o caso sintético;
nenhuma dependência/configuração existente do repositório foi editada. Proveniência
dos dados/script e versões em `proveniencia_a.json`. Partes A/B têm testes de
rendimento, sobreposição, Chapman com resposta conhecida, Poisson com célula
ausente conhecida e avaliação declarativa de proxies. Verificação de testes deve
ser conferida com a saída do executor, não presumida pelo renderizador.

Exige acesso controlado às bases identificadas e reexecução do ORL:
1. conferir a configuração real de todos os 30 passos, a divergência do passo 11
   e o momento exato da exclusão/exportação;
2. executar cada chave sobre o mesmo universo original, **sem exclusão entre
   listas**, preservando histórico multilista por par, inclusive pares não aceitos;
3. auditar comparações de data (especialmente o zero estrutural do passo 6),
   equivalência das normalizações/códigos fonéticos e valores ausentes;
4. testar propostas com índices de igualdade reais, medir pares gerados, tempo,
   memória e sobreposição incremental; não executar os proxies como classificador;
5. revisar amostra independente de novos candidatos e pares fora da blocagem para
   estimar sensibilidade absoluta e vieses. Só então reconsiderar recaptura e IC.

Não se alteraram módulos `gzcmd/*.py`, YAML, capítulo, sprint3b ou scripts
proibidos; não foi feito commit. Artefatos M8 contêm somente agregados/metadados.
"""
    (ROOT / "gzcmd/M8_blocagem.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/m8")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-usd-mtok", type=float)
    parser.add_argument("--output-usd-mtok", type=float)
    args = parser.parse_args()
    folder = args.out_dir
    provenance = read_json(folder / "proveniencia_a.json")
    if provenance["input_sha256"] != hashlib.sha256(args.input.read_bytes()).hexdigest():
        raise ValueError("Extrato diverge da proveniência da parte A; não misturar agregados e avaliação")
    context = chapter_context()
    write_json(folder / "contexto_capitulo.json", context)
    fields = read_json(folder / "dicionario_campos.json")
    agreements = pd.read_csv(folder / "concordancia_por_passo.csv").round(6).to_dict("records")
    inferred = call_once(folder, "chaves_inferidas", {"contexto": context, "dicionario": fields, "concordancia": agreements},
        "Infira as chaves de CADA passo a partir das frequências agregadas. Campos >=99% são pistas, não prova. Diferencie exato textual, fonético e alta similaridade. Não atribua ano se indisponível. JSON: {passos:[{passo, status:'a validar', chave_inferida, pseudocodigo, evidencia, alternativas, confianca}], limitacoes:[...]}. Pseudocódigo deve operar em registros normalizados, não no escore final. Inclua incerteza e ausência de observação de Soundex/ano.", args.offline)
    if {str(p["passo"]) for p in inferred["passos"]} != {str(r["passo"]) for r in agreements}:
        raise ValueError("Inferência deve cobrir exatamente os passos observados")
    for step in inferred["passos"]:
        step["status"] = "a validar"
    write_json(folder / "chaves_inferidas.json", inferred)
    profiles = pd.read_csv(folder / "perfil_vinculos.csv").round(6)
    # JSON null, nunca NaN, para grupos vazios.
    profiles_json = json.loads(profiles.to_json(orient="records"))
    proposals = call_once(folder, "chaves_propostas", {
        "contexto": context, "dicionario": fields,
        "perfil_vinculos": profiles_json, "concordancia": agreements,
        "precondicao": read_json(folder / "precondicao.json"),
        "captura_recaptura": pd.read_csv(folder / "blocagem_captura_recaptura.csv").fillna("não estimável").to_dict("records"),
        "chaves_inferidas": inferred},
        "Proponha de 1 a 5 NOVAS chaves alternativas, recuperando erros nos vínculos tardios/nome baixo e relaxando pelo menos um componente nominal. Não use nota, PAR, identificadores, composição comum/raro ou presença de abreviação como chave. Chave real deve ser igualdade de campos normalizados, fonéticos ou componentes demográficos, não classificador posterior. Para cada chave, traduza em proxy avaliável SOMENTE com campos do dicionário. Ano/sexo/Soundex sem proxy confiável devem ser declarados, nunca tratados como observados. Custo absoluto em bases originais é não identificável: expresse apenas ordem de grandeza hipotética no extrato de 61696 linhas e indique isso. JSON: {chaves:[{id, chave, pseudocodigo, justificativa, campos:[...], erro_recuperado, custo_ordem_grandeza, limitacao_proxy, proxy_dnf:[[{campo,op:'>=|==|<=',valor:numero}]]}],limitacoes:[...]}. DNF é OR de grupos AND. Use apenas nomes exatos do dicionário, números e operadores >=, ==, <=. Em vez de inventar correspondência fonética, explicite aproximação por fragmentos iguais e a potencial subestimação. No máximo 5 chaves.", args.offline)
    results = evaluate(args.input, folder, proposals)
    calls = read_json(folder / "chamadas.json")
    n_in = sum(c.get("input_tokens", 0) for c in calls)
    n_out = sum(c.get("output_tokens", 0) for c in calls)
    # Recuperação explícita da primeira truncada: mesmo prompt, tokens de entrada
    # inferidos pela tentativa concluída. Não reclassificar como usage observado.
    for call in calls:
        if call.get("output_tokens_estimados") and not call.get("input_tokens_estimados"):
            twin = next((c for c in calls if c["prompt_sha256"] == call["prompt_sha256"] and "input_tokens" in c), None)
            if twin:
                call["input_tokens_estimados"] = twin["input_tokens"]
                call["fonte_estimativa_input"] = "usage de tentativa com prompt_sha256 idêntico; não usage preservado da falha"
    write_json(folder / "chamadas.json", calls)
    missing_in = sum(c.get("input_tokens_estimados", 0) for c in calls)
    missing_out = sum(c.get("output_tokens_estimados", 0) for c in calls)
    priced = args.input_usd_mtok is not None and args.output_usd_mtok is not None
    write_json(folder / "custo.json", {"tentativas": len(calls), "concluidas": sum(c["status"] == "concluída" for c in calls),
        "input_tokens": n_in, "output_tokens": n_out,
        "input_tokens_estimados_falhas": missing_in, "output_tokens_estimados_falhas": missing_out,
        "input_usd_mtok": args.input_usd_mtok, "output_usd_mtok": args.output_usd_mtok,
        "subtotal_usage_usd": (n_in * args.input_usd_mtok + n_out * args.output_usd_mtok) / 1e6 if priced else None,
        "estimativa_usd": ((n_in + missing_in) * args.input_usd_mtok + (n_out + missing_out) * args.output_usd_mtok) / 1e6 if priced else None,
        "preco_fonte": "https://platform.claude.com/docs/en/about-claude/pricing (02/11/2026)",
        "nota": "Inclui reconstrução explicitada de usage da truncada, não fatura. Sem preço confirmado, custo monetário não verificado."})
    if priced:
        render_report(folder, inferred, proposals, results)
    print(results[["id", "vinculos_concordantes", "recall_proxy", "linhas_concordantes", "volume_relativo_proxy"]].to_string(index=False))
    print(json.dumps(read_json(folder / "custo.json"), ensure_ascii=False))


if __name__ == "__main__":
    main()
