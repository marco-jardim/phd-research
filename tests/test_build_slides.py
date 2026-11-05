"""Contrato de compilação e critérios editoriais dos slides de defesa."""

import csv
from pathlib import Path
import re
import runpy
import subprocess
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "text" / "slides"
HOURS = ROOT / "data" / "carga_abaixo_corte_horas.csv"


def read_csv(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one_row(rows, **criteria):
    selected = [
        row for row in rows if all(row[key] == value for key, value in criteria.items())
    ]
    assert len(selected) == 1, criteria
    return selected[0]


def integer(value):
    return str(int(float(value)))


def pt_number(value, places=1):
    return f"{float(value):,.{places}f}".translate(str.maketrans(",.", ".,"))


def pt_percent(value, places=1):
    return pt_number(100 * float(value), places)


def slide_frames():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    return re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)


def assert_fragments(frame, required):
    normalized = frame.replace("{,}", ",")
    for fragment in required:
        assert fragment in normalized, fragment


def load_builder():
    return runpy.run_path(str(SLIDES / "build_slides.py"))["build"]


def test_build_requires_pdflatex(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="pdflatex não encontrado"):
        load_builder()()


def test_build_runs_two_passes(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(SLIDES / "build_slides.py"))
    build = namespace["build"]
    build.__globals__["__file__"] = str(tmp_path / "build_slides.py")
    (tmp_path / "defesa.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("shutil.which", lambda _: "pdflatex")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    assert build() == tmp_path / "defesa.pdf"
    assert run.call_count == 2
    for call in run.call_args_list:
        assert call.kwargs["cwd"] == tmp_path
        assert call.kwargs["check"] is True
        assert call.kwargs["timeout"] == 600
        assert isinstance(call.kwargs["env"], dict)
        assert call.args[0][-1] == "defesa.tex"


def test_build_propagates_compilation_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "pdflatex")
    run = Mock(side_effect=subprocess.CalledProcessError(1, "pdflatex"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        load_builder()()
    assert run.call_count == 1


def test_slide_structure_and_editorial_contract():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    main, backup = source.split(r"\appendix")
    assert main.count(r"\begin{frame}") == 30
    backup_frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", backup, re.S)
    assert backup_frames
    assert len(backup_frames) == 16
    assert len(backup_frames) == backup.count(r"\begin{frame}")
    for frame in backup_frames:
        title = re.match(r"\{Backup:\s*([^{}]+)\}", frame)
        assert title and title.group(1).strip()
        assert re.search(r"\\fonte(?:\[[^\]]*\])?\{[^}]+\}", frame)
    frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)
    assert sum("Fellegi" in frame for frame in frames) == 1
    assert source.index("Limitações: referência") < source.index("Conclusões:")
    cutoff = one_row(
        read_csv("arranjo_corte_metricas.csv"),
        modo="vigilancia", braco="corte_sozinho", faixa="TODAS",
    )
    assert (
        f"{integer(cutoff['aceitos'])}/{integer(cutoff['vinculos'])} = "
        f"{pt_percent(cutoff['sens_auto'])}"
    ) in source
    modal_f1 = max(
        float(row["f1"]) for row in read_csv("ablation_results.csv")
        if row["category"] == "ml-only" and "XGBoost" in row["config"]
    )
    assert f"Fluxo Modal & XGBoost & {pt_number(modal_f1, 3)}" in source
    assert any(
        "base inteira (comparação)" in frame
        and "REFREC" in frame
        and "tab:cap8-cascata-v31" in frame
        for frame in backup_frames
    )
    for path in [SLIDES / "defesa.tex", SLIDES / "roteiro_defesa.md"]:
        text = path.read_text(encoding="utf-8")
        assert not re.search("zona (?:cinza|cinzenta)|[\u2013\u2014]", text, re.I)
        assert "[ATUALIZAR" not in text
        if not HOURS.exists():
            assert text.count("% CONFERIR horas") == 1
            text = text.replace("% CONFERIR horas", "")
        assert "CONFERIR" not in text


def test_script_has_twenty_nine_short_answers():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    answers = re.findall(r"### \d+\..*?\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 29
    assert [int(n) for n in re.findall(r"^### (\d+)\.", questions, re.M)] == list(
        range(1, 30)
    )
    assert all(len(answer.split()) <= 70 for answer in answers)
    counts = [len(answer.split()) for answer in answers]
    recorded = re.search(r"São 29 respostas, com (.*?) palavras, respectivamente", source)[1]
    assert [int(n) for n in re.findall(r"\d+", recorded)] == counts


def scientific_requirements(csv_loader=read_csv):
    metrics = csv_loader("arranjo_corte_metricas.csv")
    par2 = csv_loader("arranjo_corte_par2.csv")
    llm = csv_loader("arranjo_corte_vpp_llm.csv")
    mechanisms = csv_loader("arranjo_corte_perdas_mecanismo.csv")
    robustness = csv_loader("robustez_arranjo_resumo.csv")
    groups = csv_loader("equidade_arranjo_group_metrics.csv")
    load = csv_loader("carga_abaixo_corte.csv")
    steps = csv_loader("par_por_passo.csv")
    cutoff = one_row(metrics, modo="vigilancia", braco="corte_sozinho", faixa="TODAS")
    principal = {
        mode: one_row(metrics, modo=mode, braco="P", faixa="TODAS")
        for mode in ("vigilancia", "confirmacao")
    }
    lower = one_row(load, limiar="por_passo", passo="TOTAL")
    global_lower = one_row(load, limiar="global", passo="TOTAL")
    requirements = {
        3: ["preservar o corte operacional", "mesmo orçamento humano", "falsos positivos"],
        4: ["Hipótese de ordenação", "fora da amostra", "não superou essa ordenação simples",
            "duas fronteiras", "não registrada", "mesmo XGBoost OOF", "(LLM)", "(CdR)", "P:"],
        5: [],
        6: ["Cenário exploratório", "se a revisão tivesse seguido", "Não é uma reconstrução histórica",
            pt_number(lower["pares_acima_limiar"], 0), pt_number(global_lower["pares_acima_limiar"], 0)],
        7: ["Revisora única, sem classificação independente", "CdR no passo aberto",
            "não nos passos 6, 9 e 11", "viés de verificação e incorporação"],
        13: ["OOF", "REFREC", "A política v3.1 compara custos, não usa esse corte único"],
        19: [],
        20: ["figures/curva_esforco.pgf", "não registrada", "Revisor perfeito é hipótese",
             "erro de 5,0", "favorece O1 por construção", "mesmo XGBoost OOF"],
        21: ["impede interpretação prospectiva", "Recomendação, não resultado observado",
             "encaminhar MATCH do LLM ao revisor humano", "não à aceitação automática",
             "avaliação registrada própria"],
        22: ["principal porque foi registrado antes", "dominado por P-Platt",
             "análise de sensibilidade", "não confirmatória", "não sustentou a dominância",
             "A política não superou a ordenação simples", "validação em dados novos"],
        23: ["proporção de vínculos", "definida pelo corte", "FP abaixo do corte"],
        24: [r"Disparidade herdada do corte e do \textit{label}", "Benjamini-Hochberg (vig./conf.)"],
        26: ["Respostas históricas reutilizadas", "conhecido antes do registro",
             "recalibração, braços e avaliação", "desvios de protocolo", "não generalização"],
        28: ["não superou a ordenação simples", "Horas são ordem de grandeza, não medida",
             "o serviço decide se aceita a troca", "Não há superioridade em acurácia sobre o serviço"],
        29: ["Preservar o corte", "ordem decrescente", r"\texttt{p\_cal\_ml}",
             "MATCH do LLM ao revisor humano, não à aceitação automática", "registro interno",
             "validação registrada em dados novos", "Decomposição por PAR", "Curva de esforço",
             "Falha de transferência", "ancoragem", "quatro ações", "benchmark"],
        30: ["não uma melhora de acurácia sobre o serviço", "não reproduz",
             "A política não superou a ordenação simples", "recomendação exploratória",
             "validação registrada em dados novos"],
        43: [],
        44: [],
        45: [],
        46: ["não confirmatória", "não generalização", "em relação à partição original",
             "A réplica original é um ponto dessa distribuição"],
    }
    reference = csv_loader("referencia_comparador_pos_orl.csv")
    two_frontiers = one_row(reference, estrategia="pos_orl_revisao_humana_area_cinza")
    requirements[6].extend((
        f"{pt_number(two_frontiers['pares_ao_humano'], 0)} ao revisor humano",
        f"{pt_percent(two_frontiers['sens_com_humano'])}\\% projetados sob erro de "
        f"{pt_percent(two_frontiers['erro_humano_hipotese'])}\\%",
    ))
    requirements[6].append(f"{integer(cutoff['aceitos'])}/{integer(cutoff['vinculos'])} = {pt_percent(cutoff['sens_auto'])}")
    baseline = one_row(reference, estrategia="orl_corte_servico")
    for band in ("lt5", "area_cinza", "ge9"):
        requirements[7].append(
            f"{pt_number(baseline['n_' + ('nota_' + band if band != 'area_cinza' else band)], 0)} & "
            f"{integer(baseline['vinculos_' + ('nota_' + band if band != 'area_cinza' else band)])}"
        )
    oof = one_row(csv_loader("desempenho_faixas_oof_refrec.csv"), faixa="area_cinza")
    requirements[13].extend((f"AUC {pt_number(oof['auc'], 3)}", f"(AP) {pt_number(oof['ap'], 3)}"))
    requirements[13].extend((f"{integer(oof['vp'])} verdadeiros positivos", f"{integer(oof['fp'])} falsos positivos",
                             f"{integer(oof['fn'])} falsos negativos"))
    mechanisms_order = ("guarda_nota_lt_5", "AUTO_NONMATCH_custo", "NONMATCH_LLM")
    labels = {"vigilancia": "Vigilância", "confirmacao": "Confirmação"}
    for mode, label in labels.items():
        p = principal[mode]
        observed = one_row(par2, modo=mode, braco="P")
        published = one_row(par2, modo=mode, braco="publicada")
        requirements[19].append(f"{integer(cutoff['aceitos'])}+{integer(observed['aceitos'])}")
        table = [label]
        for field in ("aceitos", "humano", "perdidos"):
            table.extend((integer(observed[field]), integer(published[field])))
        requirements[45].extend((" & ".join(table), pt_percent(observed["perdidos_proporcao"])))
        requirements[45].append("/".join(
            integer(one_row(mechanisms, modo=mode, braco="P", mecanismo=mechanism)["vinculos_perdidos"])
            for mechanism in mechanisms_order
        ))
        reserve = 43 if mode == "vigilancia" else 44
        requirements[reserve].append(
            f"({integer(cutoff['aceitos'])}+{integer(observed['aceitos'])})+{integer(p['perdidos'])}+"
            f"{integer(p['vinc_humano'])}={integer(cutoff['vinculos'])}"
        )
        requirements[reserve].append(f"IC 95\\% {pt_percent(p['ic95_inf'])} a {pt_percent(p['ic95_sup'])}")
        published_metrics = one_row(metrics, modo=mode, braco="publicada", faixa="TODAS")
        requirements[reserve].append(f"& {pt_percent(published_metrics['sens_auto'])}\\%")
        requirements[reserve].append(
            f"Fechamento: ${integer(published_metrics['aceitos'])}+{integer(published_metrics['perdidos'])}+"
            f"{integer(published_metrics['vinc_humano'])}={integer(published_metrics['vinculos'])}$"
        )
    cost_losses = one_row(
        mechanisms, modo="confirmacao", braco="P", mecanismo="AUTO_NONMATCH_custo"
    )
    requirements[30].extend((
        f"{integer(principal['vigilancia']['perdidos'])} vínculos na vigilância e "
        f"{integer(principal['confirmacao']['perdidos'])} na confirmação",
        f"{integer(principal['vigilancia']['fp'])} e "
        f"{integer(principal['confirmacao']['fp'])} falsos positivos",
    ))
    requirements[45].extend((
        f"{integer(cost_losses['vinculos_perdidos'])} perdas por AUTO\\_NONMATCH",
        "Sem varredura de custos no arranjo",
    ))
    requirements[45].append(
        f"{integer(principal['vigilancia']['perdidos'])} e "
        f"{integer(principal['confirmacao']['perdidos'])}"
    )
    for mode, arm, label in (("vigilancia", "P", "P, vigilância"),
                             ("confirmacao", "P", "P, confirmação"),
                             ("vigilancia", "P-Platt", "P-Platt, vigilância")):
        row = one_row(llm, modo=mode, braco=arm)
        requirements[21].append(
            f"{label} & {integer(row['VP'])}/{integer(row['aceitacoes_llm_abaixo'])} & "
            f"{pt_percent(row['VPP'])}\\% & "
            f"{pt_percent(row['ic95_inf'])} a {pt_percent(row['ic95_sup'])}\\%"
        )
    llm_p = one_row(llm, modo="vigilancia", braco="P")
    requirements[21].append(
        f"{integer(llm_p['FP'])} FP observados contra {pt_number(llm_p['e_FP_previstas'], 2)} previstos"
    )
    platt = one_row(metrics, modo="vigilancia", braco="P-Platt", faixa="TODAS")
    requirements[45].append(
        f"P-Platt aceita {integer(platt['aceitos'])}, com {integer(platt['fp'])} FP, "
        f"{integer(platt['pares_humano'])} ao revisor humano, {integer(platt['pares_llm'])} ao LLM "
        f"e US\\$ {pt_number(platt['custo_llm_usd'], 2)}"
    )
    for mode, label in labels.items():
        dominance = [one_row(robustness, tipo="dominancia", modo=mode, bloco=block)
                     for block in ("N", "P")]
        counts = [f"{integer(row['k'])}/{integer(row['n'])}" for row in dominance]
        requirements[46].append(label + " & " + " & ".join(counts))
        inverse = [f"{integer(row['inversa'])}/{integer(row['n'])}" for row in dominance]
        requirements[46].append(" & ".join(inverse))
        lost_more = [one_row(robustness, tipo="PAR2", modo=mode, bloco=block, braco="P")
                     for block in ("N", "P")]
        fractions = [
            f"{round(float(row['perde_mais_aposteriori']) * float(dom['n']))}/{integer(dom['n'])}"
            for row, dom in zip(lost_more, dominance)
        ]
        short = "vig." if mode == "vigilancia" else "conf."
        requirements[46].append(
            rf"P perde mais PAR~=~2 que o \textit{{a posteriori}}, {short} & " + " & ".join(fractions)
        )
    fp = one_row(robustness, tipo="desfecho", modo="vigilancia", bloco="P",
                 braco="P", faixa="TODAS", metrica="fp_abaixo")
    lost = one_row(robustness, tipo="desfecho", modo="vigilancia", bloco="P",
                   braco="P", faixa="TODAS", metrica="perdidos")
    jaccard = one_row(robustness, tipo="estabilidade", modo="vigilancia", bloco="P",
                      braco="P", metrica="jaccard_llm")
    requirements[46].extend((
        f"FP de {integer(fp['minimo'])} a {integer(fp['maximo'])}",
        f"perdas de {integer(lost['minimo'])} a {integer(lost['maximo'])}",
        f"de {pt_number(jaccard['mediana'], 2)} em relação à partição original",
    ))
    for block in ("N", "P"):
        delta = one_row(robustness, tipo="estabilidade", modo="vigilancia", bloco=block,
                        braco="P", metrica="delta_p_max")
        block_name = "numérico (N)" if block == "N" else "de partição (R)"
        requirements[46].append(f"{pt_number(delta['maximo'], 2)} no bloco {block_name}")
    sex = one_row(csv_loader("equidade_arranjo_pairwise_tests.csv"),
                  decisao="arranjo_P_vigilancia", dimension="sexo", metric="ppv")
    requirements[23].extend((
        f"p_{{BH}}={pt_number(sex['p_value_fdr'], 3)}",
        "sem rejeição" if sex["reject_fdr"] == "False" else "com rejeição",
    ))
    for step in ("6", "9", "11"):
        cutoff_n = sum(int(row["n"]) for row in steps
                       if row["modo"] == "vigilancia" and row["passo"] == step and row["par"] == "1")
        total = sum(int(row["n"]) for row in steps
                    if row["modo"] == "vigilancia" and row["passo"] == step and row["par"] in ("1", "2"))
        requirements[24].append(pt_percent(cutoff_n / total))
        if step == "11":
            requirements[24].append(f"{cutoff_n}/{total}")
        requirements[5].append(f"{pt_number(sum(int(row['n']) for row in steps if row['modo'] == 'vigilancia' and row['passo'] == step), 0)} & {total}")
    par2_step11 = sum(int(row["n"]) for row in steps
                      if row["modo"] == "vigilancia" and row["passo"] == "11" and row["par"] == "2")
    requirements[24].append(f"Dos {par2_step11} PAR~=~2 do passo 11")
    step11_p = one_row(groups, decisao="arranjo_P_vigilancia", dimension="passo", group="passo_11")
    step11_cascade = one_row(groups, decisao="cascata_v31_vigilancia", dimension="passo", group="passo_11")
    requirements[24].append(
        f"{pt_percent(step11_p['sens'])} contra {pt_percent(step11_cascade['sens'])}"
    )
    confirmation_rows = [i for i, row in enumerate(groups, 2)
                         if row["decisao"] == "arranjo_P_confirmacao" and row["dimension"] == "passo"]
    requirements[24].append(f"{min(confirmation_rows)} a {max(confirmation_rows)}")
    comparator = csv_loader("comparador_mesma_carga.csv")
    regions = csv_loader("fp_arranjo_por_regiao.csv")
    for mode, label in (("vigilancia", "vigilância"), ("confirmacao", "confirmação")):
        high = one_row(regions, modo=mode, braco="P", limiar="global", passo="TOTAL", regiao_SCORE="maior_ou_igual")
        low = one_row(regions, modo=mode, braco="P", limiar="global", passo="TOTAL", regiao_SCORE="menor")
        requirements[45].append(f"{label}, {high['FP']} acima ou iguais e {low['FP']} abaixo")
    for mode in ("vigilancia", "confirmacao"):
        p = principal[mode]
        label = "vigilância" if mode == "vigilancia" else "confirmação"
        extra = integer(float(p['aceitos']) - float(cutoff['aceitos']))
        requirements[19].append(
            rf"\textbf{{P, {label}}} & \textbf{{{integer(cutoff['aceitos'])}+{extra} ({pt_percent(p['sens_auto'])}\%)}}"
            f" & {integer(p['fp'])} & {integer(p['perdidos'])} & {integer(p['pares_humano'])} & {integer(p['pares_llm'])}"
        )
        for arm in ("P", "O3"):
            effort = one_row(csv_loader("esforco_arranjo.csv"), modo=mode, braco=arm, carga="K1_humana")
            if arm == "P":
                requirements[28].extend((pt_number(effort['revisoes_poupadas'], 0), pt_number(effort['horas_poupadas'])))
            else:
                requirements[28].append(integer(effort['par2_nao_recuperados']))
        row = one_row(comparator, modo=mode, ordenacao="O3_p_cal_ml", carga="K1_humana")
        requirements[28].append(integer(row['k_min_equivalente_P']))
        requirements[28].append(integer(row['P_par2_recuperados']))
    vig = one_row(comparator, modo="vigilancia", ordenacao="O3_p_cal_ml", carga="K1_humana")
    requirements[22].append(f"recuperam {integer(vig['par2_topk'])} PAR~=~2 em {integer(vig['k'])} revisões")
    requirements[22].append(f"contra {integer(vig['P_par2_recuperados'])} de P com {integer(vig['P_fp'])} FP e {integer(vig['P_pares_llm'])} ao LLM")
    for mode in ("vigilancia", "confirmacao"):
        for arm in ("P", "P-Platt", "A3", "P-PAR2-grey", "B1", "aposteriori"):
            row = one_row(metrics, modo=mode, braco=arm, faixa="TODAS")
            accepted = f"({integer(cutoff['aceitos'])}+{integer(float(row['aceitos']) - float(cutoff['aceitos']))})" if arm == "P" else integer(row['aceitos'])
            requirements[45].append("/".join((accepted, integer(row['fp']), integer(row['pares_humano']), integer(row['pares_llm']))))
    return requirements


@pytest.mark.parametrize("frame_number", [3, 4, 5, 6, 7, 13, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 43, 44, 45, 46])
def test_slide_scientific_contract_from_csv(frame_number):
    assert_fragments(slide_frames()[frame_number - 1], scientific_requirements()[frame_number])


def test_slide_contract_rejects_changed_csv_dominance():
    def changed_csv(name):
        rows = read_csv(name)
        if name == "robustez_arranjo_resumo.csv":
            row = one_row(rows, tipo="dominancia", modo="vigilancia", bloco="N")
            row["k"] = str(float(row["k"]) + 1)
        return rows

    with pytest.raises(AssertionError):
        assert_fragments(slide_frames()[45], scientific_requirements(changed_csv)[46])


def test_slide_final_message_rejects_missing_false_positives():
    frame = slide_frames()[29]
    frame = re.sub(r"\d+ e \d+ falsos positivos", "", frame)
    with pytest.raises(AssertionError, match="falsos positivos"):
        assert_fragments(frame, scientific_requirements()[30])


def test_slide_and_script_s7_terminology_and_attribution():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    for text, singular, plural in ((source, r"\textit{label}", r"\textit{labels}"),
                                    (script, "*label*", "*labels*")):
        assert not re.search(r"rótulos?|calibração sigmoidal", text, re.I)
        assert singular in text and plural in text
    speech = script.split("## Fala-guia por slide")[1].split("## Perguntas prováveis")[0]
    assert "porque foi registrado antes" in speech.split("### 22.")[1].split("### 23.")[0]
    assert "na base S6a" not in speech
    assert "Preservar o corte pode reduzir a revisão" not in slide_frames()[29]
    assert "Preservar o corte pode reduzir a revisão" not in speech.split("### 30.")[1].split("### 31.")[0]
    questions = script.split("## Perguntas prováveis")[1].split("## Figuras e critérios")[0]
    answer = questions.split("### 14.")[1].split("### 15.")[0]
    mechanisms = read_csv("arranjo_corte_perdas_mecanismo.csv")
    for mode, label in (("vigilancia", "vigilância"), ("confirmacao", "confirmação")):
        counts = [integer(one_row(mechanisms, modo=mode, braco="P", mecanismo=mechanism)["vinculos_perdidos"])
                  for mechanism in ("guarda_nota_lt_5", "AUTO_NONMATCH_custo", "NONMATCH_LLM")]
        assert f"{label}: " + "/".join(counts) in answer
    assert "guarda/custo/LLM" in answer


def test_effort_figure_is_regenerated_from_csv(tmp_path):
    plot = runpy.run_path(str(ROOT / "scripts/figura_curva_esforco.py"))
    output = plot["generate"](
        ROOT / "data/comparador_mesma_carga_curva.csv",
        ROOT / "data/comparador_mesma_carga.csv", tmp_path / "curve.pgf",
    )
    assert output.read_bytes() == (SLIDES / "figures/curva_esforco.pgf").read_bytes()
    content = output.read_text(encoding="utf-8")
    for mode, label in (("vigilancia", "P vig."), ("confirmacao", "P conf.")):
        row = one_row(read_csv("comparador_mesma_carga.csv"), modo=mode,
                      ordenacao="O3_p_cal_ml", carga="K1_humana")
        assert f"{label}: ({integer(row['P_pares_humano'])}, {integer(row['P_par2_recuperados'])}), {integer(row['P_fp'])} FP" in content
    curves = read_csv("comparador_mesma_carga_curva.csv")
    for order, _, _ in plot["ORDERS"]:
        expected = sorted((int(r["k"]), int(r["par2_topk"])) for r in curves if r["ordenacao"] == order)
        assert plot["curve_series"](curves, order) == expected


def test_slide_backup_cascade_rows_from_csv():
    metrics = read_csv("arranjo_corte_metricas.csv")
    frames = slide_frames()
    for mode, reserve in (("vigilancia", 43), ("confirmacao", 44)):
        row = one_row(metrics, modo=mode, braco="publicada", faixa="TODAS")
        for field, heading in (("perdidos", "Vínculos perdidos por não-vínculo"),
                               ("pares_humano", "Potenciais pares ao humano"),
                               ("pares_llm", "Potenciais pares ao LLM"),
                               ("fp", "Falsos positivos")):
            assert re.search(
                rf"{heading} & \d+(?:\.\d+)* & {re.escape(pt_number(row[field], 0))}",
                frames[reserve - 1],
            ), (reserve, field)


def test_effort_figure_rejects_invalid_curve_and_tracks_csv_changes():
    plot = runpy.run_path(str(ROOT / "scripts/figura_curva_esforco.py"))
    curves = read_csv("comparador_mesma_carga_curva.csv")
    comparator = read_csv("comparador_mesma_carga.csv")
    original = plot["render"](curves, comparator)
    row = one_row(comparator, modo="vigilancia", ordenacao="O3_p_cal_ml", carga="K1_humana")
    row["P_par2_recuperados"] = str(int(row["P_par2_recuperados"]) + 1)
    assert original != plot["render"](curves, comparator)
    with pytest.raises(ValueError, match="ausente"):
        plot["curve_series"]([], "O3_p_cal_ml")
    with pytest.raises(ValueError, match="duplicado"):
        plot["curve_series"]([curves[0], curves[0]], "O1_SCORE")
    with pytest.raises(ValueError, match="não monotônica"):
        plot["curve_series"]([
            {"ordenacao": "O1_SCORE", "k": "1", "par2_topk": "2"},
            {"ordenacao": "O1_SCORE", "k": "2", "par2_topk": "1"},
        ], "O1_SCORE")


def test_script_s8_comparator_effort_and_false_positive_regions_from_csv():
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    for row in read_csv("comparador_mesma_carga.csv"):
        if row["carga"] == "K1_humana":
            expected = " | ".join((row["modo"], row["ordenacao"], integer(row["k"]),
                                   integer(row["par2_topk"]), integer(row["k_min_equivalente_P"]),
                                   integer(row["P_par2_recuperados"]), integer(row["P_fp"])))
            assert expected in script
    for row in read_csv("esforco_arranjo.csv"):
        expected = " | ".join((row["modo"], row["braco"], row["carga"], integer(row["revisoes"]),
                               pt_number(row["horas"]), pt_number(row["horas_poupadas"]),
                               pt_number(row["horas_poupadas_por_vinculo_perdido"]),
                               pt_number(row["horas_poupadas_por_fp"]) if row["horas_poupadas_por_fp"] else "não se aplica"))
        assert expected in script
        assert row["cenario"] == "True" and row["horas_medidas"] == "False"
    for row in read_csv("fp_arranjo_por_regiao.csv"):
        if row["passo"] == "TOTAL":
            assert " | ".join((row["modo"], row["limiar"], row["regiao_SCORE"], row["FP"])) in script
    assert "ordem histórica não foi documentada" in script
    assert "denominadores mínimos" in script
    assert "exploratória *post hoc*, não registrada" in script
    assert "Slides 6, 9 e 10" in script
    assert "CIDACS" in script.split("### 3.")[-1].split("### 4.")[0]


def test_slide_s8_word_list_structure_and_robustness_block_names():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    for text in (source, script):
        for forbidden in (r"rótulos?", "calibração sigmoidal", "revocação", r"\bexplicam\b",
                          "denominadores pequenos", "pior em todo desfecho", "os intervalos se sobrepõem"):
            assert not re.search(forbidden, text, re.I), forbidden
        assert not re.search(r"(?<![0-9])(?:96/17|13/27|4/8)(?![0-9])", text)
    frames = slide_frames()
    assert not re.search(r"\d", frames[2].split("\\fonte")[0])
    assert frames[21].count(r"\item ") == 2
    assert "IC 95" not in frames[18] and "Fechamentos" not in frames[18]
    assert "01:22:17" not in frames[18]
    assert "02:11:52 UTC (metadados na versão" in frames[44]
    assert "Numérico (N) & Partição (R)" in frames[45]
    assert "bloco de partição (R), Jaccard" in frames[45]


def test_script_s6_questions_and_provenance():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    llm = one_row(read_csv("arranjo_corte_vpp_llm.csv"), modo="vigilancia", braco="P")
    lost = one_row(read_csv("arranjo_corte_par2.csv"), modo="confirmacao", braco="P")
    for topic in [
        "carga comparável",
        f"VPP de {pt_percent(llm['VPP'], 0)}%",
        "vínculos perdidos",
        f"perder {pt_percent(lost['perdidos_proporcao'], 0)}%",
        "taxas congeladas",
        "viés de incorporação",
        "somente com o Zenodo",
        "desvios de protocolo",
        "robustez confirma",
        "varredura de custos",
    ]:
        assert topic in questions
    assert "recalibração, braços e avaliação" in questions
    metrics = read_csv("arranjo_corte_metricas.csv")
    cutoff = one_row(metrics, modo="vigilancia", braco="corte_sozinho", faixa="TODAS")
    principal = one_row(metrics, modo="vigilancia", braco="P", faixa="TODAS")
    assert f"arranjo_corte_metricas.csv:{metrics.index(cutoff) + 2},{metrics.index(principal) + 2}" in questions
    assert "`capitulo3.tex`, `sec:objetivos-especificos`" in questions
    assert "objetivo 4 autorizado" not in questions
    assert "em relação à partição original" in questions
    assert "Jaccard mediano do lote LLM entre partições" not in source
    robustness = read_csv("robustez_arranjo_resumo.csv")
    answer11 = questions.split("### 11.")[1].split("### 12.")[0]
    for mode in ("vigilancia", "confirmacao"):
        rows = [one_row(robustness, tipo="dominancia", modo=mode, bloco=block)
                for block in ("N", "P")]
        assert " e ".join(f"{integer(row['k'])}/{integer(row['n'])}" for row in rows) in answer11
    assert "não houve varredura no arranjo" in questions.lower()
    assert "árvore compilada" in source
    assert re.search(r"SHA-256[^\n]*`[0-9a-f]{64}`", source)
