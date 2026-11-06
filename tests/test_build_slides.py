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
    assert len(backup_frames) == 19
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


def test_script_has_fifty_two_short_answers():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    answers = re.findall(r"### \d+\..*?\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 52
    assert [int(n) for n in re.findall(r"^### (\d+)\.", questions, re.M)] == list(
        range(1, 53)
    )
    assert all(len(answer.split()) <= 70 for answer in answers)
    recorded = re.search(r"São (\d+) respostas, com no máximo 70 palavras cada", source)[1]
    assert int(recorded) == len(answers)


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
        4: ["hipótese gerada nesta avaliação", "Hipótese exploratória, a validar", "fora da amostra",
            "duas fronteiras", "mesmo XGBoost", "LLM:", "CdR:", "OOF:", "FP:", "VPP:", "P:", "O3:", "O4:"],
        5: [],
        6: ["Cenário exploratório", "se a revisão tivesse seguido", "Não é uma reconstrução histórica",
            pt_number(lower["pares_acima_limiar"], 0), pt_number(global_lower["pares_acima_limiar"], 0)],
        7: ["Revisora única, sem classificação independente", "CdR no passo aberto",
             "não nos passos 6, 9 e 11", "viés de verificação e incorporação",
             "A nota do CdR não é o escore probabilístico do OpenRecLink"],
        13: ["OOF", "REFREC", "A política v3.1 compara custos, não usa esse corte único"],
        19: [r"S3 (avaliação com os \textit{labels})"],
        18: [r"p\geq C_H/C_{FN}", "Os pares restantes recebem não vínculo", "restrito às ações revisor humano e não vínculo"],
        20: ["figures/curva_esforco.pgf", "não registrada", "Revisor perfeito é hipótese dos dois lados",
             "Se a revisão histórica seguiu o escore", "favoreceria O1", "mesmos", "verificados"],
        21: ["impede interpretação prospectiva", "Hipótese de uso, não política validada",
              "O LLM nunca aceita sozinho", "Filtro antes do revisor humano",
               "validação registrada própria", "VPN (valor preditivo negativo)"],
         22: ["o registro fixou a isotônica como calibração padrão", "análise de sensibilidade", "não confirmatória",
              "sem dominância sustentada", "Hipótese de revisor perfeito dos dois lados",
              "Não é empate sem qualificação", "Não se substitui o revisor humano pelo LLM",
              "LLM como aceitador não se paga abaixo do corte",
              "Uma diferença de um vínculo em 36 não sustenta superioridade"],
        23: ["proporção de vínculos", "definida pelo corte", "FP abaixo do corte"],
        24: [r"Disparidade herdada do corte e do \textit{label}", "Benjamini-Hochberg: vigilância"],
        26: ["Respostas históricas reutilizadas", "conhecido antes do registro", "favoreceria O1",
              "recalibração, braços e avaliação", "desvios de protocolo", "não de generalização"],
        28: ["perdas são de cobertura",
             "O serviço decide se aceita a troca", "95", "99", "100", "especificidade humana é assumida perfeita"],
        29: ["Preservar o corte", "ordem decrescente", "uma probabilidade ou escore",
              "Se usar probabilidade aprendida", "1:1 e 1:5 pioraram o topo (13 e 18,5 contra 22)",
                "1:20 e 1:50 são indistinguíveis da referência", "Brier mediano da isotônica piorou em todas as razões",
              "uma execução contra dez sementes", "Revisor humano no topo até o orçamento", "restante como não vínculo",
             "LLM apenas como filtro de NONMATCH", "nunca aceitando sozinho", "registro interno",
             "validação registrada em dados novos", "Decomposição por PAR", "curva de esforço",
             "Falha de transferência", "ancoragem", "quatro ações", "benchmark"],
        30: ["cobertura do topo da fila", "não reproduz", "no máximo 1 vínculo",
             "recomendações a validar", "filtro experimental com H=22, nunca aceitando sozinho"],
        43: [],
        44: [],
        45: [],
        46: ["não confirmatória", "não generalização", "em relação à partição original",
              "A réplica original é um ponto dessa distribuição"],
        47: ["Exploratória", "prevalência original", "Brier mediano", "sem demonstrar melhora de equidade",
             "1:1 e 1:5 pioram o topo (13 e 18,5 contra 22)"],
        48: ["MATCH e inconclusivos vão ao revisor humano", "Nunca aceita sozinho",
             "protocolo fixado antes dos desfechos", "respostas de datas diferentes", "Contagens iguais não significam todas as decisões iguais"],
        49: ["especificidade humana perfeita", "sem modelar FP humanos", "metadados na versão",
             "Horas são ordem de grandeza, não medida",
             r"\multicolumn{2}{c}{Aceitos}", r"\multicolumn{2}{c}{Ao humano}",
             r"\multicolumn{2}{c}{Perdidos}", "Modo & P & Cascata & P & Cascata & P & Cascata"],
    }
    reference = csv_loader("referencia_comparador_pos_orl.csv")
    two_frontiers = one_row(reference, estrategia="pos_orl_revisao_humana_area_cinza")
    requirements[6].extend((
        f"{pt_number(two_frontiers['pares_ao_humano'], 0)} ao revisor humano",
        f"{pt_percent(two_frontiers['sens_com_humano'])}\\% no cenário humano de "
        f"{pt_percent(1-float(two_frontiers['erro_humano_hipotese']), 0)}\\%",
        r"cenários de sensibilidade humana de 95\%, 99\% e 100\% (quadro 28)",
    ))
    requirements[6].append(f"{integer(cutoff['aceitos'])}/{integer(cutoff['vinculos'])} = {pt_percent(cutoff['sens_auto'])}")
    hours = one_row(csv_loader("carga_abaixo_corte_horas.csv"), limiar="por_passo")
    requirements[6].append(f"{integer(hours['horas_inteiras'])} h {integer(hours['minutos_restantes'])} min")
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
        requirements[49].extend((" & ".join(table), pt_percent(observed["perdidos_proporcao"])))
        requirements[49].append("/".join(
            integer(one_row(mechanisms, modo=mode, braco="P", mecanismo=mechanism)["vinculos_perdidos"])
            for mechanism in mechanisms_order
        ))
        reserve = 43 if mode == "vigilancia" else 44
        requirements[reserve].append(
            f"({integer(cutoff['aceitos'])}+{integer(observed['aceitos'])})+{integer(p['perdidos'])}+"
            f"{integer(p['vinc_humano'])}={integer(cutoff['vinculos'])}"
        )
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
    requirements[49].extend((
        f"{integer(cost_losses['vinculos_perdidos'])} perdas por AUTO\\_NONMATCH",
        "Sem varredura de custos no arranjo",
    ))
    requirements[49].append(
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
    requirements[45].append("Na vigilância, P-Platt dominou P na avaliação original, como análise de sensibilidade")
    requirements[45].append("limiar global do escore")
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
    sex = one_row(csv_loader("equidade_arranjo_den30_pairwise_tests.csv"),
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
        for threshold in ("global", "por_passo"):
            high = one_row(regions, modo=mode, braco="P", limiar=threshold, passo="TOTAL", regiao_SCORE="maior_ou_igual")
            low = one_row(regions, modo=mode, braco="P", limiar=threshold, passo="TOTAL", regiao_SCORE="menor")
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
            if arm == "P" and mode == "vigilancia":
                o1 = one_row(comparator, modo=mode, ordenacao="O1_SCORE", carga="K1_humana")
                requirements[49].append(f"O1 requer {integer(o1['k_min_equivalente_P'])} revisões")
        p = principal[mode]
        mode_text = "vigilância" if mode == "vigilancia" else "confirmação"
        requirements[28].append(f"{integer(p['perdidos'])} {'vínculos na ' if mode == 'vigilancia' else 'na '}{mode_text}")
    requirements[28].append(f"{integer(principal['vigilancia']['fp'])} e {integer(principal['confirmacao']['fp'])} FP")
    par2_counts = [str(sum(int(r['n']) for r in steps if r['modo']=='vigilancia' and r['passo']==step and r['par']=='2')) for step in ('6','9','11')]
    requirements[7].append(", ".join(par2_counts[:-1]) + " e " + par2_counts[-1])
    for mode in ("vigilancia", "confirmacao"):
        for arm in ("P", "P-Platt", "A3", "P-PAR2-grey", "B1", "aposteriori"):
            row = one_row(metrics, modo=mode, braco=arm, faixa="TODAS")
            accepted = f"({integer(cutoff['aceitos'])}+{integer(float(row['aceitos']) - float(cutoff['aceitos']))})" if arm == "P" else integer(row['aceitos'])
            requirements[45].append("/".join((accepted, integer(row['fp']), integer(row['pares_humano']), integer(row['pares_llm']))))
    for number, fragments in final_scientific_requirements(csv_loader).items():
        requirements[number].extend(fragments)
    requirements[28].extend(budget_requirements(csv_loader))
    return requirements


def budget_requirements(csv_loader=read_csv):
    comparator = csv_loader("comparador_mesma_carga.csv")
    effort = csv_loader("esforco_arranjo.csv")
    required = []
    for mode, label in (("vigilancia", "Vigilância"), ("confirmacao", "Confirmação")):
        o3 = one_row(comparator, modo=mode, ordenacao="O3_p_cal_ml", carga="K1_humana")
        o4 = one_row(comparator, modo=mode, ordenacao="O4_P_recalibrada", carga="K1_humana")
        loss = one_row(effort, modo=mode, braco="O3", carga="K1_humana")
        required.append(
            f"{label}, k={o3['k']} & {o3['P_par2_recuperados']} & {o3['par2_topk']} / "
            f"{loss['par2_nao_recuperados']} & {o3['k_min_equivalente_P']} / {o4['k_min_equivalente_P']}"
        )
    o1 = one_row(comparator, modo="vigilancia", ordenacao="O1_SCORE", carga="K1_humana")
    o3 = one_row(comparator, modo="vigilancia", ordenacao="O3_p_cal_ml", carga="K1_humana")
    required.append(f"Para os 36 PAR~=~2, O1 requer {pt_number(o1['k_min_36'], 0)} e O3, {pt_number(o3['k_min_36'], 0)}")
    return required


def final_scientific_requirements(csv_loader=read_csv):
    required = {number: [] for number in (19, 21, 22, 28, 47, 48, 49)}
    match = csv_loader("comparador_p_match_humano.csv")
    metrics = csv_loader("arranjo_corte_metricas.csv")
    llm = csv_loader("arranjo_corte_vpp_llm.csv")
    for mode, label in (("vigilancia", "Vigilância"), ("confirmacao", "Confirmação")):
        row = one_row(match, modo=mode)
        cells = [label,
            f"{row['P_revisoes_humanas_total']} / {row['P_par2_recuperados']}",
            f"{row['O3_k_mesma_carga']} / {row['O3_par2_mesma_carga']}",
            f"{row['O3_k_min_igualar_P']} / {row['O3_par2_no_k_min']}"]
        required[22].append(" & ".join(cells))
        p = one_row(metrics, modo=mode, braco="P", faixa="TODAS")
        required[22].append(f"{p['pares_llm']} {'pares avaliados pelo LLM na vigilância' if mode == 'vigilancia' else 'na confirmação'}")
        accepted = one_row(llm, modo=mode, braco="P")
        required[19].append(f"{accepted['aceitacoes_llm_abaixo']} {'pares na vigilância' if mode == 'vigilancia' else 'na confirmação'}")
    vp = [one_row(llm, modo=mode, braco="P") for mode in ("vigilancia", "confirmacao")]
    required[19].extend((f"{vp[0]['VP']} e {vp[1]['VP']} são PAR~=~2",
                         f"os {vp[0]['FP']} e {vp[1]['FP']} FP foram rejeitados ou não examinados"))
    scenarios = csv_loader("cenarios_sensibilidade_humana.csv")
    for strategy, words in (("humano_integral", "na revisão integral"), (None, "nas estratégias de carga reduzida")):
        amplitude = max(float(r["amplitude_pp"]) for r in scenarios
                        if (r["estrategia"] == strategy if strategy else r["estrategia"] != "humano_integral"))
        required[28].append(f"{pt_number(amplitude, 2)} p.p. {words}")
        required[49].append(f"{pt_number(amplitude, 2)} p.p.")
    required[22].append(f"no máximo {max(int(r['diferenca_P_O3_PAR2_mesma_carga']) for r in match)} vínculo a mais")
    subset = csv_loader("subamostragem_treino_resumo.csv")
    for ratio in ("original", "1:1", "1:5", "1:20", "1:50"):
        row = one_row(subset, ratio=ratio, variant="isotonic", population="below", metric="par2_top_96")
        def count(value):
            return integer(value) if float(value).is_integer() else pt_number(value)
        if ratio == "original":
            assert row["seeds"] == "1"
            assert row["median"] == row["minimum"] == row["maximum"]
            required[47].append(f"Original & {count(row['median'])} & Referência: uma execução")
        else:
            required[47].append(f"{ratio} & {count(row['median'])} & {count(row['minimum'])} a {count(row['maximum'])}")
    for metric, group in (("sens96_passo_6", "passo 6"), ("sens96_sexo_M", "sexo masculino")):
        row = one_row(subset, ratio="1:1", variant="isotonic", population="below", metric=metric)
        required[47].append(f"{pt_number(-100*float(row['delta_median']), 0)} p.p. no {group}")
    denominator = one_row(subset, ratio="1:1", variant="isotonic", population="below", metric="sens96_passo_6_den")
    required[47].append(f"passo 6 ({integer(denominator['median'])} PAR~=~2)")
    filters = csv_loader("filtro_llm.csv")
    selected = []
    for mode, label, k in (("vigilancia", "Vigilância", "96"), ("confirmacao", "Confirmação", "125")):
        row = one_row(filters, modo=mode, k=k)
        selected.append(row)
        required[48].append(
            f"{label}, {k} & {row['H']} & {row['par2_recuperados']} & {row['o3_par2_mesma_carga']} & "
            f"{pt_percent(row['vpn_nonmatch'])}\\% & {row['par2_descartados']} de {row['par2_topk']}"
        )
    required[21].extend((
        f"{selected[0]['nonmatch']} pares avaliados pelo LLM como NONMATCH e {selected[0]['H']} revisões humanas em k={selected[0]['k']}",
        f"recupera {selected[0]['par2_recuperados']} PAR~=~2 contra {selected[0]['o3_par2_mesma_carga']}",
    ))
    for k in ("96", "58"):
        row = one_row(filters, modo="vigilancia", k=k)
        for number in (21, 29, 48):
            required.setdefault(number, []).append(f"{pt_percent(row['vpn_nonmatch'])}\\% em k={k}")
    required[48].append(f"{selected[0]['nonmatch']} pares avaliados pelo LLM como NONMATCH e {selected[0]['H']} revisões humanas")
    concordance = csv_loader("filtro_llm_concordancia_modos.csv")
    p = one_row(concordance, k="125", classe="PAR2")
    n = one_row(concordance, k="125", classe="nao_vinculo")
    required[48].extend((f"{p['coincidentes']} coincidentes, {p['diferentes']} diferentes",
                         f"há {n['diferentes']} diferenças"))
    return required


@pytest.mark.parametrize("frame_number", [3, 4, 5, 6, 7, 13, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 43, 44, 45, 46, 47, 48, 49])
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
    assert "o registro fixou a isotônica como calibração padrão" in speech.split("### 22.")[1].split("### 23.")[0]
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
        assert f"+{integer(row['P_pares_llm'])} LLM" in content
    curves = read_csv("comparador_mesma_carga_curva.csv")
    for order, _, _ in plot["ORDERS"]:
        expected = sorted((int(r["k"]), int(r["par2_topk"])) for r in curves if r["ordenacao"] == order)
        assert plot["curve_series"](curves, order) == expected
    assert len(set(plot["LINE_STYLES"].values())) == len(plot["ORDERS"])
    assert_figure_series_styles(content, plot)
    assert "Eixo x limitado a 2.522 revisões" in content


def assert_figure_series_styles(content, plot):
    for i, (order, label, color) in enumerate(plot["ORDERS"]):
        style = plot["LINE_STYLES"][order]
        # Vincular cor, estilo e caminho da série: os pontos de P também usam sólido.
        curve = "\n".join((rf"\color{{{color}}}", style, r"\pgfsetlinewidth{0.9pt}", r"\pgfpathmoveto"))
        assert content.count(curve) == 1, order
        x, y = (i % 2)*5.9, -1.0-(i // 2)*0.38
        legend = "\n".join((rf"\color{{{color}}}", style,
            rf"\pgfpathmoveto{{\pgfpoint{{{x:.2f}cm}}{{{y:.2f}cm}}}}",
            rf"\pgfpathlineto{{\pgfpoint{{{x+0.45:.2f}cm}}{{{y:.2f}cm}}}}\pgfusepath{{stroke}}"))
        assert content.count(legend) == 1, order
        assert label in content


@pytest.mark.parametrize("order_index", range(4))
def test_effort_styles_reject_missing_series_style_even_with_p_markers(order_index):
    plot = runpy.run_path(str(ROOT / "scripts/figura_curva_esforco.py"))
    content = (SLIDES / "figures/curva_esforco.pgf").read_text(encoding="utf-8")
    order, _, color = plot["ORDERS"][order_index]
    original = rf"\color{{{color}}}" + "\n" + plot["LINE_STYLES"][order]
    content = content.replace(original, rf"\color{{{color}}}", 1)
    assert r"\color{red}" + "\n" + r"\pgfsetdash{}{0pt}" in content
    with pytest.raises(AssertionError, match=order):
        assert_figure_series_styles(content, plot)


def test_effort_figure_cli_has_two_default_destinations(monkeypatch, tmp_path):
    plot = runpy.run_path(str(ROOT / "scripts/figura_curva_esforco.py"))
    plot["main"].__globals__["ROOT"] = tmp_path
    monkeypatch.setattr("sys.argv", ["figura_curva_esforco.py", "--curve",
        str(ROOT / "data/comparador_mesma_carga_curva.csv"), "--comparator",
        str(ROOT / "data/comparador_mesma_carga.csv")])
    plot["main"]()
    thesis = tmp_path / "text/latex/figures/curva_esforco.pgf"
    slides = tmp_path / "text/slides/figures/curva_esforco.pgf"
    assert thesis.read_bytes() == slides.read_bytes()
    assert slides.read_bytes() == (SLIDES / "figures/curva_esforco.pgf").read_bytes()


def test_slide_backup_cascade_rows_from_csv():
    metrics = read_csv("arranjo_corte_metricas.csv")
    frames = slide_frames()
    for mode, reserve in (("vigilancia", 43), ("confirmacao", 44)):
        row = one_row(metrics, modo=mode, braco="publicada", faixa="TODAS")
        for field, heading in (("perdidos", "Vínculos perdidos por não vínculo"),
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
    with pytest.raises(ValueError, match="não contínuo"):
        plot["curve_series"]([
            {"ordenacao": "O1_SCORE", "k": "1", "par2_topk": "1"},
            {"ordenacao": "O1_SCORE", "k": "3", "par2_topk": "2"},
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
                               pt_number(row["horas"], 0), pt_number(row["horas_poupadas"], 0),
                               pt_number(row["horas_poupadas_por_vinculo_perdido"], 2),
                               pt_number(row["horas_poupadas_por_fp"], 2) if row["horas_poupadas_por_fp"] else "não se aplica"))
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
        assert "registro prévio" not in text.lower()
        assert "bloco P" not in text and "bloco de partição (R), bloco de partição (R)" not in text
        assert not re.search(r"(?<![0-9])(?:96/17|13/27|4/8|121/125|29/29|7/3|9/1)(?![0-9])", text)
    frames = slide_frames()
    assert not re.search(r"\d", frames[2].split("\\fonte")[0])
    assert frames[21].count(r"\item ") == 2
    assert "IC 95" not in frames[18] and "Fechamentos" not in frames[18]
    assert "01:22:17" not in frames[18]
    assert "02:11:52 UTC (metadados na versão" in frames[48]
    assert "Numérico (N) & Partição (R)" in frames[45]
    assert "bloco de partição (R), Jaccard" in frames[45]
    assert frames[45].count("bloco de partição (R)") == 3
    assert not re.search(r"IC 95\\% (?:82,3 a 90,7|84,1 a 92,0)", source)


def test_script_s81_new_artifacts_provenance_and_hours():
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    for row in read_csv("comparador_p_match_humano.csv"):
        keys = ("modo", "P_revisoes_humanas_originais", "P_MATCH_llm_ao_humano", "P_revisoes_humanas_total",
                "P_par2_recuperados", "O3_k_mesma_carga", "O3_par2_mesma_carga", "O3_k_min_igualar_P",
                "diferenca_P_O3_PAR2_mesma_carga")
        assert " | ".join(row[key] for key in keys) in script
        assert row["revisor_perfeito_ambos"] == "True"
    for row in read_csv("filtro_llm.csv"):
        assert " | ".join([row[k] for k in ("modo", "k", "H", "par2_recuperados", "o3_par2_mesma_carga")]
                          + [pt_percent(row["vpn_nonmatch"]), row["par2_descartados"]]) in script
    for row in read_csv("cenarios_sensibilidade_humana.csv"):
        assert " | ".join([row[k] for k in ("modo", "estrategia", "k", "revisoes_humanas", "vinculos_perdidos_cobertura")]
                          + [pt_percent(row["s"], 0), pt_percent(row["sensibilidade_total"], 2),
                             pt_number(row["amplitude_pp"], 2)]) in script
    for row in read_csv("subamostragem_treino_resumo.csv"):
        if row["variant"] == "isotonic" and row["population"] == "below" and row["metric"] in (
                "par2_top_96", "brier", "sens96_passo_6", "sens96_sexo_M"):
            digits = 6 if row["metric"] == "brier" else (2 if row["metric"].startswith("sens") else 1)
            assert " | ".join([row["ratio"], row["metric"]] + [pt_number(row[k], digits)
                              for k in ("median", "minimum", "maximum", "reference")]) in script
    questions = script.split("## Perguntas prováveis")[1].split("## Figuras e critérios")[0]
    answer12 = questions.split("### 12.")[1].split("### 13.")[0]
    for row in read_csv("carga_abaixo_corte_horas.csv"):
        assert f"{row['horas_inteiras']} h {row['minutos_restantes']} min" in answer12
    for number, topic in ((30, "filtro"), (31, "prevalência original"), (32, "99%"), (33, "revisor humano pelo LLM")):
        assert topic in questions.split(f"### {number}.")[1].split("### ")[0]
    assert "não superou" not in slide_frames()[3]
    assert "hipótese gerada" in script


def assert_budget_script(script, csv_loader=read_csv):
    comparator = csv_loader("comparador_mesma_carga.csv")
    effort = csv_loader("esforco_arranjo.csv")
    for row in comparator:
        if row["carga"] != "K1_humana":
            continue
        # O CSV comparador materializa recuperação e k; perda = total PAR=2 - recuperação.
        metrics = csv_loader("arranjo_corte_metricas.csv")
        total = one_row(metrics, modo=row["modo"], braco="corte_sozinho", faixa="TODAS")
        count = int(total["vinculos"]) - int(total["aceitos"])
        expected = " | ".join((row["modo"], row["ordenacao"], row["P_par2_recuperados"],
            str(count - int(row["par2_topk"])), row["k_min_equivalente_P"], row["k_min_36"]))
        assert expected in script
        if row["ordenacao"] == "O3_p_cal_ml":
            loss = one_row(effort, modo=row["modo"], braco="O3", carga="K1_humana")
            assert int(loss["par2_nao_recuperados"]) == count - int(row["par2_topk"])
    speech = script.split("## Fala-guia por slide")[1].split("## Perguntas prováveis")[0]
    answer20 = script.split("## Perguntas prováveis")[1].split("### 20.")[1].split("### 21.")[0]
    sections = {n: speech.split(f"### {n}.")[1].split("### ")[0] for n in (20, 28)}
    v3, c3 = [one_row(comparator, modo=mode, ordenacao="O3_p_cal_ml", carga="K1_humana")
              for mode in ("vigilancia", "confirmacao")]
    v4, c4 = [one_row(comparator, modo=mode, ordenacao="O4_P_recalibrada", carga="K1_humana")
              for mode in ("vigilancia", "confirmacao")]
    vl, cl = [one_row(effort, modo=mode, braco="O3", carga="K1_humana")
              for mode in ("vigilancia", "confirmacao")]
    assert (f"O4 iguala P com {v4['k_min_equivalente_P']} revisões na vigilância e {c4['k_min_equivalente_P']} "
            f"na confirmação, contra {v3['k_min_equivalente_P']} e {c3['k_min_equivalente_P']} de O3") in sections[20]
    assert (f"P recupera {v3['P_par2_recuperados']} e {c3['P_par2_recuperados']}; O3 recupera "
            f"{v3['par2_topk']} e {c3['par2_topk']}, deixando {vl['par2_nao_recuperados']} e "
            f"{cl['par2_nao_recuperados']} dos 36 fora") in sections[20]
    assert f"P recupera {v3['P_par2_recuperados']} e {c3['P_par2_recuperados']} dos 36" in sections[28]
    assert (f"Em {v3['k']} e {c3['k']} revisões, O3 recupera {v3['par2_topk']} e {c3['par2_topk']}, "
            f"deixando {vl['par2_nao_recuperados']} e {cl['par2_nao_recuperados']} fora") in sections[28]
    assert (f"Para igualar P, O3 requer {v3['k_min_equivalente_P']} e {c3['k_min_equivalente_P']}; "
            f"O4, {v4['k_min_equivalente_P']} e {c4['k_min_equivalente_P']}") in sections[28]
    assert (f"Para igualar os {v3['P_par2_recuperados']} e {c3['P_par2_recuperados']} vínculos de P, "
            f"O4 requer {v4['k_min_equivalente_P']} e {c4['k_min_equivalente_P']} revisões; "
            f"O3, {v3['k_min_equivalente_P']} e {c3['k_min_equivalente_P']}") in answer20
    assert (f"Em {v3['k']} e {c3['k']} revisões, O3 deixa {vl['par2_nao_recuperados']} e "
            f"{cl['par2_nao_recuperados']} dos 36 fora") in answer20
    o1 = one_row(comparator, modo="vigilancia", ordenacao="O1_SCORE", carga="K1_humana")
    for text in (*sections.values(), answer20):
        assert f"O1 requer {pt_number(o1['k_min_36'], 0)} e O3, {pt_number(v3['k_min_36'], 0)}" in text


def test_s9_budget_script_preserves_all_materialized_fields():
    assert_budget_script((SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize("artifact,field", [
    ("comparador_mesma_carga.csv", "k_min_equivalente_P"),
    ("comparador_mesma_carga.csv", "P_par2_recuperados"),
    ("comparador_mesma_carga.csv", "k_min_36"),
    ("esforco_arranjo.csv", "par2_nao_recuperados"),
])
def test_s9_budget_contract_rejects_csv_mutations_in_slide_and_script(artifact, field):
    def changed(name):
        rows = read_csv(name)
        if name == artifact:
            criteria = {"modo": "vigilancia", "carga": "K1_humana"}
            criteria["ordenacao" if name.startswith("comparador") else "braco"] = (
                "O3_p_cal_ml" if name.startswith("comparador") else "O3")
            row = one_row(rows, **criteria)
            row[field] = str(int(row[field]) + 1)
        return rows
    with pytest.raises(AssertionError):
        assert_fragments(slide_frames()[27], budget_requirements(changed))
    with pytest.raises(AssertionError):
        assert_budget_script((SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8"), changed)


def test_s9_recommendation_filter_and_paired_difference_are_qualified():
    frames = slide_frames()
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    speech = script.split("## Fala-guia por slide")[1].split("## Perguntas prováveis")[0]
    for number in (4, 28, 29, 30):
        for text in (frames[number - 1], speech.split(f"### {number}.")[1].split("### ")[0]):
            assert "Preservar o corte" in text or "preservar o corte" in text
            assert "probabilidade ou escore" in text
            assert "curva de esforço" in text and "orçamento humano" in text
            assert "na vigilância, nenhuma ordenação domina" in text.lower()
            assert "na confirmação, o2 domina fracamente p" in text.lower()
            assert "fila por p_cal_ml" not in text and r"\texttt{p\_cal\_ml}" not in text
    for text in (script, (SLIDES / "defesa.tex").read_text(encoding="utf-8")):
        assert "não-vínculo" not in text
        assert "N/P" not in text and "no P |" not in text
        assert "cerca de 90" not in text and "cerca de 88 a 90" not in text
        assert "cabe na variação" not in text and "dentro da variação de perdas" not in text
        assert "Uma diferença de um vínculo em 36 não sustenta superioridade" in text
    filters = read_csv("filtro_llm.csv")
    for number in (21, 48):
        section = speech.split(f"### {number}.")[1].split("### ")[0]
        row = one_row(filters, modo="vigilancia", k="96")
        assert f"{row['nonmatch']} pares avaliados pelo LLM como NONMATCH e {row['H']} revisões humanas" in section
        for k in ("96", "58"):
            row = one_row(filters, modo="vigilancia", k=k)
            assert f"{pt_percent(row['vpn_nonmatch'])}% em k={k}" in section
    answer30 = script.split("## Perguntas prováveis")[1].split("### 30.")[1].split("### 31.")[0]
    assert "confirmação k=125" in answer30
    assert "protocolo fixado antes dos desfechos" in answer30
    for number in (21, 22, 26, 28):
        assert len(frames[number - 1].split("\\fonte")[0].split()) <= 210


def test_slide_s81_contract_rejects_changed_human_comparison():
    def changed(name):
        rows = read_csv(name)
        if name == "comparador_p_match_humano.csv":
            row = one_row(rows, modo="vigilancia")
            row["P_par2_recuperados"] = str(int(row["P_par2_recuperados"]) + 1)
        return rows
    with pytest.raises(AssertionError):
        assert_fragments(slide_frames()[21], scientific_requirements(changed)[22])


def test_human_scenarios_are_coverage_not_measured_error():
    rows = read_csv("cenarios_sensibilidade_humana.csv")
    assert {float(r["s"]) for r in rows} == {0.95, 0.99, 1.0}
    assert all(float(r["especificidade_humana_assumida"]) == 1 for r in rows)
    for row in rows:
        assert float(row["sensibilidade_total"]) == pytest.approx(
            (float(row["vinculos_aceitos_sem_humano"]) + float(row["s"])*float(row["vinculos_ao_humano"]))
            / float(row["vinculos_total"])
        )
    for mode in ("vigilancia", "confirmacao"):
        keys = {(r["estrategia"], r["k"]) for r in rows if r["modo"] == mode}
        series = {key: [one_row(rows, modo=mode, estrategia=key[0], k=key[1], s=s)
                        for s in ("0.95", "0.99", "1.0")] for key in keys}
        for left in series.values():
            for right in series.values():
                if float(left[0]["sensibilidade_total"]) > float(right[0]["sensibilidade_total"]):
                    assert all(float(a["sensibilidade_total"]) >= float(b["sensibilidade_total"]) - 1e-12
                               for a, b in zip(left, right))
    text = slide_frames()[27]
    assert all(f"{value}\\%" in text for value in (95, 99, 100))
    assert "As perdas são de cobertura" in text


def test_slides_no_control_characters_and_definitions_before_results():
    for path in SLIDES.glob("*.tex"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", text), path
    frames = slide_frames()
    for definition in ("OOF: fora da amostra", "FP: falso positivo", "VPP: valor preditivo positivo", "O3:", "O4:"):
        assert definition in frames[3]
    assert "Na vigilância, nenhuma ordenação domina" in frames[3]
    assert "na confirmação, O2 domina fracamente P" in frames[3]
    for row in read_csv("subamostragem_treino_resumo.csv"):
        if row["population"] == "below" and row["variant"] == "isotonic" and row["metric"] == "brier" and row["ratio"] != "original":
            assert float(row["median"]) > float(row["reference"])


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
        "proporções congeladas",
        "viés de incorporação",
        "somente com o Zenodo",
        "desvios de protocolo",
        "estabilidade confirma",
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
    assert "SHA-256 dos PDFs: a atualizar no build final" in source


def thesis_title():
    source = (ROOT / "text" / "latex" / "tese.tex").read_text(encoding="utf-8")
    raw = re.search(r"\\Large \\bf (Aprendizado[^\n]*)\}$", source, re.M)[1]
    for latex, char in (("\\'a", "á"), ("\\'o", "ó"), ("\\'i", "í"), ("\\'u", "ú"), ("\\\\", " ")):
        raw = raw.replace(latex, char)
    return " ".join(raw.split())


def test_banca_e_title_terminology_and_new_answers():
    source = (SLIDES / "defesa.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{(.*)\}$", source, re.M)[1]
    assert " ".join(title.split()) == thesis_title()
    assert r"\textit{Linkage} Probabilístico" in title
    assert "Relacionamento Probabilístico" not in source
    frame_titles = re.findall(r"\\begin\{frame\}\{([^\n]*)\}", source)
    assert not any("Fellegi" in frame_title for frame_title in frame_titles)
    assert "decisão de limiar" in slide_frames()[7]
    assert not re.search(r"\btaxas?\b|estágio 1", source, re.I)
    script = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = script.split("## Perguntas prováveis")[1].split("## Figuras e critérios")[0]
    answers = {n: questions.split(f"### {n}.")[1].split("### ")[0] for n in range(34, 43)}
    for number, fragments in {
        34: ("0,954", "21.620 para 41 pares", "18 min", "Platt ajustado no treino",
             "esperado, não observado", "sem chamada a LLM"),
        35: ("211 dos 247 vínculos (85,4%)", "preserva"),
        36: ("homonímia", "gzcmd/guardrails.py", "exploratórios"),
        37: ("README", "notebooks/README.md", "apendice_pacote.tex"),
        38: ("Anthropic", "OpenCode Go", "Levanto", "Fireworks AI", "OpenAI", "Modal",
             "ambiente local"),
        39: ("retirado",),
        40: ("Comparador de Registros", "sem interface gráfica"),
        41: ("não consumiu a saída do OpenRecLink", "OpenRecLink original, em C",
             "Daniela Medina Coeli Bastos", "extrato pós-CdR"),
        42: ("LinkDataPop", "acesso restrito", "agregados", "MIT vale somente para o código"),
    }.items():
        for fragment in fragments:
            assert fragment in answers[number], (number, fragment)
    costs = {row["braco"]: row for row in read_csv("custos_equivalentes_exploratorios.csv")
             if row["modo"] == "vigilancia"}
    frame22 = slide_frames()[21].replace("{,}", ",")
    for arm in ("P", "O3_recuperacao_equivalente", "P_MATCH_ao_humano"):
        value = float(costs[arm]["subtotal_revisao_fp"])
        text = pt_number(value, 2) if not value.is_integer() else integer(value)
        assert text in frame22, arm


def test_roteiro_repassada_final_preserva_reprodutibilidade_e_limite_da_estabilidade():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    section26 = source.split("### 26.", 1)[1].split("### 27.", 1)[0]
    assert "a análise de estabilidade mede estabilidade nos mesmos dados, não generalização" in section26
    verification = source.split("### Verificação técnica do build oficial", 1)[1].split(
        "### Rastreabilidade dos números do arranjo", 1
    )[0]
    assert "as folhas da ficha correspondem às páginas físicas do PDF" in verification
    assert "As 52 respostas têm limite de 70 palavras, conforme contagem por espaços" in verification
    assert "Em outra máquina, é necessário instalar Beamer, translator, PGF, babel com português e Latin Modern, ou disponibilizá-los em diretório local de dependências." in verification
