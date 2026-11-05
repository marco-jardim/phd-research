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


def test_script_has_twenty_three_short_answers():
    source = (SLIDES / "roteiro_defesa.md").read_text(encoding="utf-8")
    questions = source.split("## Perguntas prováveis")[1].split(
        "## Figuras e critérios"
    )[0]
    answers = re.findall(r"### \d+\..*?\n\n(.*?)\n\nFonte:", questions, re.S)
    assert len(answers) == 23
    assert [int(n) for n in re.findall(r"^### (\d+)\.", questions, re.M)] == list(
        range(1, 24)
    )
    assert all(len(answer.split()) <= 70 for answer in answers)


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
        3: [pt_number(lower["pares_acima_limiar"], 0),
            pt_number(global_lower["pares_acima_limiar"], 0),
            "hipótese de revisão", "teto teórico", "serviço define"],
        4: ["Hipótese condicionada à carga", "não aceitos pelo corte", "duas fronteiras"],
        19: [],
        20: [],
        21: ["impede interpretação prospectiva", "Recomendação, não resultado observado",
             "encaminhar MATCH do LLM ao revisor humano", "não à aceitação automática",
             "avaliação registrada própria"],
        22: ["principal porque foi registrado antes", "dominado por P-Platt",
             "análise de sensibilidade", "não confirmatória", "dominância não sustentada",
             "Preservar o corte", "nenhuma variante superou a prática do serviço",
             "validação em dados novos"],
        23: ["proporção de vínculos", "definida pelo corte", "FP abaixo do corte"],
        24: ["Disparidade herdada do corte e do rótulo"],
        26: ["Respostas históricas reutilizadas", "conhecido antes do registro",
             "recalibração, braços e avaliação", "desvios de protocolo", "não generalização"],
        28: ["hipótese é condicionada à carga", pt_number(lower["pares_acima_limiar"], 0),
             "pior em todo desfecho medido contra o rótulo", "limite inferior derivado do rótulo",
             "avaliação registrada e o resultado negativo"],
        29: ["MATCH do LLM ao revisor humano, não aceitação automática", "registro próprio"],
        46: ["não confirmatória", "não generalização", "em relação à partição original",
             "A réplica original é um ponto dessa distribuição"],
    }
    if HOURS.exists():
        hours = csv_loader("carga_abaixo_corte_horas.csv")
        for threshold in ("por_passo", "global"):
            row = one_row(hours, limiar=threshold)
            requirements[3].append(
                f"{integer(row['horas_inteiras'])} h {integer(row['minutos_restantes'])} min"
            )
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
        requirements[20].extend((" & ".join(table), pt_percent(observed["perdidos_proporcao"])))
        requirements[20].append("/".join(
            integer(one_row(mechanisms, modo=mode, braco="P", mecanismo=mechanism)["vinculos_perdidos"])
            for mechanism in mechanisms_order
        ))
        requirements[28].append(integer(p["perdidos"]))
    cost_losses = one_row(
        mechanisms, modo="confirmacao", braco="P", mecanismo="AUTO_NONMATCH_custo"
    )
    requirements[20].extend((
        f"{integer(cost_losses['vinculos_perdidos'])} perdas por AUTO\\_NONMATCH",
        "Sem varredura de custos no arranjo",
    ))
    requirements[19].append(
        f"+{integer(principal['vigilancia']['vinc_humano'])} e "
        f"+{integer(principal['confirmacao']['vinc_humano'])} vínculos ao humano"
    )
    requirements[20].append(
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
    requirements[22].append(
        f"P-Platt aceita {integer(platt['aceitos'])}, com {integer(platt['fp'])} FP, "
        f"{integer(platt['pares_humano'])} ao humano, {integer(platt['pares_llm'])} ao LLM "
        f"e US\\$ {pt_number(platt['custo_llm_usd'], 2)}"
    )
    for mode, label in labels.items():
        dominance = [one_row(robustness, tipo="dominancia", modo=mode, bloco=block)
                     for block in ("N", "P")]
        counts = [f"{integer(row['k'])}/{integer(row['n'])}" for row in dominance]
        requirements[22].append(" e ".join(counts))
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
            f"P perde mais PAR~=~2 que o a posteriori, {short} & " + " & ".join(fractions)
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
        requirements[46].append(f"{pt_number(delta['maximo'], 2)} no bloco {block}")
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
    return requirements


@pytest.mark.parametrize("frame_number", [3, 4, 19, 20, 21, 22, 23, 24, 26, 28, 29, 46])
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
