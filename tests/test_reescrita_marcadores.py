"""Blindagem de números na reescrita por LLM: mascarar, verificar e restaurar."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import reescrita_marcadores as rm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reescrita_marcadores.py"
FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

SAMPLE = (
    r"A amostra tem 61.696 pares (75,0\% do total), com VPP de 0,931." "\n"
    r"O intervalo foi de 6,4 a 34,7 pontos (Tabela~\ref{tab:res2}; \cite{silva2020})." "\n"
    r"A diferença foi de $-5{,}648$ unidades." "\n"
)
ANTES = "Antes 11\n"
DEPOIS = "Depois 22\n"
ARQ = "text/latex/cap.tex"


def M(k):
    return f"\u27e6{k}\u27e7"


def preparar(tmp_path, eol="\n"):
    """Escreve o .tex (linhas 2-4 são o trecho) e roda 'mascarar'. Devolve (bytes, caminho)."""
    original = (ANTES + SAMPLE + DEPOIS).replace("\n", eol).encode("utf-8")
    arq = tmp_path / ARQ
    arq.parent.mkdir(parents=True, exist_ok=True)
    arq.write_bytes(original)
    rc = rm.main(["mascarar", "--raiz", str(tmp_path), "--arquivo", ARQ, "--linhas", "2-4", "--id", "t"])
    assert rc == 0
    return original, arq


def editar(tmp_path, texto):
    ed = tmp_path / "editado.tex"
    ed.write_bytes(texto.encode("utf-8"))
    return str(ed)


def rodar(tmp_path, comando, texto=None, *extra):
    args = [comando, "--raiz", str(tmp_path), "--id", "t", *extra]
    if texto is not None:
        args += ["--entrada", editar(tmp_path, texto)]
    return rm.main(args)


# ---------------------------------------------------------------------------
# Mascarar
# ---------------------------------------------------------------------------


def test_numeros_de_prosa_sao_mascarados():
    mascarado, itens = rm.mascarar_texto(SAMPLE)
    assert [i["texto"] for i in itens] == ["61.696", "75,0", "0,931", "6,4", "34,7", "-5{,}648"]
    assert [i["k"] for i in itens] == [1, 2, 3, 4, 5, 6]
    assert f"tem {M(1)} pares ({M(2)}\\% do total), com VPP de {M(3)}." in mascarado
    assert f"de {M(4)} a {M(5)} pontos" in mascarado
    assert f"foi de ${M(6)}$ unidades" in mascarado
    assert r"\ref{tab:res2}" in mascarado and r"\cite{silva2020}" in mascarado


def test_k_e_por_ocorrencia_nao_por_valor():
    mascarado, itens = rm.mascarar_texto("Foram 5 casos e 5 óbitos.\n")
    assert [i["k"] for i in itens] == [1, 2]
    assert mascarado == f"Foram {M(1)} casos e {M(2)} óbitos.\n"


def test_espaco_fino_e_ano_sao_mascarados():
    mascarado, itens = rm.mascarar_texto("Em 2020 havia 61\\,696 registros.\n")
    assert [i["texto"] for i in itens] == ["2020", "61\\,696"]


def test_estruturas_latex_nao_sao_mascaradas():
    texto = (
        r"\begin{figure}[htbp]" "\n"
        r"\includegraphics[width=0.5\textwidth]{figuras/fig1.pdf}" "\n"
        r"\caption{Resultado 7 \label{fig:2}}" "\n"
        r"\end{figure}" "\n"
        r"\begin{tabular}{lrr}" "\n"
        r"\toprule" "\n"
        r"\multicolumn{2}{c}{Total} & 12 \\[2pt]" "\n"
        r"\cmidrule(lr){2-3}" "\n"
        r"\hspace{2cm} \vspace{-1mm} Ver Tabela~\ref{tab:3} e \cite{silva2020a,costa2019}." "\n"
        r"\begin{tabular}{p{3cm}r}" "\n"
        r"\end{tabular}" "\n"
        r"Largura 0.5\textwidth e recuo de 2cm." "\n"
        r"% comentário com 99" "\n"
    )
    mascarado, itens = rm.mascarar_texto(texto)
    assert [i["texto"] for i in itens] == ["7", "12"]
    for trecho in (
        r"\begin{figure}[htbp]",
        r"\includegraphics[width=0.5\textwidth]{figuras/fig1.pdf}",
        r"\label{fig:2}",
        r"\begin{tabular}{lrr}",
        r"\multicolumn{2}{c}{Total}",
        r"\\[2pt]",
        r"\cmidrule(lr){2-3}",
        r"\hspace{2cm}",
        r"\vspace{-1mm}",
        r"\ref{tab:3}",
        r"\cite{silva2020a,costa2019}",
        r"\begin{tabular}{p{3cm}r}",
        r"Largura 0.5\textwidth e recuo de 2cm.",
        r"% comentário com 99",
    ):
        assert trecho in mascarado


def test_mascarar_grava_so_em_local_reescrita_com_mapa(tmp_path):
    original, _ = preparar(tmp_path)
    pasta = tmp_path / ".local" / "reescrita"
    assert sorted(p.name for p in pasta.iterdir()) == ["t.mapa.json", "t.mascarado.tex"]
    mapa = json.loads((pasta / "t.mapa.json").read_text(encoding="utf-8"))
    assert mapa["arquivo"] == ARQ and mapa["linhas"] == [2, 4]
    assert mapa["sha256"] == rm.sha256_texto(SAMPLE)
    assert [m["texto"] for m in mapa["marcadores"]][:2] == ["61.696", "75,0"]
    assert mapa["marcadores"][0]["linha"] == 2 and "61.696" in mapa["marcadores"][0]["contexto"]
    # O .tex não muda em 'mascarar'.
    assert (tmp_path / ARQ).read_bytes() == original


# ---------------------------------------------------------------------------
# Ida e volta, movimento, remoção
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_ida_e_volta_sem_edicao_e_byte_a_byte(tmp_path, eol, capsys):
    original, arq = preparar(tmp_path, eol)
    assert rodar(tmp_path, "restaurar") == 0
    assert arq.read_bytes() == original
    assert b"\xef\xbb\xbf" not in arq.read_bytes()
    out = capsys.readouterr().out
    assert "6 mantidos, 0 removidos, 0 movidos" in out


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_marcador_movido_e_removido_passa_e_preserva_eol(tmp_path, eol, capsys):
    original, arq = preparar(tmp_path, eol)
    editado = (
        f"O VPP foi {M(3)} e a amostra tem {M(1)} pares ({M(2)}\\%, ver \\ref{{tab:res2}}).\n"
        f"Diferença de ${M(6)}$.\n"
    )
    assert rodar(tmp_path, "restaurar", editado) == 0
    esperado = (
        ANTES
        + "O VPP foi 0,931 e a amostra tem 61.696 pares (75,0\\%, ver \\ref{tab:res2}).\n"
        + "Diferença de $-5{,}648$.\n"
        + DEPOIS
    ).replace("\n", eol)
    assert arq.read_bytes() == esperado.encode("utf-8")
    if eol == "\r\n":
        assert b"\n" not in arq.read_bytes().replace(b"\r\n", b"")
    out = capsys.readouterr().out
    assert out.count("AVISO: número removido") == 2
    assert "'6,4'" in out and "'34,7'" in out
    assert "4 mantidos, 2 removidos, 1 movidos" in out
    assert M(3) in out.split("movidos", 1)[1]


def test_verificar_nao_escreve(tmp_path, capsys):
    original, arq = preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"Só {M(1)}.\n") == 0
    assert arq.read_bytes() == original
    assert "nada escrito" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Violações
# ---------------------------------------------------------------------------


def test_marcador_duplicado_falha(tmp_path, capsys):
    original, arq = preparar(tmp_path)
    assert rodar(tmp_path, "restaurar", f"{M(1)} e de novo {M(1)}.\n") == 1
    out = capsys.readouterr().out
    assert "marcador duplicado" in out and M(1) in out
    assert arq.read_bytes() == original


def test_marcador_inventado_falha(tmp_path, capsys):
    original, arq = preparar(tmp_path)
    assert rodar(tmp_path, "restaurar", f"{M(1)} e {M(99)}.\n") == 1
    assert "marcador desconhecido" in capsys.readouterr().out
    assert arq.read_bytes() == original


def test_marcador_malformado_falha(tmp_path, capsys):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"{M(1)} e \u27e6x\u27e7.\n") == 1
    assert "malformado" in capsys.readouterr().out


def test_digito_livre_falha(tmp_path, capsys):
    original, arq = preparar(tmp_path)
    assert rodar(tmp_path, "restaurar", f"{M(1)} pares em 42 casos.\n") == 1
    out = capsys.readouterr().out
    assert "dígito livre '42'" in out and "linha 2 do .tex" in out
    assert arq.read_bytes() == original


def test_digito_colado_a_marcador_falha(tmp_path):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"{M(1)}5 pares.\n") == 1


def test_estrutura_isenta_nova_com_digitos_falha(tmp_path, capsys):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"{M(1)} \\hspace{{3cm}} pares.\n") == 1
    assert "estrutura LaTeX isenta" in capsys.readouterr().out


def test_permitir_novos_libera(tmp_path, capsys):
    _, arq = preparar(tmp_path)
    texto = f"{M(1)} pares em 42 casos.\n"
    assert rodar(tmp_path, "verificar", texto) == 1
    capsys.readouterr()
    assert rodar(tmp_path, "restaurar", texto, "--permitir-novos", "42, 7") == 0
    out = capsys.readouterr().out
    assert "LIBERADO: número novo autorizado 42" in out
    assert "42" in arq.read_text(encoding="utf-8") and "61.696 pares em 42 casos." in arq.read_text(
        encoding="utf-8"
    )


def test_permitir_novos_com_virgula_decimal_usa_ponto_e_virgula(tmp_path, capsys):
    preparar(tmp_path)
    texto = f"{M(1)} com 0,95 e 12.\n"
    assert rodar(tmp_path, "verificar", texto, "--permitir-novos", "0,95") == 1
    capsys.readouterr()
    assert rodar(tmp_path, "verificar", texto, "--permitir-novos", "0,95;12") == 0
    assert rodar(tmp_path, "verificar", texto, "--permitir-novos", "0,95", "--permitir-novos", "12") == 0
    assert "0,95" in capsys.readouterr().out


def test_parse_permitir_regras():
    assert rm.parse_permitir(["0,95"]) == ["0,95"]
    assert rm.parse_permitir(["0,95;12"]) == ["0,95", "12"]
    assert rm.parse_permitir(["12, 7"]) == ["12", "7"]
    assert rm.parse_permitir(["61.696", "-5{,}648"]) == ["61.696", "-5{,}648"]
    assert rm.parse_permitir(None) == []


def test_valor_liberado_vale_para_todas_as_ocorrencias(tmp_path, capsys):
    preparar(tmp_path)
    texto = f"{M(1)} com 42 e de novo 42.\n"
    assert rodar(tmp_path, "verificar", texto, "--permitir-novos", "42") == 0
    assert capsys.readouterr().out.count("LIBERADO") == 2


def test_permitir_ids_libera_id_exato(tmp_path, capsys):
    preparar(tmp_path)
    texto = f"{M(1)} pares ordenados por O3 e F4.\n"
    assert rodar(tmp_path, "verificar", texto) == 1
    capsys.readouterr()
    assert rodar(tmp_path, "verificar", texto, "--permitir-ids", "O3,F4") == 0


def test_permitir_ids_nao_libera_id_fora_da_lista(tmp_path, capsys):
    preparar(tmp_path)
    texto = f"{M(1)} pares ordenados por O3 e F4.\n"
    assert rodar(tmp_path, "verificar", texto, "--permitir-ids", "O3") == 1
    out = capsys.readouterr().out
    assert "dígito livre '4'" in out and "dígito livre '3'" not in out


@pytest.mark.parametrize("trecho", ["O35", "BO3", "O3x"])
def test_permitir_ids_exige_identificador_inteiro(tmp_path, capsys, trecho):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"{M(1)} pares em {trecho}.\n", "--permitir-ids", "O3") == 1
    assert "dígito livre" in capsys.readouterr().out


def test_permitir_ids_nao_libera_numero_solto(tmp_path, capsys):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", f"{M(1)} pares em O3 e 3 casos.\n", "--permitir-ids", "O3") == 1
    out = capsys.readouterr().out
    assert "dígito livre '3'" in out and out.count("dígito livre") == 1


def test_permitir_ids_aparece_no_resumo(tmp_path, capsys):
    _, arq = preparar(tmp_path)
    texto = f"{M(1)} pares por O3, depois F4 e de novo O3.\n"
    assert rodar(tmp_path, "restaurar", texto, "--permitir-ids", "O3", "--permitir-ids", "F4") == 0
    out = capsys.readouterr().out
    assert out.count("LIBERADO-ID") == 3
    assert "LIBERADO-ID: identificador autorizado O3" in out and "F4" in out
    assert "61.696 pares por O3, depois F4 e de novo O3." in arq.read_text(encoding="utf-8")


def test_permitir_ids_formato_invalido_e_erro_de_uso(tmp_path, capsys):
    preparar(tmp_path)
    assert rodar(tmp_path, "verificar", None, "--permitir-ids", "3,O") == 2
    assert "--permitir-ids inválido" in capsys.readouterr().err


def test_sha256_divergente_falha(tmp_path, capsys):
    original, arq = preparar(tmp_path)
    mudado = original.replace(b"6,4", b"6,5")
    arq.write_bytes(mudado)
    assert rodar(tmp_path, "restaurar") == 1
    assert "sha256" in capsys.readouterr().out
    assert arq.read_bytes() == mudado
    assert rodar(tmp_path, "verificar") == 1


def test_restaurar_duas_vezes_falha_por_sha(tmp_path, capsys):
    preparar(tmp_path)
    editado = f"Só {M(3)}.\n"
    assert rodar(tmp_path, "restaurar", editado) == 0
    assert rodar(tmp_path, "restaurar", editado) == 1
    assert "sha256" in capsys.readouterr().out


def test_mapa_ausente_e_erro_de_uso(tmp_path, capsys):
    assert rodar(tmp_path, "verificar") == 2
    assert "mapa não encontrado" in capsys.readouterr().err


def test_linhas_fora_do_arquivo_e_erro_de_uso(tmp_path):
    arq = tmp_path / ARQ
    arq.parent.mkdir(parents=True)
    arq.write_bytes(b"uma linha 1\n")
    assert rm.main(["mascarar", "--raiz", str(tmp_path), "--arquivo", ARQ, "--linhas", "1-5"]) == 2


@pytest.mark.parametrize("rel", [
    "text/latex/tables/tab_x.tex",
    "text/latex/tables/editorial/tab_y.tex",
    "text/latex/cap/../tables/editorial/tab_y.tex",
])
def test_mascarar_recusa_tabelas_da_tese(tmp_path, capsys, rel):
    tabela = tmp_path / "text/latex/tables/editorial/tab_y.tex"
    tabela.parent.mkdir(parents=True)
    tabela.write_bytes(b"a & 0,931 \\\\\n")
    antes = tabela.read_bytes()
    rc = rm.main(["mascarar", "--raiz", str(tmp_path), "--arquivo", rel, "--linhas", "1-1", "--id", "tab"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "tabelas mudam só pelo gerador" in err
    assert not (tmp_path / rm.SAIDA_REL).exists()
    assert tabela.read_bytes() == antes


def test_mascarar_aceita_arquivo_fora_de_tables(tmp_path):
    arq = tmp_path / "text/latex/tablesx/cap.tex"
    arq.parent.mkdir(parents=True)
    arq.write_bytes(b"valor 12\n")
    assert rm.main(["mascarar", "--raiz", str(tmp_path), "--arquivo", "text/latex/tablesx/cap.tex",
                    "--linhas", "1-1", "--id", "ok"]) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ponta_a_ponta(tmp_path):
    original = (ANTES + SAMPLE + DEPOIS).encode("utf-8")
    arq = tmp_path / ARQ
    arq.parent.mkdir(parents=True)
    arq.write_bytes(original)

    def cli(*args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args, "--raiz", str(tmp_path)],
            capture_output=True, text=True, encoding="utf-8", creationflags=FLAGS,
        )

    r = cli("mascarar", "--arquivo", ARQ, "--linhas", "2-4", "--id", "cli")
    assert r.returncode == 0, r.stderr
    assert "6 números" in r.stdout
    assert cli("verificar", "--id", "cli").returncode == 0
    assert cli("restaurar", "--id", "cli").returncode == 0
    assert arq.read_bytes() == original
