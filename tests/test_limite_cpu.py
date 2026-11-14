"""Testes do limite de CPU da suite (tests/_limite_cpu.py, aplicado no conftest)."""

from __future__ import annotations

import os
import subprocess
import sys

import _limite_cpu as lc
import pytest

ATIVO = not lc.desativado()
CI = lc.em_ci()


def test_variaveis_de_threads_definidas():
    if not ATIVO:
        pytest.skip("PYTEST_CPU_LIMIT desativa o limite")
    for nome in lc.VARIAVEIS_THREADS:
        assert os.environ.get(nome), nome


def test_afinidade_limitada_a_n_cpus():
    if not ATIVO or CI:
        pytest.skip("limite desativado ou ambiente de CI")
    atual = lc.afinidade_atual()
    if atual is None:
        pytest.skip("afinidade de CPU não suportada nesta plataforma")
    assert 1 <= atual <= lc.limite_de_cpus()


def test_prioridade_abaixo_do_normal():
    if not ATIVO or CI:
        pytest.skip("limite desativado ou ambiente de CI")
    baixa = lc.prioridade_baixa()
    if baixa is None:
        pytest.skip("prioridade não suportada nesta plataforma")
    assert baixa


def test_limite_de_cpus_respeita_pytest_max_cpus():
    assert lc.limite_de_cpus({"PYTEST_MAX_CPUS": "1"}) == 1
    assert lc.limite_de_cpus({}) == max(1, (os.cpu_count() or 1) // 2)
    with pytest.warns(RuntimeWarning, match="inválido"):
        assert lc.limite_de_cpus({"PYTEST_MAX_CPUS": "0"}) >= 1


def test_desativado_e_ci_reconhecem_valores():
    for valor in ("0", "false", "OFF"):
        assert lc.desativado({"PYTEST_CPU_LIMIT": valor})
    assert not lc.desativado({"PYTEST_CPU_LIMIT": "1"})
    assert not lc.desativado({})
    assert lc.em_ci({"CI": "true"})
    assert not lc.em_ci({})


def _sonda(env_extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in lc.VARIAVEIS_THREADS}
    env.pop("CI", None)
    env.pop("PYTEST_CPU_LIMIT", None)
    env.pop("PYTEST_MAX_CPUS", None)
    env.update(env_extra)
    codigo = (
        "import os, _limite_cpu as lc; "
        "lc.aplicar_limite_de_cpu(); "
        "print(lc.afinidade_atual(), lc.prioridade_baixa(), "
        "os.environ.get('OMP_NUM_THREADS'))"
    )
    saida = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", codigo],
        env={**env, "PYTHONPATH": os.path.dirname(__file__)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return dict(zip(("afinidade", "baixa", "omp"), saida, strict=True))


def test_opt_out_nao_altera_nada_em_subprocesso():
    # O filho herda afinidade/prioridade do pai; só as variáveis de threads
    # provam o opt-out.
    r = _sonda({"PYTEST_CPU_LIMIT": "0"})
    assert r["omp"] == "None"


def test_limite_em_subprocesso_define_threads_e_afinidade():
    if lc.afinidade_atual() is None:
        pytest.skip("afinidade de CPU não suportada nesta plataforma")
    r = _sonda({"PYTEST_MAX_CPUS": "1"})
    assert r["omp"] == "1"
    assert r["afinidade"] == "1"
    assert r["baixa"] == "True"


def test_ci_aplica_so_threads_em_subprocesso():
    r = _sonda({"CI": "true", "PYTEST_MAX_CPUS": "1"})
    assert r["omp"] == "1"
