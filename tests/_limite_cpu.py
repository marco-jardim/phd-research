"""Limite de CPU da suite de testes (stdlib apenas).

Por padrão a suite usa metade das CPUs lógicas, em prioridade baixa e com as
bibliotecas de álgebra linear limitadas ao mesmo número de threads. Isso impede
que ``pytest -n 4`` (e os ``n_jobs`` dos modelos) saturem a máquina sem alterar
nenhum parâmetro da tese: a afinidade limita o uso de CPU sem mudar a numérica.

Variáveis de ambiente:
- ``PYTEST_MAX_CPUS=<n>``: número de CPUs lógicas permitidas (inteiro >= 1).
- ``PYTEST_CPU_LIMIT=0`` (ou ``false``/``off``): desativa tudo.
- ``CI`` definida: só as variáveis de threads são aplicadas (sem afinidade nem
  prioridade).

Falhas viram ``RuntimeWarning`` com o código de erro; nunca são engolidas.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping

VARIAVEIS_THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_DESLIGADO = {"0", "false", "off"}
_BELOW_NORMAL_PRIORITY_CLASS = 0x4000
_NICE_ALVO = 10


def desativado(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("PYTEST_CPU_LIMIT", "").strip().lower() in _DESLIGADO


def em_ci(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("CI", "").strip())


def limite_de_cpus(env: Mapping[str, str] | None = None) -> int:
    """N = PYTEST_MAX_CPUS (inteiro >= 1) ou metade das CPUs lógicas (mín. 1)."""
    env = os.environ if env is None else env
    total = os.cpu_count() or 1
    bruto = env.get("PYTEST_MAX_CPUS", "").strip()
    if bruto:
        try:
            valor = int(bruto)
        except ValueError:
            valor = 0
        if valor >= 1:
            return min(valor, total)
        warnings.warn(
            f"PYTEST_MAX_CPUS={bruto!r} inválido (esperado inteiro >= 1); "
            "usando metade das CPUs lógicas",
            RuntimeWarning,
            stacklevel=2,
        )
    return max(1, total // 2)


def _kernel32():
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k32.GetPriorityClass.restype = ctypes.c_uint32
    k32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return k32


def _aviso(o_que: str, codigo: object) -> None:
    warnings.warn(
        f"limite de CPU dos testes: falha ao {o_que} (código {codigo})",
        RuntimeWarning,
        stacklevel=3,
    )


def definir_threads(n: int, env: dict[str, str] | None = None) -> None:
    alvo = os.environ if env is None else env
    for nome in VARIAVEIS_THREADS:
        alvo.setdefault(nome, str(n))


def baixar_prioridade() -> None:
    if sys.platform == "win32":
        import ctypes

        k32 = _kernel32()
        if not k32.SetPriorityClass(
            k32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS
        ):
            _aviso("baixar a prioridade", ctypes.get_last_error())
    elif hasattr(os, "nice"):
        try:
            atual = os.nice(0)
            if atual < _NICE_ALVO:
                os.nice(_NICE_ALVO - atual)
        except OSError as exc:
            _aviso("baixar a prioridade", exc.errno)


def restringir_afinidade(n: int) -> None:
    if sys.platform == "win32":
        import ctypes

        k32 = _kernel32()
        proc = ctypes.c_size_t()
        sistema = ctypes.c_size_t()
        handle = k32.GetCurrentProcess()
        if not k32.GetProcessAffinityMask(
            handle, ctypes.byref(proc), ctypes.byref(sistema)
        ):
            _aviso("ler a afinidade", ctypes.get_last_error())
            return
        mascara, escolhidos = 0, 0
        for bit in range(sistema.value.bit_length()):
            if escolhidos == n:
                break
            if sistema.value >> bit & 1:
                mascara |= 1 << bit
                escolhidos += 1
        if not k32.SetProcessAffinityMask(handle, mascara):
            _aviso("restringir a afinidade", ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))[:n]
            os.sched_setaffinity(0, cpus)
        except OSError as exc:
            _aviso("restringir a afinidade", exc.errno)
    # macOS (sem API de afinidade): nada a fazer.


def afinidade_atual() -> int | None:
    """Número de CPUs lógicas em que o processo pode rodar; None se não suportado."""
    if sys.platform == "win32":
        import ctypes

        k32 = _kernel32()
        proc = ctypes.c_size_t()
        sistema = ctypes.c_size_t()
        if not k32.GetProcessAffinityMask(
            k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(sistema)
        ):
            return None
        return bin(proc.value).count("1")
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def prioridade_baixa() -> bool | None:
    """True se a prioridade está abaixo do normal; None se não suportado."""
    if sys.platform == "win32":
        k32 = _kernel32()
        classe = k32.GetPriorityClass(k32.GetCurrentProcess())
        return classe in (_BELOW_NORMAL_PRIORITY_CLASS, 0x40)  # below normal, idle
    if hasattr(os, "nice"):
        return os.nice(0) > 0
    return None


def aplicar_limite_de_cpu(env: Mapping[str, str] | None = None) -> int | None:
    """Aplica o limite ao processo atual; devolve N, ou None se desativado."""
    if desativado(env):
        return None
    n = limite_de_cpus(env)
    definir_threads(n, os.environ if env is None else None)
    if not em_ci(env):
        baixar_prioridade()
        restringir_afinidade(n)
    return n
