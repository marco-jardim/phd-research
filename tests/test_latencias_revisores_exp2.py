import pytest

from scripts.latencias_revisores_exp2 import resumir_latencias


def test_latency_percentiles_are_not_parallel_batch_duration():
    data = {"mode": "vigilancia", "workers": 2,
            "metrics": {"total_elapsed_s": 4.0},
            "results": [{"total_latency_s": value} for value in [1, 2, 3, 4]]}
    result = resumir_latencias(data)
    assert result["mediana_s"] == 2.5
    assert result["p90_s"] == pytest.approx(3.7)
    assert result["duracao_lote_s"] == 4
    assert result["trabalhadores"] == 2


def test_rejects_invalid_latencies():
    with pytest.raises(ValueError, match="Latências"):
        resumir_latencias({"mode": "vigilancia", "results": []})
