import json
import os
from pathlib import Path

import pandas as pd
import pytest

from scripts.area_cinza_por_par import carregar, gravar, resumir
from scripts.sensibilidade_gzcmd_por_par import executar, identificar
from local_data import DATA_ROOT, requires_files

EXTRATO = Path(os.environ.get("EXTRATO_CSV", DATA_ROOT / "data/COMPARADORSEMIDENT.csv"))
FUNIL = Path(os.environ.get("FUNIL_DATA_DIR", DATA_ROOT / "data"))


def test_saida_agregada_metadados_e_regra_de_parada(tmp_path):
    path = tmp_path / "extrato.csv"
    path.write_text(
        "PAR;COMPREC,C,12,0;REFREC,C,12,0;PASSO;nota final\n"
        "1;a;r1;6;5\n1;b;r2;9;8\n2;c;r3;11;7\n0;d;r4;11;4\n",
        encoding="utf-8",
    )
    frame = identificar(carregar(path, identificadores=True))
    funnel = frame[["pair_id", "nota"]].copy()
    funnel["y"] = [1, 1, 1, 0]
    funnel["final"] = ["MATCH", "DEFERRED", "NONMATCH", "NONMATCH"]
    metrics = []
    for mode in ("vigilancia", "confirmacao"):
        funnel.to_csv(tmp_path / f"funil_gzcmd_{mode}.csv", index=False)
        metrics.append({
            "modo": mode, "versao": "v3.1", "VP_cadeia": 1,
            "FP_cadeia": 0, "FN_cadeia": 2, "VN_cadeia": 1,
            "sens_cadeia_observada": 1 / 3,
        })
    pd.DataFrame(metrics).to_csv(tmp_path / "funil_gzcmd_final_metricas.csv", index=False)
    out = tmp_path / "sensibilidade.csv"
    result = executar(path, tmp_path, out)
    metadata = json.loads(out.with_name("sensibilidade_meta.json").read_text(encoding="utf-8"))
    assert metadata["regra_de_parada"] is True
    assert metadata["area_cinza"] == {
        "vinculos": 3, "corte_orl": 2,
        "cascata": {"vigilancia": 1, "confirmacao": 1},
    }
    assert len(metadata["fontes"]) == 4
    assert all(len(source["sha256"]) == 64 for source in metadata["fontes"])
    assert len(result) == 8
    assert "pair_id" not in out.read_text(encoding="utf-8").splitlines()[0]
    area = tmp_path / "area.csv"
    gravar(resumir(carregar(path)), area, {"somente_agregados": True})
    assert pd.read_csv(area).vinculos.sum() == 3
    assert area.with_name("area_meta.json").exists()


@requires_files(EXTRATO, FUNIL / "funil_gzcmd_vigilancia.csv",
                FUNIL / "funil_gzcmd_confirmacao.csv", FUNIL / "funil_gzcmd_final_metricas.csv")
def test_sensibilidade_local_quando_disponivel(tmp_path):
    extrato, data_dir = EXTRATO, FUNIL
    out = tmp_path / "sensibilidade.csv"
    table = executar(extrato, data_dir, out)
    assert table.aceitos_automaticamente.tolist() == [184, 10, 105, 8, 124, 8, 61, 4]
