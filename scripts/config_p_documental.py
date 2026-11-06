"""Generate a documented P configuration without rewriting the frozen YAML.

Historical S1 still pins gzcmd_v3_config.yaml. This new snapshot documents the
P protocol and Haiku call parameters; it is not a standalone replay of S1.
"""
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
HEADER = """# Gerado por scripts/config_p_documental.py; perfil documental v3.1/P.
# O YAML original gzcmd_v3_config.yaml é congelado e deve permanecer intacto.
# Os runs históricos usaram aquele arquivo, não este snapshot posterior.
# Não basta carregar este YAML: executar o protocolo S1/S2/S3 de M_arranjo_corte.md.
# calibration abaixo descreve a inicialização/base; P substitui as probabilidades
# por recalibração abaixo do corte e as taxas por estimativas cross-fitted.
# anchors são alternativas históricas inativas em P, não a calibração de P.
# consistency / conflict_margin_delta: RESERVADO, NÃO IMPLEMENTADO.
# Não há restrição N:1 nem encaminhamento automático por margem de conflito.
# Parâmetros LLM documentam o experimento; este arquivo não faz chamadas.
"""


def generate(output=ROOT / "gzcmd/gzcmd_v31_p_config.yaml"):
    frozen = subprocess.check_output(["git", "show", "eaf00b5:gzcmd/gzcmd_v3_config.yaml"], cwd=ROOT)
    config = yaml.safe_load(frozen)
    for model in config["llm_review"]["model"].values():
        model.pop("temperature", None)
        model["max_tokens"] = 2500
    config["anchors"]["pos_anchor"]["target_mean_probability"] = 0.979
    config["anchors"]["neg_anchor"]["target_mean_probability"] = 0.0001
    config["evaluation"]["seeds"] = [42]
    config["evaluation"]["split_by"] = "refrec"
    config["arranjo_P_protocolo"] = {
        "consumido_pelo_runner_generico": False,
        "entrypoint": "scripts/arranjo_corte_s1_simular.py",
        "seed": 42, "outer_folds": 5, "inner_folds": 5, "group": "REFREC",
        "classifier_n_jobs": 8, "calibration_scope": "treino abaixo do corte",
        "calibration": "isotonica; Platt quando menos de 10 PAR=2",
        "reviewer_rates": "cross-fitted; seis coortes Haiku v4 congeladas",
        "prompt": "gzcmd_v4_llm_prompt.md", "dossier": "sem model_outputs",
        "historical_config": "gzcmd_v3_config.yaml",
    }
    output.write_text(HEADER + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                      encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    generate()
