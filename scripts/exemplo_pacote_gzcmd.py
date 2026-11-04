"""Compare synthetic triage under v3 and v3.1 in gzcmd-record-linkage 0.4.0."""

import pandas as pd
from gzcmd_record_linkage.policy_engine import (
    Budget, Costs, LLMError, PolicyEngineV3,
)

pares = pd.DataFrame({
    "p_cal": [0.01, 0.50, 0.99],
    "band": ["grey_mid"] * 3,
    "nota_final": [6.0, 7.0, 8.0],
})
saida = pares[["p_cal"]].copy()
for politica in ("v3", "v3.1"):
    motor = PolicyEngineV3(
        costs=Costs(false_positive=10, false_negative=50,
                    llm_review=0.25),
        llm_error_by_band={"grey_mid": LLMError(0.06, 0.10)},
        budget=Budget(llm_max=1),
        policy_version=politica,
    )
    saida[politica] = motor.triage(pares)["action"]
print(saida.to_string(index=False))
