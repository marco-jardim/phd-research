"""Minimal synthetic triage example for gzcmd-record-linkage 0.3.0."""

import pandas as pd
from gzcmd_record_linkage.policy_engine import (
    Budget, Costs, LLMError, PolicyEngineV3,
)

pares = pd.DataFrame({
    "p_cal": [0.01, 0.50, 0.99],
    "band": ["grey_mid"] * 3,
})
motor = PolicyEngineV3(
    costs=Costs(false_positive=10, false_negative=50,
                llm_review=0.25),
    llm_error_by_band={"grey_mid": LLMError(0.06, 0.10)},
    budget=Budget(llm_max=1),
)
saida = motor.triage(pares)
print(saida[["p_cal", "action", "review_requested"]]
      .to_string(index=False))
