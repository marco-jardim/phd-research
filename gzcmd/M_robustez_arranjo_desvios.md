# Robustez: limites da execução e da validação

Este complemento não altera o pré-registro, as notas anteriores ou os fingerprints históricos.

- B1: os treinos foram executados com o teste de fingerprint vermelho. A posterior restauração dos bytes históricos e a emenda técnica documentam a execução, mas não convertem a checagem anterior em aprovada.
- B2: a checagem da identidade efetiva do dossiê foi transferida para o wrapper novo de robustez. Essa proteção não passou a integrar os validadores históricos restaurados.
- B3: `frozen_responses` restringe as fontes usadas para estimar taxas, mas não valida o vocabulário nem a completude dessas respostas. A auditoria posterior não substitui essa validação na entrada.

Os testes adicionais da S6 ficam em arquivo novo, fora dos fingerprints fixados. Exercitam partições efetivas das réplicas, fontes congeladas e a regeneração agregada da fase C a partir dos CSVs versionados. Não repetem treino real, chamadas ao LLM ou avaliação de dados privados.
