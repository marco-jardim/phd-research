# Agregados das explorações posteriores sobre filas de revisão e LLM

Resultados agregados que sustentam os números do Apêndice "Explorações posteriores: filas com aprendizado e LLM abaixo do corte" (`text/latex/apendice_gzcmd.tex`, seção `sec:e4-ap-filas-llm`).

- **Só agregados.** Nenhum arquivo contém identificador de par, chave ou decisão por par. As decisões do LLM por par permanecem locais (no repositório do pacote, `data/llm_fixtures/` é ignorado pelo git).
- **Origem:** pacote GZ-CMD (`gzcmd-record-linkage`), ramo `feat/fila-autocalibravel`, pasta `docs/`; tag `v0.7.0-exp` (commit `9c52c414768e21acf1804fcd9d73ff1633fa6180`, não publicada no PyPI). Os resultados do passo 1 vêm do commit `7fa1e1f`.
- **Extrato:** o mesmo da tese (61.696 potenciais pares; 247 pares verdadeiros, 211 aceitos pelo corte e 36 abaixo dele).
- **Ressalvas:** revisor humano simulado e perfeito; veredito do LLM em uma única realização; um só extrato.

## Arquivos esperados

| Arquivo | Sustenta |
|---|---|
| `llm_real_haiku_sem_nota.csv` | Tabela do Haiku 5.5 sem a nota final; VPP projetado |
| `llm_real_fila_ponta_a_ponta.csv` | Execução ponta a ponta da fila autocalibrável |
| `llm_real_desempate_zona_cinza.csv` | Desempate pelo LLM (tabela de janelas; IC; chamadas) |
| `sim_fila_autocalibravel_resultados.csv` | Simulação da fila autocalibrável |
| `engenharia_de_filas_resultados.csv` | Hipóteses de engenharia de filas (sintético e real) |
| `engenharia_de_filas_real_h8_h11.csv` | *Bandit* e auditor no extrato real |
| `engenharia_de_filas_replicacao_passo.csv` | Replicação por passo de blocagem |
| `llm_fixture_haiku-5-5_v4_sem-nota.rates.csv` | Proporções do Haiku por faixa da nota final |
| `comparacao_o1.csv` | Comparação das estratégias com a ordenação O1 (escore do OpenRecLink) |

## Manifesto

`MANIFEST.sha256` lista o sha256 de cada arquivo acima. Foi gerado, junto com a cópia, pelo comando registrado em `PROPOSTA_FILAS_LLM.md` (seção "Pendências do passo 1"). Para conferir:

```powershell
Get-Content data/filas_llm/MANIFEST.sha256
Get-ChildItem data/filas_llm/*.csv | Get-FileHash -Algorithm SHA256
```
