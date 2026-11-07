# Agregados W5a

Os geradores não alteram o motor nem o perfil v3 histórico e não chamam LLM.

- `scripts/ablacao_dossie_proveniencia.py --source <raiz>/data/sprint3b`: lê somente campos permitidos dos JSONs privados e publica metadados e SHA-256 dos arquivos inteiros. Não publica resultados, identificadores ou hashes individuais de dossiês. Metadados ausentes não são inferidos do código atual. O histórico do runner é consultado com `git log -p --follow` e seu hash registrado. Sem prova de identidade, o prompt v4 das coortes completas não é atribuído aos braços selecionados.
- `scripts/ablacao_teste_par2_agregado.py`: `dtype=str`, conversão explícita, `engineer_features` e `split_data(random_state=42, test_size=0.3)`, sem retreino. Recompõe o teste dos classificadores (18.509 potenciais pares, 66 PAR=1, oito PAR=2), não o experimento distinto GZ-CMD por COMPREC do apêndice.
- `scripts/perfil_obitos_agregado.py`: deriva idade pelas funções clínicas existentes (datas de nascimento/óbito do SIM, dias/365,25), restringe a 0 a 120, usa DP amostral e conserva sexo textual. REFREC identifica SIM; COMPREC identifica Sinan. Nos grupos de registros, demografia só é conservada se completa e invariante entre candidatos; a nota é média das notas dos pares. Não se atribui escore de par ao óbito. As linhas dos 27 recuperados e 40 comuns são importadas do agregado histórico já versionado `data/epidemiological_profile.json`, sem novo ajuste do RF, mantendo arredondamento e ausências originais.

Saídas públicas: `data/sprint3b/ablacao/ablacao_dossie_proveniencia.json`,
`data/ablacao_teste_composicao.csv`, `data/perfil_obitos_agregado.csv` e tabelas
editoriais. Os testes portáveis usam os agregados; verificações que leem
insumos privados utilizam `tests/local_data.py` e `@requires_files`.
