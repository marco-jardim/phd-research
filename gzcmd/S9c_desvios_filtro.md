# Conferência S9c do filtro e limites da execução histórica

## Duplicatas e código usado nos desfechos

O protocolo `gzcmd/protocolo_filtro_llm.yaml` determina a última ocorrência
válida no mesmo arquivo. O coletor em `scripts/filtro_llm.py`, entretanto,
levanta `ValueError("Duplicate pair within source")` quando encontra duas
ocorrências de um identificador no mesmo arquivo. Trata-se de desvio da regra
registrada, sem efeito nos desfechos publicados: a conferência das 14 fontes
usadas encontrou zero linhas duplicadas por identificador em cada arquivo.
O total de 2.651 linhas é a soma das fontes, não uma contagem de potenciais
pares distintos entre fontes. As sobreposições entre coortes seguem a ordem
de precedência do coletor e não são duplicatas internas.

O SHA-256, com LF, do código usado nos desfechos é
`190c599e04a55b4c10e4e9801f06ce971b9da203efba0f13ed843f16ea5df190`
(`scripts/filtro_llm.py`). A evidência histórica é
`data/filtro_llm_metadata.json`, campo `code_sha256_lf`. O protocolo,
o coletor e esse metadata têm hashes fixados e não foram editados na S9c.
As contagens por fonte estão em `data/s9c_filtro_duplicatas.json`; somente
agregados e nomes de arquivos são publicados, sem identificadores por par.

## Pares e chamadas não são a mesma unidade

A execução adicional compreendeu **83 pares**: 12 na vigilância e 71 na
confirmação. Foram **201 chamadas de API**, não 83 chamadas. Um par pode
exigir mais de uma chamada aos agentes. Esses totais constam em
`data/filtro_llm_metadata.json`, objeto `new_execution`, campos `total_pairs`
e `total_api_calls`. A expressão histórica "83 chamadas de pares pendentes"
na autorização refere-se às listas de pares, não ao número de requisições
efetivamente realizadas. Esta conferência não executou chamadas de API.

## Monitor de custo congelado: limitação preservada, não correção silenciosa

`scripts/filtro_llm_executar.py` inicializa o custo em zero e soma os arquivos
`*.transport.json` encontrados. Se nenhum arquivo existir, a busca vazia
mantém o custo em zero, sem erro. Isso não satisfaz a condição de segurança
solicitada na S9c: ausência de journals deve interromper a execução, não ser
interpretada como custo nulo. O comportamento não foi alterado porque o
arquivo tem SHA-256 fixado, tanto em `code_sha256_lf` do metadata quanto em
`amendments[].execution_code_sha256` do protocolo:
`b884df01c7678b8d7b79619dd86d5f145ce651a68ad18759a06c2a35dbd74b7e`.

Não se autoriza nova execução por este documento. Antes de reutilizar o
executor para novas chamadas, é necessária uma versão nova e uma autorização
que fixe seu hash. Essa versão deverá validar que há ao menos um journal
antes da soma e interromper o processo filho se a observabilidade do custo
estiver ausente. O teste S9c caracteriza a limitação do executor congelado;
não a transforma em comportamento aceito para execuções futuras.

## Testes e EOL

Os testes acrescentados exercitam o gate de referência da subamostragem sem
treinar modelos: referência ausente, identidade divergente e reprodução falsa
impedem o carregamento dos dados; identidade correspondente e reprodução
confirmada permitem chegar ao carregamento. O script histórico permanece
inalterado. O teste privado da concordância exige agora todas as respostas
dos diretórios `RUN_DIRS`, resolvidas pelo mesmo mecanismo do coletor.

`scripts/auditar_eol.py` descobre referências conservadoramente pelos testes
de bytes/hash, por fingerprints versionados e pela família de tabelas
editoriais. Para cada caminho, registra as fontes da evidência, os bytes LF
ou CRLF e os atributos efetivos de checkout. Não reescreve referências.
As exceções CRLF históricas prevalecem sobre as regras LF de família.

Há uma exceção herdada no worktree de origem: `gzcmd/prereg_arranjo_corte.yaml`
contém 265 finais CRLF e 122 LF, mas o blob versionado é integralmente LF.
Nenhuma regra `eol` reproduz essa mistura. Preservou-se o arquivo congelado
sem reescrita; a regra LF reproduz o blob do Git nos clones novos. A auditoria
registra explicitamente a mistura e só a admite se a normalização dos bytes
do worktree for idêntica ao blob LF. As suítes nos dois clones verificam os
contratos históricos no checkout canônico, sem alterar hashes esperados.
