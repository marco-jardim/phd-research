# M4: prompt v4 para a revisão sem saídas do motor

O executor `scripts/run_exp2_sem_model_outputs.py` aceita `--prompt-version v3|v4`.
O padrão continua sendo v3. O arquivo v3, o parser e o protocolo Agent-A + Agent-B
com árbitro em caso de divergência não foram alterados.

## Política e compatibilidade

O v4 decide pelos campos disponíveis, com os limiares por feature do v3, sem
probabilidades, decisões, bandas ou explicações provenientes do motor. A nota
do comparador é somente contexto. Nome forte e nascimento forte ou compatível,
sem contradição forte, sustentam MATCH; contradição forte não explicada sustenta
NONMATCH; sinais mistos ou evidência insuficiente justificam UNSURE.

O schema mantém os 14 códigos aceitos pelo parser. `MODEL_HIGH_P` e
`MODEL_AMBIGUOUS` permanecem no enum por compatibilidade, mas o v4 proíbe sua
atribuição. O cabeçalho usa “dossiê sem esse grupo” para não conter a string
`model_outputs`, que a verificação de ausência também proíbe no cabeçalho.

O carregador em `gzcmd/llm_review.py` extrai o trecho system até o primeiro
separador `---`. Por isso, a política do v4 fica dentro desse trecho, antes
do separador. O v3 é preservado integralmente, inclusive sua extração histórica.

Não existe instrução de modo no prompt v3 nem injeção do modo nas mensagens
do revisor nesses executores. `--modo` continua controlando a reconstrução e a
coorte histórica, sem introduzir instrução nova de vigilância/confirmação no v4.
Sexo, UF e datas de evento são considerados somente se existirem no dossiê;
esta mudança não acrescenta campos ao serializador.

## Seleção, hash e retomada

Ambos os provedores usam o módulo compartilhado `gzcmd.llm_review`. O executor
seleciona o arquivo antes de criar os revisores, incluindo o árbitro.
`metadata.prompt_version` registra a versão. Para v4, `prompt_sha256` é o SHA256
dos bytes do arquivo completo; `metadata.prompt_file_sha256` também o registra.
Para v3, `prompt_sha256` preserva o hash histórico da concatenação dos trechos
system e user, permitindo retomar checkpoints anteriores sem `prompt_version`.
Checkpoints de versões distintas não podem ser misturados.

`--saida-dir` define a pasta do JSON, journal e auditoria dos campos. O nome do
JSON é `exp2_llm_sync_<modo>.json`. É alternativa a `--saida`, não combinação.
Para não reutilizar resultados anteriores, use pastas novas com sufixo `_v4`.
Os comandos de rerun não precisam de `--summarize-only`; os relatórios históricos
desse modo de sumarização ainda descrevem a ablação com prompt v3.

## Dossiê e testes sem inferência

`DossierView` remove integralmente `model_outputs` e preserva `features`, inclusive
`features.nota_final`, os subescores e os indicadores MACD. `audit_view` rejeita
chaves proibidas recursivamente. O executor já grava somente nomes de campos em
`dossier_field_audit_<modo>.json`, sem valores dos campos. Testes serializam um
dossiê sintético e verificam a ausência dessas chaves e a permanência da nota.

`tests/test_exp2_prompt_v4.py` verifica ausência das referências proibidas,
política efetivamente carregada, schema, padrão v3, hash histórico, seleção v4,
mensagens dos dois agentes e árbitro, hash persistido e rejeição de retomada
com versão diferente, para ambos os modos e provedores. As chamadas de API
são bloqueadas nos testes.
