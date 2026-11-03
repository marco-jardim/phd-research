# Adoção de REFREC e fechamento observado

Por decisão do usuário, `evaluation.split_by` passou a `refrec`. O perfil
histórico continua explícito como `fit_platt`/`v3`, com as mesmas guardas e
custos. O cache por linha foi preservado em `data/pcal_ml_oof_row.csv`; o cache
REFREC previamente verificado foi promovido para `data/pcal_ml_oof.csv`,
incluindo seu fingerprint. Arquivos `_row` ficam locais e ignorados pelo git.

`scripts/preparar_refrec_final.py` preserva os artefatos por linha, tabelas e
figuras com manifesto SHA256 antes de substituí-los. A versão anterior das
taxas fica em `gzcmd/error_rates_by_band_v4_lote_antigo.yaml`. A extração usa
as taxas anteriores e exporta somente pares LLM ainda sem resposta em qualquer
coorte Haiku v4 existente, sem rótulos ou identificadores dos grupos.
Os lotes iniciais contêm 155 e 17 pares. Na vigilância, 52 pertencem a bandas
sem respostas na fonte das taxas anteriores; os pares já respondidos no
suplemento por linha não são repetidos.

Os lotes autorizados são executados com o runner existente, prompt v4,
modelo `claude-haiku-5-5`, quatro workers por modo e processos separados.
Checkpoint e journal mantêm a validação de identidade, os retries de escrita
no Windows e os contadores por par. A chave é lida do `.env` e transmitida por
ambiente, nunca por argumento nem em log. O chamador espera os processos antes
de considerar os lotes completos.

Depois das respostas iniciais:

```powershell
python scripts/fechar_refrec_final.py --reestimate
```

A reestimação inclui as respostas históricas, o suplemento por linha e o
suplemento REFREC inicial. As coortes têm de ser disjuntas por modo e manter
prompt e modelo idênticos. A tabela de reruns mostra os suplementos em linhas
separadas, sem tratá-los como novos braços pareados do experimento histórico.
As fontes e hashes utilizados nas taxas são explícitos.

Trocar taxas pode alterar a seleção LLM. O comando exporta esse conjunto
adicional, sem resposta, em arquivos de fechamento separados e registra o hash
das taxas. Após responder esse conjunto, as taxas permanecem congeladas:
não se repete indefinidamente a sequência adaptativa de reestimar e selecionar.
As respostas adicionais de fechamento entram no replay e no custo observado,
mas não na reestimação já congelada.

```powershell
python scripts/fechar_refrec_final.py --finalize
python -m pytest tests -q
```

O fechamento recusa hash de taxas divergente e qualquer pendência na cascata
final. Regera funis, perdas, varreduras de custo/NP, três figuras, tabelas com
Wilson, reconciliação e leave-out. A exclusão usa todos os pares selecionados
que constam nas fontes efetivas das taxas, não uma contagem fixa de reutilizados.
O leave-out alternativo pode selecionar pares sem respostas e continua
identificando expectativas e limites, sem classificações imputadas.

A comparação por linha/REFREC preserva o estágio histórico e identifica que
a diferença final combina split e reestimação. A sensibilidade automática
mantém vínculos encaminhados ao humano no denominador. A projeção com humano
usa erro hipotético de 5%, inclusive quando se apresenta esse cenário para
v3; não altera o replay histórico nem afirma observar resultados humanos.

Taxas LLM reestimadas e política avaliada compartilham observações adaptativamente
selecionadas. O OOF por grupo do classificador não transforma esse replay em
validação independente da confiabilidade do LLM. Custos em US$ são estimativas
tarifárias aplicadas aos tokens observados, não faturas; o gasto do rerun e o
custo alocado aos pares selecionados na cascata são reportados separadamente.
