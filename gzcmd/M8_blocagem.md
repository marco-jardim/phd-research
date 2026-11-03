# M8 — leitura da blocagem do OpenRecLink a partir de seus produtos

## 1. Pré-condição: resultado antes de qualquer estimativa

**61,696 linhas; 61,684 pares distintos; 12 pares presentes em mais de um PASSO.**
**247 linhas de vínculo; 247 vínculos distintos; 0 vínculos recapturados.**
O extrato não é perfeitamente desduplicado por par: há 12 repetições entre passos,
todas de não vínculos. Entretanto, os vínculos não têm recapturas. Além disso,
capitulo5.tex:40,48 documenta exclusão progressiva dos registros Sinan já pareados.
Assim, **captura-recaptura é inviável**, não apenas imprecisa. Não se deve substituir
os zeros por uma pseudocontagem e apresentar um total artificial.

## 2. Motivação, método e privacidade

A motivação é a leitura de um sistema a partir dos produtos que deixa observar,
inspirada em Takehi et al. (2026), *Weight-Space Programming*, post de blog indicado
no enunciado. Trata-se de analogia metodológica, não replicação nem validação desse
trabalho; a parte de edição de pesos não se aplica ao ORL. Não há alteração da
configuração, do software de origem, dos dados nominais ou de pesos de modelos.

Leitura `dtype=str`, delimitador ponto-e-vírgula e `to_numeric` explícito, convertendo
decimal vírgula. Vínculo: PAR em {1,2}. Identidade de par: (COMPREC, REFREC);
`pair_id = sha256("COMPREC|REFREC|PASSO")[:16]` é calculado apenas em memória, nunca
exportado. Matrizes usam conjuntos, não contagens repetidas. Rendimento distingue
linhas e pares distintos; quantis são por linha, frações positivas por par distinto.
Campos vêm do cabeçalho e de `gzcmd/loader.py` (a função de engenharia disponível;
`gzcmd/features.py` não existe neste checkout). Usa-se allowlist dos 29 subescores,
mais igualdade local de sexo (somente códigos válidos 1/2/M/F). Não se exportam
sexo individual, nomes, datas, municípios, identificadores, pair_id nem linhas de
pares; apenas agregados, descrições, pseudocódigo e proveniência. Ausência não
equivale a concordância: fica fora do numerador, com denominador total do passo.
Indicadores de composição, abreviação e inversão não são tratados como igualdade.
Limiar alto operacional >=0,8 não normaliza os tetos distintos dos subescores.

**Duas escalas diferentes:** aqui as faixas solicitadas [5,9) e >=9 usam **nota final
do comparador**. O capítulo 5:21 e a tabela tab_score_bands usam **SCORE do ORL**
para sua área cinza. Portanto não se espera reproduzir os 151/247 vínculos cinzentos
do capítulo com a nota final. Esta diferença é registrada, não silenciosamente
resolvida trocando a variável. Quantis e frações deste relatório são da nota final.

## 3. Rendimento por passo

| passo | linhas | pares_distintos | linhas_vinculo | vinculos | vinculos_so_neste_passo | vinculos_tambem_outro_passo | taxa_vinculo_pares | fracao_vinculos_cinza_nota | fracao_vinculos_nota_ge9 | nota_q0 | nota_q0.25 | nota_q0.5 | nota_q0.75 | nota_q0.9 | nota_q0.95 | nota_q1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2649 | 2649 | 80 | 80 | 80 | 0 | 0.030200 | 0.225000 | 0.775000 | 2.070000 | 5.800000 | 6.480000 | 6.940000 | 7.340000 | 7.730000 | 13.710000 |
| 9 | 1449 | 1449 | 127 | 127 | 127 | 0 | 0.087647 | 0.826772 | 0.165354 | 0.890000 | 3.870000 | 4.640000 | 5.730000 | 6.562000 | 7.130000 | 10.780000 |
| 11 | 57598 | 57598 | 40 | 40 | 40 | 0 | 0.000694 | 0.700000 | 0.250000 | 0.530000 | 3.800000 | 4.420000 | 5.300000 | 5.920000 | 6.310000 | 11.440000 |

São rendimentos **do que foi exportado**, condicionais à seleção dos passos e à
remoção sequencial. 'Só neste passo' significa exclusivo **neste extrato**, não que
nenhuma outra chave recuperaria o vínculo. Tardios significa PASSO > 6 (9 e 11),
não uma sequência de captura independente. A sequência numérica é a descrita no
capítulo; a execução completa dos 30 passos não foi observada.

## 4. Sobreposição

Pares candidatos (diagonal = número de pares distintos):

| passo_a | 6 | 9 | 11 |
| --- | --- | --- | --- |
| 6 | 2649 | 0 | 9 |
| 9 | 0 | 1449 | 3 |
| 11 | 9 | 3 | 57598 |

Vínculos (diagonal = número de vínculos distintos):

| passo_a | 6 | 9 | 11 |
| --- | --- | --- | --- |
| 6 | 80 | 0 | 0 |
| 9 | 0 | 127 | 0 |
| 11 | 0 | 0 | 40 |

Entre 6 e 11 há 9 pares comuns; entre 9 e 11, 3; entre 6 e 9, zero. Todos os
vínculos exportados são exclusivos. Isso **não demonstra completude da blocagem**.

## 5. Captura-recaptura: não estimar neste extrato

`blocagem_captura_recaptura.csv` registra status inviável; total, não capturados e
IC95% ficam ausentes, não zero. Os 247 observados coincidem com capitulo5.tex:43–48
(80+127+40) e capitulo6.tex:9. Essas seções não oferecem número independente de
vínculos 'esperados' para comparação: 247 é o total observado, não a população
verdadeira. Não inferimos sensibilidade absoluta ou número de vínculos perdidos.

Para futuros históricos completos, o script oferece Chapman para duas listas:
N̂=((nA+1)(nB+1)/(m+1))-1;
Var(N̂)=(nA+1)(nB+1)(nA-m)(nB-m)/((m+1)²(m+2)); IC normal aproximado 95%,
truncado inferiormente à união observada. Sem sobreposição positiva a função
recusa a estimativa. Para 3+ listas há Poisson GLM via statsmodels sobre todos os
2^k−1 perfis, incluindo frequências zero: intercepto + efeitos principais +
interações duplas (interações de primeira ordem entre listas); a célula ausente é
exp(intercepto), com IC95% delta na escala log. Com 3 listas o modelo é saturado
nos 7 perfis observados: não há graus de liberdade residuais para testar ajuste.
O teste sintético confirma a fórmula, não sua adequação a este extrato.

Hipóteses: população fechada, captura homogênea e classificação correta; para
Chapman, independência entre listas; para GLM, ausência de interações de ordem
superior às incluídas e identificabilidade. Dependência positiva de chaves tende
a subestimar o total sob independência (leitura de limite inferior, não garantia
matemática universal). Exclusão progressiva induz dependência estrutural/negativa
e truncamento por desenho, que **não** se corrige chamando a estimativa de limite
inferior. A hipótese de listas comparáveis foi violada; não há IC válido aqui.

## 6. Perfil dos vínculos tardios e de baixa concordância nominal

Fração abaixo de 0,8 e fração zero são discordância/baixa similaridade **proxy**,
não prova de erro nem separação entre ausência e divergência textual. Nome baixo
foi pré-definido como `NOME qtd frag iguais < 0,5` (1 vínculo; grupo muito pequeno,
sem inferência populacional). Os tardios exclusivos somam 167 vínculos.

| grupo | n_vinculos | campo | fracao_abaixo_0.8 | fracao_zero | fracao_ausente |
| --- | --- | --- | --- | --- | --- |
| todos_vinculos | 247 | NOME qtd frag iguais | 0.141700 | 0.000000 | 0.000000 |
| todos_vinculos | 247 | NOMEMAE qtd frag iguais | 0.688259 | 0.518219 | 0.000000 |
| todos_vinculos | 247 | DTNASC dt iguais | 0.538462 | 0.538462 | 0.000000 |
| todos_vinculos | 247 | CODMUNRES local igual | 0.056680 | 0.056680 | 0.000000 |
| todos_vinculos | 247 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |
| so_passos_tardios | 167 | NOME qtd frag iguais | 0.155689 | 0.000000 | 0.000000 |
| so_passos_tardios | 167 | NOMEMAE qtd frag iguais | 0.958084 | 0.766467 | 0.000000 |
| so_passos_tardios | 167 | DTNASC dt iguais | 0.317365 | 0.317365 | 0.000000 |
| so_passos_tardios | 167 | CODMUNRES local igual | 0.059880 | 0.059880 | 0.000000 |
| so_passos_tardios | 167 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |
| nome_baixo_lt_0.5 | 1 | NOME qtd frag iguais | 1.000000 | 0.000000 | 0.000000 |
| nome_baixo_lt_0.5 | 1 | NOMEMAE qtd frag iguais | 1.000000 | 1.000000 | 0.000000 |
| nome_baixo_lt_0.5 | 1 | DTNASC dt iguais | 0.000000 | 0.000000 | 0.000000 |
| nome_baixo_lt_0.5 | 1 | CODMUNRES local igual | 0.000000 | 0.000000 | 0.000000 |
| nome_baixo_lt_0.5 | 1 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |
| passo_11 | 40 | NOME qtd frag iguais | 0.325000 | 0.000000 | 0.000000 |
| passo_11 | 40 | NOMEMAE qtd frag iguais | 0.825000 | 0.475000 | 0.000000 |
| passo_11 | 40 | DTNASC dt iguais | 0.925000 | 0.925000 | 0.000000 |
| passo_11 | 40 | CODMUNRES local igual | 0.000000 | 0.000000 | 0.000000 |
| passo_11 | 40 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |
| passo_6 | 80 | NOME qtd frag iguais | 0.112500 | 0.000000 | 0.000000 |
| passo_6 | 80 | NOMEMAE qtd frag iguais | 0.125000 | 0.000000 | 0.000000 |
| passo_6 | 80 | DTNASC dt iguais | 1.000000 | 1.000000 | 0.000000 |
| passo_6 | 80 | CODMUNRES local igual | 0.050000 | 0.050000 | 0.000000 |
| passo_6 | 80 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |
| passo_9 | 127 | NOME qtd frag iguais | 0.102362 | 0.000000 | 0.000000 |
| passo_9 | 127 | NOMEMAE qtd frag iguais | 1.000000 | 0.858268 | 0.000000 |
| passo_9 | 127 | DTNASC dt iguais | 0.125984 | 0.125984 | 0.000000 |
| passo_9 | 127 | CODMUNRES local igual | 0.078740 | 0.078740 | 0.000000 |
| passo_9 | 127 | SEXO igualdade | 0.000000 | 0.000000 | 0.000000 |

No passo 11, 37/40 não têm concordância exata de data, 33/40 têm concordância da
mãe <0,8 e 13/40 têm concordância do nome <0,8. Nos tardios combinados, 128/167 têm
zero em fragmentos iguais da mãe, 53/167 não têm data exata e 26/167 têm nome <0,8.
No passo 6, data exata é zero em todos os 80 vínculos: investigar o comparador e a
exportação antes de interpretar isso como qualidade intrínseca das datas.

## 7. Chaves inferidas — a validar

A primeira chamada recebe apenas a tabela agregada de todos os candidatos por
passo, dicionário e descrição geral do ORL extraída do capítulo 5:21,40,73. As
chaves enumeradas nas linhas 43–45 foram **retidas do prompt**, para não fabricar
uma 'inferência' que apenas copia a documentação. Soundex, comparação APROX e
distância de edição são documentados; Jaro-Winkler específico não foi confirmado
pelas passagens consultadas e não é atribuído como fato. Ano não é observado
diretamente nos subescores; inversão de ano não significa igualdade de ano.

Frações de concordância exata proxy em todos os pares de cada passo:

| campo | 6 | 9 | 11 |
| --- | --- | --- | --- |
| CODMUNRES local igual | 0.857682 | 0.859213 | 1.000000 |
| CODMUNRES uf igual | 1.000000 | 0.996549 | 1.000000 |
| NOME prim frag igual | 0.741034 | 0.793651 | 0.753064 |
| NOME ult frag igual | 0.967535 | 0.964803 | 0.960207 |
| NOMEMAE prim frag igual | 0.937335 | 0.071084 | 0.085402 |
| NOMEMAE ult frag igual | 0.963760 | 0.322291 | 0.335654 |
| SEXO igualdade | 0.884107 | 1.000000 | 1.000000 |

| passo | inferência LLM | status |
| --- | --- | --- |
| 6 | Exato textual sobre UF de residência, com chave de sexo improvável. Não há evidência de chave fonética nem de ano. | a validar |
| 9 | Exato textual sobre sexo, com UF de residência como pista possível. Não há evidência de chave fonética nem de ano. | a validar |
| 11 | Exato textual sobre UF de residência, município de residência e sexo. Chave ainda não confirmada. Os agregados não mostram nenhuma chave fonética (Soundex) nem chave de ano de nascimento. | a validar |

**Confronto posterior com a documentação, não configuração validada:**

| Passo | Chave descrita no capítulo 5:43–45 | Conferência na configuração ORL |
| --- | --- | --- |
| 6 | Soundex primeiro+último bloco do nome e primeiro+último bloco da mãe | a validar |
| 9 | Soundex primeiro+último bloco do nome + ano de nascimento + sexo | a validar |
| 11 | Soundex primeiro+último bloco do nome + sexo + município | a validar; linha 70 menciona só primeiro fragmento |

O LLM não recuperou componentes fonéticos nem ano. UF≈100% é principalmente
compatível com homogeneidade geográfica, e não confirma que UF seja chave. A
inferência UF+município inclui redundância. **Discordância textual não exclui
igualdade fonética**, ao contrário do que alguns trechos brutos do LLM sugerem.
Sua frase de que não há número de candidatos por passo também é incorreta: `n`
estava no agregado. Preservam-se respostas brutas como material auditável, não
como configuração reconstruída. As inferências completas e pseudocódigo estão
em `chaves_inferidas.json`, todas com status 'a validar'.

## 8. Novas chaves e avaliação local

Segunda tarefa LLM: recebe agregados do perfil, inviabilidade da recaptura,
concordâncias, contexto geral e inferências. Propostas estruturadas em DNF são
validadas por allowlist; sem `eval`, código gerado ou execução de pseudocódigo.
São **chaves alternativas a testar**, não melhorias demonstradas. Para retirar
dependência do primeiro nome, prioriza-se discussão de K1/K4 (data exata + sexo e
geografia), K2 (último fragmento + sexo + município), K3 (últimos fragmentos do
nome e da mãe + município). O custo declarado pelo LLM é hipótese, separado do
volume efetivamente medido:

| id | chave | vinculos_concordantes | vinculos_conhecidos | recall_proxy | linhas_concordantes | linhas_extrato | volume_relativo_proxy | ordem_grandeza_no_extrato |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K1 | (DTNASC_norm, CODMUNRES_local_norm, SEXO_norm) | 104 | 247 | 0.421053 | 116 | 61696 | 0.001880 | 100 |
| K2 | (NOME_ult_frag_norm, CODMUNRES_local_norm, SEXO_norm) | 228 | 247 | 0.923077 | 58451 | 61696 | 0.947403 | 10000 |
| K3 | (NOME_ult_frag_norm, NOMEMAE_ult_frag_norm, CODMUNRES_local_norm) | 83 | 247 | 0.336032 | 21855 | 61696 | 0.354237 | 10000 |
| K4 | (DTNASC_norm, CODMUNRES_uf_norm, SEXO_norm) | 114 | 247 | 0.461538 | 128 | 61696 | 0.002075 | 100 |

Recall proxy = vínculos conhecidos concordantes / 247; volume relativo = linhas
concordantes / 61.696. Os 12 pares repetidos ficam no denominador de linhas pedido;
CSV também registra pares distintos concordantes. Não se estima incremento: todos
os 247 vínculos já foram gerados por alguma chave. Os proxies de igualdade literal
não simulam uma chave Soundex; não se multiplicam estes volumes para obter pares
nas bases originais. Regras, justificativas, tipo de erro, custo sugerido, limitações
e pseudocódigo estão em `chaves_propostas.json` e no CSV de avaliação.

**Auditoria crítica das propostas:** K1: 116 linhas (ordem 10²), compatível com a
faixa sugerida 10²–10³; K2: 58.451 (10⁴), mas cobre 94,74% do extrato, custo proxy
muito alto; K3: 21.855 (10⁴), não os 10³ sugeridos; K4: 128 (10²), não os 10³–10⁴
sugeridos. Em K4 o LLM inverteu a métrica: 0,317365 é a fração **sem** data exata
nos tardios, portanto a concordância é 0,682635, não 0,317. Suas alegações de
quantos casos com erro nominal seriam recuperados não foram medidas por aquelas
interseções e não são resultados da auditoria. K2 pode já ser parte de uma chave
existente; 'nova' significa combinação candidata, não novidade comprovada diante
dos outros 27 passos não exportados. A seleção de propostas sobre o próprio
gabarito torna o recall descritivo/in-sample, não validação externa.

## 9. Chamadas e custo

3 tentativas, 2 concluídas; uma resposta de propostas truncada em 8.000
tokens foi descartada pelo wrapper original. Não houve retry automático; a nova
tentativa usou o mesmo prompt com max_tokens=16.000. As chamadas iniciais usaram
8.000; todas >=4.000, somente Claude Haiku 5.5, via cliente de
`scripts/run_exp2_anthropic.py`, sem parâmetro temperature, Fireworks ou OpenCode Go.
Ledger persistido antes de cada request, teto de 10, checkpoints reusados offline.

Usage preservado das concluídas: 54151 tokens de entrada e
15836 de saída. Estimativa adicional da truncada: 37080
de entrada (prompt com SHA256 idêntico à tentativa concluída) e
8000 de saída (teto atingido); esses valores são
**reconstruídos, não usage preservado**. Subtotal pelas chamadas com usage:
USD 0.01333310; total incluindo reconstrução:
**USD 0.02104110**, estimativa e não fatura.
Tarifa pública consultada em 02/11/2026: entrada USD 0,10/MTok e saída 0,50/MTok
para prompts até 100.000 tokens (todos abaixo do limite), sem cache/prêmios.
Fonte: https://platform.claude.com/docs/en/about-claude/pricing.
Contagens e premissas estão em `chamadas.json` e `custo.json`.

## 10. Reproduzir e próximos passos

```powershell
# Python Miniconda; pandas, numpy, scipy; statsmodels para funções/teste GLM.
python scripts/m8_auditoria_blocagem.py
# anthropic + python-dotenv e ANTHROPIC_API_KEY no .env, nunca impresso/gravado:
python scripts/m8_chaves_llm.py --input-usd-mtok 0.10 --output-usd-mtok 0.50
# Respostas e relatório existentes, sem rede:
python scripts/m8_chaves_llm.py --offline --input-usd-mtok 0.10 --output-usd-mtok 0.50
python -m pytest tests -q
```

A parte A não importa o cliente e não acessa rede. `statsmodels` não estava
instalado no Python disponível e foi instalado para verificar o caso sintético;
nenhuma dependência/configuração existente do repositório foi editada. Proveniência
dos dados/script e versões em `proveniencia_a.json`. Partes A/B têm testes de
rendimento, sobreposição, Chapman com resposta conhecida, Poisson com célula
ausente conhecida e avaliação declarativa de proxies. Verificação de testes deve
ser conferida com a saída do executor, não presumida pelo renderizador.

Exige acesso controlado às bases identificadas e reexecução do ORL:
1. conferir a configuração real de todos os 30 passos, a divergência do passo 11
   e o momento exato da exclusão/exportação;
2. executar cada chave sobre o mesmo universo original, **sem exclusão entre
   listas**, preservando histórico multilista por par, inclusive pares não aceitos;
3. auditar comparações de data (especialmente o zero estrutural do passo 6),
   equivalência das normalizações/códigos fonéticos e valores ausentes;
4. testar propostas com índices de igualdade reais, medir pares gerados, tempo,
   memória e sobreposição incremental; não executar os proxies como classificador;
5. revisar amostra independente de novos candidatos e pares fora da blocagem para
   estimar sensibilidade absoluta e vieses. Só então reconsiderar recaptura e IC.

Não se alteraram módulos `gzcmd/*.py`, YAML, capítulo, sprint3b ou scripts
proibidos; não foi feito commit. Artefatos M8 contêm somente agregados/metadados.
