# S8c: esforço humano e falsos positivos por região

As contagens desta análise foram produzidas depois da avaliação da S3 e não constituem desfechos confirmatórios registrados. Os arquivos históricos com hashes fixados foram preservados; os desvios anteriores estão descritos em `M_arranjo_corte_desvios.md`.

## Esforço humano hipotético

`scripts/esforco_arranjo.py` produz `data/esforco_arranjo.csv` a partir de `data/comparador_mesma_carga.csv` e do campo explícito `minutos_por_par` de `data/esforco_arranjo_parametros.json`. Adotam-se 20 minutos por potencial par, hipótese presente nos comentários da configuração de custos e não uma duração medida no serviço. O cenário de referência, 2.156 revisões, pressupõe ordem decrescente de SCORE dentro de cada passo; não há fonte que documente essa ordem histórica.

K1 corresponde ao orçamento humano de P. K2 soma os potenciais pares encaminhados ao humano e ao LLM, sem converter chamadas ao LLM em horas de trabalho humano. P mantém suas decisões nas duas comparações; O3 utiliza o orçamento inteiro para revisão humana. Assim, na vigilância, P exige 96 revisões humanas nas duas cargas, enquanto O3 exige 96 em K1 e 440 em K2. Na confirmação, esses valores são 17 para P e 17 ou 58 para O3.

| Modo | Braço e carga | Revisões humanas | Horas hipotéticas | PAR=2 recuperados com humano perfeito | FP | Revisões poupadas | Horas poupadas |
|---|---|---:|---:|---:|---:|---:|---:|
| Vigilância | P, K1 e K2 | 96 | 32,00 | 23 | 21 | 2.060 | 686,67 |
| Vigilância | O3, K1 | 96 | 32,00 | 22 | 0 | 2.060 | 686,67 |
| Vigilância | O3, K2 | 440 | 146,67 | 26 | 0 | 1.716 | 572,00 |
| Confirmação | P, K1 e K2 | 17 | 5,67 | 9 | 4 | 2.139 | 713,00 |
| Confirmação | O3, K1 | 17 | 5,67 | 6 | 0 | 2.139 | 713,00 |
| Confirmação | O3, K2 | 58 | 19,33 | 20 | 0 | 2.098 | 699,33 |

Os vínculos recuperados são condicionais ao revisor humano perfeito: para P somam aceitação automática e encaminhamento humano; para O3 correspondem aos PAR=2 presentes nas primeiras posições da ordenação. Os FP de P são aceitações automáticas observadas contra o padrão-ouro, enquanto zero FP de O3 decorre da hipótese de revisão perfeita. A coluna `par2_recuperados_erro05` apresenta separadamente a projeção com 5% de erro humano, sem atribuir esse erro às aceitações automáticas de P.

`horas_por_vinculo_perdido` divide as horas de revisão pelo número de PAR=2 não recuperados. A coluna `horas_poupadas_por_vinculo_perdido` explicita a troca de esforço poupado por vínculo não recuperado frente aos 36 do padrão-ouro. As razões por FP seguem a mesma distinção; quando não há FP, a razão é indefinida e permanece vazia no CSV. Essas horas expressam ordem de grandeza, não medida de eficiência do serviço ou recomendação validada em dados novos.

## Falsos positivos por região de SCORE

`scripts/fp_arranjo_por_regiao.py` lê somente as colunas necessárias do extrato como texto, converte PAR, PASSO e SCORE com `to_numeric` e alinha as decisões privadas de P pelas chaves internas. Nenhuma chave ou informação pessoal é gravada. `data/fp_arranjo_por_regiao.csv` contém apenas contagens por modo, limiar, passo e região, com as linhas TOTAL para uso no texto e nos slides.

O limiar global é −5,64846544302. Os limiares por passo são −5,64846544302 nos passos 6 e 11 e −4,35228276097 no passo 9, conforme `data/carga_abaixo_corte_metadata.json`. A região superior inclui os empates com o limiar.

| Modo | Limiar | FP com SCORE maior ou igual ao limiar | FP abaixo do limiar |
|---|---|---:|---:|
| Vigilância | Global | 17 | 4 |
| Vigilância | Por passo | 11 | 10 |
| Confirmação | Global | 4 | 0 |
| Confirmação | Por passo | 4 | 0 |

A posição em relação ao limiar não comprova que o serviço tenha examinado esses potenciais pares. A contagem informa a distribuição dos FP condicionais ao padrão-ouro, sem resolver o viés de verificação ou a ausência de documentação da parada histórica.

## Evidência de implementação

- M1 e M2: desvios documentados, sem edição dos testes fixados; teste complementar de hash exato do comparador e das taxas em `tests/test_s8c.py`.
- M3: `scripts/tabelas_s8c_editorial.py` regenera quatro tabelas em `tables/editorial/`; as substituições editoriais exigem alvo único. A legenda histórica da cascata foi restaurada e é reproduzida pelo gerador. As tabelas antigas de comparação de equidade não foram restauradas porque seu teste está congelado no manifesto.
- m3: respostas e journals privados da robustez e PDFs de verificação são ignorados, sem ignorar `tese.pdf` ou o script de preservação versionado. Nenhum `identity.json` da robustez estava versionado.
- m5: o teste congelado da robustez permanece intacto. O teste novo observa os registros efetivamente recebidos pelo estimador no caminho de produção para duas réplicas numéricas, em vez de reaplicar o estimador às mesmas entradas. O contrato não congelado da S6a exige a qualificação completa do cenário nos dois idiomas.
- m8: dependência histórica dos comentários documentada; o novo artefato de esforço usa campo explícito, sem modificar a configuração fixada.

Cada um dos três scripts novos tem teste de regeneração em `tests/test_s8c.py`. A regeneração dos FP recebe `skipif` quando faltam extrato, políticas ou respostas privadas; esforço e tabelas são regenerados apenas dos artefatos versionados.
