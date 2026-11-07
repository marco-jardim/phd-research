# Sensibilidade das conclusões ao erro da referência operacional PAR

Análise quantitativa de viés, exploratória post hoc, CPU e sem LLM. Não estima a frequência real de erro nem corrige a referência como se fosse verdade conhecida.

## Convenções

Frações dos FP abaixo/acima do limiar **por passo** variam independentemente em 0/25/50/75/100%; remoção uniforme entre os 36 PAR=2: 0/5/10/20%. Todos esses FP estão abaixo do corte operacional ORL; não confundir os dois limiares. Confirmação: zero FP abaixo do limiar por passo, quatro acima. Aceitos significa VP automáticos; sensibilidade automática não conta encaminhamentos. Também se publica sensibilidade com revisor perfeito.

Ordenações O1/O3 e RRF(60) F4 permanecem fixas; não há retreino sob os novos rótulos. 1.000 sorteios por cenário, seed 20261107. Número de trocas: n×fração com arredondamento estocástico não enviesado; subconjuntos uniformes sem reposição. Mediana e P2,5–P97,5 são dispersão de cenários, não IC inferencial. Diferenças são pareadas por sorteio. Metas 25/64/100%: inteiro mais próximo (metades para cima), reproduzindo 9/23/36 na referência (23/36=63,89%, rotulado 64% no estudo); denominador recalculado em cada sorteio. Para evitar que arredondamento seja confundido com atingir literalmente a fração, as métricas adicionais `revisoes_minimo_25/64/100` usam teto(n×fração): na referência exigem 9/24/36. As viradas C3/C4 mantêm a convenção nominal do estudo; 9 e 36 são metas de referência, não contagens absolutas fixas após alteração do total de vínculos.

## Viradas condicionais (outros eixos em zero)

| Modo | Conclusão operacional | Eixo | Válida no início? | Primeira mudança na grade |
|---|---|---|---|---|
| vigilancia | C1_VPP_menor_50 | fp_abaixo | sim | 100% |
| vigilancia | C1_VPP_menor_50 | fp_acima | sim | 100% |
| vigilancia | C1_VPP_menor_50 | par2_nao_vinculo | sim | não observada |
| vigilancia | C1_VPP_menor_50 | fp_uniforme | sim | 50% |
| vigilancia | C2_perdidos_13_27 | fp_abaixo | sim | não observada |
| vigilancia | C2_perdidos_13_27 | fp_acima | sim | não observada |
| vigilancia | C2_perdidos_13_27 | par2_nao_vinculo | sim | 5% |
| vigilancia | C2_perdidos_13_27 | fp_uniforme | sim | não observada |
| vigilancia | C3_O3_margem25_meta64 | fp_abaixo | sim | não observada |
| vigilancia | C3_O3_margem25_meta64 | fp_acima | sim | não observada |
| vigilancia | C3_O3_margem25_meta64 | par2_nao_vinculo | sim | não observada |
| vigilancia | C3_O3_margem25_meta64 | fp_uniforme | sim | não observada |
| vigilancia | C4_F4_O1_meta25 | fp_abaixo | sim | não observada |
| vigilancia | C4_F4_O1_meta25 | fp_acima | sim | não observada |
| vigilancia | C4_F4_O1_meta25 | par2_nao_vinculo | sim | não observada |
| vigilancia | C4_F4_O1_meta25 | fp_uniforme | sim | não observada |
| vigilancia | C4_F4_O1_meta100 | fp_abaixo | sim | não observada |
| vigilancia | C4_F4_O1_meta100 | fp_acima | sim | não observada |
| vigilancia | C4_F4_O1_meta100 | par2_nao_vinculo | sim | não observada |
| vigilancia | C4_F4_O1_meta100 | fp_uniforme | sim | não observada |
| confirmacao | C1_VPP_menor_50 | fp_abaixo | não | não observada |
| confirmacao | C1_VPP_menor_50 | fp_acima | não | não observada |
| confirmacao | C1_VPP_menor_50 | par2_nao_vinculo | não | não observada |
| confirmacao | C1_VPP_menor_50 | fp_uniforme | não | não observada |
| confirmacao | C2_perdidos_13_27 | fp_abaixo | sim | não observada |
| confirmacao | C2_perdidos_13_27 | fp_acima | sim | não observada |
| confirmacao | C2_perdidos_13_27 | par2_nao_vinculo | sim | 5% |
| confirmacao | C2_perdidos_13_27 | fp_uniforme | sim | não observada |
| confirmacao | C3_O3_margem25_meta64 | fp_abaixo | sim | não observada |
| confirmacao | C3_O3_margem25_meta64 | fp_acima | sim | não observada |
| confirmacao | C3_O3_margem25_meta64 | par2_nao_vinculo | sim | não observada |
| confirmacao | C3_O3_margem25_meta64 | fp_uniforme | sim | não observada |
| confirmacao | C4_F4_O1_meta25 | fp_abaixo | sim | não observada |
| confirmacao | C4_F4_O1_meta25 | fp_acima | sim | não observada |
| confirmacao | C4_F4_O1_meta25 | par2_nao_vinculo | sim | não observada |
| confirmacao | C4_F4_O1_meta25 | fp_uniforme | sim | não observada |
| confirmacao | C4_F4_O1_meta100 | fp_abaixo | sim | não observada |
| confirmacao | C4_F4_O1_meta100 | fp_acima | sim | não observada |
| confirmacao | C4_F4_O1_meta100 | par2_nao_vinculo | sim | não observada |
| confirmacao | C4_F4_O1_meta100 | fp_uniforme | sim | não observada |

Todas as superfícies condicionais, inclusive combinações, estão em vies_referencia_viradas.csv. Virada é mudança do indicador baseado na mediana; o intervalo completo deve ser consultado no CSV de métricas. Não se interpola entre pontos.

## Leitura curta

- vigilancia: VPP inicial 16.0%; perdidos iniciais 13. Reclassificar FP aumenta VP, não recupera vínculos perdidos; retirar PAR=2 reduz os perdidos esperados proporcionalmente (hipótese de erro uniforme).
- confirmacao: VPP inicial 66.7%; perdidos iniciais 27. Reclassificar FP aumenta VP, não recupera vínculos perdidos; retirar PAR=2 reduz os perdidos esperados proporcionalmente (hipótese de erro uniforme).
- C1 é apenas o diagnóstico VPP<50%, não prova de segurança de aceitação autônoma. Na confirmação ele já é falso no cenário inicial (8/12); não se generaliza a recomendação da vigilância com esse limiar arbitrário.
- C2 literal (13/27) muda com qualquer fração positiva de PAR=2 removida; na grade a primeira é 5%. Isso não elimina perdas: a 20% restam 10,4/21,6 esperadas.
- C3 exige redução de pelo menos 25% em O3 frente a O1 na meta nominal de 64%; C4 exige diferença F4−O1 estritamente negativa, nas metas móveis correspondentes a 9 e 36 na referência. Não são testes de significância.

## Robustez nas combinações da vigilância

| Indicador pareado (negativo favorece a conclusão) | Maior mediana entre os 100 cenários | Maior P97,5 |
|---|---:|---:|
| O3-O1_margem25_64 | -229.75 | 5.87 |
| F4-O1_revisoes_25 | -39.00 | -13.00 |
| F4-O1_revisoes_100 | -1134.00 | -1134.00 |

C3 mantém o sinal pela mediana nos 100/100 cenários, mas P97,5 cruza zero em 1/100: não se deve traduzir robustez da mediana em garantia em todos os sorteios.

C1, cálculo contínuo sob reclassificação uniforme dos 21 FP: VPP=50% em 40,48% dos FP se nenhum PAR=2 for removido (primeiro ponto da grade: 50%). Se apenas os dez FP abaixo do limiar por passo mudam, são necessários 85%; se apenas os onze acima mudam, 77,27% (ambos: primeiro ponto 100%). Com remoção de 20% dos PAR=2, o limiar uniforme é 44,29%. Essas são soluções algébricas do VPP esperado, não estimativas da prevalência de erro.

## Identificadores fortes (somente agregados)

Máximo empírico no extrato completo de cada um dos sete subescores: qtd/primeiro/último fragmento iguais para NOME e NOMEMAE, e DTNASC dt iguais. Concordância forte requer os sete máximos simultaneamente, sem imputação. Máximos registrados no metadado. Não equivale à igualdade dos nomes originais nem à confirmação de vínculo. Ausência de conflito significa que nem COMPREC nem REFREC constam dos 211 PAR=1 acima do corte (mesmo critério de fp_cardinalidade). Não avalia conflitos entre candidatos abaixo do corte.

| Grupo | n | Máximos simultâneos | Sem conflito | Ambos |
|---|---:|---:|---:|---:|
| vigilancia/FP_P | 21 | 0 | 20 | 0 |
| confirmacao/FP_P | 4 | 0 | 4 | 0 |
| referencia/PAR2 | 36 | 0 | 30 | 0 |
| referencia/amostra_PAR0 | 1000 | 0 | 915 | 0 |

Nenhum dos FP, PAR=2 ou dos 1.000 PAR=0 amostrados atinge os sete máximos simultâneos (todos os máximos são 1). Esse critério estrito não sustenta que estes FP sejam vínculos fortes ignorados; tampouco comprova que sejam não vínculos. Ausência de conflito de cardinalidade não confirma vínculo. Viés de verificação dos PAR=2 pode favorecer O1; a análise não elimina esse viés nem modela erros nos 211 PAR=1.

## Reprodução

`python scripts/vies_referencia.py --data-root <dados> --policy-dir <politicas>`

Testes portáveis em tests/test_vies_referencia.py; entradas privadas opcionais via tests/local_data.py e PHD_RESEARCH_POLICY_DIR. Metadado registra grade antes da leitura, hashes, versões e regras. Nenhum identificador ou valor por par é publicado.
