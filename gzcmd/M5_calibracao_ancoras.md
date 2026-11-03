# M5b: calibração Beta por momentos

## Modelo e derivação

Referência verificada: Kull, Silva Filho e Flach (2017), AISTATS, PMLR 54:623-631, https://proceedings.mlr.press/v54/kull17a.html. O artigo descreve calibração supervisionada; seu uso por momentos agregados aqui é uma extensão experimental, não validação do procedimento pelo artigo.
Com densidades condicionais Beta(s;α1,β1) e Beta(s;α0,β0), o logaritmo da razão posterior é c+(α1−α0)ln(s)+(β1−β0)ln(1−s). Definindo a=α1−α0 e b=β0−β1, obtém-se p=expit(a ln(s)−b ln(1−s)+c). Impõe-se a,b≥0: dp/ds=p(1−p)[a/s+b/(1−s)]≥0. A identidade está contida: a=b=1,c=0. A calibração Platt da margem é o caso a=b (antes do clipping).
A motivação gaussiana usual de Platt decorre de distribuições condicionais de mesma variância para o escore. Isso é condição suficiente para log-odds linear, não pressuposto necessário de toda regressão logística. Árvores podem produzir escores saturados e distribuições assimétricas, motivando os dois coeficientes da Beta, sem garantir sua correção.
s=expit(margem OOF do XGBoost), truncada em [10^-6,1−10^-6]. As margens são as mesmas do M5, 5 folds externos com seed 42; caches são validados pela fonte, hash do CSV, cobertura de pair_id e folds. Não se usa logit da isotônica. Não houve retreinamento nem chamada LLM.

## Momentos e algoritmo

```text
conjunto     n  positivos  prevalencia
      A- 39981          3     0.000075
     A+1    18         18     1.000000
     A+2    77         75     0.974026
```

Três médias: A− com nota<5, A+1 com 9≤nota<9,5 e A+2 com nota≥9,5. O alvo é a prevalência medida exata de cada âncora. A+1 tem menos de 30 pares e taxa de fronteira; isso é reportado, não ocultado por agrupamento ou suavização do alvo. Beta+ acrescenta a média cinza estimada sem ler PAR. Nenhum rótulo cinzento individual entra no ajuste.
Objetivo: soma_j [(mean_j(p)−r_j)/SE_j]^2, SE_j=sqrt(rtilde_j(1−rtilde_j)/n_j), rtilde=(n_j*r_j+0,5)/(n_j+1). Jeffreys estabiliza SOMENTE o peso, não altera r_j. Beta usa três momentos; Beta+ usa quatro para três parâmetros, logo não há garantia de satisfazer todos simultaneamente. Otimização least_squares com Jacobiano analítico e escala adaptativa x_scale=jac, 12 inicializações fixas/aleatórias, seed 42, a,b∈[0,1000], c∈[−100,100], até 5.000 avaliações e tolerâncias 10^-11. Uma primeira tentativa sem escala adaptativa e com a≤100 atingiu 2.000 avaliações em Beta+ sem convergir; seus resultados provisórios foram substituídos pelos ajustes auditados abaixo. Reportam-se limites ativos, resíduos, valores singulares do Jacobiano e dispersão das previsões entre inicializações quase ótimas (objetivo≤melhor+10^-6*(1+melhor)). Três momentos não garantem identificabilidade quando as distribuições de s são semelhantes ou saturadas.

## Estimador da restrição cinza sem rótulos

```json
{
  "method": "FS fixed likelihood ratio, unlabeled prior EM",
  "base": 2.0,
  "prior": 0.002845823921132141,
  "iterations": 16,
  "log_likelihood_relative": 305.36489674097095,
  "prior_range_multistart": [
    0.0028458239209682747,
    0.0028458239212637987
  ],
  "total_expected": 175.5759526347573,
  "seed": 42,
  "starts": 8,
  "score_column": "SCORE",
  "unit_audit": {
    "m": [
      0.99,
      0.75,
      0.95
    ],
    "u": [
      0.01,
      0.02,
      0.05
    ],
    "observed_floor": -12.84813778786,
    "expected_floor": -12.848137787863678,
    "errors_by_base": {
      "2.0": 3.6770586575585185e-12,
      "2.718281828459045": 3.9424873047566003,
      "10.0": 8.980462925289165
    }
  },
  "grey_expected": 99.91677175101105,
  "grey_prevalence": 0.004621497305782195,
  "grey_observed_evaluation_only": 151,
  "difference_expected_minus_observed": -51.083228248988945
}
```

O extrato possui SCORE,C,15,0, peso de Fellegi e Sunter do OpenRecLink, distinto da nota final do comparador. Usou-se esse peso, não a mistura binomial. A base 2 é auditada pelo piso de discordância total: soma log2((1−m)/(1−u)), com m=(0,99;0,75;0,95), u=(0,01;0,02;0,05), parâmetros documentados na Tabela tab:parametros-orl do capítulo 5. O piso coincide com o mínimo do extrato dentro de 10^-6. Não se pressupõe que sigmoid(SCORE) seja posterior.
Define-se LR_i=2^SCORE_i e estima-se π sem rótulos por EM sobre todos os pares: q_i=expit(logit(π)+ln(2)*SCORE_i), π_novo=mean(q). Isso maximiza soma ln[(1−π)+π LR_i], omitindo a densidade de não vínculos constante em π. O alvo cinza é E=sum(q_i para nota em [5,9)). EM usa multistart e não consulta PAR. A comparação com 151 é somente avaliação posterior.
A razão FS configurada pode não ser a razão correta para o universo bloqueado/exportado, com campos dependentes e parâmetros fixos. O erro do alvo cinza pode ser elevado. O SE plug-in do quarto momento ignora incerteza/viés do modelo FS e pode conferir peso excessivo a uma estimativa errada; os resultados devem ser lidos como teste dessa hipótese, não prevalência conhecida. O fallback binomial está implementado para fontes sem SCORE: k=round(20*clip(nota/10,0,1)), EM de duas binomiais, componente de maior média presumido vínculo. Não foi usado aqui e não seria justificado como contagem real de critérios.

## Avaliação e critério pré-definido

Sucesso conjunto: ECE cinza≤1,2×ECE da isotônica e |média prevista−frequência observada|≤0,02 em p≥0,9, com pelo menos uma observação nesse intervalo. Avaliação descritiva no mesmo extrato, não teste de equivalência. ECE usa dez intervalos uniformes; a curva adicional usa decis por quantis sem partir empates. Log-loss trunca todas as probabilidades em [10^-6,1−10^-6]. Isotônica é supervisionada estritamente aninhada; Platt da nota é histórico in-sample. As calibrações globais por momentos sobre margens OOF não são totalmente aninhadas: há dependência indireta entre folds; M5 já examinou a variante aninhada. Os rótulos das âncoras são usados como prevalências, e os cinzentos somente para avaliar. O escore FS e a nota compartilham campos, portanto o quarto momento não é evidência independente.

```text
        calibracao escopo     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09  high_n  high_predicted  high_observed   high_gap  success_ece  success_high  success_joint  optimizer_success
        platt_nota  cinza 21620        151 0.00614992 0.03155659 0.00074180 0.87315686 0.21632791        0.02633672            0.00164519               0.70860927               0.00000000       0             NaN            NaN        NaN         True         False          False               True
        platt_nota  total 61696        247 0.00226109 0.01177376 0.00038723 0.96775760 0.56247462        0.19891266            0.00010941               0.44534413               0.23076923      59      0.97971816     0.96610169 0.01361646        False          True          False               True
        ancoras_ml  cinza 21620        151 0.00265412 0.01512235 0.00253659 0.99722687 0.87260298        0.99972137            0.00000000               0.13907285               0.69536424     124      0.99455744     0.84677419 0.14778324        False         False          False               True
        ancoras_ml  total 61696        247 0.00101638 0.00593387 0.00092200 0.99800635 0.91505810        0.99983413            0.00000000               0.09311741               0.78137652     213      0.99491043     0.90610329 0.08880714        False         False          False               True
      beta_ancoras  cinza 21620        151 0.00286914 0.01890190 0.00288390 0.99722687 0.87260298        0.99999905            0.00002604               0.23178808               0.70198675     128      0.99623215     0.82812500 0.16810715        False         False          False               True
      beta_ancoras  total 61696        247 0.00109458 0.00720109 0.00106368 0.99800635 0.91505810        0.99999964            0.00002604               0.15384615               0.79352227     219      0.99663108     0.89497717 0.10165391        False         False          False               True
beta_ancoras_cinza  cinza 21620        151 0.00293588 0.03026710 0.00291990 0.96599841 0.85520277        0.99996621            0.00000000               0.26490066               0.69536424     125      0.99772048     0.84000000 0.15772048        False         False          False               True
beta_ancoras_cinza  total 61696        247 0.00112577 0.01144911 0.00112085 0.97525432 0.90503544        0.99997842            0.00000000               0.17408907               0.78137652     214      0.99805321     0.90186916 0.09618405        False         False          False               True
      isotonica_ml  cinza 21620        151 0.00220576 0.00854096 0.00112513 0.99514223 0.86226195        0.89473684            0.00002248               0.12582781               0.48344371      75      0.98845695     0.97333333 0.01512361         True          True           True               True
      isotonica_ml  total 61696        247 0.00088626 0.00369240 0.00030701 0.99553380 0.91072734        0.95238095            0.00002248               0.08502024               0.55060729     138      0.98885378     0.98550725 0.00334654         True          True           True               True
```

## Sensibilidade

```text
    s_plus    s_split    s_minus  restricao_cinza   n_0  n_1  n_2        r_0        r_1        r_2     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09  high_n  high_predicted  high_observed   high_gap  success_ece  success_high  success_joint            a           b            c   objective  optimizer_success  nfev
9.00000000 9.50000000 5.00000000            False 39981   18   77 0.00007504 1.00000000 0.97402597 21620        151 0.00286914 0.01890190 0.00288390 0.99722687 0.87260298        0.99999905            0.00002604               0.23178808               0.70198675     128      0.99623215     0.82812500 0.16810715        False         False          False   0.00000000  2.95999910 -10.55602847  0.09898624               True    30
9.00000000 9.50000000 5.00000000             True 39981   18   77 0.00007504 1.00000000 0.97402597 21620        151 0.00293588 0.03026710 0.00291990 0.96599841 0.85520277        0.99996621            0.00000000               0.26490066               0.69536424     125      0.99772048     0.84000000 0.15772048        False         False          False 277.72308754  1.32395887  -0.55592731 12.33251007               True    94
9.00000000 9.50000000 4.00000000            False 18783   18   77 0.00000000 1.00000000 0.97402597 21620        151 0.00305578 0.04068879 0.00308570 0.94937703 0.75062713        1.00000000            0.00000000               0.26490066               0.72185430     133      0.99998785     0.81954887 0.18043898        False         False          False 832.25790173 19.73033819 -60.49925530  0.00000000               True    46
9.00000000 9.50000000 4.00000000             True 18783   18   77 0.00000000 1.00000000 0.97402597 21620        151 0.00293690 0.03045817 0.00294273 0.99722687 0.87260298        1.00000000            0.00000000               0.26490066               0.69536424     124      0.99919617     0.84677419 0.15242197        False         False          False   0.00000000  6.18170964 -25.72278325 11.78516682               True    52
9.50000000 9.75000000 5.00000000            False 39981   18   59 0.00007504 1.00000000 0.96610169 21620        151 0.00285464 0.01858206 0.00278513 0.99722687 0.87260298        0.99999832            0.00002789               0.23841060               0.69536424     125      0.99771908     0.84000000 0.15771908        False         False          False   0.00000000  2.88230374 -10.48721186  0.00310378               True    25
9.50000000 9.75000000 5.00000000             True 39981   18   59 0.00007504 1.00000000 0.96610169 21620        151 0.00276171 0.01738556 0.00274474 0.99722687 0.87260298        0.99990943            0.00002059               0.25165563               0.66887417     118      0.99295963     0.85593220 0.13702742        False         False          False   0.00000000  2.43613120 -10.79092545 11.26884285               True    14
9.50000000 9.75000000 4.00000000            False 18783   18   59 0.00000000 1.00000000 0.96610169 21620        151 0.00303232 0.03822468 0.00305441 0.96271275 0.83580198        1.00000000            0.00000000               0.26490066               0.72185430     133      0.99953604     0.81954887 0.17998717        False         False          False 382.56638294  9.24892466 -27.92443120  0.00000000               True    42
9.50000000 9.75000000 4.00000000             True 18783   18   59 0.00000000 1.00000000 0.96610169 21620        151 0.00292916 0.03989682 0.00295094 0.95284823 0.82036897        1.00000000            0.00000000               0.30463576               0.69536424     122      0.99999699     0.86065574 0.13934125        False         False          False 583.17066941 17.13124125 -74.67469946  9.30824090              False  5000
```

Avaliação cinza permanece [5,9). Cortes A−<5/<4 e início A+≥9/≥9,5. Para manter três âncoras não vazias, A+≥9,5 é dividido em [9,5;9,75) e ≥9,75; simplesmente manter divisão 9,5 produziria A+1 vazio. Taxas zero/um são mantidas no alvo, com pesos estabilizados. Nenhum corte é selecionado pelo desempenho cinzento.

## Decisão histórica sem guardas

```text
       modo         calibracao  revisao  auto_match  auto_nonmatch  fn_auto  fp_auto  perda_revisor_perfeito  perda_revisor_config
 vigilancia         platt_nota     1590           0          20030       69        0                 3847.50               4816.70
confirmacao         platt_nota      210           0          21410      107        0                 2192.50               2777.30
 vigilancia         ancoras_ml      100         128          21392       17       22                 1095.00               1265.60
confirmacao         ancoras_ml       70         116          21434       21       15                 1937.50               2203.90
 vigilancia       beta_ancoras       44         131          21445       22       24                 1351.00               1465.80
confirmacao       beta_ancoras       27         122          21471       35       17                 2406.75               2484.35
 vigilancia beta_ancoras_cinza       18         125          21477       40       20                 2204.50               2238.90
confirmacao beta_ancoras_cinza       17         122          21481       40       17                 2504.25               2560.25
 vigilancia       isotonica_ml      311          93          21216        6        4                  417.75                797.95
confirmacao       isotonica_ml      153          56          21411       19        0                  418.25                939.85
```

Reusa a simulação v3 de M5, com seus custos, tetos, erros por banda e orçamento históricos; não altera a política/configuração de outros experimentos. Perdas configuradas são expectativas do revisor, não revisões executadas.

## Auditoria dos ajustes

```json
{
  "beta": {
    "a": 1.4694673097240867e-23,
    "b": 2.9599991030461807,
    "c": -10.556028469567098,
    "success": true,
    "message": "`ftol` termination condition is satisfied.",
    "nfev": 30,
    "objective": 0.09898623711065019,
    "n": [
      39981,
      18,
      77
    ],
    "targets": [
      7.503564192991671e-05,
      1.0,
      0.974025974025974
    ],
    "means": [
      7.405204647877804e-05,
      0.9893589342835839,
      0.9767928043685487
    ],
    "residuals": [
      -9.835954511386736e-07,
      -0.010641065716416098,
      0.0027668303425746332
    ],
    "standard_errors_plugin": [
      4.679031067211756e-05,
      0.03772949678527466,
      0.020072594669230247
    ],
    "weighted_residuals": [
      -0.021021349014568525,
      -0.282035718021269,
      0.13784119034775175
    ],
    "jacobian_singular_values": [
      6.486691109180198,
      2.815009058926822,
      0.036608536802663294
    ],
    "jacobian_condition": 177.19066850845257,
    "active_bounds": [
      -1,
      0,
      0
    ],
    "bounds_upper": [
      1000,
      1000,
      100
    ],
    "near_optimal_start_count": 9,
    "near_optimal_parameter_min": [
      7.706931265731209e-29,
      2.9599990739755873,
      -10.556028473574278
    ],
    "near_optimal_parameter_max": [
      9.718745917669977e-19,
      2.9599991089375353,
      -10.556028413357577
    ],
    "near_optimal_max_prediction_spread": 1.657458525983202e-08,
    "seed": 42,
    "starts": 12
  },
  "beta_plus": {
    "a": 277.72308753956963,
    "b": 1.3239588690269255,
    "c": -0.555927309457396,
    "success": true,
    "message": "`ftol` termination condition is satisfied.",
    "nfev": 94,
    "objective": 12.33251007264823,
    "n": [
      39981,
      18,
      77,
      21620
    ],
    "targets": [
      7.503564192991671e-05,
      1.0,
      0.974025974025974,
      0.004621497305782195
    ],
    "means": [
      4.2043407514354175e-05,
      0.9828092457072777,
      0.9507869843306485,
      0.006104624369804915
    ],
    "residuals": [
      -3.2992234415562535e-05,
      -0.01719075429272232,
      -0.023238989695325563,
      0.0014831270640227202
    ],
    "standard_errors_plugin": [
      4.679031067211756e-05,
      0.03772949678527466,
      0.020072594669230247,
      0.00046240926220662756
    ],
    "weighted_residuals": [
      -0.7051082572790582,
      -0.4556316876039453,
      -1.1577471711192953,
      3.2073904768801644
    ],
    "jacobian_singular_values": [
      3.424031969870732,
      0.035495964302477125,
      7.640051975308253e-05
    ],
    "jacobian_condition": 44816.86748907991,
    "active_bounds": [
      0,
      0,
      0
    ],
    "bounds_upper": [
      1000,
      1000,
      100
    ],
    "near_optimal_start_count": 12,
    "near_optimal_parameter_min": [
      277.63915222043596,
      1.3238162255953978,
      -0.5630025592471636
    ],
    "near_optimal_parameter_max": [
      277.7328540823938,
      1.3253127201225507,
      -0.5551667294983449
    ],
    "near_optimal_max_prediction_spread": 3.2846358230531614e-05,
    "seed": 42,
    "starts": 12
  }
}
```

## Reprodução

```powershell
python scripts/validar_ancoras_beta.py
python -m pytest tests/test_validar_ancoras_beta.py -q -o addopts=
```

**Beta por âncoras: sucesso conjunto=False**. Brier=0.00286914, ECE=0.00288390; intervalo superior previsto=0.99623215, observado=0.82812500, diferença=0.16810715.

**Beta + restrição cinza: sucesso conjunto=False**. Brier=0.00293588, ECE=0.00291990; intervalo superior previsto=0.99772048, observado=0.84000000, diferença=0.15772048.

Ajuste Beta: convergiu=True, condição do Jacobiano=177.191, amplitude máxima de p entre soluções quase ótimas=0.000000. Isso avalia identificabilidade prática, não só sucesso numérico do algoritmo.
Ajuste Beta+: convergiu=True; massa cinza ajustada=131.981979, alvo FS=99.916772; a restrição adicional é um momento ponderado, não igualdade imposta exatamente quando incompatível com os demais. Limites ativos=[0, 0, 0].

Se o critério falha, ampliar a família não basta para atribuir confiabilidade às âncoras: separar a incerteza do alvo FS da flexibilidade da curva, medir prevalências internas em amostra independente e regularizar curvas monotônicas com pooling são prioridades. Não se recomenda substituir a isotônica supervisionada sem validação independente e por pessoa.
