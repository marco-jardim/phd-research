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
  "score_column": "SCORE,C,15,0",
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
        calibracao escopo     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09  high_n  high_predicted  high_observed    high_gap  success_ece  success_high  success_joint  optimizer_success
        platt_nota  cinza 21620        151 0.00614992 0.03155659 0.00074180 0.87315686 0.21632791        0.02633672            0.00164519               0.70860927               0.00000000       0             NaN            NaN         NaN         True         False          False               True
        platt_nota  total 61696        247 0.00226109 0.01177376 0.00038723 0.96775760 0.56247462        0.19891266            0.00010941               0.44534413               0.23076923      59      0.97971816     0.96610169  0.01361646        False          True          False               True
        ancoras_ml  cinza 21620        151 0.00153921 0.00956310 0.00148309 0.99642531 0.92870563        0.99972311            0.00000000               0.11258278               0.82781457     134      0.99532366     0.93283582  0.06248784        False         False          False               True
        ancoras_ml  total 61696        247 0.00064089 0.00405243 0.00056510 0.99713350 0.94571879        0.99977885            0.00000000               0.08097166               0.85829960     222      0.99591956     0.95495495  0.04096460        False         False          False               True
      beta_ancoras  cinza 21620        151 0.00171681 0.01012899 0.00179616 0.99642531 0.92870563        0.99250618            0.00000000               0.09933775               0.82781457     135      0.98600637     0.92592593  0.06008045        False         False          False               True
      beta_ancoras  total 61696        247 0.00070369 0.00426060 0.00068963 0.99713350 0.94571879        0.99319468            0.00000000               0.07287449               0.85829960     224      0.98707514     0.94642857  0.04064657        False         False          False               True
beta_ancoras_cinza  cinza 21620        151 0.00153978 0.01422755 0.00148463 0.99242940 0.92754657        0.99202430            0.00000000               0.13907285               0.81456954     131      0.98817581     0.93893130  0.04924451        False         False          False               True
beta_ancoras_cinza  total 61696        247 0.00064303 0.00582877 0.00059361 0.99340637 0.94500839        0.99262062            0.00000000               0.09716599               0.84615385     218      0.98896736     0.95871560  0.03025176        False         False          False               True
      isotonica_ml  cinza 21620        151 0.00151761 0.00723098 0.00075774 0.99292018 0.90696445        0.98181818            0.00003193               0.09271523               0.66887417     105      0.98657041     0.98095238  0.00561803         True          True           True               True
      isotonica_ml  total 61696        247 0.00063745 0.00317664 0.00023683 0.99403231 0.93391840        1.00000000            0.00003193               0.06477733               0.70445344     183      0.98618440     0.98907104 -0.00288663         True          True           True               True
```

## Sensibilidade

```text
    s_plus    s_split    s_minus  restricao_cinza   n_0  n_1  n_2        r_0        r_1        r_2     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09  high_n  high_predicted  high_observed   high_gap  success_ece  success_high  success_joint            a          b           c   objective  optimizer_success  nfev
9.00000000 9.50000000 5.00000000            False 39981   18   77 0.00007504 1.00000000 0.97402597 21620        151 0.00171681 0.01012899 0.00179616 0.99642531 0.92870563        0.99250618            0.00000000               0.09933775               0.82781457     135      0.98600637     0.92592593 0.06008045        False         False          False   5.58667949 0.66701582 -0.76987227  0.33047229               True    27
9.00000000 9.50000000 5.00000000             True 39981   18   77 0.00007504 1.00000000 0.97402597 21620        151 0.00153978 0.01422755 0.00148463 0.99242940 0.92754657        0.99202430            0.00000000               0.13907285               0.81456954     131      0.98817581     0.93893130 0.04924451        False         False          False 106.64374459 0.51842141  0.44855113 17.62755866               True    61
9.00000000 9.50000000 4.00000000            False 18783   18   77 0.00000000 1.00000000 0.97402597 21620        151 0.00189996 0.00778858 0.00197223 0.99642531 0.92870563        0.99259553            0.00000008               0.06622517               0.82781457     134      0.98646635     0.93283582 0.05363053        False         False          False   1.31146983 0.68666370 -0.92531481  0.32344132               True    30
9.00000000 9.50000000 4.00000000             True 18783   18   77 0.00000000 1.00000000 0.97402597 21620        151 0.00153571 0.01414391 0.00148297 0.99237912 0.92754657        0.99291146            0.00000000               0.13907285               0.81456954     131      0.98838919     0.93893130 0.04945789        False         False          False  98.53962111 0.59482199 -0.08229336 17.19435420               True    50
9.50000000 9.75000000 5.00000000            False 39981   18   59 0.00007504 1.00000000 0.96610169 21620        151 0.00153317 0.00952288 0.00144193 0.99642531 0.92870563        0.99708934            0.00000000               0.10596026               0.81456954     131      0.99118622     0.93893130 0.05225492        False         False          False   3.40390844 1.02842764 -2.88527540  0.24646011               True    33
9.50000000 9.75000000 5.00000000             True 39981   18   59 0.00007504 1.00000000 0.96610169 21620        151 0.00152617 0.01420076 0.00145199 0.99244791 0.92754657        0.98638110            0.00000000               0.13907285               0.80794702     130      0.98232805     0.93846154 0.04386651        False         False          False 112.72858259 0.44647001  0.51929308 16.19476998               True    78
9.50000000 9.75000000 4.00000000            False 18783   18   59 0.00000000 1.00000000 0.96610169 21620        151 0.00153846 0.00741668 0.00148886 0.99642531 0.92870563        0.99695633            0.00000011               0.09271523               0.81456954     131      0.99092040     0.93893130 0.05198910        False         False          False   1.11123222 1.02429030 -2.89547643  0.24325171               True    38
9.50000000 9.75000000 4.00000000             True 18783   18   59 0.00000000 1.00000000 0.96610169 21620        151 0.00151830 0.01403352 0.00144452 0.99237820 0.92754657        0.98807981            0.00000000               0.13907285               0.80132450     129      0.98325627     0.93798450 0.04527177        False         False          False  98.13730620 0.55882816 -0.30173343 15.70206567               True    64
```

Avaliação cinza permanece [5,9). Cortes A−<5/<4 e início A+≥9/≥9,5. Para manter três âncoras não vazias, A+≥9,5 é dividido em [9,5;9,75) e ≥9,75; simplesmente manter divisão 9,5 produziria A+1 vazio. Taxas zero/um são mantidas no alvo, com pesos estabilizados. Nenhum corte é selecionado pelo desempenho cinzento.

## Decisão histórica sem guardas

```text
       modo         calibracao  revisao  auto_match  auto_nonmatch  fn_auto  fp_auto  perda_revisor_perfeito  perda_revisor_config
 vigilancia         platt_nota     1590           0          20030       69        0                 3847.50               4816.70
confirmacao         platt_nota      210           0          21410      107        0                 2192.50               2777.30
 vigilancia         ancoras_ml       69         138          21413       14       12                  837.25                927.05
confirmacao         ancoras_ml       41         125          21454       17        6                  950.25               1164.65
 vigilancia       beta_ancoras       72         141          21407       12       13                  748.00                836.80
confirmacao       beta_ancoras       93         103          21424       15        1                  423.25                849.25
 vigilancia beta_ancoras_cinza       13         134          21473       21        9                 1143.25               1177.45
confirmacao beta_ancoras_cinza       40         105          21475       21        1                  530.00                654.80
 vigilancia       isotonica_ml      230         126          21264        7        8                  487.50                714.70
confirmacao       isotonica_ml      119          81          21420       14        1                  409.75                863.35
```

Reusa a simulação v3 de M5, com seus custos, tetos, erros por banda e orçamento históricos; não altera a política/configuração de outros experimentos. Perdas configuradas são expectativas do revisor, não revisões executadas.

## Auditoria dos ajustes

```json
{
  "beta": {
    "a": 5.586679487358167,
    "b": 0.6670158190373664,
    "c": -0.7698722663554288,
    "success": true,
    "message": "`ftol` termination condition is satisfied.",
    "nfev": 27,
    "objective": 0.33047228704740145,
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
      7.511023018786428e-05,
      0.9804726151663639,
      0.9790480742766173
    ],
    "residuals": [
      7.458825794757508e-08,
      -0.01952738483363614,
      0.005022100250643313
    ],
    "standard_errors_plugin": [
      4.679031067211756e-05,
      0.03772949678527466,
      0.020072594669230247
    ],
    "weighted_residuals": [
      0.0015940962322360125,
      -0.5175628221274722,
      0.2501968646007588
    ],
    "jacobian_singular_values": [
      5.889255947266835,
      0.7774443311236142,
      1.675105681429821e-09
    ],
    "jacobian_condition": 3515751879.153045,
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
      5.586655358327286,
      0.6670138716446975,
      -0.7698955159144028
    ],
    "near_optimal_parameter_max": [
      5.586690297790285,
      0.6670200030607352,
      -0.7698617270553243
    ],
    "near_optimal_max_prediction_spread": 4.072439920399518e-06,
    "seed": 42,
    "starts": 12
  },
  "beta_plus": {
    "a": 106.64374458621643,
    "b": 0.5184214124274462,
    "c": 0.4485511279164646,
    "success": true,
    "message": "`ftol` termination condition is satisfied.",
    "nfev": 61,
    "objective": 17.62755865968196,
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
      4.4471189257395917e-05,
      0.9545526998873215,
      0.961460718842038,
      0.006433648466866884
    ],
    "residuals": [
      -3.056445267252079e-05,
      -0.0454473001126785,
      -0.01256525518393603,
      0.0018121511610846887
    ],
    "standard_errors_plugin": [
      4.679031067211756e-05,
      0.03772949678527466,
      0.020072594669230247,
      0.00046240926220662756
    ],
    "weighted_residuals": [
      -0.65322183660418,
      -1.2045562221867747,
      -0.6259905802411089,
      3.918933527492645
    ],
    "jacobian_singular_values": [
      8.244685624060322,
      0.010733052140595537,
      0.0009469505163912844
    ],
    "jacobian_condition": 8706.564367776931,
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
    "near_optimal_start_count": 4,
    "near_optimal_parameter_min": [
      106.63845571557819,
      0.5184214124274462,
      0.44830158472541864
    ],
    "near_optimal_parameter_max": [
      106.64374458621643,
      0.5184555886532385,
      0.4485511279164646
    ],
    "near_optimal_max_prediction_spread": 8.006683791861313e-06,
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

**Beta por âncoras: sucesso conjunto=False**. Brier=0.00171681, ECE=0.00179616; intervalo superior previsto=0.98600637, observado=0.92592593, diferença=0.06008045.

**Beta + restrição cinza: sucesso conjunto=False**. Brier=0.00153978, ECE=0.00148463; intervalo superior previsto=0.98817581, observado=0.93893130, diferença=0.04924451.

Ajuste Beta: convergiu=True, condição do Jacobiano=3.51575e+09, amplitude máxima de p entre soluções quase ótimas=0.000004. Isso avalia identificabilidade prática, não só sucesso numérico do algoritmo.
Ajuste Beta+: convergiu=True; massa cinza ajustada=139.095480, alvo FS=99.916772; a restrição adicional é um momento ponderado, não igualdade imposta exatamente quando incompatível com os demais. Limites ativos=[0, 0, 0].

Se o critério falha, ampliar a família não basta para atribuir confiabilidade às âncoras: separar a incerteza do alvo FS da flexibilidade da curva, medir prevalências internas em amostra independente e regularizar curvas monotônicas com pooling são prioridades. Não se recomenda substituir a isotônica supervisionada sem validação independente e por pessoa.

## Discussão dos resultados finais

As duas variantes convergiram após a correção de escala numérica, em 27 e 61 avaliações na melhor inicialização, sem limites ativos. Beta apresentou ECE cinza de 0,00179616 e diferença superior prevista menos observada de 0,06008045; Beta+ apresentou 0,00148463 e 0,04924451. Nenhuma satisfaz o critério conjunto. Beta+ melhora o Brier relativamente à Beta simples (0,00153978 contra 0,00171681), mas a log-loss piora (0,01422755 contra 0,01012899), e não melhora a calibração relativamente à isotônica supervisionada.

Há um limite analítico para a restrição interna. Para quaisquer dez bins, a desigualdade triangular dá ECE ≥ |média(p)−média(Y)|. Se o quarto momento fosse imposto exatamente, o alvo FS implicaria ECE ≥ |99,91677175101105−151|/21620 = aproximadamente 0,002363, acima do limite de sucesso de aproximadamente 0,000909. Logo, impor exatamente esse alvo enviesado tornaria o critério impossível neste extrato, independentemente da família calibradora. O ajuste WLS não o impõe exatamente e conserva uma massa cinza maior, como registrado no JSON.

O Jacobiano da Beta tem condição aproximada de 3,52×10^9, evidência de identificação local fraca dos momentos. Isso não deve ser confundido com dispersão efetivamente observada entre inicializações: as doze soluções quase ótimas diferem em no máximo 0,0000041 nas previsões. A convergência do algoritmo não garante que as restrições externas transportem calibração para a área cinza.

A Beta é monotônica no escore bruto, mas coeficientes elevados em Beta+ fazem probabilidades extremamente pequenas subfluírem numericamente para zero. Clipping e empates numéricos podem reduzir a AUC calculada sobre p, sem inverter a ordenação teórica dos escores. Essa redução não é evidência isolada de perda de informação discriminativa pelo XGBoost.

Na decisão histórica, Beta simples reduz a perda configurada frente ao Platt ancorado: 836,80 contra 927,05 em vigilância e 849,25 contra 1164,65 em confirmação. Beta+ reduz a perda em confirmação para 654,80, mas produz 21 falsos negativos automáticos, contra 14 na isotônica; em vigilância sua perda é 1177,45, contra 714,70 da isotônica. Portanto, eventual ganho de custo não é demonstração de calibração ou superioridade geral.

Prioridades para novos experimentos, ainda não executados nesta rodada: auditar LR e estimar priors FS por passo de blocagem antes de transportar posteriores ao universo selecionado; representar incerteza do total cinza por intervalo ou penalização robusta em vez de pseudo-prevalência precisa; obter âncoras internas independentes por consistência de campos, com validação por pessoa e fora da amostra. Os dados atuais não sustentam apresentar Beta por momentos como correção resolvida da sobreconfiança.

## Verificação técnica

Na execução integrada, `python -m pytest tests -q` retornou **97 passed in 8.61s**. O verificador de artefatos confirmou cinco modelos, dez linhas de métricas/decisões, oito sensibilidades, consistência entre ECE e bins e ausência de campos dos registros nos CSVs. O fingerprint do script de calibração coincide com o JSON de auditoria.

A tese foi compilada pela cadeia completa do repositório (pdflatex, BibTeX, pdflatex duas vezes), com jobname separado `tese_m5b_bd1849b8`. O log foi verificado sem erro de LaTeX ou citação indefinida, e a extração do PDF confirmou a presença da tabela. A referência Kull2017beta é unívoca, e seus metadados foram conferidos no PMLR. A subseção tem cerca de 230 palavras de prosa, equação e tabela; uma quebra de página local evita separá-la em duas páginas.

Os PDFs `text/latex/tese.pdf` e `text/pdf/tese.pdf` já estavam modificados antes desta tarefa. Seus hashes foram preservados pela compilação com nome separado, em vez de executar checkout e descartar alterações preexistentes. Não houve commit, chamada LLM, alteração do cache M1 ou edição dos arquivos Python/YAML de gzcmd nesta tarefa. A falha inicial de testes da política v3.1 foi observada durante alterações concorrentes de custos; os testes desse outro escopo não foram editados nesta tarefa.
