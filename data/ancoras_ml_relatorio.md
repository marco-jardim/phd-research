# M5 — Calibração por âncoras da margem supervisionada

## Veredito: NÃO: o critério conjunto não é satisfeito

Brier próximo: True; ECE próximo: False; mediana Y=1 >0,5: True; mediana Y=0 <0,1: True.
Proximidade operacional: erro ≤1,25×isotônica +0,0001 (não é margem de equivalência inferencial).

## Síntese dos resultados e recomendação

Âncoras vs isotônica na área cinza: Brier +20.33%, ECE +125.45% e log-loss +77.06%. A mediana alta dos vínculos (0.99972137) resolve o problema de ordenação da nota, mas NÃO demonstra probabilidades bem calibradas.
E1: Brier=0.00265412, ECE=0.00253659, resíduo máximo=9.14e-10. Igualar quase exatamente os momentos não melhora a calibração cinzenta; não é um efeito dos limites α/β, da penalização ou da inicialização examinados.
E2: Brier=0.00264855, ECE=0.00245503 (ainda 118.20% acima da isotônica), resíduo máximo logit=0.528643. Mesmo fornecendo taxas internas oracle, uma única sigmoide de dois parâmetros não satisfaz todos os momentos nem remove a sobreconfiança. Só adicionar momentos por banda ao mesmo Platt não basta.
E3: Brier=0.00353017, ECE=0.00367571, log-loss=0.02977637. O aninhamento rigoroso não resgata o resultado (ECE 226.69% acima da isotônica). Inclinações variam de 1.777052 a 8.432205 entre folds, coerente com instabilidade das taxas extremas raras e da escala das margens.
Conclusão diagnóstica: os controles não encontraram falha numérica que explique a discrepância. Há evidência de especificação/transporte inadequados dos dois momentos extremos para a área cinza e de sobreconfiança, não de perda da capacidade discriminativa do classificador. Isso não prova uma impossibilidade teórica geral das âncoras: testa esta seleção pela nota e esta família Platt neste extrato.
Melhoria prioritária: manter a isotônica supervisionada aninhada como referência operacional quando há gabarito. Para uma alternativa de poucos rótulos, medir âncoras internas por consistência independente de campos em amostra separada e substituir o Platt global por curva monotônica regularizada com restrições agregadas e pooling entre estratos/folds; avaliar por pessoa e fora da amostra. E2 mostra que a informação interna sozinha, sem flexibilidade de forma, teve benefício insuficiente.

## Reprodução e protocolo

```powershell
python scripts/validar_ancoras_ml.py
python scripts/validar_ancoras_ml.py --reuse-margins
```

Python Miniconda; seed 42; split_by=refrec; mesma fábrica XGBoost/allowlist de M1; 5 folds externos × (5 internos + 1 ajuste externo), 300 árvores por ajuste. Nenhum LLM executado.
Equivalência do adaptador à API sobre a nota: max |Δp|=0. Reprodução da isotônica M1: max |Δp|=1.11022302463e-16; folds iguais (asserções no comando).
Margem extraída por predict(output_margin=True), não pelo logit da isotônica (que perde informação e possui 0/1). O cache M1 original é somente lido e validado por hashes, features, seed e versões; margens M5 têm cache separado com hashes de fonte/código/pacote/CSV e versões.
Gabarito PAR∈{1,2}, leitura dtype=str, decimal vírgula convertido. pair_id é o mesmo hash de M1, com unicidade/cobertura verificadas. Somente hashes, margens, probabilidades e agregados são persistidos; nenhum identificador ou campo pessoal.

### Limite real da API do pacote

fit_anchor_platt seleciona A+/A- por cortes no próprio argumento nota. Passar a margem com cortes 9/5, ou recalcular cortes para a margem, TROCARIA os conjuntos. Para preservar exatamente as âncoras originais, implementou-se extensão local por máscaras, reutilizando _logit_mean_and_gradient do pacote e reproduzindo objetivo, inicialização, limites, L2 e tolerâncias. A chamada pública é executada como controle sobre a nota e comparada ao adaptador a cada execução. O pacote externo não foi alterado. A extensão por máscaras é uma adaptação explícita, não uma alegação de que a API pública aceita máscaras.

### Rótulos, OOF e limites de interpretação

A margem de cada par nunca é treinada no rótulo desse par. O ajuste principal usa apenas as duas taxas agregadas das âncoras extremas; não usa rótulos cinzentos diretamente. Taxas medidas exatas (sem arredondamento a cinco casas do script histórico). Os rótulos são reutilizados para avaliação offline, não para ajustar individualmente o Platt ancorado.
O Platt da nota é o comparador histórico in-sample de pcal_ml_oof.csv, NÃO calibração por âncoras da nota. A isotônica é estritamente aninhada. Já ajustar um único Platt às margens OOF globais não é uma avaliação totalmente aninhada: escores de outros folds podem depender do rótulo de um par avaliado, e taxas extremas incluem seus próprios rótulos no total. E3 elimina essa dependência: cada calibrador recebe apenas margens OOF internas e prevalências do treino externo. Separação por par, não por pessoa; registros compartilhados continuam sendo uma limitação. Nenhum resultado prova calibração sem gabarito nem validação externa.

## Métricas copiadas da saída do comando

```text
             calibracao escopo     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09
             platt_nota  total 61696        247 0.00226109 0.01177376 0.00038723 0.96775760 0.56247462        0.19891266            0.00010941               0.44534413               0.23076923
             platt_nota  cinza 21620        151 0.00614992 0.03155659 0.00074180 0.87315686 0.21632791        0.02633672            0.00164519               0.70860927               0.00000000
             ancoras_ml  total 61696        247 0.00101638 0.00593387 0.00092200 0.99800635 0.91505810        0.99983413            0.00000000               0.09311741               0.78137652
             ancoras_ml  cinza 21620        151 0.00265412 0.01512235 0.00253659 0.99722687 0.87260298        0.99972137            0.00000000               0.13907285               0.69536424
           isotonica_ml  total 61696        247 0.00088626 0.00369240 0.00030701 0.99553380 0.91072734        0.95238095            0.00002248               0.08502024               0.55060729
           isotonica_ml  cinza 21620        151 0.00220576 0.00854096 0.00112513 0.99514223 0.86226195        0.89473684            0.00002248               0.12582781               0.48344371
exp1_sem_limites_ativos  total 61696        247 0.00101638 0.00593389 0.00092200 0.99800635 0.91505810        0.99983414            0.00000000               0.09311741               0.78137652
exp1_sem_limites_ativos  cinza 21620        151 0.00265412 0.01512241 0.00253659 0.99722687 0.87260298        0.99972139            0.00000000               0.13907285               0.69536424
  exp2_ancoras_internas  total 61696        247 0.00101400 0.00589643 0.00090135 0.99800635 0.91505810        0.99980507            0.00000000               0.09311741               0.78137652
  exp2_ancoras_internas  cinza 21620        151 0.00264855 0.01502303 0.00245503 0.99722687 0.87260298        0.99967390            0.00000000               0.13907285               0.69536424
 exp3_ancoras_aninhadas  total 61696        247 0.00133433 0.01164795 0.00137542 0.98697208 0.86858497        1.00000000            0.00000000               0.07287449               0.90688259
 exp3_ancoras_aninhadas  cinza 21620        151 0.00353017 0.02977637 0.00367571 0.97930606 0.81849270        0.99999999            0.00000000               0.10596026               0.86092715
```

Brier/log-loss menores são melhores; log-loss trunca p em [10^-6,1−10^-6] igualmente para todos, inclusive isotônica. AUC/AP são discriminação, não calibração. ECE usa dez intervalos uniformes [0,0,1), …, [0,9,1], ponderados por N. Com 151 vínculos em 21.620 cinzentos, Brier/ECE podem esconder erro na classe rara: por isso medianas e frações condicionais também são exigidas.
Curvas completas em ancoras_ml_confiabilidade.csv: intervalos uniformes (inclusive vazios) e decis por quantis. Empates isotônicos não são repartidos artificialmente; qcut remove bordas duplicadas, podendo haver menos de dez grupos. A contribuição ECE dos intervalos uniformes soma o ECE da validação.

### Sobreconfiança nos dois extremos cinzentos

```text
  calibracao escopo tipo_bin  bin     n  positivos      p_min      p_max    p_media  taxa_observada  contribuicao_ece
  platt_nota  cinza uniforme    1 21410        107 0.00034229 0.09876476 0.00508268      0.00499766        0.00008419
  platt_nota  cinza uniforme   10     0          0        NaN        NaN        NaN             NaN        0.00000000
  ancoras_ml  cinza uniforme    1 21434         21 0.00000000 0.06949813 0.00008782      0.00097975        0.00088426
  ancoras_ml  cinza uniforme   10   124        105 0.91769327 0.99999940 0.99455744      0.84677419        0.00084760
isotonica_ml  cinza uniforme    1 21411         19 0.00000000 0.07142857 0.00053745      0.00088739        0.00034656
isotonica_ml  cinza uniforme   10    75         73 0.95121951 1.00000000 0.98845695      0.97333333        0.00005246
```

Compare p_media com taxa_observada, não apenas a quantidade de pares: o bin de baixa probabilidade pode subestimar vínculos raros mesmo tendo ótimo desempenho majoritário; o bin ≥0,9 pode superestimar o valor preditivo positivo. Esses são intervalos de probabilidade, não bandas da nota.

## Sensibilidade aos cortes

```text
    s_plus    s_minus  n_plus  n_minus  r_plus_medida  r_minus_medida  r_plus_alvo  r_minus_alvo  max_logit_residual     n  positivos      brier   log_loss     ece_10        auc         ap  mediana_vinculos  mediana_nao_vinculos  fracao_vinculos_p_lt_01  fracao_vinculos_p_gt_09
9.00000000 5.00000000      95    39981     0.97894737      0.00007504   0.97894737    0.00007504          0.00001382 21620        151 0.00265412 0.01512235 0.00253659 0.99722687 0.87260298        0.99972137            0.00000000               0.13907285               0.69536424
9.00000000 4.00000000      95    18783     0.97894737      0.00000000   0.97894737    0.00002662          0.00000108 21620        151 0.00281473 0.01063355 0.00256408 0.99722687 0.87260298        0.99447172            0.00001593               0.09271523               0.69536424
9.50000000 5.00000000      77    39981     0.97402597      0.00007504   0.97402597    0.00007504          0.00001304 21620        151 0.00262900 0.01458822 0.00238541 0.99722687 0.87260298        0.99950441            0.00000000               0.13907285               0.69536424
9.50000000 4.00000000      77    18783     0.97402597      0.00000000   0.97402597    0.00002662          0.00000124 21620        151 0.00268496 0.01016078 0.00255455 0.99722687 0.87260298        0.99234366            0.00001630               0.09271523               0.69536424
```

Mediana cinzenta Y=1 varia de 0.99234366 a 0.99972137 (amplitude 0.00737771). O conjunto de avaliação [5,9) permanece fixo.
Se uma âncora tem prevalência 0/1, a restrição logit não tem solução finita: usa-se Jeffreys (pos+0,5)/(N+1) SOMENTE nessa fronteira; taxas medidas e alvos distintos ficam registrados. Cortes mais estritos não garantem estimativas mais estáveis: A+ é pequeno e A- tem pouquíssimos positivos.

## Diagnóstico: implementação versus hipótese de transporte

```text
{
  "slope": 1.6039883544311664,
  "intercept": -5.04849562172188,
  "success": true,
  "message": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
  "objective": 2.8060279854229138e-05,
  "max_logit_residual": 1.381521485832593e-05,
  "means": [
    0.9789473370949056,
    7.503667849280674e-05
  ],
  "targets": [
    0.9789473684210527,
    7.503564192991671e-05
  ],
  "residuals_logit": [
    -1.5199907119622935e-06,
    1.381521485832593e-05
  ],
  "l2": 1e-06,
  "alpha_max": 10.0,
  "beta_max": 50.0
}
```

```text
grupo  y     n  margem_q0  margem_q10  margem_q50  margem_q90  margem_q100  fracao_sigmoid_lt_001  fracao_sigmoid_gt_099
   A+  0     2   3.159263    3.235784    3.541869    3.847953     3.924474               0.000000               0.000000
   A+  1    93   3.924474    5.221572    9.184315   11.203357    11.963095               0.000000               0.946237
   A-  0 39978 -14.440290  -12.907588  -11.751497  -10.156830     5.322026               0.992121               0.000025
   A-  1     3 -10.703617   -9.738482   -5.877944    0.684871     2.325575               0.666667               0.000000
cinza  0 21469 -14.878944  -13.129839  -11.837603   -9.340353     8.493429               0.986585               0.000885
cinza  1   151  -9.824228   -0.184763    8.250595   10.995582    12.083411               0.033113               0.695364
```

O resíduo logit e o status do otimizador distinguem falha de encaixe dos momentos de falha de calibração fora das âncoras. Igualar duas médias não identifica a curva P(Y=1|margem) na área intermediária. As âncoras são selecionadas pela nota, que também é feature do classificador: P(Y|margem,âncora) não precisa ser P(Y|margem,cinza). Distribuições/quantis por classe acima permitem verificar sobreposição e saturação sem presumir que não há suporte comum. scale_pos_weight do XGBoost altera a escala probabilística bruta; a margem não é um log-odds populacional já calibrado. Platt continua monótono na margem e preserva sua ordenação, ao contrário de corrigir a ordenação insuficiente da nota.
Nas âncoras positivas, os 2 não vínculos têm margem entre 3.159263 e 3.924474; entre não vínculos cinzentos a margem máxima é 8.493429. Nas âncoras negativas, os 3 vínculos têm mediana de margem -5.877944, contra 8.250595 nos vínculos cinzentos. As distribuições têm suporte sobreposto: 'âncoras só cobrem extremos' refere-se à seleção pela nota, não a separação perfeita das margens. Os números são consistentes com seleção não representativa e saturação dos escores fáceis, sem provar isoladamente violação da independência condicional. AUC da isotônica pode mudar por patamares/transformações diferentes por fold; AUC maior do Platt ancorado global não implica calibração melhor.

## Três experimentos rápidos executados

1. **E1 — limites/L2/ótimo local:** mesmas duas âncoras, L2=0, limites α≤100, |β|≤500 e quatro inicializações. Compara com α≤10, |β|≤50 e L2=10^-6. Se não resolver a discrepância, afrouxar o otimizador não basta.
2. **E2 — âncoras internas estratificadas:** adiciona quatro momentos de prevalência nas bandas [5,6), …, [8,9), com igual peso logit; ainda só dois parâmetros Platt. É experimento oracle/otimista: usa informação agregada dos próprios cinzentos avaliados, NÃO estimativa independente nem âncora disponível sem rótulos. Incompatibilidade dos seis momentos aparece nos resíduos do JSON. Próximo experimento prospectivo: âncoras internas definidas por consistência independente de campos, prevalências medidas numa amostra separada e ajuste monotônico mais flexível com restrições de médias por estrato.
3. **E3 — aninhamento rigoroso:** calibra por âncoras nas margens OOF internas, mede taxas só no treino externo, aplica à margem do teste externo. Controla dependência indireta do ajuste OOF global e flutuação de prevalências entre folds. Não muda os modelos ou o cache de M1. Auditoria por fold em ancoras_ml_ajustes.json registra taxas, N e resíduos.

### Ajustes dos experimentos (saída numérica)

```json
{
  "exp1": {
    "slope": 1.603997569733112,
    "intercept": -5.048532243002667,
    "success": true,
    "message": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    "objective": 9.51862448933292e-19,
    "max_logit_residual": 9.137526291169706e-10,
    "means": [
      0.9789473684140056,
      7.503564186135777e-05
    ],
    "targets": [
      0.9789473684210527,
      7.503564192991671e-05
    ],
    "residuals_logit": [
      -3.419335925514133e-10,
      -9.137526291169706e-10
    ],
    "l2": 0,
    "alpha_max": 100,
    "beta_max": 500
  },
  "exp2": {
    "rates": [
      {
        "n": 95,
        "positivos": 93,
        "taxa_medida": 0.9789473684210527,
        "alvo": 0.9789473684210527,
        "suavizada": false
      },
      {
        "n": 39981,
        "positivos": 3,
        "taxa_medida": 7.503564192991671e-05,
        "alvo": 7.503564192991671e-05,
        "suavizada": false
      },
      {
        "n": 14263,
        "positivos": 17,
        "taxa_medida": 0.0011918951132300357,
        "alvo": 0.0011918951132300357,
        "suavizada": false
      },
      {
        "n": 6175,
        "positivos": 57,
        "taxa_medida": 0.009230769230769232,
        "alvo": 0.009230769230769232,
        "suavizada": false
      },
      {
        "n": 1080,
        "positivos": 43,
        "taxa_medida": 0.03981481481481482,
        "alvo": 0.03981481481481482,
        "suavizada": false
      },
      {
        "n": 102,
        "positivos": 34,
        "taxa_medida": 0.3333333333333333,
        "alvo": 0.3333333333333333,
        "suavizada": false
      }
    ],
    "fit": {
      "slope": 1.5912688751623885,
      "intercept": -5.100945174457932,
      "success": true,
      "message": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
      "objective": 0.3994723083314969,
      "max_logit_residual": 0.5286429931708003,
      "means": [
        0.9772089207568875,
        7.312134958000315e-05,
        0.0014459764087554053,
        0.007575479727119959,
        0.03319827986336549,
        0.4589664476694963
      ],
      "targets": [
        0.9789473684210527,
        7.503564192991671e-05,
        0.0011918951132300357,
        0.009230769230769232,
        0.03981481481481482,
        0.3333333333333333
      ],
      "residuals_logit": [
        -0.0811210424864246,
        -0.025844756786213452,
        0.19349465311465242,
        -0.1992950229871795,
        -0.1886082722622877,
        0.5286429931708003
      ],
      "l2": 1e-06,
      "alpha_max": 10.0,
      "beta_max": 50.0
    }
  }
}
```

Isotônica com restrições agregadas é uma melhoria concreta para a próxima rodada, mas duas médias sozinhas deixam muitos graus de liberdade: exigir regularização/suavidade e âncoras internas independentes, escolher hiperparâmetros só no treino e avaliar por fold/pessoa. EM com prevalência conhecida só é justificável sob hipóteses de mistura/label shift verificáveis; a seleção pela nota pode violá-las. Não foi executado um quarto experimento nem apresentado EM/isotônica restrita como solução comprovada.

## Decisão — mesma política histórica, sem guardas

```text
       modo   calibracao  revisao  auto_match  auto_nonmatch  fn_auto  fp_auto  perda_revisor_perfeito  perda_revisor_config
 vigilancia   platt_nota     1590           0          20030       69        0                 3847.50               4816.70
 vigilancia   ancoras_ml      100         128          21392       17       22                 1095.00               1265.60
 vigilancia isotonica_ml      311          93          21216        6        4                  417.75                797.95
confirmacao   platt_nota      210           0          21410      107        0                 2192.50               2777.30
confirmacao   ancoras_ml       70         116          21434       21       15                 1937.50               2203.90
confirmacao isotonica_ml      153          56          21411       19        0                  418.25                939.85
```

PolicyEngineV3 policy_version=v3; vigilância: Cfp=10, Cfn=50, Cllm=0,25, tetos 0,85/0,15, orçamento 2.000; confirmação: 100/20/0,25, tetos 0,95/0,10, orçamento 1.000. Erros por banda idênticos ao validar_ancoras.py. Não se usa v3.1, guardas ou custo humano: objetivo é isolar calibração nas tabelas históricas. Perda perfeita soma FP/FN automáticos + custo de revisão; perda configurada acrescenta erros esperados do revisor sobre as classes observadas. Não houve revisão LLM real.

## Artefatos

Script scripts/validar_ancoras_ml.py; CSVs ancoras_ml_{validacao,confiabilidade,decisao,sensibilidade,diagnostico,margens_oof}.csv; JSONs de cache/ajustes; tabela nova tab_cap8_ancoras_ml.tex. Tabelas/capítulo/configurações existentes e pacote externo não são modificados.
