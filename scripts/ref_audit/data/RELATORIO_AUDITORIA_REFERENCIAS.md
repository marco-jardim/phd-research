# Relatório de Auditoria de Referências

**Gerado em:** 2026-02-07 14:15

## Resumo

| Score | Qtd | % |
|-------|-----|---|
| OK REF_SOLID | 1 | 4% |
| OK REF_OK | 1 | 4% |
| ! REF_REVIEW | 13 | 48% |
| X REF_PROBLEM | 12 | 44% |
| XX REF_CRITICAL | 0 | 0% |
| **TOTAL** | **27** | |

## REF_PROBLEM (12 referências)

### `Barreto2019The`

**Título:** The Centre for Data and Knowledge Integration for Health (CIDACS): Linking Health and Social Data in Brazil  
**Ano:** 2019  
**DOI:** 10.23889/ijpds.v4i2.1140  
**L1 (Existência):** L1_FAIL_GHOST (title_sim=74%, year=True, author=False)
**L2 (Journal):** L2_PASS_HIGH (journal=International Journal of Population Data Science, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the Minha Casa Minha Vida program and its association with leprosy incidence, which is relevant to the broader context of health data and identifiers, but does not directly address the completeness or quality of the Cartão Nacional de Saúde (CNS) as a primary key for data linkage.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on the association between a housing program and leprosy incidence, which does not relate to the analytical potential of data linkage for tuberculosis underreporting or data quality assessment mentioned in the thesis.

### `Breiman2001rf`

**Título:** Random Forests  
**Ano:** 2001  
**DOI:** 10.1023/A:1010933404324  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Machine Learning, quartile=, evidence=B)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the use of random forests to analyze complex composite endpoints, which aligns with the claim about machine learning techniques capturing complex patterns, but it does not explicitly mention the incorporation of a larger number of predictor variables or interactions as stated in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract discusses the use of random forests in the context of composite endpoints in medical research, but it does not provide evidence or support for the specific claim regarding the construction of decision trees from random samples of training data and variable selection as described in the thesis sentence.

### `Camargo2000reclink`

**Título:** Reclink: aplicativo para o relacionamento de bases de dados, implementando o m\'etodo probabil\'\istico  
**Ano:** 2000  
**DOI:** 10.1590/S0102-311X2000000200014  
**L1 (Existência):** L1_PASS (title_sim=89%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Cadernos de Saúde Pública, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses a system for database linkage, which aligns with the claim about the need to relate records from different health information systems due to their fragmentation.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract discusses a system for database linkage but does not mention the absence of a unique identifier for linking records, which is the specific claim made in the thesis.
  - TOOL: L3_PASS [HIGH] — The abstract describes a system for probabilistic record linkage, which aligns with the claim that OpenRecLink is a widely used tool for probabilistic linkage in health databases.
  - SUPPORT_FACT: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a probabilistic record linkage system and its performance, which is relevant to the claim about similarity scores, but it does not explicitly mention the concept of similarity scores or their role in determining thresholds for classification.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses a system for database linkage, which aligns with the claim about the necessity of record linkage for integrating fragmented health data in Brazil.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract discusses a system for database linkage using probabilistic record linkage techniques, which directly supports the claim about the need for computational methods to classify records as true pairs, non-pairs, or doubtful pairs.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract discusses the use of a probabilistic record linkage system that improves processing speed and maintains sensitivity, directly supporting the claim about the application of probabilistic methods in epidemiological studies.
  - DEFINE: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a probabilistic record linkage system and its performance, but does not explicitly address the empirical or convenience-based definition of classification thresholds mentioned in the claim.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the development and testing of a database linkage system, which aligns with the claim of the laboratory's extensive experience in health data linkage in Brazil.
  - TOOL: L3_PASS [HIGH] — The abstract describes the OpenRecLink program as a system for probabilistic record linkage, which directly supports its use in generating candidate pairs for analysis in the thesis.
**Flags:**
  - STALE_REF [MÉDIA]: Published 2000, only 109 citations. Threshold: <2005 AND <200 cit.

### `Christen2012book`

**Título:** Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection  
**Ano:** 2012  
**DOI:** 10.1007/978-3-642-31164-2  
**L1 (Existência):** L1_WARN_META (title_sim=23%, year=True, author=True)
**L2 (Journal):** L2_NA (journal=, quartile=, evidence=B)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract indicates that the book addresses data matching, which is relevant to record linkage, but does not provide specific evidence or methods related to the claim about techniques used in record linkage.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide specific evidence or details regarding the incorporation of machine learning approaches for automated classification of candidate pairs, which is the focus of the claim.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract indicates that the book addresses data matching aspects, which is relevant to the claim about indirect methods of linking data, but it does not provide specific evidence or details supporting the claim about the challenges and errors associated with these methods.
  - BACKGROUND: L3_PASS [HIGH] — The abstract indicates that the book addresses data quality and data matching, which aligns with the limitations of manual review mentioned in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide specific evidence or details regarding the advantages and limitations of different data matching strategies, which is necessary to support the claim made in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide specific evidence or details regarding the deterministic relationship or criteria for matching records as described in the thesis sentence.
  - DATA_SOURCE: L3_PASS [HIGH] — The abstract indicates that the cited paper addresses data matching, which is relevant to the process of classifying pairs into categories based on similarity scores as described in the thesis sentence.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not mention the specific concept of classification thresholds or their limitations in data matching, which are central to the claim made in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide specific evidence or details regarding the use of machine learning techniques for candidate pair classification in record linkage, which is the focus of the thesis claim.
  - DEFINE: L3_WARN_PARTIAL [MEDIUM] — The abstract mentions data matching, which is relevant to the claim about class imbalance in record linkage, but it does not specifically address the severity of class imbalance or provide quantitative details as stated in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide any specific evidence or details regarding the decision model proposed by Fellegi and Sunter or the concept of the 'gray area' in classification, which is essential for supporting the claim made in the thesis.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide specific evidence or support for the claim regarding the impact of incomplete fields, spelling errors, or homonyms on the gray area in data matching.
  - BACKGROUND: L3_PASS [HIGH] — The abstract indicates that the book addresses data quality and data matching, which aligns with the limitations of manual review mentioned in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide any evidence or specific information regarding the proportion of true pairs to total candidates in the context of record linkage, which is essential for supporting the claim made in the thesis.
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract indicates that the book addresses data quality and data matching, which is relevant to the discussion of cut-off points and their impact on false positives, but does not explicitly support the specific claim about low cut-off points leading to false positives.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses data matching, which is relevant to the claim about the necessity of record linkage for integrating fragmented health data systems in Brazil.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide specific evidence or support for the claim regarding the challenges and costs associated with clerical review in data linkage processes.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract indicates that the cited paper addresses recent research advances and open challenges in data matching, which is relevant to the claim about the application of machine learning techniques, but it does not specifically support the methodological aspects mentioned in the thesis.
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract indicates a focus on data quality and matching, which is relevant to the claim about the challenges of manual review in data processes, but does not explicitly address the artisanal and non-reproducible nature of manual reviews as stated in the thesis sentence.
  - BACKGROUND: L3_PASS [HIGH] — The abstract indicates that the cited paper addresses data matching, which is relevant to the discussion of class imbalance in the context of data quality.

### `Coeli2002blocking`

**Título:** Avalia\cc\~ao de diferentes estrat\'egias de blocagem no relacionamento probabil\'\istico de registros  
**Ano:** 2002  
**DOI:** 10.1590/S0034-89102002000400006  
**L1 (Existência):** L1_FAIL_GHOST (title_sim=43%, year=True, author=False)
**L2 (Journal):** L2_PASS_HIGH (journal=Revista de Saúde Pública, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on hospital admissions among asthmatic children and does not provide any information or evidence related to advancements in text field comparison metrics, blocking strategies, or machine learning approaches as mentioned in the thesis sentence.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not mention any concepts related to the blocking strategies or record linkage processes, which are essential for defining the claim made in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on hospital admissions among asthmatic children and does not address the issue of record linkage or the proportion of true pairs in candidate record comparisons, which is central to the claim made in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on hospital admissions and care for asthmatic children, which does not provide any evidence or support for the probabilistic record linkage methods mentioned in the thesis sentence.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not mention probabilistic linkage protocols or the empirical definition of classification thresholds, which are central to the claim made in the thesis.
  - BACKGROUND: L3_PASS [HIGH] — The cited paper discusses the organization of health care services and their impact on hospital admissions for asthma children, which aligns with the background claim about the experience and capabilities of the IESC in managing health data and methodologies.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not mention anything related to 'blocking steps' or the classification of candidate pairs, which are essential to the definition provided in the thesis sentence.
**Flags:**
  - STALE_REF [MÉDIA]: Published 2002, only 16 citations. Threshold: <2005 AND <200 cit.

### `Fellegi1969theory`

**Título:** A Theory for Record Linkage  
**Ano:** 1969  
**DOI:** 10.1080/01621459.1969.10501049  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Journal of the American Statistical Association, quartile=Q1, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a record linkage study, which aligns with the thesis's mention of record linkage in epidemiological studies, but it does not explicitly support the specific claims about longitudinal follow-up, identification of underreporting, or performance indicators.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses a study on congenital anomalies and breech presentation, which does not provide evidence or support for the claim regarding the formalization of a probabilistic record linkage method by Fellegi and Sunter.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not mention or define the concept of probabilistic linkage or the weighted similarity scores as described in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses congenital anomalies and breech presentation, which does not relate to the probabilistic methods or parameters (m-probability and u-probability) mentioned in the thesis sentence.
  - DATA_SOURCE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses congenital anomalies and breech presentation, which is unrelated to the classification of similarity scores into categories of matches, non-matches, and gray areas as described in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses congenital anomalies and breech presentation, which is unrelated to the classification thresholds and the concept of the 'gray area' in the decision model proposed by Fellegi and Sunter.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses congenital anomalies and breech presentation, which is unrelated to the classification thresholds and 'gray area' concept described in the thesis sentence.
  - DEFINE: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not mention or define the concept of 'thresholds' for classification, which is central to the claim made in the thesis sentence.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on the association between breech presentation and congenital anomalies, and does not provide any evidence or methods related to record linkage or the classification of health data records as true pairs, non-pairs, or doubtful pairs.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper focuses on the association between breech presentation and congenital anomalies, which does not provide evidence or support for the probabilistic record linkage method discussed in the thesis.

### `Hand2018fmeasure`

**Título:** A Note on Using the F-Measure for Evaluating Record Linkage Algorithms  
**Ano:** 2018  
**DOI:** 10.1007/s11222-017-9746-6  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Statistics and Computing, quartile=, evidence=B)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - DEFINE: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the importance of precision and recall in the context of the F-measure but does not explicitly define or elaborate on the prioritization of pairs or non-pairs as described in the thesis sentence.
  - DEFINE: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the importance of precision and recall in the context of the F-measure but does not explicitly define or elaborate on the prioritization of pairs or non-pairs as stated in the thesis sentence.
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the F-measure's conceptual weaknesses, which is relevant to the claim about using different evaluation metrics, but does not explicitly support the claim regarding the appropriateness of AUC-PR for imbalanced classes.
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the F-measure's conceptual weaknesses, which is relevant to the performance metrics mentioned in the thesis, but does not directly support the specific claim about the use of these metrics in evaluating classifiers.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper discusses the conceptual weaknesses of the F-measure but does not provide evidence or support for the effectiveness of the strategy mentioned in the thesis for constructing reliable analytical sets.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the F-measure, which is directly related to the evaluation metrics mentioned in the thesis sentence, thereby supporting the claim about recognized metrics in the literature.

### `Hastie2009elements`

**Título:** The Elements of Statistical Learning: Data Mining, Inference, and Prediction  
**Ano:** 2009  
**DOI:** 10.1007/978-0-387-84858-7  
**L1 (Existência):** L1_WARN_META (title_sim=64%, year=True, author=True)
**L2 (Journal):** L2_NA (journal=Springer Series in Statistics, quartile=, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract of the cited paper discusses statistical learning methods, which may imply the ability to capture complex patterns, but does not explicitly support the specific claim regarding non-linear patterns and comparison variables.
  - SUPPORT_FACT: L3_WARN_PARTIAL [MEDIUM] — The abstract of the cited paper does not explicitly support the specific claim about logistic regression as a generalized linear model estimating binary event probabilities, but it is relevant to the broader context of statistical learning and methods discussed in the paper.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide any specific evidence or details about the sequential nature of Gradient Boosting methods or their mechanism of correcting residual errors, which is essential to support the claim.
  - BACKGROUND: L3_PASS [HIGH] — The abstract of the cited paper discusses the concepts of Support Vector Machines and their ability to find optimal separating hyperplanes, which aligns with the thesis sentence's description of SVMs.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide specific evidence or details regarding the Multilayer Perceptron or its learning process through backpropagation, which is necessary to support the claim made in the thesis.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not provide specific evidence or support for the claim regarding the use of machine learning algorithms as a post-processing layer to improve accuracy in record linkage.
  - BACKGROUND: L3_PASS [HIGH] — The cited paper provides foundational knowledge on statistical learning methods, which supports the background claim regarding the evaluation of various machine learning classifiers and their performance metrics.
  - BACKGROUND: L3_PASS [HIGH] — The cited paper is a foundational text in statistical learning and machine learning, which supports the claim that precision is a recognized metric in the literature of record linkage and machine learning.

### `He2009imbalanced`

**Título:** Learning from Imbalanced Data  
**Ano:** 2009  
**DOI:** 10.1109/TKDE.2008.239  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=IEEE Transactions on Knowledge and Data Engineering, quartile=Q1, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - DEFINE: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses class imbalance and mentions the use of SMOTE to address it, which relates to the claim about severe class imbalance in large datasets, but does not explicitly define or quantify the imbalance ratio mentioned in the thesis.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract mentions the use of the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance, which is relevant to the claim about challenges in classification algorithms due to imbalanced datasets, but it does not explicitly discuss the bias towards the majority class or the implications for model accuracy in identifying true pairs.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract explicitly mentions the use of the Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance, which directly supports the claim regarding combined techniques like SMOTE-Tomek that integrate oversampling with noise removal.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide evidence or discussion regarding the choice of appropriate balancing strategies or the lack of consensus in the literature, which is central to the claim made in the thesis.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper does not address the specific issue of class imbalance in record linkage or the fraction of true pairs in mortality and notification data, which is the focus of the claim.
  - BACKGROUND: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses class imbalance and the use of SMOTE to address it, which is relevant to the claim about severe class imbalance, but does not specifically address the context of the dataset or the specific proportions mentioned in the thesis.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the use of machine learning models and mentions addressing class imbalance, which aligns with the claim about AUC-PR being informative in scenarios of severe class imbalance.

### `Jaro1989advances`

**Título:** Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida  
**Ano:** 1989  
**DOI:** 10.1080/01621459.1989.10478785  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Journal of the American Statistical Association, quartile=Q1, evidence=A)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide specific evidence or details regarding the incorporation of machine learning approaches for automated classification of candidate pairs, which is the focus of the claim.
  - SUPPORT_FACT: L3_FAIL_UNSUPPORTED [HIGH] — The abstract does not provide specific evidence or details regarding the metrics of textual similarity, such as Jaro's distance or Winkler's extension, which are central to the claim made in the thesis.

### `Newcombe1959automatic`

**Título:** Automatic Linkage of Vital Records  
**Ano:** 1959  
**DOI:** 10.1126/science.130.3381.954  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Science, quartile=, evidence=B)
**L3 (Relevância):** L3_FAIL_UNSUPPORTED (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_INCONCLUSIVE [LOW] — The abstract of the cited paper is empty, providing no information to evaluate whether it supports the claim regarding the use of record linkage in epidemiological studies.
  - SUPPORT_METHOD: L3_FAIL_UNSUPPORTED [HIGH] — The abstract of the cited paper is empty, providing no evidence to support the claim regarding the formalization of the method by Fellegi and Sunter.

### `Rocha2015causes`

**Título:** Do que morrem os pacientes com tuberculose: causas m\'ultiplas de \'obito de uma coorte de casos notificados  
**Ano:** 2015  
**DOI:** 10.1590/0102-311X00026614  
**L1 (Existência):** L1_FAIL_GHOST (title_sim=0%, year=False, author=False)
**L2 (Journal):** L2_PASS_HIGH (journal=Cadernos de Sa\'ude P\'ublica, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_NA (abstract=não)

## REF_REVIEW (13 referências)

### `Brasil2024tb`

**Título:** Boletim Epidemiol\'ogico de Tuberculose 2024  
**Ano:** 2024  
**DOI:** N/A  
**L1 (Existência):** L1_WARN_NO_DOI (title_sim=0%, year=False, author=False)
**L2 (Journal):** L2_NA (journal=, quartile=, evidence=C)
**L3 (Relevância):** L3_NA (abstract=não)
**Flags:**
  - MISSING_DOI [MÉDIA]: No DOI in .bib and none resolved via APIs.

### `Chawla2002smote`

**Título:** SMOTE: Synthetic Minority Over-sampling Technique  
**Ano:** 2002  
**DOI:** 10.1613/jair.953  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Journal of Artificial Intelligence Research, quartile=, evidence=B)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract explicitly mentions the use of the SMOTE algorithm for improving the dataset in the context of constructing an early warning model, directly supporting the claim regarding the minority class oversampling strategy.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the use of the SMOTE algorithm in constructing a predictive model, which is relevant to the claim about class balancing strategies, but it does not explicitly evaluate or compare multiple strategies as stated in the thesis.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract mentions the use of the SMOTE algorithm in constructing an early warning model, which aligns with the thesis's focus on class balancing techniques, but it does not provide specific evidence for the combination of methods or classifiers mentioned in the thesis.

### `Chen2016xgboost`

**Título:** XGBoost: A Scalable Tree Boosting System  
**Ano:** 2016  
**DOI:** 10.1145/2939672.2939785  
**L1 (Existência):** L1_WARN_META (title_sim=30%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, quartile=, evidence=B)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the effectiveness of tree boosting and mentions its scalability and efficiency, which aligns with the general capabilities of machine learning methods to incorporate more predictors, but it does not explicitly address the incorporation of derived variables or interactions as stated in the thesis sentence.
  - SUPPORT_FACT: L3_WARN_PARTIAL [MEDIUM] — The abstract mentions that XGBoost is widely used and achieves state-of-the-art results, but it does not explicitly confirm the specific features such as L1 and L2 regularization, handling of missing values, or parallelization mentioned in the thesis.
  - BACKGROUND: L3_PASS [HIGH] — The abstract of the cited paper describes XGBoost as a highly effective machine learning method, which aligns with the thesis's mention of using XGBoost as a classifier for maximizing precision.

### `Enamorado2019fastlink`

**Título:** Using a Probabilistic Model to Assist Merging of Large-Scale Administrative Records  
**Ano:** 2019  
**DOI:** 10.1017/S0003055418000783  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=American Political Science Review, quartile=, evidence=B)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a scalable algorithm for probabilistic record linkage, which aligns with the claim about advancements in methodologies, but it does not explicitly mention machine learning approaches or automated classification of candidate pairs.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a probabilistic record linkage methodology that addresses issues relevant to merging data sets, which aligns with the use of machine learning for automating decision-making in record linkage, but it does not explicitly mention machine learning techniques or their application in classification as stated in the thesis sentence.
  - SUPPORT_METHOD: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses a probabilistic record linkage methodology that addresses issues relevant to the claim about machine learning applications in record linkage, but it does not specifically mention the use of supervised classifiers or the context of Brazilian literature.

### `Oliveira2012uso`

**Título:** Uso do sistema de informa\cc\~ao sobre mortalidade para identificar subnotifica\cc\~ao de casos de tuberculose no Brasil  
**Ano:** 2012  
**DOI:** 10.1590/S1415-790X2012000300003  
**L1 (Existência):** L1_PASS (title_sim=93%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Revista Brasileira de Epidemiologia, quartile=Q3, evidence=B)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract discusses the underreporting of tuberculosis deaths and emphasizes the importance of linking databases to improve the quality of the TB surveillance system, directly supporting the claim about identifying unreported deaths in public health.
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract explicitly discusses the underreporting of tuberculosis deaths and the analysis of data from the Mortality Information System (SIM) and the Reportable Disease Information System (Sinan), directly supporting the claim about identifying underreporting of tuberculosis.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the importance of accurately linking health records, specifically in the context of tuberculosis deaths, which aligns with the claim about the relevance of true pair loss in health record relationships affecting epidemiological study validity.
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract clearly discusses the underreporting of TB deaths and the relationship between the SIM and Sinan databases, which directly supports the claim about identifying unreported TB deaths and highlighting significant underreporting.
  - SUPPORT_FACT: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the Reportable Disease Information System (Sinan) and its role in tuberculosis reporting, but it primarily focuses on underreporting of deaths rather than detailing the specific variables and patient data included in the Sinan-TB system as stated in the thesis.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the relevance of tuberculosis in the context of health care quality and emphasizes the importance of data relationships, aligning well with the thesis's justification for using these data sources.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract discusses the analysis of underreporting of deaths from tuberculosis and emphasizes the importance of identifying true pairs to improve the quality of the TB surveillance system, directly supporting the thesis's focus on maximizing sensitivity to recover true pairs in studies of underreporting.

### `Oliveira2016accuracy`

**Título:** Acur\'acia do relacionamento probabil\'\istico e determin\'\istico de registros: o caso da tuberculose  
**Ano:** 2016  
**DOI:** 10.1590/S1518-8787.2016050006327  
**L1 (Existência):** L1_WARN_META (title_sim=66%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Revista de Saúde Pública, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_PASS (abstract=sim)
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract clearly states that the deterministic method has lower sensitivity due to issues like missing values and low similarity measures, which aligns with the thesis claim about the limitations of deterministic methods in identifying true pairs.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract clearly states that the deterministic method had lower sensitivity due to issues like missing values and low similarity measures, which aligns with the claim about its reduced capacity to recover pairs in the presence of incomplete or erroneous identification fields.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the experience and methodologies used in record linkage, which aligns with the claim about the institutional experience in linking health databases in Brazil.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the relevance of tuberculosis (TB) records and the methods used for record linkage, aligning with the thesis's emphasis on the importance of TB as a marker for healthcare quality.

### `Paim2011brazilian`

**Título:** The Brazilian Health System: History, Advances, and Challenges  
**Ano:** 2011  
**DOI:** 10.1016/S0140-6736(11)60054-8  
**L1 (Existência):** L1_PASS (title_sim=100%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=The Lancet, quartile=Q1, evidence=A)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the historical development and components of the Brazilian health system, including the establishment of the Unified Health System (SUS), which aligns with the thesis sentence about the SUS's role in organizing health services in Brazil.
  - DATA_SOURCE: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the Brazilian health system and its reforms, which provides topical relevance to the Sistema de Informação sobre Mortalidade (SIM) as a data source, but does not specifically mention the SIM or its components.

### `Santos2018factors`

**Título:** Fatores associados \`a subnotifica\cc\~ao de tuberculose a partir do linkage SINAN-AIDS e SINAN-TB  
**Ano:** 2018  
**DOI:** 10.1590/1980-549720180019  
**L1 (Existência):** L1_WARN_META (title_sim=69%, year=True, author=True)
**L2 (Journal):** L2_PASS (journal=Revista Brasileira de Epidemiologia, quartile=Q3, evidence=B)
**L3 (Relevância):** L3_PASS (abstract=sim)
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract provides evidence of a study that identified underreporting of tuberculosis, directly supporting the claim made in the thesis.
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract discusses the TB Notification System (Sinan TB) as a data source for estimating underreporting of tuberculosis cases, which directly supports the claim about the system's role in registering tuberculosis cases in Brazil.

### `Sousa2011obitos`

**Título:** \'Obitos e interna\cc\~oes por tuberculose n\~ao notificados no munic\'\ipio do Rio de Janeiro  
**Ano:** 2011  
**DOI:** 10.1590/S0034-89102011000100004  
**L1 (Existência):** L1_WARN_META (title_sim=78%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Revista de Saúde Pública, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_PASS (abstract=sim)
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract discusses the identification of unreported tuberculosis deaths and hospitalizations, directly supporting the claim about the potential of record linkage in public health to identify unreported deaths.
  - SUPPORT_FACT: L3_PASS [HIGH] — The abstract provides evidence of significant underreporting of tuberculosis deaths and hospitalizations, directly supporting the claim about the relationship between SIM and Sinan-TB for identifying unreported tuberculosis deaths.
  - SUPPORT_METHOD: L3_PASS [HIGH] — The abstract discusses the issue of underreporting in tuberculosis deaths and hospitalizations, which directly supports the claim regarding the importance of identifying true pairs to avoid underestimation of adverse outcomes in health system performance evaluation.

### `Viacava2012avaliacao`

**Título:** Avalia\cc\~ao de desempenho de sistemas de sa\'ude: um modelo de an\'alise  
**Ano:** 2012  
**DOI:** 10.1590/S1413-81232012000400014  
**L1 (Existência):** L1_PASS (title_sim=89%, year=True, author=True)
**L2 (Journal):** L2_PASS_HIGH (journal=Ciência &amp; Saúde Coletiva, quartile=Q2, evidence=A)
**L3 (Relevância):** L3_WARN_PARTIAL (abstract=sim)
  - SUPPORT_FACT: L3_WARN_PARTIAL [MEDIUM] — The abstract discusses the evaluation of health service performance and the improvement of indicators, which aligns with the thesis's claim about evaluating health systems, but it does not specifically address the identification of unreported tuberculosis deaths or the improvement of data quality.
  - BACKGROUND: L3_PASS [HIGH] — The abstract discusses the evaluation of health service performance, which aligns with the need for timely and reproducible data in health decision-making during health crises, as stated in the thesis sentence.

### `WHO2024tb`

**Título:** Global Tuberculosis Report 2024  
**Ano:** 2024  
**DOI:** N/A  
**L1 (Existência):** L1_WARN_NO_DOI (title_sim=0%, year=False, author=False)
**L2 (Journal):** L2_NA (journal=, quartile=, evidence=C)
**L3 (Relevância):** L3_NA (abstract=não)
**Flags:**
  - MISSING_DOI [MÉDIA]: No DOI in .bib and none resolved via APIs.

### `Winkler1990string`

**Título:** String Comparator Metrics and Enhanced Decision Rules in the Fellegi--Sunter Model of Record Linkage  
**Ano:** 1990  
**DOI:** N/A  
**L1 (Existência):** L1_PASS (title_sim=99%, year=True, author=True)
**L2 (Journal):** L2_NA (journal=, quartile=, evidence=C)
**L3 (Relevância):** L3_NA (abstract=não)
**Flags:**
  - MISSING_DOI [MÉDIA]: No DOI in .bib and none resolved via APIs.

### `silva2012linkage`

**Título:** Linkage de Bases de Dados Identificadas em Saúde: Consentimento, Privacidade e Segurança da Informação  
**Ano:** 2012  
**DOI:** N/A  
**L1 (Existência):** L1_WARN_NO_DOI (title_sim=0%, year=False, author=False)
**L2 (Journal):** L2_NA (journal=, quartile=, evidence=C)
**L3 (Relevância):** L3_NA (abstract=não)
**Flags:**
  - MISSING_DOI [MÉDIA]: No DOI in .bib and none resolved via APIs.

## REF_SOLID + REF_OK (2 referências)

| Chave | Score | L1 | L2 | L3 |
|-------|-------|----|----|----|
| `Bartholomay2014improved` | REF_OK | L1_PASS | L2_PASS_HIGH | L3_NA |
| `Macinko2015family` | REF_SOLID | L1_PASS | L2_PASS_HIGH | L3_PASS |
