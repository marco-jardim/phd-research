# M5/M5b/M7 reprocessados por REFREC

OOF aninhado agrupado por REFREC, com cinco folds externos e cinco internos, seed 42. A margem bruta exigiu reconstruir os classificadores; a isotônica foi comparada com o cache existente.
Diferença máxima na reprodução da isotônica: 1.11022302463e-16.
As alternativas globais por momentos não são calibrações inteiramente aninhadas; o agrupamento das margens não elimina essa limitação.
## Cinco calibrações na área cinza

21.620 potenciais pares e 151 vínculos. ECE usa dez intervalos uniformes. Mediana é calculada nos vínculos verdadeiros.
```
        calibracao       brier      ece_10  mediana_vinculos  high_predicted  high_observed  success_joint
        platt_nota 0.006149922 0.000741802       0.026336722             NaN            NaN          False
        ancoras_ml 0.002654122 0.002536587       0.999721374     0.994557436    0.846774194          False
      beta_ancoras 0.002869136 0.002883903       0.999999050     0.996232150    0.828125000          False
beta_ancoras_cinza 0.002935882 0.002919899       0.999966212     0.997720481    0.840000000          False
      isotonica_ml 0.002205761 0.001125128       0.894736842     0.988456948    0.973333333           True
```
## Vereditos
```json
{
  "M5_row": {
    "brier_proximo": true,
    "ece_proximo": false,
    "sucesso_conjunto": false
  },
  "M5_refrec": {
    "brier_proximo": true,
    "ece_proximo": false,
    "sucesso_conjunto": false
  },
  "M5b_row": [
    {
      "calibracao": "beta_ancoras",
      "success_joint": false
    },
    {
      "calibracao": "beta_ancoras_cinza",
      "success_joint": false
    }
  ],
  "M5b_refrec": [
    {
      "calibracao": "beta_ancoras",
      "success_joint": false
    },
    {
      "calibracao": "beta_ancoras_cinza",
      "success_joint": false
    }
  ]
}
```
M5 permanece sem sucesso conjunto: Brier dentro da tolerância e ECE fora dela. As duas variantes Beta continuam sem atender ao critério conjunto. Não se recomenda substituir a isotônica com base nesses ajustes.
## M7 sem chamadas LLM
```
                  model         auc          ap       brier
             programado 0.975158391 0.805358493 0.002965677
programado_platt_direto 0.975158391 0.805358493 0.005470316
                   nota 0.873156860 0.216327906 0.007573945
           logistica_l2 0.991479166 0.866642084 0.002793589
      xgb_oof_isotonica 0.995142233 0.862261947 0.002205761
```
Escore existente mantido fixo, sem geração, refinamento ou nova seleção. O hash do módulo e o hash do ledger API foram conferidos. O novo resumo registra zero chamadas LLM. Transporte por PASSO não foi reexecutado neste recálculo; os artefatos anteriores ficam preservados por linha.
## Texto, tabelas e PDFs

As três marcas editoriais foram substituídas em capitulo_gzcmd.tex, capitulo8.tex e na tabela Beta, com identificação da partição. O limiar de ECE da seção Beta foi atualizado para 0,001350. As tabelas M5 e Beta foram regeneradas.
Build executado: python scripts/build_pdf.py --jobname tese_refrec2 --outname tese_refrec2. Zero erros e zero citações/referências indefinidas no log final. Há 14 avisos Overfull hbox, que não foram classificados como erros nem ocultados.
Os dois tese.pdf foram substituídos por PDFs idênticos, de 890565 bytes, SHA256 71dfab83f9456263d625477c70801cc1c36a71071e4a3be4f46672f623adf061.
Removidos 130 artefatos de builds anteriores e 12 marcas do build final. Fontes .tex e os dois tese.pdf foram excluídos da rotina de limpeza.
As 23 versões por linha foram preservadas com SHA256 em data/m5_m7_preservados_row.json. As tabelas históricas são cópias literais _row.tex.txt em data/, fora da árvore LaTeX ativa, preservando seus bytes e as antigas marcas editoriais sem deixá-las em .tex ativos.
rg -n "a confirmar" text/latex --glob '*.tex': sem ocorrências, exit 1.
## Testes
```text
python -m pytest tests -q
203 passed, 2 warnings in 15.59s
```
Avisos do statsmodels no teste M8 de célula ausente (divisão por zero e separação perfeita), sem supressão.
## Arquivos de implementação

scripts/validar_ancoras_ml.py; scripts/validar_ancoras_beta.py; scripts/m7_common.py; scripts/m7_escore_programado.py; scripts/preservar_m5_m7_row.py; scripts/finalizar_pdf_refrec.py; scripts/relatorio_refrec_ancoras_m7.py; .gitignore; gzcmd/M_refrec_ancoras_m7.md.
Testes: tests/test_validar_ancoras_ml.py; tests/test_m7_escore_programado.py; tests/test_finalizar_pdf_refrec.py.
Sem commit, sem chamadas LLM e sem persistência de novos campos pessoais.
