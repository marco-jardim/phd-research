# Segundo review: agregados e limite do dossiê histórico

## D16: checkpoints privados da ablação

`scripts/ablacao_dossie_json_agregado.py` grava contagens por braço e SHA-256
dos bytes originais. `--remove-tracked` retira os checkpoints do índice, mas
preserva os arquivos locais ignorados. A reprodução dos hashes exige os
originais privados byte a byte; os agregados não permitem reconstituir decisões
pareadas. As transições históricas agregadas continuam em
`data/sprint3b/ablacao/resumo.json`. Não houve reescrita de histórico.

## Fronteira dos campos

`scripts/dossie_campos_enviados.py` usa somente o cabeçalho do extrato, as
auditorias v4 dos dois modos e uma linha sintética para verificar a interface.
A saída `data/dossie_campos_enviados.csv` contém os 29 subescores, categoria e
inclusão. O gerador também produz a tabela editorial correspondente.

Os campos de nome composto e componentes separados da data que não existem no
extrato foram retirados da lista do construtor. Não se acrescentaram campos:
continuam 14 subescores, além dos compostos, indicadores e nota final já usados.
O subescore municipal é um valor escalar chamado `municipality`; por isso seu
nome original não aparece como chave na auditoria do JSON. O teste de fronteira
cobre os 29 campos reais e os 14 efetivamente enviados.

## Taxa-base não identifica mecanismo

`scripts/ablacao_dossie_taxa_base.py` separa seis coortes históricas: dois
modelos, com saídas do motor v3, sem essas saídas v3 e sem elas v4.
O denominador de cada linha é 1.845 revisões (modo, par), não 1.845 pares únicos:
um mesmo par pode ter sido revisto em vigilância e confirmação. Não entram
suplementos nem coortes de fechamento.

As contagens das 18 decisões e as transições históricas são apresentadas ao
lado das taxas-base, mas não se afirma equivalência entre a retirada de
`model_outputs` e a retirada adicional dos proxies no braço `sem_pcal`.
Não existe execução de `sem_pcal` na coorte completa. A taxa-base não é um
controle negativo e não distingue ancoragem de abstenção generalizada.

## Sensibilidade e exclusões de denominador

`scripts/equidade_passo_revisor_perfeito.py` usa os agregados de equidade
versionados. Para cada decisão e passo, a sensibilidade automática é
`tp/n_true`; o limite superior com revisor perfeito é
`(tp+n_humano_true)/n_true`. O segundo valor não é desempenho humano observado.
O gerador exige o fechamento dos vínculos e que os encaminhados verdadeiros
sejam um subconjunto dos falsos negativos automáticos.

`data/equidade_arranjo_den30_excluidos.csv` conserva todas as comparações
excluídas por `n1<30` ou `n2<30`, com os denominadores, numeradores e taxas
originais. Não se reaplica BH aos excluídos nem se chama ausência na família
de evidência de paridade. Passo é estrato operacional, não atributo demográfico.

## Configuração e reprodução histórica

`gzcmd_v3_config.yaml` não é um retrato único de P. Seleciona `ml_xgb` e
`policy_version=v3.1`, mas conserva campos descritivos/defaults de outros
perfis. O arranjo P é construído pelo protocolo e código específicos descritos
em `M_arranjo_corte.md`, com recalibração abaixo do corte e taxas cross-fitted;
não se obtém P apenas trocando o YAML. Seus parâmetros não devem ser
confundidos com as sementes e partições das avaliações históricas da base
inteira. A seção `consistency`, inclusive `conflict_margin_delta`, é reservada,
não implementada; não há restrição de cardinalidade N:1.

Os valores de geração Haiku do YAML (temperature 0,0 e max_tokens 500) não
documentam as chamadas do experimento descritas no E4, que omitem temperature
e usam max_tokens 2.500. O YAML histórico não foi regravado nesta etapa para
não invalidar silenciosamente os fingerprints congelados. A criação de um
perfil P executável e a conciliação de seus metadados permanecem pendentes.

Um checkout novo não recupera os bytes mistos do pré-registro da S1;
`S10_eol_prereg.md` é a referência para esse impedimento. Nenhum hash congelado
ou pré-registro foi atualizado para contornar o gate.
