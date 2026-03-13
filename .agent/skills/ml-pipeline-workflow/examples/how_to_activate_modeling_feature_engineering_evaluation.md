# Exemplo: Como Ativar Esta Skill da Melhor Forma

Use este exemplo quando quiser que o agente produza um fluxo de alta qualidade para Modelagem, Feature Engineering e Avaliacao a partir de um dataset existente ou de uma EDA ja realizada.

## Objetivo

O objetivo deste prompt e ativar a skill com contexto, restricoes e expectativa de qualidade suficientes para que o agente:

- comece com um plano de implementacao
- leia a EDA anterior antes de modelar
- implemente primeiro o pipeline fonte de verdade
- gere um notebook profissional depois que a implementacao estiver validada
- escreva insights com base na execucao real, e nao em suposicoes genericas
- faca uma revisao final de coerencia do notebook

## Prompt recomendado para ativacao

```text
Use a skill `ml-pipeline-workflow`.

Quero um fluxo completo de Modelagem, Feature Engineering e Avaliacao para este dataset/projeto.

Contexto:
- Revise a EDA existente, os notebooks anteriores e o dataset limpo antes de propor a modelagem.
- Trate o trabalho exploratorio anterior como contexto obrigatorio.
- Considere que a entrega precisa ser tecnicamente rigorosa, reproduzivel e adequada para portfolio e apresentacao a stakeholders.

Requisitos de implementacao:
- Comece criando um plano detalhado de implementacao.
- Use esse plano para definir as etapas da modelagem, as features propostas, os passos de validacao, os artefatos e as secoes do notebook.
- Implemente primeiro o pipeline de modelagem como fonte de verdade, seguindo boas praticas de engenharia.
- Utilize pelo menos 5 modelos relevantes, quando fizer sentido.
- Use metricas adequadas ao problema, especialmente metricas apropriadas para desbalanceamento em classificacao.
- Valide com cross-validation e holdout.
- Compare features originais versus features derivadas quando feature engineering for parte central da tarefa.
- Gere graficos, artefatos estruturados e uma recomendacao final.

Requisitos do notebook:
- Depois que a implementacao estiver pronta, gere um notebook em `notebooks/` seguindo o template da skill.
- O notebook deve ser segmentado, bem comentado e bem explicado.
- Inclua feature notes detalhadas com regra de negocio e racional de modelagem.
- Inclua uma secao de validacao das features cobrindo correlacao e multicolinearidade.
- Para cada grafico principal, explique o que ele mostra, por que ele importa e qual insight ele gera.
- Todos os textos de resultados devem ser baseados nos outputs realmente executados.
- Depois da conclusao, adicione um resumo executivo com tabelas-resumo elegantes.
- Antes de finalizar, faca uma validacao completa do notebook para confirmar que metricas, graficos, conclusoes e resumo executivo estao coerentes.

Nivel de qualidade:
- Nao invente conclusoes sem suporte no codigo executado.
- Nao pule a etapa de plano de implementacao.
- Nao pare em recomendacoes genericas.
- Faça a entrega final em nivel profissional, adequada para avaliacao tecnica e apresentacao de negocio.
```

## Por que esta ativacao funciona bem

Essa estrutura e forte porque da ao agente:

- o gatilho explicito para usar a skill
- o contexto do projeto antes do inicio da modelagem
- a obrigacao de planejar antes de implementar
- a expectativa de que a modelagem seja executada, e nao apenas proposta
- a expectativa de que o notebook seja uma entrega polida, e nao apenas um rascunho
- a regra de que os insights precisam vir dos resultados executados
- a exigencia de uma checagem final de qualidade do notebook

## Versao curta

Se quiser um prompt menor, use este:

```text
Use a skill `ml-pipeline-workflow`.

Leia a EDA anterior e o dataset, crie um plano de implementacao e depois construa um fluxo completo de Modelagem, Feature Engineering e Avaliacao com pelo menos 5 modelos, metricas adequadas, graficos, validacao de features e um notebook profissional em `notebooks/`.

Todos os insights devem vir dos resultados executados, e o notebook deve ser validado ao final para garantir coerencia entre os outputs do codigo, os graficos e a narrativa.
```

## Melhores informacoes para o usuario fornecer

Sempre que possivel, inclua:

- o caminho para o notebook anterior de EDA
- o caminho para o dataset limpo
- a variavel target
- se o problema e classificacao ou regressao
- restricoes de negocio relacionadas a precision, recall, explicabilidade ou deploy
- se o notebook deve ser orientado a portfolio, negocio ou investigacao tecnica

## Comportamento esperado do agente apos a ativacao

Depois de ativada corretamente, a skill deve levar o agente a:

1. revisar a EDA existente ou o dataset limpo
2. apresentar um plano detalhado de implementacao
3. implementar o fluxo de modelagem de ponta a ponta
4. executar e validar os resultados
5. gerar o notebook com narrativa clara e interpretacao dos graficos
6. fazer uma revisao final de coerencia do notebook antes de encerrar
