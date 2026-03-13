# Notebook Template: Modelagem, Feature Engineering e Avaliacao

Use este template quando o objetivo for gerar um notebook profissional de modelagem como artefato principal da implementacao.

## Regra principal

- Antes de implementar, o agent deve apresentar um plano detalhado com todas as etapas, secoes, artefatos e validacoes que serao criados.
- O artefato principal desta entrega deve ser `notebooks/02_modeling_evaluation.ipynb`.
- Esse nome deve ser usado apenas quando a entrega for especificamente um notebook de Modelagem, Feature Engineering e Avaliacao.
- Para outros tipos de notebook, manter a pasta `notebooks/`, mas escolher um nome coerente com o objetivo da entrega.
- O notebook pode ser criado manualmente ou programaticamente com a biblioteca `nbformat`, que e a ferramenta oficial do ecossistema do IPython/Jupyter para ler, escrever e manipular a estrutura interna dos arquivos `.ipynb`.
- O notebook final deve conter tanto a logica de modelagem quanto a narrativa, segmentacao e explicacao.
- Os insights devem ser claros, objetivos e baseados na execucao real do codigo implementado.
- Sempre que possivel, citar numeros, rankings, thresholds e diferencas que realmente apareceram nos resultados gerados.
- O notebook deve ser bem comentado e bem dividido em celulas, com markdown bem formatado e explicativo.

## Sequencia recomendada

1. Titulo e descricao macro do notebook
2. Plano de implementacao
3. Imports
4. Variaveis e configuracoes
5. Funcoes e blocos do pipeline
6. Contexto e leitura do dataset
7. Hipotese de feature engineering
8. Feature notes detalhadas
9. Validacao das features por correlacao e multicolinearidade
10. Execucao completa do pipeline
11. Resultados tabulares detalhados
12. Graficos com interpretacao detalhada
13. Conclusao
14. Resumo executivo
15. Validacao final do notebook

## Estrutura esperada

### 1. Titulo e descricao macro

- Titulo claro, por exemplo: `Modelagem, Feature Engineering e Avaliacao`
- Um paragrafo descrevendo o que o notebook cobre de forma ampla
- Explicitar os blocos principais do notebook

### 2. Plano de implementacao

- Logo no inicio do notebook, incluir uma secao em markdown com o plano detalhado do que sera implementado
- Explicitar etapas, artefatos esperados, modelos a comparar, metricas e validacoes
- O plano deve servir como contrato do notebook antes da execucao tecnica

### 3. Imports

- Uma celula exclusiva para imports
- Manter imports organizados e legiveis

### 4. Variaveis e configuracoes

- Uma celula separada para constantes, paths e configuracoes globais
- Exemplo: `RANDOM_STATE`, paths, pastas de artefatos

### 5. Funcoes e blocos do pipeline

- Uma ou mais celulas para as funcoes e blocos que implementam o pipeline no proprio notebook
- Comentarios uteis seguindo o padrao de qualidade de codigo sobre o objetivo de cada bloco de funcao

### 6. Contexto e leitura do dataset

- Explicar de onde vem o dataset
- Mostrar shape e distribuicao da target
- Explicar rapidamente por que esse dataset e o ponto de entrada da modelagem

### 7. Hipotese de feature engineering

- Explicar os sinais vindos da EDA
- Descrever por que as features derivadas fazem sentido
- Conectar feature engineering com comportamento, operacao e negocio

### 8. Feature notes detalhadas

Criar uma secao em markdown preferencialmente em formato de tabela com as colunas:

| Feature | Categoria | O que mede | Regra de negocio | Hipotese de impacto no churn |
|---|---|---|---|---|

Orientacao para preenchimento:

- `Feature`: nome tecnico da feature derivada
- `Categoria`: tipo de sinal que a feature representa, como intensidade de uso, monetizacao, qualidade operacional, relacionamento ou eficiencia
- `O que mede`: descricao objetiva e curta da medida
- `Regra de negocio`: explicacao da logica de negocio e do significado analitico da feature
- `Hipotese de impacto no churn`: justificativa de por que a feature pode melhorar explicacao ou previsao de churn

### 9. Validacao das features

Adicionar uma secao chamada:

`Validacao das Features (Correlacao e Multicolinearidade)`

Essa secao deve conter:

- correlacao das features com a target
- matriz de correlacao cruzada
- tabela com pares de maior correlacao absoluta
- explicacao do que isso significa
- ressalva de que correlacao nao substitui teste em modelo

### 10. Execucao completa do pipeline

- Separar `X` e `y`
- Construir modelos
- Rodar cross-validation
- Rodar holdout
- Calcular feature importance
- Comparar raw vs engineered
- Calcular threshold trade-off
- Salvar artefatos

### 11. Resultados tabulares

Incluir tabelas para:

- benchmark em cross-validation
- benchmark em holdout
- impacto do feature engineering
- As interpretacoes dessas tabelas devem refletir exatamente os resultados executados, sem antecipar conclusoes nao observadas

### 12. Graficos com interpretacao

Cada grafico deve vir com markdown proprio contendo:

- o que o grafico mostra
- por que ele e importante
- quais insights ele gera
- Os insights precisam ser extraidos do grafico efetivamente gerado, mencionando valores, ranking dos modelos, formatos das curvas, threshold ou sinais relevantes quando aplicavel

Graficos recomendados:

- impacto do feature engineering
- comparacao de metricas entre modelos
- curvas ROC
- curvas Precision-Recall
- matriz de confusao
- importancia das features
- trade-off de threshold

### 13. Conclusao

- Retomar a recomendacao tecnica
- Reforcar os principais sinais encontrados
- Indicar proxima evolucao tecnica

### 14. Resumo executivo

Deve vir depois da conclusao tecnica.

Conteudo esperado:

- narrativa executiva
- quadro-resumo do projeto
- scorecard elegante dos modelos
- principais drivers do churn
- leitura final para negocio
- O resumo executivo deve ser consistente com a conclusao tecnica e com os resultados efetivamente observados

### 15. Validacao final do notebook

Antes de considerar o notebook concluido, fazer uma revisao final para validar:

- se as metricas citadas no texto batem com as tabelas geradas
- se a interpretacao dos graficos bate com os graficos efetivamente exibidos
- se o modelo recomendado bate com o benchmark executado
- se os insights de feature engineering batem com a validacao feita nas features
- se a conclusao tecnica nao contradiz o resumo executivo
- se nao existe narrativa generica ou hipotetica sem suporte nos outputs produzidos

## Padrao de comentarios no codigo

- Comentarios devem explicar intencao, nao obviedades
- Comentar blocos importantes: carga do dataset, feature engineering, benchmark, holdout, threshold, persistencia
- Evitar comentario por linha sem necessidade

## Regra de qualidade

- Notebook bem segmentado
- Notebook explicativo
- Notebook profissional
- Notebook como fonte principal da implementacao
- Notebook com linguagem clara para publico tecnico e de negocio
- Texto de resultados sempre ancorado na execucao real
- Revisao final obrigatoria de coerencia entre codigo, tabelas, graficos e narrativa
