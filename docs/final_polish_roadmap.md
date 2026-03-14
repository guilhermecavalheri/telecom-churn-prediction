# Roadmap de Acabamento Final

Este documento consolida os pontos de maior impacto para levar o projeto a um nivel final de portfolio, com foco em clareza executiva, reproducibilidade e demonstracao de maturidade em ML end-to-end.

## Objetivo

O projeto ja cobre o nucleo tecnico necessario para um bom portfolio de Machine Learning:

- EDA e limpeza
- feature engineering guiado por negocio
- benchmark de modelos
- persistencia do melhor pipeline
- inferencia reutilizavel
- API de scoring
- banco operacional local
- monitoramento e alerta de drift
- documentacao tecnica

O objetivo deste roadmap e atacar os acabamentos que mais aumentam percepcao de profissionalismo sem inflar desnecessariamente o escopo.

## Prioridade alta

### 1. README com leitura executiva

Status: implementado

Melhorar o topo do projeto para responder rapidamente:

- qual problema de negocio esta sendo resolvido
- qual modelo foi escolhido
- qual foi o desempenho final
- por que este projeto e relevante como exemplo de ML end-to-end

Valor para portfolio:

- acelera entendimento por gestores
- melhora leitura por recrutadores
- reduz atrito para avaliadores tecnicos

### 2. Secao de limitacoes e proximos passos

Status: implementado

Documentar explicitamente:

- o que foi implementado
- o que foi simplificado para portfolio
- o que seria a evolucao natural para producao real

Valor para portfolio:

- demonstra maturidade
- evita vender o projeto como "producao completa" quando nao e
- mostra senso critico de engenharia

### 3. Teste de integracao end-to-end

Criar um teste que cubra, em uma unica trilha:

1. treino e persistencia do melhor modelo
2. carregamento do bundle
3. previsao em lote
4. gravacao no banco local
5. rodada de monitoramento

Valor para portfolio:

- prova que o fluxo completo funciona
- reforca confiabilidade
- demonstra pensamento de engenharia alem do notebook

## Prioridade media

### 4. Health check mais robusto

Status: implementado

Expandir a API para distinguir:

- servico vivo
- servico pronto para operar

O endpoint mais robusto deve verificar, no minimo:

- modelo carregado
- metadata carregada
- acesso ao banco local
- tabelas operacionais disponiveis
- versao do modelo ativa

Valor para portfolio:

- mostra preocupacao com readiness
- aproxima o projeto de uma operacao real

### 5. Exemplo operacional de request e response

Status: implementado

Adicionar ao README e possivelmente ao notebook operacional:

- payload valido para `/predict`
- resposta esperada
- exemplo de `batch`

Valor para portfolio:

- torna o projeto mais concreto
- facilita teste rapido por avaliadores

### 6. Camada executiva de observabilidade

Status: parcialmente implementado

Ja existe uma base para isso em `notebooks/04_duckdb_and_api_operations_demo.ipynb`, com consultas ao banco operacional, leitura de logs, alertas e consumo da API. O ponto pendente aqui e transformar essa camada em uma leitura mais executiva, com indicadores consolidados e narrativa voltada a acompanhamento de operacao.

Consolidar uma visao curta da operacao, por exemplo em notebook ou documento:

- versao atual do modelo
- volume recente de previsoes
- score medio por origem
- alertas mais recentes
- features com maior PSI

Valor para portfolio:

- conecta tecnicidade com leitura de negocio
- ajuda a demonstrar valor do monitoramento

## Prioridade baixa

### 7. Containerizacao

Adicionar:

- `Dockerfile`
- opcionalmente `docker-compose.yml`

Valor para portfolio:

- melhora reproducibilidade
- facilita demonstracao do projeto

### 8. Fluxo de batch scoring dedicado

Criar um script explicito para ler um arquivo, prever e salvar resultados.

Valor para portfolio:

- mostra um uso mais real de inferencia fora do notebook e da API

## Itens que nao sao obrigatorios neste momento

Para este portfolio, os itens abaixo sao interessantes, mas nao essenciais agora:

- deploy em cloud
- CI/CD completo
- autenticacao da API
- retreinamento automatico
- scheduler de monitoramento
- feature store

## Recomendacao de ordem de implementacao

1. teste de integracao end-to-end
2. dashboard ou notebook executivo de observabilidade
3. containerizacao
4. fluxo de batch scoring dedicado

## Resultado esperado

Ao concluir esse roadmap, o projeto tende a passar uma imagem mais forte para liderancas tecnicas e executivas, porque deixa de ser apenas um bom projeto de modelagem e passa a se apresentar como uma solucao de ML organizada, auditavel, observavel e bem comunicada.
