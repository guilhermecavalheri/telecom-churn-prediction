# Referencia Operacional de Predicao e Monitoramento

Este documento explica, em detalhe, como o fluxo operacional do projeto funciona depois que o modelo ja foi treinado e persistido. O foco aqui nao e a etapa de modelagem, e sim o que acontece quando o modelo passa a ser usado para inferencia, registro em banco, monitoramento de drift e geracao de alertas.

O objetivo deste material e permitir que qualquer pessoa que continue o projeto entenda:

- como as previsoes sao geradas
- como os logs operacionais sao gravados
- como a amostra de entradas vai para o banco
- como o monitoramento calcula estatisticas
- quando uma mudanca vira alerta
- quais arquivos participam de cada etapa

## Visao geral do fluxo

O fluxo operacional atual esta separado em quatro responsabilidades:

1. `src/modules/predict.py`
   - valida os dados brutos de entrada
   - aplica o feature engineering necessario para inferencia
   - executa o pipeline salvo
   - gera o resultado de previsao

2. `src/modules/ops_store.py`
   - persiste logs de previsao
   - persiste amostras dos dados de entrada
   - persiste snapshots de monitoramento
   - persiste alertas de drift

3. `src/modules/monitoring.py`
   - compara uma base de referencia com uma base atual
   - calcula PSI e estatisticas descritivas
   - classifica severidade de drift
   - gera alertas quando necessario

4. `src/drift_demo.py`
   - orquestra uma simulacao completa
   - gera dados sinteticos baseline e drifted
   - roda previsoes nos dois cenarios
   - registra os resultados no banco
   - dispara o monitoramento

## Glossario rapido dos termos em ingles

- `registry`: registro oficial ou cadastro
- `log`: historico de eventos gravados
- `payload`: dados efetivamente enviados em uma requisicao
- `snapshot`: foto ou instantaneo de um estado em um momento especifico
- `drift`: mudanca de distribuicao dos dados ao longo do tempo
- `score`: probabilidade prevista pelo modelo
- `threshold`: limite usado para transformar probabilidade em classe
- `severity`: severidade ou gravidade
- `bundle`: pacote do modelo salvo com contexto e metadados
- `baseline`: referencia inicial para comparacao
- `current`: conjunto atual observado

## Fluxo de predicao

### Entrada dos dados

Os dados podem chegar por dois caminhos principais:

- `api/app.py`
  - via endpoint `/health` para liveness
  - via endpoint `/ready` para readiness operacional
  - via endpoint `/predict`
  - via endpoint `/predict/batch`
- `src/drift_demo.py`
  - para simulacao operacional com dados sinteticos

Em ambos os casos, o fluxo de inferencia converge para `src/modules/predict.py`.

### Diferenca entre `health` e `ready`

O endpoint `GET /health` responde apenas se o servico esta vivo. Ele funciona como uma verificacao simples de liveness.

O endpoint `GET /ready` responde se o servico esta realmente pronto para operar. Na implementacao atual, ele verifica:

- se o bundle do modelo foi carregado
- se a metadata do modelo foi carregada
- qual versao do modelo esta ativa
- se o banco local pode ser acessado
- se as tabelas operacionais esperadas existem

Essa separacao e importante porque uma API pode estar no ar, mas ainda assim nao estar pronta para servir previsoes com seguranca.

### Validacao da entrada

Arquivo responsavel: `src/modules/schema.py`

Funcao principal: `validate_input_dataframe(...)`

Essa validacao faz quatro verificacoes principais:

- o `DataFrame` nao pode estar vazio
- todas as colunas obrigatorias precisam existir
- colunas inesperadas sao rejeitadas, salvo configuracao contraria
- valores nulos e negativos em campos criticos sao rejeitados

Tambem existe uma camada de padronizacao numerica:

- colunas esperadas sao convertidas para tipo numerico
- se a conversao falhar, a entrada e rejeitada

Essa etapa e importante porque evita previsao com payload quebrado, schema incompleto ou tipos invalidos.

### Geração das features de inferencia

Arquivo responsavel: `src/modules/predict.py`

Funcao principal: `prepare_inference_features(...)`

Depois da validacao:

1. o `DataFrame` bruto de entrada e validado
2. o `engineer_features(...)` de `src/churn_modeling.py` e aplicado
3. as colunas finais sao alinhadas com `feature_columns` salvas no bundle do modelo

Se alguma feature esperada estiver ausente depois do feature engineering, a inferencia e interrompida com erro.

### Geracao da previsao

Arquivo responsavel: `src/modules/predict.py`

Funcao principal: `predict_dataframe(...)`

Essa funcao:

1. carrega o modelo salvo, caso ele nao tenha sido passado em memoria
2. gera o `feature_df`
3. recupera o `threshold` salvo no bundle
4. executa `pipeline.predict_proba(feature_df)[:, 1]`
5. transforma probabilidade em classe com:
   - `predicted_label = (score >= threshold).astype(int)`
6. monta um `DataFrame` final com colunas operacionais

Esse `DataFrame` de previsao e a base para o log operacional.

## Como a tabela `prediction_logs` e preenchida

Arquivo responsavel pela persistencia: `src/modules/ops_store.py`

Funcao principal: `log_predictions(...)`

O fluxo e:

1. `predict_dataframe(...)` gera `predictions_df`
2. o chamador passa esse `predictions_df` junto com o `input_df` original para `log_predictions(...)`
3. a funcao grava uma versao completa da previsao em `prediction_logs`

Os campos inseridos em `prediction_logs` sao:

- `prediction_timestamp_utc`
  - timestamp UTC da previsao
  - gerado em `predict.py`
- `request_id`
  - identificador unico da linha prevista
  - gerado em `predict.py` com UUID se nao vier informado
- `source`
  - origem da previsao
  - definida pelo chamador, como `api_single`, `api_batch`, `synthetic_baseline` ou `synthetic_drift`
- `model_name`
  - nome do algoritmo salvo no bundle
- `model_version`
  - versao do modelo salva no bundle
- `score`
  - probabilidade prevista de churn
- `predicted_label`
  - classe final apos aplicar o threshold
- `threshold`
  - limiar usado na decisao

Esses campos sao importantes porque permitem auditoria completa da saida operacional do modelo.

## Como a tabela `prediction_inputs_sample` e preenchida

Essa e uma das partes que mais geram duvida.

A tabela `prediction_inputs_sample` nao recebe todos os inputs sempre. Ela recebe uma amostra simples do lote de entrada. Hoje, essa amostra e formada pelas primeiras linhas do lote.

Arquivo responsavel: `src/modules/ops_store.py`

Funcao principal: `log_predictions(...)`

### Passo a passo do que o codigo faz

Depois de gravar `prediction_logs`, a funcao faz:

1. `payload_rows = input_df.head(sample_size).copy()`
   - pega as primeiras linhas do lote de entrada
   - por padrao, `sample_size = 20`

2. adiciona os identificadores operacionais correspondentes:
   - `request_id`
   - `prediction_timestamp_utc`

3. transforma cada linha em JSON:
   - remove colunas auxiliares adicionadas para ligacao
   - converte a linha para dicionario
   - serializa em string JSON

4. reduz a estrutura final para:
   - `prediction_timestamp_utc`
   - `request_id`
   - `payload_json`

5. insere em `prediction_inputs_sample`

### Por que esse desenho foi adotado

Essa tabela foi desenhada para auditoria leve e portfolio, nao para armazenar todo o historico bruto em grande escala.

As vantagens desse desenho sao:

- cria rastreabilidade entre input e output
- permite investigar exemplos reais de payload
- evita crescimento excessivo do banco
- mantem a demonstracao enxuta e profissional

### O que isso significa na pratica

Cada previsao vai para `prediction_logs`, mas apenas uma amostra do input do lote vai para `prediction_inputs_sample`.

Hoje, essa amostra nao e aleatoria. Ela e simplesmente composta pelas primeiras `N` linhas do lote processado.

Se o projeto evoluir, isso pode ser trocado por:

- amostragem aleatoria
- amostragem estratificada
- armazenamento condicional de casos criticos

## Como as rodadas de monitoramento sao executadas

Hoje, o monitoramento e executado de forma controlada pelo `src/drift_demo.py`.

Isso significa que:

- a API salva previsoes no banco
- mas nao dispara monitoramento automaticamente
- a rodada de monitoramento e uma execucao separada e intencional

### O que e uma rodada de monitoramento

Uma rodada de monitoramento e uma comparacao entre:

- uma base de referencia
- uma base atual

No projeto atual, o `drift_demo.py` usa:

- `baseline_df` como referencia
- `drift_df` como base atual

### O que acontece no `src/drift_demo.py`

O script faz, em ordem:

1. gera dados sinteticos baseline e drifted
2. carrega apenas as colunas brutas esperadas pelo modelo
3. roda previsao para baseline
4. roda previsao para drifted
5. grava ambos os lotes em `prediction_logs` e `prediction_inputs_sample`
6. chama `run_monitoring(...)`

Ou seja, cada execucao do `drift_demo.py` gera um ciclo novo de:

- dados sinteticos
- previsao
- persistencia
- monitoramento

Se a geracao sintetica usar semente fixa, os dados podem ser reproduziveis entre rodadas. Se a semente for variavel, cada rodada produz uma nova amostra.

## Como `monitoring_snapshots` e calculada

Arquivo responsavel: `src/modules/monitoring.py`

Funcao principal: `compute_monitoring_snapshot(...)`

Essa funcao percorre coluna por coluna da base de referencia. Para cada feature, ela calcula uma foto estatistica comparando referencia e atual.

Campos calculados:

- `run_id`
  - identificador unico da rodada
  - gerado automaticamente, salvo se passado externamente
- `observed_at_utc`
  - timestamp da rodada
- `dataset_label`
  - nome logico do conjunto atual monitorado
- `feature_name`
  - nome da variavel analisada
- `reference_mean`
  - media da feature na referencia
- `current_mean`
  - media da feature no conjunto atual
- `reference_std`
  - desvio padrao na referencia
- `current_std`
  - desvio padrao no atual
- `reference_missing_rate`
  - proporcao de valores ausentes na referencia
- `current_missing_rate`
  - proporcao de valores ausentes no atual
- `psi`
  - Population Stability Index
- `drift_severity`
  - classificacao do drift com base no PSI

### Como o PSI e calculado

Funcao: `calculate_psi(...)`

Passos:

1. remove nulos e converte as series para `float`
2. cria bins com base nos quantis da referencia
3. conta a distribuicao da referencia nesses bins
4. conta a distribuicao do conjunto atual nos mesmos bins
5. transforma contagens em proporcoes
6. aplica a formula do PSI

Se a referencia ou o atual estiverem vazios, o PSI retorna `0.0`.

Se a variavel nao tiver dispersao suficiente para formar bins uteis, o PSI tambem retorna `0.0`.

### Como a severidade e definida

Funcao: `classify_drift(...)`

Regras atuais:

- `psi >= 0.20` -> `high`
- `psi >= 0.10` e `< 0.20` -> `moderate`
- abaixo disso -> `low`

Essas regras sao simples, mas suficientes para uma demonstracao de portfolio.

## Como `drift_alerts` e gerada

Arquivo responsavel: `src/modules/monitoring.py`

Funcao principal: `build_drift_alerts(...)`

Essa funcao transforma a estatistica em sinal acionavel.

### Regra principal por feature

Para cada linha de `snapshot_df`:

- se `drift_severity == "low"`, nenhum alerta e criado
- se `drift_severity` for `moderate` ou `high`, um alerta e gerado

Cada alerta inclui:

- `run_id`
- `alert_timestamp_utc`
- `feature_name`
- `psi`
- `severity`
- `message`
- `recommended_action`

### Regra adicional por mudanca da taxa prevista de churn

Existe uma segunda condicao que nao depende diretamente de uma feature individual.

Se `prediction_rate_reference` e `prediction_rate_current` forem informados:

1. o codigo calcula:
   - `delta = prediction_rate_current - prediction_rate_reference`

2. se `abs(delta) >= 0.08`, um alerta extra e criado para:
   - `feature_name = "predicted_churn_rate"`

3. a severidade desse alerta segue:
   - `high` se `abs(delta) >= 0.15`
   - `moderate` caso contrario

Essa regra e importante porque captura mudancas agregadas no comportamento do modelo, mesmo quando a leitura feature a feature nao conta toda a historia sozinha.

## Como interpretar corretamente um aumento da taxa prevista de churn

Um ponto importante desta implementacao e a interpretacao do alerta ligado a `predicted_churn_rate`.

Se a taxa prevista de churn sobe no conjunto atual, isso **nao significa automaticamente que o modelo melhorou**. O que esse sinal mostra, na implementacao atual, e que a nova massa de dados passou a receber scores mais altos e, portanto, mais registros cruzaram o threshold de classificacao.

Isso pode acontecer por alguns motivos:

- os novos dados realmente ficaram mais parecidos com perfis historicamente associados a churn
- houve uma mudanca operacional ou de comportamento dos clientes
- houve drift de dados suficiente para deslocar a distribuicao das entradas
- o modelo pode estar ficando mais agressivo na previsao da classe positiva

Sem os rotulos reais do novo lote, isto e, sem `y_true`, **nao e possivel concluir se o modelo esta acertando mais ou piorando sua qualidade operacional**.

Sem `y_true`, nao sabemos se:

- o modelo ficou mais sensivel de forma correta
- ou se comecou a superestimar churn

Essa distincao e fundamental:

- `data drift` significa mudanca na distribuicao dos dados de entrada
- `model degradation` significa perda de desempenho real do modelo
- `concept drift` significa mudanca na relacao entre variaveis e target

No fluxo atual de portfolio, o que esta sendo simulado principalmente e `data drift`. O monitoramento mostra que o comportamento das entradas e da saida prevista mudou. Ele nao prova, sozinho, que houve melhora ou piora real da performance, porque isso exigiria observacao posterior do target verdadeiro.

## Como os resultados de monitoramento sao persistidos

Funcao principal: `run_monitoring(...)`

Essa funcao orquestra toda a rodada:

1. chama `compute_monitoring_snapshot(...)`
2. chama `build_drift_alerts(...)`
3. salva arquivos em `artifacts/monitoring/`
4. gera graficos
5. persiste `snapshot_df` e `alerts_df` no banco com `log_monitoring_results(...)`

Arquivos gerados:

- `drift_summary.csv`
- `drift_alerts.csv`
- `drift_report.json`
- `drift_overview.png`
- `predicted_churn_rate_shift.png`
- `alerted_feature_distributions.png`

Tabelas populadas no banco:

- `monitoring_snapshots`
- `drift_alerts`

## Resumo do fluxo ponta a ponta

### Fluxo de previsao

1. um payload entra por API ou script
2. `predict_dataframe(...)` valida e calcula previsoes
3. `log_predictions(...)` grava:
   - a saida completa em `prediction_logs`
   - uma amostra dos inputs em `prediction_inputs_sample`

### Fluxo de monitoramento

1. o `drift_demo.py` cria ou carrega dois conjuntos comparaveis
2. executa previsoes nos dois cenarios
3. chama `run_monitoring(...)`
4. o sistema gera um snapshot estatistico por feature
5. desvios relevantes viram alerta
6. snapshots e alertas sao persistidos no banco

## O que ainda nao esta automatizado

Hoje, o fluxo foi desenhado para portfolio com responsabilidades separadas e demonstracao clara. Algumas coisas ainda nao estao automaticas por design:

- a API nao dispara monitoramento a cada request
- o monitoramento nao roda por agendamento automatico
- o retreinamento nao e automatico

Em um cenario mais proximo de producao, o caminho natural seria:

- acumular previsoes por lote ou janela de tempo
- rodar monitoramento periodicamente
- avaliar persistencia do drift
- abrir processo formal de revisao e possivel retreinamento

## Por que esse desenho e importante

Esse desenho operacional mostra que o projeto nao termina no treino do modelo. Ele cobre:

- inferencia validada
- versionamento de modelo
- logging operacional
- rastreabilidade de payload
- monitoramento estatistico
- alertas de drift

Para portfolio, isso demonstra preocupacao com confiabilidade, auditoria, governanca e system design, que sao aspectos muito valorizados em projetos reais de Machine Learning.
