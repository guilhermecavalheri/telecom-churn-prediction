# Telecom Churn Prediction

Projeto de Machine Learning para previsao de churn de clientes de telecomunicacoes, com foco em EDA, feature engineering, benchmark de modelos, avaliacao robusta e artefatos reutilizaveis.

## Resumo Executivo

O problema de negocio resolvido por este projeto e a identificacao antecipada de clientes com maior risco de churn, permitindo priorizar acoes de retencao com base em comportamento de uso, friccao operacional e sinais de relacionamento. O caso foi estruturado sobre uma base de telecom coletada ao longo de 12 meses, com evolucao desde a analise exploratoria ate a operacao basica do modelo.

O modelo escolhido foi o `XGBoost`, que apresentou o melhor equilibrio geral no benchmark. No conjunto de teste final (`holdout`), ele entregou aproximadamente `ROC AUC = 0.9881`, `Average Precision = 0.9450`, `Recall = 0.8876`, `Precision = 0.8587` e `F1 = 0.8729`, alem de ganho real com o feature engineering em relacao ao conjunto bruto.

O projeto e relevante como exemplo de ML end-to-end porque nao para na modelagem: ele inclui persistencia do pipeline vencedor, inferencia reutilizavel, API de scoring, banco local para logs operacionais, simulacao de drift, monitoramento com alertas e documentacao de arquitetura. Isso o torna um caso de portfolio mais proximo de um fluxo real de MLOps do que de um notebook isolado.

## Destaques do Projeto

- pipeline reproduzivel de modelagem e avaliacao
- persistencia do melhor modelo com metadata
- inferencia reutilizavel via codigo e API
- banco local para logs operacionais e alertas
- simulacao de data drift com monitoramento e explicacao dos sinais gerados
- notebooks investigativos e documentacao tecnica estruturada

## Objetivo

Desenvolver um pipeline preditivo capaz de identificar clientes com maior probabilidade de cancelamento, apoiando a priorizacao de acoes de retencao.

O projeto usa uma base de churn de telecom coletada ao longo de 12 meses e evolui o fluxo desde a analise exploratoria ate a persistencia do melhor pipeline treinado.

## Estrutura do Projeto

Observacao sobre organizacao:

- `src/` contem codigo executavel e modulos reutilizaveis do projeto.
- `src/modules/` concentra a logica compartilhada entre treino, inferencia, API e monitoramento.
- `artifacts/` contem apenas saidas persistidas do projeto, como modelos, logs operacionais, relatorios e graficos.
- Portanto, `src/modules/ops_store.py` e o codigo que gerencia o banco local, enquanto `artifacts/database/ml_ops.duckdb` e o banco em si.

```text
telecom-churn-prediction/
|-- data/
|   |-- raw/
|   |   `-- iranian_churn_telecom.parquet   # Base original
|   |-- trusted/
|   |   `-- train.parquet                   # Base validada para modelagem
|   |-- refined/
|       |-- train.parquet                   # Base refinada apos limpeza
|       `-- train_engineered.parquet        # Base com features derivadas
|   `-- synthetic/
|       |-- api_test_baseline.parquet       # Lote sintetico de referencia
|       `-- api_test_drifted.parquet        # Lote sintetico com drift simulado
|-- artifacts/
|   |-- modeling/                           # CSVs, graficos e resumo da modelagem
|   |-- models/
|   |   |-- best_churn_pipeline.joblib      # Pipeline vencedor serializado
|   |   `-- best_churn_pipeline_metadata.json
|   |                                      # Metadata do modelo salvo
|   |-- monitoring/
|   |   |-- drift_summary.csv               # Snapshot de drift por feature
|   |   |-- drift_alerts.csv                # Alertas gerados pelo monitoramento
|   |   |-- drift_overview.png              # Ranking visual de PSI por feature
|   |   |-- predicted_churn_rate_shift.png  # Comparacao da taxa prevista de churn
|   |   |-- alerted_feature_distributions.png
|   |   |                                  # Distribuicoes baseline vs drift
|   |   `-- drift_report.json               # Resumo executivo do monitoramento
|   `-- database/
|       `-- ml_ops.duckdb                   # Banco local com logs operacionais
|-- notebooks/
|   |-- 01_eda_and_cleaning.ipynb           # EDA, limpeza e validacao inicial
|   |-- 02_modeling_evaluation.ipynb        # Modelagem, feature engineering e avaliacao
|   |-- 03_drift_monitoring_demo.ipynb      # Simulacao e interpretacao de drift
|   `-- 04_duckdb_and_api_operations_demo.ipynb
|                                          # Consultas operacionais e chamadas da API
|-- api/
|   `-- app.py                              # FastAPI para scoring unitario e em lote
|-- docs/
|   |-- end_to_end_ml_evolution.md          # Arquitetura, plano e fluxo em Mermaid
|   |-- final_polish_roadmap.md             # Roadmap de acabamento final do portfolio
|   |-- operational_monitoring_reference.md # Fluxo operacional, tabelas, calculos e alertas
|   `-- retraining_playbook.md              # Passo a passo documentado do retreinamento
|-- src/
|   |-- churn_modeling.py                   # Entrypoint do pipeline de treinamento e avaliacao
|   |-- drift_demo.py                       # Entrypoint da demo de drift e monitoramento
|   `-- modules/
|       |-- model_io.py                     # Persistencia e carga do modelo vencedor
|       |-- schema.py                       # Validacao de schema para inferencia
|       |-- predict.py                      # Inferencia reutilizavel fora do notebook
|       |-- synthetic_data.py               # Geracao de dados sinteticos baseline e drift
|       |-- monitoring.py                   # Monitoramento e alertas de drift
|       `-- ops_store.py                    # Persistencia operacional local em DuckDB
|-- tests/
|   `-- ...                                 # Testes de persistencia, inferencia, API e monitoramento
|-- .agent/
|   `-- skills/
|       `-- ml-pipeline-workflow/           # Skill local para fluxos de ML guiados
|-- Makefile                                # Automacao do ambiente
|-- requirements.txt                        # Dependencias do projeto
`-- README.md
```

## Fluxo do Projeto

1. `01_eda_and_cleaning.ipynb`
   - leitura e entendimento dos dados
   - limpeza, padronizacao e validacao inicial
   - exportacao da base pronta para modelagem

2. `src/churn_modeling.py`
   - criacao das features derivadas
   - benchmark de multiplos modelos
   - validacao em cross-validation e holdout
   - geracao de metricas, graficos e resumo final

3. `notebooks/02_modeling_evaluation.ipynb`
   - reproducao do pipeline em formato investigativo
   - interpretacao detalhada dos graficos e resultados
   - conclusao tecnica e resumo executivo

4. `notebooks/03_drift_monitoring_demo.ipynb`
   - demonstracao detalhada do fluxo de drift
   - interpretacao dos graficos e alertas de monitoramento

5. `notebooks/04_duckdb_and_api_operations_demo.ipynb`
   - consultas ao banco local de MLOps
   - consumo da API e leitura dos registros operacionais

6. `api/app.py`, `src/modules/predict.py` e `src/modules/ops_store.py`
   - inferencia reutilizavel
   - API de scoring
   - persistencia operacional em banco local DuckDB

7. `src/modules/synthetic_data.py`, `src/modules/monitoring.py` e `src/drift_demo.py`
   - geracao de dados sinteticos
   - simulacao de drift
   - alerta de possivel retreinamento

## Principais Entregas

- Feature engineering guiado por hipotese de negocio
- Benchmark com pelo menos 5 modelos de classificacao
- Avaliacao com ROC AUC, Average Precision, Recall, Precision, F1 e Balanced Accuracy
- Graficos de ROC, Precision-Recall, matriz de confusao, importancia de features e threshold trade-off
- Comparacao entre base original e base com features derivadas
- Persistencia do melhor pipeline treinado para reuso futuro
- API local de inferencia com FastAPI
- Banco operacional local com logs de previsao e monitoramento
- Simulacao de drift com alertas e artefatos de observabilidade

## Como Configurar o Ambiente

### Pre-requisitos

- Python 3.11 ou superior
- `make` instalado no sistema
  - Windows: `choco install make`
  - Linux/macOS: normalmente ja disponivel

### Instalacao rapida

```bash
git clone https://github.com/seu-usuario/telecom-churn-prediction.git
cd telecom-churn-prediction
make setup
```

### Instalacao manual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=telecom-churn --display-name="Python (Telecom Churn)"
```

## Comandos do Makefile

| Comando | Descricao |
|---|---|
| `make help` | Exibe os comandos disponiveis |
| `make venv` | Cria o ambiente virtual |
| `make install` | Instala dependencias e registra o kernel |
| `make install-dev` | Instala dependencias de desenvolvimento |
| `make setup` | Executa a configuracao completa |
| `make activate` | Mostra como ativar o ambiente virtual |
| `make clean` | Remove o ambiente virtual |
| `make clean-cache` | Remove caches Python |
| `make clean-all` | Remove ambiente e caches |
| `make freeze` | Gera `requirements-frozen.txt` |
| `make update-deps` | Atualiza dependencias no venv existente |

## Como Executar

### Notebooks

Abra o projeto no VS Code ou Jupyter e selecione o kernel `Python (Telecom Churn)`.

Ordem recomendada:

1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_modeling_evaluation.ipynb`
3. notebooks auxiliares de revisao metodologica ou persistencia, se desejar

### Pipeline de modelagem

Para executar a modelagem fora do notebook:

```bash
python src/churn_modeling.py
```

Esse comando recria a base enriquecida e atualiza os artefatos em `artifacts/modeling/`.

### API de inferencia

Para subir a API localmente:

```bash
uvicorn api.app:app --reload
```

Endpoints principais:

- `GET /health`: verifica se o servico esta vivo
- `GET /ready`: verifica se o servico esta pronto para operar, incluindo modelo, metadata e banco local
- `GET /model/info`: retorna informacoes da versao ativa do modelo
- `POST /predict`: inferencia unitaria
- `POST /predict/batch`: inferencia em lote

### Demo de drift

Para gerar dados sinteticos baseline e driftado, registrar previsoes e produzir artefatos de monitoramento:

```bash
python src/drift_demo.py
```

Esse fluxo salva datasets em `data/synthetic/`, registra eventos em `artifacts/database/ml_ops.duckdb` e gera alertas em `artifacts/monitoring/`.
Os principais visuais do monitoramento sao:

- `drift_overview.png`: ranking de PSI por feature com severidade do drift
- `predicted_churn_rate_shift.png`: comparacao entre a taxa prevista de churn no baseline e no cenario driftado
- `alerted_feature_distributions.png`: distribuicoes das features com alerta para facilitar interpretacao

## Dependencias Principais

- `pandas`, `numpy`: manipulacao e transformacao de dados
- `matplotlib`, `seaborn`, `plotly`: visualizacao
- `scikit-learn`: pipelines, validacao e metricas
- `xgboost`, `lightgbm`: modelos de gradient boosting
- `duckdb`: apoio analitico em dados
- `shap`: interpretabilidade
- `ipykernel`, `nbformat`: execucao e estruturacao de notebooks

## Limitacoes

Este projeto foi estruturado para portfolio com foco em clareza arquitetural, rastreabilidade e demonstracao de boas praticas de ML end-to-end. Ainda assim, algumas decisoes foram mantidas em nivel simplificado para preservar objetividade e escopo.

As principais limitacoes atuais sao:

- o monitoramento de drift nao e disparado automaticamente pela API; ele roda em fluxo separado de demonstracao
- a simulacao operacional usa dados sinteticos, adequados para portfolio, mas nao substitui observacao de producao real
- o banco local foi desenhado para rastreabilidade e demonstracao, nao para alta escala
- o retreinamento nao e automatico; apenas o processo recomendado esta documentado
- a avaliacao de drift nao mede degradacao real de performance em novos lotes sem `y_true`, apenas mudanca de distribuicao e mudanca no comportamento previsto do modelo
- a API nao possui autenticacao, controle de acesso ou deploy em infraestrutura produtiva
- o projeto ainda nao possui containerizacao oficial para execucao padronizada

## Proximos Passos

As evolucoes naturais do projeto sao:

- calibracao de probabilidade do modelo vencedor
- tuning adicional entre `XGBoost` e `LightGBM`
- teste de integracao end-to-end cobrindo treino, persistencia, inferencia e monitoramento
- camada executiva de observabilidade para leitura da operacao
- containerizacao com `Docker`
- evolucao controlada do processo de retreinamento

O roadmap detalhado de acabamento final esta em `docs/final_polish_roadmap.md`.

## Licenca

Este projeto esta sob a licenca MIT. Se houver um arquivo `LICENSE` no repositorio, consulte-o para os detalhes completos.
