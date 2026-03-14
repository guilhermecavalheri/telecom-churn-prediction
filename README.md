# Telecom Churn Prediction

Projeto de Machine Learning para previsao de churn de clientes de telecomunicacoes, com foco em EDA, feature engineering, benchmark de modelos, avaliacao robusta e artefatos reutilizaveis.

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
|   `-- refined/
|       |-- train.parquet                   # Base refinada apos limpeza
|       `-- train_engineered.parquet        # Base com features derivadas
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

## Proximos Passos

As evolucoes naturais do projeto sao:

- calibracao de probabilidade do modelo vencedor
- tuning adicional entre `XGBoost` e `LightGBM`
- persistencia padronizada do melhor pipeline em codigo-fonte
- modulo de inferencia dedicado, como `src/modules/predict.py`
- validacao de schema de entrada
- exposicao via API, por exemplo com `FastAPI`
- monitoramento e estrategia de retreinamento

## Licenca

Este projeto esta sob a licenca MIT. Se houver um arquivo `LICENSE` no repositorio, consulte-o para os detalhes completos.
