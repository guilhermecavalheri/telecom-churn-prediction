# Telecom Churn Prediction

Projeto de Machine Learning para previsao de churn de clientes de telecomunicacoes, com foco em EDA, feature engineering, benchmark de modelos, avaliacao robusta e artefatos reutilizaveis.

## Resumo Executivo

O problema de negocio resolvido por este projeto é a identificacao antecipada de clientes com maior risco de churn, permitindo priorizar ações de retencão com base em comportamento de uso dos clientes. O caso foi estruturado sobre uma base de churn de telecom coletada ao longo de 12 meses de uma empresa iraniana, com evolucao desde a analise exploratoria ate a operacao basica do modelo. O dataset foi obtido no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset).

O modelo escolhido foi o `XGBoost`, que apresentou o melhor resultado geral no benchmark. No conjunto de teste final (`holdout`), ele entregou aproximadamente `ROC AUC = 0.9881`, `Average Precision = 0.9450`, `Recall = 0.8876`, `Precision = 0.8587` e `F1 = 0.8729`, alem de ganho real com o feature engineering em relacao ao conjunto bruto.

O projeto é relevante como exemplo de ML end-to-end porque não para na modelagem: ele inclui persistência do pipeline vencedor, inferência reutilizável, API de scoring, banco local para logs operacionais, simulação de drift, monitoramento com alertas e documentação de arquitetura. Isso o torna um caso de portfolio mais próximo de um fluxo real de MLOps do que de um notebook isolado.

## Destaques do Projeto

- pipeline reproduzível de modelagem e avaliação
- persistência do melhor modelo com metadata
- inferência reutilizável via código e API
- banco local para logs operacionais e alertas
- simulação de data drift com monitoramento e explicação dos sinais gerados
- notebooks investigativos e documentação técnica estruturada

## Objetivo

Desenvolver um pipeline preditivo capaz de identificar clientes com maior probabilidade de cancelamento, apoiando a priorização de ações de retenção.

O projeto usa uma base de churn de telecom coletada ao longo de 12 meses e evolui o fluxo desde a análise exploratória até a persistência do melhor pipeline treinado.

## Estrutura do Projeto

Observação sobre organização:

- `src/` contém código executável e módulos reutilizáveis do projeto.
- `src/modules/` concentra a lógica compartilhada entre treino, inferência, API e monitoramento.
- `artifacts/` contém apenas saídas persistidas do projeto, como modelos, logs operacionais, relatórios e gráficos.
- Portanto, `src/modules/ops_store.py` é o código que gerencia o banco local, enquanto `artifacts/database/ml_ops.duckdb` é o banco em si.

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
   - limpeza, padronização e validação inicial
   - exportação da base pronta para modelagem

2. `src/churn_modeling.py`
   - criação das features derivadas
   - benchmark de múltiplos modelos
   - validação em cross-validation e holdout
   - geração de métricas, gráficos e resumo final

3. `notebooks/02_modeling_evaluation.ipynb`
   - reprodução do pipeline em formato investigativo
   - interpretação detalhada dos gráficos e resultados
   - conclusão técnica e resumo executivo

4. `notebooks/03_drift_monitoring_demo.ipynb`
   - demonstração detalhada do fluxo de drift
   - interpretação dos gráficos e alertas de monitoramento

5. `notebooks/04_duckdb_and_api_operations_demo.ipynb`
   - consultas ao banco local de MLOps
   - consumo da API e leitura dos registros operacionais

6. `api/app.py`, `src/modules/predict.py` e `src/modules/ops_store.py`
   - inferência reutilizável
   - API de scoring
   - persistência operacional em banco local DuckDB

7. `src/modules/synthetic_data.py`, `src/modules/monitoring.py` e `src/drift_demo.py`
   - geração de dados sintéticos
   - simulação de drift
   - alerta de possível retreinamento

## Principais Entregas

- Feature engineering guiado por hipótese de negócio
- Benchmark com pelo menos 5 modelos de classificação
- Avaliação com ROC AUC, Average Precision, Recall, Precision, F1 e Balanced Accuracy
- Gráficos de ROC, Precision-Recall, matriz de confusão, importância de features e threshold trade-off
- Comparação entre base original e base com features derivadas
- Persistência do melhor pipeline treinado para reuso futuro
- API local de inferência com FastAPI
- Banco operacional local com logs de previsão e monitoramento
- Simulação de drift com alertas e artefatos de observabilidade

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
| `make help` | Exibe os comandos disponíveis |
| `make venv` | Cria o ambiente virtual |
| `make install` | Instala dependências e registra o kernel |
| `make install-dev` | Instala dependências de desenvolvimento |
| `make setup` | Executa a configuração completa |
| `make activate` | Mostra como ativar o ambiente virtual |
| `make clean` | Remove o ambiente virtual |
| `make clean-cache` | Remove caches Python |
| `make clean-all` | Remove ambiente e caches |
| `make freeze` | Gera `requirements-frozen.txt` |
| `make update-deps` | Atualiza dependências no venv existente |

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

- `GET /health`: verifica se o servico está vivo
- `GET /ready`: verifica se o servico está pronto para operar, incluindo modelo, metadata e banco local
- `GET /model/info`: retorna informações da versão ativa do modelo
- `POST /predict`: inferencia unitária
- `POST /predict/batch`: inferencia em lote

### Demo de drift

Para gerar dados sintéticos baseline e driftado, registrar previsões e produzir artefatos de monitoramento:

```bash
python src/drift_demo.py
```

Esse fluxo salva datasets em `data/synthetic/`, registra eventos em `artifacts/database/ml_ops.duckdb` e gera alertas em `artifacts/monitoring/`.
Os principais visuais do monitoramento são:

- `drift_overview.png`: ranking de PSI por feature com severidade do drift
- `predicted_churn_rate_shift.png`: comparação entre a taxa prevista de churn no baseline e no cenario driftado
- `alerted_feature_distributions.png`: distribuições das features com alerta para facilitar interpretação

## Dependências Principais

- `pandas`, `numpy`: manipulação e transformação de dados
- `matplotlib`, `seaborn`, `plotly`: visualização
- `scikit-learn`: pipelines, validação e métricas
- `xgboost`, `lightgbm`: modelos de gradient boosting
- `duckdb`: apoio analítico em dados
- `shap`: interpretabilidade
- `ipykernel`, `nbformat`: execução e estruturação de notebooks

## Limitações

Este projeto foi estruturado para portfolio com foco em clareza arquitetural, rastreabilidade e demonstração de boas práticas de ML end-to-end. Ainda assim, algumas decisões foram mantidas em nível simplificado para preservar objetividade e escopo.

As principais limitações atuais são:

- o monitoramento de drift não é disparado automaticamente pela API; ele roda em fluxo separado de demonstração
- a simulação operacional usa dados sintéticos, adequados para portfolio, mas não substitui observação de produção real
- o banco local foi desenhado para rastreabilidade e demonstração, não para alta escala
- o retreinamento não é automático; apenas o processo recomendado está documentado
- a avaliação de drift não mede degradação real de performance em novos lotes sem `y_true`, apenas mudança de distribuição e mudança no comportamento previsto do modelo
- a API não possui autenticação, controle de acesso ou deploy em infraestrutura produtiva
- o projeto ainda não possui containerização oficial para execução padronizada

## Próximos Passos

As evoluções naturais do projeto são:

- calibração de probabilidade do modelo vencedor
- tuning adicional entre `XGBoost` e `LightGBM`
- teste de integração end-to-end cobrindo treino, persistência, inferência e monitoramento
- camada executiva de observabilidade para leitura da operação
- containerização com `Docker`
- evolução controlada do processo de retreinamento

O roadmap detalhado de acabamento final está em `docs/final_polish_roadmap.md`.