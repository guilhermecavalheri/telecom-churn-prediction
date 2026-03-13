# Telecom Churn Prediction

Projeto de Machine Learning para previsao de churn de clientes de telecomunicacoes, com foco em EDA, feature engineering, benchmark de modelos, avaliacao robusta e artefatos reutilizaveis.

## Objetivo

Desenvolver um pipeline preditivo capaz de identificar clientes com maior probabilidade de cancelamento, apoiando a priorizacao de acoes de retencao.

O projeto usa uma base de churn de telecom coletada ao longo de 12 meses e evolui o fluxo desde a analise exploratoria ate a persistencia do melhor pipeline treinado.

## Estrutura do Projeto

```text
telecom-churn-prediction/
|-- data/
|   |-- raw/
|   |   `-- iranian_churn_telecom.parquet   # Base original
|   |-- trusted/
|   |   `-- train.parquet                   # Base validada para modelagem
|   `-- refined/
|       |-- train.parquet                   # Base refinada apos limpeza
|       `-- train_engineered.parquet        # Base com features derivadas
|-- notebooks/
|   |-- 01_eda_and_cleaning.ipynb           # EDA, limpeza e validacao inicial
|   |-- 02_modeling_evaluation.ipynb        # Modelagem, feature engineering e avaliacao
|   |-- 02_modeling_evaluation_significance_review.ipynb
|   |                                      # Variante com leitura de significancia em correlacao
|   `-- 03_best_model_persistence_example.ipynb
|                                          # Exemplo de persistencia do pipeline vencedor
|-- src/
|   `-- churn_modeling.py                   # Pipeline reproduzivel de treinamento e avaliacao
|-- artifacts/
|   |-- modeling/                           # CSVs, graficos e resumo da modelagem
|   `-- models/
|       |-- best_churn_pipeline.joblib      # Pipeline vencedor serializado
|       `-- best_churn_pipeline_metadata.json
|                                          # Metadata do modelo salvo
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

3. `02_modeling_evaluation.ipynb`
   - reproducao do pipeline em formato investigativo
   - interpretacao detalhada dos graficos e resultados
   - conclusao tecnica e resumo executivo

4. `03_best_model_persistence_example.ipynb`
   - demonstracao de como salvar o melhor pipeline treinado
   - preparo do projeto para uma evolucao mais end-to-end

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
- modulo de inferencia dedicado, como `src/predict.py`
- validacao de schema de entrada
- exposicao via API, por exemplo com `FastAPI`
- monitoramento e estrategia de retreinamento

## Licenca

Este projeto esta sob a licenca MIT. Se houver um arquivo `LICENSE` no repositorio, consulte-o para os detalhes completos.
