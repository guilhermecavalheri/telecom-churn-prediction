# 📊 Telecom Churn Prediction

Projeto de Machine Learning para previsão de churn (cancelamento) de clientes de telecomunicações.

## 🎯 Objetivo

Desenvolver um modelo preditivo capaz de identificar clientes com maior probabilidade de cancelar seus serviços, permitindo ações proativas de retenção.

## 📁 Estrutura do Projeto

```
telecom-churn-prediction/
├── data/
│   ├── train.csv              # Dados de treino
│   ├── test.csv               # Dados de teste
│   └── sampleSubmission.csv   # Formato de submissão
├── notebooks/
│   └── 01_eda_and_cleaning.ipynb  # Análise exploratória
├── Makefile                   # Automação do ambiente
├── requirements.txt           # Dependências do projeto
└── README.md                  # Este arquivo
```

## 🚀 Configuração do Ambiente

### Pré-requisitos

- Python 3.11 ou superior
- `make` instalado no sistema
  - **Windows**: Instale via [Chocolatey](https://chocolatey.org/): `choco install make`
  - **Linux/Mac**: Já vem instalado por padrão

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/telecom-churn-prediction.git
cd telecom-churn-prediction

# Configure o ambiente completo (cria venv + instala dependências)
make setup
```

### Comandos do Makefile

| Comando | Descrição |
|---------|-----------|
| `make help` | Exibe todos os comandos disponíveis |
| `make venv` | Cria o ambiente virtual |
| `make install` | Cria venv e instala dependências |
| `make install-dev` | Instala dependências de desenvolvimento |
| `make setup` | Configuração completa do ambiente |
| `make activate` | Mostra como ativar o ambiente virtual |
| `make clean` | Remove o ambiente virtual |
| `make clean-cache` | Remove arquivos de cache Python |
| `make clean-all` | Remove venv e cache |
| `make freeze` | Gera requirements-frozen.txt |
| `make update-deps` | Atualiza dependências no venv existente |

### Instalação Manual (sem Make)

Se preferir não usar o Makefile:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Atualizar pip e instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# Instalar kernel do Jupyter
python -m ipykernel install --user --name=telecom-churn --display-name="Python (Telecom Churn)"
```

## 📓 Executando os Notebooks

Após configurar o ambiente, abra o projeto no **VSCode** ou **Antigravity** e selecione o kernel **"Python (Telecom Churn)"** para executar os notebooks.

## 🛠️ Dependências Principais

- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **duckdb** - Banco de dados analítico SQL
- **matplotlib/seaborn/plotly** - Visualização
- **scikit-learn** - Machine Learning
- **xgboost/lightgbm** - Modelos gradient boosting
- **shap** - Interpretabilidade de modelos
- **ipykernel** - Kernel para notebooks (VSCode/Antigravity)

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Clone seu fork: `git clone https://github.com/seu-usuario/telecom-churn-prediction.git`
3. Configure o ambiente: `make setup`
4. Crie uma branch: `git checkout -b feature/nova-feature`
5. Faça suas alterações e commit: `git commit -m 'Adiciona nova feature'`
6. Push para a branch: `git push origin feature/nova-feature`
7. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
