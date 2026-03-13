# Makefile para o projeto Telecom Churn Prediction
# Facilita a configuração do ambiente de desenvolvimento

# Variáveis
PYTHON = python
VENV_DIR = venv

# Detectar sistema operacional
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE = $(VENV_DIR)\Scripts\activate
    RM = rmdir /s /q
    PYTHON_VENV = $(VENV_DIR)\Scripts\python
else
    VENV_ACTIVATE = source $(VENV_DIR)/bin/activate
    RM = rm -rf
    PYTHON_VENV = $(VENV_DIR)/bin/python
endif

.PHONY: help venv install install-dev update-deps activate clean clean-cache clean-all freeze

# Comando padrão - mostra ajuda
help:
	@echo.
	@echo ========================================
	@echo   Telecom Churn Prediction - Makefile
	@echo ========================================
	@echo.
	@echo Comandos disponiveis:
	@echo.
	@echo   make venv          - Cria o ambiente virtual
	@echo   make install       - Instala as dependencias
	@echo   make install-dev   - Instala dependencias de desenvolvimento
	@echo   make setup         - Configura o ambiente completo (venv + install)
	@echo   make activate      - Mostra como ativar o ambiente virtual
	@echo   make clean         - Remove o ambiente virtual
	@echo   make clean-cache   - Remove arquivos de cache Python
	@echo   make clean-all     - Remove venv e cache
	@echo   make update-deps   - Atualiza dependencias no venv existente
	@echo   make freeze        - Gera requirements-frozen.txt
	@echo.

# Cria o ambiente virtual
venv:
	@echo Criando ambiente virtual...
	$(PYTHON) -m venv $(VENV_DIR)
	@echo Ambiente virtual criado em $(VENV_DIR)/

# Instala as dependências
install: venv
	@echo Atualizando pip...
	$(PYTHON_VENV) -m pip install --upgrade pip
	@echo Instalando dependencias...
	$(PYTHON_VENV) -m pip install -r requirements.txt
	@echo.
	@echo Registrando kernel para notebooks...
	$(PYTHON_VENV) -m ipykernel install --user --name=telecom-churn --display-name="Python (Telecom Churn)"
	@echo.
	@echo Instalacao concluida!

# Instala dependências de desenvolvimento adicionais
install-dev: install
	@echo Instalando dependencias de desenvolvimento...
	$(PYTHON_VENV) -m pip install black flake8 isort pytest pytest-cov pre-commit
	@echo Dependencias de desenvolvimento instaladas!

# Atualiza dependências no ambiente virtual existente (sem recriar venv)
update-deps:
	@echo Atualizando pip...
	$(PYTHON_VENV) -m pip install --upgrade pip
	@echo Atualizando dependencias do requirements.txt...
	$(PYTHON_VENV) -m pip install -r requirements.txt
	@echo.
	@echo Dependencias atualizadas com sucesso!

# Setup completo do ambiente
setup: install
	@echo.
	@echo ========================================
	@echo   Ambiente configurado com sucesso!
	@echo ========================================
	@echo.
	@echo Para ativar o ambiente virtual:
	@echo   Windows: $(VENV_DIR)\Scripts\activate
	@echo   Linux/Mac: source $(VENV_DIR)/bin/activate
	@echo.
	@echo Selecione o kernel "Python (Telecom Churn)" no VSCode/Antigravity
	@echo.

# Mostra como ativar o ambiente virtual
activate:
	@echo.
	@echo Para ativar o ambiente virtual, execute:
	@echo.
ifeq ($(OS),Windows_NT)
	@echo   .\$(VENV_DIR)\Scripts\activate
else
	@echo   source $(VENV_DIR)/bin/activate
endif
	@echo.

# Remove o ambiente virtual
clean:
	@echo Removendo ambiente virtual...
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_DIR) rmdir /s /q $(VENV_DIR)
else
	@rm -rf $(VENV_DIR)
endif
	@echo Ambiente virtual removido!

# Remove arquivos de cache
clean-cache:
	@echo Limpando cache Python...
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ rmdir /s /q __pycache__
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
else
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
endif
	@echo Cache limpo!

# Remove tudo (ambiente virtual + cache)
clean-all: clean clean-cache
	@echo Limpeza completa concluida!

# Gera requirements.txt atualizado
freeze:
	@echo Gerando requirements-frozen.txt...
	$(PYTHON_VENV) -m pip freeze > requirements-frozen.txt
	@echo Arquivo requirements-frozen.txt gerado!
