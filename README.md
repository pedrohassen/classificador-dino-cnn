# Guia de Configuração: API e Frontend Classificador Dino

Este guia unificado orienta na configuração da API Node.js (que gerencia o modelo CNN em PyTorch) e da interface Web (Frontend) para inferência de imagens.

## 📋 Pré-requisitos

* **Node.js** instalado (versão 14 ou superior)
* **Python** instalado (versão 3.7 ou superior)

---

## 📁 Estrutura do Projeto

A organização final do diretório raiz deve seguir o padrão abaixo:
```txt
CNN-API-AULAS/
├── frontend/
│   └── index.html      # Interface HTML/CSS/JS (Vanilla)
├── models_saved/
│   └── model.pth       # Pesos do modelo treinado
├── package.json        # Configurações da API Node.js
├── requirements.txt    # Bibliotecas de IA (PyTorch/Torchvision)
├── server.js           # Código do servidor back-end
└── .gitignore          # Arquivo de filtros do Git
```

---

## 1. Configuração do Ambiente e Dependências

Abra o terminal na pasta raiz do projeto (`CNN-API-AULAS/`) e execute os passos abaixo.

### Passo 1: Instalar dependências do Node.js
Instala o Express, Multer e o suporte a requisições do Frontend (CORS).
```bash
npm install
```

### Passo 2: Ativar o Ambiente Virtual Python (.venv)
Crie e ative o ambiente isolado para que o Node encontre o PyTorch corretamente.

* **No Windows (Git Bash):**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate
    ```
* **No Windows (PowerShell):**
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```
* **No Linux / macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### Passo 3: Instalar as dependências do Python
Com o terminal exibindo o prefixo `(.venv)` na linha de comando, inicie a instalação:
```bash
pip install -r requirements.txt
```

---

## 2. Preparação do Modelo (.pth)

O Node.js espera encontrar o modelo treinado em um local específico para iniciar a API.

1. Crie a pasta `models_saved/` na raiz do projeto, caso ela não exista.
2. Copie o seu arquivo de pesos treinado (`.pth`) para dentro dela.
3. Renomeie o arquivo obrigatoriamente para: **`model.pth`**

O caminho final deve ser exatamente: `models_saved/model.pth`

---

## 3. Executando a Aplicação (Passo a Passo)

Como o projeto possui uma API (Back-end) e uma Interface (Front-end), você precisará de **dois terminais distintos** abertos simultaneamente na raiz do projeto.

### Terminal 1: Iniciar a API do Node.js
No primeiro terminal (garantindo que o `.venv` esteja ativo):
```bash
npm start
```
**Mensagem de sucesso esperada:**
```text
Inicializando API de inferência CNN...
Modelo esperado em: models_saved/model.pth

API iniciada com modelo carregado.
Servidor rodando em: http://localhost:3000
Endpoint de inferência: POST http://localhost:3000/infer
```
*(Nota: O próprio processo criará a pasta interna `.runtime` e o script de inferência automaticamente ao iniciar).*

### Terminal 2: Iniciar o Servidor do Frontend
Abra um novo terminal na pasta raiz do projeto. É **obrigatório** rodar um servidor local para que o navegador não bloqueie as requisições por segurança. Escolha **apenas uma** das opções abaixo:

* **Opção A (Via Python):**
    ```bash
    cd frontend
    python -m http.server 5500
    ```
* **Opção B (Via Node.js):**
    ```bash
    cd frontend
    npx http-server -p 5500
    ```
* **Opção C (VS Code):**
    Abra a pasta `frontend` no VS Code, clique com o botão direito no arquivo `index.html` e selecione **"Open with Live Server"**.

---

## 🌐 Acesso e Teste

1. Abra o seu navegador e acesse o endereço do servidor front-end: **`http://localhost:5500`**
2. Arraste ou selecione uma imagem de um dinossauro suportado na interface gráfica.
3. Clique em **"Classificar Imagem"** para enviar o arquivo para processamento e receber o retorno em tempo real.

---

## 🔧 Resolução de Problemas Comuns

### Erro: `spawn python3 ENOENT` ou `spawn python ENOENT`
O Node.js está tentando chamar o executável do Python com uma nomenclatura que o seu sistema operacional não reconhece globalmente.
* **Solução:** Se o erro persistir mesmo com o `.venv` ativo, defina explicitamente o comando do Python antes de rodar a API:
    * **No Windows (Git Bash/CMD):** `set PYTHON_CMD=python` e depois `npm start`.
    * **No Linux/Mac:** `export PYTHON_CMD=python3` e depois `npm start`.

### Erro: `Access to XMLHttpRequest blocked by CORS policy`
O navegador bloqueou a comunicação entre a porta do front (5500) e a do back (3000).
* **Solução:** Certifique-se de que o servidor da API (Terminal 1) foi reiniciado após rodar o `npm install`. Caso persista, limpe o cache do navegador pressionando `Ctrl + F5`.