# 📈 OptyMax — MVP (B3 + OPLAB v3 + Tracking em Tempo Real)

O **OptyMax** é uma aplicação em **Streamlit** desenvolvida para auxiliar investidores pessoa física que atuam no mercado de **opções da B3**, com foco em **operações de venda coberta** e **strangles vendidos cobertos**.

A aplicação conecta-se diretamente à **API oficial da OPLAB v3** para coletar dados em tempo real das opções, calcular métricas como **Delta**, **IV Rank**, **TIO (Taxa Interna de Oportunidade)** e exibir visualmente as melhores oportunidades dentro dos critérios definidos pelo usuário.

---

## 🧠 Objetivo do Projeto

O objetivo principal do OptyMax é **otimizar a escolha de opções para estratégias de venda coberta e strangle**, maximizando o prêmio recebido e minimizando a probabilidade de exercício.

O usuário pode:
- Selecionar ativos da B3 (ex: PETR4, VALE3, PSSA3, etc);
- Filtrar opções por DTM (dias até o vencimento) e Delta mínimo/máximo;
- Analisar CALLs e PUTs separadamente;
- Montar automaticamente strangles com cálculo de TIO e IV Rank.

---

## ⚙️ Estrutura do Projeto

```
OptyMax/
│
├── app.py              → Aplicativo principal Streamlit
├── requirements.txt    → Dependências do projeto
└── README.md           → Este arquivo de documentação
```

---

## 🚀 Execução do Projeto

### 1️⃣ **Execução Local (opcional)**
Se quiser rodar o app localmente (em vez do Streamlit Cloud), siga os passos:

```bash
# Clone o repositório
git clone https://github.com/SEU_USUARIO/OptyMax-MVP.git
cd OptyMax-MVP

# Instale as dependências
pip install -r requirements.txt

# Configure seu token da OPLAB
export OPLAB_TOKEN="SEU_ACCESS_TOKEN_AQUI"

# Execute o app
streamlit run app.py
```

O app abrirá automaticamente no navegador (geralmente em `http://localhost:8501`).

---

### 2️⃣ **Deploy no Streamlit Cloud (recomendado)**

1. Acesse [https://streamlit.io/cloud](https://streamlit.io/cloud);
2. Crie um novo app conectando ao seu repositório GitHub;
3. No campo **Main file path**, digite:  
   ```bash
   app.py
   ```
4. Vá em **Settings → Secrets** e adicione seu token da OPLAB:
   ```bash
   OPLAB_TOKEN="SEU_ACCESS_TOKEN_AQUI"
   ```

O Streamlit Cloud instalará automaticamente todas as dependências e executará o aplicativo.

---

## 🔐 Integração com a API da OPLAB

O OptyMax utiliza a API oficial da OPLAB v3:

### 📘 Endpoints utilizados:
- **Listagem de opções**:  
  `GET https://api.oplab.com.br/v3/market/options/{UNDERLYING}`

  Retorna todos os contratos de opções (CALL e PUT) de um ativo base (ex: PETR4).

- **Cálculo Black-Scholes**:  
  `GET https://api.oplab.com.br/v3/market/options/bs?symbol=...`

  Retorna Delta, Vega, Theta, Rho, volatilidade implícita, preço teórico e probabilidade de exercício.

### 🔑 Autenticação:
A API requer o uso de um token de acesso:
```bash
Access-Token: SEU_ACCESS_TOKEN_AQUI
```

Esse token deve ser configurado no ambiente via variável `OPLAB_TOKEN`.

---

## 📊 Funcionalidades do Aplicativo

| Função | Descrição |
|--------|------------|
| **Seleção de Ativos** | Lista automaticamente os tickers da B3 com nomes das empresas |
| **Filtro de DTM** | Define o intervalo mínimo e máximo de dias até o vencimento |
| **Filtro de Delta** | Define a faixa de Delta (absoluto) válida para CALLs e PUTs |
| **IV Rank (Volatilidade Implícita)** | Calculado com base na volatilidade histórica via `yfinance` |
| **Cálculo do TIO** | Calcula a taxa anualizada de retorno do prêmio recebido |
| **Exportação CSV** | Permite exportar os resultados dos strangles montados |
| **Barra de Progresso** | Mostra o avanço do processamento em tempo real |
| **Log Dinâmico** | Exibe mensagens detalhadas sobre cada etapa de execução |

---

## 🧮 Fórmulas Relevantes

### **Taxa Interna de Oportunidade (TIO)**
\`\`\`text
TIO = (Prêmio Total / Preço Spot) × (365 / DTM) × 100
\`\`\`

### **IV Rank**
\`\`\`text
IV Rank = (IV Atual - IV Mínimo) / (IV Máximo - IV Mínimo) × 100
\`\`\`

---

## 📈 Fluxo de Execução do App

1. O usuário seleciona até **3 tickers** na barra lateral;
2. O app consulta a API da OPLAB para buscar todas as opções disponíveis;
3. Calcula os **Greeks (Delta, Vega, etc.)** via endpoint `/bs`;
4. Aplica os filtros definidos pelo usuário;
5. Exibe as **top 3 CALLs e top 3 PUTs** para cada ativo;
6. Monta e exibe automaticamente os **strangles** com melhor TIO.

Durante todo o processo, o app mostra:
- Uma **barra de progresso** (% de conclusão);
- Um **log dinâmico** detalhando cada etapa em tempo real.

---

## 🧰 Dependências

As dependências estão listadas em `requirements.txt`:

```
streamlit
pandas
numpy
requests
beautifulsoup4
yfinance
```

---

## 💡 Boas Práticas e Sugestões

- Atualize regularmente o token da OPLAB (ele tem validade limitada);
- Evite consultar muitos tickers ao mesmo tempo — use até 3;
- Configure corretamente as variáveis no painel de **Secrets** do Streamlit Cloud;
- Sempre utilize tickers válidos da B3 (ex: `PETR4`, `VALE3`, `PSSA3`, `ITUB4`).

---

## 🧠 Autor

Desenvolvido por **Bruno Teixeira**  
Projeto educacional para estudo e aplicação de estratégias com derivativos da B3.

---

## 📜 Licença

Este projeto é distribuído sob a licença **MIT**, permitindo livre uso e modificação, desde que citada a autoria original.

---

> “O sucesso nas opções não vem de adivinhar o mercado, mas de **gerenciar probabilidades com consistência**.”
