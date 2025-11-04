# 📈 OptyMax — MVP (B3 + OPLAB v3)

Aplicativo em **Streamlit** focado em estratégias de **venda coberta** e **strangle vendido coberto** no mercado de opções da B3.

## 🚀 Execução no Streamlit Cloud
1. Faça **fork** ou **clone** deste repositório.
2. Crie um app no [Streamlit Cloud](https://streamlit.io/cloud).
3. Caminho principal: `app.py`
4. No painel de “Secrets” do Streamlit Cloud, adicione:
   ```bash
   OPLAB_TOKEN="seu_access_token_aqui"
   ```

## ⚙️ Recursos
- Seleção de até 3 tickers (com nomes das empresas)
- Filtros de DTM, Delta (absoluto) e IV Rank
- Consultas à **API OPLAB v3**
- Cálculo de **TIO** e **IV Rank**
- Exibição dos **Top 3 CALLs** e **Top 3 PUTs**
- Montagem automática de **Strangles** com exportação CSV
