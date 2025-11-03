# OptyMax MVP — Streamlit + OPLAB API

Este projeto é um MVP em Streamlit que integra com a API pública da OPLAB (v3) para buscar dados de opções, calcular métricas e sugerir estratégias de venda coberta e strangle vendido coberto.

## 🚀 Execução Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Configuração do Token OPLAB

O token de acesso deve ser configurado como variável de ambiente:

**Linux/Mac**
```bash
export OPLAB_TOKEN='seu_token_aqui'
```

**Windows (PowerShell)**
```powershell
$env:OPLAB_TOKEN='seu_token_aqui'
```

## 🧩 Principais Endpoints Utilizados

- `/v3/market/options/details/{symbol}` — Consulta de detalhes da opção
- `/v3/market/options/bs` — Cálculo Black-Scholes (Delta, Gamma, Vega, etc.)

## 📈 Funcionalidades

- Consulta dinâmica de opções CALL e PUT
- Cálculo de métricas (TIO, Delta, IV proxy)
- Sugestão de Strangles vendidas cobertas
- Exportação CSV dos resultados

## ☁️ Deploy

Pode ser hospedado gratuitamente em [Streamlit Cloud](https://streamlit.io/cloud) ou em qualquer ambiente Python com acesso à internet.
