# app.py
"""
OptyMax — MVP Minimalista
- Lista opções via OPLAB (/market/options/{UNDERLYING})
- Mostra tabela com close (último preço), bid, ask, strike, dtm, spot
- Permite selecionar uma opção por ticker para calcular Greeks via:
    GET /market/options/details/{symbol}
    GET /market/options/bs?...
- NÃO faz recomendações automáticas — você escolhe o contrato para analisar.
"""

import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# --------------------
# Config
# --------------------
OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
st.set_page_config(page_title="OptyMax — MVP Simples", layout="wide")

st.title("OptyMax — MVP Simples (Lista + Análise de uma série)")

# --------------------
# Helpers
# --------------------
@st.cache_data(ttl=600)
def fetch_tickers():
    """Busca tickers (código e nome) - fallback simples"""
    try:
        url = "https://www.dadosdemercado.com.br/acoes"
        r = requests.get(url, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        tickers = []
        for tr in soup.find_all("tr"):
            cols = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cols) >= 2:
                code, name = cols[0].upper(), cols[1]
                if len(code) in (5,6) and any(ch.isdigit() for ch in code):
                    tickers.append((code, name))
        if not tickers:
            return [("PETR4", "Petrobras PN"), ("VALE3", "Vale ON"), ("PSSA3", "PSSA3")]
        # unique preserve order
        seen = {}
        for t,n in tickers:
            if t not in seen:
                seen[t]=n
        return list(seen.items())
    except Exception:
        return [("PETR4", "Petrobras PN"), ("VALE3", "Vale ON"), ("PSSA3", "PSSA3")]

def fetch_options_chain(underlying: str):
    """Chama /market/options/{UNDERLYING} e retorna DataFrame"""
    url = f"{OPLAB_BASE}/market/options/{underlying}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        if isinstance(data, list):
            for it in data:
                rows.append({
                    "symbol": it.get("symbol"),
                    "type": (it.get("type") or it.get("category") or "").upper(),
                    "strike": float(it.get("strike") or 0),
                    "expiration": it.get("due_date"),
                    "bid": float(it.get("bid") or 0),
                    "ask": float(it.get("ask") or 0),
                    "close": float(it.get("close") or 0),
                    "spot": float(it.get("spot_price") or it.get("spot") or 0),
                    "dtm": int(it.get("days_to_maturity") or 0),
                    "open_interest": int(it.get("open_interest") or 0),
                    "volume": int(it.get("volume") or 0),
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Erro ao consultar OPLAB para {underlying}: {e}")
        return pd.DataFrame()

def fetch_option_details(symbol: str):
    """GET /market/options/details/{symbol}"""
    try:
        url = f"{OPLAB_BASE}/market/options/details/{symbol}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Detalhes não encontrados para {symbol}: {e}")
        return None

def fetch_bs(symbol: str, params: dict):
    """GET /market/options/bs with params (expects dict)"""
    try:
        url = f"{OPLAB_BASE}/market/options/bs"
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"BS API falhou para {symbol}: {e}")
        return None

# --------------------
# UI - Sidebar
# --------------------
st.sidebar.header("Configuração")
tickers = fetch_tickers()
options_display = [f"{t} — {n}" for t,n in tickers]
ticker_map = {f"{t} — {n}": t for t,n in tickers}
sel = st.sidebar.selectbox("Escolha um ticker", [""] + options_display)
dtm_min = st.sidebar.number_input("DTM mínimo (dias) - listagem", min_value=0, value=0, step=1)
dtm_max = st.sidebar.number_input("DTM máximo (dias) - listagem", min_value=0, value=999, step=1)
st.sidebar.markdown("---")
st.sidebar.write("Coloque seu OPLAB token em `OPLAB_TOKEN` (Streamlit Secrets).")
st.sidebar.caption("Este MVP não recomenda posições automaticamente — apenas mostra dados e calcula Greeks para a série selecionada.")

# --------------------
# Main flow: list chain when ticker selected
# --------------------
if sel:
    underlying = ticker_map[sel]
    st.header(f"Opções: {underlying}")
    with st.spinner("Consultando OPLAB..."):
        df = fetch_options_chain(underlying)
    if df.empty:
        st.info("Nenhuma série retornada ou erro na API.")
    else:
        # sanitize dtm numeric
        df["dtm"] = pd.to_numeric(df["dtm"], errors="coerce").fillna(0).astype(int)
        # apply DTM filter for display
        df_display = df[(df["dtm"] >= int(dtm_min)) & (df["dtm"] <= int(dtm_max))].copy()
        # show counts
        st.write(f"Total retornado: {len(df)} — Após filtro DTM: {len(df_display)}")
        # present table with relevant columns
        st.dataframe(df_display[["symbol","type","strike","close","bid","ask","dtm","expiration","spot","volume","open_interest"]].sort_values(["type","strike"]))

        st.markdown("### Seleção manual para análise")
        # allow choosing one symbol to analyze (or multiple single-select per ticker)
        selected_symbol = st.selectbox("Escolha a série para calcular Greeks", [""] + df_display["symbol"].tolist())
        if selected_symbol:
            st.write(f"Analisando: **{selected_symbol}**")
            details = fetch_option_details(selected_symbol)
            if details:
                st.json(details)  # show raw details for transparency
                # Build params for BS call using details where possible
                premium = details.get("close") or details.get("bid") or details.get("ask") or 0.01
                params = {
                    "symbol": details.get("symbol"),
                    "irate": 0.1,
                    "type": (details.get("category") or details.get("type") or "").upper(),
                    "spotprice": details.get("spot_price") or details.get("spot") or df_display["spot"].iloc[0],
                    "strike": details.get("strike") or 0,
                    "premium": premium,
                    "dtm": details.get("days_to_maturity") or df_display["dtm"].iloc[0],
                    "vol": details.get("volatility") or 0.25,
                    "duedate": details.get("due_date") or details.get("expiration"),
                    "amount": 100
                }
                st.write("Parâmetros enviados para `/market/options/bs`:")
                st.json(params)
                if st.button("Calcular Greeks"):
                    with st.spinner("Consultando /market/options/bs ..."):
                        bs = fetch_bs(selected_symbol, params)
                        if bs:
                            # show key fields in readable layout
                            st.success("BS retornado com sucesso")
                            # pick common fields if present
                            fields = ["price","delta","gamma","vega","theta","rho","volatility","poe","spotprice","strike","margin"]
                            out = {f: bs.get(f) for f in fields if f in bs}
                            st.table(pd.DataFrame([out]).T.rename(columns={0:"value"}))
                        else:
                            st.error("BS não retornou dados — verifique parâmetros e token.")
            else:
                st.warning("Não foi possível buscar detalhes para essa série.")
else:
    st.info("Selecione um ticker no menu lateral para começar.")
