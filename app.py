# app.py
"""
OptyMax MVP - Streamlit single-file app (Option B)
Integrado corretamente com a API OPLAB (v3)
"""
import os
import math
from datetime import datetime, date
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ---------------- Configurações Gerais ----------------
OPLAB_TOKEN = os.environ.get(
    "OPLAB_TOKEN",
    "AnJFCmWtZiSCL9Up1F2slrKpbhg/SIUuWj7ohDwxQ4Uvk1/2CY9bUI8KaPofVzT0--X8vvuqmk7JeKDuYquob/lA==--MzVlYTVhYzY0ODkyM2Y0Y2ZlOTkwMjcyNTM2ZWFjNDg="
)
OPLAB_BASE = "https://api.oplab.com.br/v3"
LOT_SIZE = 100

# ---------------- Funções Auxiliares ----------------
def days_to_maturity(expiration_iso: str) -> int:
    try:
        exp = datetime.fromisoformat(expiration_iso).date()
        today = date.today()
        return max((exp - today).days, 0)
    except Exception:
        return 0

def fetch_option_details(symbol: str) -> dict:
    """Consulta detalhes de uma opção individual pela API OPLAB v3."""
    url = f"{OPLAB_BASE}/market/options/details/{symbol}"
    headers = {"Access-Token": OPLAB_TOKEN}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Erro ao consultar detalhes da opção {symbol}: {e}")
        return {}

def fetch_bs_data(symbol: str, params: dict) -> dict:
    """Consulta os dados de Black-Scholes de uma opção específica."""
    url = f"{OPLAB_BASE}/market/options/bs"
    headers = {"Access-Token": OPLAB_TOKEN}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Erro ao consultar Black-Scholes para {symbol}: {e}")
        return {}

def load_option_chain(parent_symbol: str, strikes: list) -> pd.DataFrame:
    """Gera um DataFrame com opções individuais de um ativo base (parent_symbol)."""
    data = []
    for strike in strikes:
        for opt_type in ["CALL", "PUT"]:
            opt_symbol = f"{parent_symbol}{'E' if opt_type=='CALL' else 'F'}{int(strike*100)}"  # placeholder naming
            details = fetch_option_details(opt_symbol)
            if not details:
                continue
            bs_data = fetch_bs_data(opt_symbol, {
                "symbol": opt_symbol,
                "irate": 0.1,
                "type": opt_type,
                "spotprice": details.get("spot_price", 0),
                "strike": details.get("strike", 0),
                "premium": details.get("bid", 0),
                "dtm": details.get("days_to_maturity", 0),
                "vol": 0.3,
                "duedate": details.get("due_date"),
                "amount": LOT_SIZE
            })
            data.append({
                "symbol": parent_symbol,
                "option_symbol": opt_symbol,
                "type": opt_type.lower(),
                "strike": details.get("strike", 0),
                "expiration": details.get("due_date"),
                "bid": details.get("bid", 0),
                "ask": details.get("ask", 0),
                "spot": details.get("spot_price", 0),
                "dtm": details.get("days_to_maturity", 0),
                "delta": bs_data.get("delta", 0),
                "iv": bs_data.get("volatility", 0),
                "poe": bs_data.get("poe", 0)
            })
    return pd.DataFrame(data)

def compute_tio(premium, spot, dtm, lot=LOT_SIZE):
    if dtm <= 0:
        return 0.0
    capital = spot * lot
    returns = premium * lot
    tio = (returns / capital) * (365.0 / dtm) * 100.0
    return round(tio, 3)

def pick_top(df, n=3):
    return df.sort_values(by=["bid", "dtm"], ascending=[False, True]).head(n)

def propose_strangles(df, spot, dtm_target=None, max_pairs=5):
    pairs = []
    for dtm, group in df.groupby("dtm"):
        calls = pick_top(group[group["type"]=="call"], 3)
        puts = pick_top(group[group["type"]=="put"], 3)
        if calls.empty or puts.empty:
            continue
        if dtm_target and abs(dtm - dtm_target) > 15:
            continue
        for _, c in calls.iterrows():
            for _, p in puts.iterrows():
                total_premium = c["bid"] + p["bid"]
                tio = compute_tio(total_premium, spot, dtm)
                pairs.append({
                    "symbol": c["symbol"],
                    "dtm": dtm,
                    "call_symbol": c["option_symbol"],
                    "put_symbol": p["option_symbol"],
                    "total_premium": total_premium,
                    "tio": tio
                })
    return pd.DataFrame(pairs).sort_values(by="tio", ascending=False).head(max_pairs)

# ---------------- Interface Streamlit ----------------
st.set_page_config(page_title="OptyMax MVP (OPLAB API)", layout="wide")
st.title("OptyMax — MVP com integração real OPLAB v3")

with st.sidebar:
    st.header("Parâmetros")
    tickers = st.text_input("Tickers", "PETR4,VALE3")
    strikes = st.text_input("Strikes (separados por vírgula)", "25,30,35")
    dtm_target = st.number_input("DTM preferencial (dias)", 0, 365, 45)
    btn = st.button("Executar")

if btn:
    for tk in [t.strip().upper() for t in tickers.split(",") if t.strip()]:
        st.subheader(f"Ticker: {tk}")
        df = load_option_chain(tk, [float(s) for s in strikes.split(",") if s.strip()])
        if df.empty:
            st.warning(f"Sem dados retornados para {tk}")
            continue
        st.write("Amostra das opções:")
        st.dataframe(df)
        st.write("CALLs principais:")
        st.dataframe(pick_top(df[df["type"]=="call"]))
        st.write("PUTs principais:")
        st.dataframe(pick_top(df[df["type"]=="put"]))
        st.write("Strangles sugeridos:")
        st.dataframe(propose_strangles(df, df['spot'].mean(), dtm_target))
        st.download_button("Baixar CSV", df.to_csv(index=False), f"{tk}_options.csv")
else:
    st.info("Insira os tickers e strikes e clique em Executar.")
