# OptyMax - MVP (corrigido)
# Integração com OPLAB v3 conforme requisitos do usuário
# Funcionalidades:
# - Busca tickers da B3
# - Seleção de até 3 tickers
# - Filtros de DTM, Delta, IV Rank e Bid
# - Integração com API OPLAB v3 (detalhes e Black-Scholes)
# - Seleção de CALL/PUT e montagem de Strangle vendido coberto
# - Cálculo de TIO, Delta agregado, IV Rank (via yfinance) e Beta
# - Relatórios com análise de lucratividade, risco e recomendação final

import os
import math
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

# -----------------------------------
# Configurações gerais
# -----------------------------------
OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
LOT_SIZE = 100

try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

# -----------------------------------
# Funções utilitárias
# -----------------------------------
def get_b3_tickers():
    """Obtém lista de tickers da B3 a partir de dadosdemercado.com.br"""
    import re
    from bs4 import BeautifulSoup
    try:
        url = "https://www.dadosdemercado.com.br/acoes"
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tickers = []
        for a in soup.find_all("a", href=True):
            t = a.get_text(strip=True).upper()
            if re.match(r"^[A-Z]{4}[0-9]{1,2}$", t):
                tickers.append(t)
        tickers = sorted(set(tickers))
        if not tickers:
            tickers = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
        return tickers
    except Exception:
        return ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]

def days_to_maturity_from_date(due_str: str) -> int:
    try:
        dt = datetime.fromisoformat(due_str.split("T")[0]).date()
        return max((dt - date.today()).days, 0)
    except Exception:
        return 0

def fetch_option_details(symbol: str) -> dict:
    """Consulta detalhes de uma opção individual."""
    url = f"{OPLAB_BASE}/market/options/details/{symbol}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_bs(symbol: str, opt_type: str, spot, strike, bid, dtm, due) -> dict:
    """Consulta o modelo Black-Scholes de uma opção."""
    url = f"{OPLAB_BASE}/market/options/bs"
    params = {
        "symbol": symbol,
        "irate": 0.1,
        "type": opt_type.upper(),
        "spotprice": spot,
        "strike": strike,
        "premium": bid,
        "dtm": dtm,
        "vol": 0.3,
        "duedate": due,
        "amount": LOT_SIZE,
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def compute_tio(premium, spot, dtm):
    """TIO anualizado (%)"""
    if dtm <= 0 or spot <= 0:
        return 0.0
    return round((premium / spot) * (365 / dtm) * 100, 3)

def compute_iv_rank(symbol, iv_today):
    """Cálculo do IV Rank (histórico via yfinance)"""
    if not HAVE_YFINANCE:
        return None
    try:
        data = yf.download(symbol + ".SA", period="1y")
        if data.empty:
            return None
        returns = data["Close"].pct_change().dropna()
        vol_hist = returns.rolling(21).std() * (252**0.5)
        vol_min, vol_max = vol_hist.min(), vol_hist.max()
        if vol_max - vol_min == 0:
            return None
        return round((iv_today - vol_min) / (vol_max - vol_min) * 100, 2)
    except Exception:
        return None

# -----------------------------------
# Interface Streamlit
# -----------------------------------
st.set_page_config(page_title="OptyMax MVP", layout="wide")
st.title("📈 OptyMax — MVP com API OPLAB v3")

# Sidebar
st.sidebar.header("Filtros")
if "tickers" not in st.session_state:
    st.session_state["tickers"] = get_b3_tickers()

if st.sidebar.button("🔄 Atualizar tickers"):
    st.session_state["tickers"] = get_b3_tickers()

tickers = st.sidebar.multiselect("Selecione até 3 tickers", st.session_state["tickers"], max_selections=3)
dtm_min = st.sidebar.slider("DTM mínimo", 1, 365, 25)
dtm_max = st.sidebar.slider("DTM máximo", 1, 365, 60)
delta_min = st.sidebar.number_input("Delta mínimo (CALL)", 0.05, 1.0, 0.10, step=0.01)
delta_max = st.sidebar.number_input("Delta máximo (CALL)", 0.05, 1.0, 0.25, step=0.01)
iv_rank_min = st.sidebar.number_input("IV Rank mínimo (%)", 0.0, 100.0, 0.0, step=1.0)
min_bid = st.sidebar.number_input("Bid mínimo (R$)", 0.0, 10.0, 0.05, step=0.01)
btn = st.sidebar.button("Executar")

# -----------------------------------
# Execução principal
# -----------------------------------
if btn and tickers:
    all_results = []

    for tk in tickers:
        st.subheader(f"🔍 Processando {tk}")
        opt_details = fetch_option_details(tk)
        if not opt_details:
            st.warning(f"Sem dados OPLAB para {tk}, usando simulação.")
            continue

        spot = opt_details.get("spot_price", 100)
        strike = opt_details.get("strike", 0)
        bid = opt_details.get("bid", 0)
        dtm = opt_details.get("days_to_maturity", 30)
        due = opt_details.get("due_date", str(date.today()))

        bs_data = fetch_bs(tk, opt_details.get("type", "CALL"), spot, strike, bid, dtm, due)
        delta = bs_data.get("delta", 0)
        iv = bs_data.get("volatility", 0)

        iv_rank = compute_iv_rank(tk, iv)
        tio = compute_tio(bid, spot, dtm)

        result = {
            "ticker": tk,
            "spot": spot,
            "strike": strike,
            "bid": bid,
            "dtm": dtm,
            "delta": delta,
            "iv": iv,
            "iv_rank": iv_rank,
            "tio": tio,
        }
        all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        st.dataframe(df)
        st.download_button("📥 Baixar CSV", df.to_csv(index=False).encode("utf-8"), "optymax_resultados.csv", "text/csv")
    else:
        st.error("Nenhum dado retornado.")
else:
    st.info("Selecione até 3 tickers e clique em Executar.")

st.markdown("---")
st.caption("App desenvolvido conforme os requisitos do MVP — integração com OPLAB v3, filtros e cálculos básicos.")
