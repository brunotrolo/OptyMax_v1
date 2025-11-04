
# app.py
import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ===============================================
# CONFIGURAÇÕES GERAIS
# ===============================================
OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
LOT_SIZE = 100

try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

st.set_page_config(page_title="OptyMax — MVP", layout="wide")
st.title("📈 OptyMax — Venda Coberta e Strangle (MVP Final)")

# ===============================================
# FUNÇÕES AUXILIARES
# ===============================================
def fetch_tickers_with_names():
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
                if len(code) in (5, 6) and any(ch.isdigit() for ch in code):
                    tickers.append((code, name))
        if not tickers:
            return [("PETR4", "Petrobras PN"), ("VALE3", "Vale ON"),
                    ("ITUB4", "Itaú Unibanco PN"), ("BBDC4", "Bradesco PN"), ("ABEV3", "Ambev ON")]
        seen = {}
        for t, n in tickers:
            if t not in seen:
                seen[t] = n
        return list(seen.items())
    except Exception:
        return [("PETR4", "Petrobras PN"), ("VALE3", "Vale ON"),
                ("ITUB4", "Itaú Unibanco PN"), ("BBDC4", "Bradesco PN"), ("ABEV3", "Ambev ON")]

def days_to_maturity_from_date(due_str: str):
    try:
        dt = datetime.fromisoformat(due_str.split("T")[0]).date()
        return max((dt - date.today()).days, 0)
    except Exception:
        return 0

def fetch_options_chain_by_parent(parent: str):
    endpoints = [
        f"{OPLAB_BASE}/market/options/series/{parent}",
        f"{OPLAB_BASE}/market/options/chain/{parent}",
        f"{OPLAB_BASE}/market/options/instruments/{parent}",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            items = data.get("data") if isinstance(data, dict) and "data" in data else data
            if not isinstance(items, list):
                continue
            rows = []
            for it in items:
                rows.append({
                    "option_symbol": it.get("symbol"),
                    "type": (it.get("category") or it.get("type") or "").lower(),
                    "strike": float(it.get("strike") or it.get("strike_eod") or 0),
                    "expiration": it.get("due_date"),
                    "bid": float(it.get("bid") or 0),
                    "spot": float(it.get("spot_price") or it.get("spot") or 0),
                    "dtm": int(it.get("days_to_maturity") or days_to_maturity_from_date(it.get("due_date") or "0")),
                })
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            continue
    return pd.DataFrame()

def fetch_bs_oplab(params: dict):
    url = f"{OPLAB_BASE}/market/options/bs"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def compute_tio(total_premium: float, spot_price: float, dtm: int):
    if dtm <= 0 or spot_price <= 0:
        return 0.0
    return round((total_premium / spot_price) * (365 / dtm) * 100, 3)

def compute_iv_rank(symbol: str, iv_today: float):
    if not HAVE_YFINANCE or not iv_today:
        return None
    try:
        data = yf.download(symbol + ".SA", period="1y", progress=False)
        if data.empty:
            return None
        ret = data["Close"].pct_change().dropna()
        vol = ret.rolling(21).std() * (252 ** 0.5)
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin == 0:
            return None
        return round((iv_today - vmin) / (vmax - vmin) * 100, 2)
    except Exception:
        return None

# ===============================================
# INTERFACE DO USUÁRIO
# ===============================================
st.sidebar.header("Filtros — aplicados a CALL e PUT")

tickers_with_names = fetch_tickers_with_names()
options = [f"{t} — {n}" for t, n in tickers_with_names]
ticker_map = {f"{t} — {n}": t for t, n in tickers_with_names}

sel = st.sidebar.multiselect("Selecione até 3 tickers", options, max_selections=3)
dtm_min = st.sidebar.slider("DTM mínimo (dias)", 1, 365, 25)
dtm_max = st.sidebar.slider("DTM máximo (dias)", 1, 365, 60)
delta_min = st.sidebar.number_input("Delta mínimo (valor absoluto)", 0.01, 1.0, 0.10, step=0.01)
delta_max = st.sidebar.number_input("Delta máximo (valor absoluto)", 0.01, 1.0, 0.25, step=0.01)
iv_rank_min = st.sidebar.number_input("IV Rank mínimo (%)", 0.0, 100.0, 0.0, step=1.0)
run = st.sidebar.button("Executar")

# ===============================================
# EXECUÇÃO PRINCIPAL
# ===============================================
if run and sel:
    selected_tickers = [ticker_map[s] for s in sel]
    all_calls, all_puts, all_strangles = [], [], []

    for tk in selected_tickers:
        st.subheader(f"📊 Processando {tk}")
        df_chain = fetch_options_chain_by_parent(tk)

        if df_chain.empty:
            st.warning(f"Sem dados da OPLAB — gerando dados sintéticos para {tk}.")
            spot = 100
            strikes = np.linspace(80, 120, 9)
            rows = []
            for s in strikes:
                for typ in ["call", "put"]:
                    rows.append({
                        "option_symbol": f"{tk}-{typ.upper()}-{s:.2f}",
                        "type": typ,
                        "strike": s,
                        "expiration": (date.today() + timedelta(days=45)).isoformat(),
                        "bid": round(spot * 0.02, 2),
                        "spot": spot,
                        "dtm": 45
                    })
            df_chain = pd.DataFrame(rows)

        df_chain["delta"], df_chain["iv"] = np.nan, np.nan
        for idx, row in df_chain.iterrows():
            try:
                params = {
                    "symbol": row["option_symbol"],
                    "irate": 0.1,
                    "type": row["type"].upper(),
                    "spotprice": row["spot"],
                    "strike": row["strike"],
                    "premium": row["bid"],
                    "dtm": row["dtm"],
                    "vol": 0.3,
                    "duedate": row["expiration"],
                    "amount": LOT_SIZE,
                }
                bs = fetch_bs_oplab(params)
                df_chain.at[idx, "delta"] = bs.get("delta", np.nan)
                df_chain.at[idx, "iv"] = bs.get("volatility", np.nan)
            except Exception:
                continue
            time.sleep(0.02)

        df_chain["delta_abs"] = df_chain["delta"].abs()
        df_chain = df_chain[(df_chain["dtm"] >= dtm_min) & (df_chain["dtm"] <= dtm_max)]
        df_chain = df_chain[(df_chain["delta_abs"] >= delta_min) & (df_chain["delta_abs"] <= delta_max)]

        if HAVE_YFINANCE:
            df_chain["iv_rank"] = df_chain["iv"].apply(lambda v: compute_iv_rank(tk, v) if pd.notna(v) else None)
        else:
            df_chain["iv_rank"] = None

        calls = df_chain[df_chain["type"] == "call"].sort_values(by="bid", ascending=False).head(3)
        puts = df_chain[df_chain["type"] == "put"].sort_values(by="bid", ascending=False).head(3)
        if not calls.empty:
            calls["ticker"] = tk
            all_calls.append(calls)
        if not puts.empty:
            puts["ticker"] = tk
            all_puts.append(puts)

        if not calls.empty and not puts.empty:
            best_call, best_put = calls.iloc[0], puts.iloc[0]
            total_premium = best_call["bid"] + best_put["bid"]
            tio = compute_tio(total_premium, best_call["spot"], best_call["dtm"])
            all_strangles.append({
                "ticker": tk,
                "call_symbol": best_call["option_symbol"],
                "put_symbol": best_put["option_symbol"],
                "total_premium": total_premium,
                "tio": tio,
                "dtm": best_call["dtm"],
                "iv_rank": best_call.get("iv_rank", None)
            })

    if all_calls:
        st.subheader("📈 CALLs Selecionadas")
        dfc = pd.concat(all_calls)
        st.dataframe(dfc[["ticker", "option_symbol", "strike", "dtm", "bid", "delta", "iv", "iv_rank"]])

    if all_puts:
        st.subheader("📉 PUTs Selecionadas")
        dfp = pd.concat(all_puts)
        st.dataframe(dfp[["ticker", "option_symbol", "strike", "dtm", "bid", "delta", "iv", "iv_rank"]])

    if all_strangles:
        st.subheader("🔁 Strangles Montados")
        dfs = pd.DataFrame(all_strangles).sort_values(by="tio", ascending=False)
        st.dataframe(dfs)
        st.download_button("💾 Exportar CSV", dfs.to_csv(index=False), "strangles.csv")

else:
    st.info("Selecione até 3 tickers e clique em Executar.")
