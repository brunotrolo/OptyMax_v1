# app.py
"""
OptyMax — MVP Final com integração real com API OPLAB e tracking em tempo real
-----------------------------------------------------------------------------
- Busca real das opções via endpoint /market/options/{UNDERLYING}
- Barra de progresso e log dinâmico de execução
- Delta min/max aplicados a CALL e PUT
- Cálculo de TIO e IV Rank (via yfinance)
- Sem geração de dados sintéticos
"""

import os
import time
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
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
st.title("📈 OptyMax — Venda Coberta e Strangle (OPLAB v3 + Tracking Tempo Real)")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def fetch_tickers_with_names():
    """Obtém lista de (ticker, nome da empresa) da B3 via dadosdemercado.com.br"""
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


def fetch_options_chain_by_parent(parent: str, log_box):
    """Obtém lista de opções de um ativo base diretamente da API OPLAB"""
    url = f"{OPLAB_BASE}/market/options/{parent}"
    try:
        log_box.text(f"[{parent}] 🔍 Consultando opções na OPLAB...")
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                rows = []
                for it in data:
                    rows.append({
                        "option_symbol": it.get("symbol"),
                        "type": (it.get("type") or "").upper(),
                        "strike": float(it.get("strike") or 0),
                        "expiration": it.get("due_date"),
                        "bid": float(it.get("bid") or 0),
                        "ask": float(it.get("ask") or 0),
                        "spot": float(it.get("spot_price") or 0),
                        "dtm": int(it.get("days_to_maturity") or 0),
                        "open_interest": int(it.get("open_interest") or 0),
                        "volume": int(it.get("volume") or 0),
                        "parent_symbol": parent
                    })
                log_box.text(f"[{parent}] ✅ {len(rows)} opções carregadas da OPLAB.")
                return pd.DataFrame(rows)
            else:
                log_box.text(f"[{parent}] ⚠️ Nenhum dado de opção retornado.")
        else:
            log_box.text(f"[{parent}] ❌ Erro HTTP {r.status_code}")
    except Exception as e:
        log_box.text(f"[{parent}] ❌ Erro ao consultar API OPLAB: {e}")
    return pd.DataFrame()


def fetch_bs_oplab(params: dict, log_box):
    """Consulta modelo Black-Scholes na OPLAB"""
    url = f"{OPLAB_BASE}/market/options/bs"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log_box.text(f"Erro Black-Scholes: {e}")
    return {}


def compute_tio(total_premium: float, spot_price: float, dtm: int):
    """TIO anualizado"""
    if dtm <= 0 or spot_price <= 0:
        return 0.0
    return round((total_premium / spot_price) * (365 / dtm) * 100, 3)


def compute_iv_rank(symbol: str, iv_today: float):
    """Calcula IV Rank com base na volatilidade histórica"""
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

# ============================================================
# INTERFACE DO USUÁRIO
# ============================================================
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

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if run and sel:
    selected_tickers = [ticker_map[s] for s in sel]
    all_calls, all_puts, all_strangles = [], [], []

    progress_text = st.empty()
    log_box = st.empty()
    progress_bar = st.progress(0)

    total = len(selected_tickers)
    for i, tk in enumerate(selected_tickers, start=1):
        progress_text.markdown(f"🚀 **Processando `{tk}` ({i}/{total})**")
        log_box.text(f"[{tk}] Iniciando processamento...")

        # 1️⃣ Buscar opções da OPLAB
        df_chain = fetch_options_chain_by_parent(tk, log_box)
        if df_chain.empty:
            st.warning(f"Nenhum dado retornado para {tk}. Verifique token OPLAB.")
            progress_bar.progress(i / total)
            continue

        # 2️⃣ Consultar Black-Scholes
        log_box.text(f"[{tk}] Calculando Greeks via Black-Scholes...")
        df_chain["delta"], df_chain["iv"] = np.nan, np.nan
        for idx, row in df_chain.iterrows():
            try:
                params = {
                    "symbol": row["option_symbol"],
                    "irate": 0.1,
                    "type": row["type"],
                    "spotprice": row["spot"],
                    "strike": row["strike"],
                    "premium": row["bid"],
                    "dtm": row["dtm"],
                    "vol": 0.3,
                    "duedate": row["expiration"],
                    "amount": LOT_SIZE,
                }
                bs = fetch_bs_oplab(params, log_box)
                df_chain.at[idx, "delta"] = bs.get("delta", np.nan)
                df_chain.at[idx, "iv"] = bs.get("volatility", np.nan)
            except Exception:
                continue
            time.sleep(0.02)

        # 3️⃣ Aplicar filtros
        log_box.text(f"[{tk}] Aplicando filtros e selecionando melhores CALLs/PUTs...")
        df_chain["delta_abs"] = df_chain["delta"].abs()
        df_chain = df_chain[(df_chain["dtm"] >= dtm_min) & (df_chain["dtm"] <= dtm_max)]
        df_chain = df_chain[(df_chain["delta_abs"] >= delta_min) & (df_chain["delta_abs"] <= delta_max)]

        # 4️⃣ Calcular IV Rank
        if HAVE_YFINANCE:
            df_chain["iv_rank"] = df_chain["iv"].apply(lambda v: compute_iv_rank(tk, v) if pd.notna(v) else None)
        else:
            df_chain["iv_rank"] = None

        # 5️⃣ Selecionar CALLs e PUTs
        calls = df_chain[df_chain["type"] == "CALL"].sort_values(by="bid", ascending=False).head(3)
        puts = df_chain[df_chain["type"] == "PUT"].sort_values(by="bid", ascending=False).head(3)
        if not calls.empty:
            calls["ticker"] = tk
            all_calls.append(calls)
        if not puts.empty:
            puts["ticker"] = tk
            all_puts.append(puts)

        # 6️⃣ Montar Strangle
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
        log_box.text(f"[{tk}] ✅ Finalizado.")
        progress_bar.progress(i / total)

    progress_text.markdown("✅ **Processamento concluído!**")
    log_box.text("Todos os tickers foram processados com sucesso.")
    progress_bar.empty()

    # Exibir resultados finais
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
