# app.py
"""
OptyMax — MVP Final (Estratégia Conservadora)
-----------------------------------------------------------
Critérios:
- CALL: strike > spot * 1.03 (mínimo 3% OTM)
- PUT: strike < spot * 0.97 (mínimo 3% OTM)
- Liquidez mínima: volume ≥ 10 e open interest ≥ 50
- Delta: 0.12 ≤ |Δ| ≤ 0.28
- Sem fallback de delta estimado (somente dados reais da API)
"""

import os
import time
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

st.set_page_config(page_title="OptyMax — MVP", layout="wide")
st.title("📈 OptyMax — Estratégia Conservadora (Venda Coberta e Strangle OTM)")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def fetch_tickers_with_names():
    """Obtém lista de tickers e nomes da B3"""
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
            return [("PETR4","Petrobras PN"),("VALE3","Vale ON"),("ITUB4","Itaú Unibanco PN")]
        seen = {}
        for t, n in tickers:
            if t not in seen: seen[t] = n
        return list(seen.items())
    except Exception:
        return [("PETR4","Petrobras PN"),("VALE3","Vale ON"),("ITUB4","Itaú Unibanco PN")]

def fetch_options_chain_by_parent(parent: str, log_box):
    """Obtém lista de opções via API OPLAB"""
    url = f"{OPLAB_BASE}/market/options/{parent}"
    try:
        log_box.text(f"[{parent}] 🔍 Consultando opções na OPLAB...")
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            rows = []
            for it in data:
                rows.append({
                    "option_symbol": it.get("symbol"),
                    "type": (it.get("type") or "").upper(),
                    "strike": float(it.get("strike") or 0),
                    "expiration": it.get("due_date"),
                    "bid": float(it.get("bid") or 0),
                    "ask": float(it.get("ask") or 0),
                    "close": float(it.get("close") or 0),
                    "spot": float(it.get("spot_price") or 0),
                    "dtm": int(it.get("days_to_maturity") or 0),
                    "open_interest": int(it.get("open_interest") or 0),
                    "volume": int(it.get("volume") or 0),
                    "parent_symbol": parent
                })
            log_box.text(f"[{parent}] ✅ {len(rows)} opções carregadas.")
            return pd.DataFrame(rows)
        else:
            log_box.text(f"[{parent}] ❌ Erro HTTP {r.status_code}")
    except Exception as e:
        log_box.text(f"[{parent}] ❌ Erro ao consultar API: {e}")
    return pd.DataFrame()

def fetch_option_details(symbol: str):
    """Obtém dados reais de uma opção via /market/options/details/{symbol}"""
    try:
        url = f"{OPLAB_BASE}/market/options/details/{symbol}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def fetch_bs_oplab_accurate(symbol: str, log_box):
    """Consulta Black-Scholes com dados reais da OPLAB"""
    details = fetch_option_details(symbol)
    if not details:
        return {}
    try:
        params = {
            "symbol": details.get("symbol"),
            "irate": 0.1,
            "type": details.get("category", "").upper(),
            "spotprice": float(details.get("spot_price", 0)),
            "strike": float(details.get("strike", 0)),
            "premium": float(details.get("close", 0) or details.get("bid", 0) or details.get("ask", 0) or 0.01),
            "dtm": int(details.get("days_to_maturity", 0)),
            "vol": float(details.get("volatility", 0.25)),
            "duedate": details.get("due_date", ""),
            "amount": 100
        }
        r = requests.get(f"{OPLAB_BASE}/market/options/bs", headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log_box.text(f"Erro em BS detalhado: {e}")
    return {}

def compute_tio(total_premium: float, spot_price: float, dtm: int):
    if dtm <= 0 or spot_price <= 0:
        return 0.0
    return round((total_premium / spot_price) * (365 / dtm) * 100, 3)

# ============================================================
# INTERFACE
# ============================================================
st.sidebar.header("Filtros — aplicados a CALL e PUT")
tickers = fetch_tickers_with_names()
opts = [f"{t} — {n}" for t, n in tickers]
ticker_map = {f"{t} — {n}": t for t, n in tickers}

sel = st.sidebar.multiselect("Selecione até 3 tickers", opts, max_selections=3)
dtm_min = st.sidebar.slider("DTM mínimo (dias)", 1, 365, 25)
dtm_max = st.sidebar.slider("DTM máximo (dias)", 1, 365, 60)

listar = st.sidebar.button("📋 Listar Opções")
processar = st.sidebar.button("⚙️ Gerar Recomendações (Conservadora)")

if "opcoes" not in st.session_state:
    st.session_state["opcoes"] = {}

# ============================================================
# ETAPA 1 — LISTAGEM
# ============================================================
if listar and sel:
    st.session_state["opcoes"].clear()
    progress_text, log_box = st.empty(), st.empty()
    progress_bar = st.progress(0)
    selected = [ticker_map[s] for s in sel]
    total = len(selected)

    for i, tk in enumerate(selected, start=1):
        progress_text.markdown(f"📊 **Listando `{tk}` ({i}/{total})**")
        df = fetch_options_chain_by_parent(tk, log_box)
        if not df.empty:
            df["dtm"] = pd.to_numeric(df["dtm"], errors="coerce").fillna(0).astype(int)
            df = df[(df["dtm"] >= dtm_min) & (df["dtm"] <= dtm_max)]
            if not df.empty:
                st.session_state["opcoes"][tk] = df
                st.subheader(f"📈 {tk} — {len(df)} opções (DTM {dtm_min}-{dtm_max})")
                st.dataframe(df[["option_symbol","type","strike","bid","ask","close","expiration","dtm","spot"]])
            else:
                st.warning(f"⚠️ Nenhuma opção no intervalo DTM para {tk}.")
        progress_bar.progress(i/total)
    progress_text.markdown("✅ **Listagem concluída!**")
    progress_bar.empty()

# ============================================================
# ETAPA 2 — PROCESSAMENTO CONSERVADOR
# ============================================================
if processar:
    if not st.session_state["opcoes"]:
        st.warning("⚠️ Nenhuma opção listada. Clique primeiro em '📋 Listar Opções'.")
    else:
        progress_text, log_box = st.empty(), st.empty()
        progress_bar = st.progress(0)
        all_calls, all_puts, all_strangles = [], [], []
        total = len(st.session_state["opcoes"])

        for i, tk in enumerate(st.session_state["opcoes"], start=1):
            df_chain = st.session_state["opcoes"][tk]
            progress_text.markdown(f"⚙️ **Processando `{tk}` ({i}/{total})**")
            log_box.text(f"[{tk}] Calculando Black-Scholes (modo conservador)...")

            df_chain["delta"], df_chain["iv"] = np.nan, np.nan
            for idx, row in df_chain.iterrows():
                try:
                    bs = fetch_bs_oplab_accurate(row["option_symbol"], log_box)
                    if bs and "delta" in bs:
                        df_chain.at[idx,"delta"] = float(bs.get("delta", np.nan))
                        df_chain.at[idx,"iv"] = float(bs.get("volatility", np.nan))
                except Exception as e:
                    log_box.text(f"[{tk}] Erro BS: {e}")
                    continue
                time.sleep(0.02)

            # 🔹 Filtros conservadores
            df_chain["dtm"]=pd.to_numeric(df_chain["dtm"],errors="coerce").fillna(0).astype(int)
            df_chain["delta_abs"]=df_chain["delta"].abs()
            df_chain=df_chain[(df_chain["dtm"]>=dtm_min)&(df_chain["dtm"]<=dtm_max)]
            df_chain=df_chain[(df_chain["delta_abs"]>=0.12)&(df_chain["delta_abs"]<=0.28)]
            df_chain=df_chain[(df_chain["volume"]>=10)&(df_chain["open_interest"]>=50)]

            # 🔹 Apenas OTM (3% fora do dinheiro)
            df_chain["is_otm"]=np.where(
                ((df_chain["type"]=="CALL")&(df_chain["strike"]>df_chain["spot"]*1.03))|
                ((df_chain["type"]=="PUT")&(df_chain["strike"]<df_chain["spot"]*0.97)),
                True,False)
            df_chain=df_chain[df_chain["is_otm"]]

            # 🔹 Seleção final
            calls=(df_chain[df_chain["type"]=="CALL"]
                .sort_values(by=["strike"],ascending=True).head(3))
            puts=(df_chain[df_chain["type"]=="PUT"]
                .sort_values(by=["strike"],ascending=False).head(3))

            if not calls.empty:
                calls["ticker"]=tk; all_calls.append(calls)
            if not puts.empty:
                puts["ticker"]=tk; all_puts.append(puts)

            if not calls.empty and not puts.empty:
                best_call,best_put=calls.iloc[0],puts.iloc[0]
                total_prem=best_call["close"]+best_put["close"]
                tio=compute_tio(total_prem,best_call["spot"],best_call["dtm"])
                all_strangles.append({
                    "ticker":tk,
                    "call_symbol":best_call["option_symbol"],
                    "put_symbol":best_put["option_symbol"],
                    "total_premium":total_prem,
                    "tio":tio,
                    "dtm":best_call["dtm"]
                })
            progress_bar.progress(i/total)

        progress_text.markdown("✅ **Processamento concluído — Estratégia Conservadora!**")
        progress_bar.empty()

        if all_calls:
            st.subheader("📈 CALLs OTM Selecionadas (Conservadoras)")
            st.dataframe(pd.concat(all_calls)[["ticker","option_symbol","strike","spot","dtm","close","delta","volume","open_interest"]])
        if all_puts:
            st.subheader("📉 PUTs OTM Selecionadas (Conservadoras)")
            st.dataframe(pd.concat(all_puts)[["ticker","option_symbol","strike","spot","dtm","close","delta","volume","open_interest"]])
        if all_strangles:
            st.subheader("🔁 Strangles Montados (Conservadores)")
            dfs=pd.DataFrame(all_strangles).sort_values(by="tio",ascending=False)
            st.dataframe(dfs)
            st.download_button("💾 Exportar CSV",dfs.to_csv(index=False),"strangles.csv")
