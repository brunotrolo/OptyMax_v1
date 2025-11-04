# app.py
"""
OptyMax — MVP (ajustes conforme pedido)
- Carrega tickers B3 automaticamente (com nome da empresa quando possível)
- Delta min/max aplicados a CALL e PUT (usando valor absoluto)
- Sem campo 'Bid mínimo'
- Integração básica com OPLAB v3 (endpoints usados quando disponíveis)
"""
import os
import time
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict, Optional
import requests
import pandas as pd
import numpy as np
import streamlit as st

# optional yfinance for IV Rank / beta (if installed)
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
LOT_SIZE = 100

st.set_page_config(page_title="OptyMax — MVP (ajustes)", layout="wide")
st.title("OptyMax — Venda Coberta & Strangle (B3) — MVP (ajustes)")

# ---------------- Utilities ----------------
def fetch_tickers_with_names() -> List[Tuple[str, str]]:
    """
    Tenta fazer scraping em dadosdemercado.com.br/acoes para retornar lista de (ticker, company_name).
    Se falhar, retorna lista padrão de tickers (sem nomes).
    """
    try:
        url = "https://www.dadosdemercado.com.br/acoes"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        # heurística: procurar linhas/tags que contenham ticker e nome
        results = []
        # procura por tabelas que parecem ter nome e código
        tables = soup.find_all("table")
        for table in tables:
            for row in table.find_all("tr"):
                cols = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
                if not cols or len(cols) < 2:
                    continue
                # heurística: um dos cols é ticker (ex: PETR4) e outro é nome extenso
                for c in cols:
                    txt = c.strip().upper()
                    if len(txt) >= 4 and len(txt) <= 6 and any(ch.isdigit() for ch in txt):
                        ticker = txt
                        # company name = first other col with length>3
                        name_candidates = [x for x in cols if x and x != c and len(x) > 3]
                        name = name_candidates[0] if name_candidates else ""
                        results.append((ticker, name))
        # fallback: search anchors
        if not results:
            anchors = soup.find_all("a")
            for a in anchors:
                txt = a.get_text(strip=True)
                if not txt:
                    continue
                txtu = txt.upper()
                if len(txtu) >= 4 and len(txtu) <= 6 and any(ch.isdigit() for ch in txtu):
                    # try to find nearby element as company name
                    parent = a.parent
                    name = ""
                    if parent:
                        # siblings text
                        sibs = [s.get_text(strip=True) for s in parent.find_all() if s != a]
                        name = next((s for s in sibs if len(s) > 3), "")
                    results.append((txtu, name))
        # deduplicate preserving first
        seen = {}
        for t, n in results:
            if t not in seen:
                seen[t] = n
        items = list(seen.items())
        if not items:
            # fallback hardcoded
            return [("PETR4","Petrobras PN"), ("VALE3","Vale ON"), ("ITUB4","Itau Unibanco PN"), ("BBDC4","Bradesco PN"), ("ABEV3","Ambev ON")]
        return items
    except Exception:
        return [("PETR4","Petrobras PN"), ("VALE3","Vale ON"), ("ITUB4","Itau Unibanco PN"), ("BBDC4","Bradesco PN"), ("ABEV3","Ambev ON")]

def days_to_maturity_from_date(due_str: str) -> int:
    try:
        dt = datetime.fromisoformat(due_str.split("T")[0]).date()
        return max((dt - date.today()).days, 0)
    except Exception:
        try:
            return int(due_str)
        except Exception:
            return 0

def fetch_options_chain_by_parent(parent: str) -> pd.DataFrame:
    """
    Tenta vários endpoints plausíveis da OPLAB para retornar a cadeia de opções do ativo parent.
    Retorna DataFrame com colunas mínimas: option_symbol, type (call/put), strike, expiration, bid, ask, spot, dtm, open_interest, volume
    """
    endpoints = [
        f"{OPLAB_BASE}/market/options/series/{parent}",
        f"{OPLAB_BASE}/market/options/chain/{parent}",
        f"{OPLAB_BASE}/market/options/instruments/{parent}",
        f"{OPLAB_BASE}/market/options/list/{parent}"
    ]
    for url in endpoints:
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            j = r.json()
            items = j.get("data") if isinstance(j, dict) and "data" in j else j
            if not isinstance(items, list):
                items = list(items)
            rows = []
            for it in items:
                try:
                    rows.append({
                        "option_symbol": it.get("symbol") or it.get("option_symbol") or it.get("name"),
                        "type": (it.get("category") or it.get("type") or "").lower(),
                        "strike": float(it.get("strike") or it.get("strike_eod") or 0),
                        "expiration": it.get("due_date") or it.get("dueDate") or it.get("due_date"),
                        "bid": float(it.get("bid") or 0),
                        "ask": float(it.get("ask") or 0),
                        "spot": float(it.get("spot_price") or it.get("spotprice") or it.get("spot") or 0),
                        "dtm": int(it.get("days_to_maturity") or days_to_maturity_from_date(it.get("due_date") or "0")),
                        "open_interest": int(it.get("open_interest") or it.get("openInterest") or 0),
                        "volume": int(it.get("volume") or 0),
                        "parent_symbol": it.get("parent_symbol") or parent
                    })
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            continue
    # fallback vazio (o chamador decide usar dados sintéticos)
    return pd.DataFrame()

def fetch_option_details_oplab(option_symbol: str) -> dict:
    url = f"{OPLAB_BASE}/market/options/details/{option_symbol}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def fetch_bs_oplab(params: dict) -> dict:
    url = f"{OPLAB_BASE}/market/options/bs"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

def compute_tio(total_premium: float, spot_price: float, dtm: int) -> float:
    if dtm <= 0 or spot_price <= 0:
        return 0.0
    return round((total_premium / spot_price) * (365.0 / dtm) * 100.0, 3)

def compute_iv_rank(symbol: str, iv_today: float) -> Optional[float]:
    if not HAVE_YFINANCE or iv_today is None:
        return None
    try:
        yf_sym = symbol + ".SA"
        t = yf.Ticker(yf_sym)
        hist = t.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 30:
            return None
        ret = hist['Close'].pct_change().dropna()
        rolling_vol = ret.rolling(window=21).std() * (252**0.5)
        rv = rolling_vol.dropna()
        if rv.empty:
            return None
        v_min = rv.min()
        v_max = rv.max()
        if v_max - v_min == 0:
            return None
        rank = (iv_today - v_min) / (v_max - v_min) * 100.0
        return float(max(0.0, min(100.0, round(rank, 2))))
    except Exception:
        return None

# ---------------- UI (inputs) ----------------
st.sidebar.header("Filtros principais (aplicam a CALL e PUT)")
tickers_with_names = fetch_tickers_with_names()  # carregamento automático
# build mapping label -> ticker
options = [f\"{t} — {n}\" if n else t for t, n in tickers_with_names]
ticker_map = { (f\"{t} — {n}\" if n else t): t for t, n in tickers_with_names }

sel = st.sidebar.multiselect(\"Selecione até 3 tickers\", options=options, max_selections=3)
if not sel:
    st.sidebar.info(\"Selecione até 3 tickers para iniciar a busca.\")
dtm_min = st.sidebar.slider(\"DTM mínimo (dias)\", 1, 365, 25)
dtm_max = st.sidebar.slider(\"DTM máximo (dias)\", 1, 365, 60)
delta_min = st.sidebar.number_input(\"Delta mínimo (valor absoluto, para CALL e PUT)\", 0.01, 1.0, 0.10, step=0.01)
delta_max = st.sidebar.number_input(\"Delta máximo (valor absoluto, para CALL e PUT)\", 0.01, 1.0, 0.25, step=0.01)
iv_rank_min = st.sidebar.number_input(\"IV Rank mínimo (%) (quando disponível)\", 0.0, 100.0, 0.0, step=1.0)

run = st.sidebar.button(\"Executar\")

# ---------------- Execução ----------------
if run and sel:
    selected_tickers = [ticker_map[s] for s in sel]
    all_calls = []
    all_puts = []
    all_strangles = []

    for tk in selected_tickers:
        st.subheader(f\"Processando {tk}\")
        df_chain = fetch_options_chain_by_parent(tk)

        if df_chain.empty:
            st.warning(f\"Cadeia não disponível para {tk} — usando fallback sintético.\")
            # synthetic fallback (45 dias)
            spot_guess = 100.0
            strikes = np.round(np.linspace(spot_guess * 0.8, spot_guess * 1.2, 13), 2)
            rows = []
            for s in strikes:
                for typ in ['call','put']:
                    rows.append({
                        'option_symbol': f\"{tk}-{typ[0].upper()}-{s}\",
                        'type': typ,
                        'strike': s,
                        'expiration': (date.today() + timedelta(days=45)).isoformat(),
                        'bid': round(max(0.01, (spot_guess * 0.03) * (1 + abs((spot_guess - s) / spot_guess))), 2),
                        'ask': 0.0,
                        'spot': spot_guess,
                        'dtm': 45,
                        'open_interest': 10,
                        'volume': 10
                    })
            df_chain = pd.DataFrame(rows)

        # try to fetch greeks via BS endpoint for each option (best-effort)
        if 'delta' not in df_chain.columns or df_chain['delta'].isnull().all():
            df_chain['delta'] = np.nan
            df_chain['iv'] = np.nan
            for idx, row in df_chain.iterrows():
                try:
                    params = {
                        'symbol': row.get('option_symbol'),
                        'irate': 0.10,
                        'type': (row.get('type') or 'CALL').upper(),
                        'spotprice': row.get('spot', 0),
                        'strike': row.get('strike', 0),
                        'premium': row.get('bid', 0),
                        'dtm': row.get('dtm', 0),
                        'vol': 0.3,
                        'duedate': row.get('expiration'),
                        'amount': LOT_SIZE
                    }
                    bs = fetch_bs_oplab(params)
                    if bs:
                        df_chain.at[idx, 'delta'] = float(bs.get('delta', 0))
                        df_chain.at[idx, 'iv'] = float(bs.get('volatility') or bs.get('vol') or 0)
                    time.sleep(0.03)
                except Exception:
                    continue

        # filter by DTM range
        df_chain = df_chain[(df_chain['dtm'] >= dtm_min) & (df_chain['dtm'] <= dtm_max)].copy()

        # compute iv_rank for each option if possible (best-effort)
        if HAVE_YFINANCE:
            df_chain['iv_rank'] = df_chain['iv'].apply(lambda iv: compute_iv_rank(tk, iv) if iv and not pd.isna(iv) else None)
        else:
            df_chain['iv_rank'] = None

        # filter by delta absolute range (aplica a CALL e PUT)
        # ensure delta numeric; for puts delta may be negative - use abs()
        df_chain['delta_abs'] = df_chain['delta'].apply(lambda v: abs(v) if pd.notna(v) else np.nan)
        df_chain = df_chain[(df_chain['delta_abs'] >= delta_min) & (df_chain['delta_abs'] <= delta_max)]

        # top-3 calls and puts by bid
        calls = df_chain[df_chain['type'] == 'call'].sort_values(by='bid', ascending=False).head(3)
        puts = df_chain[df_chain['type'] == 'put'].sort_values(by='bid', ascending=False).head(3)

        if not calls.empty:
            calls['ticker'] = tk
            all_calls.append(calls)
        if not puts.empty:
            puts['ticker'] = tk
            all_puts.append(puts)

        # find best individual call/put to form strangle (closest delta to delta_min)
        def best_by_delta(group, target_abs):
            if group.empty:
                return None
            group = group.copy()
            group['score'] = (group['delta_abs'] - target_abs).abs()
            group = group.sort_values(by=['score', 'bid'], ascending=[True, False])
            return group.iloc[0]

        best_call = best_by_delta(df_chain[df_chain['type']=='call'], delta_min)
        best_put  = best_by_delta(df_chain[df_chain['type']=='put'], delta_min)

        if best_call is not None and best_put is not None:
            total_premium = float(best_call.get('bid', 0)) + float(best_put.get('bid', 0))
            spot_price = float(best_call.get('spot') or best_put.get('spot') or 100)
            dtm = int(best_call.get('dtm') or best_put.get('dtm') or 0)
            tio = compute_tio(total_premium, spot_price, dtm)
            iv_rank_avg = None
            if HAVE_YFINANCE:
                ivs = [v for v in [best_call.get('iv'), best_put.get('iv')] if v and not pd.isna(v)]
                if ivs:
                    iv_rank_avg = compute_iv_rank(tk, sum(ivs)/len(ivs))
            all_strangles.append({
                'ticker': tk,
                'call_symbol': best_call.get('option_symbol'),
                'call_strike': best_call.get('strike'),
                'call_bid': best_call.get('bid'),
                'call_delta': best_call.get('delta'),
                'put_symbol': best_put.get('option_symbol'),
                'put_strike': best_put.get('strike'),
                'put_bid': best_put.get('bid'),
                'put_delta': best_put.get('delta'),
                'total_premium': round(total_premium, 4),
                'dtm': dtm,
                'tio': tio,
                'iv_rank': iv_rank_avg,
                'spot': spot_price
            })

    # display results
    if all_calls:
        df_calls_all = pd.concat(all_calls, ignore_index=True)
        st.subheader("Top CALLs (por ticker)")
        st.dataframe(df_calls_all[['ticker','option_symbol','strike','expiration','dtm','bid','delta','iv','iv_rank','open_interest']])
    else:
        st.write("Nenhuma CALL encontrada com os filtros.")

    if all_puts:
        df_puts_all = pd.concat(all_puts, ignore_index=True)
        st.subheader("Top PUTs (por ticker)")
        st.dataframe(df_puts_all[['ticker','option_symbol','strike','expiration','dtm','bid','delta','iv','iv_rank','open_interest']])
    else:
        st.write("Nenhuma PUT encontrada com os filtros.")

    if all_strangles:
        df_str = pd.DataFrame(all_strangles)
        # ordenar priorizando TIO e proximidade de DTM=45 e IV Rank >= iv_rank_min
        df_str['iv_rank_filled'] = df_str['iv_rank'].fillna(0)
        df_str_sorted = df_str.sort_values(by=['tio','iv_rank_filled', 'dtm'], ascending=[False, False, True])
        st.subheader("Strangles gerados")
        st.dataframe(df_str_sorted[['ticker','total_premium','dtm','tio','iv_rank','spot']])
        st.download_button("Exportar Strangles CSV", df_str_sorted.to_csv(index=False), "strangles.csv")
    else:
        st.write("Nenhuma combinação de Strangle gerada com os filtros atuais.")

else:
    st.info("Selecione até 3 tickers e ajuste os filtros na barra lateral. Em seguida clique em Executar.")

st.markdown("---")
st.caption("Observação: IV Rank e Beta requerem yfinance e internet; sem yfinance o app exibirá IV (API) como proxy.")
