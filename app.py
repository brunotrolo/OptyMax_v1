#!/usr/bin/env python3
# optymax_analysis.py
"""
OptyMax — Análise Conservadora minimalista (CLI)

Resumo:
- Rejeita séries com close == 0
- Calcula TIO (bruta) por série: TIO = (close / spot) * (365 / dtm) * 100
- Calcula IV Rank por ativo (percentil do IV entre as séries do mesmo ativo)
- Calcula Beta do ativo vs IBOV (via yfinance)
- Gera relatório com formato RÍGIDO:
    1. Análise por Lucratividade (TIO Bruta)
    2. Análise por Risco e Segurança
    3. Conclusão e Recomendação Final

Modo de uso:
  (A) Usando OPLAB (recomendado para dados reais)
    - Defina variável de ambiente OPLAB_TOKEN com seu token
      export OPLAB_TOKEN="SEU_TOKEN_AQUI"
    - Rode:
      python optymax_analysis.py PETR4 VALE3 PSSA3

  (B) Modo offline (CSVs)
    - Prepare CSV 'options.csv' com colunas:
        parent,option_symbol,type,strike,close,spot,dtm,volume,open_interest,volatility
      (volatility = implied vol decimal, e.g. 0.25)
    - Prepare CSV 'betas.csv' (opcional) com:
        parent,beta
    - Rode:
      python optymax_analysis.py --csv options.csv --betas betas.csv

Observações:
- Para precisão do delta usamos o endpoint /market/options/bs quando disponível,
  mas o script NÃO depende do delta para ranking (ranking é por TIO e IV Rank).
- O script tenta calcular IV Rank com as volatilidades disponíveis na cadeia.
- Recomenda executar com poucos tickers (1-5) quando usando OPLAB ao vivo.
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np

# optional: progress bar if available
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **_): return x

# yfinance optional
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# -------------
# CONFIG
# -------------
OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
SLEEP_BETWEEN_REQUESTS = 0.05     # gentle pause
BETA_MARKET_SYMBOL = "^BVSP"      # IBOV for yfinance
MIN_IVRANK_TRUST = 50.0          # % threshold to consider TIO trustworthy

# -------------
# HELPERS: HTTP safe get
# -------------
def safe_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: int = 10):
    try:
        h = headers if headers is not None else HEADERS
        r = requests.get(url, params=params, headers=h, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except Exception:
        return None

# -------------
# OPLAB calls
# -------------
def fetch_chain_oplab(underlying: str) -> pd.DataFrame:
    url = f"{OPLAB_BASE}/market/options/{underlying}"
    data = safe_get(url)
    # Return empty DF on any issue
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for it in data:
        rows.append({
            "parent": underlying,
            "option_symbol": it.get("symbol"),
            "type": (it.get("type") or it.get("category") or "").upper(),
            "strike": float(it.get("strike") or 0),
            "expiration": it.get("due_date"),
            "bid": float(it.get("bid") or 0),
            "ask": float(it.get("ask") or 0),
            "close": float(it.get("close") or 0),
            "spot": float(it.get("spot_price") or it.get("spot") or 0),
            "dtm": int(it.get("days_to_maturity") or 0),
            "volume": int(it.get("volume") or 0),
            "open_interest": int(it.get("open_interest") or 0),
            "volatility": (float(it.get("volatility")) if it.get("volatility") not in (None,"") else None)
        })
    df = pd.DataFrame(rows)
    return df

def fetch_details_oplab(symbol: str) -> Dict[str,Any]:
    url = f"{OPLAB_BASE}/market/options/details/{symbol}"
    return safe_get(url)

def fetch_bs_oplab_with_details(symbol: str) -> Dict[str,Any]:
    # fetch details then /bs
    details = fetch_details_oplab(symbol)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if not details:
        return {}
    params = {
        "symbol": details.get("symbol"),
        "irate": 0.1,
        "type": details.get("category") or details.get("type"),
        "spotprice": float(details.get("spot_price") or details.get("spot") or 0),
        "strike": float(details.get("strike") or 0),
        "premium": float(details.get("close") or details.get("bid") or details.get("ask") or 0.01),
        "dtm": int(details.get("days_to_maturity") or 0),
        "vol": float(details.get("volatility") or 0.25),
        "duedate": details.get("due_date") or details.get("expiration"),
        "amount": 100
    }
    bs = safe_get(f"{OPLAB_BASE}/market/options/bs", params=params)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return bs or {}

# -------------
# Calculations
# -------------
def compute_tio(close: float, spot: float, dtm: int) -> float:
    try:
        c = float(close); s = float(spot); d = int(dtm)
        if c <= 0 or s <= 0 or d <= 0:
            return 0.0
        return float((c / s) * (365.0 / d) * 100.0)
    except Exception:
        return 0.0

def iv_rank_percentile(iv_list: List[float], value: float) -> float:
    ivs = [v for v in iv_list if v is not None and not (isinstance(v,float) and math.isnan(v))]
    if not ivs:
        return 0.0
    ivs_sorted = sorted(ivs)
    count_le = sum(1 for v in ivs_sorted if v <= value)
    return float((count_le / len(ivs_sorted)) * 100.0)

def compute_beta_yf(ticker: str) -> float:
    if not HAVE_YF:
        return float("nan")
    try:
        sym = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        market = BETA_MARKET_SYMBOL
        df_t = yf.download(sym, period="1y", progress=False)
        df_m = yf.download(market, period="1y", progress=False)
        if df_t.empty or df_m.empty:
            return float("nan")
        rt = df_t["Close"].pct_change().dropna()
        rm = df_m["Close"].pct_change().dropna()
        df = pd.concat([rt, rm], axis=1, join="inner").dropna()
        if df.shape[0] < 30:
            return float("nan")
        cov = df.iloc[:,0].cov(df.iloc[:,1])
        var_m = df.iloc[:,1].var()
        if var_m == 0 or math.isnan(cov):
            return float("nan")
        beta = cov / var_m
        return float(beta)
    except Exception:
        return float("nan")

# -------------
# Pipeline: analyze tickers (either via API or CSV)
# -------------
def analyze_with_api(tickers: List[str]) -> Dict[str, Any]:
    per_underlying = []
    for t in tqdm(tickers):
        chain = fetch_chain_oplab(t)
        if chain.empty:
            continue
        # reject close == 0
        chain = chain[chain["close"] > 0].copy()
        if chain.empty:
            continue
        # ensure numeric
        chain["dtm"] = pd.to_numeric(chain["dtm"], errors="coerce").fillna(0).astype(int)
        chain["spot"] = pd.to_numeric(chain["spot"], errors="coerce").fillna(0.0)
        chain["close"] = pd.to_numeric(chain["close"], errors="coerce").fillna(0.0)

        # compute TIO per series
        chain["tio_pct"] = chain.apply(lambda r: compute_tio(r["close"], r["spot"], int(r["dtm"])), axis=1)

        # collect IVs from chain if available; otherwise try to call /bs for top candidates only
        ivs_available = chain["volatility"].dropna().tolist()
        # if volatility missing for many, attempt to fetch bs for options with close>0 to get iv and delta (but keep limited)
        missing_iv_count = chain["volatility"].isna().sum()
        if missing_iv_count > 0 and len(chain) <= 200:  # only auto-call bs for reasonable chain sizes
            # cache bs responses
            bs_cache = {}
            for sym in chain["option_symbol"].tolist():
                bs = fetch_bs_oplab_with_details(sym)
                if bs:
                    iv = None
                    if "volatility" in bs and bs["volatility"] is not None:
                        try: iv = float(bs["volatility"])
                        except: iv = None
                    elif "vol" in bs and bs["vol"] is not None:
                        try: iv = float(bs["vol"])
                        except: iv = None
                    # store
                    bs_cache[sym] = {"iv": iv, "delta": bs.get("delta")}
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            # map back
            chain["iv"] = chain["option_symbol"].apply(lambda s: bs_cache.get(s, {}).get("iv"))
            chain["delta"] = chain["option_symbol"].apply(lambda s: bs_cache.get(s, {}).get("delta"))
            # fill from chain volatility if present
            chain["iv"] = chain.apply(lambda r: r["iv"] if (r["iv"] is not None and not math.isnan(r["iv"])) else r["volatility"], axis=1)
        else:
            # use chain volatility column directly
            chain["iv"] = chain["volatility"]

        # compute iv_rank per series (percentil within this underlying)
        iv_list = [v for v in chain["iv"].tolist() if v is not None and not (isinstance(v,float) and math.isnan(v))]
        chain["iv_rank_pct"] = chain["iv"].apply(lambda v: iv_rank_percentile(iv_list, v) if v is not None and not math.isnan(v) else 0.0)

        # pick best series per underlying by tio_pct
        best_idx = chain["tio_pct"].idxmax() if not chain["tio_pct"].isnull().all() else None
        best = chain.loc[best_idx].to_dict() if best_idx is not None else None

        per_underlying.append({
            "parent": t,
            "chain": chain,
            "best_series": best,
            "max_tio": float(chain["tio_pct"].max()),
            "best_option": best["option_symbol"] if best else None,
            "best_iv": float(best.get("iv")) if best and best.get("iv") is not None else None,
            "best_delta": float(best.get("delta")) if best and best.get("delta") is not None else None,
            "iv_rank_of_best": float(best.get("iv_rank_pct")) if best and best.get("iv_rank_pct") is not None else 0.0,
            "n_series": len(chain)
        })
    # summary df
    summary = pd.DataFrame([{
        "parent": u["parent"],
        "max_tio": u["max_tio"],
        "best_option": u["best_option"],
        "best_iv": u["best_iv"],
        "best_delta": u["best_delta"],
        "iv_rank_of_best": u["iv_rank_of_best"],
        "n_series": u["n_series"]
    } for u in per_underlying])
    # compute betas
    betas = []
    for p in summary["parent"].tolist():
        b = compute_beta_yf(p)
        betas.append({"parent": p, "beta": b})
    betas_df = pd.DataFrame(betas)
    if not betas_df.empty:
        summary = summary.merge(betas_df, on="parent", how="left")
    # mark tio trustworthy
    summary["tio_trustworthy"] = summary["iv_rank_of_best"] >= MIN_IVRANK_TRUST
    return {"per_underlying": per_underlying, "summary": summary, "betas": betas_df}

def analyze_with_csv(options_csv: str, betas_csv: str = None) -> Dict[str, Any]:
    df = pd.read_csv(options_csv)
    # expected columns: parent,option_symbol,type,strike,close,spot,dtm,volume,open_interest,volatility
    df = df.copy()
    # reject close==0
    df = df[df["close"] > 0]
    if df.empty:
        return {"per_underlying": [], "summary": pd.DataFrame(), "betas": pd.DataFrame()}
    per = []
    for parent, group in df.groupby("parent"):
        g = group.copy()
        g["tio_pct"] = g.apply(lambda r: compute_tio(r["close"], r["spot"], int(r["dtm"])), axis=1)
        # iv_rank within group
        iv_list = [v for v in g["volatility"].tolist() if v is not None and not (isinstance(v,float) and math.isnan(v))]
        g["iv_rank_pct"] = g["volatility"].apply(lambda v: iv_rank_percentile(iv_list, v) if v is not None and not math.isnan(v) else 0.0)
        best_idx = g["tio_pct"].idxmax() if not g["tio_pct"].isnull().all() else None
        best = g.loc[best_idx].to_dict() if best_idx is not None else None
        per.append({
            "parent": parent,
            "chain": g,
            "best_series": best,
            "max_tio": float(g["tio_pct"].max()),
            "best_option": best["option_symbol"] if best else None,
            "best_iv": float(best.get("volatility")) if best and best.get("volatility") is not None else None,
            "best_delta": None,
            "iv_rank_of_best": float(best.get("iv_rank_pct")) if best and best.get("iv_rank_pct") is not None else 0.0,
            "n_series": len(g)
        })
    summary = pd.DataFrame([{
        "parent": u["parent"],
        "max_tio": u["max_tio"],
        "best_option": u["best_option"],
        "best_iv": u["best_iv"],
        "best_delta": u["best_delta"],
        "iv_rank_of_best": u["iv_rank_of_best"],
        "n_series": u["n_series"]
    } for u in per])
    # betas
    if betas_csv and os.path.exists(betas_csv):
        betas_df = pd.read_csv(betas_csv)
    else:
        betas = []
        for p in summary["parent"].tolist():
            b = compute_beta_yf(p)
            betas.append({"parent": p, "beta": b})
        betas_df = pd.DataFrame(betas)
    if not betas_df.empty:
        summary = summary.merge(betas_df, on="parent", how="left")
    summary["tio_trustworthy"] = summary["iv_rank_of_best"] >= MIN_IVRANK_TRUST
    return {"per_underlying": per, "summary": summary, "betas": betas_df}

# -------------
# Output formatting: Strict as requested
# -------------
def build_report(result: Dict[str, Any]) -> str:
    summary: pd.DataFrame = result["summary"]
    betas_df: pd.DataFrame = result["betas"]
    # 1) Top 3 by TIO
    top3 = summary.sort_values("max_tio", ascending=False).head(3)
    tio_comments = []
    for _, r in top3.iterrows():
        if r["tio_trustworthy"]:
            tio_comments.append(f"{r['parent']} tem TIO elevada ({r['max_tio']:.2f}%) suportada por IV Rank {r['iv_rank_of_best']:.1f}%.")
        else:
            tio_comments.append(f"{r['parent']} tem TIO elevada ({r['max_tio']:.2f}%) MAS IV Rank baixo ({r['iv_rank_of_best']:.1f}%) — TIO pode ser enganosa.")
    tio_comment_par = " ".join(tio_comments)

    # 2) Risk & Safety: top3 lowest betas (prefer beta<1.0)
    if not betas_df.empty and not betas_df["beta"].isnull().all():
        betas_sorted = betas_df.sort_values("beta", ascending=True).head(3)
    else:
        # fallback: choose smallest max_tio as proxy (not ideal)
        betas_sorted = summary.sort_values("max_tio", ascending=True).head(3)[["parent"]].assign(beta=np.nan)

    # which asset allows more leverage in lots? heuristic: larger open interest and lower spot -> higher leverage score
    leverage_list = []
    for item in result["per_underlying"]:
        best = item.get("best_series")
        if not best:
            continue
        spot = float(best.get("spot") or 0)
        oi = int(best.get("open_interest") or 0)
        vol = int(best.get("volume") or 0)
        score = (oi + 1) / (spot + 1e-9)
        leverage_list.append({"parent": item["parent"], "score": score, "spot": spot, "open_interest": oi, "volume": vol})
    leverage_df = pd.DataFrame(leverage_list) if leverage_list else pd.DataFrame()
    if not leverage_df.empty:
        best_lev = leverage_df.sort_values("score", ascending=False).iloc[0]
        leverage_comment = f"{best_lev['parent']} permite maior alocação por lote (proxy score = {best_lev['score']:.2f}; spot={best_lev['spot']}, open_interest={best_lev['open_interest']})."
    else:
        leverage_comment = "Dados insuficientes para estimar alavancagem por lotes."

    # 3) Conclusion: Renda Máxima (principal) and Estabilidade (secondary)
    candidates = summary[summary["tio_trustworthy"] == True]
    if not candidates.empty:
        renda = candidates.sort_values("max_tio", ascending=False).iloc[0]
    elif not summary.empty:
        renda = summary.sort_values("max_tio", ascending=False).iloc[0]
    else:
        renda = None

    if not betas_df.empty and not betas_df["beta"].isnull().all():
        estabilidade = betas_df.sort_values("beta", ascending=True).iloc[0]
    elif not summary.empty:
        estabilidade = summary.sort_values("max_tio", ascending=True).iloc[0]
    else:
        estabilidade = None

    # Build strict text
    lines = []
    lines.append("1. Análise por Lucratividade (TIO Bruta)")
    lines.append("    - Apresentar o ranking dos 3 melhores ativos.")
    for _, r in top3.iterrows():
        lines.append(f"        • {r['parent']}: TIO={r['max_tio']:.2f}% — melhor série: {r['best_option']} — IV(best)={r['best_iv'] if not pd.isna(r['best_iv']) else 'N/D'} — delta={r['best_delta'] if not pd.isna(r['best_delta']) else 'N/D'} — IV Rank={r['iv_rank_of_best']:.2f}%")
    lines.append("    - Comentar a relação entre TIO e IV Rank.")
    lines.append(f"        {tio_comment_par}")
    lines.append("")
    lines.append("2. Análise por Risco e Segurança")
    lines.append("    - Apresentar os 3 ativos com melhor perfil de segurança (Beta < 1.0 ou mais próximos de zero).")
    for _, r in betas_sorted.iterrows():
        lines.append(f"        • {r['parent']}: Beta={r['beta'] if not pd.isna(r['beta']) else 'N/D'}")
    lines.append("    - Comentar qual ativo permite a maior alavancagem em lotes.")
    lines.append(f"        {leverage_comment}")
    lines.append("")
    lines.append("3. Conclusão e Recomendação Final")
    lines.append("    - Indicar o Ativo de Renda Máxima (Principal) e o Ativo de Estabilidade (Secundário)")
    if renda is not None:
        lines.append(f"    - Ativo de Renda Máxima (Principal): {renda['parent']} — TIO={renda['max_tio']:.2f}% — série {renda['best_option']}")
    else:
        lines.append("    - Ativo de Renda Máxima (Principal): N/D")
    if estabilidade is not None:
        lines.append(f"    - Ativo de Estabilidade (Secundário): {estabilidade['parent']} — Beta={stabilidade['beta'] if 'beta' in estabilidade else 'N/D'}")
    else:
        lines.append("    - Ativo de Estabilidade (Secundário): N/D")
    return "\n".join(lines)

# -------------
# CLI entry
# -------------
def main():
    parser = argparse.ArgumentParser(description="OptyMax minimal analysis (TIO / IV Rank / Beta)")
    parser.add_argument("tickers", nargs="*", help="Tickers (ex: PETR4 VALE3). If --csv provided, tickers ignored.")
    parser.add_argument("--csv", help="Use local options CSV instead of OPLAB API (columns: parent,option_symbol,type,strike,close,spot,dtm,volume,open_interest,volatility)")
    parser.add_argument("--betas", help="Optional CSV with betas (columns: parent,beta)")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_REQUESTS, help="Sleep between API calls (seconds)")
    args = parser.parse_args()

    global SLEEP_BETWEEN_REQUESTS
    SLEEP_BETWEEN_REQUESTS = args.sleep

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"CSV file not found: {args.csv}")
            return
        res = analyze_with_csv(args.csv, args.betas)
    else:
        if not OPLAB_TOKEN:
            print("ERROR: OPLAB_TOKEN not set. Either export OPLAB_TOKEN or use --csv mode.")
            return
        if not args.tickers:
            print("Provide tickers or use --csv. Example: python optymax_analysis.py PETR4 VALE3")
            return
        res = analyze_with_api(args.tickers)

    report = build_report(res)
    print("\n" + report + "\n")

if __name__ == "__main__":
    main()
