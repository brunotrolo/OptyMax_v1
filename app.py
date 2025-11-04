#!/usr/bin/env python3
# analysis_optymax.py
"""
Análise OptyMax — TIO / IV Rank / Beta / Recomendação

Uso:
  - Defina OPLAB_TOKEN no ambiente:
      export OPLAB_TOKEN="SEU_TOKEN_AQUI"
  - Instale dependências:
      pip install requests pandas numpy yfinance tqdm
  - Rode:
      python analysis_optymax.py PETR4 VALE3 PSSA3

O que o script faz (resumido):
  1) Para cada ticker informado, consulta /market/options/{UNDERLYING}
  2) Rejeita séries com close == 0
  3) Para cada série restante, chama /market/options/bs (via dados de details para parâmetros precisos) para obter delta e volatilidade
  4) Calcula TIO por série: (close / spot) * (365 / dtm) * 100
  5) Calcula IV Rank por ativo como percentil do IV entre as séries do mesmo ativo
  6) Calcula Beta do ativo vs IBOV (usando yfinance) para o período de 1 ano
  7) Gera relatório com o formato rígido exigido
Observação: chame com poucos tickers para testes — cada ticker pode gerar muitas requisições.
"""

import os
import sys
import time
import math
from typing import List, Dict, Any
import requests
import pandas as pd
import numpy as np

# opcional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **_): return x

# yfinance para beta
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ---------------------------
# Configuração
# ---------------------------
OPLAB_BASE = "https://api.oplab.com.br/v3"
OPLAB_TOKEN = os.environ.get("OPLAB_TOKEN", "")
HEADERS = {"Access-Token": OPLAB_TOKEN} if OPLAB_TOKEN else {}
LOT_SIZE = 100

# parâmetros conservadores / defaults (ajustáveis)
MIN_IVRANK_PERCENT = 50.0   # IV Rank mínimo (%) para considerar TIO "confiável"
BETA_LOOKBACK_DAYS = 252    # ~1 ano de pregões
MARKET_INDEX = "^BVSP"      # índice Bovespa (yfinance symbol)
REQUEST_SLEEP = 0.05        # pausa entre requisições para não saturar API

# ---------------------------
# Utils e chamadas OPLAB
# ---------------------------
def safe_get(url: str, params=None, headers=None, timeout=10) -> Any:
    try:
        r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            # Retornar None para erros, log externo
            # print(f"HTTP {r.status_code} -> {url} params={params}")
            return None
    except Exception as e:
        # print(f"Exception fetching {url}: {e}")
        return None

def get_options_chain(underlying: str) -> pd.DataFrame:
    """GET /market/options/{UNDERLYING} -> retorna DataFrame com as colunas essenciais"""
    url = f"{OPLAB_BASE}/market/options/{underlying}"
    data = safe_get(url)
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
        })
    df = pd.DataFrame(rows)
    return df

def get_option_details(symbol: str) -> Dict[str,Any]:
    url = f"{OPLAB_BASE}/market/options/details/{symbol}"
    return safe_get(url)

def get_bs_for_symbol_using_details(symbol: str) -> Dict[str,Any]:
    """Consulta /market/options/details/{symbol} e em seguida /market/options/bs com parâmetros consistentes."""
    details = get_option_details(symbol)
    time.sleep(REQUEST_SLEEP)
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
        "amount": LOT_SIZE
    }
    bs = safe_get(f"{OPLAB_BASE}/market/options/bs", params=params)
    time.sleep(REQUEST_SLEEP)
    return bs or {}

# ---------------------------
# Cálculos
# ---------------------------
def compute_tio(close: float, spot: float, dtm: int) -> float:
    """TIO = (close / spot) * (365 / dtm) * 100 — retorna em %"""
    try:
        close_f = float(close)
        spot_f = float(spot)
        dtm_i = int(dtm)
        if close_f <= 0 or spot_f <= 0 or dtm_i <= 0:
            return 0.0
        # cálculo passo-a-passo (para precisão)
        ratio = close_f / spot_f
        factor = 365.0 / dtm_i
        tio = ratio * factor * 100.0
        return float(round(tio, 6))
    except Exception:
        return 0.0

def iv_rank_from_list(iv_list: List[float], value: float) -> float:
    """Calcula percentil (0-100) de 'value' na lista iv_list."""
    ivs = [v for v in iv_list if v is not None and not math.isnan(v)]
    if not ivs:
        return 0.0
    ivs_sorted = sorted(ivs)
    # percentile rank (percentage of items <= value)
    count_le = sum(1 for v in ivs_sorted if v <= value)
    rank = (count_le / len(ivs_sorted)) * 100.0
    return float(round(rank, 4))

def compute_beta(ticker: str, market_symbol: str = MARKET_INDEX, period_days: int = BETA_LOOKBACK_DAYS) -> float:
    """Calcula Beta do ticker vs market usando yfinance (1y ~ 252 dias). Retorna NaN se indisponível."""
    if not HAVE_YF:
        return float("nan")
    try:
        # yfinance symbols for BR tickers: add '.SA'
        t = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        m = market_symbol
        df_t = yf.download(t, period="1y", progress=False)
        df_m = yf.download(m, period="1y", progress=False)
        if df_t.empty or df_m.empty:
            return float("nan")
        rt = df_t["Close"].pct_change().dropna()
        rm = df_m["Close"].pct_change().dropna()
        # align
        df = pd.concat([rt, rm], axis=1, join="inner").dropna()
        if df.shape[0] < 30:
            return float("nan")
        cov = df.iloc[:,0].cov(df.iloc[:,1])
        var_m = df.iloc[:,1].var()
        if var_m == 0 or math.isnan(cov):
            return float("nan")
        beta = cov / var_m
        return float(round(beta, 4))
    except Exception:
        return float("nan")

# ---------------------------
# Pipeline principal
# ---------------------------
def analyze_tickers(tickers: List[str]) -> Dict[str, Any]:
    """
    Retorna um dicionário com:
      - per_underlying: dataframe com melhores séries, TIO, IV, delta, iv_rank, etc.
      - betas: dataframe com betas
      - report fields
    """
    per_underlying = []

    for t in tqdm(tickers):
        chain_df = get_options_chain(t)
        if chain_df.empty:
            # nenhum dado
            continue

        # Rejeitar séries com close == 0
        chain_df = chain_df[chain_df["close"] > 0].copy()
        if chain_df.empty:
            continue

        # Garantir tipos
        chain_df["dtm"] = pd.to_numeric(chain_df["dtm"], errors="coerce").fillna(0).astype(int)
        chain_df["spot"] = pd.to_numeric(chain_df["spot"], errors="coerce").fillna(0.0)
        chain_df["close"] = pd.to_numeric(chain_df["close"], errors="coerce").fillna(0.0)

        # Para cada série, obter BS (delta e iv) usando endpoint de detalhes + bs
        ivs = []
        deltas = []
        bs_cache = {}  # option_symbol -> bs result
        for idx, row in chain_df.iterrows():
            sym = row["option_symbol"]
            bs = get_bs_for_symbol_using_details(sym)
            if bs and isinstance(bs, dict):
                # tentativa de extrair volatilidade e delta
                iv = None
                delta = None
                # possíveis chaves: 'volatility', 'vol', 'iv', 'delta'
                if "volatility" in bs and bs["volatility"] is not None:
                    try:
                        iv = float(bs["volatility"])
                    except:
                        iv = None
                elif "vol" in bs and bs["vol"] is not None:
                    try:
                        iv = float(bs["vol"])
                    except:
                        iv = None
                if "delta" in bs and bs["delta"] is not None:
                    try:
                        delta = float(bs["delta"])
                    except:
                        delta = None
                # armazenar
                bs_cache[sym] = {"iv": iv, "delta": delta, "raw": bs}
                ivs.append(iv)
                deltas.append(delta)
            else:
                # sem bs válido -> append None placeholders
                bs_cache[sym] = {"iv": None, "delta": None, "raw": bs}
                ivs.append(None)
                deltas.append(None)

        # calcular TIO por série
        chain_df["tio_pct"] = chain_df.apply(lambda r: compute_tio(r["close"], r["spot"], int(r["dtm"])), axis=1)

        # associar bs data ao df
        chain_df["iv"] = chain_df["option_symbol"].apply(lambda s: bs_cache.get(s, {}).get("iv"))
        chain_df["delta"] = chain_df["option_symbol"].apply(lambda s: bs_cache.get(s, {}).get("delta"))

        # calcular iv_rank relativo ao conjunto de ivs deste ativo (percentil)
        # para cada série, se iv presente, compute percentile
        iv_list = [v for v in chain_df["iv"].tolist() if v is not None and not math.isnan(v)]
        chain_df["iv_rank_pct"] = chain_df["iv"].apply(lambda v: iv_rank_from_list(iv_list, v) if v is not None and not math.isnan(v) else 0.0)

        # escolher a melhor série por TIO bruta (maior tio)
        best_idx = chain_df["tio_pct"].idxmax() if not chain_df["tio_pct"].isnull().all() else None
        if best_idx is not None and not pd.isna(best_idx):
            best = chain_df.loc[best_idx].to_dict()
        else:
            best = None

        per_underlying.append({
            "parent": t,
            "chain_df": chain_df,
            "best_series": best,
            # estatísticas resumidas
            "max_tio": float(chain_df["tio_pct"].max()),
            "best_option": best["option_symbol"] if best else None,
            "best_iv": float(best.get("iv")) if best and best.get("iv") is not None else None,
            "best_delta": float(best.get("delta")) if best and best.get("delta") is not None else None,
            "iv_rank_of_best": float(best.get("iv_rank_pct")) if best and best.get("iv_rank_pct") is not None else 0.0,
            "n_series": len(chain_df)
        })

    # DataFrame resumo
    summary_rows = []
    for u in per_underlying:
        summary_rows.append({
            "parent": u["parent"],
            "max_tio": u["max_tio"],
            "best_option": u["best_option"],
            "best_iv": u["best_iv"],
            "best_delta": u["best_delta"],
            "iv_rank_of_best": u["iv_rank_of_best"],
            "n_series": u["n_series"]
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(by="max_tio", ascending=False).reset_index(drop=True)

    # calcular betas para cada underlying (usa yfinance se disponível)
    betas = []
    for u in tqdm(summary_df["parent"].tolist()):
        if HAVE_YF:
            b = compute_beta(u)
        else:
            b = float("nan")
        betas.append({"parent": u, "beta": b})
    betas_df = pd.DataFrame(betas)

    # mesclar betas no summary
    if not betas_df.empty:
        summary_df = summary_df.merge(betas_df, on="parent", how="left")

    # marcar ativos com alta TIO mas IV Rank < MIN_IVRANK_PERCENT como "não confiável"
    summary_df["tio_trustworthy"] = summary_df["iv_rank_of_best"] >= MIN_IVRANK_PERCENT

    return {
        "per_underlying": per_underlying,
        "summary_df": summary_df,
        "betas_df": betas_df
    }

# ---------------------------
# Formatação de saída conforme requisitado
# ---------------------------
def build_rigorous_report(result: Dict[str, Any]) -> str:
    summary_df: pd.DataFrame = result["summary_df"]
    betas_df: pd.DataFrame = result["betas_df"]

    # 1) Análise por Lucratividade (TIO Bruta) — top 3
    top3 = summary_df.sort_values(by="max_tio", ascending=False).head(3)

    # comentário sobre relação TIO e IV Rank:
    # para cada top3, ver se tio_trustworthy True
    comments = []
    for _, r in top3.iterrows():
        if r.get("tio_trustworthy"):
            comments.append(f"{r['parent']} tem TIO elevada ({r['max_tio']:.2f}%) suportada por IV Rank {r['iv_rank_of_best']:.1f}%.")
        else:
            comments.append(f"{r['parent']} tem TIO elevada ({r['max_tio']:.2f}%) MAS IV Rank baixo ({r['iv_rank_of_best']:.1f}%) — cuidado: prêmio possivelmente enganoso.")

    tio_comment_paragraph = " ".join(comments)

    # 2) Risco e Segurança — 3 ativos com melhor perfil de segurança (beta baixo) — Beta < 1.0 preferred
    # Use betas_df; se betas nan, we'll fallback to summary ordering
    if not betas_df.empty:
        betas_sorted = betas_df.sort_values(by="beta", ascending=True).head(3)
    else:
        betas_sorted = summary_df.sort_values(by="max_tio", ascending=True).head(3)[["parent"]].assign(beta=np.nan)

    # determinar qual ativo permite maior alavancagem em lotes = normalmente ativo com menor preço spot e boa liquidez
    # Como proxy, usamos spot do best option dividido por strike? Melhor: recuperar spot do per_underlying
    # Vamos pegar spot médio da melhor série de cada underlying (se disponível) e open interest/volume summation
    leverage_rows = []
    for u in result["per_underlying"]:
        best = u.get("best_series")
        if not best:
            continue
        spot = float(best.get("spot") or 0)
        oi = int(best.get("open_interest") or 0)
        vol = int(best.get("volume") or 0)
        leverage_rows.append({"parent": u["parent"], "spot": spot, "open_interest": oi, "volume": vol})
    leverage_df = pd.DataFrame(leverage_rows)
    if not leverage_df.empty:
        # heurística: menor spot e maior open_interest = mais "alavancável"
        # score = normalized open_interest / spot
        leverage_df["score"] = leverage_df.apply(lambda r: (r["open_interest"] + 1) / (r["spot"] + 1e-9), axis=1)
        best_leverage_row = leverage_df.sort_values(by="score", ascending=False).iloc[0]
        leverage_comment = f"{best_leverage_row['parent']} permite maior alocação por lote (score liquidity/spot = {best_leverage_row['score']:.2f})."
    else:
        leverage_comment = "Dados de liquidez insuficientes para estimar alavancagem por lotes."

    # 3) Conclusão e Recomendação Final: escolher Ativo de Renda Máxima (principal) e Ativo de Estabilidade (secundário)
    # Renda Máxima: top asset by max_tio but must be tio_trustworthy True if possible
    candidates = summary_df[summary_df["tio_trustworthy"] == True]
    if candidates.empty:
        renda = summary_df.sort_values("max_tio", ascending=False).iloc[0]
    else:
        renda = candidates.sort_values("max_tio", ascending=False).iloc[0]

    # Estabilidade: lowest beta (<1.0 preferred)
    if not betas_df.empty and not betas_df["beta"].isnull().all():
        estabilidade = betas_df.sort_values("beta", ascending=True).iloc[0]
    else:
        # fallback to asset with lower tio volatility? choose asset with lowest max_tio
        estabilidade = summary_df.sort_values("max_tio", ascending=True).iloc[0]

    # Montar string formatada rigidamente
    lines = []
    lines.append("1. Análise por Lucratividade (TIO Bruta)")
    lines.append("    - Apresentar o ranking dos 3 melhores ativos.")
    for _, r in top3.iterrows():
        lines.append(f"        • {r['parent']}: TIO={r['max_tio']:.2f}% — melhor série: {r['best_option']} — IV(best)={r['best_iv'] if r['best_iv'] is not None else 'N/D'} — delta={r['best_delta'] if r['best_delta'] is not None else 'N/D'} — IV Rank={r['iv_rank_of_best']:.2f}%")
    lines.append("    - Comentar a relação entre TIO e IV Rank.")
    lines.append(f"        {tio_comment_paragraph}")
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
    lines.append(f"    - Ativo de Renda Máxima (Principal): {renda['parent']} — TIO={renda['max_tio']:.2f}% — série {renda['best_option']}")
    lines.append(f"    - Ativo de Estabilidade (Secundário): {estabilidade['parent']} — Beta={estabilidade['beta'] if 'beta' in estabilidade else 'N/D'}")

    return "\n".join(lines)

# ---------------------------
# Main entry
# ---------------------------
def main(argv):
    if not OPLAB_TOKEN:
        print("ERRO: variável de ambiente OPLAB_TOKEN não configurada. Defina antes de rodar.")
        return

    if len(argv) < 2:
        print("Uso: python analysis_optymax.py PETR4 VALE3 ...")
        return

    tickers = argv[1:]
    print(f"Analisando {len(tickers)} tickers: {', '.join(tickers)}")
    res = analyze_tickers(tickers)

    report = build_rigorous_report(res)
    print("\n\n" + report + "\n\n")

if __name__ == "__main__":
    main(sys.argv)
