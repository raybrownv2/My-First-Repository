import os, json
import pandas as pd
import yfinance as yf
from backtest.tickers import TICKERS, BENCHMARK, START_DATE, END_DATE

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')

def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_prices(force_refresh=False):
    _ensure_cache()
    cache_file = os.path.join(CACHE_DIR, 'prices.csv')
    if os.path.exists(cache_file) and not force_refresh:
        print("  [cache] Loading prices...")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print("  [fetch] Downloading prices...")
    df = yf.download(TICKERS + [BENCHMARK], start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)['Close']
    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_csv(cache_file)
    return df

def get_fundamentals(force_refresh=False):
    _ensure_cache()
    cache_file = os.path.join(CACHE_DIR, 'fundamentals.json')
    if os.path.exists(cache_file) and not force_refresh:
        print("  [cache] Loading fundamentals...")
        with open(cache_file) as f: return json.load(f)
    print("  [fetch] Downloading fundamentals...")
    fundamentals = {}
    for t in TICKERS:
        try:
            info = yf.Ticker(t).info
            fundamentals[t] = {
                'priceToBook': info.get('priceToBook'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
            }
        except: fundamentals[t] = {}
    with open(cache_file, 'w') as f: json.dump(fundamentals, f)
    return fundamentals
