"""
データ取得モジュール
yfinanceを使って株価・財務データを取得し、キャッシュする
"""

import os
import json
import pandas as pd
import yfinance as yf

from backtest.tickers import TICKERS, BENCHMARK, START_DATE, END_DATE

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'cache')


def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    全銘柄 + ベンチマークの日次終値を取得。
    キャッシュがあれば再利用。

    Returns:
        DataFrame: index=日付, columns=ティッカー
    """
    _ensure_cache()
    cache_file = os.path.join(CACHE_DIR, 'prices.csv')

    if os.path.exists(cache_file) and not force_refresh:
        print("  [cache] 株価データ読み込み中...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print("  [fetch] 株価データをダウンロード中（時間がかかる場合があります）...")
    all_tickers = TICKERS + [BENCHMARK]
    df = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
    )['Close']

    # カラム名を整理
    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_csv(cache_file)
    print(f"  [cache] 株価データ保存: {cache_file}")
    return df


def get_fundamentals(force_refresh: bool = False) -> dict:
    """
    各銘柄の財務指標を取得（現時点の値）。
    ⚠️ 注意: 過去時点の財務データではない。ルックアヘッドバイアスの可能性あり。

    Returns:
        dict: {ticker: {priceToBook, returnOnEquity, ...}}
    """
    _ensure_cache()
    cache_file = os.path.join(CACHE_DIR, 'fundamentals.json')

    if os.path.exists(cache_file) and not force_refresh:
        print("  [cache] 財務データ読み込み中...")
        with open(cache_file, 'r') as f:
            return json.load(f)

    print("  [fetch] 財務データをダウンロード中...")
    fundamentals = {}
    for ticker in TICKERS:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {
                'priceToBook':      info.get('priceToBook'),
                'returnOnEquity':   info.get('returnOnEquity'),
                'debtToEquity':     info.get('debtToEquity'),
                'trailingPE':       info.get('trailingPE'),
                'revenueGrowth':    info.get('revenueGrowth'),
                'grossMargins':     info.get('grossMargins'),
            }
            print(f"    {ticker}: OK")
        except Exception as e:
            print(f"    {ticker}: ERROR ({e})")
            fundamentals[ticker] = {}

    with open(cache_file, 'w') as f:
        json.dump(fundamentals, f, ensure_ascii=False, indent=2)
    print(f"  [cache] 財務データ保存: {cache_file}")
    return fundamentals
