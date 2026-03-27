"""
投資戦略モジュール
ルールベースで銘柄スコアを計算する。学習・フィードバックなし。

実装戦略:
  MOM  : モメンタム（12-1ヶ月リターン）
  VAL  : バリュー（低PBR）         ⚠️ 現時点財務データ使用
  QUAL : クオリティ（高ROE・低負債） ⚠️ 現時点財務データ使用
  COMP : 複合（MOM + VAL + QUAL の正規化合算）
"""

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score正規化。標準偏差が0の場合は0を返す。"""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


# ─────────────────────────────────────────
# 1. モメンタム戦略
# ─────────────────────────────────────────
def momentum_score(prices: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """
    12-1ヶ月モメンタム（Jegadeesh & Titman 1993）。
    - 直近12ヶ月のリターンから直近1ヶ月を除く
    - 純粋に価格データのみ使用 → ルックアヘッドバイアスなし

    Args:
        prices: 全銘柄の日次終値DataFrame
        date:   判断時点（この日以前のデータのみ使用）
    """
    end   = date - pd.DateOffset(months=1)
    start = date - pd.DateOffset(months=13)

    window = prices.loc[start:end].dropna(axis=1, how='all')
    if len(window) < 20:
        return pd.Series(dtype=float)

    first = window.ffill().iloc[0]
    last  = window.ffill().iloc[-1]
    ret   = (last / first) - 1

    # データ不足銘柄を除外
    valid = ret.dropna()
    return valid


# ─────────────────────────────────────────
# 2. バリュー戦略
# ─────────────────────────────────────────
def value_score(fundamentals: dict) -> pd.Series:
    """
    低PBR戦略（Fama & French 1992）。
    PBRが低いほどスコア高。
    ⚠️ 現時点財務データ使用のため、過去のバックテストには近似値。

    Args:
        fundamentals: {ticker: {priceToBook: float, ...}}
    """
    pb_dict = {}
    for ticker, info in fundamentals.items():
        pb = info.get('priceToBook')
        if pb and pb > 0:
            pb_dict[ticker] = pb

    if not pb_dict:
        return pd.Series(dtype=float)

    pb_series = pd.Series(pb_dict)
    return -pb_series  # 低いほど良い → 符号反転


# ─────────────────────────────────────────
# 3. クオリティ戦略
# ─────────────────────────────────────────
def quality_score(fundamentals: dict) -> pd.Series:
    """
    高ROE・低負債戦略（Novy-Marx 2013）。
    ROEが高く、負債比率が低いほどスコア高。
    ⚠️ 現時点財務データ使用のため、過去のバックテストには近似値。

    Args:
        fundamentals: {ticker: {returnOnEquity, debtToEquity, ...}}
    """
    scores = {}
    for ticker, info in fundamentals.items():
        roe = info.get('returnOnEquity')
        de  = info.get('debtToEquity')
        if roe is None or np.isnan(roe):
            continue
        score = roe
        if de is not None and not np.isnan(de) and de > 0:
            score -= 0.3 * np.log1p(de)  # 高負債をペナルティ
        scores[ticker] = score

    if not scores:
        return pd.Series(dtype=float)

    return pd.Series(scores)


# ─────────────────────────────────────────
# 4. 複合戦略
# ─────────────────────────────────────────
def composite_score(
    prices: pd.DataFrame,
    fundamentals: dict,
    date: pd.Timestamp,
    weights: dict = None,
) -> pd.Series:
    """
    MOM + VAL + QUAL をZ-score正規化して合算。
    デフォルトウェイト: MOM=0.5, VAL=0.25, QUAL=0.25
    モメンタムは価格のみで計算できるため、信頼度が高い分ウェイトを高めに設定。

    Args:
        prices:       全銘柄の日次終値DataFrame
        fundamentals: {ticker: {...}} 財務データ
        date:         判断時点
        weights:      {'mom': float, 'val': float, 'qual': float}
    """
    if weights is None:
        weights = {'mom': 0.5, 'val': 0.25, 'qual': 0.25}

    mom  = _zscore(momentum_score(prices, date))
    val  = _zscore(value_score(fundamentals))
    qual = _zscore(quality_score(fundamentals))

    # 共通ティッカーのみ
    tickers = set(mom.index)
    df = pd.DataFrame({
        'mom':  mom.reindex(list(tickers)),
        'val':  val.reindex(list(tickers)),
        'qual': qual.reindex(list(tickers)),
    }).fillna(0.0)

    df['score'] = (
        weights['mom']  * df['mom'] +
        weights['val']  * df['val'] +
        weights['qual'] * df['qual']
    )
    return df['score']


# ─────────────────────────────────────────
# 銘柄選定
# ─────────────────────────────────────────
def select_stocks(scores: pd.Series, n: int = 10) -> list:
    """
    スコア上位n銘柄を選定する。
    スコアが計算できない場合は空リストを返す。
    """
    if scores.empty:
        return []
    return scores.nlargest(n).index.tolist()


# 全戦略のマップ（engine.pyから参照）
STRATEGY_NAMES = {
    'MOM':  'モメンタム戦略',
    'VAL':  'バリュー戦略',
    'QUAL': 'クオリティ戦略',
    'COMP': '複合戦略',
}
