"""
バックテストエンジン
2015〜2024年を3ヶ月ごとに区切り、各戦略の実績を計算する。

ルール:
  - 四半期ごとに上位10銘柄を等ウェイトで保有
  - 取引コスト・税金なし（簡略化）
  - ベンチマーク: 日経225（^N225）
  - 評価指標: アルファ / シャープレシオ / 最大ドローダウン / 勝率
"""

import pandas as pd
import numpy as np

from backtest.tickers import BENCHMARK, TICKERS
from backtest.strategies import (
    momentum_score,
    value_score,
    quality_score,
    composite_score,
    select_stocks,
    STRATEGY_NAMES,
)


# ─────────────────────────────────────────
# 四半期日付生成
# ─────────────────────────────────────────
def _quarter_dates(start: str, end: str) -> list:
    """テスト期間の四半期開始日リストを生成。"""
    dates = pd.date_range(start=start, end=end, freq='QS')  # Quarter Start
    return list(dates)


# ─────────────────────────────────────────
# 1クォーターのリターン計算
# ─────────────────────────────────────────
def _quarter_return(prices: pd.DataFrame, tickers: list, buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    """
    buy_date〜sell_date の等ウェイトポートフォリオリターンを計算。
    データが取得できない銘柄は除外。
    """
    if not tickers:
        return np.nan

    returns = []
    for t in tickers:
        if t not in prices.columns:
            continue
        # buy_date以降の最初の有効価格
        price_window = prices.loc[buy_date:sell_date, t].dropna()
        if len(price_window) < 2:
            continue
        ret = price_window.iloc[-1] / price_window.iloc[0] - 1
        returns.append(ret)

    return float(np.mean(returns)) if returns else np.nan


# ─────────────────────────────────────────
# メイン: バックテスト実行
# ─────────────────────────────────────────
def run_backtest(
    prices: pd.DataFrame,
    fundamentals: dict,
    test_start: str,
    test_end: str,
    n_stocks: int = 10,
) -> dict:
    """
    全戦略のバックテストを実行する。

    Returns:
        {
          'quarterly': DataFrame（四半期ごとの詳細）,
          'summary':   DataFrame（戦略別サマリー）,
          'holdings':  DataFrame（保有銘柄ログ）,
        }
    """
    quarter_starts = _quarter_dates(test_start, test_end)
    print(f"\n  テスト期間: {quarter_starts[0].date()} 〜 {quarter_starts[-1].date()}")
    print(f"  クォーター数: {len(quarter_starts)}")
    print(f"  保有銘柄数: {n_stocks}\n")

    # 戦略ごとの四半期リターンを格納
    records = []
    holdings_log = []

    for i, buy_date in enumerate(quarter_starts):
        sell_date = buy_date + pd.DateOffset(months=3)
        label = f"{buy_date.year}-Q{(buy_date.month - 1) // 3 + 1}"

        # ──── 各戦略でスコア計算 ────
        strategy_scores = {
            'MOM':  momentum_score(prices, buy_date),
            'VAL':  value_score(fundamentals),
            'QUAL': quality_score(fundamentals),
            'COMP': composite_score(prices, fundamentals, buy_date),
        }

        # ──── 銘柄選定 & リターン計算 ────
        row = {'quarter': label, 'buy_date': buy_date, 'sell_date': sell_date}

        for strat, scores in strategy_scores.items():
            selected = select_stocks(scores, n=n_stocks)
            ret = _quarter_return(prices, selected, buy_date, sell_date)
            row[f'{strat}_ret'] = ret
            row[f'{strat}_tickers'] = ','.join(selected)

            holdings_log.append({
                'quarter':  label,
                'strategy': strat,
                'tickers':  selected,
                'return':   ret,
            })

        # ──── ベンチマーク（日経225）リターン ────
        bm_ret = _quarter_return(prices, [BENCHMARK], buy_date, sell_date)
        row['BM_ret'] = bm_ret

        records.append(row)
        _print_progress(i, len(quarter_starts), label, row)

    quarterly = pd.DataFrame(records)
    summary   = _calc_summary(quarterly)
    holdings  = pd.DataFrame(holdings_log)

    return {
        'quarterly': quarterly,
        'summary':   summary,
        'holdings':  holdings,
    }


# ─────────────────────────────────────────
# サマリー指標の計算
# ─────────────────────────────────────────
def _calc_summary(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    各戦略の評価指標を計算。
      - 年率リターン（CAGR近似）
      - アルファ（対日経225）
      - シャープレシオ（四半期ベース、無リスク金利=0）
      - 最大ドローダウン
      - 勝率（ベンチマーク超過の四半期割合）
    """
    strategies = list(STRATEGY_NAMES.keys()) + ['BM']
    rows = []

    bm_rets = quarterly['BM_ret'].dropna()
    bm_cum  = (1 + bm_rets).prod()
    n       = len(bm_rets)

    for strat in strategies:
        col = f'{strat}_ret' if strat != 'BM' else 'BM_ret'
        if col not in quarterly.columns:
            continue

        rets = quarterly[col].dropna()
        if rets.empty:
            continue

        cum_ret  = (1 + rets).prod() - 1
        cagr     = (1 + cum_ret) ** (4 / len(rets)) - 1  # 四半期→年率
        alpha    = rets.mean() - bm_rets.reindex(rets.index).mean()
        alpha_yr = alpha * 4

        # シャープレシオ（年率換算）
        sharpe = (rets.mean() / rets.std() * np.sqrt(4)) if rets.std() > 0 else np.nan

        # 最大ドローダウン
        cum_curve = (1 + rets).cumprod()
        rolling_max = cum_curve.cummax()
        drawdowns = (cum_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        # 勝率（対ベンチマーク）
        if strat != 'BM':
            excess = rets.values - bm_rets.reindex(rets.index).values
            win_rate = (excess > 0).mean()
        else:
            win_rate = np.nan

        rows.append({
            '戦略':           STRATEGY_NAMES.get(strat, '日経225'),
            '累計リターン':    f"{cum_ret*100:.1f}%",
            '年率リターン':    f"{cagr*100:.1f}%",
            '年率アルファ':    f"{alpha_yr*100:+.1f}%",
            'シャープレシオ':  f"{sharpe:.2f}" if not np.isnan(sharpe) else '—',
            '最大ドローダウン': f"{max_dd*100:.1f}%",
            '勝率(対BM)':     f"{win_rate*100:.1f}%" if not np.isnan(win_rate) else '—',
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 進捗表示
# ─────────────────────────────────────────
def _print_progress(i: int, total: int, label: str, row: dict):
    bar_len = 20
    filled = int(bar_len * (i + 1) / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    mom  = row.get('MOM_ret',  float('nan'))
    comp = row.get('COMP_ret', float('nan'))
    bm   = row.get('BM_ret',   float('nan'))
    print(
        f"  [{bar}] {i+1:2d}/{total} {label} | "
        f"MOM:{mom*100:+5.1f}% COMP:{comp*100:+5.1f}% BM:{bm*100:+5.1f}%"
    )
