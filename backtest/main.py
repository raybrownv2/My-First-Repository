"""
エントリーポイント
GitHub Actions または手動で実行するメインスクリプト。

使用方法:
  python -m backtest.main
  python -m backtest.main --refresh   # キャッシュを無視して再取得
  python -m backtest.main --stocks 15 # 保有銘柄数を変更
"""

import argparse
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import get_prices, get_fundamentals
from backtest.engine import run_backtest
from backtest.report import generate_all
from backtest.tickers import TEST_START, TEST_END


def main():
    parser = argparse.ArgumentParser(description='日本株バックテストシステム')
    parser.add_argument('--refresh', action='store_true', help='キャッシュを無視してデータ再取得')
    parser.add_argument('--stocks',  type=int, default=10, help='保有銘柄数（デフォルト: 10）')
    args = parser.parse_args()

    print("=" * 60)
    print("  日本株 戦略バックテストシステム")
    print("  対象: 日経225主要構成銘柄")
    print(f"  期間: {TEST_START} 〜 {TEST_END}")
    print(f"  保有銘柄数: {args.stocks}")
    print("=" * 60)

    # ── Step 1: データ取得 ──
    print("\n[Step 1] データ取得")
    prices       = get_prices(force_refresh=args.refresh)
    fundamentals = get_fundamentals(force_refresh=args.refresh)
    print(f"  株価データ: {prices.shape[1]}銘柄 × {len(prices)}営業日")
    print(f"  財務データ: {len(fundamentals)}銘柄")

    # ── Step 2: バックテスト実行 ──
    print("\n[Step 2] バックテスト実行")
    results = run_backtest(
        prices=prices,
        fundamentals=fundamentals,
        test_start=TEST_START,
        test_end=TEST_END,
        n_stocks=args.stocks,
    )

    # ── Step 3: レポート生成 ──
    print("\n[Step 3] レポート生成")
    generate_all(results)

    print("\n✅ 完了！results/ ディレクトリを確認してください。")


if __name__ == '__main__':
    main()
