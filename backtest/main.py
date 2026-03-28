import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.data import get_prices, get_fundamentals
from backtest.engine import run_backtest
from backtest.report import generate_all
from backtest.tickers import TEST_START, TEST_END

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--stocks', type=int, default=10)
    args = p.parse_args()
    print(f"Nikkei225 Backtest | {TEST_START} - {TEST_END} | {args.stocks} stocks")
    prices       = get_prices(force_refresh=args.refresh)
    fundamentals = get_fundamentals(force_refresh=args.refresh)
    results      = run_backtest(prices, fundamentals, TEST_START, TEST_END, args.stocks)
    generate_all(results)
    print("Done! Check results/ directory.")

if __name__ == '__main__': main()
