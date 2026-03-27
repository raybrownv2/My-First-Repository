"""
レポート生成モジュール
バックテスト結果をCSV・グラフとして results/ ディレクトリに保存する。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ヘッドレス環境対応
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
COLORS = {
    'MOM':  '#5c73f2',
    'VAL':  '#f59e0b',
    'QUAL': '#06b6d4',
    'COMP': '#22c55e',
    'BM':   '#94a3b8',
}


def _ensure_results():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_csv(results: dict):
    """四半期データ・サマリー・保有銘柄ログをCSV保存。"""
    _ensure_results()

    results['quarterly'].to_csv(
        os.path.join(RESULTS_DIR, 'quarterly_returns.csv'), index=False, encoding='utf-8-sig'
    )
    results['summary'].to_csv(
        os.path.join(RESULTS_DIR, 'summary.csv'), index=False, encoding='utf-8-sig'
    )
    results['holdings'].to_csv(
        os.path.join(RESULTS_DIR, 'holdings_log.csv'), index=False, encoding='utf-8-sig'
    )
    print("  [save] CSVファイル保存完了")


def plot_cumulative_returns(results: dict):
    """各戦略の累積リターン推移グラフ。"""
    _ensure_results()
    quarterly = results['quarterly']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    strategies = ['MOM', 'VAL', 'QUAL', 'COMP', 'BM']
    labels = {
        'MOM':  'モメンタム',
        'VAL':  'バリュー',
        'QUAL': 'クオリティ',
        'COMP': '複合',
        'BM':   '日経225',
    }

    x = range(len(quarterly))
    quarters = quarterly['quarter'].tolist()

    for strat in strategies:
        col = f'{strat}_ret' if strat != 'BM' else 'BM_ret'
        if col not in quarterly.columns:
            continue
        rets = quarterly[col].fillna(0)
        cum = (1 + rets).cumprod() - 1
        lw = 2.5 if strat == 'COMP' else 1.5
        ls = '-' if strat != 'BM' else '--'
        ax.plot(x, cum * 100, label=labels[strat], color=COLORS[strat],
                linewidth=lw, linestyle=ls)

    ax.axhline(y=0, color='#2e3248', linewidth=1)
    ax.set_xticks(list(x)[::4])
    ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), 4)],
                       rotation=45, ha='right', fontsize=9, color='#94a3b8')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.tick_params(colors='#94a3b8')
    ax.set_title('累積リターン推移（2015〜2024）', color='#e2e8f0', fontsize=14, pad=15)
    ax.set_ylabel('累積リターン', color='#94a3b8')
    ax.legend(facecolor='#232635', labelcolor='#e2e8f0', edgecolor='#2e3248', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3248')
    ax.grid(True, color='#2e3248', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'cumulative_returns.png')
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [save] グラフ保存: {path}")


def plot_quarterly_alpha(results: dict):
    """各四半期の対ベンチマーク超過リターン（棒グラフ）。"""
    _ensure_results()
    quarterly = results['quarterly']
    quarters  = quarterly['quarter'].tolist()
    bm_rets   = quarterly['BM_ret'].fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('対日経225 四半期超過リターン', color='#e2e8f0', fontsize=14, y=1.01)

    strats = ['MOM', 'VAL', 'QUAL', 'COMP']
    labels_map = {
        'MOM': 'モメンタム', 'VAL': 'バリュー',
        'QUAL': 'クオリティ', 'COMP': '複合'
    }

    for ax, strat in zip(axes.flat, strats):
        ax.set_facecolor('#1a1d27')
        col = f'{strat}_ret'
        if col not in quarterly.columns:
            continue
        rets   = quarterly[col].fillna(0)
        excess = (rets - bm_rets) * 100
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in excess]
        ax.bar(range(len(excess)), excess, color=colors, width=0.7)
        ax.axhline(0, color='#2e3248', linewidth=1)
        ax.set_title(labels_map[strat], color='#e2e8f0', fontsize=11)
        ax.tick_params(colors='#94a3b8', labelsize=7)
        ax.set_xticks(range(0, len(quarters), 4))
        ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), 4)],
                           rotation=45, ha='right')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3248')
        ax.grid(True, color='#2e3248', linewidth=0.5, alpha=0.5, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'quarterly_alpha.png')
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [save] グラフ保存: {path}")


def plot_summary_table(results: dict):
    """サマリーテーブルを画像として保存。"""
    _ensure_results()
    summary = results['summary']

    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    ax.axis('off')

    cols = list(summary.columns)
    rows = summary.values.tolist()

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # スタイリング
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1d27' if r % 2 == 0 else '#232635')
        cell.set_text_props(color='#e2e8f0')
        cell.set_edgecolor('#2e3248')
        if r == 0:
            cell.set_facecolor('#2e3248')
            cell.set_text_props(color='#94a3b8', fontweight='bold')

    plt.title('戦略別パフォーマンスサマリー', color='#e2e8f0', fontsize=13, pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'summary_table.png')
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"  [save] グラフ保存: {path}")


def print_summary(results: dict):
    """コンソールにサマリーを表示。"""
    print("\n" + "=" * 70)
    print("  バックテスト結果サマリー")
    print("=" * 70)
    print(results['summary'].to_string(index=False))
    print("=" * 70)
    print("\n⚠️  注意:")
    print("  - VAL・QUALは現時点財務データ使用（ルックアヘッドバイアスあり）")
    print("  - MOM・COMPの価格部分はバイアスなし")
    print("  - 取引コスト・税金は未考慮")
    print()


def generate_all(results: dict):
    """全レポートを生成。"""
    save_csv(results)
    plot_cumulative_returns(results)
    plot_quarterly_alpha(results)
    plot_summary_table(results)
    print_summary(results)
