import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
COLORS = {'MOM':'#5c73f2','VAL':'#f59e0b','QUAL':'#06b6d4','COMP':'#22c55e','BM':'#94a3b8'}
LABELS = {'MOM':'Momentum','VAL':'Value','QUAL':'Quality','COMP':'Composite','BM':'Nikkei225'}

def _ensure(): os.makedirs(RESULTS_DIR, exist_ok=True)

def save_csv(r):
    _ensure()
    r['quarterly'].to_csv(os.path.join(RESULTS_DIR,'quarterly_returns.csv'),index=False)
    r['summary'].to_csv(os.path.join(RESULTS_DIR,'summary.csv'),index=False)
    r['holdings'].to_csv(os.path.join(RESULTS_DIR,'holdings_log.csv'),index=False)
    print("  [save] CSV files saved")

def plot_all(r):
    _ensure()
    qt = r['quarterly']
    x  = range(len(qt))
    qs = qt['quarter'].tolist()

    # Cumulative returns
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d27')
    for s in ['MOM','VAL','QUAL','COMP','BM']:
        col = 'BM_ret' if s=='BM' else f'{s}_ret'
        if col not in qt.columns: continue
        cum = (1+qt[col].fillna(0)).cumprod()-1
        ax.plot(x, cum*100, label=LABELS[s], color=COLORS[s],
                linewidth=2.5 if s=='COMP' else 1.5, linestyle='--' if s=='BM' else '-')
    ax.axhline(0, color='#2e3248', lw=1)
    ax.set_xticks(list(x)[::4])
    ax.set_xticklabels([qs[i] for i in range(0,len(qs),4)], rotation=45, ha='right', fontsize=9, color='#94a3b8')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}%'))
    ax.tick_params(colors='#94a3b8'); ax.set_title('Cumulative Returns 2015-2024', color='#e2e8f0', fontsize=13)
    ax.legend(facecolor='#232635', labelcolor='#e2e8f0', edgecolor='#2e3248')
    ax.grid(True, color='#2e3248', lw=0.5, alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor('#2e3248')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'cumulative_returns.png'), dpi=150, facecolor='#0f1117')
    plt.close(); print("  [save] cumulative_returns.png")

    # Quarterly excess returns (COMP only)
    fig, ax = plt.subplots(figsize=(12,4))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d27')
    bm = qt['BM_ret'].fillna(0)
    exc = (qt['COMP_ret'].fillna(0) - bm)*100
    ax.bar(x, exc, color=['#22c55e' if v>=0 else '#ef4444' for v in exc], width=0.7)
    ax.axhline(0, color='#2e3248', lw=1)
    ax.set_xticks(list(x)[::4])
    ax.set_xticklabels([qs[i] for i in range(0,len(qs),4)], rotation=45, ha='right', fontsize=9, color='#94a3b8')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.1f}%'))
    ax.tick_params(colors='#94a3b8'); ax.set_title('Composite vs Nikkei225 Quarterly Excess Return', color='#e2e8f0', fontsize=13)
    ax.grid(True, color='#2e3248', lw=0.5, alpha=0.5, axis='y')
    for sp in ax.spines.values(): sp.set_edgecolor('#2e3248')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'quarterly_excess.png'), dpi=150, facecolor='#0f1117')
    plt.close(); print("  [save] quarterly_excess.png")

def generate_all(r):
    save_csv(r); plot_all(r)
    print("\n" + "="*60)
    print(r['summary'].to_string(index=False))
    print("="*60)
    print("Note: VAL/QUAL use current fundamentals (possible look-ahead bias)")
