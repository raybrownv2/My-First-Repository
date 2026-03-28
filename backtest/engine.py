import pandas as pd
import numpy as np
from backtest.tickers import BENCHMARK
from backtest.strategies import momentum_score, value_score, quality_score, composite_score, select_stocks, STRATEGY_NAMES

def run_backtest(prices, fundamentals, test_start, test_end, n_stocks=10):
    quarters = list(pd.date_range(start=test_start, end=test_end, freq='QS'))
    print(f"  Period: {quarters[0].date()} to {quarters[-1].date()} ({len(quarters)} quarters)")
    records, holdings_log = [], []
    for i, buy in enumerate(quarters):
        sell  = buy + pd.DateOffset(months=3)
        label = f"{buy.year}-Q{(buy.month-1)//3+1}"
        row   = {'quarter': label, 'buy_date': buy, 'sell_date': sell}
        for strat, fn in [('MOM', lambda: momentum_score(prices, buy)),
                          ('VAL', lambda: value_score(fundamentals)),
                          ('QUAL',lambda: quality_score(fundamentals)),
                          ('COMP',lambda: composite_score(prices, fundamentals, buy))]:
            scores  = fn()
            picks   = select_stocks(scores, n_stocks)
            rets    = [prices.loc[buy:sell, t].dropna() for t in picks if t in prices.columns]
            rets    = [r.iloc[-1]/r.iloc[0]-1 for r in rets if len(r)>=2]
            row[f'{strat}_ret']     = float(np.mean(rets)) if rets else np.nan
            row[f'{strat}_tickers'] = ','.join(picks)
            holdings_log.append({'quarter':label,'strategy':strat,'tickers':picks,'return':row[f'{strat}_ret']})
        bm = prices.loc[buy:sell, BENCHMARK].dropna()
        row['BM_ret'] = float(bm.iloc[-1]/bm.iloc[0]-1) if len(bm)>=2 else np.nan
        records.append(row)
        bar = chr(9608)*int(20*(i+1)/len(quarters)) + chr(9617)*(20-int(20*(i+1)/len(quarters)))
        print(f"  [{bar}] {i+1:2d}/{len(quarters)} {label} | MOM:{row['MOM_ret']*100:+5.1f}%  COMP:{row['COMP_ret']*100:+5.1f}%  BM:{row['BM_ret']*100:+5.1f}%")
    qt = pd.DataFrame(records)
    return {'quarterly': qt, 'summary': _summary(qt), 'holdings': pd.DataFrame(holdings_log)}

def _summary(qt):
    bm = qt['BM_ret'].fillna(0)
    rows = []
    for strat, name in list(STRATEGY_NAMES.items()) + [('BM','Nikkei225')]:
        col  = 'BM_ret' if strat=='BM' else f'{strat}_ret'
        if col not in qt.columns: continue
        r    = qt[col].fillna(0)
        cum  = (1+r).prod()-1
        cagr = (1+cum)**(4/len(r))-1
        alp  = (r.mean()-bm.mean())*4
        sh   = r.mean()/r.std()*np.sqrt(4) if r.std()>0 else np.nan
        cc   = (1+r).cumprod()
        mdd  = ((cc-cc.cummax())/cc.cummax()).min()
        wr   = ((r.values-bm.values)>0).mean() if strat!='BM' else np.nan
        rows.append({'Strategy':name,'CumReturn':f"{cum*100:.1f}%",'CAGR':f"{cagr*100:.1f}%",
                     'Alpha':f"{alp*100:+.1f}%" if strat!='BM' else '-',
                     'Sharpe':f"{sh:.2f}" if not np.isnan(sh) else '-',
                     'MaxDD':f"{mdd*100:.1f}%",'WinRate':f"{wr*100:.1f}%" if not np.isnan(wr) else '-'})
    return pd.DataFrame(rows)
