import numpy as np
import pandas as pd

def _zscore(s):
    std = s.std()
    return pd.Series(0.0, index=s.index) if std == 0 or np.isnan(std) else (s - s.mean()) / std

def momentum_score(prices, date):
    end, start = date - pd.DateOffset(months=1), date - pd.DateOffset(months=13)
    w = prices.loc[start:end].dropna(axis=1, how='all')
    if len(w) < 20: return pd.Series(dtype=float)
    return ((w.ffill().iloc[-1] / w.ffill().iloc[0]) - 1).dropna()

def value_score(fundamentals):
    pb = {t: v['priceToBook'] for t, v in fundamentals.items() if v.get('priceToBook') and v['priceToBook'] > 0}
    return -pd.Series(pb) if pb else pd.Series(dtype=float)

def quality_score(fundamentals):
    scores = {}
    for t, v in fundamentals.items():
        roe = v.get('returnOnEquity')
        de  = v.get('debtToEquity')
        if roe is None or np.isnan(roe): continue
        scores[t] = roe - (0.3 * np.log1p(de) if de and de > 0 else 0)
    return pd.Series(scores) if scores else pd.Series(dtype=float)

def composite_score(prices, fundamentals, date):
    mom  = _zscore(momentum_score(prices, date))
    val  = _zscore(value_score(fundamentals))
    qual = _zscore(quality_score(fundamentals))
    df = pd.DataFrame({'mom': mom, 'val': val, 'qual': qual}).reindex(list(set(mom.index))).fillna(0.0)
    return df['mom']*0.5 + df['val']*0.25 + df['qual']*0.25

def select_stocks(scores, n=10):
    return scores.nlargest(n).index.tolist() if not scores.empty else []

STRATEGY_NAMES = {'MOM':'Momentum','VAL':'Value','QUAL':'Quality','COMP':'Composite'}
