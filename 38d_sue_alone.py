# 38d_sue_alone.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from factorlab.core import load_panel, smooth_score, replay, benchmark
from factorlab.factors import get_factor


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


p = load_panel()
sue = get_factor("sue", p)
print(f"SUE 覆盖 {float(sue.notna().values.mean()):.1%}（全网格）")

bench = benchmark(p)
b_ann = geom(bench)

print(f"\n{'rebal':>5} {'年化':>8} {'对标等权':>10} {'换手':>7} {'IR':>7} {'持仓数':>7}")
for rebal in (5, 20, 60):
    s = smooth_score(sue, rebal=rebal)
    r = replay(s, p, rebal=rebal)
    ex = (r.net - bench.reindex(r.net.index).fillna(0.0)).dropna()
    ir = float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else np.nan
    n_hold = int(np.mean([len(v) for v in r.holdings.values()])) if r.holdings else 0
    print(f"{rebal:>5} {r.ann():>+8.2%} {(r.ann()-b_ann)*100:>+9.2f}pp "
          f"{r.ann_turnover():>6.0%} {ir:>+7.2f} {n_hold:>7}")
