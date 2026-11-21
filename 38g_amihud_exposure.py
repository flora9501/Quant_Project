# 38g_amihud_exposure.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from factorlab.core import load_panel, composite, smooth_score, replay
from factorlab.factors import load_cached, load_price_data

p = load_panel()
f = load_cached("amihud").reindex(index=p.idx, columns=p.cols)

base = replay(smooth_score(composite(p)), p)
aug = replay(smooth_score(composite(p) * 0.8 + f.fillna(0.0) * 0.2), p)

daily, basic = load_price_data()
cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
            .reindex(index=p.idx, columns=p.cols))
turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
             .reindex(index=p.idx, columns=p.cols))
lnmv = np.log(cmv * 1e4)


def exposure(holdings, mat):
    out = {}
    for d, names in holdings.items():
        if d not in mat.index:
            continue
        cols = [c for c in names if c in mat.columns]
        if cols:
            out[d] = float(mat.loc[d, cols].mean())
    return np.mean(list(out.values())) if out else np.nan


print(f"{'组合':<10} {'ln市值':>9} {'换手率%':>9} {'只数':>6}")
for label, r in (("基准", base), ("+amihud", aug)):
    n = np.mean([len(v) for v in r.holdings.values()])
    print(f"{label:<10} {exposure(r.holdings, lnmv):>9.2f} "
          f"{exposure(r.holdings, turn):>9.2f} {n:>6.0f}")
