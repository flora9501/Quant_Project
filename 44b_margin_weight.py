# 44b_margin_weight.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay, benchmark
from factorlab.factors import load_cached, load_price_data


def log(m):
    print(m, flush=True)


p = load_panel()
f = load_cached("margin_chg").reindex(index=p.idx, columns=p.cols)
b = benchmark(p)

base = replay(smooth_score(composite(p)), p)
base_ann = base.ann()
log(f"基准：年化 {base_ann:.2%}  换手 {base.ann_turnover():.0%}")

for w in (0.20, 0.10):
    aug = replay(smooth_score(composite(p) * (1 - w) + f.fillna(0.0) * w), p)
    d = (aug.net - base.net).dropna()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan
    log(f"w={w:<5} 年化 {aug.ann():.2%}  Δ {(aug.ann()-base_ann)*100:+.2f}pp  "
        f"t {t:+.2f}  换手 {aug.ann_turnover():.0%}")

# 池子偏差：持仓的市值与换手暴露
daily, basic = load_price_data()
cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
            .reindex(index=p.idx, columns=p.cols))
turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
             .reindex(index=p.idx, columns=p.cols))
lnmv = np.log(cmv * 1e4)


def expo(r, mat):
    vals = []
    for d_, names in r.holdings.items():
        if d_ not in mat.index:
            continue
        cols = [x for x in names if x in mat.columns]
        if cols:
            vals.append(float(mat.loc[d_, cols].mean()))
    return float(np.mean(vals)) if vals else np.nan


aug20 = replay(smooth_score(composite(p) * 0.8 + f.fillna(0.0) * 0.2), p)
log(f"\n持仓暴露  ln市值  基准 {expo(base, lnmv):.2f}  +margin {expo(aug20, lnmv):.2f}")
log(f"          换手率  基准 {expo(base, turn):.2f}  +margin {expo(aug20, turn):.2f}")

# 两融覆盖率的代价：两融标的的平均市值 vs 全池
rz = f.notna()
log(f"\n两融标的占比 {float(rz.where(p.mask).to_numpy().mean()/p.mask.to_numpy().mean()):.1%}")
