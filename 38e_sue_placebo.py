# 38e_sue_placebo.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from factorlab.core import load_panel, smooth_score, replay, benchmark
from factorlab.factors import get_factor


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


p = load_panel()
sue = get_factor("sue", p)
avail = sue.notna()
bench = benchmark(p)
b = geom(bench)

n_av = avail.sum(axis=1)
print(f"每日可用标的：均值 {n_av.mean():.0f}，最少 {int(n_av.min())}")
print(f"等权基准年化 {b:.2%}\n")

print(f"{'seed':>6} {'年化':>9} {'对标等权':>11} {'换手':>7}")
rng = np.random.default_rng(20261121)
vals = []
for k in range(5):
    z = pd.DataFrame(rng.standard_normal(sue.shape),
                     index=sue.index, columns=sue.columns)
    rnd = z.where(avail)                      # ★ 同样只在这 59% 里选
    r = replay(smooth_score(rnd, rebal=20), p, rebal=20)
    d = (r.ann() - b) * 100
    vals.append(d)
    print(f"{k:>6} {r.ann():>+9.2%} {d:>+10.2f}pp {r.ann_turnover():>6.0%}")

print(f"\nplacebo 均值 {np.mean(vals):+.2f}pp  "
      f"区间 [{min(vals):+.2f}, {max(vals):+.2f}]")
print("对照：真实 SUE 在 rebal=20 时是 +3.43pp")
