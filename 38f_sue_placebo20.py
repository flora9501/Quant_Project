# 38f_sue_placebo20.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from factorlab.core import load_panel, smooth_score, replay, benchmark
from factorlab.factors import get_factor

START = "20190101"          # ★ 去掉 2018 空仓的假收益
SEEDS = 20
REBAL = 20


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def excess_pp(r, bench):
    net = r.net[r.net.index >= START]
    b = bench.reindex(net.index).fillna(0.0)
    return (geom(net) - geom(b)) * 100


p = load_panel()
sue = get_factor("sue", p)
avail = sue.notna()
bench = benchmark(p)

real = excess_pp(replay(smooth_score(sue, rebal=REBAL), p, rebal=REBAL), bench)
print(f"真实 SUE（rebal={REBAL}，2019 起算）：{real:+.2f}pp\n")

rng = np.random.default_rng(20261121)
vals = []
print(f"{'seed':>6} {'超额pp':>10}")
for k in range(SEEDS):
    z = pd.DataFrame(rng.standard_normal(sue.shape),
                     index=sue.index, columns=sue.columns).where(avail)
    r = replay(smooth_score(z, rebal=REBAL), p, rebal=REBAL)
    v = excess_pp(r, bench)
    vals.append(v)
    print(f"{k:>6} {v:>+10.2f}")

vals = np.array(vals)
pval = float((vals >= real).mean())
print(f"\nplacebo：均值 {vals.mean():+.2f}pp  sd {vals.std(ddof=1):.2f}  "
      f"区间 [{vals.min():+.2f}, {vals.max():+.2f}]")
print(f"经验 p 值（placebo ≥ 真实）：{pval:.2f}   （{SEEDS} 次抽样）")
print(f"真实 − placebo 均值 = {real - vals.mean():+.2f}pp")
