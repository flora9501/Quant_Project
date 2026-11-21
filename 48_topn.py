# 48_topn.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from factorlab.core import load_panel, composite, smooth_score, replay, benchmark
from factorlab.factors import load_cached

p = load_panel()
comp = smooth_score(composite(p))
b = benchmark(p)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


def run(score, n, label):
    r = replay(score, p, top_n=n)
    ex = (geom(r.net) - geom(b.reindex(r.net.index).fillna(0))) * 100
    print(f"{label:<18} N={n:<4} 年化 {r.ann():+.2%}  超额 {ex:+.2f}pp  "
          f"换手 {r.ann_turnover():.0%}")
    return r


base = run(comp, 50, "基准（现行）")
for n in (100, 200, 400):
    run(comp, n, "基准")

print()
f = load_cached("margin_chg").reindex(index=p.idx, columns=p.cols)
for n in (50, 100, 200, 400):
    aug = smooth_score(composite(p) * 0.8 + f.fillna(0.0) * 0.2)
    run(aug, n, "+margin_chg 20%")
