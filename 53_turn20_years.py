# 53_turn20_years.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import load_panel, composite, smooth_score, replay, benchmark


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


p = load_panel()
b = benchmark(p)
s4 = smooth_score(composite(p))
s20 = smooth_score(p.fac["turn_20"])

r4 = replay(s4, p, top_n=400)
r20 = replay(s20, p, top_n=400)

print(f"{'年':<6} {'4因子N400':>12} {'turn20N400':>12} {'差':>9}")
for y in ("2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"):
    row = {}
    for lab, r in (("a", r4), ("b", r20)):
        net = r.net[r.net.index.str[:4] == y]
        bb = b.reindex(net.index).fillna(0.0)
        row[lab] = (geom(net) - geom(bb)) * 100
    print(f"{y:<6} {row['a']:>+11.2f}% {row['b']:>+11.2f}% {row['b']-row['a']:>+8.2f}")

# 两年一组，看稳定性
print("\n二年一组（更稳）：")
for lo, hi in (("2018", "2019"), ("2020", "2021"), ("2022", "2023"), ("2024", "2025")):
    out = []
    for lab, r in (("4因子", r4), ("turn20", r20)):
        net = r.net[(r.net.index.str[:4] >= lo) & (r.net.index.str[:4] <= hi)]
        bb = b.reindex(net.index).fillna(0.0)
        out.append((geom(net) - geom(bb)) * 100)
    print(f"{lo}-{hi}  4因子 {out[0]:+6.2f}%  turn20 {out[1]:+6.2f}%  "
          f"差 {out[1]-out[0]:+6.2f}pp")
