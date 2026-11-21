#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
40b_bucket_universe.py
把股票池限制到某个换手率桶，看超额从哪来。

两个基准（这是全部要点）
  vs 全池 EW  = 池子选择的 beta + 池内选股
  vs 同池 EW  = 只有池内选股 —— 这才是 alpha

用法
  .venv/bin/python -u 40b_bucket_universe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from factorlab.core import load_panel, composite, smooth_score, replay, benchmark
from factorlab.factors import load_price_data


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def metrics(r, panel, m):
    net = r.net
    b_full = benchmark(panel).reindex(net.index).fillna(0.0)
    b_pool = panel.fwd.where(m).mean(axis=1).reindex(net.index).fillna(0.0)
    ex_pool = (net - b_pool).dropna()
    return {
        "net": r.ann(),
        "turn": r.ann_turnover(),
        "vs_full": (r.ann() - geom(b_full)) * 100,
        "vs_pool": (r.ann() - geom(b_pool)) * 100,
        "te_pool": float(ex_pool.std() * np.sqrt(252)),
        "dev": max(r.ind_dev.values()) if r.ind_dev else np.nan,
    }


def main():
    p = load_panel()
    _, basic = load_price_data()
    turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
                 .reindex(index=p.idx, columns=p.cols))
    q = turn.where(p.mask).rank(axis=1, pct=True)      # 每日截面分位

    score = smooth_score(composite(p))

    cases = [
        ("全池（基准）",   p.mask),
        ("只留低+中换手",  p.mask & (q <= 2 / 3)),
        ("只留低换手",     p.mask & (q <= 1 / 3)),
        ("只留高换手",     p.mask & (q > 2 / 3)),
    ]

    rows = []
    log(f"{'池子':<14} {'年化':>8} {'换手':>7} {'vs全池':>9} {'vs同池':>9} "
        f"{'同池TE':>8} {'行业偏离':>9} {'只数':>6}")
    for label, m in cases:
        m = m.fillna(False).astype(bool)
        r = replay(score.where(m), p)
        d = metrics(r, p, m)
        d["池子"] = label
        d["只数"] = int(round(float(m.sum(axis=1).mean())))
        rows.append(d)
        log(f"{label:<14} {d['net']:>+8.2%} {d['turn']:>6.0%} "
            f"{d['vs_full']:>+8.2f} {d['vs_pool']:>+8.2f} "
            f"{d['te_pool']:>7.2%} {d['dev']:>8.2%} {d['只数']:>6}")

    tab = pd.DataFrame(rows).set_index("池子")
    tab.to_csv("data/output_zoo/bucket_universe.csv", encoding="utf-8-sig")

    log("\n【判读】")
    log("  vs全池 为正、vs同池 接近 0  → 收益来自池子选择（beta）")
    log("  vs同池 明显为正            → 池内选股真的更强（可能可用）")
    log("  vs同池 为负                → 这个池子里因子失效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
