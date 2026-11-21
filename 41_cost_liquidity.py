#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
41_cost_liquidity.py
把"一律 12bp"换成按流动性计算的冲击成本。

成本模型（平方根律）
  c_i = k · σ_i,daily · sqrt(Q / ADV_i) + floor
  Q    = AUM × (1/TOP_N)        每次调仓单只票的交易额（近似）
  ADV  = 20 日平均成交额（元）
  k = 0.7，floor = 3bp，单股上限 2%

输出：不同 AUM 下与"扁平 12bp"的对照。
用法
  .venv/bin/python -u 41_cost_liquidity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, TOP_N, COST)

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def cost_matrix(p, daily, aum, k=0.7, floor=0.0003, part=1.0 / TOP_N,
                cap=0.02):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0        # 千元 → 元
    adv = amt.rolling(20, min_periods=20).mean()
    q = aum * part
    c = k * sig * np.sqrt(q / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def main():
    log("① 加载面板与成交额...")
    p = load_panel()
    daily, basic = load_price_data = None, None
    from factorlab.factors import load_price_data
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    score = smooth_score(composite(p))
    b = benchmark(p)
    b_ann = geom(b)

    rows = []

    log("\n② 扁平 12bp（现行口径）...")
    r0 = replay(score, p)
    rows.append({
        "方案": "扁平 12bp", "AUM": "—", "年化": round(r0.ann(), 4),
        "vs EW": round((r0.ann() - b_ann) * 100, 2),
        "换手": round(r0.ann_turnover(), 4),
        "平均单边bp": round(COST * 1e4, 1),
        "成本pp/年": round(float(r0.cost_amt.mean() * 252 * 100)
                           if r0.cost_amt is not None else r0.ann() * 0, 2),
    })
    log(f"   年化 {r0.ann():.2%}  vs EW {(r0.ann() - b_ann)*100:+.2f}pp  "
        f"换手 {r0.ann_turnover():.0%}")

    for aum in (0.5e8, 1e8, 2e8, 3e8):
        log(f"\n③ AUM {aum/1e8:.1f} 亿，逐股成本...")
        cm = cost_matrix(p, daily, aum)
        med = float(np.nanmedian(cm.to_numpy()))
        p90 = float(np.nanpercentile(cm.to_numpy(), 90))
        log(f"   单股成本 中位 {med*1e4:.1f}bp  p90 {p90*1e4:.1f}bp")

        r = replay(score, p, cost=cm)
        dw_tot = float(r.turnover.sum() * 2)
        avg_bp = float(r.cost_amt.sum() / dw_tot * 1e4) if dw_tot > 0 else np.nan
        cost_pp = float(r.cost_amt.mean() * 252 * 100)
        rows.append({
            "方案": "逐股流动性成本", "AUM": f"{aum/1e8:.1f}亿",
            "年化": round(r.ann(), 4),
            "vs EW": round((r.ann() - b_ann) * 100, 2),
            "换手": round(r.ann_turnover(), 4),
            "平均单边bp": round(avg_bp, 1) if np.isfinite(avg_bp) else None,
            "成本pp/年": round(cost_pp, 2),
        })
        log(f"   年化 {r.ann():.2%}  vs EW {(r.ann()-b_ann)*100:+.2f}pp  "
            f"换手 {r.ann_turnover():.0%}  平均单边 {avg_bp:.1f}bp  "
            f"成本 {cost_pp:.2f}pp/年")

    tab = pd.DataFrame(rows)
    log("\n" + "=" * 96)
    log("【成本模型对照】")
    log("=" * 96)
    log(tab.to_string(index=False))
    tab.to_csv(OUT / "cost_liquidity.csv", index=False, encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'cost_liquidity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
