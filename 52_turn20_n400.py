#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
52_turn20_n400.py
最后一个组合尝试：turn_20（低换手因子）+ 宽组合。

为什么值得试
  turn_20 单因子 N=50：超额 6.68pp、IR 0.98，但持仓换手率 0.85（vs 1.04）
  → 疑为低换手 level beta。
  加宽到 N=400 后，每个行业要选 10–13 只「最低换手」，
  平均换手率会被抬回接近全池 → **beta 暴露可能自动下降**。
  如果暴露回落而超额不降，那它更可能是真信号。

对照
  4 因子 N=400：超额 5.15 / 毛 6.22 / TE 3.90% / IR 1.32 / 换手 447% / 主动 60%

用法
  .venv/bin/python -u 52_turn20_n400.py     # 约 5–6 分钟
输出
  data/output_zoo/turn20_n400.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark)
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
CAP = 0.02
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def cost_matrix(p, daily, aum, k=0.7, floor=0.0003, part=1.0 / 50.0):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=CAP)


def stats(r, b):
    net = r.net
    bb = b.reindex(net.index).fillna(0.0)
    ex = (net - bb).dropna()
    te = float(ex.std() * np.sqrt(252))
    return {"年化": geom(net),
            "超额pp": (geom(net) - geom(bb)) * 100,
            "毛超额pp": (geom(net) - geom(bb)) * 100 + r.ann_turnover() * 0.0012 * 2 * 100,
            "TE": te,
            "IR": ((geom(net) - geom(bb)) / te) if te > 0 else np.nan,
            "换手": r.ann_turnover()}


def active_share(holdings, mask):
    v = []
    for d, names in holdings.items():
        if d not in mask.index or not names:
            continue
        n = int(mask.loc[d].sum())
        if n:
            v.append(0.5 * (len(names) * abs(1 / len(names) - 1 / n)
                            + (n - len(names)) / n))
    return float(np.mean(v)) if v else np.nan


def expo(holdings, mat):
    v = []
    for d, names in holdings.items():
        if d not in mat.index:
            continue
        cols = [x for x in names if x in mat.columns]
        if cols:
            v.append(float(mat.loc[d, cols].mean()))
    return float(np.mean(v)) if v else np.nan


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=p.idx, columns=p.cols))
    turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
                 .reindex(index=p.idx, columns=p.cols))
    lnmv = np.log(cmv * 1e4)
    b = benchmark(p)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    s4 = smooth_score(composite(p))
    s20 = smooth_score(p.fac["turn_20"])

    combos = [("4 因子", s4, 400), ("turn_20", s20, 50), ("turn_20", s20, 400)]

    log("\n② 12bp 回放...")
    rows, reps = [], {}
    for name, sc, n in combos:
        r = replay(sc, p, top_n=n)
        reps[(name, n)] = r
        d = stats(r, b)
        d.update({"组合": name, "N": n,
                  "主动份额": active_share(r.holdings, p.mask),
                  "持仓换手率": expo(r.holdings, turn),
                  "持仓ln市值": expo(r.holdings, lnmv)})
        rows.append(d)
        log(f"   {name:<9} N={n:<4} 超额 {d['超额pp']:+.2f}pp  毛 {d['毛超额pp']:+.2f}pp  "
            f"TE {d['TE']:.2%}  IR {d['IR']:.2f}  换手 {d['换手']:.0%}  "
            f"持仓换手率 {d['持仓换手率']:.2f}  ln市值 {d['持仓ln市值']:.2f}")

    log("\n③ turn_20 N=400 的逐股成本...")
    cms = {lab: cost_matrix(p, daily, aum) for aum, lab in AUMS}
    for lab in cms:
        r = replay(s20, p, top_n=400, cost=cms[lab])
        v = stats(r, b)["超额pp"]
        for row in rows:
            if row["组合"] == "turn_20" and row["N"] == 400:
                row[lab] = v
        log(f"   {lab}: {v:+.2f}pp")

    tab = pd.DataFrame(rows).set_index(["组合", "N"])
    cols = ["年化", "超额pp", "毛超额pp", "TE", "IR", "换手",
            "主动份额", "持仓换手率", "持仓ln市值"] + [l for _, l in AUMS]
    tab = tab[[c for c in cols if c in tab.columns]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手", "主动份额"):
        tab[c] = tab[c].round(4)
    for c in ("IR", "持仓换手率", "持仓ln市值"):
        tab[c] = tab[c].round(2)

    log("\n" + "=" * 112)
    log("【汇总】")
    log("=" * 112)
    log(tab.to_string())

    r400 = tab.loc[("turn_20", 400)]
    r50 = tab.loc[("turn_20", 50)]
    base = tab.loc[("4 因子", 400)]

    log("\n【判读：宽组合有没有把 beta 洗掉？】")
    log(f"  turn_20 持仓换手率  N=50: {r50['持仓换手率']:.2f}  →  "
        f"N=400: {r400['持仓换手率']:.2f}   （全池约 1.1）")
    log(f"  ln 市值差（vs 4因子 N=400 {base['持仓ln市值']:.2f}）："
        f"{r400['持仓ln市值'] - base['持仓ln市值']:+.2f}")
    log(f"  超额：N=50 {r50['超额pp']:+.2f}pp → N=400 {r400['超额pp']:+.2f}pp")
    log(f"  IR  ：N=50 {r50['IR']:.2f} → N=400 {r400['IR']:.2f}")
    log("")
    log("  若持仓换手率明显回升（→1.0）且超额不降 → beta 被洗掉，信号更可信")
    log("  若持仓换手率仍低（<0.7）         → 纯流动性下沉，关闭")
    log("  若超额随 N 大幅下降               → 说明原来吃的是集中度 + beta")

    tab.to_csv(OUT / "turn20_n400.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'turn20_n400.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
