#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
51_final_checks.py
两个收尾检查，一个脚本跑完。

A. TOP_N 网格补点：300 / 400 / 500 / 600
   为什么：N=100 是洼地（毛 5.02 vs 50 的 6.18），说明曲线不单调。
   所以不能只拿 4 个点里的最大值说事，要看是「平台」还是「尖峰」。

B. turn_20 单飞的定性
   它单独 +6.67pp、毛 7.80pp（比 4 因子还高）。
   但它就是「买最不活跃的 50 只票」，而 9 桶表早写着低换手桶年化 10–19%。
   判据：持仓日均换手率。4 因子是 1.04%，若 turn_20 明显更低 → 纯流动性下沉。

用法
  .venv/bin/python -u 51_final_checks.py     # 约 8–10 分钟
输出
  data/output_zoo/topn_grid.csv
  data/output_zoo/turn20_solo.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, FACTORS)
from factorlab.factors import load_price_data, load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

NS = [300, 400, 500, 600]
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]
CAP = 0.02


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
    return {
        "年化": geom(net),
        "超额pp": (geom(net) - geom(bb)) * 100,
        "毛超额pp": (geom(net) - geom(bb)) * 100
                    + r.ann_turnover() * 0.0012 * 2 * 100,
        "TE": te,
        "IR": ((geom(net) - geom(bb)) / te) if te > 0 else np.nan,
        "换手": r.ann_turnover(),
    }


def active_share(holdings, mask):
    vals = []
    for d, names in holdings.items():
        if d not in mask.index or not names:
            continue
        n = int(mask.loc[d].sum())
        if n == 0:
            continue
        wb = 1.0 / n
        wp = 1.0 / len(names)
        vals.append(0.5 * (len(names) * abs(wp - wb) + (n - len(names)) * wb))
    return float(np.mean(vals)) if vals else np.nan


def expo(holdings, mat):
    vals = []
    for d, names in holdings.items():
        if d not in mat.index:
            continue
        cols = [x for x in names if x in mat.columns]
        if cols:
            vals.append(float(mat.loc[d, cols].mean()))
    return float(np.mean(vals)) if vals else np.nan


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
    score4 = smooth_score(composite(p))
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    # 参照：4 因子 N=50
    r4 = replay(score4, p, top_n=50)
    s4 = stats(r4, b)
    t4 = expo(r4.holdings, turn)
    m4 = expo(r4.holdings, lnmv)
    log(f"   参照 4 因子 N=50：超额 {s4['超额pp']:+.2f}pp  "
        f"持仓换手率 {t4:.2f}  持仓 ln市值 {m4:.2f}")

    # ---------- A. TOP_N 网格 ----------
    log("\n② A. TOP_N 网格 300–600 ...")
    rows = []
    for n in NS:
        r = replay(score4, p, top_n=n)
        d = stats(r, b)
        d["N"] = n
        d["主动份额"] = active_share(r.holdings, p.mask)
        # 1 亿成本
        cm = cost_matrix(p, daily, 1e8)
        r1 = replay(score4, p, top_n=n, cost=cm)
        d["1.0亿"] = stats(r1, b)["超额pp"]
        rows.append(d)
        log(f"   N={n:<4} 超额 {d['超额pp']:+.2f}pp  毛 {d['毛超额pp']:+.2f}pp  "
            f"TE {d['TE']:.2%}  IR {d['IR']:.2f}  换手 {d['换手']:.0%}  "
            f"主动 {d['主动份额']:.0%}  1亿 {d['1.0亿']:+.2f}pp")

    tabA = pd.DataFrame(rows).set_index("N")[
        ["年化", "超额pp", "毛超额pp", "TE", "IR", "换手", "主动份额", "1.0亿"]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手", "主动份额"):
        tabA[c] = tabA[c].round(4)
    for c in ("IR", "1.0亿"):
        tabA[c] = tabA[c].round(2)

    log("\n" + "=" * 106)
    log("【A】TOP_N 网格（含 50 / 100 / 200 的已知点作为对照）")
    log("=" * 106)
    log(tabA.to_string())
    inc = tabA["超额pp"]
    log(f"\n  超额范围 {inc.min():+.2f} ~ {inc.max():+.2f}pp（极差 {inc.max()-inc.min():.2f}pp）")
    log(f"  1 亿口径范围 {tabA['1.0亿'].min():+.2f} ~ {tabA['1.0亿'].max():+.2f}pp")
    log("  极差 <0.8pp → 平台（可以取中间值，不必挑最大）；"
        ">1.5pp → 尖峰（警惕在 4 个点里选最大）")

    # ---------- B. turn_20 单飞 ----------
    log("\n③ B. turn_20 单飞定性 ...")
    t20 = p.fac["turn_20"]
    r20 = replay(smooth_score(t20), p, top_n=50)
    s20 = stats(r20, b)
    t20_expo = expo(r20.holdings, turn)
    m20_expo = expo(r20.holdings, lnmv)
    log(f"   超额 {s20['超额pp']:+.2f}pp  毛 {s20['毛超额pp']:+.2f}pp  "
        f"换手 {s20['换手']:.0%}  TE {s20['TE']:.2%}  IR {s20['IR']:.2f}")
    log(f"   持仓日均换手率 {t20_expo:.2f}  （4 因子 {t4:.2f}）")
    log(f"   持仓 ln市值     {m20_expo:.2f}  （4 因子 {m4:.2f}）")

    rowsb = []
    for label, r in (("4 因子 N=50", r4), ("turn_20 单飞", r20)):
        d = stats(r, b)
        d["组合"] = label
        d["持仓换手率"] = expo(r.holdings, turn)
        d["持仓ln市值"] = expo(r.holdings, lnmv)
        for aum, lab in AUMS:
            d[lab] = stats(replay(
                smooth_score(composite(p) if label.startswith("4")
                             else t20), p, top_n=50,
                cost=cost_matrix(p, daily, aum)), b)["超额pp"]
        rowsb.append(d)

    tabB = pd.DataFrame(rowsb).set_index("组合")[
        ["年化", "超额pp", "毛超额pp", "TE", "IR", "换手",
         "持仓换手率", "持仓ln市值", "0.5亿", "1.0亿", "2.0亿"]]
    log("\n" + "=" * 106)
    log("【B】turn_20 单飞 vs 4 因子（含逐股成本）")
    log("=" * 106)
    log(tabB.round(3).to_string())

    ratio = t20_expo / t4 if t4 else np.nan
    log("\n【判读】")
    log(f"  持仓换手率之比 turn_20 / 4因子 = {ratio:.2f}")
    log(f"    <0.7 → 明显的流动性下沉，+6.67pp 是低换手 beta，不能算改进")
    log(f"    >=0.9 → 暴露相近，那还真的只是因子更集中，值得再看")
    log(f"  持仓 ln 市值差 {abs(m20_expo - m4):.2f}（>0.2 说明也下沉了小市值）")

    tabA.to_csv(OUT / "topn_grid.csv", encoding="utf-8-sig")
    tabB.to_csv(OUT / "turn20_solo.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT}/topn_grid.csv、turn20_solo.csv")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
