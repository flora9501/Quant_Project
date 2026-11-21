#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
49_topn_verify.py
验证「加宽组合」的三件事。

背景（48 号）
  N=50 → 400：净超额 4.47 → 5.15pp，换手 714% → 447%
  但 **毛超额几乎不变**（6.18 → 6.22），增益全部来自成本
  且 N=100 是洼地（净 3.51），疑因行业配额取整

三个检验
  ① 逐股成本 0.5 / 1 / 2 亿：换手降了，容量到底改善多少（这是它唯一真正的价值）
  ② TE / IR / 主动份额：N 变大后是不是"半个基准"
  ③ 持仓暴露：ln 市值、换手率

判据（预注册）
  ① N=400 在 1 亿下明显优于 N=50 → 有容量价值；若差异 <0.5pp → 只是账面数字
  ② 主动份额 <40% 或 TE 掉到 3% 以下 → 已经接近基准，不能算主动策略
  ③ 暴露差 >0.2（ln 市值）→ 赢的是 beta

用法
  .venv/bin/python -u 49_topn_verify.py
输出
  data/output_zoo/topn_verify.csv
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

NS = [50, 100, 200, 400]
NS_COST = [50, 200, 400]
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]
COST = 0.0012
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
    q = aum * part
    c = k * sig * np.sqrt(q / adv.where(adv > 0))
    return (c + floor).clip(upper=CAP)


def stats(r, bench):
    net = r.net
    b = bench.reindex(net.index).fillna(0.0)
    ex = (net - b).dropna()
    te = float(ex.std() * np.sqrt(252))
    ir = float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else np.nan
    ir_ratio = (geom(net) - geom(b)) / te if te > 0 else np.nan
    return {
        "年化": geom(net),
        "超额pp": (geom(net) - geom(b)) * 100,
        "TE": te,
        "IR(算术)": ir,
        "IR(比值)": ir_ratio,
        "换手": r.ann_turnover(),
    }


def active_share(holdings, mask):
    vals = []
    for d, names in holdings.items():
        if d not in mask.index:
            continue
        n = int(mask.loc[d].sum())
        if n == 0 or not names:
            continue
        wb = 1.0 / n
        wp = 1.0 / len(names)
        s = len(names) * abs(wp - wb) + (n - len(names)) * wb
        vals.append(0.5 * s)
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
    score = smooth_score(composite(p))
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    log("\n② 各 N 回放（12bp）...")
    reps, rows = {}, []
    for n in NS:
        r = replay(score, p, top_n=n)
        reps[n] = r
        st = stats(r, b)
        st["N"] = n
        st["毛超额pp"] = st["超额pp"] + r.ann_turnover() * COST * 2 * 100
        st["主动份额"] = active_share(r.holdings, p.mask)
        st["ln市值"] = expo(r.holdings, lnmv)
        st["换手率"] = expo(r.holdings, turn)
        rows.append(st)
        log(f"   N={n:<4} 年化 {st['年化']:+.2%}  超额 {st['超额pp']:+.2f}pp  "
            f"毛 {st['毛超额pp']:+.2f}pp  TE {st['TE']:.2%}  "
            f"IR {st['IR(比值)']:.2f}  换手 {st['换手']:.0%}  "
            f"主动 {st['主动份额']:.0%}")

    tab = pd.DataFrame(rows).set_index("N")
    tab = tab[["年化", "超额pp", "毛超额pp", "TE", "IR(算术)", "IR(比值)",
               "换手", "主动份额", "ln市值", "换手率"]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手", "主动份额"):
        tab[c] = tab[c].round(4)
    for c in ("IR(算术)", "IR(比值)"):
        tab[c] = tab[c].round(2)
    for c in ("ln市值", "换手率"):
        tab[c] = tab[c].round(2)

    log("\n" + "=" * 104)
    log("【② TE / IR / 主动份额】")
    log("=" * 104)
    log(tab.to_string())

    log("\n③ 逐股成本（0.5 / 1 / 2 亿）...")
    cost_rows = []
    for n in NS_COST:
        row = {"N": n, "12bp": round(float(tab.loc[n, "超额pp"]), 2)}
        for aum, label in AUMS:
            cm = cost_matrix(p, daily, aum)
            r = replay(score, p, top_n=n, cost=cm)
            row[label] = round(stats(r, b)["超额pp"], 2)
        cost_rows.append(row)
        log("   " + str(row))
    cost_tab = pd.DataFrame(cost_rows).set_index("N")

    log("\n" + "=" * 104)
    log("【① 成本敏感性（超额 pp）】")
    log("=" * 104)
    log(cost_tab.to_string())

    # ---------- 判读 ----------
    g50 = float(tab.loc[50, "毛超额pp"])
    g400 = float(tab.loc[400, "毛超额pp"])
    d_1e = float(cost_tab.loc[400, "1.0亿"] - cost_tab.loc[50, "1.0亿"])
    as400 = float(tab.loc[400, "主动份额"])
    te400 = float(tab.loc[400, "TE"])
    dmv = abs(float(tab.loc[400, "ln市值"] - tab.loc[50, "ln市值"]))

    log("\n" + "=" * 104)
    log("【判读】")
    log("=" * 104)
    log(f"  毛超额  N=50 {g50:+.2f}pp  →  N=400 {g400:+.2f}pp   "
        f"（差 {g400 - g50:+.2f}pp，≈0 说明不是 alpha 改善）")
    log(f"  ① 1 亿 AUM 下 N=400 − N=50 = {d_1e:+.2f}pp   "
        f"（>+0.5pp 才算有容量价值）")
    log(f"  ② N=400 主动份额 {as400:.0%}、TE {te400:.2%}   "
        f"（<40% 或 TE<3% → 接近基准）")
    log(f"  ③ ln 市值差 {dmv:.2f}   （>0.2 → 赢的是 beta）")

    ok_cap = d_1e > 0.5
    ok_active = (as400 >= 0.40) and (te400 >= 0.03)
    ok_expo = dmv <= 0.2
    log(f"\n  → 容量价值 {'✅' if ok_cap else '❌'}   "
        f"仍是主动策略 {'✅' if ok_active else '❌'}   "
        f"暴露未漂移 {'✅' if ok_expo else '❌'}")
    log("  三条都过 → 值得作为「实现层面的改进」写进文档；"
        "否则只是账面数字。")

    tab.to_csv(OUT / "topn_verify.csv", encoding="utf-8-sig")
    cost_tab.to_csv(OUT / "topn_cost.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT}/topn_verify.csv、topn_cost.csv")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
