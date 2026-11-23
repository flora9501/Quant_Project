#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
52b_exposure.py
turn_20 宽组合的暴露检查：+6.56pp 是信号，还是 beta 换皮？

背景
  52 号显示 turn_20 N=400：12bp 超额 6.56pp、IR 1.80、换手 329%，
  持仓换手率 0.85(N=50) → 1.28(N=400)，看似「低换手 beta 被宽度洗掉」。
  但 52 只查了「ln 市值」和「原始换手率」两个维度，不足以支撑这个结论。

本脚本补四个中性化 z 因子（rev_5 / turn_20 / vol_20 / ep）的持仓均值，
与 4 因子 N=400 逐项对比。z 因子已对行业 + ln 市值做过残差化，
所以这里的暴露差就是「风格差」，不掺行业和规模。

判据（预注册，先定不改）
  任一 |Δ暴露| > 0.15 z → 「beta 被洗掉」不成立，必须补归因
  四项 |Δ| 均 ≤ 0.15     → 暴露相近，超额更可能来自选股/宽度，值得进一步验证

用法
  .venv/bin/python -u 52b_exposure.py           # 3 次回放，约 4 分钟
  .venv/bin/python -u 52b_exposure.py --cost    # 加 1 亿成本，约 8 分钟
输出
  data/output_zoo/turn20_exposure.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, FACTORS)
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

CAP = 0.02
WITH_COST = "--cost" in sys.argv
CONFIGS = [("4因子", "s4", 400), ("turn_20", "s20", 50), ("turn_20", "s20", 400)]
BASE = ("4因子", 400)
THRESH_Z = 0.15


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


def mstats(r, bench):
    net = r.net
    b = bench.reindex(net.index).fillna(0.0)
    ex = (net - b).dropna()
    te = float(ex.std() * np.sqrt(252))
    g_net, g_bm = geom(net), geom(b)
    gross = r.gross.reindex(net.index) if getattr(r, "gross", None) is not None else None
    return {
        "年化": g_net,
        "超额pp": (g_net - g_bm) * 100,
        "毛超额pp": (geom(gross) - g_bm) * 100 if gross is not None else np.nan,
        "TE": te,
        "IR": (g_net - g_bm) / te if te > 0 else np.nan,
        "换手": r.ann_turnover(),
    }


def expo(holdings, mat, skipna=True):
    v = []
    for d, names in holdings.items():
        if d not in mat.index or not names:
            continue
        cols = [c for c in names if c in mat.columns]
        if not cols:
            continue
        x = mat.loc[d, cols]
        if skipna:
            x = x.dropna()
        if len(x):
            v.append(float(x.mean()))
    return float(np.mean(v)) if v else np.nan


def nan_share(holdings, mat):
    num = den = 0
    for d, names in holdings.items():
        if d not in mat.index:
            continue
        for c in names:
            if c in mat.columns:
                den += 1
                num += int(pd.isna(mat.at[d, c]))
    return num / den if den else np.nan


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

    # 全池参照（检验「全池约 1.1」这个说法）
    tv = turn.where(p.mask)
    pool_naive = float(np.nanmean(tv.to_numpy()))
    pool_nonzero = float(np.nanmean(tv.replace(0, np.nan).to_numpy()))
    log(f"   全池 turnover_rate：含 0 口径 {pool_naive:.2f}%  /  "
        f"剔 0 口径 {pool_nonzero:.2f}%   ← 两者差很多时，"
        f"「全池约1.1」是被 0 摊薄的假象")

    score = {"s4": smooth_score(composite(p)),
             "s20": smooth_score(p.fac["turn_20"])}

    log("\n② 回放（12bp）...")
    reps, rows = {}, []
    for name, skey, n in CONFIGS:
        r = replay(score[skey], p, top_n=n)
        reps[(name, n)] = r
        d = mstats(r, b)
        d.update({"组合": name, "N": n,
                  "ln市值": expo(r.holdings, lnmv),
                  "持仓换手率": expo(r.holdings, turn),
                  "ep缺失占比": nan_share(r.holdings, p.fac["ep"])})
        for f in FACTORS:
            d[f"z_{f}"] = expo(r.holdings, p.fac[f])
        rows.append(d)
        log(f"   {name:<8} N={n:<4} 超额 {d['超额pp']:+.2f}pp  IR {d['IR']:.2f}  "
            f"TE {d['TE']:.2%}  换手 {d['换手']:.0%}  "
            f"lnMV {d['ln市值']:.2f}  持仓换手率 {d['持仓换手率']:.2f}")

    # ---------- 成本（可选）----------
    if WITH_COST:
        log("\n③ 逐股成本（1 亿，part = 1/n）...")
        sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
        amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                    .reindex(index=p.idx, columns=p.cols)) * 1000.0
        adv = amt.rolling(20, min_periods=20).mean()
        for name, skey, n in CONFIGS:
            c = 0.7 * sig * np.sqrt((1e8 * (1.0 / n)) / adv.where(adv > 0))
            cm = (c + 0.0003).clip(upper=CAP)
            r = replay(score[skey], p, top_n=n, cost=cm)
            v = mstats(r, b)["超额pp"]
            for row in rows:
                if row["组合"] == name and row["N"] == n:
                    row["1亿"] = v
            log(f"   {name:<8} N={n:<4} {v:+.2f}pp")

    tab = pd.DataFrame(rows).set_index(["组合", "N"])
    zcols = [f"z_{f}" for f in FACTORS]
    show = tab[["年化", "超额pp", "毛超额pp", "TE", "IR", "换手",
                "ln市值", "持仓换手率", "ep缺失占比"] + zcols +
               ([c for c in ["1亿"] if c in tab.columns])]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手", "ep缺失占比"):
        show[c] = (show[c] * 100).round(2)
    for c in ("IR", "ln市值", "持仓换手率"):
        show[c] = show[c].round(2)
    for c in zcols:
        show[c] = show[c].round(3)

    log("\n" + "=" * 116)
    log("【暴露汇总】（z 因子已中性化：正 = 更偏向「好」的一侧）")
    log("=" * 116)
    log(show.to_string())

    # ---------- 与基线逐项比 ----------
    base = tab.loc[BASE]
    log("\n【Δ暴露 vs 4因子 N=400】")
    breach = []
    for name, n in tab.index:
        if (name, n) == BASE:
            continue
        ds = {f: float(tab.loc[(name, n), f"z_{f}"] - base[f"z_{f}"]) for f in FACTORS}
        s = "  ".join(f"{f} {v:+.3f}" for f, v in ds.items())
        log(f"   {name} N={n}: {s}")
        for f, v in ds.items():
            if abs(v) > THRESH_Z:
                breach.append((name, n, f, v))
    log(f"   持仓换手率差：turn_20 N=400 {float(tab.loc[('turn_20', 400), '持仓换手率'] - base['持仓换手率']):+.2f}")

    log("\n" + "=" * 116)
    log("【判读（阈值 |Δz| > 0.15 先定）】")
    log("=" * 116)
    if breach:
        for name, n, f, v in breach:
            log(f"  ⚠ {name} N={n} 在 {f} 上偏离 {v:+.3f} → 超阈值")
        log("\n  → 「beta 被洗掉」不成立。这些超额里至少有一部分是"
            "另一个风格的暴露，需要做风格归因才能判断。")
    else:
        log("  → 四项暴露均在 ±0.15 内，与 4 因子 N=400 相近：")
        log("     不能说它只是换了一个 beta；超额更可能来自宽度带来的选股分散。")
        log("     但这仍不等于 alpha——turn_20 本身是已知的低关注度溢价。")

    tab.to_csv(OUT / "turn20_exposure.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'turn20_exposure.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
