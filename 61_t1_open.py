#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
61_t1_open.py  (v2)
执行口径敏感性：收盘成交 vs 次日开盘成交。

为什么做
  现行回测假设「收盘决策、收盘成交」，收益是 close_{t+1}/close_t − 1。
  真实交易很难成交在收盘价——收盘集合竞价只有 14:57–15:00 三分钟。
  改成「收盘决策、次日开盘成交」，收益变成 open_{t+2}/open_{t+1} − 1。

v2 修复（v1 的两个 bug）
  1. 隔夜 gap 用 np.nanmean，v1 用 np.mean 导致整列 nan
  2. 判读的 elif 链在 dlt ∈ (−2, −1] 时错标成「开盘更好」

口径分解
  收盘口径 = 隔夜gap(t+1) + 日内(t+1)
  开盘口径 = 日内(t+1) + 隔夜gap(t+2)
  两者之差 = gap(t+2) − gap(t+1)
  → 若开盘口径更低，说明「买入当天的隔夜 gap」高于「持有期的隔夜 gap」

判据（预注册）
  dlt < −2pp       硬约束：alpha 有一大块在隔夜缺口里
  −2 ≤ dlt < −0.5  实质成本：约吃掉 1/4 左右的超额
  |dlt| ≤ 0.5      执行中性
  dlt > 0.5        开盘执行更好（隔夜是负漂移）

注意
  这是敏感性分析，不替换生产口径。daily_run 按收盘记净值，
  回测必须与实盘一致，收盘口径仍是主口径。

用法
  .venv/bin/python -u 61_t1_open.py
输出
  data/output_zoo/t1_open.csv
"""
import copy
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
NS = [50, 400]


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def stats(r, b):
    net = r.net
    bb = b.reindex(net.index).fillna(0.0)
    ex = (net - bb).dropna()
    te = float(ex.std() * np.sqrt(252))
    return {
        "年化": geom(net),
        "超额pp": (geom(net) - geom(bb)) * 100,
        "毛超额pp": (geom(net) - geom(bb)) * 100 + r.ann_turnover() * 0.0012 * 2 * 100,
        "TE": te,
        "IR": ((geom(net) - geom(bb)) / te) if te > 0 else np.nan,
        "换手": r.ann_turnover(),
    }


def judge(dlt):
    if dlt < -2.0:
        return "硬约束：alpha 有一大块在隔夜缺口里"
    if dlt < -0.5:
        return "实质成本：约吃掉 1/4 左右的超额"
    if abs(dlt) <= 0.5:
        return "执行中性，可按容量与冲击择优"
    return "开盘执行更好，隔夜是负漂移"


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, _ = load_price_data(daily_cols=("pct_chg", "open", "close"))
    op = (daily.pivot(index="trade_date", columns="ts_code", values="open")
              .reindex(index=p.idx, columns=p.cols).replace(0, np.nan))
    cl = (daily.pivot(index="trade_date", columns="ts_code", values="close")
              .reindex(index=p.idx, columns=p.cols).replace(0, np.nan))
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    fwd_oo = op.shift(-2) / op.shift(-1) - 1          # 次日开盘成交
    gap = op.shift(-1) / cl - 1                        # 被跳过的隔夜段
    gap_mask = gap.where(p.mask)

    log(f"   全池日均：close-to-close {np.nanmean(p.fwd.to_numpy())*1e4:+.1f}bp  "
        f"open-to-open {np.nanmean(fwd_oo.to_numpy())*1e4:+.1f}bp  "
        f"隔夜gap {np.nanmean(gap.to_numpy())*1e4:+.1f}bp")

    p_oo = copy.copy(p)
    p_oo.fwd = fwd_oo

    score = smooth_score(composite(p))

    log("\n② 两种口径回放...")
    rows = []
    for n in NS:
        for label, pp in (("收盘成交", p), ("次日开盘", p_oo)):
            r = replay(score, pp, top_n=n)
            d = stats(r, benchmark(pp))
            d.update({"N": n, "口径": label})
            rows.append(d)
            log(f"   N={n:<4} {label:<6} 年化 {d['年化']:+.2%}  "
                f"超额 {d['超额pp']:+.2f}pp  毛 {d['毛超额pp']:+.2f}pp  "
                f"TE {d['TE']:.2%}  IR {d['IR']:.2f}")

    tab = pd.DataFrame(rows).set_index(["N", "口径"])[
        ["年化", "超额pp", "毛超额pp", "TE", "IR", "换手"]]
    for c in ("年化", "超额pp", "毛超额pp", "TE", "换手"):
        tab[c] = tab[c].round(4)
    tab["IR"] = tab["IR"].round(2)

    log("\n" + "=" * 96)
    log("【对比】")
    log("=" * 96)
    log(tab.to_string())

    # ---------- 机制：入场当天的隔夜 gap ----------
    log("\n③ 机制：入场当天的隔夜 gap vs 全池 ----------")
    pool_gap = float(np.nanmean(gap_mask.to_numpy())) * 1e4
    log(f"   全池（在册）隔夜 gap 均值：{pool_gap:+.1f}bp")

    mech = {}
    for n in NS:
        r = replay(score, p, top_n=n)          # 用收盘口径的持仓
        vals = []
        for d, names in r.holdings.items():
            if d not in gap.index:
                continue
            s = gap.loc[d, [c for c in names if c in gap.columns]].dropna()
            if len(s) >= 20:
                vals.append(float(s.mean()))
        if not vals:
            log(f"   N={n:<4} 无有效样本")
            continue
        arr = np.array(vals)
        mech[n] = {"mean": float(arr.mean()) * 1e4,
                   "median": float(np.median(arr)) * 1e4,
                   "share_pos": float((arr > 0).mean()),
                   "n": len(arr)}
        m = mech[n]
        log(f"   N={n:<4} 入场 gap 均值 {m['mean']:+.1f}bp  中位 {m['median']:+.1f}bp  "
            f"为正的比例 {m['share_pos']:.0%}  （{m['n']} 次调仓）")
        log(f"          相对全池：{m['mean'] - pool_gap:+.1f}bp")

    # ---------- 判读 ----------
    log("\n" + "=" * 96)
    log("【判读】")
    log("=" * 96)
    for n in NS:
        cc = float(tab.loc[(n, "收盘成交"), "超额pp"])
        oo = float(tab.loc[(n, "次日开盘"), "超额pp"])
        dlt = oo - cc
        ir_cc = float(tab.loc[(n, "收盘成交"), "IR"])
        ir_oo = float(tab.loc[(n, "次日开盘"), "IR"])
        log(f"   N={n:<4} 收盘 {cc:+.2f}pp → 开盘 {oo:+.2f}pp  "
            f"差 {dlt:+.2f}pp（占 {abs(dlt)/abs(cc):.0%}）  "
            f"IR {ir_cc:.2f} → {ir_oo:.2f}")
        log(f"          {judge(dlt)}")

    if mech:
        m = mech[400] if 400 in mech else list(mech.values())[0]
        log(f"\n   机制：入场日隔夜 gap {m['mean']:+.1f}bp vs 全池 {pool_gap:+.1f}bp")
        if m["mean"] > pool_gap + 3:
            log("   → 买入的票在次日开盘平均上涨，收盘买入能吃到这段，开盘买入吃不到。")
            log("     这就是超额下降的来源。")
        elif abs(m["mean"] - pool_gap) <= 3:
            log("   → 入场 gap 与全池接近，超额下降主要来自 TE 上升（开盘价更噪）。")
        else:
            log("   → 入场 gap 低于全池，开盘执行本应更好；检查 open/close 对齐。")

    log("\n  加宽组合的改进是否稳健：")
    for n in NS:
        log(f"   N={n:<4} 开盘口径超额 {tab.loc[(n,'次日开盘'),'超额pp']:+.2f}pp  "
            f"IR {tab.loc[(n,'次日开盘'),'IR']:.2f}")

    log("\n  说明：敏感性分析，不替换生产口径（daily_run 按收盘记账）。")
    log("  真实交易中开盘集合竞价容量大于收盘竞价，大规模下开盘的实际优势可能更大。")

    tab.to_csv(OUT / "t1_open.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 't1_open.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
