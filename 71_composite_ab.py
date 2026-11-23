#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
71_composite_ab.py
A/B：composite 的部分缺失处理

  A（现状, core.composite）: acc / len(FACTORS)   → 缺 1 个因子，分数缩到 75%
  B（候选）               : acc / 有效因子数      → 按有效数平均

只读 factorlab.core，不改 core.py，不写 paper/。
输出 data/output_zoo/composite_ab.csv

用法
  .venv/bin/python -u 71_composite_ab.py --skip-cost   # 约 3 分钟
  .venv/bin/python -u 71_composite_ab.py               # 含 0.5/1/2 亿，约 10 分钟
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

 # noqa
from factorlab.core import load_panel, smooth_score, replay, benchmark, FACTORS, TOP_N
from factorlab.factors import load_price_data

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

N = TOP_N                      # 50
CAP = 0.02
AUMS = [(0.5e8, "0.5亿"), (1e8, "1.0亿"), (2e8, "2.0亿")]
SKIP_COST = "--skip-cost" in sys.argv


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


# ---------- 两个变体 ----------
def composite_variant(panel, mode):
    parts = [panel.fac[n] for n in FACTORS]
    acc = sum(d.fillna(0.0) for d in parts)
    cnt = sum(d.notna().astype(int) for d in parts)
    if mode == "len":
        denom = float(len(parts))
    else:
        denom = cnt.astype(float).replace(0, np.nan)
    return (acc / denom).where(cnt > 0)


# ---------- 指标 ----------
def metrics(r, b):
    net = r.net
    bb = b.reindex(net.index).fillna(0.0)
    n = len(net)
    navp = (1 + net).cumprod()
    navb = (1 + bb).cumprod()
    ann = float(navp.iloc[-1] ** (252 / n) - 1)
    annb = float(navb.iloc[-1] ** (252 / n) - 1)

    ex_diff = (ann - annb) * 100                                   # 49 口径
    ex_ratio = float(((navp.iloc[-1] / navb.iloc[-1]) ** (252 / n) - 1) * 100)  # 33 口径

    te = float((net - bb).std() * np.sqrt(252))
    gross = np.nan
    if r.cost_amt is not None:
        cm = r.cost_amt.reindex(net.index).fillna(0.0)
        gross = geom(net + cm) - annb

    return {
        "年化": ann,
        "超额_差值": ex_diff,
        "超额_比值": ex_ratio,
        "毛超额_差值": (gross * 100) if np.isfinite(gross) else np.nan,
        "TE": te,
        "IR_差值": ex_diff / (te * 100) if te > 0 else np.nan,
        "IR_比值": ex_ratio / (te * 100) if te > 0 else np.nan,
        "换手": r.ann_turnover(),
        "最大回撤": float((navp / navp.cummax() - 1).min()),
        "年化成本pp": float(r.cost_amt.mean() * 252 * 100) if r.cost_amt is not None else np.nan,
    }


def expo(holdings, mat):
    v = []
    for d, names in holdings.items():
        if d not in mat.index or not names:
            continue
        cols = [c for c in names if c in mat.columns]
        if not cols:
            continue
        x = mat.loc[d, cols]
        x = x[x.notna()]
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
    b = benchmark(p)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    # ---------- 覆盖率诊断 ----------
    parts = [p.fac[n] for n in FACTORS]
    cnt = sum(d.notna().astype(int) for d in parts)

    m = p.mask.to_numpy()
    vals = cnt.to_numpy()[m]
    vals = vals[np.isfinite(vals)]
    dist = pd.Series(vals).value_counts(normalize=True).sort_index()

    log("\n② 覆盖率（只在 mask=True 的格子上统计）")
    for k, v in dist.items():
        log(f"   有效因子数 {int(k)}/4  {v*100:6.2f}%")
    log("   单因子缺失率：")
    for n_ in FACTORS:
        miss = float(p.fac[n_].where(p.mask).isna().to_numpy().mean())
        log(f"     {n_:<8} {miss*100:6.2f}%")

    # ---------- 两个变体回放 ----------
    log("\n③ 回放（12bp）...")
    res, reps = {}, {}
    for mode in ("len", "valid"):
        sc = smooth_score(composite_variant(p, mode))
        r = replay(sc, p)
        reps[mode] = r
        res[mode] = metrics(r, b)
        log(f"   {mode:<6} 年化 {res[mode]['年化']:+.2%}  "
            f"超额(差值) {res[mode]['超额_差值']:+.2f}pp  "
            f"换手 {res[mode]['换手']:.0%}  TE {res[mode]['TE']:.2%}")

    # ---------- 持仓差异 ----------
    ra, rb = reps["len"], reps["valid"]
    dates = sorted(set(ra.holdings) & set(rb.holdings))
    inter, jac, moved = [], [], []
    for d in dates:
        A, B = set(ra.holdings[d]), set(rb.holdings[d])
        inter.append(len(A & B))
        jac.append(len(A & B) / max(len(A | B), 1))
        moved.append(len(A - B))
    mean_inter = float(np.mean(inter)) if inter else np.nan
    mean_moved = float(np.mean(moved)) if moved else np.nan

    # ---------- 暴露 ----------
    log("\n④ 持仓暴露")
    expo_rows = {}
    for mode, r in reps.items():
        row = {"持仓ln市值": expo(r.holdings, None) if False else np.nan}
        row["ep缺失占比"] = nan_share(r.holdings, p.fac["ep"])
        for n_ in FACTORS:
            row[f"mean_{n_}"] = expo(r.holdings, p.fac[n_])
        expo_rows[mode] = row
        log(f"   {mode:<6} " + "  ".join(
            f"{k}={v:.3f}" for k, v in row.items() if np.isfinite(v)))

    # ---------- 成本 ----------
    cost_rows = {}
    if not SKIP_COST:
        log("\n⑤ 逐股成本（N=50，part=1/50）...")
        daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
        sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
        amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                    .reindex(index=p.idx, columns=p.cols)) * 1000.0
        adv = amt.rolling(20, min_periods=20).mean()

        def cmat(aum, n=N, k=0.7):
            c = k * sig * np.sqrt((aum * (1.0 / n)) / adv.where(adv > 0))
            return (c + 0.0003).clip(upper=CAP)

        for mode in ("len", "valid"):
            sc = smooth_score(composite_variant(p, mode))
            cost_rows[mode] = {}
            for aum, lab in AUMS:
                rr = replay(sc, p, top_n=N, cost=cmat(aum))
                v = metrics(rr, b)["超额_差值"]
                cost_rows[mode][lab] = v
                log(f"   {mode:<6} {lab}  {v:+.2f}pp")

    # ---------- 汇总 ----------
    tab = pd.DataFrame(res).T
    log("\n" + "=" * 100)
    log("【12bp 口径】")
    log("=" * 100)
    show = tab[["年化", "超额_差值", "超额_比值", "毛超额_差值", "TE",
                "IR_差值", "IR_比值", "换手", "最大回撤", "年化成本pp"]].copy()
    for c in ("年化", "TE", "换手", "最大回撤"):
        show[c] = (show[c] * 100).round(2)
    for c in ("超额_差值", "超额_比值", "毛超额_差值", "IR_差值", "IR_比值", "年化成本pp"):
        show[c] = show[c].round(2)
    log(show.to_string())

    d_ann = abs(res["len"]["年化"] - res["valid"]["年化"]) * 100
    d_ex = abs(res["len"]["超额_差值"] - res["valid"]["超额_差值"])
    log(f"\n  年化差 {d_ann:.2f}pp   超额(差值)差 {d_ex:.2f}pp   "
        f"持仓重合 {mean_inter:.1f}/{N}（每期平均换掉 {mean_moved:.1f} 只）")

    if cost_rows:
        log("\n【逐股成本（pp，差值口径）】")
        log(pd.DataFrame(cost_rows).T.round(2).to_string())

    log("\n【暴露差异】")
    log(pd.DataFrame(expo_rows).T.round(3).to_string())
    log(f"  其中 ep缺失占比：A {expo_rows['len']['ep缺失占比']:.1%} → "
        f"B {expo_rows['valid']['ep缺失占比']:.1%}")

    # ---------- 预注册判据 ----------
    log("\n" + "=" * 100)
    log("【判读（阈值在看到结果前已定）】")
    log("=" * 100)
    if d_ann < 0.2 and mean_inter >= 47:
        verdict = "A 无害：保留 core.composite，只写进 limitations"
    elif d_ann <= 0.5 and mean_inter >= 40:
        verdict = "A 敏感但非致命：保留 A，limitations 里给出敏感性区间"
    else:
        verdict = "⚠ 结论依赖填充约定：改用 B，全文数字（9.65/4.25/4.47/容量）重算"
    log(f"  年化差 {d_ann:.2f}pp，重合 {mean_inter:.1f}/{N}  →  {verdict}")
    log("\n  若改用 B：legacy/ 必须保持 A（冻结），OOS 记录与回测口径要分开写明。")

    out = tab.copy()
    if cost_rows:
        cu = pd.DataFrame(cost_rows).T
        out = out.join(cu, how="left")
    out.to_csv(OUT / "composite_ab.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'composite_ab.csv'}")
    log(f"总耗时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
