#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
72_marginal_at_cost.py  (v3)
把 9 个候选因子的「组合增量」从扁平 12bp 换算到真实逐股成本，
并把「死于成本」与「死于稀释」分开。

v3 修改
  ★ 修 pd.to_numeric(errors="ignore") —— pandas 2.2 起已移除，只接受 raise/coerce
  ★ 修双重平滑：基准与增强都从「未平滑 composite」出发，只加一层平滑
  ★ 12bp 口径改用解析式 net12 = gross - 2·turnover·COST（用基准回放自检）
      → 回放次数从 20 次降到 11 次，约 11 分钟
  ★ 打印每个因子的覆盖率；<70% 给出警告
  ★ 新增 Δcost 与「主导」列：Δnet = Δgross − Δcost

为什么不算新实验
  replay 的选股只读分数，成本只进 net = gross − cost_amt；
  持仓与成本档位无关，换成本档是换一把尺子。

口径
  基准 = 4 因子 composite（smooth_score, top_n=50）
  增量 = 新因子按目标 20% 方差权重并入（权重自动校准）
  判据 = Δnet ≤ 0 或 |配对 t| < 2 → 不采纳

用法
  .venv/bin/python -u 72_marginal_at_cost.py                 # 只算 1 亿，约 11 分钟
  .venv/bin/python -u 72_marginal_at_cost.py --aums 0.5,1,2  # 三档，约 30 分钟
输出
  data/output_zoo/marginal_at_cost.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, TOP_N, COST)
from factorlab.factors import load_price_data, load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

CAP = 0.02
N = TOP_N
TARGET_W = 0.2
ARGV = sys.argv
AUMS = ([float(x) for x in ARGV[ARGV.index("--aums") + 1].split(",")]
        if "--aums" in ARGV else [1.0])

FACTORS_EXT = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn",
               "q_ag", "ovn_mom", "idr_rev", "margin_chg"]

# 文档里的原始死因（FACTOR_ZOO §3 / §4）
REASON_DOC = {
    "rev_20": "成本（换手 714%→1543%）",
    "amihud": "残差 IC −0.0064；增量来自降换手，非 alpha",
    "max_ret": "成本为主（换手→972%）",
    "high_52w": "h=5 完全失效（频率问题，非成本）",
    "turn_abn": "权重叠加导致选股恶化",
    "q_ag": "IC 0.0030 伪信号（非成本）",
    "ovn_mom": "方向事后修正 + 增量 0",
    "idr_rev": "成本（换手↑）",
    "margin_chg": "正交但边际 t 仅 +0.39（无增量）",
}

# FACTOR_ZOO §3.1 的「20% Δ」（在实为 14.3% 权重下算出，仅供对照）
DOC_DELTA = {
    "rev_20": -1.21, "max_ret": -1.80, "turn_abn": -2.31,
    "amihud": 2.34, "q_ag": 1.30,
}


def log(m):
    print(m, flush=True)


def geom(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1)


def build_cost_inputs(p, daily):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    return sig, adv


def cost_from(sig, adv, aum, n=N, k=0.7, floor=0.0003, cap=CAP):
    part = 1.0 / n
    c = k * sig * np.sqrt((aum * part) / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def paired_t(a, b):
    d = (a - b).dropna()
    if len(d) < 10 or d.std() == 0:
        return np.nan
    return float(d.mean() / d.std() * np.sqrt(len(d)))


def calibrate_weight(comp_raw, fac, mask, target=TARGET_W):
    """返回 (并入权重, 0.8/0.2 下的实际方差占比)"""
    vc = float(np.nanvar(comp_raw.where(mask).to_numpy()))
    vf = float(np.nanvar(fac.where(mask).to_numpy()))
    if not (np.isfinite(vc) and np.isfinite(vf)) or vf <= 0:
        return 0.2, np.nan
    rr = np.sqrt(target / (1 - target)) * np.sqrt(vc / vf)
    w_f = rr / (1 + rr)
    naive = (0.2 ** 2 * vf) / (0.8 ** 2 * vc + 0.2 ** 2 * vf)
    return float(w_f), float(naive)


def main():
    t0 = time.time()
    log("① 加载...")
    p = load_panel()
    daily, _ = load_price_data(daily_cols=("pct_chg", "amount"))
    sig, adv = build_cost_inputs(p, daily)
    b = benchmark(p)

    comp_raw = composite(p)                      # ★ 未平滑
    comp_sm = smooth_score(comp_raw)             # ★ 只平滑一次
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只；N={N}；AUM={AUMS} 亿")
    log(f"   composite 截面 std = {np.nanstd(comp_raw.where(p.mask).to_numpy()):.3f}"
        f"（≠0.5 → 0.8/0.2 拿不到 20%）")

    # ---------- 基准 ----------
    log("\n② 基准回放（4 因子）...")
    rb12 = replay(comp_sm, p, top_n=N)
    g_base = geom(rb12.gross.reindex(rb12.net.index))

    # ★ 自检：net12 的解析式
    derived12 = (rb12.gross - 2 * rb12.turnover * COST).iloc[:-1]
    gap = float((derived12 - rb12.net).abs().max())
    log(f"   自检 net12 = gross − 2·turnover·COST：max|gap| = {gap:.2e}")
    if not (gap < 1e-10):
        raise SystemExit("net12 解析式不成立，改用逐次 12bp 回放")

    base = {"12bp": rb12}
    for aum in AUMS:
        key = f"{aum:g}亿"
        base[key] = replay(comp_sm, p, top_n=N, cost=cost_from(sig, adv, aum * 1e8))
        log(f"   {key}  换手 {base[key].ann_turnover():.0%}  "
            f"年化 {geom(base[key].net)*100:+.2f}%")

    # ---------- 逐因子 ----------
    rows = []
    for name in FACTORS_EXT:
        f = load_cached(name)
        if f is None:
            log(f"\n   缺 {name}，跳过")
            continue
        f = f.reindex(index=p.idx, columns=p.cols)
        cov = float((p.mask & f.notna()).to_numpy().sum() / p.mask.to_numpy().sum())
        w_f, naive = calibrate_weight(comp_raw, f, p.mask)
        aug = smooth_score(comp_raw * (1 - w_f) + f.fillna(0.0) * w_f)   # ★ 单层平滑

        log(f"\n③ {name}  (w_f={w_f:.3f}；0.8/0.2 实际只有 {naive:.1%})")
        if cov < 0.7:
            log(f"   ⚠ 覆盖 {cov:.1%}，边际只在部分股票上生效")

        row = {"factor": name, "文档死因": REASON_DOC.get(name, ""),
               "覆盖%": cov * 100,
               "w_f": round(w_f, 3), "0.8/0.2实际": round(naive, 3),
               "文档Δ": DOC_DELTA.get(name, np.nan)}

        reps = {}
        for aum in AUMS:
            key = f"{aum:g}亿"
            reps[key] = replay(aug, p, top_n=N, cost=cost_from(sig, adv, aum * 1e8))
        r0 = reps[list(reps)[0]]

        g_aug = geom(r0.gross.reindex(r0.net.index))
        row["Δgross_pp"] = (g_aug - g_base) * 100
        net12_aug = (r0.gross - 2 * r0.turnover * COST).iloc[:-1]        # ★ 解析式
        row["Δnet@12bp"] = (geom(net12_aug) - geom(base["12bp"].net)) * 100
        row["换手_base"] = rb12.ann_turnover() * 100
        row["换手_aug"] = r0.ann_turnover() * 100
        row["Δ换手_pp"] = row["换手_aug"] - row["换手_base"]
        log(f"   12bp  Δgross {row['Δgross_pp']:+.2f}  "
            f"Δnet {row['Δnet@12bp']:+.2f}  换手 "
            f"{row['换手_base']:.0f}%→{row['换手_aug']:.0f}%")

        for aum in AUMS:
            key = f"{aum:g}亿"
            ra = reps[key]
            dn = (geom(ra.net) - geom(base[key].net)) * 100
            row[f"Δnet@{key}"] = dn
            row[f"t@{key}"] = paired_t(ra.net, base[key].net)
            row[f"Δcost@{key}"] = row["Δgross_pp"] - dn
            row[f"主导@{key}"] = ("成本" if abs(row[f"Δcost@{key}"])
                                  > abs(row["Δgross_pp"]) else "信号")
            log(f"   {key}  Δnet {dn:+.2f}  t {row[f't@{key}']:+.2f}  "
                f"Δcost {row[f'Δcost@{key}']:+.2f}  → {row[f'主导@{key}']}主导")

        rows.append(row)

    tab = pd.DataFrame(rows).set_index("factor")

    # ---------- 汇总（★ 逐列数值转换，不用 errors="ignore"） ----------
    log("\n" + "=" * 132)
    log("【结论表】Δnet = Δgross − Δcost；主导 = 谁贡献了更大的负值")
    log("=" * 132)
    cols = ["文档死因", "覆盖%", "Δgross_pp", "Δnet@12bp", "文档Δ",
            "w_f", "0.8/0.2实际", "换手_base", "换手_aug", "Δ换手_pp"]
    for aum in AUMS:
        k = f"{aum:g}亿"
        cols += [f"Δnet@{k}", f"Δcost@{k}", f"t@{k}", f"主导@{k}"]

    show = tab[[c for c in cols if c in tab.columns]].copy()
    str_cols = {"文档死因"} | {c for c in show.columns if c.startswith("主导")}
    for c in show.columns:
        if c in str_cols:
            continue
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    log(show.to_string())

    # ---------- 判读 ----------
    log("\n【判读】")
    for aum in AUMS:
        k = f"{aum:g}亿"
        n_cost = int((tab[f"主导@{k}"] == "成本").sum())
        n_sig = int((tab[f"主导@{k}"] == "信号").sum())
        still_pos = tab[tab[f"Δnet@{k}"] > 0].index.tolist()
        log(f"  {k}：成本主导 {n_cost} 个 / 信号主导 {n_sig} 个；"
            f"仍为正 {still_pos if still_pos else '无'}")

    log("\n  读法：")
    log("    Δgross < 0            → 加进去毛利就变差（稀释），与成本无关")
    log("    Δgross > 0 且 Δnet < 0 → 真的被成本杀了")
    log("  口径警告：本表 Δnet 与 FACTOR_ZOO §3.1 的「文档Δ」不同源——")
    log("  §3.1 实际是在 14.3% 方差权重、成本档未记录的条件下算的，")
    log("  差 0.5–1.2pp 属预期，不要直接引用旧值。")

    tab.to_csv(OUT / "marginal_at_cost.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT/'marginal_at_cost.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
