#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
74_evaluate.py  (v6)
统一的因子评估台 CLI。

v6 修改
  ★ --bh 改用 resid_t 的单边 p，并列打印 t_used / p / q_bh（科学计数法）
  ★ 排行榜的「可并入 / 可用」在未跑 deep 时显示「—」

用法
  .venv/bin/python -u 74_evaluate.py --all-cache --no-deep    # 约 6 分钟
  .venv/bin/python -u 74_evaluate.py --all-cache              # auto，约 20 分钟
  .venv/bin/python -u 74_evaluate.py --factors margin_chg --deep
  .venv/bin/python -u 74_evaluate.py --bh
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.evaluate import (build_context, evaluate, format_report,
                                bh_from_log, TRIALS)
from factorlab.factors import load_cached

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn",
         "q_ag", "ovn_mom", "idr_rev", "margin_chg"]


def _tri(v):
    return "—" if v is None else ("✅" if v else "❌")


def main():
    np.seterr(divide="ignore", invalid="ignore")
    argv = sys.argv

    # ---------- BH ----------
    if "--bh" in argv:
        df = bh_from_log()
        if df is None:
            print("没有 trial log，先跑一次评估")
            return 1
        out = df[["name", "resid_ic", "resid_t", "t_used", "p", "q_bh",
                  "n_trials", "pass_bh"]].sort_values("q_bh").copy()
        out["p"] = out["p"].map(lambda x: f"{x:.1e}")
        out["q_bh"] = out["q_bh"].map(lambda x: f"{x:.1e}")
        print("=" * 86)
        print("【BH-FDR】单边检验 H1: 残差 IC > 0；分母 = 去重后的试验数")
        print("=" * 86)
        print(out.to_string(index=False))
        print("\n  t_used 即 resid_t（残差 IC 的 NW t），不是原始 IC 的 t")
        return 0

    # ---------- 模式 ----------
    if "--no-deep" in argv:
        deep = False
    elif "--deep" in argv:
        deep = True
    else:
        deep = "auto"

    names = (argv[argv.index("--factors") + 1].split(",")
             if "--factors" in argv else NAMES)
    aums = ([float(x) for x in argv[argv.index("--aums") + 1].split(",")]
            if "--aums" in argv else (1.0,))

    t0 = time.time()
    print(f"① 构建上下文...  (deep={deep})")
    ctx = build_context()
    print(f"   {len(ctx.panel.idx)} 天 × {len(ctx.panel.cols)} 只")
    if deep is True:
        print("   预热成本矩阵（读 daily，约 20–40 秒）...")
        ctx.ensure_cost()

    results = []
    print("\n② 逐个评估")
    for nm in names:
        f = load_cached(nm)
        if f is None:
            print(f"   缺 {nm}，跳过")
            continue
        f = f.reindex(index=ctx.panel.idx, columns=ctx.panel.cols)
        t1 = time.time()
        r = evaluate(ctx, f, name=nm, deep=deep, aums=aums)
        results.append(r)
        print(format_report(r))
        print(f"  （{time.time() - t1:.1f}s）\n")

    if not results:
        print("没有可评估的因子")
        return 1

    # ---------- 覆盖率自检 ----------
    covs = [r["coverage"] for r in results]
    bars = "=" * 132
    print(bars)
    print(f"【覆盖率自检】min {min(covs):.1%}  max {max(covs):.1%}  "
          f"median {np.median(covs):.1%}")
    if max(covs) < 0.20:
        print("   ⚠ 全部 <20% → coverage 分母补丁没生效"
              "（应为 valid[m].sum()/m.sum()）")
    print(bars)

    # ---------- 排行榜 ----------
    rows = [{
        "factor": r["name"],
        "覆盖": r["coverage"],
        "起始年": r["first_effective_year"],
        "覆盖趋势": r["coverage_trend"],
        "低覆盖年": ",".join(r["low_cov_years"]) or "-",
        "IC": r["ic"][r["h0"]]["ic"],
        "ICIR": r["ic"][r["h0"]]["icir"],
        "t_nw": r["ic"][r["h0"]]["t_nw"],
        "best_h": r["best_h"],
        "残差IC": r["resid_ic"],
        "残差t": r["resid_ic_t"],
        "最大相关": r["corr_max"],
        "单调": r["monotone"],
        "质量门": r["pass_gates"],
        "可并入": _tri(r["delta_ok"]),
        "可用": _tri(r["usable"]),
    } for r in results]

    tab = (pd.DataFrame(rows).set_index("factor")
           .sort_values("IC", key=lambda s: s.abs(), ascending=False))
    print("【排行榜】（按 |IC| 排序；质量门按 h0=5 判）")
    print(tab.round(4).to_string())

    # ---------- 三段结论 ----------
    fail_q = [(r["name"], [k for k, v in r["gate"].items() if not v])
              for r in results if not r["pass_gates"]]
    if fail_q:
        print("\n【质量门未过】")
        for nm, gates in fail_q:
            print(f"   {nm:<11} ❌ {', '.join(gates)}")

    diluted = [r for r in results if r["pass_gates"] and r["delta_ok"] is False]
    if diluted:
        print("\n【质量全过、但等权并入会稀释】")
        for r in diluted:
            print(f"   {r['name']:<11} Δ {r['delta_excess']:+.2f}pp  "
                  f"t {r['delta_t']:+.2f}  → 问题是组合构造，不是因子")

    pending = [r for r in results if r["pass_gates"] and r["delta_ok"] is None]
    if pending:
        print("\n【质量全过、但未测 deep】（待定，不是通过）")
        for r in pending:
            warn = "  ⚠ 覆盖漂移" if r["coverage_drifting"] else ""
            print(f"   {r['name']:<11} 残差 IC {r['resid_ic']:+.5f} "
                  f"(t {r['resid_ic_t']:+.2f}, 依据 {r['resid_ic_via']}){warn}")

    drift = [r for r in results if r["coverage_drifting"]]
    if drift:
        print("\n【覆盖随时间漂移 >0.30】（历史不平稳，IC 可能是 regime 效应）")
        for r in drift:
            print(f"   {r['name']:<11} 趋势 {r['coverage_trend']:+.2f}  "
                  f"起始年 {r['first_effective_year']}")

    tab.to_csv(OUT / "eval_leaderboard.csv", encoding="utf-8-sig")
    print(f"\n✅ 输出 {OUT/'eval_leaderboard.csv'}")
    print(f"   trial log {TRIALS}（BH 的分母，别删）")
    print(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
