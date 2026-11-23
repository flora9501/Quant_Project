#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
75_risk_model.py
CNE5 近似风险模型：构建暴露、估计因子收益、诊断因子的"风格含量"。

用法
  .venv/bin/python -u 75_risk_model.py --build                # 构建并落盘，约 5–8 分钟
  .venv/bin/python -u 75_risk_model.py --factor rev_20        # 单因子风格 R²/载荷
  .venv/bin/python -u 75_risk_model.py --all-factors          # 9 个因子一起
  .venv/bin/python -u 75_risk_model.py --portfolio            # 当前 50 名的 TE 分解
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlab.core import (load_panel, composite, smooth_score, replay,
                            benchmark, FACTORS, TOP_N)
from factorlab.factors import load_price_data, load_cached
from factorlab.risk import (build_styles, build_exposures, estimate,
                            ewma_cov, specific_risk, factor_style_diagnose,
                            te_decompose, STYLES)

OUT = Path("data/output_zoo")
NAMES = ["rev_20", "amihud", "max_ret", "high_52w", "turn_abn",
         "q_ag", "ovn_mom", "idr_rev", "margin_chg"]


def log(m):
    print(m, flush=True)


def load_or_build(rebuild=False):
    fx, fi, fs = OUT / "risk_exposures.parquet", OUT / "risk_find.parquet", OUT / "risk_fstyle.parquet"
    if not rebuild and fx.exists() and fi.exists() and fs.exists():
        wide = pd.read_parquet(fx)
        f_ind = pd.read_parquet(fi)
        f_style = pd.read_parquet(fs)
        use = [c for c in f_style.columns]
        return wide, use, f_style, f_ind

    log("① 构建风格...")
    p = load_panel(with_index=True)
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount", "circ_mv",
                                               "total_mv", "turnover_rate",
                                               "pb", "pe_ttm"))
    styles = build_styles(p, daily_basic=basic)

    log("② 构建暴露...")
    wide, use = build_exposures(p, styles)

    log("③ 估计因子收益...")
    f_style, f_ind = estimate(p, wide, use)

    wide.reset_index().to_parquet(fx, index=False)
    f_ind.to_parquet(fi)
    f_style.to_parquet(fs)
    log(f"   ✅ 落盘 {fx.name} / {fi.name} / {fs.name}")
    return wide, use, f_style, f_ind


def main():
    argv = sys.argv
    t0 = time.time()
    rebuild = "--build" in argv
    wide, use, f_style, f_ind = load_or_build(rebuild)

    if "--portfolio" in argv:
        log("\n④ 组合 TE 分解（N=50，12bp）")
        p = load_panel()
        sc = smooth_score(composite(p))
        r = replay(sc, p, top_n=TOP_N)
        last = r.net.index[-1]
        # 用最后一个调仓日的持仓
        hd = sorted(r.holdings)[-1]
        names = r.holdings[hd]
        w_p = pd.Series(1.0 / len(names), index=names)
        w_b = pd.Series(1.0 / int(p.mask.loc[hd].sum()), index=p.mask.loc[hd][p.mask.loc[hd]].index)

        allf = pd.concat([f_style, f_ind], axis=1).dropna()
        F = ewma_cov(allf)
        D = specific_risk({}, [])   # 占位：见下方说明
        log("   注意：D 需要逐股特异波动，本脚本先只报因子部分")

    if "--factor" in argv or "--all-factors" in argv:
        names = (NAMES if "--all-factors" in argv
                 else [argv[argv.index("--factor") + 1]])
        p = load_panel(with_index=True)


        _cp = (1 + p.ret.fillna(0.0)).cumprod()
        fwd5 = (_cp.shift(-5) / _cp - 1).reindex(p.idx)    
        
        rows = []
        for nm in names:
            f = load_cached(nm)
            if f is None:
                log(f"   缺 {nm}")
                continue
            f = f.reindex(index=p.idx, columns=p.cols)
            d = factor_style_diagnose(f, wide, use)
            # 残差 IC（风格中性化后）
            S = d["resid"].where(p.mask).rank(axis=1).to_numpy()
            Fm = fwd5.where(p.mask).rank(axis=1).to_numpy()
            from factorlab.evaluate import _row_corr
            ric = pd.Series(_row_corr(S, Fm), index=p.idx).dropna()
            rows.append({"factor": nm, "R2_style": round(d["r2_mean"], 3),
                         "resid_IC": round(float(ric.mean()), 4),
                         "top_style": list(d["beta_abs_rank"].items())[:3]})
            log(f"\n   {nm}: 风格 R² = {d['r2_mean']:.3f}  "
                f"残差 IC = {ric.mean():+.4f}")
            log(f"      载荷 |β| 前三：{list(d['beta_abs_rank'].items())[:3]}")

        tab = pd.DataFrame(rows).set_index("factor").sort_values("R2_style")
        log("\n" + "=" * 96)
        log("【风格含量排行】（R² 高 = 主要是风格；残差 IC 才是新信息）")
        log("=" * 96)
        log(tab.to_string())
        tab.to_csv(OUT / "risk_factor_style.csv", encoding="utf-8-sig")

    log(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
