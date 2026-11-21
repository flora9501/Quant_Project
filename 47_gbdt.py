#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
47_gbdt.py
非线性模型：GBDT 打分 vs 线性复合，同一套组合构建。

设计（判据已预注册，跑完不许改）
  特征：**只用 4 个已中性化的因子**（行业+市值已回归掉）。
        绝不把 circ_mv / turnover_rate 原始值当特征 —— 那是把 beta 直接喂进去。
  目标：5 日 forward return 的**每日横截面百分位排名**（去掉市场涨跌）。
  验证：walk-forward，逐年滚动（2018–2020 训练 → 2021 预测，… → 2025）。
        只有 2021–2025 有预测值，所以对比窗口就是 2021–2025。
  组合：完全复用 replay + quota_select + Δ=0.25 + 双边 12bp，只换 score 来源。

判据
  1. 净超额 —— 记录，但**不作为判据**
  2. 三个代理指标（真正的 alpha 归因要 08 号，这里用代理）：
       a) GBDT 分数被 4 因子线性解释的 R²（>0.85 → 只是线性组合的换皮）
       b) 两个组合 top-50 的重合度（>0.6 → 选出来的是同一批票）
       c) 持仓的 ln 市值 / 换手率暴露（差异大 → 赢的是 beta）
  3. 换手 > 900% → 模型在追噪声

用法
  .venv/bin/python -u 47_gbdt.py
输出
  data/output_zoo/gbdt_vs_linear.csv
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
from factorlab.gate import forward_returns

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("缺 lightgbm：.venv/bin/pip install lightgbm")

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)

H = 5
START_EVAL = "20210101"
FEATS = list(FACTORS)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def metrics(r, bench, start=START_EVAL):
    net = r.net[r.net.index >= start]
    b = bench.reindex(net.index).fillna(0.0)
    ex = (net - b).dropna()
    return {
        "年化": geom(net),
        "超额pp": (geom(net) - geom(b)) * 100,
        "换手": float(r.turnover.replace(0, np.nan)
                      [r.turnover.index >= start].mean() * (252 / r.rebal)),
        "IR": (float(ex.mean() / ex.std() * np.sqrt(252))
               if ex.std() > 0 else np.nan),
    }


def main():
    t0 = time.time()
    log("① 加载面板与价格...")
    p = load_panel()
    daily, basic = load_price_data()
    cmv = (basic.pivot(index="trade_date", columns="ts_code", values="circ_mv")
                .reindex(index=p.idx, columns=p.cols))
    turn = (basic.pivot(index="trade_date", columns="ts_code", values="turnover_rate")
                 .reindex(index=p.idx, columns=p.cols))
    lnmv = np.log(cmv * 1e4)
    fwd5 = forward_returns(p, H)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    log("\n② 构造训练表（特征=4 因子，目标=横截面百分位）...")
    y_rank = fwd5.rank(axis=1, pct=True)

    def flat(wide, name):
        return wide.where(p.mask).stack().rename(name)

    parts = {n: flat(p.fac[n], n) for n in FEATS}
    parts["y"] = flat(y_rank, "y")
    df = pd.DataFrame(parts).reset_index().rename(
        columns={"trade_date": "date"})
    df = df.dropna()
    df["year"] = df["date"].str[:4]
    log(f"   {len(df):,} 行，年份 {sorted(df['year'].unique())}")

    log("\n③ walk-forward 训练...")
    years = sorted(df["year"].unique())
    preds = []
    for Y in years[3:]:
        tr = df[df["year"] < Y]
        te = df[df["year"] == Y]
        if te.empty or len(tr) < 100000:
            continue
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[FEATS].to_numpy(), tr["y"].to_numpy())
        pr = te[["date", "ts_code"]].copy()
        pr["pred"] = m.predict(te[FEATS].to_numpy())
        preds.append(pr)
        log(f"   训练 {tr['year'].min()}–{int(Y) - 1} → 预测 {Y}  "
            f"({len(pr):,} 行)")

    if not preds:
        log("❌ 训练失败")
        return 1
    pred = pd.concat(preds, ignore_index=True)
    pred_wide = (pred.pivot(index="date", columns="ts_code", values="pred")
                     .reindex(index=p.idx, columns=p.cols))

    log("\n④ 两个组合回放（2021 起可比）...")
    lin_score = smooth_score(composite(p))
    gb_score = smooth_score(pred_wide)
    r_lin = replay(lin_score, p)
    r_gb = replay(gb_score, p)
    b = benchmark(p)

    rows = [{"模型": "线性 4 因子", **metrics(r_lin, b)},
            {"模型": "GBDT", **metrics(r_gb, b)}]
    tab = pd.DataFrame(rows).set_index("模型")
    for c in ("年化", "超额pp", "换手"):
        tab[c] = tab[c].round(4)
    tab["IR"] = tab["IR"].round(2)
    log(tab.to_string())

    log("\n⑤ 代理诊断（这三条才是判据）...")

    # a) GBDT 分数被 4 因子线性解释的比例
    r2s = []
    for d in p.idx:
        if d < START_EVAL or d not in pred_wide.index:
            continue
        m = pd.concat([pred_wide.loc[d].rename("y")] +
                      [p.fac[f].loc[d].rename(f) for f in FEATS],
                      axis=1).dropna()
        if len(m) < 100:
            continue
        X = np.c_[np.ones(len(m)), m[FEATS].to_numpy()]
        y = m["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        r2s.append(r2)
    log(f"   a) GBDT 分数被 4 因子解释的 R²：{np.nanmean(r2s):.3f}"
        f"   （>0.85 → 只是线性组合的换皮）")

    # b) top-50 重合度
    ov = []
    for d in p.idx:
        if d < START_EVAL:
            continue
        a = lin_score.loc[d].where(p.mask.loc[d]).dropna()
        c = gb_score.loc[d].where(p.mask.loc[d]).dropna()
        if len(a) < 50 or len(c) < 50:
            continue
        ov.append(len(set(a.nlargest(50).index) & set(c.nlargest(50).index)) / 50)
    log(f"   b) top-50 平均重合度：{np.mean(ov):.1%}"
        f"   （>60% → 选出来是同一批票）")

    # c) 持仓暴露
    def expo(r, mat):
        vals = []
        for d, names in r.holdings.items():
            if d < START_EVAL or d not in mat.index:
                continue
            cols = [x for x in names if x in mat.columns]
            if cols:
                vals.append(float(mat.loc[d, cols].mean()))
        return float(np.mean(vals)) if vals else np.nan

    log(f"   c) 持仓暴露")
    log(f"        ln市值  线性 {expo(r_lin, lnmv):.2f}   GBDT {expo(r_gb, lnmv):.2f}")
    log(f"        换手率  线性 {expo(r_lin, turn):.2f}   GBDT {expo(r_gb, turn):.2f}")

    log("\n" + "=" * 90)
    log("【判读】")
    log("=" * 90)
    log(f"  线性超额 {tab.loc['线性 4 因子','超额pp']:+.2f}pp  →  "
        f"GBDT {tab.loc['GBDT','超额pp']:+.2f}pp  "
        f"（差 {tab.loc['GBDT','超额pp'] - tab.loc['线性 4 因子','超额pp']:+.2f}pp）")
    log("  R² 高 + 重合度高 → 模型没学到新东西")
    log("  超额变好但暴露变化大 → 赢的是 beta，不是 alpha")
    log("  ⚠️ 真正的 alpha 检验需要 08 号的风格归因；这里只有代理指标")

    tab.to_csv(OUT / "gbdt_vs_linear.csv", encoding="utf-8-sig",
               float_format="%.4f")
    log(f"\n✅ 输出 {OUT / 'gbdt_vs_linear.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
