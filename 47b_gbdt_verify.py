#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
47b_gbdt_verify.py
GBDT 结果的三步验证。每一步都能单独杀死它。

  ① 成本敏感性：12bp / 0.5亿 / 1亿 / 2亿（逐股冲击成本）
       —— 41 号已证明 12bp 低估 2–4 倍，GBDT 换手翻倍，最可能死在这里
  ② 按年拆：2021–2025 每年的超额，看是不是集中在某一两年
  ③ OLS 对照：同特征、同目标、同 walk-forward，换成线性回归
       —— 若 OLS ≈ GBDT，增量来自"重新估权重"而非非线性

判据（预注册）
  ① 1 亿下 GBDT 仍领先线性 >1pp → 继续；否则关闭
  ② 若 +1.30pp 全部来自单一年份 → 噪声
  ③ OLS ≈ GBDT（差 <0.5pp）→ 非线性没起作用

用法
  .venv/bin/python -u 47b_gbdt_verify.py [--rebuild]
输出
  data/output_zoo/gbdt_verify.csv
  data/output_zoo/gbdt_pred.parquet（预测缓存，重跑秒出）
"""
import argparse
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
    raise SystemExit("缺 lightgbm")

OUT = Path("data/output_zoo")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "gbdt_pred.parquet"

H = 5
START_EVAL = "20210101"
FEATS = list(FACTORS)
TOP_N = 50
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def log(m):
    print(m, flush=True)


def geom(s):
    s = s.dropna()
    return float((1 + s).cumprod().iloc[-1] ** (252 / len(s)) - 1) if len(s) else np.nan


def cost_matrix(p, daily, aum, k=0.7, floor=0.0003, part=1.0 / TOP_N,
                cap=0.02):
    sig = p.ret.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=20).std()
    amt = (daily.pivot(index="trade_date", columns="ts_code", values="amount")
                .reindex(index=p.idx, columns=p.cols)) * 1000.0
    adv = amt.rolling(20, min_periods=20).mean()
    q = aum * part
    c = k * sig * np.sqrt(q / adv.where(adv > 0))
    return (c + floor).clip(upper=cap)


def excess_pp(r, bench, start=START_EVAL):
    net = r.net[r.net.index >= start]
    b = bench.reindex(net.index).fillna(0.0)
    return (geom(net) - geom(b)) * 100


def train_preds(p, df, rebuild=False):
    """walk-forward，同时产出 GBDT 与 OLS 的预测"""
    if CACHE.exists() and not rebuild:
        log("   命中预测缓存")
        return pd.read_parquet(CACHE)

    years = sorted(df["year"].unique())
    outs = []
    for Y in years[3:]:
        tr = df[df["year"] < Y]
        te = df[df["year"] == Y]
        if te.empty or len(tr) < 100000:
            continue

        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[FEATS].to_numpy(), tr["y"].to_numpy())
        pg = m.predict(te[FEATS].to_numpy())

        X = np.c_[np.ones(len(tr)), tr[FEATS].to_numpy()]
        beta, *_ = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)
        Xt = np.c_[np.ones(len(te)), te[FEATS].to_numpy()]
        po = Xt @ beta

        o = te[["date", "ts_code"]].copy()
        o["pred_gb"] = pg
        o["pred_ols"] = po
        outs.append(o)
        log(f"   训练 {tr['year'].min()}–{int(Y) - 1} → {Y}  ({len(o):,} 行)")

    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    log("① 加载...")
    p = load_panel()
    daily, basic = load_price_data(daily_cols=("pct_chg", "amount"))
    fwd5 = forward_returns(p, H)
    log(f"   {len(p.idx)} 天 × {len(p.cols)} 只")

    log("\n② 训练表 + walk-forward（GBDT 与 OLS）...")
    y_rank = fwd5.rank(axis=1, pct=True)

    def flat(w, n):
        return w.where(p.mask).stack().rename(n)

    parts = {n: flat(p.fac[n], n) for n in FEATS}
    parts["y"] = flat(y_rank, "y")
    df = (pd.DataFrame(parts).reset_index()
            .rename(columns={"trade_date": "date"}).dropna())
    df["year"] = df["date"].str[:4]
    pred = train_preds(p, df, a.rebuild)

    def wide(col):
        return (pred.pivot(index="date", columns="ts_code", values=col)
                    .reindex(index=p.idx, columns=p.cols))

    log("\n③ 三个 score + 回放...")
    s_lin = smooth_score(composite(p))          # 等权线性（现行）
    s_gb = smooth_score(wide("pred_gb"))
    s_ols = smooth_score(wide("pred_ols"))
    b = benchmark(p)

    r_lin = replay(s_lin, p)
    r_gb = replay(s_gb, p)
    r_ols = replay(s_ols, p)
    log(f"   线性 {excess_pp(r_lin, b):+.2f}pp   "
        f"OLS {excess_pp(r_ols, b):+.2f}pp   "
        f"GBDT {excess_pp(r_gb, b):+.2f}pp")

    # ---------- ① 成本敏感性 ----------
    log("\n④ 检验 1：成本敏感性（逐股冲击成本）...")
    rows = []
    for label, r in (("线性 4 因子", r_lin), ("GBDT", r_gb)):
        row = {"模型": label, "12bp": round(excess_pp(r, b), 2)}
        for aum in (0.5e8, 1e8, 2e8):
            cm = cost_matrix(p, daily, aum)
            rr = replay(s_lin if label.startswith("线性") else s_gb, p, cost=cm)
            row[f"{aum/1e8:.1f}亿"] = round(excess_pp(rr, b), 2)
        rows.append(row)
    tab_cost = pd.DataFrame(rows).set_index("模型")
    log(tab_cost.to_string())
    gap_1e = float(tab_cost.loc["GBDT", "1.0亿"] - tab_cost.loc["线性 4 因子", "1.0亿"])
    log(f"   → 1 亿下 GBDT − 线性 = {gap_1e:+.2f}pp")

    # ---------- ② 按年 ----------
    log("\n⑤ 检验 2：按年超额...")
    yrs = []
    for y in ("2021", "2022", "2023", "2024", "2025"):
        row = {"年": y}
        for label, r in (("线性", r_lin), ("GBDT", r_gb)):
            net = r.net[r.net.index.str[:4] == y]
            bb = b.reindex(net.index).fillna(0.0)
            row[label] = round((geom(net) - geom(bb)) * 100, 2)
        row["差"] = round(row["GBDT"] - row["线性"], 2)
        yrs.append(row)
    tab_year = pd.DataFrame(yrs).set_index("年")
    log(tab_year.to_string())
    pos_years = int((tab_year["差"] > 0).sum())

    # ---------- ③ OLS 对照 ----------
    log("\n⑥ 检验 3：OLS 对照（同特征/同目标/同 walk-forward）...")
    tab3 = pd.DataFrame([
        {"模型": "线性等权（现行）", "超额pp": round(excess_pp(r_lin, b), 2),
         "换手": round(r_lin.ann_turnover(), 4)},
        {"模型": "OLS（重估权重）", "超额pp": round(excess_pp(r_ols, b), 2),
         "换手": round(r_ols.ann_turnover(), 4)},
        {"模型": "GBDT（非线性）", "超额pp": round(excess_pp(r_gb, b), 2),
         "换手": round(r_gb.ann_turnover(), 4)},
    ]).set_index("模型")
    log(tab3.to_string())
    d_ols_gb = float(tab3.loc["GBDT（非线性）", "超额pp"]
                     - tab3.loc["OLS（重估权重）", "超额pp"])

    # ---------- 汇总 ----------
    log("\n" + "=" * 92)
    log("【判读】")
    log("=" * 92)
    log(f"  ① 1 亿 AUM 下 GBDT − 线性 = {gap_1e:+.2f}pp"
        f"   （>+1pp 才继续，否则关闭）")
    log(f"  ② 2021–2025 中 GBDT 领先的年份：{pos_years}/5"
        f"   （<3 年 → 集中在少数年份，噪声）")
    log(f"  ③ GBDT − OLS = {d_ols_gb:+.2f}pp"
        f"   （<0.5pp → 非线性没起作用）")

    verdict = ("继续：跑 08 号做真正的 alpha 归因"
               if (gap_1e > 1.0 and pos_years >= 3 and d_ols_gb >= 0.5)
               else "关闭：GBDT 的增量不可靠")
    log(f"\n  → {verdict}")
    log("\n  ⚠️ 无论结论如何，不要动 legacy/ 和 paper/。")

    pd.concat({
        "cost": tab_cost, "year": tab_year, "models": tab3,
    }, axis=1).to_csv(OUT / "gbdt_verify.csv", encoding="utf-8-sig")
    log(f"\n✅ 输出 {OUT / 'gbdt_verify.csv'}")
    log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
