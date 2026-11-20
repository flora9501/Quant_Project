#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
27_impact_cost.py
用平方根律给两个配置重新定价，输出 AUM 容量曲线。

冲击模型（每只股票、每次调仓）
    rate_i = k * σ_i * sqrt(Q_i / ADV_i)
    Q_i    = AUM * |Δw_i|             单边成交金额
    σ_i    = 过去 20 日收益标准差
    ADV_i  = 过去 20 日平均成交额
    k      = 0.7（A 股常用的经验值，不做拟合）

总成本 = 固定价差（3bp/边）+ 冲击，全部以组合收益的形式计入。

对比
    Baseline（Stratified，换手 2514%）
    A+B 配额约束（THRESH=0.25，换手 714%）

输出
    data/output_turnover/capacity.csv
    终端打印：AUM × 单边成本 × 净超额，以及超额归零的临界 AUM
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

RAW = DATA_DIR / "raw"
REF = DATA_DIR / "ref"
PANEL = DATA_DIR / "panel"
OUT = DATA_DIR / "output_turnover"
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = ["rev_5", "turn_20", "vol_20", "ep"]
TOP_N = 50
REBAL = 5
THRESH = 0.25
K_IMPACT = 0.7
SPREAD_BP = 3.0
AUM_LIST = [1e8, 3e8, 5e8, 1e9, 2e9, 5e9]


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg", "amount"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0).sort_index()
amt = daily.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()  # 千元
del daily
fwd = ret.shift(-1)

# 交易当日的成本用 T-1 及之前的统计量（避免前视）
sigma = ret.rolling(20, min_periods=10).std().shift(1)
adv = (amt.rolling(20, min_periods=10).mean() * 1000).shift(1)   # 元

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}
idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns
for m in [sigma, adv, fwd]:
    m.reindex(index=idx, columns=cols)

sigma = sigma.reindex(index=idx, columns=cols)
adv = adv.reindex(index=idx, columns=cols)
fwd = fwd.reindex(index=idx, columns=cols)
for n in FACTORS:
    wide[n] = wide[n].reindex(index=idx, columns=cols)

mask = (pd.read_parquet(PANEL / "tradable.parquet")
        .reindex(index=idx, columns=cols).fillna(False).astype(bool))

sb = pd.concat([pd.read_parquet(REF / "stock_basic.parquet"),
                pd.read_parquet(REF / "stock_delisted.parquet")],
               ignore_index=True).drop_duplicates("ts_code")
industry = sb.set_index("ts_code")["industry"].reindex(cols).fillna("UNKNOWN")
log(f"   {len(idx)} 天 × {len(cols)} 只")


# ==================== 2. 分数 ====================
def combine(dfs):
    return sum(d.fillna(0.0) for d in dfs).where(
        sum(d.notna().astype(int) for d in dfs) > 0)


score_raw = combine([wide[n] for n in FACTORS])

log("② 平滑分数...")
acc = score_raw.fillna(0.0).copy()
cnt = score_raw.notna().astype(np.int16).copy()
for k in (REBAL, 2 * REBAL):
    sh = score_raw.shift(k)
    acc = acc + sh.fillna(0.0)
    cnt = cnt + sh.notna().astype(np.int16)
    del sh
score_smooth = acc / cnt.replace(0, np.nan)
del acc, cnt


# ==================== 3. 选股 ====================
def industry_quota(df, top_n=TOP_N):
    share = df.groupby("ind").size() / len(df)
    raw = share * top_n
    base = np.floor(raw).astype(int)
    remain = top_n - int(base.sum())
    if remain > 0:
        for key in list((raw - base).sort_values(ascending=False).index)[:remain]:
            base.loc[key] += 1
    return base


def select_stratified(s):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)
    base = industry_quota(df)
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        picks.extend(df[df["ind"] == ind_].nlargest(int(k), "score").index.tolist())
    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


def select_quota_hysteresis(s, held, sigma_mult=THRESH):
    df = pd.DataFrame({"score": s, "ind": industry.reindex(s.index)}).dropna()
    if df.empty:
        return []
    if len(df) <= TOP_N:
        return list(df.nlargest(TOP_N, "score").index)

    base = industry_quota(df)
    delta = sigma_mult * float(df["score"].std())
    picks = []
    for ind_, k in base.items():
        if k <= 0:
            continue
        pool = df[df["ind"] == ind_].sort_values("score", ascending=False)
        if pool.empty:
            continue
        cut = float(pool["score"].iloc[min(int(k), len(pool)) - 1])
        floor = cut - delta
        held_in = [c for c in held if c in pool.index]
        keep = [c for c in held_in if float(pool.loc[c, "score"]) >= floor]
        keep = sorted(keep, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(keep)][:int(k) - len(keep)]
        picks.extend(keep + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 4. 回放，记录每期交易向量 ====================
def run(score_df, use_hyst):
    w_live = pd.Series(0.0, index=cols)
    held = []
    gross_s = pd.Series(0.0, index=idx)
    flat_to = pd.Series(0.0, index=idx)
    trades = {}

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                picks = (select_quota_hysteresis(s, held) if use_hyst
                         else select_stratified(s))
                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                delta = w_new - w_drift
                flat_to.loc[d] = float(delta.abs().sum() / 2)

                nz = delta[delta.abs() > 1e-6]
                trades[d] = nz

                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return gross_s, flat_to, trades


log("\n③ 回放 Baseline...")
gross_b, to_b, trades_b = run(score_raw, False)
log("④ 回放 A+B 配额约束...")
gross_a, to_a, trades_a = run(score_smooth, True)

bench = fwd.where(mask).mean(axis=1).reindex(idx).fillna(0.0)
bench_nav = (1 + bench).cumprod()


# ==================== 5. 冲击成本 ====================
def realized_cost(trades, aum):
    """返回每期的组合层面成本（收益率形式）。"""
    cost = pd.Series(0.0, index=idx)
    miss = 0
    for d, delta in trades.items():
        if delta.empty:
            continue
        codes = delta.index
        q = aum * delta.abs()                       # 单边成交金额（元）
        sg = sigma.loc[d].reindex(codes)
        ad = adv.loc[d].reindex(codes)

        ok = q.notna() & sg.notna() & ad.notna() & (ad > 0)
        miss += int((~ok).sum())

        rate = pd.Series(0.0, index=codes)
        rate[ok] = K_IMPACT * sg[ok] * np.sqrt(q[ok] / ad[ok])
        rate = rate + SPREAD_BP / 1e4

        cost.loc[d] = float((rate * delta.abs()).sum())
    return cost, miss


def evaluate(gross, to_flat, trades, aum):
    cost, miss = realized_cost(trades, aum)
    # 换手为 0 的日子没有成本；模型里的成本已经包含价差+冲击
    net = (gross - cost).iloc[:-1]
    nav = (1 + net).cumprod()
    ann = nav.iloc[-1] ** (252 / len(net)) - 1
    ex_nav = nav / bench_nav.reindex(nav.index).ffill()
    ex_ann = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
    return {"net": ann * 100, "ex": ex_ann * 100,
            "cost_bp_yr": cost.mean() * 252 * 1e4,
            "flat_turnover": to_flat.replace(0, np.nan).mean() * (252 / REBAL) * 100,
            "miss": miss}


log("\n⑤ 计算 AUM 容量曲线...")
rows = []
for aum in AUM_LIST:
    rb = evaluate(gross_b, to_b, trades_b, aum)
    ra = evaluate(gross_a, to_a, trades_a, aum)
    rows.append({
        "AUM(亿)": aum / 1e8,
        "Baseline_net": rb["net"], "Baseline_ex": rb["ex"],
        "AB_net": ra["net"], "AB_ex": ra["ex"],
        "AB_年化成本bp": ra["cost_bp_yr"],
    })
    log(f"   {aum/1e8:>4.0f} 亿完成")

tab = pd.DataFrame(rows).set_index("AUM(亿)")


# ==================== 6. 汇总 ====================
log("\n" + "=" * 92)
log("【AUM 容量曲线】（成本 = 3bp 价差 + 平方根律冲击，k=0.7）")
log("=" * 92)
log(tab.round(2).to_string())

log("\n【相对等权基准的超额】")
log(f"   基准年化 5.18%")
for aum, r in tab.iterrows():
    flag_b = "✅" if r["Baseline_ex"] > 0 else "❌"
    flag_a = "✅" if r["AB_ex"] > 0 else "❌"
    log(f"   {aum:>4.0f} 亿   Baseline {r['Baseline_ex']:+6.2f}% {flag_b}   "
        f"A+B {r['AB_ex']:+6.2f}% {flag_a}")

# 线性插值找归零点
def zero_point(x, y):
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return x[i] + t * (x[i + 1] - x[i])
    return np.nan

xs = tab.index.to_numpy()
zb = zero_point(xs, tab["Baseline_ex"].to_numpy())
za = zero_point(xs, tab["AB_ex"].to_numpy())

log("\n【容量上限（超额归零的 AUM）】")
log(f"   Baseline : {zb:.1f} 亿" if np.isfinite(zb) else "   Baseline : 在测试区间内未归零")
log(f"   A+B      : {za:.1f} 亿" if np.isfinite(za) else "   A+B      : 在测试区间内未归零")

log("\n【未匹配到 σ/ADV 的交易笔数】")
log(f"   Baseline {evaluate(gross_b, to_b, trades_b, 1e8)['miss']} 笔，"
    f"A+B {evaluate(gross_a, to_a, trades_a, 1e8)['miss']} 笔")
log("   （这些笔按 0 成本计，会略微低估真实成本）")

tab.to_csv(OUT / "capacity.csv", encoding="utf-8-sig")
log(f"\n✅ 输出到 {OUT}/capacity.csv")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
