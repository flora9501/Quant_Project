#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
28_fix_cost_convention.py
统一成本口径，重算 Phase 1 与 Phase 3 的全部净收益。

问题
  07/25 里 cost = turnover(单边) × c，只对单边收一次费。
  实际卖出 14.2% 要收一次、买入 14.2% 再收一次，应为 Σ|Δw| × c = 2 × 单边 × c。
  换言之，历史数字的成本被少算了一半。

三个口径，逐个叠加，看每个改动贡献多少
  (1) 原 README      固定权重 + 单边收一次
  (2) + 权重漂移      buy-and-hold + 单边收一次
  (3) + 双边计费      buy-and-hold + 每笔成交都收   <- 最终采用

四个配置
  EW Baseline / EW + A+B（无行业约束）/ Strat Baseline / Strat + A+B（配额约束）

四组成本
  flat 12bp（旧口径，仅作对照）
  flat 12bp（新口径）
  flat 6bp / 10bp / 20bp（新口径，敏感性）
  冲击模型 @ 1 / 2 / 3 亿

输出：data/output_turnover/final_numbers.csv
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
AUM_LIST = [1e8, 2e8, 3e8]


def log(m):
    print(m, flush=True)


t0 = time.time()

# ==================== 1. 数据 ====================
log("① 读取数据...")
frames = [pd.read_parquet(f, columns=["ts_code", "trade_date", "pct_chg", "amount"])
          for f in sorted(RAW.glob("daily_[0-9][0-9][0-9][0-9].parquet"))]
daily = pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])

ret = (daily.pivot(index="trade_date", columns="ts_code", values="pct_chg") / 100.0).sort_index()
amt = daily.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
del daily
fwd = ret.shift(-1)

sigma = ret.rolling(20, min_periods=10).std().shift(1)
adv = (amt.rolling(20, min_periods=10).mean() * 1000).shift(1)

fac = pd.read_parquet(PANEL / "factors_neutral.parquet")
wide = {n: fac.pivot(index="trade_date", columns="ts_code", values=n) for n in FACTORS}
idx = ret.index.intersection(wide["rev_5"].index).sort_values()
cols = ret.columns

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


def select_plain(s):
    return list(s.nlargest(TOP_N).index)


def select_plain_hyst(s, held, sigma_mult=THRESH):
    target = select_plain(s)
    if not held:
        return target
    floor = s.reindex(target).min() - sigma_mult * float(s.std())
    keep = [c for c in held if c in s.index and s[c] >= floor][:TOP_N]
    return keep + [c for c in target if c not in set(keep)][:TOP_N - len(keep)]


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


def select_quota_hyst(s, held, sigma_mult=THRESH):
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
        held_in = [c for c in held if c in pool.index]
        keep = [c for c in held_in if float(pool.loc[c, "score"]) >= cut - delta]
        keep = sorted(keep, key=lambda c: -float(pool.loc[c, "score"]))[:int(k)]
        fill = [c for c in pool.index if c not in set(keep)][:int(k) - len(keep)]
        picks.extend(keep + fill)

    picks = list(dict.fromkeys(picks))[:TOP_N]
    if len(picks) < TOP_N:
        rest = df[~df.index.isin(set(picks))].nlargest(TOP_N - len(picks), "score")
        picks += rest.index.tolist()
    return list(dict.fromkeys(picks))[:TOP_N]


# ==================== 4. 回放 ====================
def run(score_df, mode):
    w_live = pd.Series(0.0, index=cols)
    held = []
    gross_s = pd.Series(0.0, index=idx)
    to_s = pd.Series(0.0, index=idx)
    trades = {}

    for i, d in enumerate(idx):
        if i % REBAL == 0:
            valid = mask.loc[d]
            s = score_df.loc[d].where(valid).dropna()
            if len(s) >= TOP_N:
                if mode == "plain":
                    picks = select_plain(s)
                elif mode == "plain_hyst":
                    picks = select_plain_hyst(s, held)
                elif mode == "strat":
                    picks = select_stratified(s)
                else:
                    picks = select_quota_hyst(s, held)

                picks = list(dict.fromkeys(picks))[:TOP_N]
                w_new = pd.Series(1.0 / len(picks), index=picks).reindex(cols).fillna(0.0)

                tot = w_live.sum()
                w_drift = (w_live / tot) if tot > 0 else w_live
                delta = w_new - w_drift
                to_s.loc[d] = float(delta.abs().sum() / 2)

                nz = delta[delta.abs() > 1e-6]
                if not nz.empty:
                    trades[d] = nz

                held = picks
                w_live = w_new

        r = fwd.loc[d].fillna(0.0)
        gross_s.loc[d] = float((w_live * r).sum())
        w_live = w_live * (1 + r)

    return gross_s, to_s, trades


CONFIGS = [
    ("EW Baseline",    score_raw,    "plain"),
    ("EW + A+B",       score_smooth, "plain_hyst"),
    ("Strat Baseline", score_raw,    "strat"),
    ("Strat + A+B",    score_smooth, "quota_hyst"),
]

log("\n③ 回放四个配置...")
res = {}
for label, sc, mode in CONFIGS:
    res[label] = run(sc, mode)
    log(f"   {label} 完成")

bench = fwd.where(mask).mean(axis=1).reindex(idx).fillna(0.0)
bench_nav = (1 + bench).cumprod()


# ==================== 5. 成本 ====================
def flat_cost(to, c, two_sided):
    """to 是每期单边换手；two_sided=True 时每笔成交都收费。"""
    return to * (2.0 * c if two_sided else c)


def impact_cost(trades, aum):
    cost = pd.Series(0.0, index=idx)
    for d, delta in trades.items():
        codes = delta.index
        q = aum * delta.abs()
        sg = sigma.loc[d].reindex(codes)
        ad = adv.loc[d].reindex(codes)
        ok = q.notna() & sg.notna() & ad.notna() & (ad > 0)
        rate = pd.Series(0.0, index=codes)
        rate[ok] = K_IMPACT * sg[ok] * np.sqrt(q[ok] / ad[ok])
        rate = rate + SPREAD_BP / 1e4
        cost.loc[d] = float((rate * delta.abs()).sum())
    return cost


def summarize(gross, cost):
    net = (gross - cost).iloc[:-1]
    nav = (1 + net).cumprod()
    ann = nav.iloc[-1] ** (252 / len(net)) - 1
    ex_nav = nav / bench_nav.reindex(nav.index).ffill()
    ex = ex_nav.iloc[-1] ** (252 / len(ex_nav)) - 1
    ex_ret = ex_nav.pct_change().dropna()
    return ann * 100, ex * 100, ex / (ex_ret.std() * np.sqrt(252)), cost.mean() * 252 * 100


log("\n④ 计算成本口径...")
rows = []
for label, (gross, to, trades) in res.items():
    ann_to = to.replace(0, np.nan).mean() * (252 / REBAL) * 100
    row = {"配置": label, "年化换手": ann_to,
           "毛利": ((1 + gross.iloc[:-1]).cumprod().iloc[-1] ** (252 / (len(gross) - 1)) - 1) * 100}

    for tag, cost in [
        ("旧12bp", flat_cost(to, 0.0012, False)),
        ("新12bp", flat_cost(to, 0.0012, True)),
        ("新6bp", flat_cost(to, 0.0006, True)),
        ("新20bp", flat_cost(to, 0.0020, True)),
    ]:
        a, e, ir, c = summarize(gross, cost)
        row[f"net_{tag}"] = a
        row[f"ex_{tag}"] = e
        row[f"cost_{tag}"] = c

    for aum in AUM_LIST:
        a, e, ir, c = summarize(gross, impact_cost(trades, aum))
        row[f"net_{aum/1e8:.0f}亿"] = a
        row[f"ex_{aum/1e8:.0f}亿"] = e
        row[f"IR_{aum/1e8:.0f}亿"] = ir

    rows.append(row)

tab = pd.DataFrame(rows).set_index("配置")


# ==================== 6. 输出 ====================
log("\n" + "=" * 104)
log("【口径重算】")
log("=" * 104)

c1 = ["年化换手", "毛利", "cost_旧12bp", "cost_新12bp",
      "net_旧12bp", "net_新12bp", "ex_旧12bp", "ex_新12bp"]
log(tab[c1].round(2).to_string())

log("\n【不同成本假设下的净收益】")
c2 = ["net_新6bp", "net_新12bp", "net_新20bp"] + \
     [f"net_{a/1e8:.0f}亿" for a in AUM_LIST]
log(tab[c2].round(2).to_string())

log("\n【不同成本假设下的超额】")
c3 = ["ex_新6bp", "ex_新12bp", "ex_新20bp"] + \
     [f"ex_{a/1e8:.0f}亿" for a in AUM_LIST]
log(tab[c3].round(2).to_string())

log("\n【Phase 1 头条的三次修订】")
ew = tab.loc["EW Baseline"]
log(f"   原 README（固定权重 + 单边收一次）      : 9.79%")
log(f"   + buy-and-hold 权重漂移                : {ew['net_旧12bp']:.2f}%")
log(f"   + 双边计费（最终口径）                 : {ew['net_新12bp']:.2f}%")
log(f"   差额来源：权重口径约 +0.7pp，成本口径约 -2.9pp")

log("\n【Phase 3 的最终对比（新12bp）】")
for lbl in ["Strat Baseline", "Strat + A+B"]:
    r = tab.loc[lbl]
    log(f"   {lbl:<16} 净 {r['net_新12bp']:6.2f}%   超额 {r['ex_新12bp']:+6.2f}%   "
        f"成本 {r['cost_新12bp']:5.2f}%")

tab.to_csv(OUT / "final_numbers.csv", encoding="utf-8-sig")
log(f"\n✅ 输出到 {OUT}/final_numbers.csv")
log(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
